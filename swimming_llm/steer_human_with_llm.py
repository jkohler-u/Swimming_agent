import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import mujoco
import mujoco.viewer
import numpy as np
import requests
import json
import time
import cv2
import queue
import threading
import copy
from scipy.spatial.transform import Rotation as R 

from helper import format_for_print, SimulationRecorder

from openai import OpenAI

# Initialize the client. 
# This assumes you have set the environment variable OPENAI_API_KEY
client = OpenAI(api_key='sk-...')
MODEL_NAME = "gpt-4o-mini"

# --- Configuration ---
MODEL_PATH = 'swimming_llm/humanoid_laying.xml'
WATER_SURFACE_HEIGHT = 3.0
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4" #"llama3"

# Physics step
KP = 350.0
KD = 12.0
KI =  0.0 #12.0

POSITION_THRESHOLD = 0.03
VELOCITY_THRESHOLD = 0.1
MAX_CARTESIAN_FORCE = 100.0
INTEGRAL_LIMIT = 0.15
PHYSICS_STEPS_PER_CONTROL = 1

# LLM control
REQUIRED_ACTION_KEYS = { "left_arm","right_arm","left_leg","right_leg"}

ACTION_TO_OBSERVATION_KEY = {
    "left_arm": "left_arm_pos",
    "right_arm": "right_arm_pos",
    "left_leg": "left_leg_pos",
    "right_leg": "right_leg_pos",
}

# Maximum change for each Cartesian coordinate between two actions.
MAX_ACTION_DELTA = 0.20

# Broad physical workspaces.
# These are safety limits, not desired poses.
ACTION_LIMITS = {
    "left_arm": {
        "min": np.array([-0.40, -0.40, -0.45]),
        "max": np.array([ 0.40,  0.40,  0.45]),
    },
    "right_arm": {
        "min": np.array([-0.40, -0.40, -0.45]),
        "max": np.array([ 0.40,  0.40,  0.45]),
    },
    "left_leg": {
        "min": np.array([-0.25, -0.20, -1.18]),
        "max": np.array([ 0.25,  0.20, -0.75]),
    },
    "right_leg": {
        "min": np.array([-0.25, -0.20, -1.18]),
        "max": np.array([ 0.25,  0.20, -0.75]),
    },
}

# general limits 
MAX_ACTIONS = 50
STEPS_PER_ACTION = 250
VIEWER_SYNC_INTERVAL = 10
RECORD_INTERVAL = 5

metric = []

class LLMHumanoidController:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.step_count = 0
        self.current_targets = {}
        self.limb_bases = {
            'hand_left': 'torso',
            'hand_right': 'torso',
            'foot_left': 'torso',
            'foot_right': 'torso',
        }
        self.integral_error = {limb: np.zeros(3) for limb in self.limb_bases.keys()}
        self.prev_error = {}
    
    def get_local_pos(self, body_name):
        base_name = self.limb_bases.get(body_name)
        base_pos = self.data.xpos[self.model.body(base_name).id]
        target_pos = self.data.xpos[self.model.body(body_name).id]
        relative_vec_global = target_pos - base_pos
        base_mat = self.data.xmat[self.model.body(base_name).id].reshape(3, 3)
        return (base_mat.T @ relative_vec_global).tolist()

    def get_obs(self):
        head_height = self.data.xpos[self.model.body('head').id][2]
        torso_pos = self.data.xpos[self.model.body('torso').id]
        return {
            "head_height": head_height, "body_height": torso_pos[2],
            "forward": self.data.qvel[0],
            "full_state": {
                "left_arm_pos": self.get_local_pos('hand_left'),
                "right_arm_pos": self.get_local_pos('hand_right'),
                "left_leg_pos": self.get_local_pos('foot_left'),
                "right_leg_pos": self.get_local_pos('foot_right'),
            }
        }

    def set_targets(self, action_json,reset_integral=False):
        """
        Update the current Cartesian targets.
        reset_integral : bool
            If True, reset the integral error for the updated limbs.
            Leave False during continuous swimming.
        """

        mapping = {
            "left_arm": "hand_left",
            "right_arm": "hand_right",
            "left_leg": "foot_left",
            "right_leg": "foot_right",
        }

        self.current_targets = {
            body_name: np.asarray(action_json[action_name],dtype=np.float64 )
            for action_name, body_name in mapping.items()
            if action_name in action_json and action_json[action_name] is not None   
        }

        if reset_integral:
            for body_name in self.current_targets:
                self.integral_error[body_name][:] = 0.0

        self.prev_error = {
            body_name: float("inf")
            for body_name in self.current_targets
        }
    
    def physics_step(self, debug=False) -> bool:
        """
        Move active end-effectors toward torso-relative Cartesian targets
        using Jacobian-transpose PD control.

        Returns True when every active target is within the position
        threshold and below the velocity threshold.
        """
        self.model.opt.timestep = 0.002

        # Clear controls once before processing all active targets.
        self.data.ctrl[:] = 0.0
        reached_count = 0
        active_target_count = 0

        for body_name, local_target in self.current_targets.items():
            if local_target is None: continue
            active_target_count += 1
            target_local = np.asarray(local_target,dtype=np.float64)

            body_id = self.model.body(body_name).id
            base_name = self.limb_bases[body_name]
            base_id = self.model.body(base_name).id

            # ---------------------------------
            # 1. Current torso-relative position
            # ---------------------------------
            body_pos_world = self.data.xpos[body_id].copy()
            base_pos_world = self.data.xpos[base_id].copy()
            base_rot_world = self.data.xmat[base_id].reshape(3, 3).copy()

            relative_position_world = body_pos_world - base_pos_world
            current_local = base_rot_world.T @ relative_position_world
            error_local = target_local - current_local
            distance = float(np.linalg.norm(error_local))

            # ---------------------------------
            # 2. Hand and torso Jacobians
            # ---------------------------------
            jac_hand_pos = np.zeros((3, self.model.nv), dtype=np.float64)
            jac_hand_rot = np.zeros((3, self.model.nv),dtype=np.float64)
            jac_base_pos = np.zeros((3, self.model.nv),dtype=np.float64)
            jac_base_rot = np.zeros((3, self.model.nv),dtype=np.float64)

            mujoco.mj_jacBody(self.model,self.data,jac_hand_pos, jac_hand_rot,body_id)
            mujoco.mj_jacBody(self.model,self.data,jac_base_pos,jac_base_rot,base_id,)

            # ---------------------------------
            # 3. Exact torso-local Jacobian
            # ---------------------------------
            def skew(vector):
                x, y, z = vector

                return np.array([
                    [0.0, -z, y],
                    [z, 0.0, -x],
                    [-y, x, 0.0],
                ])

            # Local position: p_local = R_base.T @ (p_hand - p_base)
            jac_relative_world = jac_hand_pos - jac_base_pos + skew(relative_position_world) @ jac_base_rot
            jac_local = base_rot_world.T @ jac_relative_world
            velocity_local = jac_local @ self.data.qvel
            speed = float(np.linalg.norm(velocity_local) )

            # ---------------------------------
            # 4. Check whether target is reached
            # ---------------------------------
            target_reached = (distance <= POSITION_THRESHOLD and speed <= VELOCITY_THRESHOLD )

            if target_reached:
                reached_count += 1
                if debug:
                    print(f"\nBody: {body_name}")
                    print("Target reached")
                    print( "Position error:",round(distance, 5),)
                    print("Relative speed:",round(speed, 5), )
                continue

            if not target_reached and self.step_count % 30 == 0:
                reasons = []

                if distance > POSITION_THRESHOLD:
                    reasons.append(f"position error {distance:.4f} m > {POSITION_THRESHOLD:.4f} m")

                if speed > VELOCITY_THRESHOLD:
                    reasons.append(f"speed {speed:.4f} m/s > {VELOCITY_THRESHOLD:.4f} m/s")

                print(f"{body_name} not reached: " + "; ".join(reasons), flush=True)

            # ---------------------------------
            # 5. Cartesian PD in torso frame
            # ---------------------------------
            dt = self.model.opt.timestep
            if distance < 0.005: self.integral_error[body_name] *= 0.9

            # Accumulate Cartesian error.
            self.integral_error[body_name] += error_local * dt

            # Anti-windup.
            self.integral_error[body_name] = np.clip( self.integral_error[body_name],-INTEGRAL_LIMIT,INTEGRAL_LIMIT)
            p_force = KP * error_local
            i_force = KI  * self.integral_error[body_name]
            d_force = -KD * velocity_local

            cartesian_force = p_force + i_force + d_force

            force_norm = float( np.linalg.norm(cartesian_force))

            if force_norm > MAX_CARTESIAN_FORCE:
                cartesian_force *=  MAX_CARTESIAN_FORCE/ force_norm

            # The Jacobian and force are both torso-local.
            desired_torque = (jac_local.T @ cartesian_force )

            # ---------------------------------
            # 5. Map generalized torque to actuator controls
            # ---------------------------------
            for actuator_id in range(self.model.nu):
                transmission_type =  self.model.actuator_trntype[actuator_id]
                if transmission_type != mujoco.mjtTrn.mjTRN_JOINT: continue

                joint_id = int( self.model.actuator_trnid[actuator_id, 0,])
                if joint_id < 0:  continue

                dof_id = int( self.model.jnt_dofadr[joint_id])
                gear = float(self.model.actuator_gear[actuator_id, 0] )

                if abs(gear) < 1e-8: continue

                joint_torque = desired_torque[dof_id]
                self.data.ctrl[actuator_id] += joint_torque / gear
                

            if debug:
                print(f"\nBody: {body_name}")
                print("Target local:",np.round(target_local, 4))
                print("Current local:",np.round(current_local, 4))
                print("Error local:", np.round(error_local, 5))
                print( "Distance:",round(distance, 5) )
                print("Velocity local:", np.round(velocity_local, 5))
                print("Speed:",round(speed, 5))
                print("P-force norm:", round(np.linalg.norm(p_force),5))
                print("D-force norm:", round(np.linalg.norm(d_force),5))
                print( "I-force norm:",round(np.linalg.norm(i_force),5) )
                print("Cartesian force norm:",round( np.linalg.norm(cartesian_force),5))
                print( "Max |desired torque|:", round( np.max(np.abs(desired_torque)),5))

        # ---------------------------------
        # 7. Apply actuator control limits
        # ---------------------------------
        for actuator_id in range(self.model.nu):
            if self.model.actuator_ctrllimited[actuator_id]:
                ctrl_min, ctrl_max = self.model.actuator_ctrlrange[actuator_id]
                self.data.ctrl[actuator_id] = np.clip(self.data.ctrl[actuator_id], ctrl_min, ctrl_max)

        for _ in range(PHYSICS_STEPS_PER_CONTROL):
            mujoco.mj_step(self.model,self.data )

        self.step_count += 1

        if active_target_count == 0:
            print( "ERROR: No active controller targets.",flush=True)
            return False

        return reached_count == active_target_count
    
# ---------------------------------------------------------------------
# LLM SWIMMING CONTROL
# ---------------------------------------------------------------------

# def get_llm_action(prompt):
#     """
#     Request one swimming action from Ollama.
#     Returns:
#         dict | None
#     """
#     payload = {
#         "model": MODEL_NAME,
#         "prompt": prompt,
#         "stream": False,
#         "format": "json",
#         "keep_alive": "30m",
#         "options": {"temperature": 0, "num_predict": 250},
#     }

#     print(f"Sending request to local Ollama model {MODEL_NAME}...",flush=True, )
#     print(prompt)
#     try:
#         response = requests.post(OLLAMA_URL, json=payload, timeout=300)
#         response.raise_for_status()
#         response_data = response.json()
#         raw_response = response_data.get("response","", )
#         print("Raw model response:",raw_response, flush=True,)

#         if isinstance(raw_response, dict):
#             return raw_response
        
#         return json.loads(raw_response)

#     except requests.RequestException as exc:
#         print( f"Ollama request error: {exc}", flush=True, )
#         return None

#     except ( json.JSONDecodeError,TypeError,ValueError,KeyError, ) as exc:
#         print( f"Ollama response error: {exc}",flush=True,)
#         return None



def get_llm_action(prompt):
    """
    Request one swimming action from GPT-4o-mini.
    Returns:
        dict | None
    """
    print(f"Sending request to OpenAI model {MODEL_NAME}...", flush=True)
    print(prompt)

    try:
        # OpenAI uses a ChatCompletion interface
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
            ],
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=500,
        )

        # Extract the content string from the response object
        raw_response = response.choices[0].message.content
        print("Raw model response:", raw_response, flush=True)

        # Parse the string into a dictionary
        return json.loads(raw_response)

    except Exception as exc:
        # Catching general exceptions as the OpenAI library 
        # raises specific APIError/RateLimitError types
        print(f"OpenAI request/response error: {exc}", flush=True)
        return None

def validate_action(action):
    """
    Verify that the LLM returned all four limbs as finite 3-D vectors.
    """
    if not isinstance(action, dict): return False
    if not REQUIRED_ACTION_KEYS.issubset(action.keys()): return False

    for action_name in REQUIRED_ACTION_KEYS:
        target = action[action_name]

        if not isinstance(target,(list, tuple, np.ndarray)): return False
        
        try:
            target_array = np.asarray( target, dtype=np.float64)
        except (TypeError, ValueError):
            return False

        if target_array.shape != (3,):
            return False

        if not np.all( np.isfinite(target_array)):
            return False
    return True

def sanitize_action(requested_action):
    """
    Convert targets to floats and clip only to the limb workspaces.
    """
    sanitized_action = {}
    for action_name in REQUIRED_ACTION_KEYS:
        requested_target = np.asarray(requested_action[action_name], dtype=np.float64)
        limits = ACTION_LIMITS[action_name]
        safe_target = np.clip(requested_target, limits["min"], limits["max"])
        sanitized_action[action_name] = safe_target.tolist()

    return sanitized_action

def build_llm_prompt( observation, action_history, vel_history):
    """
    Construct the swimming prompt from the latest state.
    """
    obs = observation["full_state"]
    recent_actions = action_history[-3:]
    recent_vel = vel_history[-3:]
    print("LNSSS", len(recent_vel), len(recent_actions))
    head_height_relative = (observation["head_height"]- WATER_SURFACE_HEIGHT)
    body_height_relative = ( observation["body_height"]- WATER_SURFACE_HEIGHT )

    left_arm_pos = ", ".join([f"{val:.2f}" for val in obs['left_arm_pos']])
    right_arm_pos = ", ".join([f"{val:.2f}" for val in obs['right_arm_pos']])
    left_leg_pos = ", ".join([f"{val:.2f}" for val in obs['left_leg_pos']])
    right_leg_pos = ", ".join([f"{val:.2f}" for val in obs['right_leg_pos']])

    return f"""
        You are the motion-control AI for a humanoid robot performing the backstroke. 
        Your goal is to generate the next set of coordinates for the hands and feet so it generates forward movement.

        ### COORDINATE SYSTEM 
        You set the next positions for the humanoids hands and feet.
        The coordinates are given relative to the humanoids torso.
        - X-Axis (Vertical): Positive (+) = Towards Sky | Negative (-) = Towards Floor
        - Y-Axis (Lateral): Positive (+) = Robot's Left | Negative (-) = Robot's Right
        - Z-Axis (Longitudinal): Positive (+) = Above Head | Negative (-) = Towards Feet
        Positions are given in meters.

         ### MOTION PATTERN: THE ALTERNATE KICK (Legs)
        Legs alternate towards the water surface and towards the floor while remaining extended:
        1. Down beat left leg (negative X), right leg up beat (positive X) 
        2. Up beat left leg (positive X), right leg down beat (negative X) 
        -> Return to Phase 1. Important: Alternation occurs on the X-axis.
         
        ### MOTION PATTERN: THE WINDMILL (Arms)
        The arms must operate in opposite phases. One arm follows this sequence while the other is 180 degrees offset:
        1. Right: Recovery -> negative Z / Left: Overhead -> positive Z
        2. Right: Lift -> positive X / Left: Pull -> negative X
        3. Right: Overhead -> positive Z / Left: Recovery -> negative Z
        4. Right: Pull -> negative X / Left: Lift -> positive X
        -> Return to Phase 1.

        ### STRICT PHYSICAL CONSTRAINTS
        1. NO CLIPPING: Ensure L_Hand / L_Foot Y is always positive and R_Hand/L_Foot Y is always negative.
        2. ALWAYS IN MOTION: The generated coordinates have to be different from the current coordinates,
           to be effective, the difference between the current and last action should be larger than 0.15
        3. MOVEMENT RANGE: The legs extend 1.25m from the Torso, the arms extend 0.6m from the torso.

        ### PREVIOUS STATES :
        Previous positions you choose: {'; '.join([f"State {i}: {recent_actions[i]} (Vel: {recent_vel[i]:.2f})" for i in range(len(recent_actions))])}

        ### CURRENT STATE
        - Current positions:  L_Hand (x, y, z): {left_arm_pos}, R_Hand (x, y, z): {right_arm_pos}
                              L_Foot (x, y, z): {left_leg_pos}, R_Foot (x, y, z): {right_leg_pos}
        - Velocity: {observation["forward"]:.2f}
        - Head hight relative to the waterline: {head_height_relative:.2f}
        
        Formatting (IMPORTANT: the output must contain NEW positions for EVERY limb):
        {{"left_arm": [X, Y, Z], "right_arm": [X, Y, Z], "left_leg": [X, Y, Z], "right_leg": [X, Y, Z]}}
        Return ONLY a valid JSON object. No explanation.
"""

def llm_worker( action_queue,controller,controller_lock, action_history, vel_history, history_lock, stop_event):
    """
    Generate actions without blocking the simulation thread.
    """
    while not stop_event.is_set():
        # Do not request another action when one is already waiting to be executed.
        if action_queue.full():
            time.sleep(0.05)
            continue

        # MuJoCo data is shared between threads - read it while holding the controller lock.
        with controller_lock: observation = copy.deepcopy(controller.get_obs())
        with history_lock: history_snapshot = copy.deepcopy(action_history)
        with history_lock: vel_snapshot = copy.deepcopy(vel_history)

        prompt = build_llm_prompt(observation,history_snapshot,vel_snapshot)
        raw_action = get_llm_action(prompt)
        print("Raw action:",format_for_print(raw_action))

        if raw_action is None:
            print("No valid Ollama action received. Retrying in 5 seconds.",flush=True)
            time.sleep(5.0)
            continue

        if not validate_action(raw_action):
            print("Rejected invalid LLM action.")
            time.sleep(0.2)
            continue

        safe_action = sanitize_action(raw_action)

        try:
            action_queue.put(safe_action,timeout=0.5,)
        except queue.Full:
            pass


# ---------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------
print("Initializing LLM Humanoid Environment...")

controller = LLMHumanoidController(MODEL_PATH)
recorder = SimulationRecorder(controller.model, controller.data)

# Ensure body positions and orientations are initialized before obtaining the first observation.
mujoco.mj_forward( controller.model,controller.data)

action_queue = queue.Queue(maxsize=1)

controller_lock = threading.Lock()
history_lock = threading.Lock()
stop_event = threading.Event()

action_history = []
vel_history = []

with controller_lock:
    initial_observation = copy.deepcopy(controller.get_obs())

initial_action = {
    "left_arm": initial_observation["full_state"]["left_arm_pos"],
    "right_arm": initial_observation["full_state"]["right_arm_pos"],
    "left_leg": initial_observation[ "full_state" ]["left_leg_pos"],
    "right_leg": initial_observation["full_state"]["right_leg_pos"],
}

action_history.append(copy.deepcopy(initial_action))
vel_history.append(int(0))
worker_thread = threading.Thread(target=llm_worker, args=(action_queue,controller, controller_lock, action_history, vel_history, history_lock, stop_event), daemon=True)
worker_thread.start()
executed_actions = 0

try:
    with mujoco.viewer.launch_passive(controller.model,controller.data,) as viewer:
        while (viewer.is_running() and executed_actions < MAX_ACTIONS):
            try:
                queued_action = action_queue.get(timeout=0.1)
            except queue.Empty:
                viewer.sync()
                continue

            if not validate_action(queued_action):
                print("Rejected invalid queued action.")
                continue

            # The humanoid may have moved - Sanitize once more using the newest state.
            with controller_lock: current_observation = copy.deepcopy(controller.get_obs())

            action = sanitize_action(queued_action)
            executed_actions += 1


            print("=" * 70)
            print(f"Executing LLM action {executed_actions}/{MAX_ACTIONS}")
            print(format_for_print(action))


            with controller_lock:
                controller.set_targets(action)
                print("Controller targets:", {body_name: target.tolist() for body_name, target in controller.current_targets.items()}, flush=True)

            target_reached_at_least_once = False

            for action_step in range(STEPS_PER_ACTION):
                if not viewer.is_running(): break

                with controller_lock:
                    targets_reached = controller.physics_step()
                    current_step = controller.step_count

                if targets_reached:
                    target_reached_at_least_once = True
                    print(f"All targets reached after {action_step + 1} steps.",flush=True)
                    break

                # Record only when a recorder exists.
                if "recorder" in globals() and current_step % RECORD_INTERVAL == 0:
                    action_text=f"Action:{executed_actions} {action}"
                    with controller_lock: recorder.record_frame(action_text = action_text)

                if current_step % VIEWER_SYNC_INTERVAL == 0:
                    viewer.sync()

            with controller_lock: observation_after = copy.deepcopy(controller.get_obs())
            with history_lock: action_history.append(copy.deepcopy(action))
            with history_lock: vel_history.append(observation_after["forward"])
            metric.append([observation_after["forward"], observation_after["head_height"]])

            print("Positions after action:",format_for_print(observation_after["full_state"]))
            print("Reached strict controller ""threshold:", target_reached_at_least_once)
            print("=" * 70)
            print("Action history", action_history)
            


finally:
    stop_event.set()
    with controller_lock: controller.data.ctrl[:] = 0.0
    worker_thread.join(timeout=2.0)
import csv
filename = "wo_example_gpt.csv"
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(["forward_velocity", "head_height"])
    
    # Write all the data rows at once
    writer.writerows(metric)

recorder.save_video("wo_example_gpt.mp4")
print("LLM swimming simulation complete.")