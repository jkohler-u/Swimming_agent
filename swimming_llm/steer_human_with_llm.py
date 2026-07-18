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
from scipy.spatial.transform import Rotation as R  # <--- Add this import
class SimulationRecorder:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.renderer = mujoco.Renderer(model)
        self.frames = []

    def record_frame(self):
        self.renderer.update_scene(self.data)
        pixels = self.renderer.render()
        frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        self.frames.append(frame_bgr)

    def save_video(self, filename, fps=30):
        if not self.frames: return
        height, width, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        for frame in self.frames: video.write(frame)
        video.release()

# --- Configuration ---
MODEL_PATH = 'underwater_swimming_simple_human/humanoid_laying.xml'
WATER_SURFACE_HEIGHT = 3.0
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3" #"llama3"
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

    def get_euler_angles(self):
        quat_mujoco = self.data.qpos[3:7] 
        quat_scipy = np.array([quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0]])
        return R.from_quat(quat_scipy).as_euler('xyz', degrees=False) 
    
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
            "full_state": {
                "left_arm_pos": self.get_local_pos('hand_left'),
                "right_arm_pos": self.get_local_pos('hand_right'),
                "left_leg_pos": self.get_local_pos('foot_left'),
                "right_leg_pos": self.get_local_pos('foot_right'),
            }
        }

    def set_targets(
        self,
        action_json,
        reset_integral=False,
    ):
        """
        Update the current Cartesian targets.

        Parameters
        ----------
        action_json : dict
            {
                "left_arm": [...],
                "right_arm": [...],
                "left_leg": [...],
                "right_leg": [...],
            }

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
            body_name: np.asarray(
                action_json[action_name],
                dtype=np.float64,
            )
            for action_name, body_name in mapping.items()
            if (
                action_name in action_json
                and action_json[action_name] is not None
            )
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

        KP = 100.0
        KD = 12.0
        KI =  0.0 #12.0

        

        POSITION_THRESHOLD = 0.03
        VELOCITY_THRESHOLD = 0.1
        MAX_CARTESIAN_FORCE = 100.0
        INTEGRAL_LIMIT = 0.15

        PHYSICS_STEPS_PER_CONTROL = 1

        self.model.opt.timestep = 0.002

        # Clear controls once before processing all active targets.
        self.data.ctrl[:] = 0.0

        reached_count = 0
        active_target_count = 0

        for body_name, local_target in self.current_targets.items():
            if local_target is None:
                continue

            active_target_count += 1

            target_local = np.asarray(
                local_target,
                dtype=np.float64,
            )

            body_id = self.model.body(body_name).id

            base_name = self.limb_bases[body_name]
            base_id = self.model.body(base_name).id

            # ---------------------------------
            # 1. Current torso-relative position
            # ---------------------------------
            body_pos_world = self.data.xpos[body_id].copy()
            base_pos_world = self.data.xpos[base_id].copy()

            base_rot_world = (
                self.data.xmat[base_id]
                .reshape(3, 3)
                .copy()
            )

            relative_position_world = (
                body_pos_world - base_pos_world
            )

            current_local = (
                base_rot_world.T
                @ relative_position_world
            )

            error_local = target_local - current_local
            distance = float(np.linalg.norm(error_local))

            # ---------------------------------
            # 2. Hand and torso Jacobians
            # ---------------------------------
            jac_hand_pos = np.zeros(
                (3, self.model.nv),
                dtype=np.float64,
            )
            jac_hand_rot = np.zeros(
                (3, self.model.nv),
                dtype=np.float64,
            )

            jac_base_pos = np.zeros(
                (3, self.model.nv),
                dtype=np.float64,
            )
            jac_base_rot = np.zeros(
                (3, self.model.nv),
                dtype=np.float64,
            )

            mujoco.mj_jacBody(
                self.model,
                self.data,
                jac_hand_pos,
                jac_hand_rot,
                body_id,
            )

            mujoco.mj_jacBody(
                self.model,
                self.data,
                jac_base_pos,
                jac_base_rot,
                base_id,
            )

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

            # Local position:
            # p_local = R_base.T @ (p_hand - p_base)
            #
            # Its velocity includes:
            # - hand translation
            # - torso translation
            # - torso rotation
            jac_relative_world = (
                jac_hand_pos
                - jac_base_pos
                + skew(relative_position_world)
                @ jac_base_rot
            )

            jac_local = (
                base_rot_world.T
                @ jac_relative_world
            )

            velocity_local = (
                jac_local @ self.data.qvel
            )

            speed = float(
                np.linalg.norm(velocity_local)
            )

            # ---------------------------------
            # 4. Check whether target is reached
            # ---------------------------------
            target_reached = (
                distance <= POSITION_THRESHOLD
                and speed <= VELOCITY_THRESHOLD
            )

            if target_reached:
                reached_count += 1

                if debug:
                    print(f"\nBody: {body_name}")
                    print("Target reached")
                    print(
                        "Position error:",
                        round(distance, 5),
                    )
                    print(
                        "Relative speed:",
                        round(speed, 5),
                    )

                continue


            if (
                not target_reached
                and self.step_count % 30 == 0
            ):
                reasons = []

                if distance > POSITION_THRESHOLD:
                    reasons.append(
                        f"position error {distance:.4f} m "
                        f"> {POSITION_THRESHOLD:.4f} m"
                    )

                if speed > VELOCITY_THRESHOLD:
                    reasons.append(
                        f"speed {speed:.4f} m/s "
                        f"> {VELOCITY_THRESHOLD:.4f} m/s"
                    )

                print(
                    f"{body_name} not reached: "
                    + "; ".join(reasons),
                    flush=True,
                )

            # ---------------------------------
            # 5. Cartesian PD in torso frame
            # ---------------------------------
            dt = self.model.opt.timestep

            if distance < 0.005:
                self.integral_error[body_name] *= 0.9

            # Accumulate Cartesian error.
            self.integral_error[body_name] += (
                error_local * dt
            )

            # Anti-windup.
            self.integral_error[body_name] = np.clip(
                self.integral_error[body_name],
                -INTEGRAL_LIMIT,
                INTEGRAL_LIMIT,
            )

            p_force = KP * error_local

            i_force = (
                KI
                * self.integral_error[body_name]
            )

            d_force = -KD * velocity_local

            cartesian_force = (
                p_force
                + i_force
                + d_force
            )

            force_norm = float(
                np.linalg.norm(cartesian_force)
            )

            if force_norm > MAX_CARTESIAN_FORCE:
                cartesian_force *= (
                    MAX_CARTESIAN_FORCE
                    / force_norm
                )

            # The Jacobian and force are both torso-local.
            desired_torque = (
                jac_local.T @ cartesian_force
            )

            # ---------------------------------
            # 5. Map generalized torque
            #    to actuator controls
            # ---------------------------------
            for actuator_id in range(self.model.nu):
                transmission_type = (
                    self.model.actuator_trntype[
                        actuator_id
                    ]
                )

                if (
                    transmission_type
                    != mujoco.mjtTrn.mjTRN_JOINT
                ):
                    continue

                joint_id = int(
                    self.model.actuator_trnid[
                        actuator_id,
                        0,
                    ]
                )

                if joint_id < 0:
                    continue

                dof_id = int(
                    self.model.jnt_dofadr[joint_id]
                )

                gear = float(
                    self.model.actuator_gear[
                        actuator_id,
                        0,
                    ]
                )

                if abs(gear) < 1e-8:
                    continue

                joint_torque = desired_torque[dof_id]

                # Use += so multiple active targets can
                # contribute to the same actuator.
                self.data.ctrl[actuator_id] += (
                    joint_torque / gear
                )

            # ---------------------------------
            # 6. Debug output
            # ---------------------------------


            if debug:
                print(f"\nBody: {body_name}")

                print(
                    "Target local:",
                    np.round(target_local, 4),
                )
                print(
                    "Current local:",
                    np.round(current_local, 4),
                )
                print(
                    "Error local:",
                    np.round(error_local, 5),
                )
                print(
                    "Distance:",
                    round(distance, 5),
                )
                print(
                    "Velocity local:",
                    np.round(velocity_local, 5),
                )
                print(
                    "Speed:",
                    round(speed, 5),
                )
                print(
                    "P-force norm:",
                    round(
                        np.linalg.norm(p_force),
                        5,
                    ),
                )
                print(
                    "D-force norm:",
                    round(
                        np.linalg.norm(d_force),
                        5,
                    ),
                )

                print(
                    "I-force norm:",
                    round(
                        np.linalg.norm(i_force),
                        5,
                    ),
                )
                print(
                    "Cartesian force norm:",
                    round(
                        np.linalg.norm(cartesian_force),
                        5,
                    ),
                )
                print(
                    "Max |desired torque|:",
                    round(
                        np.max(np.abs(desired_torque)),
                        5,
                    ),
                )

            

        # ---------------------------------
        # 7. Apply actuator control limits
        # ---------------------------------
        for actuator_id in range(self.model.nu):
            if self.model.actuator_ctrllimited[
                actuator_id
            ]:
                ctrl_min, ctrl_max = (
                    self.model.actuator_ctrlrange[
                        actuator_id
                    ]
                )

                self.data.ctrl[actuator_id] = (
                    np.clip(
                        self.data.ctrl[actuator_id],
                        ctrl_min,
                        ctrl_max,
                    )
                )


        for _ in range(PHYSICS_STEPS_PER_CONTROL):
            mujoco.mj_step(
                self.model,
                self.data,
            )

        self.step_count += 1

        if active_target_count == 0:
            print(
                "ERROR: No active controller targets.",
                flush=True,
            )
            return False

        return reached_count == active_target_count
# ---------------------------------------------------------------------
# LLM SWIMMING CONTROL
# ---------------------------------------------------------------------

REQUIRED_ACTION_KEYS = {
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
}

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


def get_llm_action(prompt):
    """
    Request one Cartesian swimming action from Ollama.
    Returns:
        dict | None
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {
            "temperature": 0.2,
            "num_predict": 250,
        },
    }

    request_start = time.perf_counter()

    print(f"Sending request to local Ollama model {MODEL_NAME}...",flush=True, )

    try:
        response = requests.post(OLLAMA_URL, json=payload,timeout=300,)

        response.raise_for_status()

        elapsed = time.perf_counter() - request_start

        print(f"Ollama replied after {elapsed:.1f} seconds.",flush=True,)

        response_data = response.json()
        raw_response = response_data.get("response","", )

        print("Raw model response:",raw_response, flush=True,)
        print("prompt", prompt)

        if isinstance(raw_response, dict):
            return raw_response
        
        return json.loads(raw_response)

    except requests.Timeout:
        elapsed = time.perf_counter() - request_start
        print( f"Ollama timed out after " f"{elapsed:.1f} seconds.", flush=True,)
        return None

    except requests.RequestException as exc:
        print( f"Ollama request error: {exc}", flush=True, )
        return None

    except ( json.JSONDecodeError,TypeError,ValueError,KeyError, ) as exc:
        print( f"Ollama response error: {exc}",flush=True,)
        return None

def validate_action(action):
    """
    Verify that the LLM returned all four limbs as finite 3-D vectors.
    """
    if not isinstance(action, dict):
        return False

    if not REQUIRED_ACTION_KEYS.issubset(
        action.keys()
    ):
        return False

    for action_name in REQUIRED_ACTION_KEYS:
        target = action[action_name]

        if not isinstance(
            target,
            (list, tuple, np.ndarray),
        ):
            return False

        try:
            target_array = np.asarray(
                target,
                dtype=np.float64,
            )
        except (TypeError, ValueError):
            return False

        if target_array.shape != (3,):
            return False

        if not np.all(
            np.isfinite(target_array)
        ):
            return False

    return True


def sanitize_action(requested_action):
    """
    Convert targets to floats and clip only to the limb workspaces.
    """
    sanitized_action = {}

    for action_name in REQUIRED_ACTION_KEYS:
        requested_target = np.asarray(
            requested_action[action_name],
            dtype=np.float64,
        )

        limits = ACTION_LIMITS[action_name]

        safe_target = np.clip(
            requested_target,
            limits["min"],
            limits["max"],
        )

        sanitized_action[action_name] = safe_target.tolist()

    return sanitized_action

def format_for_print(value):
    """
    Produce shorter readable output for logs.
    """
    if isinstance(value, dict):
        return {
            key: format_for_print(item)
            for key, item in value.items()
        }

    if isinstance(value, np.ndarray):
        return format_for_print(
            value.tolist()
        )

    if isinstance(value, (list, tuple)):
        return [
            format_for_print(item)
            for item in value
        ]

    if isinstance(
        value,
        (float, np.floating),
    ):
        return round(float(value), 3)

    return value


def build_llm_prompt( observation, action_history):
    """
    Construct the swimming prompt from the latest state.
    """
    obs = observation["full_state"]

    recent_actions = action_history[-4:]

    head_height_relative = (
        observation["head_height"]
        - WATER_SURFACE_HEIGHT
    )

    body_height_relative = (
        observation["body_height"]
        - WATER_SURFACE_HEIGHT
    )

    left_arm_pos = ", ".join([f"{val:.2f}" for val in obs['left_arm_pos']])
    right_arm_pos = ", ".join([f"{val:.2f}" for val in obs['right_arm_pos']])
    left_leg_pos = ", ".join([f"{val:.2f}" for val in obs['left_leg_pos']])
    right_leg_pos = ", ".join([f"{val:.2f}" for val in obs['right_leg_pos']])


    # return f"""You are an AI controlling a humanoid robot learning to swim backstroke.
    #             The coordinates are LOCAL to the torso: 
    #             Current State:
    #             - Head Height Error: {head_height_relative:.2f} (Positive = above water)
                
    #             - Limb Positions (Relative to Torso):
    #             L_Hand (x, y, z): {left_arm_pos}
    #             R_Hand (x, y, z): {right_arm_pos}
    #             eg.:
    #             L_Hand: [0, 0.01, -0.63] R_Hand: [0, 0.01, -0.63] arms parallel to the torso, pointing towards the feet
    #             if x is positive - the hand reaches up towards the sky, if its negative, it reaches towards the floor
    #             if y is positive . the hand reaches towards the left, if its negative, it reaches towards the right
    #             if z is positive - the hand reaches above the head, if its negative it reaches towards the feet (see example)

    #             L_Foot (x, y, z): {left_leg_pos}
    #             R_Foot (x, y, z): {right_leg_pos}
    #             L_Foot: [-0.2, -0.01, -1.25] - the leg is extended, pointing slightly below the torso towards the floor
    #             If the x is positive, the foot is reaching towards the sky, if its negative the foot reaches towards the floor.
    #             If the y is positive, the foot is to the left side of the torso; if negative, it is on the right side.
    #             As the legs are below the torso, z should be negative. For a fully extended leg z = -1.25.

    #             Keep in mind that the movements have to be incremental and within the physical limits of the humanoid.
    #             Moving the hand from in front of the torso eg. x = 0.5 to behind the torso x = -0.5 in one step is not physically possible.
                 
    #             Goal:
    #             You start out lying on your back at the water surface, looking up to the sky.
    #              Backstroke Motion: 
    #             - Arms: The arms alternatingly perform the following, windmill like motion. The streched out arm points towards the feed (parallet to body), then towards the sky(perpendicular to body), then up above the head (parallel to body), then towards the floor (perpendicular to body), and the back towards the feet (parallel to body).
    #             - Legs: Alternatingly, move one leg slightly up towards the sky while the other is moved slightly down towards the floor. Both legs are stretched out.

    #             Physical Constraints (STRICT):
    #             - Move incrementally. Max change per step: 0.25m.
    #             - Reach Limit: Do not suggest coordinates that exceed the physical length of the limbs.
    #             - Arms: Max reach is approx 0.6m from the shoulder. Keep X larger than 0.3 (left) and smaller than -0.3 (right) to prevent clipping, and Z within [-0.6, 0.6] .
    #             - Legs: Max reach is approx 1.2m from the hip. Keep X and Y within [-0.3, 0.3] and Z within [-1, 0.1].

    #             Constraint: 
    #             Return ONLY a valid JSON object.

    #             Example Output Positions:
    #             {{"left_arm": [0.1, 0.1, -0.6] , "right_arm": [0.1, 0.1, -0.6], "left_leg": [0.2, 0.09, -1], "right_leg": [-0.05, -0.09, -1]}}
    #             Last chosen positions: {recent_actions}
    #             Your turn - choose where the humanoid should move its limbs next (tip: if available, consider the last actions, their trajectory and build upon them) (JSON only):""".strip()

    return f"""You are the motion-control AI for a humanoid robot learning the backstroke. 
        Your goal is to generate the next set of coordinates for the limbs based on a continuous fluid motion.

        ### COORDINATE SYSTEM (Torso-Relative)
        - X-Axis (Vertical): Positive (+) = Towards Sky | Negative (-) = Towards Floor
        - Y-Axis (Lateral): Positive (+) = Robot's Left | Negative (-) = Robot's Right
        - Z-Axis (Longitudinal): Positive (+) = Above Head | Negative (-) = Towards Feet

        ### MOTION PATTERN: THE WINDMILL (Arms)
        The arms must operate in opposite phases. One arm follows this sequence while the other is 180 degrees offset:
        1. Recovery (Parallel to body): [X: 0, Y: ±0.2, Z: -0.6]
        2. Lift (Perpendicular to body): [X: 0.5, Y: ±0.2, Z: -0.3]
        3. Overhead (Parallel to body): [X: 0, Y: ±0.2, Z: 0.6]
        4. Pull (Perpendicular to body): [X: -0.5, Y: ±0.2, Z: -0.3]
        -> Return to Phase 1.

        ### MOTION PATTERN: THE FLUTTER KICK (Legs)
        Legs alternate small vertical oscillations while remaining extended:
        - Leg A: X increases (up) while Leg B: X decreases (down).
        - Keep Z constant near -1.2 and Y near 0.

        ### STRICT PHYSICAL CONSTRAINTS
        1. INCREMENTAL MOVEMENT: The distance between the 'Last chosen positions' and the new positions must NOT exceed 0.25m per limb.
        2. ARM BOUNDARIES: X: [-0.6, 0.6] | Y: [0.2, 0.4] (Left) / [-0.2, -0.4] (Right) | Z: [-0.6, 0.6].
        3. LEG BOUNDARIES: X: [-0.3, 0.3] | Y: [-0.3, 0.3] | Z: [-1.25, -1.0].
        4. NO CLIPPING: Ensure L_Hand Y is always positive and R_Hand Y is always negative.

        ### CURRENT STATE
        - Head Height Error: {head_height_relative:.2f}
        - Current positions:  L_Hand (x, y, z): {left_arm_pos}, R_Hand (x, y, z): {right_arm_pos}
                              L_Foot (x, y, z): {left_leg_pos}, R_Foot (x, y, z): {right_leg_pos}

        ### TASK
        Analyze the trajectory of the {recent_actions}. Determine which phase of the windmill each arm is in and advance them incrementally toward the next phase. Alternate the legs.

        Return ONLY a valid JSON object. No explanation.

        Example Output:
        {{"left_arm": [0.1, 0.2, -0.5], "right_arm": [-0.1, -0.2, 0.3], "left_leg": [0.1, 0.0, -1.2], "right_leg": [-0.1, 0.0, -1.2]}}"""

def llm_worker(
    action_queue,
    controller,
    controller_lock,
    action_history,
    history_lock,
    stop_event,
):
    """
    Generate actions without blocking the simulation thread.
    """
    while not stop_event.is_set():
        # Do not request another action when one is
        # already waiting to be executed.
        if action_queue.full():
            time.sleep(0.05)
            continue

        # MuJoCo data is shared between threads, so read it
        # while holding the controller lock.
        with controller_lock:
            observation = copy.deepcopy(
                controller.get_obs()
            )

        with history_lock: history_snapshot = copy.deepcopy(action_history)

        prompt = build_llm_prompt(observation,history_snapshot)

        raw_action = get_llm_action(prompt)

        if raw_action is None:
            print(
                "No valid Ollama action received. "
                "Retrying in 5 seconds.",
                flush=True,
            )
            time.sleep(5.0)
            continue

        if not validate_action(raw_action):
            print(
                "Rejected invalid LLM action."
            )
            time.sleep(0.2)
            continue

        safe_action = sanitize_action(
            raw_action,
        )


        print(
            "Raw LLM action:",
            format_for_print(raw_action),
            flush=True,
        )
        print(
            "Sanitized action:",
            format_for_print(safe_action),
        )

        try:
            action_queue.put(
                safe_action,
                timeout=0.5,
            )

        except queue.Full:
            pass


# ---------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------

print(
    "Initializing LLM Humanoid Environment..."
)

controller = LLMHumanoidController(
    MODEL_PATH
)
recorder = SimulationRecorder(controller.model, controller.data)


# Ensure body positions and orientations are initialized
# before obtaining the first observation.
mujoco.mj_forward(
    controller.model,
    controller.data,
)

action_queue = queue.Queue(
    maxsize=1
)

controller_lock = threading.Lock()
history_lock = threading.Lock()
stop_event = threading.Event()

action_history = []

with controller_lock:
    initial_observation = copy.deepcopy(controller.get_obs()
)

initial_action = {
    "left_arm": initial_observation[
        "full_state"
    ]["left_arm_pos"],

    "right_arm": initial_observation[
        "full_state"
    ]["right_arm_pos"],

    "left_leg": initial_observation[
        "full_state"
    ]["left_leg_pos"],

    "right_leg": initial_observation[
        "full_state"
    ]["right_leg_pos"],
}

action_history.append(copy.deepcopy(initial_action))


worker_thread = threading.Thread(
    target=llm_worker,
    args=(
        action_queue,
        controller,
        controller_lock,
        action_history,
        history_lock,
        stop_event,
    ),
    daemon=True,
)

worker_thread.start()

MAX_ACTIONS = 15

# Fixed-duration poses are generally more suitable for
# continuous swimming than waiting for exact convergence.
STEPS_PER_ACTION = 700

VIEWER_SYNC_INTERVAL = 10
RECORD_INTERVAL = 5

executed_actions = 0

try:
    with mujoco.viewer.launch_passive(
        controller.model,
        controller.data,
    ) as viewer:

        while (
            viewer.is_running()
            and executed_actions < MAX_ACTIONS
        ):
            try:
                queued_action = (
                    action_queue.get(
                        timeout=0.1
                    )
                )

            except queue.Empty:
                viewer.sync()
                continue

            if not validate_action(
                queued_action
            ):
                print(
                    "Rejected invalid queued action."
                )
                continue

            # The humanoid may have moved while Ollama was
            # producing its response. Sanitize once more using
            # the newest state.
            with controller_lock:
                current_observation = (
                    copy.deepcopy(
                        controller.get_obs()
                    )
                )

            action = sanitize_action(
                queued_action,
            )

            executed_actions += 1

            with history_lock:
                action_history.append(
                    copy.deepcopy(action)
                )

            print()
            print("=" * 70)
            print(
                f"Executing LLM action "
                f"{executed_actions}/{MAX_ACTIONS}"
            )
            print(
                format_for_print(action)
            )


            print()
            print("=" * 70)
            print(
                f"Executing LLM action "
                f"{executed_actions}/{MAX_ACTIONS}"
            )
            print(
                format_for_print(action)
            )

            with controller_lock:
                controller.set_targets(action)

                print(
                    "Controller targets:",
                    {
                        body_name: target.tolist()
                        for body_name, target
                        in controller.current_targets.items()
                    },
                    flush=True,
                )

            target_reached_at_least_once = False

            for action_step in range(STEPS_PER_ACTION):
                if not viewer.is_running():
                    break

                with controller_lock:
                    targets_reached = controller.physics_step()
                    current_step = controller.step_count

                if targets_reached:
                    target_reached_at_least_once = True

                    print(
                        f"All targets reached after "
                        f"{action_step + 1} steps.",
                        flush=True,
                    )
                    break

                # Record only when a recorder exists.
                if (
                    "recorder" in globals()
                    and current_step
                    % RECORD_INTERVAL
                    == 0
                ):
                    with controller_lock:
                        recorder.record_frame()

                if (
                    current_step
                    % VIEWER_SYNC_INTERVAL
                    == 0
                ):
                    viewer.sync()

            with controller_lock:
                observation_after = (
                    copy.deepcopy(
                        controller.get_obs()
                    )
                )

            print(
                "Positions after action:",
                format_for_print(
                    observation_after[
                        "full_state"
                    ]
                ),
            )

            print(
                "Reached strict controller "
                "threshold:",
                target_reached_at_least_once,
            )

            print("=" * 70)

finally:
    stop_event.set()

    with controller_lock:
        controller.data.ctrl[:] = 0.0

    worker_thread.join(
        timeout=2.0
    )


recorder.save_video("aktuell.mp4")
  

print(
    "LLM swimming simulation complete."
)