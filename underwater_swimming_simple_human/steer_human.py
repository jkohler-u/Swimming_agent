import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import mujoco
import mujoco.viewer
import numpy as np
import requests
import json
import time
from scipy.spatial.transform import Rotation as R  # <--- Add this import


# --- Configuration ---
MODEL_PATH = 'underwater_swimming_simple_human/humanoid_laying.xml'
WATER_SURFACE_HEIGHT = 3.0
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"
class LLMHumanoidController:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.step_count = 0
        self.current_targets = {} # Store targets here
        self.limb_bases = {
        'hand_left': 'upper_arm_left',   # The "shoulder" area
        'hand_right': 'upper_arm_right',
        'foot_left': 'thigh_left',      # The "hip" area
        'foot_right': 'thigh_right',
        }
        self.integral_error = {limb: np.zeros(3) for limb in self.limb_bases.keys()}

    def get_euler_angles(self):
        # 1. Get the quaternion from qpos. 
        # For a standard humanoid, the root orientation is usually indices 3, 4, 5, 6
        quat_mujoco = self.data.qpos[3:7] 
        
        # 2. Reorder from MuJoCo [w, x, y, z] to Scipy [x, y, z, w]
        quat_scipy = np.array([quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0]])
        
        # 3. Convert to Euler angles (radians)
        # 'xyz' means extrinsic rotations; 'zyx' is common for Yaw-Pitch-Roll
        rotation = R.from_quat(quat_scipy)
        euler = rotation.as_euler('xyz', degrees=False) 
        
        return euler # returns [roll, pitch, yaw]
    
    def get_local_pos(self, body_name):
        """
        Transforms a body's global position into a position relative 
        to its BASE joint's current position and orientation.
        """
        # 1. Get the base body name from our mapping
        base_name = self.limb_bases.get(body_name)
        if base_name is None:
            raise ValueError(f"Body {body_name} has no defined base in limb_bases mapping")

        # 2. Get Global Positions
        base_pos = self.data.xpos[self.model.body(base_name).id]
        target_pos = self.data.xpos[self.model.body(body_name).id]
        
        # 3. Get the relative vector in global space
        relative_vec_global = target_pos - base_pos
        
        # 4. Get the rotation matrix of the BASE joint
        base_mat_flat = self.data.xmat[self.model.body(base_name).id]
        base_mat = base_mat_flat.reshape(3, 3)
        
        # 5. Transform global relative vector to local base frame
        relative_vec_local = base_mat.T @ relative_vec_global
        
        return relative_vec_local.tolist()
    

    def get_obs(self):
        """Extracts state exactly like your current_info dictionary."""
        # body heights/positions
        head_height = self.data.xpos[self.model.body('head').id][2]
        torso_pos = self.data.xpos[self.model.body('torso').id]
        euler = self.get_euler_angles()
        # Calculate forward velocity (Assuming X is forward)
        forward_vel = self.data.qvel[0] # Simple root velocity

        return {
            "head_height": head_height,
            "body_height": torso_pos[2],
            "forward_vel": forward_vel,
            "step": self.step_count,
            "full_state": {
                "roll_pitch_yaw": euler,
                # These are now coordinates relative to the torso center
                "left_arm_pos": self.get_local_pos('hand_left'),
                "right_arm_pos": self.get_local_pos('hand_right'),
                "left_leg_pos": self.get_local_pos('foot_left'),
                "right_leg_pos": self.get_local_pos('foot_right'),
             
            }
        }

    def set_targets(self, action_json):
        """Just stores the targets from the LLM and resets the integral"""
        if not action_json: return
        self.current_targets = {
            'hand_right': action_json.get('right_arm'),
            'hand_left': action_json.get('left_arm'),
            'foot_right': action_json.get('right_leg'),
            'foot_left': action_json.get('left_leg'),
        }
        # RESET integral error whenever a new goal is set
        for limb in self.integral_error:
            self.integral_error[limb] = np.zeros(3)
    
    def physics_step(self) -> bool:
        """
        Move every limb that currently has a target toward that target using a
        Jacobian‑transpose controller.  Returns True when all targets are within
        THRESHOLD distance.
        """
        # ------------------------------------------------------------------
        # 0 – constants
        # ------------------------------------------------------------------
        KP = 500.0      
        KI = 40.0              
        MAX_DELTA = 2              # maximal linear displacement per step (m)
        THRESHOLD =  0.05   # distance under which a target is considered reached
        self.model.opt.timestep = 0.002
        ACCEPTABLE_ERROR = 0.001                 # max distance from target before we consider it "too far" (m)
        self.prev_error = {body_name: float('inf') for body_name in self.current_targets.keys()}  # Initialize previous errors


        # ------------------------------------------------------------------
        # 2 – start with a clean control vector (once per physics step)
        # ------------------------------------------------------------------
        self.data.ctrl[:] = 0.0
        reached_cnt = 0

        # ------------------------------------------------------------------
        # 3 – loop over every limb that has a (non‑None) target
        # ------------------------------------------------------------------
        for body_name, local_target in self.current_targets.items():
            if local_target is None:
                continue

            # ---------- a) local (relative to joint) → global ----------
            base_name = self.limb_bases.get(body_name)
            base_id = self.model.body(base_name).id
            base_pos = self.data.xpos[base_id].copy()
            base_mat = self.data.xmat[base_id].reshape(3, 3)
            global_target = base_mat @ np.asarray(local_target, dtype=float) + base_pos

            # ---------- b) current body position ----------
            body_id = self.model.body(body_name).id
            cur_pos = self.data.xpos[body_id].copy()

            # ---------- c) error → desired step ----------
            error_vec = global_target - cur_pos
            dist = np.linalg.norm(error_vec)
            
            
            # ---------- d) already close enough? ----------
            if dist <= THRESHOLD or self.prev_error[body_name] - round(np.linalg.norm(error_vec), 5) <= ACCEPTABLE_ERROR:
                reached_cnt += 1
                # print(f"{body_name} reached target or is too far from target. Error: {round(dist, 5)}")
                continue
            self.prev_error[body_name] = dist  # Update previous error for this limb


            # ---------- e) PI Controller Logic ----------
            # 1. Update Integral (Sum of error * time)
            self.integral_error[body_name] += error_vec * self.model.opt.timestep
            
            # 2. Anti-Windup: Clamp the integral so it doesn't grow infinitely
            # This prevents the robot from "exploding" after being stuck
            max_integral = 1.0 
            norm_int = np.linalg.norm(self.integral_error[body_name])
            if norm_int > max_integral:
                self.integral_error[body_name] = (self.integral_error[body_name] / norm_int) * max_integral

            # 3. Calculate combined step: P + I
            # step_vec = (Kp * error) + (Ki * integral)
            step_vec = (KP * error_vec + KI * self.integral_error[body_name]) * self.model.opt.timestep

            # ---------- f) clamp step length ----------
            if np.linalg.norm(step_vec) > MAX_DELTA:
                step_vec = step_vec / np.linalg.norm(step_vec) * MAX_DELTA

            # ---------- f) Jacobian‑transpose torque ----------
            Jp = np.zeros((3, self.model.nv))   # linear part
            Jr = np.zeros((3, self.model.nv))   # angular part           # (3 × nv) position Jacobian
            mujoco.mj_jacBody(self.model, self.data, Jp, Jr, body_id)
            tau = Jp.T @ step_vec + Jr.T  @ np.zeros(3)
            
            # ---------- g) distribute torque onto actuators ----------
            # Iterate through all joints in the model
                
            for jnt_id in range(self.model.njnt):
                if not self.model.jnt_limited[jnt_id]:
                    continue
                dof_addr = self.model.jnt_dofadr[jnt_id]
                q = self.data.qpos[dof_addr]
                lo, hi = self.model.joint(jnt_id).range
                # if body_name == "hand_right" and self.step_count % 50 == 0:  # Print every 50 steps to reduce clutter
                #     print(f"Joint {self.model.joint(jnt_id).name} → q: {round(q, 5)}, range: [{round(lo, 5)}, {round(hi, 5)}], tau: {round(tau[dof_addr], 5)}")
                if q <= lo + 1e-6 and tau[dof_addr] < 0:
                    tau[dof_addr] = 0.0
                    # print("Limited")
                if q >= hi - 1e-6 and tau[dof_addr] > 0:
                    tau[dof_addr] = 0.0
                    # print("Limited")

            for act_id in range(self.model.nu):
                act = self.model.actuator(act_id)

                # Keep only actuators that are transmitted to a joint.
                if act.trntype != mujoco.mjtTrn.mjTRN_JOINT:
                    continue

                if len(act.trnid) > 0:
                        joint_id = act.trnid[0]
                        dof_addr = self.model.jnt_dofadr[joint_id]
                        # Safety check for index range
                        if dof_addr < len(tau):
                            torque_on_joint = tau[dof_addr]
                            
                            gear_val = float(act.gear[0])
                            if gear_val != 0.0:
                                self.data.ctrl[act_id] += torque_on_joint / gear_val * 10

                            # if body_name == "hand_right" and self.step_count % 50 == 0 and torque_on_joint != 0:  # Print every 50 steps to reduce clutter
                            #     print(f"Actuator {self.model.actuator(act_id).name} → Torque {round(torque_on_joint, 5)} / Gear {round(gear_val, 5)} = Ctrl {round(self.data.ctrl[act_id], 5)}")
                            
        
            # if body_name == "hand_right" and self.step_count % 50 == 0:  # Print every 50 steps to reduce clutter
            #         print(f" {cur_pos[0]:.2f}, {cur_pos[1]:.2f}, {cur_pos[2]:.2f} → Target {global_target[0]:.2f}, {global_target[1]:.2f}, {global_target[2]:.2f}: "
            #             f"error = {round(np.linalg.norm(error_vec), 5)}, "
            #             f"step = {round(np.linalg.norm(step_vec), 5)}"
            #             # f", torque = {round(np.linalg.norm(tau), 5)}"
            #             f", close = {reached_cnt}")
        # ------------------------------------------------------------------
        # 4 – clip, step the simulation and sync the viewer
        # ------------------------------------------------------------------
        self.data.ctrl[:] = np.clip(self.data.ctrl, -1.0, 1.0)
        for _ in range(5):  # 5 sub‑steps for stability
            mujoco.mj_step(self.model, self.data)   # ONE simulation sub‑step
        self.step_count += 1

        # ------------------------------------------------------------------
        # 5 – have *all* limbs reached their targets?
        # ------------------------------------------------------------------
        return reached_cnt == len(self.current_targets)

def get_llm_action(prompt):
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False, "format": "json"}
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        return json.loads(response.json()['response'])
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

# --- Main Execution ---
print("Initializing LLM Humanoid Environment...")
controller = LLMHumanoidController(MODEL_PATH)
runs = 0
# Launch Passive Viewer
with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
    while viewer.is_running():
        runs += 1
        if runs > 20:
            viewer.close()
            break
        obs = controller.get_obs()
        
        # 1. Get Action from LLM (Slow part)
        prompt = f"""You are an AI controlling a humanoid robot learning to swim backstroke.
                The coordinates are LOCAL to the base joint of the limb (Shoulder for arms, Hip for legs): 
                Current State:
                - Head Height Error: {obs['head_height'] - WATER_SURFACE_HEIGHT:.2f} (Positive means head is above water)
                - Body Height Error: { obs['body_height'] -WATER_SURFACE_HEIGHT} (Negative means body is below water)
                
                - Limb Positions (Relative to Torso):
                L_Hand (x, y, z): {obs['full_state']['left_arm_pos']}
                R_Hand (x, y, z): {obs['full_state']['right_arm_pos']}
                eg
                L_Hand: [0, 0.01, -0.63] R_Hand: [0, 0.01, -0.63] arms parralel to the torso, pointing towards the feet
                if x is positive - the hand reaches up towards the sky, if its negative, it reaches towards the floor
                if y is positive . the hand reaches towards the left, if its negative, it reaches towards the right
                if z is positive - the hand reaches above the head, if its negative it reaches towards the feet (see example)

                L_Foot (x, y, z): {obs['full_state']['left_leg_pos']}
                R_Foot (x, y, z): {obs['full_state']['right_leg_pos']}
                L_Foot: [-0.2, -0.01, -1] - the leg is extended, pointing slightly below the torso towards the floor
                If the x is positive, the foot is reaching towards the sky, if its negative the foot reaches towards the floor.
                If the y is positive, the foot is to the left side of the torso; if negative, it is on the right side.
                If the z is positive, the left foot is trying to reach towards the the torso; if negative, it is streched out away from the torso.

                Keep in mind that the movenets have to be incremental and within the physical limits of the humanoid.
                Moving the hand from in front of the torso eg. x = 0.5 to behind the torso x = -0.5 in one step is not physically possible.
                 
                Goal:
                You start out lying on your back at the water surface, looking up to the sky.
                 Backstroke Motion: 
                - Arms: One pulls back in water (S-shape), one recovers in a semi-circle above water.
                - Legs: Alternating flutter kick (one up, one down).

                Physical Constraints (STRICT):
                - Reach Limit: Do not suggest coordinates that exceed the physical length of the limbs.
                - Arms: Max reach is approx 0.75m from the shoulder. Keep X and Z within [-0.75, 0.75] and Y within [-0.65, 0.65].
                - Legs: Max reach is approx 1.27m from the hip. Keep X and Y within [-0.3, 0.3] and Z within [-1.27, 0.1].
                - Avoid "teleporting" limbs; suggest movements that are incremental from the current state.

                Constraint: 
                Return ONLY a valid JSON object. Do not include any conversational text, explanation, or nested keys.

                Example Output:
                {{"left_arm": [0.2, 0.3, 0.1] # , "right_arm": [-0.2, -0.3, 0.1], "left_leg": [0.1, 0.1, -0.5], "right_leg": [0.1, -0.1, -0.5]}}

                Your Turn (JSON only):"""

        action = get_llm_action(prompt)
        # action = {'left_arm': [0.75, 0.15, 0.02],
        #         'right_arm': [0.75, -0.15, 0.02],
        #         'left_leg': [1.25, 0.02, -0.2],
        #         'right_leg': [1.25, -0.02, -0.2]}
        
        # action = {'left_arm': [0.35, 0.15, 0.06],
        #         'right_arm': [0.35, -0.15, 0.06],
        #         'left_leg': [-0.01, 0.09, -1.25],
        #         'right_leg': [-0.01, -0.09, -1.25]}
        # if runs % 5 == 1:
        #     # hands parallel to the body, hands pointing ot feet - works 
        #     action = {'left_arm': [0.1, 0.1, -0.6],
        #             'right_arm': [0.1, -0.1, -0.6],
        #             'left_leg': [0.2, 0.09, -1],
        #             'right_leg': [-0.2, -0.09, -1]}
        # elif runs % 5 == 2:
        #     action = {'left_arm': [0.3, 0, -0.3],
        #             'right_arm': [0.3, 0, -0.3],
        #             'left_leg': [0.2, 0.09, -1],
        #             'right_leg': [-0.2, -0.09, -1]}
        # elif runs % 5 == 3:
        #    # arms outstreched, hands pointing away from the body - works
        #    action = {'left_arm': [0.7, 0.1, 0],
        #             'right_arm': [0.7, -0.1, 0],
        #             'left_leg': [0.2, 0.09, -1],
        #             'right_leg': [-0.2, -0.09, -1]}
        # elif runs % 5 == 4:
        #     action = {'left_arm': [-0.3, 0.1, -0.3],
        #             'right_arm': [-0.3, -0.1, -0.3],
        #             'left_leg': [0.2, 0.09, -1],
        #             'right_leg': [-0.2, -0.09, -1]}
        # elif runs % 5 == 0:
        #     action = {'left_arm': [0, 0.1, 0.6],
        #             'right_arm': [0, -0.1, 0.6],
        #             'left_leg': [0.2, 0.09, -1],
        #             'right_leg': [-0.2, -0.09, -1]}
       

        def format_for_print(value):
            if isinstance(value, dict):
                return {k: format_for_print(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [format_for_print(v) for v in value]
            if isinstance(value, float):
                return round(value, 3)
            return value
            
        print(f"LLM Action: {format_for_print(action)}")
        print(f"END {runs % 5}________________________")
        
        
        # helper to format a position list with 3‑decimal precision
        if action and (k in action for k in ['left_arm', 'right_arm', 'left_leg', 'right_leg']):
            controller.set_targets(action)
            
            # 2. PHYSICS SUB-STEPPING
            targets_reached = False
            max_target_steps = controller.step_count + 1000  # Limit to ~500 sub-steps to avoid long blocking
            
            while not targets_reached and controller.step_count < max_target_steps:
                targets_reached = controller.physics_step()
                if controller.step_count % 10 == 0:
                    viewer.sync()                           # update GUI

        else:
            print("Invalid action. Skipping...")
            for _ in range(1000): 
                controller.physics_step()
        time.sleep(1)  # slow down the loop for better visualization
        print(f"L-ARM: {obs['full_state']['right_arm_pos']}\n",
              f"R-ARM: {obs['full_state']['left_arm_pos']})\n",
              f"L_LEG: {obs['full_state']['right_leg_pos']}\n)",
              f"R_LEG: {obs['full_state']['left_leg_pos']})\n")

        