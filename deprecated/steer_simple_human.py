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
MODEL_PATH = 'underwater_swimming_simple_human/human_laying.xml'
WATER_SURFACE_HEIGHT = 3.0
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

class LLMHumanoidController:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.step_count = 0
        self.current_targets = {} 
        
        # This maps the TIP of the limb to its ROTATION BASE (the parent body)
        self.limb_to_base = {
            'left_forearm': 'left_arm', 
            'right_forearm': 'right_arm',
            'left_calf': 'left_leg',
            'right_calf': 'right_leg',
        }

    def get_euler_angles(self):
        quat_mujoco = self.data.qpos[3:7] 
        quat_scipy = np.array([quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0]])
        rotation = R.from_quat(quat_scipy)
        euler = rotation.as_euler('xyz', degrees=False) 
        return euler 
    
    def get_local_pos(self, body_name):
        # 1. Find which base this body rotates around
        base_name = self.limb_to_base.get(body_name)
        if base_name is None:
            raise ValueError(f"Body {body_name} has no defined base in limb_to_base mapping")

        # 2. Get Global Positions
        base_pos = self.data.xpos[self.model.body(base_name).id]
        target_pos = self.data.xpos[self.model.body(body_name).id]
        
        # 3. Get the relative vector in global space
        relative_vec_global = target_pos - base_pos
        
        # 4. Transform global relative vector to local base frame
        base_mat_flat = self.data.xmat[self.model.body(base_name).id]
        base_mat = base_mat_flat.reshape(3, 3)
        relative_vec_local = base_mat.T @ relative_vec_global
        
        return relative_vec_local.tolist()

    def get_obs(self):
        head_height = self.data.xpos[self.model.body('head').id][2]
        torso_pos = self.data.xpos[self.model.body('torso').id]
        euler = self.get_euler_angles()
        forward_vel = self.data.qvel[0]

        return {
            "head_height": head_height,
            "body_height": torso_pos[2],
            "forward_vel": forward_vel,
            "step": self.step_count,
            "full_state": {
                "roll_pitch_yaw": euler,
                # Now we pass the correct TIP body names
                "left_arm_pos": self.get_local_pos('left_forearm'),
                "right_arm_pos": self.get_local_pos('right_forearm'),
                "left_leg_pos": self.get_local_pos('left_calf'),
                "right_leg_pos": self.get_local_pos('right_calf'),
            }
        }

    def set_targets(self, action_json):
        if not action_json: return
        # Map the LLM's simple names to the actual XML body names
        self.current_targets = {
            'right_forearm': action_json.get('right_arm'),
            'left_forearm': action_json.get('left_arm'),
            'right_calf': action_json.get('right_leg'),
            'left_calf': action_json.get('left_leg'),
        }
    
    def physics_step(self) -> bool:
        KP = 1400.0
        MAX_DELTA = 0.3
        THRESHOLD = MAX_DELTA * 0.5
        self.model.opt.timestep = 0.002
        ACCEPTABLE_ERROR = 0.001
        
        if not hasattr(self, 'prev_error'):
            self.prev_error = {body_name: float('inf') for body_name in self.current_targets.keys()}

        self.data.ctrl[:] = 0.0
        reached_cnt = 0

        for body_name, local_target in self.current_targets.items():
            if local_target is None:
                continue

            # USE THE CORRECT MAPPING HERE
            base_name = self.limb_to_base.get(body_name)
            base_id = self.model.body(base_name).id
            base_pos = self.data.xpos[base_id].copy()
            base_mat = self.data.xmat[base_id].reshape(3, 3)
            global_target = base_mat @ np.asarray(local_target, dtype=float) + base_pos

            body_id = self.model.body(body_name).id
            cur_pos = self.data.xpos[body_id].copy()

            error_vec = global_target - cur_pos
            dist = np.linalg.norm(error_vec)
            
            if dist <= THRESHOLD or (body_name in self.prev_error and self.prev_error[body_name] - round(dist, 5) <= ACCEPTABLE_ERROR):
                reached_cnt += 1
                continue
            
            self.prev_error[body_name] = dist

            step_vec = KP * error_vec * self.model.opt.timestep
            if np.linalg.norm(step_vec) > MAX_DELTA:
                step_vec = step_vec / np.linalg.norm(step_vec) * MAX_DELTA

            Jp = np.zeros((3, self.model.nv))
            Jr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, Jp, Jr, body_id)
            tau = Jp.T @ step_vec + Jr.T @ np.zeros(3)
            
            for jnt_id in range(self.model.njnt):
                if not self.model.jnt_limited[jnt_id]: continue
                dof_addr = self.model.jnt_dofadr[jnt_id]
                q = self.data.qpos[dof_addr]
                lo, hi = self.model.joint(jnt_id).range
                if q <= lo + 1e-6 and tau[dof_addr] < 0: tau[dof_addr] = 0.0
                if q >= hi - 1e-6 and tau[dof_addr] > 0: tau[dof_addr] = 0.0

            for act_id in range(self.model.nu):
                act = self.model.actuator(act_id)
                if act.trntype != mujoco.mjtTrn.mjTRN_JOINT: continue
                if len(act.trnid) > 0:
                    joint_id = act.trnid[0]
                    dof_addr = self.model.jnt_dofadr[joint_id]
                    if dof_addr < len(tau):
                        gear_val = float(act.gear[0])
                        if gear_val != 0.0:
                            self.data.ctrl[act_id] += tau[dof_addr] / gear_val

        self.data.ctrl[:] = np.clip(self.data.ctrl, -1.0, 1.0)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

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
                The coordinates are LOCAL to the base joint of the limb (Shoulder for arms, Hip for legs): X=Forward, Y=Left, Z=Up.
                Current State:
                - Head Height Error: {obs['head_height'] - WATER_SURFACE_HEIGHT:.2f} (Positive means head is above water)
                - Body Height Error: { obs['body_height'] -WATER_SURFACE_HEIGHT} (Negative means body is below water)
                - Orientation (R/P/Y): {obs['full_state']['roll_pitch_yaw']}
                If the roll is positive, the robot is leaning to its left; if negative, leaning to its right.
                If the pitch is positive, the robot is leaning backward; if negative, leaning forward.
                For the backstroke, the pitch should be near -90 (horizontal in its back).
                If the yaw is positive, the robot is facing left; if negative, facing right.
                - Limb Positions (Relative to Torso):
                L_Hand (x, y, z): {obs['full_state']['left_arm_pos']}
                R_Hand (x, y, z): {obs['full_state']['right_arm_pos']}
                eg
                L_Hand: [0, -0.01, -0.63] R_Hand: [0, 0.01, -0.63] arms parralel to the torso, pointing towards the feet

                L_Foot (x, y, z): {obs['full_state']['left_leg_pos']}
                R_Foot (x, y, z): {obs['full_state']['right_leg_pos']}
                If the x is positive, the foot is in front of the torso; if negative, behind.
                If the y is positive, the foot is to the left side of the torso; if negative, it is on the right side.
                If the z is positive, the left foot is above the torso; if negative, below

                Keep in mind that the movenets have to be incremental and within the physical limits of the humanoid.
                Moving the hand from in front of the torso eg. x = 0.5 to behind the torso x = -0.5 in one step is not physically possible.
                 
                Goal:
                1. Keep body horizontal (Pitch ~ -90) and head above water.
                2. Backstroke Motion: 
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

        # action = get_llm_action(prompt)
        # action = {'left_arm': [0.75, 0.15, 0.02],
        #         'right_arm': [0.75, -0.15, 0.02],
        #         'left_leg': [1.25, 0.02, -0.2],
        #         'right_leg': [1.25, -0.02, -0.2]}
        
        # action = {'left_arm': [0.35, 0.15, 0.06],
        #         'right_arm': [0.35, -0.15, 0.06],
        #         'left_leg': [-0.01, 0.09, -1.25],
        #         'right_leg': [-0.01, -0.09, -1.25]}
        # RIGHT: Use the LLM labels
        action = {
            'left_arm': [0, 0.0, 0.60],
            'right_arm': [0, 0.0, 0.60],
            'left_leg': [-0.01, 0.09, -1.25],
            'right_leg': [-0.01, -0.09, -1.25]
}
        
        print(f"obs: {obs['full_state']}")
        print("----")
        print(f"LLM Action: {action}")
        print("________________________")
        
        
        # helper to format a position list with 3‑decimal precision
        if action and all(k in action for k in ['left_arm', 'right_arm', 'left_leg', 'right_leg']):
            controller.set_targets(action)
            
            # 2. PHYSICS SUB-STEPPING
            targets_reached = False
            max_target_steps = controller.step_count + 500  # Limit to ~500 sub-steps to avoid long blocking
            
            while not targets_reached and controller.step_count < max_target_steps:
                targets_reached = controller.physics_step()
                if controller.step_count % 10 == 0:
                    viewer.sync()                           # update GUI

        else:
            print("Invalid action. Skipping...")
            for _ in range(1000): 
                controller.physics_step()
        time.sleep(1)  # slow down the loop for better visualization
        print(f"Finished step: {obs['full_state']}")

        