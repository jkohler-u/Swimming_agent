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
        self.current_targets = {}
        self.limb_bases = {
            'hand_left': 'upper_arm_left', 'hand_right': 'upper_arm_right',
            'foot_left': 'thigh_left', 'foot_right': 'thigh_right',
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

    def set_targets(self, action_json):
        if not action_json: return
        self.current_targets = {
            'hand_right': action_json.get('right_arm'), 'hand_left': action_json.get('left_arm'),
            'foot_right': action_json.get('right_leg'), 'foot_left': action_json.get('left_leg'),
        }
        for limb in self.integral_error: self.integral_error[limb] = np.zeros(3)
        self.prev_error = {body_name: float('inf') for body_name in self.current_targets.keys()}
    
    def physics_step(self) -> bool:
        KP, KI, MAX_DELTA, THRESHOLD, ACCEPTABLE_ERROR = 1000.0, 40.0, 2, 0.05, 0.001
        self.model.opt.timestep = 0.002
        self.data.ctrl[:] = 0.0
        reached_cnt = 0
        for body_name, local_target in self.current_targets.items():
            if local_target is None: continue
            base_id = self.model.body(self.limb_bases[body_name]).id
            global_target = self.data.xmat[base_id].reshape(3, 3) @ np.asarray(local_target) + self.data.xpos[base_id]
            body_id = self.model.body(body_name).id
            error_vec = global_target - self.data.xpos[body_id]
            dist = np.linalg.norm(error_vec)
           
           
            # if dist <= THRESHOLD or self.prev_error.get(body_name, float('inf')) - round(dist, 5) <= ACCEPTABLE_ERROR:
            #     reached_cnt += 1
            #     continue
            self.prev_error[body_name] = dist 

            # PI controller
            self.integral_error[body_name] += error_vec * self.model.opt.timestep
            step_vec = (KP * error_vec + KI * self.integral_error[body_name]) * self.model.opt.timestep

            #caping movement
            if np.linalg.norm(step_vec) > MAX_DELTA: step_vec = step_vec / np.linalg.norm(step_vec) * MAX_DELTA

            # calculate force
            Jp = np.zeros((3, self.model.nv))
            Jr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, Jp, Jr, body_id)
            tau = Jp.T @ step_vec

            # apply force across joints
            for act_id in range(self.model.nu):
                act = self.model.actuator(act_id)
                if act.trntype == mujoco.mjtTrn.mjTRN_JOINT and len(act.trnid) > 0:
                    dof_addr = self.model.jnt_dofadr[act.trnid[0]]
                    gear_val = float(act.gear[0])
                    if gear_val != 0.0: self.data.ctrl[act_id] += (tau[dof_addr] * gear_val) * 0.004

        # self.data.ctrl[:] = np.clip(self.data.ctrl, -1.0, 1.0)
        for _ in range(5): mujoco.mj_step(self.model, self.data)
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

        
        if runs % 5 == 1:
            # hands parallel to the body, hands pointing ot feet - works 
            action = {'left_arm': [0.1, 0.1, -0.6],
                    'right_arm': [0.1, -0.1, -0.6],
                    'left_leg': [0.2, 0.09, -0.7],
                    'right_leg': [-0.05, -0.09, -0.7]}
        elif runs % 5 == 2:
            action = {'left_arm': [0.1, 0.1, 0.7],
                    'right_arm': [0.1, -0.1, 0.7],
                    'left_leg': [-0.05, 0.09, -0.7],
                    'right_leg': [0.2, -0.09, -0.7]}
        elif runs % 5 == 3:
           # arms outstreched, hands pointing away from the body - works
           action = {'left_arm': [0.5, 0.1, 0],
                    'right_arm': [0.5, -0.1, 0],
                    'left_leg': [0.2, 0.09, -0.7],
                    'right_leg': [-0.05, -0.09, -0.7]}
        elif runs % 5 == 4:
            action = {'left_arm': [0.3, 0.1, -0.3],
                    'right_arm': [0.3, -0.1, -0.3],
                    'left_leg': [-0.05, 0.09, -0.7],
                    'right_leg': [0.2, -0.09, -0.7]}
        elif runs % 5 == 0:
            action = {'left_arm': [0, 0.1, 0.6],
                    'right_arm': [0, -0.1, 0.6],
                    'left_leg': [0.2, 0.09, -0.7],
                    'right_leg': [-0.05, -0.09, -0.7]}
       

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
            max_target_steps = controller.step_count + 700  # Limit to ~500 sub-steps to avoid long blocking
            
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

        