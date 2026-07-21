import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import mujoco
import mujoco.viewer
import numpy as np
import requests
import json
import time
import threading
import queue
from scipy.spatial.transform import Rotation as R
import cv2

# --- Configuration ---
MODEL_PATH = 'swimming_llm/humanoid_laying.xml'
WATER_SURFACE_HEIGHT = 3.0
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"
VIDEO_FILENAME = "humanoid_swim_experimental.mp4"
prev_act = []

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
        
        # Movement increment
        self.STEP_SIZE = 0.15 


    def get_local_pos(self, body_name):
        """Position relative to limb base"""
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
                "left_arm": self.get_local_pos('hand_left'),
                "right_arm": self.get_local_pos('hand_right'),
                "left_leg": self.get_local_pos('foot_left'),
                "right_leg": self.get_local_pos('foot_right'),
            }
        }
    
    def get_readable_state(self):
        """Translates raw coordinates into human-readable descriptions for the LLM."""
        obs = self.get_obs()
        state = {}
        
        for limb, pos in obs['full_state'].items():
            x, y, z = pos
            desc = []
            
            # X-Axis: Sky vs Ground
            if x > 0.2: desc.append("high (sky-wards)")
            elif x < -0.2: desc.append("low (ground-wards)")
            else: desc.append("neutral height")
            
            # Y-Axis: Left vs Right
            if y > 0.2: desc.append("shifted left")
            elif y < -0.2: desc.append("shifted right")
            else: desc.append("centered")
            
            # Z-Axis: Head vs Foot
            if z > 0.1: desc.append("reaching head-wards")
            elif z < -0.5: desc.append("stretched foot-wards")
            else: desc.append("mid-torso")
            
            state[limb] = ", ".join(desc)
        return state

    def translate_action_to_coords(self, action_json):
        """Converts discrete directions into incremental coordinates."""
        obs = self.get_obs()
        new_targets = {}
        
        # Mapping of directions to axis index and sign [axis_idx, sign]
        # x: sky(+) / ground(-) | y: left(+) / right(-) | z: head(+) / foot(-)
        direction_map = {
            "sky-wards": [0, 1], "ground-wards": [0, -1],
            "left-wards": [1, 1], "right-wards": [1, -1],
            "head-wards": [2, 1], "foot-wards": [2, -1],
            "stay": None
        }

        limb_mapping = {
            'left_arm': 'hand_left', 'right_arm': 'hand_right',
            'left_leg': 'foot_left', 'right_leg': 'foot_right'
        }


        for action_key, direction in action_json.items():
            if action_key not in limb_mapping: continue
            
            current_pos = np.array(obs['full_state'][action_key])
            if direction in direction_map and direction_map[direction] is not None:
                axis, sign = direction_map[direction]
                new_pos = current_pos.copy()
                new_pos[axis] += sign * self.STEP_SIZE
                
                if 'arm' in action_key:
                    new_pos = np.clip(new_pos, [-0.6, -0.3, -0.6], [0.6, 0.3, 0.6])
                else:
                    new_pos = np.clip(new_pos, [-0.3, -0.3, -1.0], [0.3, 0.3, 1.0])
                
                new_targets[limb_mapping[action_key]] = new_pos.tolist()
            else:
                new_targets[limb_mapping[action_key]] = current_pos.tolist() # keep position
        
        return new_targets

    def set_targets(self, action_json):
        if not action_json: return
        self.current_targets = self.translate_action_to_coords(action_json)
        
        for limb in self.integral_error: self.integral_error[limb] = np.zeros(3)
        self.prev_error = {body_name: float('inf') for body_name in self.current_targets.keys()}
    
    def physics_step(self) -> bool:
        """ Move limbs to new positions """
        KP, KI, MAX_DELTA, THRESHOLD = 1000.0, 40.0, 2, 0.05
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
           
            if dist <= THRESHOLD:
                reached_cnt += 1
                continue

            self.prev_error[body_name] = dist 
            self.integral_error[body_name] += error_vec * self.model.opt.timestep
            step_vec = (KP * error_vec + KI * self.integral_error[body_name]) * self.model.opt.timestep

            if np.linalg.norm(step_vec) > MAX_DELTA: 
                step_vec = step_vec / np.linalg.norm(step_vec) * MAX_DELTA

            Jp = np.zeros((3, self.model.nv))
            Jr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, Jp, Jr, body_id)
            tau = Jp.T @ step_vec

            for act_id in range(self.model.nu):
                act = self.model.actuator(act_id)
                if act.trntype == mujoco.mjtTrn.mjTRN_JOINT and len(act.trnid) > 0:
                    dof_addr = self.model.jnt_dofadr[act.trnid[0]]
                    gear_val = float(act.gear[0])
                    if gear_val != 0.0: self.data.ctrl[act_id] += (tau[dof_addr] * gear_val) * 0.0035

        for _ in range(5): mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        return reached_cnt == len(self.current_targets)

def get_llm_action(prompt):
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False, "format": "json"}
    try:
        return json.loads(requests.post(OLLAMA_URL, json=payload).json()['response'])
    except: return None

def llm_worker(action_queue, controller):
    while True:
        obs = controller.get_obs()
        readable_state = controller.get_readable_state() # <--- Get translated coordinates
        print(readable_state)
        prompt = f"""You are an AI controlling a humanoid robot learning backstroke swimming.
                The robot is lying on its back. 
                
                Current State:
                - Head Height: {obs['head_height'] - WATER_SURFACE_HEIGHT:.2f} (Positive = above water)
                - Body Height: {obs['body_height'] - WATER_SURFACE_HEIGHT:.2f} (Negative = submerged)
                - Limb Positions:
                {json.dumps(readable_state, indent=4)}

                You must choose from these PRESET DIRECTIONS for each limb:
                - "sky-wards": Move limb UP towards the sky.
                - "ground-wards": Move limb DOWN towards the floor.
                - "left-wards": Move limb to the left.
                - "right-wards": Move limb to the right.
                - "head-wards": Move limb towards the head.
                - "foot-wards": Move limb towards the feet.
                - "stay": Keep the limb where it is.

                Goal: Backstroke. 
                - Movements only happen incrementally, you might have to move in the same direction multiple etimes 
                - Arms: Alternatingly pull from hip -> sky -> over head -> floor -> hip.
                - Legs: Alternatingly  up/down movements, stretched out.

                Return ONLY a JSON object.
                Example: {{"left_arm": "sky-wards", "right_arm": "foot-wards", "left_leg": "sky-wards", "right_leg": "floor-wards"}}
                
                Your Turn (JSON only):"""
        
        prompt += f"\nLast few actions: {prev_act[-5:]}"
        action = get_llm_action(prompt)
        if action: action_queue.put(action)
        else: time.sleep(1)

# --- Main Execution ---
controller = LLMHumanoidController(MODEL_PATH)
recorder = SimulationRecorder(controller.model, controller.data)
action_queue = queue.Queue(maxsize=1)

worker_thread = threading.Thread(target=llm_worker, args=(action_queue, controller), daemon=True)
worker_thread.start()

with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
    while viewer.is_running():
        if len(prev_act) > 10: break # Increased for the experimental run

        try:
            action = action_queue.get(block=False)
            prev_act.append(action)
            print(f"LLM Command: {action}")
        except queue.Empty:
            time.sleep(0.1)
            continue

        if action and any(k in action for k in ['left_arm', 'right_arm', 'left_leg', 'right_leg']):
            controller.set_targets(action)
            targets_reached = False
            max_target_steps = controller.step_count + 400
            
            while not targets_reached and controller.step_count < max_target_steps:
                targets_reached = controller.physics_step()
                if controller.step_count % 5 == 0: 
                    recorder.record_frame() 
                if controller.step_count % 10 == 0:
                    viewer.sync()
        else:
            time.sleep(0.1)

recorder.save_video(VIDEO_FILENAME)