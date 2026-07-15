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
import cv2  # <--- Required for video saving: pip install opencv-python

# --- Configuration ---
MODEL_PATH = 'underwater_swimming_simple_human/humanoid_laying.xml'
WATER_SURFACE_HEIGHT = 3.0
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"
VIDEO_FILENAME = "humanoid_swim.mp4"
prev_act = []

class SimulationRecorder:
    """Handles recording frames and saving them to a video file."""
    def __init__(self, model, data, width=1280, height=720): # <--- Added width and height params
        self.model = model
        self.data = data
        # Initialize renderer with specific resolution
        self.renderer = mujoco.Renderer(model, width=width, height=height) 
        self.frames = []

    def record_frame(self):
        """Captures the current scene as an RGB image."""
        self.renderer.update_scene(self.data)
        pixels = self.renderer.render()
        # MuJoCo renders in RGB, OpenCV expects BGR
        frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        self.frames.append(frame_bgr)

    def save_video(self, filename, fps=30):
        """Saves all recorded frames to an mp4 file."""
        if not self.frames:
            print("No frames recorded!")
            return

        height, width, layers = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video = cv2.VideoWriter(filename, fourcc, fps, (width, height))

        print(f"Saving video to {filename}...")
        for frame in self.frames:
            video.write(frame)
        video.release()
        print("Video saved successfully.")

# ... [LLMHumanoidController class remains exactly as before] ...
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
        KP, KI, MAX_DELTA, THRESHOLD, ACCEPTABLE_ERROR = 700.0, 40.0, 2, 0.05, 0.001
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
                    if gear_val != 0.0: self.data.ctrl[act_id] += (tau[dof_addr] * gear_val) * 0.0035

        # self.data.ctrl[:] = np.clip(self.data.ctrl, -1.0, 1.0)
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
        prompt = f"""You are an AI controlling a humanoid robot learning to swim backstroke.
                The coordinates are LOCAL to the torso: 
                Current State:
                - Head Height Error: {obs['head_height'] - WATER_SURFACE_HEIGHT:.2f} (Positive means head is above water)
                - Body Height Error: { obs['body_height'] -WATER_SURFACE_HEIGHT} (Negative means body is below water)
                
                - Limb Positions (Relative to Torso):
                L_Hand (x, y, z): {obs['full_state']['left_arm_pos']}
                R_Hand (x, y, z): {obs['full_state']['right_arm_pos']}
                eg.:
                L_Hand: [0, 0.01, -0.63] R_Hand: [0, 0.01, -0.63] arms parallel to the torso, pointing towards the feet
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
                - Arms: The arms alternatingly perform the following motion. While nearly stretched out, pull up from beside the hip towards the sky. Then enter the Water over the head and pull down towards the floor, befor exiting the water beside the hip.
                - Legs: Alternatingly, move one leg slightly up while the other is moved slightly down. Both legs are stretched out

                Physical Constraints (STRICT):
                - Move incrementally. Max change per step: 0.1m.
                - Reach Limit: Do not suggest coordinates that exceed the physical length of the limbs.
                - Arms: Max reach is approx 0.6m from the shoulder. Keep X and Z within [-0.6, 0.6] .
                - Legs: Max reach is approx 1m from the hip. Keep X and Y within [-0.3, 0.3] and Z within [-1, 0.1].

                Constraint: 
                Return ONLY a valid JSON object. Do not include any conversational text, explanation, or nested keys.

                Example Output:
                {{"left_arm": [0.1, 0.1, -0.6] , "right_arm": [0.1, 0.1, -0.6], "left_leg": [0.2, 0.09, -0.7], "right_leg": [-0.05, -0.09, -0.7]}}

                Your Turn (JSON only):"""
        prompt += f"For reference: The last 10 actions you chose: {prev_act[10:]}"
        action = get_llm_action(prompt)
        print("addded action to action queue")
        if action: action_queue.put(action)
        else: time.sleep(1)

# --- Main Execution ---
controller = LLMHumanoidController(MODEL_PATH)
recorder = SimulationRecorder(controller.model, controller.data) # Initialize recorder
action_queue = queue.Queue(maxsize=1)

worker_thread = threading.Thread(target=llm_worker, args=(action_queue, controller), daemon=True)
worker_thread.start()

with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
    while viewer.is_running():
        if len(prev_act) > 25: break

        try:
            action = action_queue.get(block=False)
            prev_act.append(action)
            print(action)
        except queue.Empty:
            time.sleep(0.1)
            continue

        if action and (k in action for k in ['left_arm', 'right_arm', 'left_leg', 'right_leg']):
            controller.set_targets(action)
            targets_reached = False
            max_target_steps = controller.step_count + 400
            
            while not targets_reached and controller.step_count < max_target_steps:
                targets_reached = controller.physics_step()
                
                # RECORD every X steps to keep video file size reasonable
                if controller.step_count % 5 == 0: 
                    recorder.record_frame() 
                
                if controller.step_count % 10 == 0:
                    viewer.sync()
        else:
            time.sleep(0.1)
        print(controller.get_obs(), controller.step_count)

# FINAL STEP: Save the video after the viewer closes
recorder.save_video(VIDEO_FILENAME)