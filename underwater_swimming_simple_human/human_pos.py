import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import mujoco
import mujoco.viewer
import numpy as np
import time

# --- Configuration ---
MODEL_PATH = 'underwater_swimming_simple_human/humanoid_laying.xml'

class PoseTester:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.step_count = 0
        self.current_targets = {}
        self.limb_bases = {
            'hand_left': 'upper_arm_left',
            'hand_right': 'upper_arm_right',
            'foot_left': 'thigh_left',
            'foot_right': 'thigh_right',
        }
        self.integral_error = {limb: np.zeros(3) for limb in self.limb_bases.keys()}

    def set_targets(self, action_json):
        self.current_targets = {
            'hand_right': action_json.get('right_arm'),
            'hand_left': action_json.get('left_arm'),
            'foot_right': action_json.get('right_leg'),
            'foot_left': action_json.get('left_leg'),
        }
        for limb in self.integral_error:
            self.integral_error[limb] = np.zeros(3)

    def physics_step(self) -> bool:
        # Copy-pasted logic from your controller to ensure identical behavior
        KP, KI = 400.0, 50.0
        MAX_DELTA = 0.3
        THRESHOLD = MAX_DELTA * 0.5
        self.model.opt.timestep = 0.002
        
        self.data.ctrl[:] = 0.0
        reached_cnt = 0

        for body_name, local_target in self.current_targets.items():
            if local_target is None: continue

            base_name = self.limb_bases.get(body_name)
            base_id = self.model.body(base_name).id
            base_pos = self.data.xpos[base_id].copy()
            base_mat = self.data.xmat[base_id].reshape(3, 3)
            global_target = base_mat @ np.asarray(local_target, dtype=float) + base_pos

            body_id = self.model.body(body_name).id
            cur_pos = self.data.xpos[body_id].copy()
            error_vec = global_target - cur_pos
            dist = np.linalg.norm(error_vec)
            
            if dist <= THRESHOLD:
                reached_cnt += 1
                continue

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

# ==========================================
# EDIT YOUR TEST POSES HERE
# ==========================================
TEST_POSES = [
    {
        "description": "Arms straight down, legs neutral",
        "action": {'left_arm': [0.1, 0.1, -0.6], 'right_arm': [0.1, -0.1, -0.6], 
                   'left_leg': [0.0, 0.1, -1.0], 'right_leg': [0.0, -0.1, -1.0]}
    },
    {
        "description": "Arms reaching forward",
        "action": {'left_arm': [0.6, 0.2, 0.1], 'right_arm': [0.6, -0.2, 0.1], 
                   'left_leg': [0.0, 0.1, -1.0], 'right_leg': [0.0, -0.1, -1.0]}
    },
    {
        "description": "Right arm recovery (up and back)",
        "action": {'left_arm': [0.1, 0.1, -0.6], 'right_arm': [-0.3, -0.3, 0.4], 
                   'left_leg': [0.2, 0.1, -1.1], 'right_leg': [-0.2, -0.1, -1.1]}
    },
]

def main():
    tester = PoseTester(MODEL_PATH)
    
    with mujoco.viewer.launch_passive(tester.model, tester.data) as viewer:
        for i, pose in enumerate(TEST_POSES):
            print(f"Testing Pose {i+1}: {pose['description']}")
            tester.set_targets(pose['action'])
            
            reached = False
            timeout = 2000 # Max steps per pose
            steps = 0
            
            while not reached and steps < timeout:
                reached = tester.physics_step()
                steps += 1
                if steps % 10 == 0:
                    viewer.sync()
            
            print(f"Finished Pose {i+1} after {steps} steps.")
            time.sleep(1.5) # Pause to let you look at the pose

if __name__ == "__main__":
    main()