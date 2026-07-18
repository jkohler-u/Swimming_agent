import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from scipy.spatial.transform import Rotation as R
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from gymnasium.wrappers import RecordVideo # <--- Added for video recording
import os
import shutil
import csv
import sys

# Constant
WATER_SURFACE_HEIGHT = 3.0

class HumanSwimmingEnv(MujocoEnv):
    def __init__(self, render_mode=None, leniency=200, survival_reward=0.01, forward_reward=40, scaling=1.0, 
                 head_punishment=5, vel_punishment=1, body_punishment=5, smothness_reward=0.01,
                 cont_head_reward=5, cont_body_reward=5, cont_body_punishment=1.0, 
                 roll_punishment=0.1, pitch_punishment=0.5):
        
        self._model_path = '/home/judith/swimming/Swimming_agent/underwater_swimming_simple_human/humanoid.xml'
        super().__init__(
            self._model_path, 
            frame_skip=5, 
            observation_space=None, 
            render_mode=render_mode,
        )
        
        self.survival_reward = survival_reward
        self.leniency = leniency
        self.forward_reward = forward_reward
        self.scaling = scaling
        self.head_punishment = head_punishment
        self.vel_punishment = vel_punishment
        self.body_punishment = body_punishment
        self.smothness_reward = smothness_reward
        self.cont_head_reward = cont_head_reward
        self.cont_body_reward = cont_body_reward
        self.cont_body_punishment = cont_body_punishment
        self.roll_punishment = roll_punishment
        self.pitch_punishment = pitch_punishment

        self.prev_action = np.zeros(self.model.nu)
        self.current_step = 0
        self.min_vel_counter = 0
        self.hight_counter = 0
        self.total_timesteps_trained = 0

        obs_size = (self.model.nq) + self.model.nv + 3 + 2 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
    
    def get_euler_angles(self):
        quat_mujoco = self.data.qpos[3:7] 
        quat_scipy = np.array([quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0]])
        rotation = R.from_quat(quat_scipy)
        return rotation.as_euler('xyz', degrees=False)

    def _get_obs(self):
        relative_qpos = self.data.qpos
        euler = self.get_euler_angles()
        body_below_surface = False if self.data.qpos[2] - WATER_SURFACE_HEIGHT < 0 else True
        head_above_surface = False if self.data.body('head').xpos[2] - WATER_SURFACE_HEIGHT < 0 else True
        return np.concatenate([relative_qpos, self.data.qvel, euler, [body_below_surface], [head_above_surface]]).astype(np.float32)

    def step(self, action):
        roll, pitch, yaw = self.get_euler_angles()
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        self.current_step += 1
        self.total_timesteps_trained += 1

        forward_vel = self.data.qvel[0] 
        reward = forward_vel * self.forward_reward  

        body_height = self.data.qpos[2]
        head_height = self.data.body('head').xpos[2]
        foot_height_1 = self.data.body('foot_left').xpos[2]
        foot_height_2 = self.data.body('foot_right').xpos[2]
        hand_height_1 = self.data.body('hand_right').xpos[2]
        hand_height_2 = self.data.body('hand_left').xpos[2]

        reward += self.cont_head_reward * (head_height - WATER_SURFACE_HEIGHT)**2 if WATER_SURFACE_HEIGHT < head_height < WATER_SURFACE_HEIGHT + 0.3 else 0
        reward += self.cont_body_reward * (WATER_SURFACE_HEIGHT - body_height)**2 if body_height < WATER_SURFACE_HEIGHT else 0
        
        reward -= self.cont_body_punishment * (foot_height_1 - WATER_SURFACE_HEIGHT - 0.5) **2 if foot_height_1 > WATER_SURFACE_HEIGHT + 0.5 else 0
        reward -= self.cont_body_punishment * (foot_height_2 - WATER_SURFACE_HEIGHT - 0.5) **2 if foot_height_2 > WATER_SURFACE_HEIGHT + 0.5 else 0
        reward -= self.cont_body_punishment * (hand_height_1 - WATER_SURFACE_HEIGHT - 0.5) **2 if hand_height_1 > WATER_SURFACE_HEIGHT + 0.5 else 0
        reward -= self.cont_body_punishment * (hand_height_2 - WATER_SURFACE_HEIGHT - 0.5) **2 if hand_height_2 > WATER_SURFACE_HEIGHT + 0.5 else 0

        reward += self.survival_reward
        reward -= self.roll_punishment * np.abs(roll) 
        reward -= self.pitch_punishment * np.abs(pitch) 
        
        action_diff = np.square(action - self.prev_action).sum()
        reward -= self.smothness_reward * action_diff  
        self.prev_action = action

        terminated = False
        self.hard_termination = self.total_timesteps_trained > 1_000_000
        if (body_height < 0 or body_height > WATER_SURFACE_HEIGHT):
            reward -= self.body_punishment
            if self.hard_termination: terminated = True

        if forward_vel < 0.05: self.min_vel_counter += 1
        else: self.min_vel_counter = 0
        if self.min_vel_counter > self.leniency:
            terminated = True
            reward -= self.vel_punishment

        if head_height < WATER_SURFACE_HEIGHT: 
            self.hight_counter += 1
        else: self.hight_counter = 0
        if self.hight_counter > self.leniency:
            terminated = True
            reward -= self.head_punishment

        reward *= self.scaling

        truncated = False
        info = {'head_height': head_height, 'forward_vel': forward_vel, 'step': self.current_step, 'body': body_height}
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qpos[2] = WATER_SURFACE_HEIGHT - 0.1 + self.np_random.uniform(-0.05, 0.05)
        qvel = np.zeros_like(self.init_qvel)
        self.set_state(qpos, qvel)
        self.current_step = 0
        self.min_vel_counter = 0
        self.hight_counter = 0
        return self._get_obs()

def make_human_env(render_mode=None, **kwargs):
    return HumanSwimmingEnv(render_mode=render_mode, **kwargs)

def main():
    lr = 0.001
    batch_size = 64
    env_params = {
        "leniency": 200, 
        "survival_reward": 0.1, 
        "forward_reward": 40, 
        "scaling": 0,
        "head_punishment": 5, "vel_punishment": 1, "body_punishment": 5, "smothness_reward": 0.01,
        "cont_head_reward": 5, "cont_body_reward": 5, "cont_body_punishment": 5,
        "roll_punishment": 0.1, "pitch_punishment": 0.5
    }

    output_dir = "human_swimmer_increased_survial_no_scaling_higher_cont_body_pun"
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(sys.argv[0], os.path.join(output_dir, "train_script.py"))

    train_env = make_vec_env(lambda: make_human_env(**env_params), n_envs=8)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    train_env.save(os.path.join(output_dir, "vec_normalize.pkl"))

    model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=lr, n_steps=4048, batch_size=batch_size)
    model.learn(total_timesteps=2_500_000)
    model.save(os.path.join(output_dir, "ppo_human_swimmer"))
    train_env.close()

    print("Running Evaluation and Recording Video...")
    # 1. Create the environment with rgb_array mode for recording
    eval_env = make_human_env(render_mode="rgb_array", **env_params)
    
    # 2. Wrap with RecordVideo
    video_folder = os.path.join(output_dir, "videos")
    eval_env = RecordVideo(
        eval_env, 
        video_folder=video_folder, 
        episode_trigger=lambda x: x == 0 # Record the first episode
    )
    
    # 3. Wrap in DummyVecEnv and Normalize to match training
    test_env = DummyVecEnv([lambda: eval_env])
    norm_env = VecNormalize.load(os.path.join(output_dir, "vec_normalize.pkl"), test_env)
    norm_env.training = False
    norm_env.norm_reward = False

    obs = norm_env.reset()
    results = []
    
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = norm_env.step(action)
        
        head_h = info[0]['head_height']
        f_vel = info[0]['forward_vel']
        alive = not done[0]
        head_above = head_h > WATER_SURFACE_HEIGHT
        
        results.append([i, alive, head_above, f_vel])
        if done[0]: break

    # Close the environment to ensure the video file is finalized
    norm_env.close()

    csv_path = os.path.join(output_dir, "test_results.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "alive", "head_above_water", "forward_velocity"])
        writer.writerows(results)

    parent_dir = os.path.dirname(os.path.abspath(output_dir))
    folder_name = os.path.basename(os.path.abspath(output_dir))
    shutil.make_archive(os.path.join(parent_dir, folder_name), 'zip', parent_dir, folder_name)
    
    print(f"All files (including videos) saved to {output_dir} and zipped.")

if __name__ == "__main__":
    main()