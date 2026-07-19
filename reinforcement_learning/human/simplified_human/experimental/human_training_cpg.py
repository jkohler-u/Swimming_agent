import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize
from scipy.spatial.transform import Rotation as R  # <--- Add this import
WATER_SURFACE_HEIGHT = 3.0  # Define the water surface height as a constant


class HumanSwimmingEnv(MujocoEnv):
    def __init__(self, render_mode=None):
        self._model_path = 'reinforcement_learning/human/breathing_human/humanoid.xml' 

        super().__init__(
            self._model_path, 
            frame_skip=5, 
            observation_space=None, 
            render_mode=render_mode,
        )
        self.prev_action = np.zeros(self.model.nu)

        # important for rewards later
        self.current_step = 0
        self.min_vel_counter = 0
        self.hight_counter = 0
        self.total_timesteps_trained = 0


        obs_size = (self.model.nq) + self.model.nv 
        obs_size += 3 + 2 # Add 3 for roll, pitch, yaw, Add 2 for relation to the water line 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
    
    def get_euler_angles(self):
        """Converts quaternions to euler angles, for better interpretability"""
        quat_mujoco = self.data.qpos[3:7]   # get quaternion        
        # convert to euler
        quat_scipy = np.array([quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0]])
        rotation = R.from_quat(quat_scipy)
        euler = rotation.as_euler('xyz', degrees=False) 
        return euler 

    def _get_obs(self):
        # calculate "manual" observation variables
        relative_qpos = self.data.qpos
        euler = self.get_euler_angles()
        body_below_surface = False if  self.data.qpos[2] - WATER_SURFACE_HEIGHT < 0 else True
        head_above_surface = False if  self.data.body('head').xpos[2] - WATER_SURFACE_HEIGHT < 0 else True        
        return np.concatenate([relative_qpos, self.data.qvel, euler, [body_below_surface], [head_above_surface]]).astype(np.float32)
   
    def step(self, action):
        obs = self._get_obs()
        
        t = self.data.time # timer for cpg
        cpg_rhythm = np.sin(2 * np.pi * 1.0 * t) # 1Hz frequency
        # apply alternatingly to left and right
        left_size = self.model.nu // 2
        right_size = self.model.nu - left_size  
        
        cpg_weight = 0.2 # adjust cpg impact
        cpg_action = np.concatenate([
            np.full(left_size, cpg_rhythm * cpg_weight), 
            np.full(right_size, -cpg_rhythm * cpg_weight)
        ])

        final_action = cpg_action + action #add to rl action        
        final_action = np.clip(final_action, -1.0, 1.0)
        self.do_simulation(final_action, self.frame_skip)
        
        # --- REWARD LOGIC ---
        self.current_step +=1
        self.total_timesteps_trained += 1

        # reward variables 
        roll, pitch, _ = self.get_euler_angles()
        forward_vel = self.data.qvel[0] 
        body_height = self.data.qpos[2]
        head_height = self.data.body('head').xpos[2]
        foot_height_1 =  self.data.body('foot_left').xpos[2]
        foot_height_2 =  self.data.body('foot_right').xpos[2]
        hand_height_1 =  self.data.body('hand_right').xpos[2]
        hand_height_2 =  self.data.body('hand_left').xpos[2]
        
        # Reward movement, higher reward for higher speed
        reward = forward_vel ** 2 * 40 * forward_vel/np.abs(forward_vel)
        # Alternative: Reward movement, linear reward 
        # reward = forward_vel * 20.0  

        # reward the head being above water, but not too high
        reward += 5.0 * (head_height - WATER_SURFACE_HEIGHT)**2 if head_height > WATER_SURFACE_HEIGHT and head_height < WATER_SURFACE_HEIGHT + 0.3 else 0
        
        # reward the body being close too, but not above, the water surface
        reward += 5.0 * (WATER_SURFACE_HEIGHT - body_height)**2 if body_height < WATER_SURFACE_HEIGHT else 0
        
        # reward the limbs being close too, but not above, the water surface
        reward -= 1.0 * (foot_height_1 - WATER_SURFACE_HEIGHT - 0.5) **2 if foot_height_1 > WATER_SURFACE_HEIGHT + 0.5 else 0
        reward -= 1.0 * (foot_height_2 - WATER_SURFACE_HEIGHT - 0.5) **2 if foot_height_2 > WATER_SURFACE_HEIGHT + 0.5 else 0
        reward -= 1.0 * (hand_height_1 - WATER_SURFACE_HEIGHT - 0.5) **2 if hand_height_1 > WATER_SURFACE_HEIGHT + 0.5 else 0
        reward -= 1.0 * (hand_height_2 - WATER_SURFACE_HEIGHT - 0.5) **2 if hand_height_2 > WATER_SURFACE_HEIGHT + 0.5 else 0

        # survival reward
        reward += 0.05

        # Punish spinning
        reward -= 0.01 * np.abs(roll) 
        reward -= 0.05 * np.abs(pitch) 
   
        # penatly to encourage smoothness
        action_diff = np.square(action - self.prev_action).sum()
        reward -= 0.01 * action_diff  
        self.prev_action = action

        # --- TERMINATION LOGIC ---
        terminated = False
       
        # Depth check: Terminate and punish if the worm sinks too low or floats too high (outside the "water")
        self.hard_termination = self.total_timesteps_trained > 1_000_000
        if  (body_height < 0 or body_height > WATER_SURFACE_HEIGHT):
            reward -= 5
            if self.hard_termination:
                terminated = True

        # terminate training is there is no forward movement for 200 steps
        if forward_vel < 0.05: self.min_vel_counter += 1
        else: self.min_vel_counter = 0
        if self.min_vel_counter > 200: 
            terminated = True
            reward -= 5

        # terminate training if the head is too low for 200 steps
        if head_height < WATER_SURFACE_HEIGHT or head_height > WATER_SURFACE_HEIGHT + 1: self.hight_counter += 1
        else:  self.hight_counter = 0
        if self.hight_counter > 200:
            terminated = True
            reward -= 5

        # Bonus for moving in harmony with the CPG rhythm
        harmony_reward = np.sum(action * cpg_action) * 0.01
        reward += harmony_reward
        
        truncated = False
        info = {}
        
        info['head_height'] = head_height
        info['forward_vel'] = forward_vel
        info['step'] = self.current_step
        info['body'] = self.data.qpos[2]
        return obs,reward , terminated, truncated, info

    def reset_model(self):

        # Reset position 
        qpos = self.init_qpos.copy()
        qpos[2] = WATER_SURFACE_HEIGHT - 0.1 + self.np_random.uniform(-0.05, 0.05)
       
        qvel = np.zeros_like(self.init_qvel)
        self.set_state(qpos, qvel)

        self.current_step = 0
        self.min_vel_counter = 0
        self.hight_counter = 0
        return self._get_obs()

def make_human_env(render_mode=None):
    return HumanSwimmingEnv(render_mode=render_mode)

if __name__ == "__main__":
    train_env = make_vec_env(lambda: make_human_env(), n_envs=8)
    train_env = VecNormalize(train_env,norm_obs=True, norm_reward=True)

    model = model = PPO(
    "MlpPolicy", 
    train_env, 
    verbose=1, 
    learning_rate=2e-4, 
    n_steps=2048,        
    batch_size=64,       
    gae_lambda=0.95, 
    gamma=0.99, 
    ent_coef=0.001        # Exploration 
)
    
    print("Training HUman Swimming Agent...")
    model.learn(total_timesteps=2000000) 
    
    model.save("ppo_human_swimmer")
    train_env.close()
