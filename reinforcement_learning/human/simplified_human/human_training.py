import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from gymnasium.wrappers import TimeLimit
from scipy.spatial.transform import Rotation as R  
from stable_baselines3.common.vec_env import VecNormalize
WATER_SURFACE_HEIGHT = 3.0 

class HumanSwimmingEnv(MujocoEnv):
    def __init__(self, render_mode=None):
        self._model_path = 'reinforcement_learning/human/simplified_human/human_arms.xml'
        super().__init__(
            self._model_path, 
            frame_skip=5, 
            observation_space=None, 
            render_mode=render_mode,
        )
        self.prev_action = np.zeros(self.model.nu)

        self.current_step = 0
        self.min_vel_counter = 0
        self.hight_counter = 0
        self.total_timesteps_trained = 0

        obs_size = (self.model.nq) + self.model.nv 
        obs_size += 3 
        obs_size += 2  # Add 5 for roll, pitch, yaw, distance from watersurface
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
    
    def get_euler_angles(self):
        """Converts quaternions to euler angles, for better interpretability"""
        quat_mujoco = self.data.qpos[3:7]   # get quaternion        
        # convert to euler
        quat_scipy = np.array([quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0]])
        rotation = R.from_quat(quat_scipy)
        euler = rotation.as_euler('xyz', degrees=False) 
        return euler 

    def _get_obs(self):
        relative_qpos = self.data.qpos
        euler = self.get_euler_angles()
        # bool whetheer agent is above water surface
        body_below_surface = False if  self.data.qpos[2] - WATER_SURFACE_HEIGHT < 0 else True
        head_above_surface = False if  self.data.body('head').xpos[2] - WATER_SURFACE_HEIGHT < 0 else True
      
        return np.concatenate([relative_qpos, self.data.qvel, euler, [body_below_surface], [head_above_surface]]).astype(np.float32)

    def step(self, action):
        roll, pitch, yaw = self.get_euler_angles()
        # angular_vel_z = self.data.qvel[5]
        # angular_vel_y= self.data.qvel[4]
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()

        self.current_step +=1
        self.total_timesteps_trained += 1

        # --- REWARD LOGIC ---
        
        # Reward movement, h
        forward_vel = self.data.qvel[0] 
        # reward = forward_vel ** 2 *  100 * forward_vel/np.abs(forward_vel) #higher reward for higher speed
        reward = forward_vel * 40.0  

       
        ## Manage in/out of water
        body_height = self.data.qpos[2]
        head_height = self.data.body('head').xpos[2]
        foot_height_1 =  self.data.body('foot_left').xpos[2]
        foot_height_2 =  self.data.body('foot_right').xpos[2]
        hand_height_1 =  self.data.body('hand_right').xpos[2]
        hand_height_2 =  self.data.body('hand_left').xpos[2]

        # reward the head being above water, but not too high
        reward += 5.0 * (head_height - WATER_SURFACE_HEIGHT)**2 if head_height > WATER_SURFACE_HEIGHT and head_height < WATER_SURFACE_HEIGHT + 0.3 else 0
        
        # reward the body being close too, but not above, the water surface
        reward += 5.0 * (WATER_SURFACE_HEIGHT - body_height)**2 if body_height < WATER_SURFACE_HEIGHT else 0
        

        # Penalize if the head is too low or feet/hands are too high
        # reward -= 5.0 * (WATER_SURFACE_HEIGHT - head_height)**2 if head_height < WATER_SURFACE_HEIGHT else 0
        reward -= 1.0 * (foot_height_1 - WATER_SURFACE_HEIGHT - 0.5) **2 if foot_height_1 > WATER_SURFACE_HEIGHT + 0.5 else 0
        reward -= 1.0 * (foot_height_2 - WATER_SURFACE_HEIGHT - 0.5) **2 if foot_height_2 > WATER_SURFACE_HEIGHT + 0.5 else 0
        reward -= 1.0 * (hand_height_1 - WATER_SURFACE_HEIGHT - 0.5) **2 if hand_height_1 > WATER_SURFACE_HEIGHT + 0.5 else 0
        reward -= 1.0 * (hand_height_2 - WATER_SURFACE_HEIGHT - 0.5) **2 if hand_height_2 > WATER_SURFACE_HEIGHT + 0.5 else 0

        # survival reward
        reward += 0.05

        # Punish spinning 
        reward -= 0.01 * np.abs(roll) 
        reward -= 0.05 * np.abs(pitch) 
        # reward -= 0.05 * np.abs(angular_vel_y) 
        # reward -= np.abs(angular_vel_z) 
   
        # penatly to encourage smoothness
        action_diff = np.square(action - self.prev_action).sum()
        reward -= 0.01 * action_diff  
        self.prev_action = action


        # --- TERMINATION LOGIC ---
        terminated = False
       
        # Depth check: Terminate and punish If the worm sinks too low or floats too high (outside the "water")
        self.hard_termination = self.total_timesteps_trained > 1_000_000
        if  (body_height < 0 or body_height > WATER_SURFACE_HEIGHT):
            reward -= 5
            if self.hard_termination:
                terminated = True

        # terminate training is there is no forward movement
        if forward_vel < 0.05: self.min_vel_counter += 1
        else: self.min_vel_counter = 0
        if self.min_vel_counter > 200:
            terminated = True
            reward -= 5

        # terminate training if the head is too low for too long
        if head_height < WATER_SURFACE_HEIGHT: self.hight_counter += 1
        else: self.hight_counter = 0
        if self.hight_counter > 200: 
            terminated = True
            reward -= 5

        # if self.current_step > 1000 and forward_vel < 0.05:
        #     terminated = True 
        #     reward -= 2 # Punish for idling
        
        # "normalize" reward
        reward *= 0.01

        truncated = False
        info = {}
        info['head_height'] = head_height
        info['forward_vel'] = forward_vel
        info['step'] = self.current_step
        info['body'] = self.data.qpos[2]
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        # Reset position to 1m height with slight randomness
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
    train_env = VecNormalize(
        train_env,
        norm_obs=True,          
        norm_reward=True)    
    
    train_env.save("vec_normalize.pkl")    

    model = PPO(
    "MlpPolicy", 
    train_env, 
    verbose=1, 
    learning_rate=2e-4, 
    n_steps=4048,       
    batch_size=64,       
    gae_lambda=0.95, 
    gamma=0.99, 
    ent_coef=0.01        
)
    
    print("Training HUman Swimming Agent...")
    model.learn(total_timesteps=2300000) 
    
    model.save("ppo_human_swimmer")
    train_env.close()

