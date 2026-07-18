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
        # Update this path to where you saved the worm XML
        self._model_path = '/home/judith/swimming/Swimming_agent/underwater_swimming_simple_human/humanoid.xml' 
        # self._model_path = 'underwater_swimming_simple_human/humanoid_laying.xml'

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
        obs_size += 3 + 2 # Add 3 for roll, pitch, yaw 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Action: Control inputs for the humans's actuators
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
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

    def _get_obs(self):
        relative_qpos = self.data.qpos
        euler = self.get_euler_angles()
        body_below_surface = False if  self.data.qpos[2] - WATER_SURFACE_HEIGHT < 0 else True
        head_above_surface = False if  self.data.body('head').xpos[2] - WATER_SURFACE_HEIGHT < 0 else True
        
        # Concatenate qpos, qvel, AND the euler angles
        return np.concatenate([relative_qpos, self.data.qvel, euler, [body_below_surface], [head_above_surface]]).astype(np.float32)
   
    def step(self, action):
        # 1. Define a basic CPG rhythm (e.g., simple sine wave)
        # Use self.np_random.uniform or a global timer
        t = self.data.time 
        obs = self._get_obs()

        # Example: Alternating arms (one is sin, other is -sin)
        cpg_rhythm = np.sin(2 * np.pi * 1.0 * t) # 1Hz frequency
        
        # Create a vector for all actuators
        # Assume first half of actuators are 'left' and second half are 'right'
        # Create a vector for all actuators
        # This ensures that even if nu is odd, the shapes match exactly
        left_size = self.model.nu // 2
        right_size = self.model.nu - left_size  # Takes the remainder if nu is odd
        
        cpg_weight = 0.2 # Adjust this value (0.1 to 0.5)
        cpg_action = np.concatenate([
            np.full(left_size, cpg_rhythm * cpg_weight), 
            np.full(right_size, -cpg_rhythm * cpg_weight)
        ])

        # 2. Combine CPG with RL action (Residual)
        # RL agent now 'fine-tunes' the basic rhythm
        final_action = cpg_action + action 

       
        
        # Clip to ensure we stay within actuator limits
        final_action = np.clip(final_action, -1.0, 1.0)

        self.do_simulation(final_action, self.frame_skip)
        
       # --- REWARD LOGIC ---
        roll, pitch, _ = self.get_euler_angles()
        self.current_step +=1
        self.total_timesteps_trained += 1
        
        # # Reward movement, higher reward for higher speed
        forward_vel = self.data.qvel[0] 
        reward = forward_vel ** 2 * 40 * forward_vel/np.abs(forward_vel)

        # try rewarding the head being above water
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
        

        reward -= 1.0 * (foot_height_1 - WATER_SURFACE_HEIGHT - 0.5) **2 if foot_height_1 > WATER_SURFACE_HEIGHT + 0.5 else 0
        reward -= 1.0 * (foot_height_2 - WATER_SURFACE_HEIGHT - 0.5) **2 if foot_height_2 > WATER_SURFACE_HEIGHT + 0.5 else 0
        reward -= 1.0 * (hand_height_1 - WATER_SURFACE_HEIGHT - 0.5) **2 if hand_height_1 > WATER_SURFACE_HEIGHT + 0.5 else 0
        reward -= 1.0 * (hand_height_2 - WATER_SURFACE_HEIGHT - 0.5) **2 if hand_height_2 > WATER_SURFACE_HEIGHT + 0.5 else 0

        
        # Reward movement, linear reward 
        # reward = forward_vel * 20.0  

        # survival reward
        reward += 0.05

        # Punish spinning (Angular velocity)
        # We use absolute value or square to punish both clockwise and counter-clockwise spin
        # angular_vel_z = self.data.qvel[5]
        # angular_vel_y= self.data.qvel[4]
        # reward -= 0.05 * np.abs(angular_vel_y) 
        # reward -= np.abs(angular_vel_z) 
        reward -= 0.01 * np.abs(roll) 
        reward -= 0.05 * np.abs(pitch) 
   
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
        if forward_vel < 0.05:
            self.min_vel_counter += 1
        else:
            self.min_vel_counter = 0

        if self.min_vel_counter > 200: # If moving too slowly for 200 steps
            terminated = True
            reward -= 5

        # terminate training if the head is too low for too long
        if head_height < WATER_SURFACE_HEIGHT or head_height > WATER_SURFACE_HEIGHT + 1:
            self.hight_counter += 1
        else:
            self.hight_counter = 0
        
        if self.hight_counter > 200: # If head is too low for 200 steps
            terminated = True
            reward -= 5


        # Bonus for moving in harmony with the CPG rhythm
        harmony_reward = np.sum(action * cpg_action) * 0.01
        reward += harmony_reward


        # if self.current_step > 1000 and forward_vel < 0.05:
        #     terminated = True 
        #     reward -= 2 # Punish for idling
        
        truncated = False
        info = {}
        
        info['head_height'] = head_height
        info['forward_vel'] = forward_vel
        info['step'] = self.current_step
        info['body'] = self.data.qpos[2]
        # return obs, reward, terminated, truncated, info, head_height, forward_vel
        return obs,reward , terminated, truncated, info

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
    # Use more envs if your CPU allows for faster training
    train_env = make_vec_env(lambda: make_human_env(), n_envs=8)
    train_env = VecNormalize(
        train_env,
        norm_obs=True,          # normalise observations
        norm_reward=True,       # normalise rewards (set False if you don’t want it)
        
    )  
    # PPO hyperparameters can be tuned, but these are a good start
    model = model = PPO(
    "MlpPolicy", 
    train_env, 
    verbose=1, 
    learning_rate=2e-4, 
    n_steps=2048,        # Increase the rollout buffer
    batch_size=64,       # Standard for PPO
    gae_lambda=0.95, 
    gamma=0.99, 
    ent_coef=0.001        # Encourage exploration of different swimming strokes
)
    
    print("Training HUman Swimming Agent...")
    # Swimming is harder to learn than walking; you might need 2M+ timesteps
    model.learn(total_timesteps=2000000) 
    
    model.save("ppo_human_swimmer")
    train_env.close()

# if __name__ == "__main__":
#     env = HumanSwimmingEnv(render_mode="human")
#     obs, info = env.reset()

#     for step in range(1000):
#         action = env.action_space.sample()
#         obs, reward, terminated, truncated, info = env.step(action)
#         env.render()

#         if step % 50 == 0:
#             print(f"\nStep {step}")
            

#         if terminated or truncated:
#             obs, info = env.reset()

#     env.close()