import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

class HumanSwimmingEnv(MujocoEnv):
    def __init__(self, render_mode=None):
        # Update this path to where you saved the worm XML
        self._model_path = '/home/judith/swimming/Swimming_agent/human.xml' 
        
        super().__init__(
            self._model_path, 
            frame_skip=5, 
            observation_space=None, 
            render_mode=render_mode
        )

         # We subtract 2 from nq because we are removing the global X, Y, Z position
        obs_size = (self.model.nq) + self.model.nv 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # to me trust always seemed very integral to pragmatics (so in like common ground, Grice maxims)
        # since you are working interdisciplinary - how do the conceptions of tust/belief updating in different fields 
        # interact


        # Action: Control inputs for the worm's actuators
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

    def _get_obs(self):
        # remove x, y, z pos
        relative_qpos = self.data.qpos 
        return np.concatenate([relative_qpos, self.data.qvel]).astype(np.float32)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        
        # --- REWARD LOGIC ---
        # In MuJoCo, for a free-floating body, qvel[0] is usually X-velocity
        # The worm wants to maximize velocity along the X-axis.
        forward_vel = self.data.qvel[0] 
        
        # Reward positive forward velocity, penalize staying still or going backward
        reward = forward_vel * 4.0 
        
        # Energy penalty: penalize huge erratic movements to encourage efficient swimming
        # ctrl_cost = 0.1 * np.square(action).sum()
        # reward -= ctrl_cost

        # --- TERMINATION LOGIC ---
        terminated = False
        
        # 1. Bound check: If the worm drifts too far from the center (X or Y), reset it
        # This keeps the agent in a controlled area of the simulation
        # if abs(self.data.qpos[0]) > 10.0 or abs(self.data.qpos[1]) > 10.0:
        #     terminated = True
        #     reward -= 1.0

        # 2. Depth check: If the worm sinks too low or floats too high (outside the "water")
        # Water surface is at 2m, worm starts at 1m. 
        if self.data.qpos[2] < 0 or self.data.qpos[2] > 5:
            terminated = True
            reward -= 1       
        # 2. Depth check: If the worm sinks too low or floats too high (outside the "water")
        # Water surface is at 2m, worm starts at 1m. 
        if self.data.qpos[2] < 1:
            reward -= 0.1
             
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        # Reset position to 1m height with slight randomness
        qpos = self.init_qpos + np.random.uniform(-0.05, 0.05, size=self.init_qpos.shape)
        qvel = np.zeros_like(self.init_qvel)
        self.set_state(qpos, qvel)
        return self._get_obs()

def make_human_env(render_mode=None):
    return HumanSwimmingEnv(render_mode=render_mode)

if __name__ == "__main__":
    # Use more envs if your CPU allows for faster training
    train_env = make_vec_env(lambda: make_human_env(), n_envs=8)

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
    model.learn(total_timesteps=1000000) 
    
    model.save("ppo_human_swimmer")
    train_env.close()