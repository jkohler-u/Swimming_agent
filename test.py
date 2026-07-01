import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np


class MyCustomHumanoidEnv(MujocoEnv):
    def __init__(self, render_mode=None):
        self._model_path = '/home/judith/swimming/Swimming_agent/humanoid2.xml' 
        
        # 1. Initialize the MuJoCo environment
        super().__init__(
            self._model_path, 
            frame_skip=5, 
            observation_space=None, 
            render_mode=render_mode
        )

        # 2. EXPLICITLY DEFINE SPACES (This fixes your AttributeError)
        # observation_space: Total of qpos (positions) and qvel (velocities)
        # self.model.nq is the number of generalized coordinates
        # self.model.nv is the number of generalized velocities
        obs_size = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # action_space: The number of actuators defined in your XML
        # self.model.nu is the number of actuators
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        
        # Your reward logic
        forward_vel = self.data.qvel[0] 
        reward = forward_vel * 2.0 if forward_vel > 0 else -0.05
            
        # Check for falling (adjust index [2] to your model's Z-axis)
        terminated = False
        if self.data.qpos[2] < 0.5: 
            terminated = True
            reward -= 10.0        

        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        # Basic reset to avoid starting in an invalid state
        qpos = self.init_qpos + np.random.uniform(-0.01, 0.01, size=self.init_qpos.shape)
        qvel = np.zeros_like(self.init_qvel)
        self.set_state(qpos, qvel)
        return self._get_obs()

# 2. Helper function for Vectorized Env
def make_custom_env(render_mode=None):
    return MyCustomHumanoidEnv(render_mode=render_mode)

# 3. Training
if __name__ == "__main__":
    # Training envs (no rendering for speed)
    train_env = make_vec_env(lambda: make_custom_env(), n_envs=4)

    model = PPO("MlpPolicy", train_env, verbose=1)
    print("Training on Custom XML Humanoid...")
    model.learn(total_timesteps=1000000)
    model.save("ppo_custom_humanoid_xml")
    train_env.close()