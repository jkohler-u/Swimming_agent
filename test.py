import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

class CustomHumanoid(MujocoEnv):
    def __init__(self, **kwargs):
        # We use the built-in humanoid xml but you can replace 
        # 'humanoid.xml' with a path to your own custom .xml file
        super().__init__(
            model_path="humanoid.xml", 
            frame_skip=5, 
            observation_space=None, # Defined by XML
            **kwargs
        )

    def step(self, action):
        # 1. Execute the action in the MuJoCo simulator
        observation, reward, terminated, truncated, info = super().step(action)
        
        # 2. CUSTOM MODIFICATION: Modify the reward here
        # Example: Extra reward for keeping the torso upright
        # torso_angle = observation[some_index] 
        # reward -= abs(torso_angle) * 0.1
        
        return observation, reward, terminated, truncated, info

# Register the custom environment
gym.register(
    id="CustomHumanoid-v0",
    entry_point="__main__.CustomHumanoid",
)

# Use v4 to avoid the ImportError and DeprecationWarning
env_id = "Walker2d-v4" 

# Train on Walker2d using 4 parallel environments.
train_env = make_vec_env(env_id, n_envs=4)
model = PPO("MlpPolicy", train_env, verbose=1)

print(f"Training on {env_id}...")
model.learn(total_timesteps=100_000)
model.save("ppo_walker2d")
train_env.close()

# Visualize
eval_env = gym.make(env_id, render_mode="human")
observation, info = eval_env.reset()
for _ in range(1000):
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = eval_env.step(action)
    if terminated or truncated:
        observation, info = eval_env.reset()
eval_env.close()