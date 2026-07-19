import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Load env and normalizer
env = DummyVecEnv([lambda: gym.make("Humanoid-v4", render_mode="human")])
env = VecNormalize.load("vec_normalize.pkl", env)

env.training = False
env.norm_reward = False

# Load model
model = PPO.load("ppo_humanoid_walker", env=env)

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()