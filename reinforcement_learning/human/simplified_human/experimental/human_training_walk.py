import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

env = make_vec_env("Humanoid-v4", n_envs=8)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    batch_size=64,
    n_steps=2048,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.0,           # No entropy coefficient needed for this env
    learning_rate=3e-4,     # Standard LR
    clip_range=0.2
)

print("Training Gym Humanoid to walk...")
model.learn(total_timesteps=2_000_000)

model.save("ppo_humanoid_walker")
env.save("vec_normalize.pkl") 
env.close()