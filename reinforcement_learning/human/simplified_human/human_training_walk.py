import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# 1. Create the standardized MuJoCo Humanoid environment
# Humanoid-v4 is specifically designed to reward forward walking
env = make_vec_env("Humanoid-v4", n_envs=8)

# 2. IMPORTANT for Humanoids: Normalize observations and rewards!
# Humanoids have many joints moving at different speeds. Normalization 
# is often the difference between learning to walk and failing entirely.
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# 3. Use hyperparameters known to work for Humanoid-v4 (from SB3 Zoo)
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
# Note: Humanoid usually needs between 2 Million and 5 Million timesteps to look "natural"
model.learn(total_timesteps=2_000_000)

model.save("ppo_humanoid_walker")
env.save("vec_normalize.pkl") # You must save the normalizer statistics too!
env.close()