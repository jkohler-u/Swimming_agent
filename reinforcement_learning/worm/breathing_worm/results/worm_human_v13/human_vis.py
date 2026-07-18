import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecFrameStack
# IMPORTANT: Import the class you actually used to train!
# If MyCustomHumanoidEnv is in training_script.py, use:
# from training_script import MyCustomHumanoidEnv 
from reinforcement_learning.worm.breathing_worm.results.worm_human_v13.train_script import HumanSwimmingEnv
# from human_training_walk import HumanWalkingEnv

def make_custom_env(render_mode="human"):
    return HumanSwimmingEnv(
        render_mode=render_mode
    )

print("Creating environment...")
env = make_custom_env(render_mode="human")
# venv = DummyVecEnv([lambda: env]) 
# venv = VecFrameStack(venv, n_stack=6) 
# env = VecNormalize.load("/home/judith/swimming/Swimming_agent/vec_normalize.pkl", venv)   
env.training = False          # stop updating running mean/std
env.norm_obs = True           # keep observation normalisation
env.norm_reward = False       # usually you turn reward‑norm off when evaluating
print("Loading model...")
# The model expects an observation space of 47 (float32)
model = PPO.load("worm_human_v13/ppo_human_swimmer.zip", env=env)
print("Model loaded")
expected_obs_shape = model.observation_space.shape
print(f"Model expects observation shape: {expected_obs_shape}")
print(f"Environment currently has shape: {env.observation_space.shape}")

# 3. Manually override the environment's space to match the model
# This "tricks" SB3 into thinking they match
env.observation_space = model.observation_space

step = 0

# obs = env.reset()

# while True:
#     step = step + 1
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, info= env.step(action)
#     current_info = info[0]
#     head = current_info['head_height']
#     vel = current_info['forward_vel']
#     step = current_info['step']
#     body = current_info['body']
#     if step % 20 == 0:
#         print(f"head: {round(head - 3,2)} body: {round(3 - body,2)} vel: {round(vel,2)} step: {step} reward: {round(reward[0],2)}")
#     env.render() # Ensure explicit render call
#     time.sleep(0.02) 

#     if done:
#         obs = env.reset()

obs, info = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    env.render() # Ensure explicit render call
    time.sleep(0.02) 

    if terminated or truncated:
        obs, info = env.reset()