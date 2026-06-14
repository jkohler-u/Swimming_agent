import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time 
from stable_baselines3 import PPO
# IMPORTANT: Import the class you actually used to train!
# If MyCustomHumanoidEnv is in training_script.py, use:
# from training_script import MyCustomHumanoidEnv 
from test import MyCustomHumanoidEnv 

def make_custom_env(render_mode="human"):
    # Use MyCustomHumanoidEnv, NOT SwimmingHumanoidEnv
    return MyCustomHumanoidEnv(
        render_mode=render_mode
    )

print("Creating environment...")
env = make_custom_env(render_mode="human")

print("Loading model...")
# The model expects an observation space of 47 (float32)
model = PPO.load("ppo_custom_humanoid_xml", env=env)
print("Model loaded")

obs, info = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    # time.sleep(0.0001) 

    if terminated or truncated:
        obs, info = env.reset()