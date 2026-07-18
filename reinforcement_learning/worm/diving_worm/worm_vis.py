
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time 
from stable_baselines3 import PPO
# IMPORTANT: Import the class you actually used to train!
# If MyCustomHumanoidEnv is in training_script.py, use:
# from training_script import MyCustomHumanoidEnv 
from worm_training import WormSwimmingEnv 

def make_custom_env(render_mode="human"):
    return WormSwimmingEnv(
        render_mode=render_mode
    )

print("Creating environment...")
env = make_custom_env(render_mode="human")

print("Loading model...")
# The model expects an observation space of 47 (float32)
model = PPO.load("underwater_swimming_worm/ppo_worm_swimmer_sucess.zip", env=env)
print("Model loaded")

obs, info = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    env.render() # Ensure explicit render call
    time.sleep(0.02) 

    if terminated or truncated:
        obs, info = env.reset()