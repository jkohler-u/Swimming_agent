import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecFrameStack
# IMPORTANT: Import the class you actually used to train!
# If MyCustomHumanoidEnv is in training_script.py, use:
# from training_script import MyCustomHumanoidEnv 
from human_training import HumanSwimmingEnv
# from human_training_walk import HumanWalkingEnv

def make_custom_env(render_mode="human"):
    return HumanSwimmingEnv(
        render_mode=render_mode
    )

print("Creating environment...")
env = make_custom_env(render_mode="human")
venv = DummyVecEnv([lambda: env]) 
venv = VecFrameStack(venv, n_stack=6) 
env = VecNormalize.load("/home/judith/swimming/Swimming_agent/vec_normalize.pkl", venv)   
env.training = False          
env.norm_obs = True          
env.norm_reward = False       
print("Loading model...")

model = PPO.load("underwater_swimming_simple_human/saves/ppo_human_swimmer_cpg_aktuell.zip", env=env)
print("Model loaded")
expected_obs_shape = model.observation_space.shape
print(f"Model expects observation shape: {expected_obs_shape}")
print(f"Environment currently has shape: {env.observation_space.shape}")

step = 0

obs, info = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    env.render() # Ensure explicit render call
    time.sleep(0.02) 

    if terminated or truncated:
        obs, info = env.reset()