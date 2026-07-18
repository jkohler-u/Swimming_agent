import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecFrameStack
# IMPORTANT: Import the class you actually used to train!
# If MyCustomHumanoidEnv is in training_script.py, use:
# from training_script import MyCustomHumanoidEnv 
from reinforcement_learning.human.simplified_human.human_training_simple import HumanSwimmingEnv
# from human_training_walk import HumanWalkingEnv

def make_custom_env(render_mode="human"):
    return HumanSwimmingEnv(
        render_mode=render_mode
    )

print("Creating environment...")
env = make_custom_env(render_mode="human")

print("Loading model...")
# Load the model. We pass the env so SB3 can check if the shapes match.
model = PPO.load("underwater_swimming_simple_human/saves/ppo_human_swimmer_in_circles.zip", env=env)
print("Model loaded")

obs, info = env.reset()

while True:
    # deterministic=True is crucial for evaluation/vis
    action, _ = model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    env.render()
    time.sleep(0.02) 

    if terminated or truncated:
        obs, info = env.reset()