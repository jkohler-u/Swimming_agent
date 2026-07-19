import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time 
from stable_baselines3 import PPO
from reinforcement_learning.human.simplified_human.human_training_simple import HumanSwimmingEnv

def make_custom_env(render_mode="human"):
    return HumanSwimmingEnv(
        render_mode=render_mode
    )

print("Creating environment...")
env = make_custom_env(render_mode="human")

print("Loading model...")
model = PPO.load("underwater_swimming_simple_human/saves/ppo_human_swimmer_in_circles.zip", env=env)
print("Model loaded")

obs, info = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.02) 

    if terminated or truncated:
        obs, info = env.reset()