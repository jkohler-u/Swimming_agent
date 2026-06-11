import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import PPO
from new_swimmer import SwimmingHumanoidEnv


def make_custom_env(render_mode="human"):
    return SwimmingHumanoidEnv(
        xml_file="humanoid.xml",
        render_mode=render_mode
    )


print("Creating environment...")
env = make_custom_env(render_mode="human")

print("Loading model...")
model = PPO.load("ppo_swimming_humanoid", env=env)
print("Model loaded")

obs, info = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()