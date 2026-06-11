import gymnasium as gym
from stable_baselines3 import PPO

# Same wrapper used during training
class CustomHumanoidWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward += 0.1
        return observation, reward, terminated, truncated, info


def make_custom_env(render_mode="human"):
    env = gym.make("Humanoid-v4", render_mode=render_mode)
    env = CustomHumanoidWrapper(env)
    return env


# Load trained model
print("Loading model...")
model = PPO.load("ppo_custom_humanoid")
print("Model loaded")

env = make_custom_env(render_mode="human")
print("Environment created")

try:
    obs, info = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Resetting episode")
            obs, info = env.reset()

except Exception as e:
    print("ERROR:", e)

finally:
    env.close()