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
    env = gym.make("Humanoid-v5", render_mode=render_mode)
    env = CustomHumanoidWrapper(env)
    return env


# Load trained model
model = PPO.load("ppo_custom_humanoid")

# Create visualization environment
env = make_custom_env(render_mode="human")

obs, info = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Episode finished. Resetting...")
        obs, info = env.reset()