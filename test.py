import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# 1. The Wrapper (unchanged)
class CustomHumanoidWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # # Classic reward 
        # reward += 0.1 

        # Let's assume observation[0] is the forward velocity of the torso
        forward_vel = observation[0] 

        if forward_vel > 0:
            reward += forward_vel * 2.0 # Bonus for moving forward
        else:
            reward -= 1.0 # Penalty for moving backward or standing still
        return observation, reward, terminated, truncated, info

# 2. Updated helper function to handle render_mode correctly
def make_custom_env(render_mode=None):
    # render_mode MUST be passed here
    env = gym.make("Humanoid-v4", render_mode=render_mode) 
    env = CustomHumanoidWrapper(env)
    return env

# 3. Training
# For training, we don't want render_mode="human" (it would be too slow)
train_env = make_vec_env(lambda: make_custom_env(), n_envs=4)

model = PPO("MlpPolicy", train_env, verbose=1)
print("Training on Custom Humanoid...")
model.learn(total_timesteps=100000)
model.save("ppo_custom_humanoid")
train_env.close()

# 4. Visualization
# Pass "human" here so the environment is created with rendering enabled
eval_env = make_custom_env(render_mode="human") 

observation, info = eval_env.reset()
for _ in range(40000):
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = eval_env.step(action)
    if terminated or truncated:
        observation, info = eval_env.reset()

eval_env.close()