import os
import gymnasium as gym
import numpy as np

from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class SwimmingHumanoidEnv(MujocoEnv, gym.utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 40,
    }

    def __init__(self, xml_file="humanoid.xml", render_mode=None):
        self.xml_file = os.path.abspath(xml_file)

        gym.utils.EzPickle.__init__(
            self,
            xml_file=xml_file,
            render_mode=render_mode,
        )

        # Temporary observation space.
        # qpos + qvel size will be checked after loading the model.
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(45,),
            dtype=np.float64,
        )

        MujocoEnv.__init__(
            self,
            model_path=self.xml_file,
            frame_skip=5,
            observation_space=observation_space,
            render_mode=render_mode,
        )

        # Correct observation space after model is loaded
        obs_size = self.data.qpos.size + self.data.qvel.size
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float64,
        )

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat.copy(),
            self.data.qvel.flat.copy(),
        ])

    def reset_model(self):
        # Use the initial state from the XML file.
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        self.set_state(qpos, qvel)

        print("Reset root z:", self.data.qpos[2])

        return self._get_obs()

    def step(self, action):
        x_before = self.data.qpos[0].copy()

        self.do_simulation(action, self.frame_skip)

        x_after = self.data.qpos[0].copy()
        x_velocity = (x_after - x_before) / self.dt

        energy_penalty = 0.001 * np.sum(np.square(action))

        reward = x_velocity - energy_penalty

        obs = self._get_obs()

        terminated = False
        truncated = False

        info = {
            "x_velocity": x_velocity,
            "energy_penalty": energy_penalty,
            "root_z": self.data.qpos[2],
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info


def make_env(render_mode=None):
    return SwimmingHumanoidEnv(
        xml_file="humanoid.xml",
        render_mode=render_mode,
    )


if __name__ == "__main__":
    # Quick test: check whether XML height is preserved
    test_env = make_env(render_mode=None)
    obs, info = test_env.reset()
    print("Initial qpos[2] root height:", test_env.data.qpos[2])
    print("Action space:", test_env.action_space)
    print("Observation space:", test_env.observation_space)
    test_env.close()

    # Training env
    train_env = make_vec_env(
        lambda: make_env(render_mode=None),
        n_envs=4,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
    )

    print("Training swimming humanoid...")
    model.learn(total_timesteps=10_000)

    model.save("ppo_swimming_humanoid")
    train_env.close()

    # Optional visual test after training
    env = make_env(render_mode="human")
    obs, info = env.reset()

    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()