from pathlib import Path

import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from gymnasium.wrappers import TimeLimit
import mujoco


class WormSwimmingEnv(MujocoEnv):
    def __init__(self, render_mode=None):
        xml_path = Path(__file__).parent / "worm_water_surface.xml"

        super().__init__(
            str(xml_path),
            frame_skip=5,
            observation_space=None,
            render_mode=render_mode,
        )

        obs_size = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32,
        )

        self.water_level = 10.0
        self.termination_depth = 0.8
        self.counter = 0

        self.max_forward_vel = 20.0
        self.max_qvel = 50.0

    def _get_obs(self):
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def set_termination_depth(self, value):
        self.termination_depth = value

    def check_unstable(self):
        if not np.all(np.isfinite(self.data.qpos)):
            return True
        if not np.all(np.isfinite(self.data.qvel)):
            return True
        if np.max(np.abs(self.data.qvel)) > self.max_qvel:
            return True
        return False

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        for _ in range(self.frame_skip):
            self.data.ctrl[:] = action
            mujoco.mj_step(self.model, self.data)

            if self.check_unstable():
                obs = self._get_obs()
                return obs, -20.0, True, False, {"unstable": True}

        obs = self._get_obs()

        forward_vel = self.data.qvel[0]

        head_id = self.model.body("head").id
        head_z = self.data.xpos[head_id, 2]

        head_radius = 0.05
        target_head_z = self.water_level + head_radius

        head_error = target_head_z - head_z
        head_depth = head_error

        ctrl_cost = 0.005 * np.square(action).sum()

        #surface_bonus = np.exp(-10.0 * max(0.0, head_error))

        #reward = (
        #    2.0 * forward_vel
        #    + 0.5 * surface_bonus
        #    + 0.1
        #    - ctrl_cost
        #)

        reward = 1.0 * forward_vel

        if head_error > 0:
            reward -= 1 * head_error
        else:
            reward += 2.0
            #print(head_depth)
            #print("above water!")

        reward -= ctrl_cost

        reward += 0.5

        terminated = False
        truncated = False


        if head_depth > self.termination_depth:
            terminated = True
            reward -= 20.0

        if np.max(np.abs(forward_vel)) > 20.0:
            print(f"I was toooooo fast!!! vel: {forward_vel}")
            terminated = True
            reward -= 20.0

        info = {
            "head_z": head_z,
            "head_depth": head_depth,
            "forward_vel": forward_vel,
            "head_error": head_error,
            "ctrl_cost": ctrl_cost,
            "reward": reward,
        }

        if self.counter % 1000 == 0:
            print(
                f"head_z={head_z:.3f}, "
                f"depth={head_depth:.3f}, "
                f"vel={forward_vel:.3f}, "
                f"reward={reward:.3f}",
                flush=True,
            )

        self.counter += 1

        return obs, reward, terminated, truncated, info

    def reset_model(self):
        qpos = self.init_qpos + np.random.uniform(
            -0.05, 0.05, size=self.init_qpos.shape
        )

        # start close to but above the surface
        qpos[2] = self.water_level + 0.05

        qvel = np.zeros_like(self.init_qvel)
        self.set_state(qpos, qvel)
        return self._get_obs()


def make_worm_env(render_mode=None):
    env = WormSwimmingEnv(render_mode=render_mode)
    return TimeLimit(env, max_episode_steps=500)

if __name__ == "__main__":


    n_envs = 16


    train_env = make_vec_env(
        make_worm_env,
        n_envs=n_envs,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=1024,
        n_epochs=5,
        device="cpu",
    )

    stages = [
        ("Stage 1: high buoyancy", 0.8, 300_000),
        ("Stage 2: medium buoyancy", 1.5, 300_000),
        ("Stage 3: near-neutral buoyancy", 3.0, 400_000),
    ]

    for i, (name, term_depth, steps) in enumerate(stages):
        #print(name)
        #train_env.env_method("set_buoyancy_factor", buoyancy)
        train_env.env_method("set_termination_depth", term_depth)

        #checkpoint_callback = CheckpointCallback(
        #    save_freq=50000,        # every 50k timesteps
        #    save_path="./checkpoints/",
        #    name_prefix="surface_worm"
        #)

        model.learn(
            total_timesteps=steps,
            reset_num_timesteps=(i == 0),
            #callback=checkpoint_callback,
            progress_bar=False,
        )

    model.save("breathing_worm/ppo_worm_swimmer_with_surface")
    print(f"Model saved as ppo_worm_swimmer_with_surface.zip")
    train_env.close()