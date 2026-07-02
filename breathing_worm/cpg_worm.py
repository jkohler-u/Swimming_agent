from pathlib import Path

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

        # PPO now controls CPG parameters, not raw torques:
        # [amplitude, frequency, phase_lag, bias]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.2, -np.pi, -0.5], dtype=np.float32),
            high=np.array([1.0, 3.0, np.pi, 0.5], dtype=np.float32),
            dtype=np.float32,
        )

        self.water_level = 3.0
        self.termination_depth = 0.8
        self.counter = 0

        self.max_forward_vel = 20.0
        self.max_qvel = 50.0

        self.cpg_phase = 0.0

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

    def cpg_to_motor_action(self, cpg_action):
        amplitude, frequency, phase_lag, bias = cpg_action

        # Advance internal oscillator phase
        self.cpg_phase += 2.0 * np.pi * frequency * self.dt

        motor_action = np.zeros(self.model.nu, dtype=np.float32)

        for i in range(self.model.nu):
            phase = self.cpg_phase + i * phase_lag
            motor_action[i] = amplitude * np.sin(phase) + bias

        return np.clip(motor_action, -1.0, 1.0)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        x_before = self.data.qpos[0]

        for _ in range(self.frame_skip):
            motor_action = self.cpg_to_motor_action(action)
            self.data.ctrl[:] = motor_action
            mujoco.mj_step(self.model, self.data)

            if self.check_unstable():
                obs = self._get_obs()
                return obs, -20.0, True, False, {"unstable": True}

        x_after = self.data.qpos[0]
        obs = self._get_obs()

        forward_progress = x_after - x_before
        forward_vel = forward_progress / (self.dt * self.frame_skip)
        vertical_vel = self.data.qvel[2]

        head_id = self.model.body("head").id
        head_z = self.data.xpos[head_id, 2]

        head_radius = 0.05
        target_head_z = self.water_level + head_radius
        head_error = head_z - target_head_z

        surface_quality = np.exp(-12.0 * head_error**2)

        forward_reward = 4 * np.clip(forward_vel, 0.0, 1.5) * surface_quality
        surface_reward = 0.3 * surface_quality

        too_deep = max(0.0, target_head_z - head_z)
        depth_penalty = 1.5 * too_deep**2

        backward_penalty = 2.0 * max(0.0, -forward_vel)
        stall_penalty = 0.1 if forward_vel < 0.05 else 0.0

        sink_vel = max(0.0, -vertical_vel)
        rise_vel = max(0.0, vertical_vel)
        vertical_penalty = 0.6 * sink_vel**2 + 0.1 * rise_vel**2

        # Penalize CPG parameter size mildly
        ctrl_cost = 0.002 * np.square(action).sum()

        reward = (
            forward_reward
            + surface_reward
            - depth_penalty
            - backward_penalty
            - stall_penalty
            - vertical_penalty
            - ctrl_cost
        )

        terminated = False
        truncated = False

        if head_z < target_head_z - self.termination_depth:
            terminated = True
            reward -= 20.0

        if head_z > target_head_z + 0.5:
            terminated = True
            reward -= 20.0

        if abs(forward_vel) > self.max_forward_vel:
            print(f"I was toooooo fast!!! vel: {forward_vel}")
            terminated = True
            reward -= 20.0

        info = {
            "head_z": head_z,
            "target_head_z": target_head_z,
            "head_error": head_error,
            "forward_vel": forward_vel,
            "vertical_vel": vertical_vel,
            "forward_reward": forward_reward,
            "surface_reward": surface_reward,
            "depth_penalty": depth_penalty,
            "backward_penalty": backward_penalty,
            "vertical_penalty": vertical_penalty,
            "ctrl_cost": ctrl_cost,
            "reward": reward,
            "cpg_amplitude": action[0],
            "cpg_frequency": action[1],
            "cpg_phase_lag": action[2],
            "cpg_bias": action[3],
        }

        if self.counter % 1000 == 0:
            print(
                f"head_z={head_z:.3f}, "
                f"vel={forward_vel:.3f}, "
                f"reward={reward:.3f}, "
                f"surface={surface_reward:.3f}, "
                f"forward={forward_progress:.3f}, "
                f"depth_penalty={depth_penalty:.3f}, "
                f"ctrl_cost={ctrl_cost:.3f}, "
                f"amp={action[0]:.3f}, "
                f"freq={action[1]:.3f}, "
                f"phase={action[2]:.3f}, "
                f"bias={action[3]:.3f}",
                flush=True,
            )

        self.counter += 1

        return obs, reward, terminated, truncated, info

    def reset_model(self):
        qpos = self.init_qpos + np.random.uniform(
            -0.05, 0.05, size=self.init_qpos.shape
        )

        qpos[2] = self.water_level + 0.05

        qvel = np.zeros_like(self.init_qvel)
        self.set_state(qpos, qvel)

        self.cpg_phase = 0.0

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
        ("Stage 2: medium buoyancy", 0.6, 300_000),
        ("Stage 3: near-neutral buoyancy", 0.5, 400_000),
    ]

    for i, (name, term_depth, steps) in enumerate(stages):
        print(name)

        train_env.env_method("set_termination_depth", term_depth)

        model.learn(
            total_timesteps=steps,
            reset_num_timesteps=(i == 0),
            progress_bar=False,
        )

    model.save("breathing_worm/ppo_worm_swimmer_with_surface_cpg")
    print("Model saved as ppo_worm_swimmer_with_surface_cpg.zip")
    train_env.close()