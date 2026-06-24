import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import mujoco

class WormSwimmingEnv(MujocoEnv):
    def __init__(self, render_mode=None):
        self._model_path = r"C:\Users\malin\Documents\Uni\Master\neuromorphic control\Swimming_agent\underwater_swimming_worm\worm_with_surface.xml"

        # Water settings
        self.water_level = 10

        self.buoyancy_factor = 1.0
        self.termination_depth = 0.8

        # Tunable coefficients
        self.water_drag = 20.0
        self.air_drag = 0.05
        self.buoyancy_strength = 0.5

        super().__init__(
            self._model_path,
            frame_skip=5,
            observation_space=None,
            render_mode=render_mode
        )

        self.segment_body_names = ["segment1", "segment2", "segment3"]
        self.segment_body_ids = [
            self.model.body(name).id for name in self.segment_body_names
        ]

        obs_size = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)


    def apply_water_forces(self):
        self.data.xfrc_applied[:] = 0.0
        self.debug_water_info = []

        g = abs(self.model.opt.gravity[2])

        # Approximate capsule vertical size.
        # Your capsule length is 0.5 and radius is 0.05.
        # This creates a realistic transition zone near the surface.
        segment_length = 0.5
        segment_radius = 0.05
        surface_band = segment_length / 2 + segment_radius  # about 0.30 m

        for body_id in self.segment_body_ids:
            z = self.data.xpos[body_id, 2]
            lin_vel = self.data.cvel[body_id, 3:6]
            mass = self.model.body_mass[body_id]

            submerged = np.clip(
                (self.water_level - z + surface_band) / (2.0 * surface_band),
                0.0,
                1.0
            )

            water_drag = -self.water_drag * lin_vel
            air_drag = -self.air_drag * lin_vel

            drag = submerged * water_drag + (1.0 - submerged) * air_drag

            buoyancy = np.array([
                0.0,
                0.0,
                submerged * self.buoyancy_factor * mass * g
            ])

            force = drag + buoyancy

            self.data.xfrc_applied[body_id, :3] += force

            self.debug_water_info.append({
                "body_id": body_id,
                "z": z,
                "submerged": submerged,
                "mass": mass,
                "lin_vel": lin_vel.copy(),
                "force": force.copy(),
            })
    def step(self, action):
        for _ in range(self.frame_skip):
            self.apply_water_forces()
            self.data.ctrl[:] = action
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        forward_vel = self.data.qvel[0]

        head_id = self.segment_body_ids[0]
        head_z = self.data.xpos[head_id, 2]

        target_head_z = self.water_level + 0.05
        head_error = abs(head_z - target_head_z)

        reward = 3.0 * forward_vel
        reward -= 10.0 * head_error
        reward += 0.5

        ctrl_cost = 0.05 * np.square(action).sum()
        reward -= ctrl_cost

        truncated = False
        terminated = False

        head_depth = max(0.0, self.water_level - head_z - 0.05)

        if head_depth > self.termination_depth:
            terminated = True
            reward -= 20.0

        info = {
            "head_z": head_z,
            "head_depth": head_depth,
        }

        return obs, reward, terminated, truncated, info

    def reset_model(self):
        qpos = self.init_qpos + np.random.uniform(
            -0.05, 0.05, size=self.init_qpos.shape
        )

        # start close to the surface
        qpos[2] = self.water_level - 0.1

        qvel = np.zeros_like(self.init_qvel)
        self.set_state(qpos, qvel)
        return self._get_obs()
    
    def set_buoyancy_factor(self, value):
        self.buoyancy_factor = value

    def set_termination_depth(self, value):
        self.termination_depth = value


def make_worm_env(render_mode=None):
    return WormSwimmingEnv(render_mode=render_mode)



#if __name__ == "__main__":
#    env = WormSwimmingEnv(render_mode="human")
#    obs, info = env.reset()

 #   for step in range(1000):
 #       action = env.action_space.sample()
 #       obs, reward, terminated, truncated, info = env.step(action)
 #       env.render()

 #       if step % 50 == 0:
 #           print(f"\nStep {step}")
 #           for item in env.debug_water_info:
 #               print(
 #                   f"body={item['body_id']} "
 #                   f"z={item['z']:.3f} "
 #                   f"submerged={item['submerged']:.2f} "
 #                   f"force={item['force']}"
 #               )
 #               print("head_z:", info["head_z"], "reward:", reward)
#
#        if terminated or truncated:
#            obs, info = env.reset()

#    env.close()


if __name__ == "__main__":
    train_env = make_vec_env(lambda: make_worm_env(), n_envs=8)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
    )

    print("Stage 1: high buoyancy, at 0.8 terminating")
    train_env.env_method("set_buoyancy_factor", 1.05)
    train_env.env_method("set_termination_depth", 0.8)
    model.learn(total_timesteps=300_000)

    print("Stage 2: medium buoyancy, at 1.5 terminating")
    train_env.env_method("set_buoyancy_factor", 1.02)
    train_env.env_method("termination_depth", 1.5)
    model.learn(total_timesteps=300_000, reset_num_timesteps=False)

    print("Stage 3: near-neutral buoyancy, at 3.0 terminating")
    train_env.env_method("set_buoyancy_factor", 1.00)
    train_env.env_method("termination_depth", 3.0)
    model.learn(total_timesteps=400_000, reset_num_timesteps=False)

    model.save("ppo_worm_swimmer_with_surface")
    train_env.close()