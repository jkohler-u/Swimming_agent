import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import mujoco
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import TimeLimit

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

class WormSwimmingEnv(MujocoEnv):
    def __init__(self, render_mode=None):
        #self._model_path = r"C:\Users\malin\Documents\Uni\Master\neuromorphic control\Swimming_agent\underwater_swimming_worm\worm_with_surface.xml"
        self._model_path = "/home/student/m/mbraatz/share/Swimming_agent/underwater_swimming_worm/worm_with_surface.xml"
        # Water settings
        self.water_level = 10

        self.buoyancy_factor = 1.0
        self.termination_depth = 0.8

        # Tunable coefficients
        self.water_drag = 20.0
        self.air_drag = 0.05
        self.buoyancy_strength = 0.5
        self.counter = 0


        super().__init__(
            self._model_path,
            frame_skip=5,
            observation_space=None,
            render_mode=render_mode
        )


        self.segment_body_names = ["segment1", "segment2", "segment3", "head"]
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

    def check_unstable(self, action):
        checks = {
            "qpos": self.data.qpos,
            "qvel": self.data.qvel,
            "qacc": self.data.qacc,
            "ctrl": self.data.ctrl,
            "xfrc_applied": self.data.xfrc_applied,
        }

        for name, arr in checks.items():
            if not np.isfinite(arr).all() or np.max(np.abs(arr)) > 1e6:
                print("\nUNSTABLE DETECTED:", name)
                print("time:", self.data.time)
                print("action:", action)
                print("qpos:", self.data.qpos)
                print("qvel:", self.data.qvel)
                print("qacc:", self.data.qacc)
                print("ctrl:", self.data.ctrl)

                for item in self.debug_water_info:
                    print(item)

                return True

        return False

        
    def step(self, action):
        for _ in range(self.frame_skip):
            self.apply_water_forces()
            self.data.ctrl[:] = action
            mujoco.mj_step(self.model, self.data)

            if self.check_unstable(action):
                obs = np.nan_to_num(self._get_obs(), nan=0.0, posinf=0.0, neginf=0.0)
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
    
    def set_buoyancy_factor(self, value):
        self.buoyancy_factor = value

    def set_termination_depth(self, value):
        self.termination_depth = value


def make_worm_env(render_mode=None):
    env = WormSwimmingEnv(render_mode=render_mode)
    return TimeLimit(env, max_episode_steps=500)



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
        ("Stage 1: high buoyancy", 1.05, 0.8, 300_000),
        ("Stage 2: medium buoyancy", 1.02, 1.5, 300_000),
        ("Stage 3: near-neutral buoyancy", 0.98, 3.0, 400_000),
    ]

    for i, (name, buoyancy, term_depth, steps) in enumerate(stages):
        print(name)
        train_env.env_method("set_buoyancy_factor", buoyancy)
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

    model.save("ppo_worm_swimmer_with_surface")
    train_env.close()