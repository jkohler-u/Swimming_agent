from pathlib import Path

import json
from datetime import datetime

import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from gymnasium.wrappers import TimeLimit
import mujoco


class WormSwimmingEnv(MujocoEnv):
    def __init__(self, render_mode=None):
        xml_path = Path(__file__).parent / "worm_water_surface.xml"

        self.log_interval = 10000
        self.log_buffer = {
            "reward": [],
            "forward_reward": [],
            "forward_vel": [],
            "head_error": [],
            "surface_reward": [],
            "depth_penalty": [],
            "ctrl_cost": [],
            "terminated": [],
            "successful_breathing": [],
        }

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

        self.water_level = 3.0
        self.termination_depth = 3
        self.counter = 0

        self.max_forward_vel = 20.0
        self.max_qvel = 50.0

        self.reward_cfg = {
            "surface_sharpness": 6.0, # how quickly the surface reward drops off as the head moves away from the target height
            "forward_weight": 12.0, # how much reward for forward progress
            #"max_rewarded_forward_vel": 1.5,
            "surface_weight": 2.0,
            "depth_weight": 3.0,
            "ctrl_weight": 0.0001,
            "termination_penalty": 1.0,
            "max_height_above_target": 0.8,
        }

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

        ### probably not needed
        # PPO can only send actions within the allowed actuator range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # =========================
        # Tunable reward parameters
        # =========================
        reward_cfg = self.reward_cfg

        # x position before the step
        x_before = self.data.qpos[0]

        for _ in range(self.frame_skip):
            self.data.ctrl[:] = action
            mujoco.mj_step(self.model, self.data)

            if self.check_unstable():
                obs = self._get_obs()
                print("Unstable state detected! Terminating episode.")
                return obs, -self.reward_cfg["termination_penalty"], True, False, {"unstable": True}

        # x position after the step
        x_after = self.data.qpos[0]

        # returns state that PPO sees (qpos and qvel)
        obs = self._get_obs()

        # actual distance moved in x direction during this step
        forward_progress = x_after - x_before

        # converts forward progress into velocity
        forward_vel = self.data.qvel[0] 
        
        vertical_vel = self.data.qvel[2]

        head_id = self.model.body("head").id

        # get the z position of the head
        head_z = self.data.xpos[head_id, 2]

        head_radius = 0.05
        target_head_z = self.water_level + head_radius
        #how far the head is from the target height <0 --> heas is too low, >0 --> head is too high
        head_error = head_z - target_head_z

        breathing_tolerance = 0.03   
        distance = max(
            0.0,
            abs(head_error) - breathing_tolerance
        )


        # smooth score for being close to the target height
        surface_quality = np.exp(
            -reward_cfg["surface_sharpness"] * distance**2
        )
        surface_reward = reward_cfg["surface_weight"] * surface_quality


        #maybe this is better *surface_quality to reward forward velocity only when the worm is breathing
        forward_reward = reward_cfg["forward_weight"] * forward_vel

        # penalty for beeing too deep, quadratic penalty for being below the target height
        depth_penalty = (
            reward_cfg["depth_weight"]
            * max(0.0, target_head_z - head_z) ** 2
        )

        ctrl_cost = reward_cfg["ctrl_weight"] * np.square(action).sum()

        reward = (
            forward_reward
            + surface_reward
            - depth_penalty
            - ctrl_cost
        )

        terminated = False
        truncated = False


        if head_z < target_head_z - self.termination_depth:
            terminated = True
            reward -= reward_cfg["termination_penalty"]

        if head_z > target_head_z + reward_cfg["max_height_above_target"]:
            print("Apperently I was flying!!!")
            terminated = True
            reward -= reward_cfg["termination_penalty"]

        if abs(forward_vel) > self.max_forward_vel:
            print(f"I was tooo fast!!! vel: {forward_vel}")
            terminated = True
            reward -= reward_cfg["termination_penalty"]

        successful_breathing = distance == 0

        self.log_buffer["reward"].append(reward)
        self.log_buffer["forward_reward"].append(forward_reward)
        self.log_buffer["successful_breathing"].append(float(successful_breathing))
        self.log_buffer["head_error"].append(head_error)
        self.log_buffer["forward_vel"].append(forward_vel)
        self.log_buffer["surface_reward"].append(surface_reward)
        self.log_buffer["depth_penalty"].append(depth_penalty)
        self.log_buffer["ctrl_cost"].append(ctrl_cost)
        self.log_buffer["terminated"].append(float(terminated))

        info = {
            "head_z": head_z,
            "reward": reward,
        }

        if self.counter % self.log_interval == 0 and self.counter > 0:
            reward_arr = np.array(self.log_buffer["reward"])
            forward_reward_arr = np.array(self.log_buffer["forward_reward"])
            forward_vel_arr = np.array(self.log_buffer["forward_vel"])
            head_error_arr = np.array(self.log_buffer["head_error"])
            surface_arr = np.array(self.log_buffer["surface_reward"])
            depth_arr = np.array(self.log_buffer["depth_penalty"])
            ctrl_arr = np.array(self.log_buffer["ctrl_cost"])
            term_arr = np.array(self.log_buffer["terminated"])
            successful_breathing_arr = np.array(self.log_buffer["successful_breathing"])

            print(
                "\n" + "-" * 50 +
                f"\nTraining diagnostics "
                f"(last {len(reward_arr)} env steps)"
                "\n\nReward:"
                f"\n  mean reward           : {reward_arr.mean():8.3f}"
                f"\n  mean breathing reward : {surface_arr.mean():8.3f}"
                f"\n  mean swimming reward  : {forward_reward_arr.mean():8.3f}"
                "\n\nSwimming values:"
                f"\n  mean forward velocity : {forward_vel_arr.mean():8.3f}"
                f"\n  mean head height error          : {head_error_arr.mean():8.3f}"
                f"\n  mean successful breathing : {successful_breathing_arr.mean():8.3f}"

                "\n\nPenalties:"
                f"\n  mean depth penalty    : {depth_arr.mean():8.3f}"
                f"\n  mean ctrl cost        : {ctrl_arr.mean():8.4f}"
                "\n\nEpisodes:"
                f"\n  terminations          : {int(term_arr.sum())}"
                "\n" + "-" * 50,
                flush=True,
            )

            for key in self.log_buffer:
                self.log_buffer[key].clear()

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
        vec_env_cls=SubprocVecEnv,
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
        ("Stage 1: high buoyancy", 1.5, 300_000),
        ("Stage 2: medium buoyancy", 1.0, 300_000),
        ("Stage 3: near-neutral buoyancy", 0.8, 400_000),
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


    save_dir = Path("models/repplicate")
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / "ppo_worm_swimmer_with_surface"
    model.save(model_path)

    metadata = {
        "model_path": str(model_path) + ".zip",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "reward_cfg": train_env.get_attr("reward_cfg")[0],
        "water_level": train_env.get_attr("water_level")[0],
        "max_forward_vel": train_env.get_attr("max_forward_vel")[0],
        "max_qvel": train_env.get_attr("max_qvel")[0],
        "stages": stages,
    }

    with open(save_dir / "ppo_worm_swimmer_with_surface_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    train_env.close()


#if __name__ == "__main__":
#    import time

 #   env = WormSwimmingEnv(render_mode="human")

  #  obs, info = env.reset()

   # while True:
        # Random actions
    #    action = env.action_space.sample()

     #   obs, reward, terminated, truncated, info = env.step(action)

      #  env.render()
       # time.sleep(0.02)

        #if terminated or truncated:
         #   print("Resetting environment...")
          #  obs, info = env.reset()