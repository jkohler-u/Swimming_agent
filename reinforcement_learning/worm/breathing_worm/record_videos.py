import os

# Required for rendering on many headless HPC nodes.
# Change "egl" to "osmesa" if EGL is unavailable on your cluster.
os.environ.setdefault("MUJOCO_GL", "egl")

import shutil
from pathlib import Path

from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import the environment definition from the training script.
# The training script must use:
#
# if __name__ == "__main__":
#     main()
#
# so importing it does not start training.
from worm_training_updated import make_env


# Directory containing this evaluation script and the experiment folders.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent

MODEL_FILENAME = "ppo_human_swimmer.zip"
NORMALIZATION_FILENAME = "vec_normalize.pkl"

MAX_EVALUATION_STEPS = 1000
DETERMINISTIC = True

BASE_ENV_PARAMS = {
    "leniency": 150,
    "survival_reward": 0.1,
    "smothness_reward": 0.2,
    "forward_reward": 3,
    "vel_punishment": 1,
    "cont_head_reward": 2,
    "head_punishment": 3,
    "cont_body_reward": 2,
    "cont_body_punishment": 1,
}


def get_environment_parameters(experiment_directory: Path) -> dict:
    """
    Reconstruct the reward parameters used for an experiment based on its
    directory name.

    Examples:
        worm_baseline
        worm_without_forward_reward
        worm_without_head_punishment
    """
    env_params = BASE_ENV_PARAMS.copy()
    folder_name = experiment_directory.name

    if folder_name == "worm_baseline":
        return env_params

    prefix = "worm_without_"

    if not folder_name.startswith(prefix):
        raise ValueError(
            f"Unrecognized experiment folder name: {folder_name}"
        )

    removed_reward = folder_name[len(prefix):]

    if removed_reward not in env_params:
        raise ValueError(
            f"Folder '{folder_name}' refers to unknown reward parameter "
            f"'{removed_reward}'."
        )

    env_params[removed_reward] = 0
    return env_params


def find_experiment_directories() -> list[Path]:
    """Find baseline and reward-ablation experiment directories."""
    directories = []

    baseline_directory = SCRIPT_DIRECTORY / "worm_baseline"

    if baseline_directory.is_dir():
        directories.append(baseline_directory)

    ablation_directories = sorted(
        directory
        for directory in SCRIPT_DIRECTORY.glob("worm_without_*")
        if directory.is_dir()
    )

    directories.extend(ablation_directories)
    return directories


def record_experiment(experiment_directory: Path) -> bool:
    """Load one trained experiment and record a deterministic episode."""
    model_path = experiment_directory / MODEL_FILENAME
    normalization_path = experiment_directory / NORMALIZATION_FILENAME

    if not model_path.exists():
        print(
            f"Skipping {experiment_directory.name}: "
            f"{MODEL_FILENAME} was not found."
        )
        return False

    if not normalization_path.exists():
        print(
            f"Skipping {experiment_directory.name}: "
            f"{NORMALIZATION_FILENAME} was not found."
        )
        return False

    try:
        env_params = get_environment_parameters(experiment_directory)
    except ValueError as error:
        print(f"Skipping {experiment_directory.name}: {error}")
        return False

    video_directory = experiment_directory / "videos"

    # Remove old videos so every run produces one clean result.
    if video_directory.exists():
        shutil.rmtree(video_directory)

    video_directory.mkdir(parents=True, exist_ok=True)

    print()
    print(f"Recording: {experiment_directory.name}")
    print(f"Model: {model_path}")
    print(f"Video directory: {video_directory}")

    # Create the unnormalized environment with image rendering enabled.
    raw_env = make_env(
        render_mode="rgb_array",
        **env_params,
    )

    # Record the first evaluation episode.
    recorded_env = RecordVideo(
        raw_env,
        video_folder=str(video_directory),
        episode_trigger=lambda episode_number: episode_number == 0,
        name_prefix=experiment_directory.name,
        disable_logger=True,
    )

    # Stable-Baselines3 models expect a vectorized environment.
    vector_env = DummyVecEnv([lambda: recorded_env])

    # Restore the observation-normalization statistics learned during training.
    normalized_env = VecNormalize.load(
        str(normalization_path),
        vector_env,
    )

    # Do not update normalization statistics during evaluation.
    normalized_env.training = False
    normalized_env.norm_reward = False

    # Load the model and attach the correctly normalized environment.
    model = PPO.load(
        str(model_path),
        env=normalized_env,
    )

    try:
        observation = normalized_env.reset()
        completed_steps = 0
        episode_finished = False

        for step in range(MAX_EVALUATION_STEPS):
            action, _ = model.predict(
                observation,
                deterministic=DETERMINISTIC,
            )

            observation, reward, done, info = normalized_env.step(action)

            completed_steps = step + 1

            if bool(done[0]):
                episode_finished = True
                break

        print(f"Recorded steps: {completed_steps}")

        if episode_finished:
            print("Episode ended because the environment terminated.")
        else:
            print(
                f"Episode reached the {MAX_EVALUATION_STEPS}-step "
                "recording limit."
            )

    finally:
        # Closing is necessary to finalize and write the MP4 file.
        normalized_env.close()

    video_files = sorted(video_directory.glob("*.mp4"))

    if video_files:
        for video_file in video_files:
            print(f"Saved video: {video_file}")
        return True

    print(
        f"Warning: no MP4 file was found in {video_directory}."
    )
    return False


def main():
    experiment_directories = find_experiment_directories()

    if not experiment_directories:
        raise FileNotFoundError(
            "No experiment folders were found. Expected folders such as "
            "'worm_baseline' or 'worm_without_forward_reward' in:\n"
            f"{SCRIPT_DIRECTORY}"
        )

    print(
        f"Found {len(experiment_directories)} experiment folder(s)."
    )

    successful_recordings = 0

    for experiment_directory in experiment_directories:
        try:
            recorded = record_experiment(experiment_directory)

            if recorded:
                successful_recordings += 1

        except Exception as error:
            # Continue with the remaining experiments when one recording fails.
            print()
            print(
                f"Failed to record {experiment_directory.name}: "
                f"{type(error).__name__}: {error}"
            )

    print()
    print(
        f"Finished: {successful_recordings} of "
        f"{len(experiment_directories)} experiment(s) recorded successfully."
    )


if __name__ == "__main__":
    main()