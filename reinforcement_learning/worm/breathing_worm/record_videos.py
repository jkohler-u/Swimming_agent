
import shutil
from pathlib import Path

from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import the environment definition from the training script.
#
# The training script must end with:
#
# if __name__ == "__main__":
#     main()
#
# This prevents training from starting when this file imports make_env.
from worm_training_updated import make_env


# Directory containing this evaluation script and the experiment folders.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent

# These filenames must match the files created by the training script.
MODEL_FILENAME = "ppo_swimmer.zip"
NORMALIZATION_FILENAME = "vec_normalize.pkl"

MAX_EVALUATION_STEPS = 5000
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
    Reconstruct the environment parameters used for an experiment from the
    experiment folder name.

    Supported folder names include:

        worm_baseline
        worm_without_survival_reward
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
            f"Unrecognized experiment folder name: '{folder_name}'."
        )

    removed_parameter = folder_name[len(prefix):]

    if removed_parameter not in env_params:
        raise ValueError(
            f"Folder '{folder_name}' refers to the unknown environment "
            f"parameter '{removed_parameter}'."
        )

    env_params[removed_parameter] = 0
    return env_params


def find_experiment_directories() -> list[Path]:
    """
    Find all experiment folders stored beside this evaluation script.

    ZIP files are ignored. Only normal directories are used.
    """
    experiment_directories: list[Path] = []

    baseline_directory = SCRIPT_DIRECTORY / "worm_baseline"

    if baseline_directory.is_dir():
        experiment_directories.append(baseline_directory)

    ablation_directories = sorted(
        directory
        for directory in SCRIPT_DIRECTORY.glob("worm_without_*")
        if directory.is_dir()
    )

    experiment_directories.extend(ablation_directories)
    return experiment_directories


def validate_experiment_files(
    experiment_directory: Path,
) -> tuple[Path, Path]:
    """
    Check that the trained model and normalization file exist.

    Returns:
        A tuple containing the model path and normalization path.

    Raises:
        FileNotFoundError: If one or both required files are missing.
    """
    model_path = experiment_directory / MODEL_FILENAME
    normalization_path = (
        experiment_directory / NORMALIZATION_FILENAME
    )

    missing_files = []

    if not model_path.is_file():
        missing_files.append(MODEL_FILENAME)

    if not normalization_path.is_file():
        missing_files.append(NORMALIZATION_FILENAME)

    if missing_files:
        missing_text = ", ".join(missing_files)

        raise FileNotFoundError(
            f"Missing required file(s) in "
            f"'{experiment_directory.name}': {missing_text}"
        )

    return model_path, normalization_path


def record_experiment(experiment_directory: Path) -> bool:
    """
    Load one trained experiment from its folder and record one deterministic
    evaluation episode.

    The resulting MP4 file is saved inside:

        <experiment folder>/videos/
    """
    model_path, normalization_path = validate_experiment_files(
        experiment_directory
    )

    env_params = get_environment_parameters(experiment_directory)

    video_directory = experiment_directory / "videos"

    # Remove previous recordings so each run leaves only the newest video.
    if video_directory.exists():
        shutil.rmtree(video_directory)

    video_directory.mkdir(parents=True, exist_ok=True)

    print()
    print(f"Recording experiment: {experiment_directory.name}")
    print(f"Model file: {model_path}")
    print(f"Normalization file: {normalization_path}")
    print(f"Video output: {video_directory}")

    # Create the original unnormalized environment with RGB rendering.
    raw_env = make_env(
        render_mode="rgb_array",
        **env_params,
    )

    # Record the first episode started by this environment.
    recorded_env = RecordVideo(
        env=raw_env,
        video_folder=str(video_directory),
        episode_trigger=lambda episode_number: episode_number == 0,
        name_prefix=experiment_directory.name,
        disable_logger=True,
    )

    # Stable-Baselines3 expects a vectorized environment.
    vector_env = DummyVecEnv(
        [lambda: recorded_env]
    )

    # Restore the observation normalization statistics from training.
    normalized_env = VecNormalize.load(
        str(normalization_path),
        vector_env,
    )

    # Evaluation must not modify the saved normalization statistics.
    normalized_env.training = False
    normalized_env.norm_reward = False

    # Load the trained PPO model.
    model = PPO.load(
        str(model_path),
        env=normalized_env,
    )

    completed_steps = 0
    episode_finished = False

    try:
        observation = normalized_env.reset()

        for step in range(MAX_EVALUATION_STEPS):
            action, _ = model.predict(
                observation,
                deterministic=DETERMINISTIC,
            )

            observation, _, done, _ = normalized_env.step(action)

            completed_steps = step + 1

            if bool(done[0]):
                episode_finished = True
                break

    finally:
        # Closing the environment finalizes the MP4 file.
        normalized_env.close()

    print(f"Recorded steps: {completed_steps}")

    if episode_finished:
        print("The episode ended because the environment terminated.")
    else:
        print(
            f"The episode reached the "
            f"{MAX_EVALUATION_STEPS}-step recording limit."
        )

    video_files = sorted(video_directory.glob("*.mp4"))

    if not video_files:
        print(
            f"Warning: no MP4 file was created in "
            f"'{video_directory}'."
        )
        return False

    for video_file in video_files:
        print(f"Saved video: {video_file}")

    return True


def main() -> None:
    experiment_directories = find_experiment_directories()

    if not experiment_directories:
        raise FileNotFoundError(
            "No experiment folders were found beside this script.\n"
            "Expected folders such as:\n"
            "  worm_baseline\n"
            "  worm_without_forward_reward\n"
            f"\nSearch directory:\n{SCRIPT_DIRECTORY}"
        )

    print(
        f"Found {len(experiment_directories)} experiment folder(s)."
    )

    successful_recordings = 0

    for experiment_directory in experiment_directories:
        try:
            recording_succeeded = record_experiment(
                experiment_directory
            )

            if recording_succeeded:
                successful_recordings += 1

        except Exception as error:
            # Continue recording the remaining experiments if one fails.
            print()
            print(
                f"Failed to record '{experiment_directory.name}': "
                f"{type(error).__name__}: {error}"
            )

    print()
    print(
        f"Finished: {successful_recordings} of "
        f"{len(experiment_directories)} experiment(s) "
        "recorded successfully."
    )


if __name__ == "__main__":
    main()
