import cv2
import os
from pathlib import Path


def extract_frames(video_path, output_dir, frame_interval=30):
    """
    Extracts every `frame_interval`-th frame from a video.

    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory where images will be saved.
        frame_interval (int): Save every x-th frame.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done! Saved {saved_count} frames to '{output_dir}'.")


if __name__ == "__main__":
    extract_frames(
        video_path="reinforcement_learning//worm_baseline_cut.mp4",
        output_dir="results_in_pictures//worm_baseline",
        frame_interval=10,  # Save every x frame
    )