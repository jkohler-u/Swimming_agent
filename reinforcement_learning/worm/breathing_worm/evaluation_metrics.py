import csv
from pathlib import Path
import numpy as np


def calculate_metrics(csv_path):
    forward_velocities = []
    head_above_values = []

    with open(csv_path, mode="r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            forward_velocities.append(float(row["forward_velocity"]))

            head_above = row["head_above_water"].strip().lower() == "true"
            head_above_values.append(head_above)

    if not forward_velocities:
        raise ValueError(f"No data rows found in {csv_path}")

    average_forward_velocity = np.mean(forward_velocities)
    head_above_percentage = 100 * np.mean(head_above_values)

    longest_below_streak = 0
    current_below_streak = 0

    for head_above in head_above_values:
        if not head_above:
            current_below_streak += 1
            longest_below_streak = max(
                longest_below_streak,
                current_below_streak
            )
        else:
            current_below_streak = 0

    return {
        "average_forward_velocity": average_forward_velocity,
        "head_above_percentage": head_above_percentage,
        "longest_head_below_streak": longest_below_streak,
        "number_of_steps": len(forward_velocities),
    }


if __name__ == "__main__":
    csv_path = Path("worm_baseline/test_results.csv")

    metrics = calculate_metrics(csv_path)

    print(f"Evaluation steps: {metrics['number_of_steps']}")
    print(
        f"Average Forward Velocity: "
        f"{metrics['average_forward_velocity']:.4f}"
    )
    print(
        f"Head Above Water: "
        f"{metrics['head_above_percentage']:.2f}%"
    )
    print(
        f"Longest streak of head below water: "
        f"{metrics['longest_head_below_streak']} steps"
    )

