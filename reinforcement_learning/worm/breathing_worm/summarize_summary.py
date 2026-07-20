import csv
from pathlib import Path

# Directory containing the experiment folders
BASE_DIR = Path(__file__).resolve().parent

OUTPUT_FILE = BASE_DIR / "all_experiment_summaries.csv"

rows = []

# Find all experiment folders
experiment_dirs = sorted(
    [d for d in BASE_DIR.iterdir()
     if d.is_dir() and
     (d.name == "worm_baseline" or d.name.startswith("worm_without_"))]
)

for exp_dir in experiment_dirs:
    summary_file = exp_dir / "summary.csv"

    if not summary_file.exists():
        print(f"Skipping {exp_dir.name}: summary.csv not found")
        continue

    with open(summary_file, newline="") as f:
        reader = csv.DictReader(f)
        summary = next(reader)

    summary["experiment"] = exp_dir.name
    rows.append(summary)

if not rows:
    raise RuntimeError("No summary.csv files were found.")

# Put experiment name as the first column
fieldnames = ["experiment"] + [
    k for k in rows[0].keys() if k != "experiment"
]

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Combined {len(rows)} experiments into:")
print(OUTPUT_FILE)