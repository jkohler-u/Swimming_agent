#!/bin/bash
#SBATCH --job-name=running_rl_training
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --error=logs/error.o%j
#SBATCH --output=logs/output.o%j

set -euo pipefail

echo "Running on node: $(hostname)"
echo "Start time: $(date)"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1


source /home/student/m/mbraatz/miniconda/etc/profile.d/conda.sh
conda activate mujoco

# Run your script
srun python -u adjusted_worm_training_updated.py

echo "Finished at: $(date)"