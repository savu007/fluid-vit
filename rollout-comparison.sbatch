#!/bin/bash
#SBATCH --job-name=vit-rollout-compare
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/global/homes/s/savu/projects/fluid-vit
#SBATCH --output=rollout-comparison-%j.out
#SBATCH --error=rollout-comparison-%j.err

set -euo pipefail
module load pytorch
export OMP_NUM_THREADS=8
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1

# One independent single-GPU worker per node, launched concurrently by the runner.
# The coordinator runs in the batch shell; each training/evaluation command uses srun.
python -u -m forecasting.compare --launcher slurm \
  --output "runs/rollout-comparison-${SLURM_JOB_ID}" "$@"
