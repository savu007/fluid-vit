#!/bin/bash
#SBATCH --job-name=validation-slides
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/global/homes/s/savu/projects/fluid-vit
#SBATCH --output=logs/slurm/validation-slides-%j.out
#SBATCH --error=logs/slurm/validation-slides-%j.err

set -euo pipefail
module load pytorch
export OMP_NUM_THREADS=8
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
checkpoint="${1:-$PWD/runs/matched-comparison-59178955/yalamanchi_vit/best.pt}"
output="$PWD/runs/validation-slides-${SLURM_JOB_ID}"
srun --ntasks=1 --gpus-per-task=1 --cpu-bind=cores \
  python -B -u -m forecasting.presentation \
  --checkpoint "$checkpoint" --architecture yalamanchi_vit \
  --device cuda --horizon 10 --output "$output"
