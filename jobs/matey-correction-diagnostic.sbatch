#!/bin/bash
#SBATCH --job-name=matey-corrections
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=/global/u1/s/savu/projects/fluid-vit
#SBATCH --output=logs/slurm/matey-corrections-%j.out
#SBATCH --error=logs/slurm/matey-corrections-%j.err
set -euo pipefail
module load pytorch/2.8.0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
checkpoint="${1:-${PSCRATCH:?}/fluid-vit/matey-memory/rollout5-59983390/best.pt}"
test -f "$checkpoint"
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores \
  "$PWD/.venv/matey-memory-torch28/bin/python" -u -B -m ornl_vit.memory.corrections \
  --checkpoint "$checkpoint" --horizon 5 --origins "${EVAL_ORIGINS:-8}" \
  --output "${PSCRATCH:?}/fluid-vit/matey-corrections-${SLURM_JOB_ID}" \
  --report "$PWD/runs/matey-corrections-${SLURM_JOB_ID}"
