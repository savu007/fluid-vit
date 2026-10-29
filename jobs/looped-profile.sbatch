#!/bin/bash
#SBATCH --job-name=looped-profile
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --array=0-3
#SBATCH --chdir=/global/u1/s/savu/projects/fluid-vit
#SBATCH --output=logs/slurm/looped-profile-%A_%a.out
#SBATCH --error=logs/slurm/looped-profile-%A_%a.err
set -euo pipefail
module load pytorch/2.8.0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
cases=(matey-control matey-looped yalamanchi-control yalamanchi-looped)
root="${1:?Pass an experiment report directory}"
python_bin="$PWD/.venv/matey-memory-torch28/bin/python"
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores \
  "$python_bin" -u -B -m forecasting.looped_experiment profile \
  --case "${cases[$SLURM_ARRAY_TASK_ID]}" --root "$root"
