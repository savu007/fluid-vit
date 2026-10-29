#!/bin/bash
#SBATCH --job-name=looped-forecast
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
#SBATCH --output=logs/slurm/looped-forecast-%j.out
#SBATCH --error=logs/slurm/looped-forecast-%j.err
# Submit through submit-looped-experiment.sh train: it overrides this placeholder
# allocation using measured profile.json. A profile is mandatory in Python too.
set -euo pipefail
module load pytorch/2.8.0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
case="${1:?Pass a case name}"
root="${2:?Pass the report directory}"
scratch="${3:?Pass the scratch directory}"
python_bin="$PWD/.venv/matey-memory-torch28/bin/python"
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores \
  "$python_bin" -u -B -m forecasting.looped_experiment train \
  --case "$case" --root "$root" --scratch "$scratch"
