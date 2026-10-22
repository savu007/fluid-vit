#!/bin/bash
#SBATCH --job-name=matey-rollout5
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --chdir=/global/u1/s/savu/projects/fluid-vit
#SBATCH --output=logs/slurm/matey-rollout5-%j.out
#SBATCH --error=logs/slurm/matey-rollout5-%j.err
set -euo pipefail
module load pytorch/2.8.0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1 MATEY_JOB_SECONDS=54000
python_bin="$PWD/.venv/matey-memory-torch28/bin/python"
test -x "$python_bin" || { echo 'Run bash jobs/setup-matey-memory-env.sh first' >&2; exit 1; }
output="${PSCRATCH:?PSCRATCH must be set}/fluid-vit/matey-memory/rollout5-${SLURM_JOB_ID}"
extra=()
if [[ ${MATEY_SMOKE_ONLY:-0} == 1 ]]; then extra+=(--smoke-only); fi
# Disposable full-size acceptance precedes fresh training, validation and test plots.
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores \
  "$python_bin" -u -B -m ornl_vit.memory.run --variant baseline --config ornl_vit/memory/rollout.yaml \
  --output "$output" --report "$PWD/runs/matey-rollout5-${SLURM_JOB_ID}" "${extra[@]}"
