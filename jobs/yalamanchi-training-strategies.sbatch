#!/bin/bash
#SBATCH --job-name=yala-training-strategies
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=09:00:00
#SBATCH --chdir=/global/homes/s/savu/projects/fluid-vit
#SBATCH --output=logs/slurm/yala-training-strategies-%j.out
#SBATCH --error=logs/slurm/yala-training-strategies-%j.err

set -euo pipefail
module load pytorch
export OMP_NUM_THREADS=8
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1

output="$PWD/runs/yalamanchi-training-strategies-${SLURM_JOB_ID}"
python -u -m forecasting.compare_training prepare --output "$output"

# Separate tasks get separate GPU bindings; neither model uses DDP.
srun --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=1 \
  --cpus-per-task=8 --mem=32G --cpu-bind=cores \
  python -u -m forecasting.compare_training worker --strategy rollout --output "$output" \
  > "$output/rollout.log" 2>&1 &
rollout_pid=$!
srun --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=1 \
  --cpus-per-task=8 --mem=32G --cpu-bind=cores \
  python -u -m forecasting.compare_training worker --strategy pushforward --output "$output" \
  > "$output/pushforward.log" 2>&1 &
pushforward_pid=$!

status=0
wait "$rollout_pid" || status=1
wait "$pushforward_pid" || status=1
python -u -m forecasting.compare_training summarize --output "$output" || status=1
exit "$status"
