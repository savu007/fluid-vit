#!/bin/bash
#SBATCH --job-name=matey-yala-compare
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=09:00:00
#SBATCH --chdir=/global/homes/s/savu/projects/fluid-vit
#SBATCH --output=logs/slurm/matey-yala-comparison-%j.out
#SBATCH --error=logs/slurm/matey-yala-comparison-%j.err

set -euo pipefail
module load pytorch
export OMP_NUM_THREADS=8
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1

output="$PWD/runs/matey-yalamanchi-comparison-${SLURM_JOB_ID}"
python -u -m forecasting.compare_matey prepare --output "$output"

# Independent fresh trainers, one explicitly bound GPU each; no DDP or RingX.
srun --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=1 \
  --gpu-bind=single:1 --cpus-per-task=8 --mem=64G --cpu-bind=cores \
  python -u -m forecasting.compare_matey worker --model matey --output "$output" \
  > "$output/matey.log" 2>&1 &
matey_pid=$!
srun --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=1 \
  --gpu-bind=single:1 --cpus-per-task=8 --mem=64G --cpu-bind=cores \
  python -u -m forecasting.compare_matey worker --model yalamanchi --output "$output" \
  > "$output/yalamanchi.log" 2>&1 &
yalamanchi_pid=$!

status=0
wait "$matey_pid" || status=1
wait "$yalamanchi_pid" || status=1
python -u -m forecasting.compare_matey summarize --output "$output" || status=1
exit "$status"
