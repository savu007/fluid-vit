#!/bin/bash
#SBATCH --job-name=yala-energy-loss
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
#SBATCH --output=logs/slurm/yala-energy-loss-%j.out
#SBATCH --error=logs/slurm/yala-energy-loss-%j.err

set -euo pipefail
module load pytorch
export OMP_NUM_THREADS=8
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1

output="$PWD/runs/yalamanchi-energy-loss-${SLURM_JOB_ID}"
python -u -m yalamanchi_vit.energy_experiment prepare --output "$output"

srun --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=1 \
  --gpu-bind=single:1 --cpus-per-task=8 --mem=64G --cpu-bind=cores \
  python -u -m yalamanchi_vit.energy_experiment worker --arm energy-1 --output "$output" \
  > "$output/energy-1.log" 2>&1 &
one_pid=$!
srun --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=1 \
  --gpu-bind=single:1 --cpus-per-task=8 --mem=64G --cpu-bind=cores \
  python -u -m yalamanchi_vit.energy_experiment worker --arm energy-10 --output "$output" \
  > "$output/energy-10.log" 2>&1 &
ten_pid=$!

status=0
wait "$one_pid" || status=1
wait "$ten_pid" || status=1
python -u -m yalamanchi_vit.energy_experiment summarize --output "$output" || status=1
exit "$status"
