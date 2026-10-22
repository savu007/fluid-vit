#!/bin/bash
#SBATCH --job-name=rollout-statistics
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --chdir=/global/u1/s/savu/projects/fluid-vit
#SBATCH --output=logs/slurm/rollout-statistics-%j.out
#SBATCH --error=logs/slurm/rollout-statistics-%j.err
set -euo pipefail
module load pytorch/2.8.0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
python_bin="$PWD/.venv/matey-memory-torch28/bin/python"
matey="${1:?Pass the full-rollout MATEY best.pt as the first argument}"
yalamanchi="${2:-$PWD/runs/matey-yalamanchi-comparison-59524397/yalamanchi/best.pt}"
output="$PWD/runs/rollout-statistics-${SLURM_JOB_ID}"
test -f "$matey"
test -f "$yalamanchi"
"$python_bin" -B -m forecasting.physical_evaluation prepare --matey "$matey" \
  --yalamanchi "$yalamanchi" --output "$output" --origins "${EVAL_ORIGINS:-8}"
pids=()
for model in matey yalamanchi; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=1 \
    --gpu-bind=single:1 --cpus-per-task=8 --mem=64G --cpu-bind=cores \
    "$python_bin" -u -B -m forecasting.physical_evaluation worker \
    --model "$model" --output "$output" > "$output/$model.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
"$python_bin" -B -m forecasting.physical_evaluation summarize --output "$output" || status=1
exit "$status"
