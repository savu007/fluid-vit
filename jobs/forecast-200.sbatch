#!/bin/bash
#SBATCH --job-name=forecast-200
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/global/u1/s/savu/projects/fluid-vit
#SBATCH --output=logs/slurm/forecast-200-%A_%a.out
#SBATCH --error=logs/slurm/forecast-200-%A_%a.err
set -euo pipefail
module load pytorch/2.8.0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
root="${1:?}";scratch="${2:?}";action="${3:?}"
if [[ "$action" == profile ]]; then
  names=(matey yalamanchi);name="${names[$SLURM_ARRAY_TASK_ID]}"
else
  name="${4:?}"
fi
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores \
  .venv/matey-memory-torch28/bin/python -u -B -m forecasting.long_training "$action" \
  --model "$name" --root "$root" --scratch "$scratch"
if [[ ! -f "$root/$name/complete.json" ]]; then
  job=$(sbatch --parsable --dependency="afterok:$SLURM_JOB_ID" --kill-on-invalid-dep=yes \
    jobs/forecast-200.sbatch "$root" "$scratch" train "$name")
  printf '%s\t%s\n' "$name" "$job" >> "$root/segments-$name.tsv"
fi
