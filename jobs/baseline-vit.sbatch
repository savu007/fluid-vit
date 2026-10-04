#!/bin/bash
#SBATCH --job-name=baseline-vit
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/global/homes/s/savu/projects/fluid-vit
#SBATCH --output=logs/slurm/vit-%j.out
#SBATCH --error=logs/slurm/vit-%j.err

set -euo pipefail

module load pytorch
export OMP_NUM_THREADS=8
export MPLBACKEND=Agg

# The configured output directory must not already exist.
# If changing training.output_dir, update the checkpoint and plot paths below.
srun --ntasks=1 --gpus=1 \
  python -u -m baseline_vit.train \
  --config baseline_vit/config.yaml

srun --ntasks=1 --gpus=1 \
  python -u -m baseline_vit.evaluate \
  --checkpoint baseline_vit/runs/baseline-01/best.pt \
  --split val \
  --plot baseline_vit/runs/baseline-01/velocity-val.png
