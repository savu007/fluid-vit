#!/bin/bash
#SBATCH --job-name=yalamanchi-vit
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
#SBATCH --output=yalamanchi-vit-%j.out
#SBATCH --error=yalamanchi-vit-%j.err

set -euo pipefail
module load pytorch
export OMP_NUM_THREADS=8
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1

# Usage: sbatch yalamanchi-vit.sbatch [config.yaml]
config="${1:-yalamanchi_vit/config.yaml}"
output=$(python -B -c 'import sys; from yalamanchi_vit.config import load_config; print(load_config(sys.argv[1])["training"]["output_dir"])' "$config")
horizon=$(python -B -c 'import sys; from yalamanchi_vit.config import load_config; print(load_config(sys.argv[1])["training"]["horizon"])' "$config")
srun --ntasks=1 --gpus=1 python -u -m yalamanchi_vit.train --config "$config"
srun --ntasks=1 --gpus=1 python -u -m yalamanchi_vit.evaluate \
  --checkpoint "$output/best.pt" --split val --horizon "$horizon" \
  --plot "$output/velocity-val.png" | tee "$output/evaluation-val.json"
