#!/bin/bash
#SBATCH --job-name=coarsen-131-160
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/coarsen-%j.out
#SBATCH --error=logs/slurm/coarsen-%j.err

set -euo pipefail

cd /global/homes/s/savu/projects/fluid-vit

module load pytorch

echo "Started: $(date -Is)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Python: $(command -v python)"
module list

# Verify that this job can actually use its allocated GPU.
srun --ntasks=1 --gpus=1 python -u -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0))"

# Measure the elapsed time of the complete coarsening command.
# This includes reading input, computation, verification, and saving output.
/usr/bin/time -p \
  srun --ntasks=1 --gpus=1 \
  python -u -m coarsening.coarsen \
  --config coarsening/config.yaml

echo "Finished: $(date -Is)"
