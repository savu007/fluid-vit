#!/bin/bash
#SBATCH --job-name=matey-rollout5-ringx
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=07:30:00
#SBATCH --chdir=/global/u1/s/savu/projects/fluid-vit
#SBATCH --output=logs/slurm/matey-rollout5-ringx-%j.out
#SBATCH --error=logs/slurm/matey-rollout5-ringx-%j.err
set -euo pipefail
module load pytorch/2.8.0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1 MATEY_JOB_SECONDS=27000
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
python_bin="$PWD/.venv/matey-memory-torch28/bin/python"
test -x "$python_bin" || { echo 'Run bash jobs/setup-matey-memory-env.sh first' >&2; exit 1; }
"$python_bin" -B -c 'import flash_attn; from ornl_vit.matey.models.ringX_attn import ringX_varlen_attn_func'
output="${PSCRATCH:?PSCRATCH must be set}/fluid-vit/matey-memory/rollout5-ringx-${SLURM_JOB_ID}"
extra=()
if [[ ${MATEY_SMOKE_ONLY:-0} == 1 ]]; then extra+=(--smoke-only); fi
# One launcher sees all four GPUs; torchrun starts one rank per GPU.
# All ranks cooperate on the SAME history; this is context parallelism, not DDP batching.
srun --ntasks=1 --gpus-per-task=4 --gpu-bind=none --cpu-bind=cores \
  "$python_bin" -u -B -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  -m ornl_vit.memory.run --variant ringx --config ornl_vit/memory/rollout.yaml --output "$output" --report "$PWD/runs/matey-rollout5-ringx-${SLURM_JOB_ID}" "${extra[@]}"
