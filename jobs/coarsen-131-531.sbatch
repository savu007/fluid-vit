#!/bin/bash
#SBATCH --job-name=coarsen-131-531
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --chdir=/global/homes/s/savu/projects/fluid-vit
#SBATCH --output=logs/slurm/coarsen-131-531-%j.out
#SBATCH --error=logs/slurm/coarsen-131-531-%j.err

set -euo pipefail
module load pytorch
export OMP_NUM_THREADS=32
export PYTHONDONTWRITEBYTECODE=1

# A separate cache for the 401 files, including the original 131-160 range.
# Refuse any existing output directory, including one from an interrupted run.
# No --force, no changes to the original cache, and no model training.
srun --ntasks=1 --gpus=1 python -u - <<'PY'
import logging
import time
from pathlib import Path

import torch

from coarsening.config import CoarsenConfig
from coarsening.coarsen import run_coarsen_pass
from coarsening.standalone import discover_files

logging.basicConfig(level=logging.INFO)
config = CoarsenConfig.from_yaml("coarsening/config-131-531.yaml")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable inside the GPU allocation")
ids = sorted(i for group in config.splits.all_groups().values() for i in group)
if ids != list(range(131, 532)):
    raise ValueError("This job requires exactly files 131 through 531")
sources = discover_files(config.root, config.file_glob, config.file_id_pattern)
missing = sorted(set(ids) - sources.keys())
if missing:
    raise RuntimeError(f"Missing source file IDs: {missing}")
cache = Path(config.coarsen.cache_dir)
# Atomic directory creation also prevents two submissions from sharing this cache.
cache.mkdir(parents=True, exist_ok=False)
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"Coarsening {len(ids)} files (4010 saved snapshots) into {cache}", flush=True)
start = time.perf_counter()
manifest = run_coarsen_pass(config, device=torch.device("cuda"))
print(f"Completed {len(manifest['file_ids'])} files in {time.perf_counter() - start:.1f} seconds",
      flush=True)
print(f"Manifest: {cache / 'manifest.json'}", flush=True)
PY
