#!/bin/bash
#SBATCH --job-name=vit-physical-loss
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --chdir=/global/homes/s/savu/projects/fluid-vit
#SBATCH --output=physical-loss-%j.out
#SBATCH --error=physical-loss-%j.err

set -euo pipefail
module load pytorch
export OMP_NUM_THREADS=8
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1

output="$PWD/runs/physical-loss-${SLURM_JOB_ID}"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ "${#nodes[@]}" -lt 2 ]]; then
  echo "Two allocated nodes are required." >&2
  exit 1
fi

# Check the shared experiment before either worker starts. No MSE-only runs.
python -u - "$output" <<'PY'
import json
import sys
from pathlib import Path
from forecasting.checkpoint import components
from forecasting.compare import validate_matched_specs
from forecasting.dataset import CacheIndex, SequenceDataset

configs = [components(name)[0](name + "/config-physical.yaml", allow_physical_loss=True)
           for name in ("baseline_vit", "yalamanchi_vit")]
if configs[0]["loss"] != configs[1]["loss"]:
    raise ValueError("Physical-loss settings must match")
specs = validate_matched_specs(configs)
for cfg in configs:
    index = CacheIndex(cfg)
    for split in ("train", "val"):
        SequenceDataset(index, split, cfg["training"]["horizon"])
root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=False)
(root / "experiment.json").write_text(json.dumps(
    {"loss": configs[0]["loss"], "matched_specs": specs}, indent=2))
print(f"Physical-loss experiment output: {root}", flush=True)
PY

srun --exclusive --exact --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --nodelist="${nodes[0]}" --gpus-per-task=1 --cpus-per-task=8 --cpu-bind=cores \
  python -u -m baseline_vit.train_physical \
  --config baseline_vit/config-physical.yaml --output "$output/baseline_vit" \
  > "$output/baseline_vit.log" 2>&1 &
baseline_pid=$!

srun --exclusive --exact --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --nodelist="${nodes[1]}" --gpus-per-task=1 --cpus-per-task=8 --cpu-bind=cores \
  python -u -m yalamanchi_vit.train_physical \
  --config yalamanchi_vit/config-physical.yaml --output "$output/yalamanchi_vit" \
  > "$output/yalamanchi_vit.log" 2>&1 &
yalamanchi_pid=$!

status=0
wait "$baseline_pid" || status=1
wait "$yalamanchi_pid" || status=1
echo "Worker exit status: $status. Results and logs: $output"
exit "$status"
