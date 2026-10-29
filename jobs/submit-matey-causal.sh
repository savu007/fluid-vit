#!/bin/bash
# Explicit invocation submits two GPU profiles, then dependent training segments.
set -euo pipefail
cd /global/u1/s/savu/projects/fluid-vit
stamp=$(date -u +%Y%m%dT%H%M%SZ)
root="$PWD/runs/matey-causal-$stamp"
scratch="${PSCRATCH:?PSCRATCH must be set}/fluid-vit/matey-causal-$stamp"
mkdir "$root"
mkdir -p logs/slurm
job=$(sbatch --parsable --array=0-1 --time=00:45:00 jobs/matey-causal.sbatch "$root" "$scratch" profile)
printf '%s\n' "$job" > "$root/profile-job.txt"
printf 'Causal and normalization-matched unmasked pipelines submitted: %s\nReports: %s\nWeights: %s\n' "$job" "$root" "$scratch"
