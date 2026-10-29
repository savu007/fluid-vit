#!/bin/bash
set -euo pipefail
cd /global/u1/s/savu/projects/fluid-vit
root="$PWD/runs/forecast-200-$(date -u +%Y%m%dT%H%M%SZ)"
scratch="${PSCRATCH:?PSCRATCH must be set}/fluid-vit/$(basename "$root")"
mkdir -p "$root" logs/slurm
job=$(sbatch --parsable --array=0-1 --time=00:45:00 jobs/forecast-200.sbatch "$root" "$scratch" profile)
printf '%s\n' "$job" > "$root/profile-job.txt"
printf '200-epoch pipelines submitted: %s\nReports: %s\nWeights: %s\n' "$job" "$root" "$scratch"
