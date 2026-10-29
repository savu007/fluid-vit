#!/bin/bash
#SBATCH --job-name=looped-coordinate
#SBATCH --account=m5401
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --chdir=/global/u1/s/savu/projects/fluid-vit
#SBATCH --output=logs/slurm/looped-coordinate-%j.out
#SBATCH --error=logs/slurm/looped-coordinate-%j.err
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
action="${1:?Missing action}"
root="${2:?Missing experiment directory}"
trap 'printf "Coordination failed: action=%s line=%s\n" "$action" "$LINENO" > "$root/pipeline-failure.txt"' ERR
case "$action" in
  train)
    bash jobs/submit-looped-experiment.sh train "$root"
    dependency=afterany
    count=0
    while IFS=$'\t' read -r name job minutes; do
      dependency+=":${job%%;*}"
      count=$((count+1))
    done < "$root/training-jobs.tsv"
    [[ "$count" == 4 ]] || { echo 'Expected four submitted training jobs' >&2; exit 1; }
    summary=$(sbatch --parsable --dependency="$dependency" jobs/looped-coordinate.sbatch summarize "$root")
    printf '%s\n' "$summary" > "$root/summary-job.txt"
    ;;
  summarize)
    # Runs after failures too; the summarizer refuses incomplete results and
    # the ERR trap records failure instead of presenting a partial comparison.
    bash jobs/submit-looped-experiment.sh summarize "$root"
    printf 'Completed\n' > "$root/pipeline-complete.txt"
    ;;
  *) echo 'Unknown coordination action' >&2; exit 2 ;;
esac
