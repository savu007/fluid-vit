#!/bin/bash
# Prepare four measured one-GPU requests. Never called automatically by Codex.
set -euo pipefail
cd /global/u1/s/savu/projects/fluid-vit
mode="${1:-profile}"
root="${2:-}"
if [[ -z "$root" ]]; then
  if [[ "$mode" == profile || "$mode" == pipeline ]]; then
    root="$PWD/runs/looped-finest-$(date -u +%Y%m%dT%H%M%SZ)"
  else
    # Timestamped directory names sort chronologically. Select the newest
    # submitted profile; never silently fall back to an older completed run.
    for candidate in "$PWD"/runs/looped-finest-*; do
      [[ ! -f "$candidate/profile-job.txt" ]] || root="$candidate"
    done
    [[ -n "$root" ]] || { echo 'No experiment found. Run bash jobs/submit-looped-profile.sh first.' >&2; exit 1; }
  fi
fi
case "$mode" in
  pipeline)
    LOOPED_AUTOMATIC=1 bash jobs/submit-looped-experiment.sh profile "$root"
    profile_job=$(cat "$root/profile-job.txt")
    coordinator=$(sbatch --parsable --dependency="afterok:${profile_job%%;*}" --kill-on-invalid-dep=yes \
      jobs/looped-coordinate.sbatch train "$root")
    printf '%s\n' "$coordinator" > "$root/coordinator-job.txt"
    printf 'Automatic pipeline scheduled. Coordinator: %s\nReports: %s\nNo further commands are required.\n' "$coordinator" "$root"
    ;;
  profile)
    [[ ! -e "$root" ]] || { echo "Directory exists: $root" >&2; exit 1; }
    mkdir -p "$root" logs/slurm
    job=$(sbatch --parsable jobs/looped-profile.sbatch "$root")
    printf '%s\n' "$job" > "$root/profile-job.txt"
    printf 'Profile array: %s\nReports: %s\n' "$job" "$root"
    if [[ ${LOOPED_AUTOMATIC:-0} != 1 ]]; then
      printf 'After success, run:\nbash jobs/submit-looped-training.sh\n'
    fi
    ;;
  train)
    test -n "${PSCRATCH:-}" || { echo 'PSCRATCH must be set' >&2; exit 1; }
    test ! -e "$root/training-jobs.tsv" || { echo 'Training already submitted for this directory' >&2; exit 1; }
    module load pytorch/2.8.0
    python_bin="$PWD/.venv/matey-memory-torch28/bin/python"
    # Validate every gate before submitting any training run. Pairwise allocate
    # the larger measured request to both treatments, retaining equal ceilings.
    requests=$("$python_bin" -B - "$root" <<'PY'
import json, sys
from pathlib import Path
from forecasting.looped_experiment import CASES, config_for, source_fingerprint
root=Path(sys.argv[1]); reports={}
for case in CASES:
    p=root/'profiles'/case/'profile.json'
    r=json.loads(p.read_text())
    if not r['accepted'] or r['case']!=case or r['config']!=config_for(case.split('-')[0]):
        raise SystemExit('Unaccepted or stale profile: '+case)
    if r['source_fingerprint']!=source_fingerprint():
        raise SystemExit('Source changed after profiling: '+case)
    if r['cuda_device'] is None:
        raise SystemExit('Need a full-size GPU profile: '+case)
    reports[case]=r
fingerprints={r['cache_fingerprint'] for r in reports.values()}
if len(fingerprints)!=1: raise SystemExit('Profiles used different cache fingerprints')
for name in ('matey','yalamanchi'):
    minutes=max(reports[name+'-'+v]['requested_minutes'] for v in ('control','looped'))
    if minutes>1440: raise SystemExit('Request exceeds 24 hours; need segmentation for '+name)
    for variant in ('control','looped'): print(name+'-'+variant,minutes)
PY
)
    scratch="$PSCRATCH/fluid-vit/$(basename "$root")"
    while read -r case minutes; do
      job=$(sbatch --parsable --time="$minutes" --job-name="$case" \
        jobs/looped-forecast.sbatch "$case" "$root" "$scratch")
      printf '%s\t%s\t%s\n' "$case" "$job" "$minutes" | tee -a "$root/training-jobs.tsv"
    done <<< "$requests"
    printf 'Weights: %s\nAfter all four finish:\nbash jobs/summarize-looped.sh\n' "$scratch"
    ;;
  summarize)
    module load pytorch/2.8.0
    "$PWD/.venv/matey-memory-torch28/bin/python" -B -m forecasting.looped_experiment summarize --root "$root"
    printf 'Comparison saved in %s\n' "$root"
    ;;
  *) echo 'Usage: bash jobs/submit-looped-experiment.sh [pipeline|profile|train|summarize]' >&2; exit 2 ;;
esac
