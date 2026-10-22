#!/bin/bash
# One submission command; separate training and evaluation allocations.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
root="$PWD"
fresh=0
if [[ ${1:-} == --fresh ]]; then fresh=1; shift; fi
yalamanchi="${1:-$root/runs/matey-yalamanchi-comparison-59524397/yalamanchi/best.pt}"
test -f "$yalamanchi" || { echo "Missing Yalamanchi checkpoint: $yalamanchi" >&2; exit 1; }
yalamanchi="$(realpath -- "$yalamanchi")"
test -x "$root/.venv/matey-memory-torch28/bin/python" || {
  echo 'Run bash jobs/setup-matey-memory-env.sh first' >&2; exit 1;
}
: "${PSCRATCH:?PSCRATCH must be set}"
if [[ ${MATEY_SMOKE_ONLY:-0} != 0 ]]; then
  echo 'Unset MATEY_SMOKE_ONLY: this pipeline requires trained weights.' >&2
  exit 1
fi
if [[ ! ${EVAL_ORIGINS:-8} =~ ^[1-9][0-9]*$ ]]; then
  echo 'EVAL_ORIGINS must be a positive integer.' >&2; exit 1
fi
command -v sbatch >/dev/null
mkdir -p "$root/logs/slurm"
# Reuse the latest completed training run unless fresh training is requested.
checkpoint="${MATEY_CHECKPOINT:-}"
training_id=""
if [[ $fresh == 0 && -z $checkpoint ]]; then
  latest_id=0
  for completed in "$root"/runs/matey-rollout5-*/complete.json; do
    [[ -f $completed ]] || continue
    directory=${completed%/complete.json}
    candidate_id=${directory##*rollout5-}
    [[ $candidate_id =~ ^[0-9]+$ ]] || continue
    candidate="$PSCRATCH/fluid-vit/matey-memory/rollout5-$candidate_id/best.pt"
    if [[ -f $candidate && $candidate_id -gt $latest_id ]]; then
      latest_id=$candidate_id
      checkpoint=$candidate
    fi
  done
fi
if [[ $fresh == 0 && -n $checkpoint ]]; then
  test -f "$checkpoint" || { echo "Missing MATEY checkpoint: $checkpoint" >&2; exit 1; }
  checkpoint="$(realpath -- "$checkpoint")"
  echo "Reusing completed MATEY training: $checkpoint"
  evaluation_response=$(sbatch --parsable "$root/jobs/rollout-statistics.sbatch" "$checkpoint" "$yalamanchi")
  evaluation_id=${evaluation_response%%;*}
  [[ $evaluation_id =~ ^[0-9]+$ ]] || { echo "Unexpected evaluation response: $evaluation_response" >&2; exit 1; }
  record="$root/runs/rollout-evaluation-$evaluation_id"
  mkdir -p "$record"
  printf 'MATEY checkpoint: %s\nYalamanchi checkpoint: %s\nEvaluation job: %s\nResults: %s/runs/rollout-statistics-%s\n' \
    "$checkpoint" "$yalamanchi" "$evaluation_id" "$root" "$evaluation_id" | tee "$record/submission.txt"
  echo "Monitor: squeue -j $evaluation_id"
  exit 0
fi
training_response=$(sbatch --parsable "$root/jobs/matey-rollout.sbatch")
training_id=${training_response%%;*}
if [[ ! $training_id =~ ^[0-9]+$ ]]; then
  echo "Unexpected training submission response: $training_response" >&2; exit 1
fi
checkpoint="$PSCRATCH/fluid-vit/matey-memory/rollout5-$training_id/best.pt"
record="$root/runs/rollout-pipeline-$training_id"
mkdir -p "$record"
printf 'Training job: %s\nMATEY checkpoint: %s\nYalamanchi checkpoint: %s\n' \
  "$training_id" "$checkpoint" "$yalamanchi" | tee "$record/submission.txt"
if ! evaluation_response=$(sbatch --parsable --dependency="afterok:$training_id" \
    --kill-on-invalid-dep=yes "$root/jobs/rollout-statistics.sbatch" "$checkpoint" "$yalamanchi"); then
  echo "Evaluation submission failed. Training $training_id remains submitted." >&2
  echo "After fixing the submission issue, submit evaluation with --dependency=afterok:$training_id and checkpoint $checkpoint. Do not rerun this launcher unless you want fresh training." >&2
  exit 1
fi
evaluation_id=${evaluation_response%%;*}
if [[ ! $evaluation_id =~ ^[0-9]+$ ]]; then
  echo "Unexpected evaluation submission response: $evaluation_response; inspect squeue before resubmitting." >&2
  exit 1
fi
printf 'Evaluation job: %s\nResults: %s/runs/rollout-statistics-%s\n' \
  "$evaluation_id" "$root" "$evaluation_id" | tee -a "$record/submission.txt"
echo "Monitor: squeue -j $training_id,$evaluation_id"
