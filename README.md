# Turbulent-flow forecasting

Active training uses normalized velocity MSE over five recursive forecast steps.
The current large-data experiment targets similar runtime for the two models.

| Location | Purpose |
|---|---|
| baseline_vit/ | Global-attention ViT, configs and architecture guide |
| yalamanchi_vit/ | Hierarchical window-attention ViT with the same file layout |
| forecasting/ | Shared dataset, rollout, training, evaluation and comparison |
| coarsening/ | Fourier filtering and scratch-cache preparation |
| jobs/ | Slurm submission scripts; run from the repository root |
| runs/ | Active and historical MSE experiment outputs and checkpoints |
| logs/slurm/ | Slurm stdout/stderr |
| archive/physical_loss/ | Retired experiment source, logs and results |
| docs/notes/ | Other research notes |

## Current experiment

```bash
sbatch jobs/runtime-comparison.sbatch
```

This prepares two concurrent single-GPU workers on two nodes; nothing is
submitted automatically. Results go to runs/runtime-comparison-JOBID/.
Baseline uses width 240, ten heads, ten blocks, and 8³ patches (512 tokens).
Runtime equivalence is not established; the job retains a 24-hour limit.
See [runtime configuration rationale](forecasting/docs/runtime-comparison.md).
The original capacity-matched experiment remains in config-matched.yaml and
jobs/matched-comparison.sbatch (now requesting 24 hours). It is not the new default.
Original small-data configs and jobs are retained for reproducibility.

## Tests

```bash
module load pytorch
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fluid-vit-matplotlib \
python -B -m unittest baseline_vit.test_baseline yalamanchi_vit.test_baseline forecasting.test_comparison coarsening.test_standalone -q
```

[Baseline guide](baseline_vit/docs/architecture-for-dns.md) ·
[Yalamanchi guide](yalamanchi_vit/docs/architecture-for-dns.md) ·
[Loss reference](forecasting/docs/loss-design-for-dns.md)
