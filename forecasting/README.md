# Matched rollout comparison

For the new capacity-controlled experiment, see [matched specs](docs/matched-specs.md)
and `matched-comparison.sbatch`. The original rollout setup below is preserved.

See the [loss-design reference for DNS practitioners](docs/loss-design-for-dns.md)
for current objective semantics, physical metadata gaps, and candidate diagnostics.

Both architectures use the same dataset, loss, trainer, and evaluation functions
in this package. The default comparison changes the architecture and keeps the
data split, seed (42), five-step horizon, batch size (2), epochs (100), AdamW
learning rate (1e-4), weight decay (0.01), clipping (1), float32 arithmetic, and
zero dropout matched. Both start from scratch with zero output heads.
Parameter counts and computational costs differ with architecture.

## Submit the comparison

From the repository root:

```bash
sbatch rollout-comparison.sbatch
```

This requests two GPU nodes and runs both models concurrently, one model using
one GPU on each node. The runner pins each worker to a distinct allocated node.
No job has been submitted by the implementation. The four-hour allocation is
an initial request; GPU memory use and runtime have not been measured.
Five-step training retains all five graphs. If memory requires a smaller batch,
change **both** rollout configurations to keep the comparison matched.

The files controlling these experiments are
[baseline configuration](../baseline_vit/config-rollout.yaml) and
[Yalamanchi configuration](../yalamanchi_vit/config-rollout.yaml).
The runner refuses mismatched data or training settings before starting.
Custom configurations can be passed as:

```bash
sbatch rollout-comparison.sbatch \
  --baseline-config /absolute/path/baseline.yaml \
  --yalamanchi-config /absolute/path/yalamanchi.yaml
```

The output root is `runs/rollout-comparison-JOBID/`, relative to the repository.
It must be new. Existing experiments are never overwritten. Each model runs in
its own process with an independent GPU. The job waits for both workers; a failed
model does not cancel the other. Successful results are retained and failures
are recorded in `failures.json`; any failure makes the job exit unsuccessfully.

## GPU and node use

Each model currently trains on one GPU. Requesting two nodes runs the two
experiments concurrently; it does not distribute either model across GPUs.
Using several GPUs for one model would require distributed data parallel training
and an explicit decision about effective batch size and update counts.

The previous serial schedule was an implementation choice, not a model restriction.
Two single-GPU experiments can also fit on separate GPUs in one Perlmutter node;
see [NERSC single-GPU task examples](https://docs.nersc.gov/systems/perlmutter/running-jobs/#single-gpu-tasks-in-parallel).
The supplied script specifically uses two distinct nodes, as requested. It leaves
other GPUs on those nodes unused. An override to one node is rejected by this
launcher; packing onto one node requires a different resource mapping.

The batch shell runs the coordinator; each training and evaluation subprocess
runs through a node-pinned `srun`. `launch.json` records those commands.
Direct `python -m forecasting.compare --output NEW_DIRECTORY` retains serial
local execution for small CPU tests or a one-GPU allocation. The Slurm script
explicitly selects concurrent execution with `--launcher slurm`.

## Training meaning

Start from a DNS field and predict five successive fields recursively.
Compute the average normalized MSE over all five predictions and differentiate
through the entire sequence. There is no teacher forcing or detached step.

Each ten-snapshot file contributes five starts. The default splits supply
100 training windows and 20 validation windows. Windows never cross files.
Normalization is fitted to training snapshots only. Both models see the same
shuffled starts because the data-loader seed is independent of model initialization.

Each model selects its own checkpoint by the lowest mean validation normalized
MSE across all five leads. Equal epochs and batch sizes give equal optimizer
updates and predicted transitions, not equal runtime or model capacity.
This is one matched-seed experiment; it does not establish variation across seeds.

## Results to compare

The comparison root contains:

- `comparison.csv`: one row per architecture, split, and forecast lead.
- `comparison.json`: combined evaluation, including all-pair one-step metrics.
- `baseline_vit.log` and `yalamanchi_vit.log`: training logs and evaluation errors.
- Resolved YAML files for the two runs.
- `baseline_vit/` and `yalamanchi_vit/`: weights, statistics, configuration,
  environment details, epoch metrics, evaluation JSON, and velocity PNGs.

Compare physical MSE, relative L2, and skill versus persistence at each lead.
All rollout leads use the same starting windows. Persistence holds the original
input field fixed throughout the forecast. The separate one-step report uses
all valid pairs, so it has more samples than the first lead of the rollout.

The PNG is a **one-step** target/prediction/error slice, not a fifth-step result.
Epoch metrics include mean normalized training loss, per-lead validation metrics,
cumulative update/forward counts, elapsed seconds, and peak CUDA allocated memory.
Check `environment.json` for parameter counts and device/software versions.

Default jobs evaluate validation only. After choosing the comparison protocol,
`--test` can be passed to the submission command to also report held-out test
metrics; test values never influence checkpoint selection.

## Individual commands and existing experiments

Inside an allocated GPU session:

```bash
python -m baseline_vit.train --config baseline_vit/config-rollout.yaml
python -m yalamanchi_vit.train --config yalamanchi_vit/config-rollout.yaml
python -m baseline_vit.evaluate --checkpoint PATH/best.pt --split val --horizon 5
python -m yalamanchi_vit.evaluate --checkpoint PATH/best.pt --split val --horizon 5
```

Original baseline one-step training and evaluation commands still work.
The baseline evaluator without `--horizon` retains its original report format.
Old baseline one-step checkpoints can also be evaluated with `--horizon 5`.
Old Yalamanchi checkpoints remain evaluable, but pushforward configuration and
training support have been removed. No warm-start or optimizer-resume option
is included in this comparison workflow.

For a local synthetic check with the PyTorch module loaded:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fluid-vit-matplotlib \
python -B -m unittest baseline_vit.test_baseline yalamanchi_vit.test_baseline forecasting.test_comparison.ComparisonTests -v
```
