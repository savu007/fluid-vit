# MATEY memory experiments

Two isolated experiments share five-step **pushforward** training and transformer
activation checkpointing. Neither modifies the existing rollout experiment or
its checkpoint format. The RingX version follows the authors' spatial partition
behavior, as approved; it is **not an execution-only equivalence comparison**.

## Submit

The project environment has been prepared with NERSC `pytorch/2.8.0` and the
official FlashAttention `2.8.3.post1` CUDA-12/Python-3.12/C++11-ABI wheel.
To recreate it, run `bash jobs/setup-matey-memory-env.sh` from the repository.
Both jobs use `.venv/matey-memory-torch28`; existing environments are unaffected.

```bash
sbatch jobs/matey-pushforward-baseline.sbatch
sbatch jobs/matey-pushforward-ringx.sbatch
```

The jobs are independent, with no dependency or sequential launcher. They request
one GPU and four GPUs, respectively, each on one node with a nine-hour wall limit.
Whether they run concurrently depends on Slurm scheduling. Each begins with a
disposable full-size acceptance run before fresh production initialization.
No jobs were submitted during preparation.

To submit only the acceptance stages:

```bash
sbatch --time=00:30:00 --export=ALL,MATEY_SMOKE_ONLY=1 jobs/matey-pushforward-baseline.sbatch
sbatch --time=00:30:00 --export=ALL,MATEY_SMOKE_ONLY=1 jobs/matey-pushforward-ringx.sbatch
```

Results and weights go to:

```text
$PSCRATCH/fluid-vit/matey-memory/baseline-JOBID/
$PSCRATCH/fluid-vit/matey-memory/ringx-JOBID/
```

Slurm stdout/stderr go to `logs/slurm/matey-pf5-{baseline,ringx}-JOBID.{out,err}`.

## Shared scientific settings

`experiment.yaml` resolves the existing `../config.yaml` and records the entire
resolved configuration with each run. Current settings are:

| Setting | Both experiments |
| --- | --- |
| Data | Coarsened 64³ velocity fields, channels u,v,w |
| Training | Files 131–237: 107 files, 214 within-file windows |
| Validation | Files 451–490: 40 files, 80 within-file windows |
| Test | Files 492–531: 40 files, 353 checked cross-file windows |
| Input | Four consecutive observed snapshots |
| Output/evaluation | Five recursive future snapshots |
| Model | Width 192, 2³ patches, three levels, four blocks per level |
| Hierarchy | Factor-two filtering, learned upsampling |
| Optimizer | AdamW, lr 1e-4, weight decay .01, gradient clip 1 |
| Training budget | Up to 100 epochs, batch size one **global sample** |
| Precision | FP32 surrounding operations, existing BF16 Q/K/V casts; no global AMP |
| Normalization | Same fixed training-channel standard deviations in the loss |
| Checkpoint selection | Minimum mean normalized MSE across all five validation leads |

Recorded times are retained with the existing 0.3% spacing tolerance. Training
statistics never use validation/test data. Both runs begin from the same seed and
model initialization. RingX uses rank-specific stochastic-depth draws afterward.

The training loss is **only on the fifth prediction**:

```text
[a,b,c,d] -> p1       no gradients
[b,c,d,p1] -> p2      no gradients
[c,d,p1,p2] -> p3     no gradients
[d,p1,p2,p3] -> p4    no gradients
[p1,p2,p3,p4] -> p5   with gradients and block checkpointing

loss = mean(((p5 - actual5) / training_channel_std)²)
```

Warmup calls remain in training mode, including stochastic depth. They have no
autograd graph. Checkpointing preserves RNG state and recomputes complete blocks
during backward. The upstream internal `no_grad` filtering remains unchanged.
This is not the old mean-of-five-losses full-gradient rollout objective.

## What the RingX version actually changes

The authors' `ringX_varlen_attn_func` is called directly through the existing
compatibility entry point. The vendored RingX implementation is unchanged. Its
FlashAttention dependency is installed; there is no custom kernel adapter or
silent CUDA fallback. Local fine/middle attention retains PyTorch SDPA.

One group of four GPU ranks cooperates on the same sample. The upstream geometry
helper partitions a 64³ field into a 1×2×2 arrangement of 64×32×32 subdomains.
Each rank patchifies, processes and decodes its own subdomain. RingX-A exchanges
information for **coarse global attention**. Fine/middle attention stays local.
Each rank feeds its own predictions back into its local history during pushforward.
Full fields are assembled for validation, test metrics and global energy spectra.

Consequences that must be acknowledged in results:

- Input normalization and block instance normalization use local subdomain
  statistics in the distributed run, rather than full-domain/global-sequence
  statistics where those were previously used.
- Learned decoder/upsampling convolutions operate at subdomain boundaries with
  the upstream padding behavior. No new halo exchange or periodic correction is
  added. Boundary artifacts are therefore possible.
- The upstream window subdivision depends on process geometry. On four ranks,
  fine/middle windows have 512 tokens instead of the baseline's 256, because the
  spatial partitions are anisotropic. Width, patch size and block counts match;
  **attention neighborhoods do not match exactly**.
- Coarse global attention spans all 2,048 tokens but distributes 512 per rank.
  Each rank's finest-level feature field has 32,768 tokens, versus 131,072 in the
  single-GPU baseline. This spatial partition reduces MLP activations too.
- Positional metadata records each subdomain's location in the full domain.
  The partition count is part of the checkpoint's execution specification;
  restore rejects a different count rather than silently changing the model.

Every rank computes mean loss over its equal-volume subdomain. RingX's backward
communicates activation gradients; afterward explicit parameter-gradient
averaging produces gradients for the global mean loss. This reduction happens
after backward to avoid interleaving DDP collectives with checkpoint replay.
Four ranks still perform **one optimizer update per global sample**, not four
independent updates. Parameters and optimizer states remain replicated.

## Acceptance, outputs and time limits

Each job executes three full-size pushforward/backward/AdamW updates, records
memory after every forecast lead and after the update, checks a checkpoint reload,
and performs five-step validation. The RingX job first compares the actual CUDA
kernel's output and Q/K/V gradients with full attention on a small distributed
case. Any failure stops the job before production training. Production reseeds
and initializes fresh weights; acceptance weights are not reused.

Files include:

- `execution.json`, `config.json`, `data.json`, `normalization.json`: provenance.
- `acceptance.json`, `acceptance.pt`: acceptance results and disposable weights.
- `metrics.jsonl`: final-lead training loss, per-lead validation, update counts,
  epoch runtime and memory on every rank.
- `best.pt`, `last.pt`: weights plus optimizer/RNG metadata. Last weights are also
  saved approximately every ten minutes. There is no automatic resume command.
- `test.json`, `first-forecast.npz`: five-step test metrics and a field example.
- `energy-data.npz`, `energy-evolution.png`, `energy-example.png`,
  `energy-spectra.png`: full-field kinetic energy and shell spectra against truth.
- `complete.json`: confirms training, selected-checkpoint testing and plots finished.
- `failure-rankN.json`: failure details if a stage raises an exception.

The jobs reserve at least one hour for final validation/checkpointing/testing,
increased if acceptance timing predicts a longer evaluation. This is an estimate,
not a guarantee against Slurm timeout. They may stop training before 100 epochs,
including at a partial epoch. Compare actual optimizer updates and elapsed time
in the output, not merely the requested wall limit. Changing the job wall time
for production also requires updating `MATEY_JOB_SECONDS` in its script.

Energy is K = .5 mean_space(u²+v²+w²), including mean flow. Spectra use integer
Fourier-mode shells and are computed after assembling the global field, never
by averaging independent subdomain spectra. These are diagnostics, not loss terms.

## Verification limits

CPU checks cover pushforward gradients/history, exact stochastic checkpoint
replay, checkpoint reload, complete baseline training/evaluation/plots, and
four-process partition/reassembly and synchronized gradients/updates. The CPU
distributed tests explicitly substitute an all-gather/SDPA **test oracle**;
they do not exercise the authors' CUDA RingX kernel and are not production code.
The actual RingX kernel and full-size A100 memory fit must pass the jobs' GPU
acceptance stages. No GPU acceptance run has been claimed during preparation.

```bash
module load pytorch/2.8.0
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
.venv/matey-memory-torch28/bin/python -B -m unittest ornl_vit.test_baseline ornl_vit.memory.test_memory
.venv/matey-memory-torch28/bin/python -B -m torch.distributed.run --standalone --nproc-per-node=4 -m ornl_vit.memory.test_distributed --pipeline
```

## One-step diagnostic (25 epochs)

Submit `sbatch jobs/matey-one-step.sbatch`. This requests one GPU for two hours;
it does not launch either comparison or RingX. Configuration: `one_step.yaml`.
Weights and reports go to `$PSCRATCH/fluid-vit/matey-memory/one-step-JOBID/`.

This experiment asks whether directly supervising the DNS-history-to-next-frame
mapping improves the weak first-step accuracy of the pushforward checkpoint.
It starts from scratch, not from the pushforward weights. Width 192, patches 2³,
12 blocks, four-frame history, stochastic depth .2, activation checkpointing,
normalization, optimizer, seed, batch size one, and data splits are unchanged.

Training makes exactly one model call with four observed frames and minimizes
`mean(((prediction1 - actual1) / training_channel_std)**2)`. There are no
unsupervised warmup predictions. The model remains in training mode, including
stochastic depth. The maximum is 25 epochs / 5,350 optimizer updates.

For a controlled diagnostic we retain the original 214 nine-frame training
windows, even though one-step training could use more windows. Only the first
future target is used in training. All five targets are retained for evaluation.
The 80 validation windows and 353 test windows also remain unchanged. Therefore
this diagnostic does not simultaneously test the benefit of additional windows.

Best-checkpoint selection uses **one-step validation normalized MSE**. Every
epoch still records five-step recursive validation. After training, the selected
checkpoint receives the existing five-step test, energy curves, and spectra.
The first test lead is the primary diagnostic; later leads are secondary and
are not directly supervised during training. Compare against the old MATEY
first-step test NMSE .100841 and Yalamanchi rollout .025143, while remembering
the different training durations, histories, and objectives.

`metrics.jsonl` uses `training_loss_one_step` and `supervised_lead: 1`.
`config.json` records training horizon 1; evaluation reports explicitly record
horizon 5. One-step checkpoints carry their own architecture/protocol identifier
and are supported by the memory runner's restore function. `best.pt` and
`last.pt` include weights and optimizer state. No automatic resume is provided.

Three disposable full-size training updates and checkpoint reload precede fresh
production initialization. Training stops after at most 90 minutes, or earlier
if the measured evaluation reserve requires it. At least 15 minutes is reserved
for evaluation, increased by the acceptance timing estimate. Completion within
two hours is not guaranteed until actual GPU timing is available. Inspect
`complete.json` for actual epochs/updates and whether training stopped for time.

Tests cover target/time alignment, a known analytical parameter update,
one-step checkpoint selection, and both objective paths through tiny training,
reload, testing, and energy plots. CPU verification does not establish full-size
GPU performance. No job is submitted automatically.

## Five-step rollout with a memory gate

Submit `sbatch jobs/matey-rollout.sbatch`: one GPU, 15 hours, up to 100 epochs.
Nothing is submitted automatically. The architecture, 107 training files,
214 training windows, batch one, optimizer, precision, seed and stochastic-depth
settings match the earlier checkpointed pushforward experiment. There is no
energy loss. The objective is normalized MSE averaged equally over all five
recursive predictions. Earlier predictions remain attached to autograd; no
teacher forcing and no pushforward detach/no-grad warmup are used.

The inherited upstream internal filtering/statistics gradient stops are unchanged.
“Full rollout” means no additional detach between model calls, not that we removed
those inherited operations. Block activation checkpointing remains enabled.

Before training, three full-size rollout/backward/AdamW updates run with disposable
weights. This tests memory after optimizer state exists, not just inference.
Peak allocated/reserved memory is recorded at each lead and after backward/update.
Checkpoint reload and validation must also pass. Production then initializes
fresh weights. An OOM or nonfinite update aborts the job: no silent batch,
resolution, precision, horizon or objective change occurs. CPU tests cannot
establish that this fits the available GPU.

The best checkpoint minimizes mean five-step validation NMSE. Test evaluation
uses the same 353 origins and produces five-step metrics, kinetic-energy curves,
spectra and first-example speed contours. Training has a 13-hour cap, with a
measured evaluation reserve that can stop it sooner. Inspect actual epochs and
stopping status in complete.json; 100 epochs within 15 hours is not guaranteed.

Weights: `$PSCRATCH/fluid-vit/matey-memory/rollout5-JOBID/{best,last}.pt`.
Home reports: `runs/matey-rollout5-JOBID/`, including learning curves, contours,
energy plots, metadata and a README pointing to scratch weights. Reports are
published on normal exit or caught failure; an uncatchable scheduler kill may
leave only scratch outputs. Failure details are retained in failure-rank0.json.
No automatic resume is implemented by the memory runner.

To request only the three-update acceptance test:

```bash
sbatch --time=00:30:00 --export=ALL,MATEY_SMOKE_ONLY=1 jobs/matey-rollout.sbatch
```

Tests include an analytic five-lead gradient that differs from detached
pushforward, tiny-model rollout training/reload/evaluation, and home-figure
publication, in addition to existing one-step/pushforward regression tests.

## RingX five-step rollout

For the checkpoint-only hierarchy correction diagnostic, see
[`forecasting/docs/matey-corrections.md`](../../forecasting/docs/matey-corrections.md).
It uses `jobs/matey-correction-diagnostic.sbatch` (one GPU, one hour), with no
training or alteration of the prediction pathway.

Submit `sbatch jobs/matey-rollout-ringx.sbatch` independently of the single-GPU
rollout job. It requests **four GPUs on one node for 7 hours 30 minutes: 30 requested
GPU-hours**, versus 15 for the single-GPU job. Both use rollout.yaml: up to 100 epochs,
214 training windows, batch one global history, normalized MSE averaged over all
five recursive predictions, gradients retained between calls, and checkpointing.
There is no energy-loss term and no silent objective change.

All four ranks cooperate on each sample using the authors' RingX kernel for
coarse global attention. This is spatial/context parallelism, not four independent
training examples. Partitioning retains the previously approved local normalization,
boundaries, and changed fine/middle attention neighborhoods described above;
this is not an execution-only architecture-equivalence comparison.

Before production, the job checks the actual CUDA RingX kernel against reference
attention, then performs three full-size rollout/backward/optimizer updates across
all ranks and verifies checkpoint reload and validation. Failure stops the job;
production begins with fresh weights only after acceptance succeeds. CPU tests
use the explicitly documented all-gather/SDPA oracle and do not verify the CUDA
kernel or full-size GPU memory fit.

Weights: `$PSCRATCH/fluid-vit/matey-memory/rollout5-ringx-JOBID/`.
Home reports: `runs/matey-rollout5-ringx-JOBID/` (published by rank zero).
For acceptance only:

```bash
sbatch --time=00:30:00 --export=ALL,MATEY_SMOKE_ONLY=1 jobs/matey-rollout-ringx.sbatch
```

CPU distributed rollout pipeline verification:

```bash
module load pytorch/2.8.0
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/matey-memory-torch28/bin/python -B -m torch.distributed.run \
  --standalone --nproc-per-node=4 -m ornl_vit.memory.test_distributed --pipeline --rollout
```

RingX allocation update: `MATEY_JOB_SECONDS=27000` matches the 07:30:00 Slurm
limit. The runner subtracts its measured evaluation reserve (at least one hour)
from that allocation before training. This deadline overrides the shared
configuration's longer training cap. The epoch target remains 100, but a
budget-limited run may stop earlier; inspect complete.json for actual epochs.
The single-GPU rollout allocation is unchanged.
