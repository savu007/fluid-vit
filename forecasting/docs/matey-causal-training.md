# MATEY temporal causal-attention experiment

## Submit both runs

From the fluid-vit directory:

```bash
bash jobs/submit-matey-causal.sh
```

Prepared, not submitted by the assistant. No arguments. This submits a two-task
GPU profile array (one GPU per task, 45 minutes), then each successful profile
submits its own dependent training chain. The two runs can execute in parallel.
Every training segment requests one GPU for 12 hours; up to 50 epochs per segment.
The runner reserves time before starting another epoch and can defer final test
evaluation to a fresh segment. Successful incomplete segments submit an afterok
continuation. Failed profiles or training do not launch successors. A new command
invocation creates a new experiment, not a duplicate-safe resume command.

There is no aggregate 12-hour cap. The previous ordinary MATEY 200-epoch run took
about 23.3 GPU-hours for training/validation. The new normalization and attention
mask can change speed and memory, so that runtime is only a reference. Initial
allocation caps are 1.5 GPU-hours for profiles plus two 12-hour training segments;
additional segments continue until each run completes 200 epochs and evaluation.
No full-size GPU memory or timing guarantee is claimed before profiling.

## Approved comparison

Both runs train fresh with seed 42 and the same parameter initialization:

| Setting | Causal | Normalization-matched control |
|---|---|---|
| Attention keys for frame i | Frames 0..i | All four observed frames |
| Within-frame spatial attention | Unrestricted in existing spatial group | Same |
| Input statistics | Per-frame history prefix | Same |
| Block normalization | Per frame, spatial tokens only | Same |
| Model parameters / architecture | Original current MATEY size | Same |
| Training | 200 epochs, five-step differentiable rollout normalized MSE | Same |

Same 107 training files 131–237, validation 451–490, test 492–531, original gaps,
training-only loss scale, cache, optimizer, learning rate, batch one, stochastic
depth, spatial hierarchy and checkpointing. Four physical input frames, 64³
velocity fields, 2³ spatial patches, width 192, three heads, three levels, four
blocks per level (12 total). No looping, auxiliary prediction loss, energy loss,
RingX, or teacher-forced rollout inputs are added.

Use the paired causal-versus-unmasked result to estimate the mask's effect. The
historical 200-epoch MATEY uses different normalization and is an additional
reference; a comparison against it cannot isolate the attention mask alone.

## Exact causal semantics

Tokens are time-major both globally and inside each upstream local spatial group.
The boolean SDPA mask allows a key at time j for a query at time i iff j <= i.
All spatial tokens of a permitted frame remain visible within that group's
existing attention footprint. This is not a triangular ordering of spatial patches.
Masks are cached per block as nonpersistent device buffers and are not learned.
QKV packing, head normalization, BF16 attention casts, residual scales, MLP and
stochastic-depth behavior are preserved.

At each hierarchy level, input frame i is normalized by detached per-channel
mean and sample standard deviation over spatial cells and input frames 0..i.
The existing 1e-4 standard-deviation floor remains. The newest input frame's
statistics exactly match the original four-frame normalization, including final
output denormalization. Earlier frames now use shorter-prefix statistics.

norm1, norm2 and mlp_norm inside attention blocks retain their original affine
parameters and InstanceNorm computation but operate separately per frame.
For local attention this is separately per frame and local spatial group; at the
coarsest level it covers the full frame's spatial token grid. These changes are
needed because original normalization pooled over all times and bypassed a mask.

The upstream recursive hierarchy and spatial tokenizers remain the same. A
private, per-instance forward-function binding substitutes prefix normalization
at the original preprocessing call. It does not patch process-global functions
or modify vendored files. The matched control uses the identical binding.

Only the final decoded frame is predicted/supervised. Earlier hidden tokens have
causal dependencies; they are not separately supervised physical-state estimates.
Causality alone does not enforce incompressibility, conservation, or improved
forecast quality. Original coarse-filter no_grad behavior remains; this experiment
does not repair that upstream gradient choice.

## Outputs and evaluation

Home: runs/matey-causal-TIMESTAMP/{causal,unmasked}/ contains profile, exact config,
metrics.jsonl, training curves, velocity contours, energy curves, spectra, raw
plotted slices, test.json and complete.json. Paired comparison.csv/json is written
when both are complete. Atomic writes avoid the unsupported filesystem lock that
failed in the previous experiment.

Scratch: $PSCRATCH/fluid-vit/matey-causal-TIMESTAMP/{causal,unmasked}/{best,last}.pt.
Checkpoints record variant, architecture ID, config, code/cache fingerprints,
optimizer, normalization, epoch and random-number states. Existing original
checkpoint loaders should not be used for these weights; rebuild
OrderedMateyForecast with the saved case and config before loading model state.

Slurm logs: logs/slurm/matey-causal-JOBID_TASK.{out,err}. Profile-job.txt and
segments-CASE.tsv identify continuation jobs. No jobs are submitted automatically
by importing Python modules.

Validation occurs after every epoch. Lowest mean five-step validation normalized
MSE selects each checkpoint. Final evaluation uses the same 353 cross-file
five-step test windows as the ordinary comparison. This prepares five-step testing,
not a new 100-step experiment. All training rollout predictions feed their own
history; no future target is inserted. Full gradients pass through five forecast
calls, subject to unchanged upstream no_grad operations.

## Checks

Tests perturb later physical input frames and verify exactly unchanged earlier
representations after every block at every level, with a noncausal negative
control. They check prefix statistics, parameter equality between treatments,
checkpointed-versus-ordinary gradients with stochastic depth, reload parity,
three-update profiling, train/resume/test, plots, source mismatch rejection, and
exact continuous-versus-segmented final weights on a tiny CPU fixture.

Full production GPU profiling checks three actual five-step forward/backward/
AdamW updates plus validation and a validation forecast, failing at >=90% allocated
GPU memory or when a conservatively estimated epoch will not fit the segment.
It does not promise that every future operation fits under all external I/O
conditions; a failed job retains committed checkpoints and does not auto-retry.
