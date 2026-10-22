# Is MATEY's finer correction too weak?

Submit from fluid-vit:

```bash
sbatch jobs/matey-correction-diagnostic.sbatch
```

One GPU, one-hour allocation limit; no runtime guarantee or retraining. Default
checkpoint is the completed single-GPU rollout run 59983390/best.pt. An optional
first sbatch script argument selects another single-GPU full-rollout checkpoint.
RingX/pushforward/one-step checkpoints are rejected. EVAL_ORIGINS defaults to 8.

The diagnostic uses eight evenly spaced eligible test origins with four observed
frames followed by five autonomous predictions. It retains the recorded intervals,
accepted 0.3% spacing tolerance and identical boundary-deduplication checks.
Reference targets are only used after prediction to analyze corrections. There
is no teacher forcing and no alteration of correction amplitudes in the rollout.

Temporary method wrappers observe decoded candidates, the two calls to learned
upsampling at each finer level, and actual hierarchy returns. They restore all
instance attributes on exit. No upstream source, checkpoint or architecture is
modified. Exact reconstruction of the observed branch algebra is checked on
every call; CPU tests also verify identical traced/untraced predictions.

For each finer level, capture the candidate q, upsampled filtered candidate U(F(q)),
and the upsampled coarser prediction b. The implemented correction is c=q-U(F(q));
the internal combined prediction is b+c. Learned U includes convolution and
normalization, so this subtraction is not a mathematically exact high-pass
projection. Its output need not be divergence-free or energy-conserving.

The source only denormalizes the final finest return. Intermediate branches use
their own input-history normalization, and coarser returns are added without
explicit renormalization. Intermediate plots apply the current level's mean/std
as an explicitly labelled diagnostic affine view. They are not separately emitted
physical predictions. At the finest level the affine map is the actual one.
The target at each resolution is obtained by applying the source's actual filter
to the physical test field. The required correction is target minus the mapped
upsampled base; it may contain both fine detail and error in the coarse branch.

Metrics across all velocity components and grid points:

- RMS ratio: norm(c)/norm(d), with d the required correction.
- Cosine alignment: dot(c,d)/(norm(c)*norm(d)).
- Projection fraction: dot(c,d)/norm(d)^2.
- Oracle alpha: dot(c,d)/norm(c)^2, the signed scalar minimizing squared residual.
- Error reduction: 1 - norm(c-d)^2/norm(d)^2; negative means correction worsens error.
- Actual combined prediction MSE and before/after correction MSE.

Undefined ratios are null, not silently set to zero. An RMS ratio below one alone
does not establish the need for amplification: direction must be appropriate and
the correction must reduce error. The oracle uses targets and is diagnostic,
not an inference-time method. Results neither establish causal architectural
superiority nor justify changing weights without a separate validation experiment.

Full-volume component arrays for every origin/lead remain in scratch:
$PSCRATCH/fluid-vit/matey-corrections-JOBID/. The coarsest arrays include its
prediction and reference; finer levels also retain internal units and mean/std.

Home reports are under runs/matey-corrections-JOBID/: metrics.json/csv, spec.json,
first-origin-slices.npz, correction-metrics.png, two signed-u correction panel
figures (base, learned, required, target and combined), README.md, report.html,
and complete.json (or failure.json). The HTML embeds images for offline viewing.
Learned and required correction panels share color limits. Other field panels
share their own limits. Aggregate ranges describe overlapping origins, not
independent-sample confidence intervals. Existing runs are never overwritten.

CPU verification:

```bash
module load pytorch/2.8.0
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/matey-memory-torch28/bin/python -B -m unittest ornl_vit.memory.test_corrections
```
