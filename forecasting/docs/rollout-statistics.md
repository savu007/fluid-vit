# Stored-checkpoint 100-step comparison

## One-command training and evaluation

```bash
bash jobs/submit-rollout-pipeline.sh
```

This reuses the latest completed single-GPU MATEY rollout training checkpoint
when available and submits only the two-GPU evaluation (4-hour allocation).
An explicit `MATEY_CHECKPOINT` environment variable selects another saved model.
To deliberately train from scratch, run `bash jobs/submit-rollout-pipeline.sh --fresh`.
When no completed checkpoint exists, the command submits single-GPU MATEY
full-rollout training (100 epochs, 15-hour allocation), followed by evaluation.
The evaluation is submitted immediately with an afterok dependency and is
cancelled if that dependency becomes impossible. No GPU allocation is consumed
while evaluation waits on training. Maximum requested total is 23 GPU-hours.
Yalamanchi uses its existing checkpoint; it is not retrained. Training may stop
before 100 epochs under its existing time-budget logic.

An optional first argument selects a different Yalamanchi checkpoint. Set
EVAL_ORIGINS to change the number of evaluation origins. Job IDs and paths are
recorded under runs/rollout-pipeline-TRAIN_JOB_ID/submission.txt for new training,
or runs/rollout-evaluation-EVAL_JOB_ID/submission.txt when weights are reused.
Each invocation submits evaluation again; do not rerun it merely to monitor jobs. If
evaluation submission fails, the launcher leaves training intact and reports
how to submit only the missing evaluation stage.

## Evaluation only

Run `jobs/rollout-statistics.sbatch MATEY_BEST_PT [YALAMANCHI_BEST_PT]` with sbatch.
Two concurrent exclusive Slurm steps each bind one GPU. Four wall-clock hours
means at most eight allocated GPU-hours. No training or submission is performed
by the evaluation code. Only single-GPU, full-rollout-trained MATEY weights are
accepted; upstream RingX weights require their original distributed execution.

Default Yalamanchi is the 107-training-file rollout checkpoint from job 59524397.
Checkpoint data settings, normalization, cache fingerprint, test sequence and
recorded timestamps must agree. MATEY retains four observed inputs; Yalamanchi
uses the last input. Training differences still prevent a causal architecture claim.
Normalization means/standard deviations may differ only by floating-point reduction
roundoff (rtol=1e-12, atol=1e-15); file IDs and sample counts must match exactly.
Each checkpoint's original normalization is preserved for inference. Diagnostic
MSE scales and histogram bins use the common MATEY reference exactly. Both saved
normalizations are recorded in spec.json and checked exactly again by workers.

Eight evenly spaced eligible test origins, each followed for 100 steps, are used
by default. Set EVAL_ORIGINS before submission to change the count. Origins overlap
and are not independent trials. The evaluator deduplicates identical file-boundary
snapshots and checks the accepted 0.3% spacing tolerance. Actual recorded times
are passed to the models. No target velocity is fed back during prediction.

Metrics use physical velocity units, except normalized MSE which divides each
channel's squared error by its saved training variance. Absolute shell spectra
include the mean mode and all Fourier modes in saved arrays; plots display modes
1 through the axial Nyquist limit (32 on this grid).

Temporal autocorrelation bands are mode magnitude (0,4], (4,12], and (12,max].
These are diagnostic choices, not asserted inertial/dissipation ranges. The last
band includes corner modes beyond axial Nyquist. For each lag 0..50, sum spatial
dot products across all valid pairs within the 100 predicted frames, divided by
the square root of the corresponding summed endpoint energies. DNS is treated
identically. Spatial means are excluded; no temporal mean is subtracted. All
three velocity components contribute. Zero-energy bands are undefined (NaN).
Each forecast has its own curve; averaging nonstationary rollout pairs is a
finite-window diagnostic, not proof of a stationary invariant measure. Curves
include Eulerian advection. Mean recorded physical lag is saved and plotted.

Longitudinal increments are u_i(x+r e_i)-u_i(x), using periodic wrapping at
separations 1,2,4,8,16 cells and pooling the three axes. PDFs are saved at leads
1,5,10,25,50,100. All models and DNS use identical physical-unit bins spanning
plus/minus 12 times the RMS training-channel standard deviation. Underflow and
overflow probabilities are saved explicitly; plotted finite-range PDFs are not
renormalized. These are spatial statistics, not an energy-transfer measurement.

Home outputs: runs/rollout-statistics-JOBID/{spec.json,README.md,mse.csv,*.png}.
Per-model NPZ files retain per-origin arrays, spectra at every lead, PDFs/tails,
correlation curves, recorded times, and speed slices. Completed origins survive
a later failure; a complete combined summary requires both workers to finish.
No full predicted 3D trajectories or new weights are written.

Submission after training (replace JOBID):

```bash
sbatch --dependency=afterok:JOBID jobs/rollout-statistics.sbatch \
  /pscratch/sd/s/savu/fluid-vit/matey-memory/rollout5-JOBID/best.pt
```

The dependency guarantees successful job exit, not 100 completed training epochs;
the evaluator records each selected checkpoint's epoch in spec.json. Allocation
time is a limit, not a measured evaluation runtime guarantee.
