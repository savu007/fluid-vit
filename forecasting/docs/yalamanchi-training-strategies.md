# Yalamanchi: five-step pushforward versus five-step rollout

Submit from the repository root when ready:

```bash
sbatch jobs/yalamanchi-training-strategies.sbatch
```

The nine-hour allocation covers training and testing for both runs concurrently
on one node with two GPUs, one per strategy. It is a wall-time limit, not a claim
that either model needs nine hours. No job is submitted by preparing these files.

## Controlled experiment

Use the current Yalamanchi architecture (8,313,683 parameters, 4³ patches),
seed 42, batch size 2, float32, AdamW lr 1e-4, weight decay 0.01, clipping 1,
and 100 epochs for BOTH runs. This is an equal-epoch/update comparison, not an
equal-time-budget comparison. Both start from scratch with identical initial
weights and shuffled training-window order.

The previous 100-epoch run used 319 training files (131–449). The new subset has
max(50, ceil(319/3)) = 107 files, selected chronologically: 131–237. This keeps
training temporally coherent and avoids selecting files based on validation or
test performance. The remaining old training files are unused. There are 535
five-step training windows and 268 optimizer updates per epoch (26,800 total).
Normalization is recomputed using ONLY the 107 training files.

Validation remains files 451–490; test remains 492–531. Gap files 450 and 491
remain excluded. The scratch cache and original configs/runs are unchanged.
The runner derives and saves two resolved YAML configs from
[config-runtime.yaml](../../yalamanchi_vit/config-runtime.yaml).

- Rollout: recursively predict five steps, average normalized velocity MSE over
  all five, and backpropagate through the entire chain.
- Pushforward: predict the first four steps with gradients disabled, then train
  the fifth prediction using terminal normalized velocity MSE only. There is no
  teacher forcing within either sequence and no spectral or other physical loss.

Both select best.pt by the same mean five-step validation normalized MSE.
Training loss magnitudes are not directly comparable because their temporal
aggregation differs. This compares practical training recipes, including both
loss placement and gradient treatment; it does not isolate gradient detachment.

## 100-step held-out test

After training, evaluate best.pt on all 261 valid 100-step windows in the test
split. Forty ten-snapshot files have 39 identical repeated boundary snapshots,
so deduplication yields 361 unique snapshots. A window contains one initial field
and 100 future fields. Starts overlap and must not be treated as independent
replicates for statistical significance. They are identical between strategies.

The loader checks every boundary: equal timestamps must have identical finite
velocity arrays; otherwise the boundary is rejected. All nonzero intervals must
match the training reference interval within the approved 0.3% tolerance.
Actual recorded intervals are supplied to the model. Gaps and overlapping times
are rejected; windows never cross data splits. This assumes the chronological
files describe one continuous physical trajectory, consistent with the cache.

Evaluation uses batch size one and no gradients, without resetting predictions
to the reference field during the 100 steps. Reports include all leads 1–100:
normalized/physical MSE, MAE, relative L2, persistence skill, and predicted/reference
kinetic energy. No test value is used for fitting, early stopping, or checkpoint
selection. This is a new long-horizon generalization test, not an assurance of
accurate forecasts through step 100.

## Outputs

runs/yalamanchi-training-strategies-JOBID/ contains experiment.json (including
exact test starts), resolved YAML files, strategy logs, comparison.json and
comparison.csv. Each strategy subdirectory contains:

- best.pt: weights selected using validation only, with saved config and normalization.
- last.pt: weights after the last completed epoch, saved atomically each epoch.
- metrics.jsonl: training loss, common validation metrics, elapsed epoch seconds,
  optimizer updates, forward examples, and peak allocated CUDA memory.
- config.json, normalization.json, environment.json.
- evaluation-test-100.json: checkpoint identity and all 100 test leads.

These are inference checkpoints; optimizer-resume state is not saved. Existing
directories are refused. Worker exceptions are recorded in strategy-failure.json;
completed peer results are retained and the job exits unsuccessfully if either
worker fails. A scheduler timeout can prevent final aggregation; epoch weights
and metrics written before interruption remain. Nonfinite forecasts fail with
the offending forecast lead recorded in the worker failure message.

At the old measured throughput, 100 epochs on one-third the training files should
be substantially shorter than the prior 5h37m MSE run, but validation costs do not
shrink and GPU sharing/node conditions can affect runtime. Nine hours includes
headroom for 100-step testing; no new full-size run has been timed or submitted.
