# Single-step 3D residual ViT

For a detailed explanation aimed at DNS practitioners, see the
[architecture guide](docs/architecture-for-dns.md).

Predict one future coarse velocity field from one observed field:

`U(t + dt) = U(t) + ViT(U(t), dt)`

Input and output are `[batch, 3, 64, 64, 64]`, in `u, v, w` order.
The default model uses a shared 3D convolution to embed non-overlapping `8^3`
patches into 512 tokens of width 192, learned spatial embeddings, a timestep
embedding, and six global-attention transformer blocks with six heads each.
A linear head reconstructs voxel-wise increments. There is no classification
token, temporal-history encoder, hierarchy, or causal mask. The head starts at
zero so the initial prediction is persistence. Backbone gradients start flowing
after the head moves away from zero.

This adapts the residual forecasting task discussed by
[Yalamanchi et al.](https://arxiv.org/abs/2604.02483) to full 3D volumes using a
plain ViT. It is **not** a reproduction of their 2D SwinV2-UNet. Global attention
uses PyTorch scaled dot-product attention. The model does not impose divergence
constraints or exact periodic translation equivariance. Position embeddings
are tied to this token grid. Patch embedding is a learned compression, not
another averaging operation.

## Files and configuration

- `config.yaml`: paths, file membership, model size, and training settings.
- `config.py`: validates those settings; relative paths resolve against the YAML directory.
- `dataset.py`: checks the cache and creates within-file temporal pairs.
- `model.py`: patch embedding, attention, transformer blocks, residual output.
- `engine.py`: normalization and one-step metrics.
- `train.py`: optimizer, validation selection, and checkpoint writing.
- `evaluate.py`: explicit validation/test evaluation using saved training statistics.
- `visualize.py`: optional PNG comparisons of predicted and true velocity slices.
- `test_baseline.py`: small synthetic checks; does not access the production dataset.

The default cache is `/global/homes/s/savu/data/hit-coarse-64`, matching the
current coarsening configuration. It must contain `.npy` arrays and the completed
`manifest.json`. Change this setting if your actual outputs are elsewhere.

| Split | File IDs | Pairs if each file contains 10 snapshots |
| --- | --- | --- |
| Train | 131–150 | 180 |
| Gap | 151 | Excluded |
| Validation | 152–155 | 36 |
| Gap | 156 | Excluded |
| Test | 157–160 | 36 |

Membership is explicit and disjoint. Timestamps establish ordering; neighboring files may share an endpoint timestamp,
but their time-range interiors cannot overlap. Train/validation/test time ranges
must remain strictly separated. Time units/calendars must agree, and all within-file
steps must match the median **training** interval within configured tolerances.
No examples bridge file boundaries, even if files are contiguous. Shared endpoint
snapshots remain in their respective files and are counted once per file when
computing training statistics. Gap files are
checked but never used for training, normalization, or evaluation. A one-file gap
does not establish statistical independence; assess physical temporal coverage.
If timestamps reset per file or files represent different trajectories, resolve
that interpretation before changing the checks.

The model receives `dt / training_reference_dt`; the scalar reference is recorded
in the checkpoint. This is a fixed-interval experiment. A timestep embedding does
not demonstrate generalization to unseen lead times.

Normalization uses per-channel population mean/std from every training snapshot,
with float64 accumulation. Validation/test fields do not enter these statistics.
Each dataset item returns physical values; the training/evaluation code applies
the frozen statistics. File metadata and array headers for all splits are checked
up front, but test field values are not used during training. Files are memory
mapped with a small per-dataset cache.

## Execution (for you to run)

No job submission is automated. These entry points run on whichever host invokes
them. On Perlmutter, run training/evaluation only inside resources you allocate.
Use the same PyTorch module/environment containing NumPy and PyYAML. CUDA is
required by default; the code fails rather than silently falling back to CPU.

From the project root, inside your existing GPU allocation:

```bash
module load pytorch
srun --ntasks=1 --gpus=1 python -u -m baseline_vit.train \
  --config baseline_vit/config.yaml
```

The initial configuration uses float32, batch size 2, AdamW with learning rate
`1e-4`, weight decay `0.01`, gradient clipping at `1.0`, and 100 epochs. These are
starting settings, not tuned results or a memory/runtime guarantee.

The default run directory is `baseline_vit/runs/baseline-01/` and contains:

- `config.json`: the resolved experiment settings.
- `normalization.json`: training statistics and their source files.
- `metrics.jsonl`: training MSE, validation metrics, and elapsed seconds per epoch.
- `best.pt`: model weights, training statistics, timestep scale, configuration,
  cache fingerprint, and epoch selected by the lowest validation normalized MSE.

The run directory must not already exist, to avoid overwriting an experiment.
Choose a new `training.output_dir` for each attempt. This initial trainer saves
the best model; it does **not** implement interrupted-training resume.

For the first diagnostic, set `training.max_train_pairs: 4`, choose a new output
directory, and check that training MSE falls substantially. Normalization still
comes from the full training split. Restore `max_train_pairs: null` for the full
training experiment. Validation curves on a four-pair overfit run are not a model
quality assessment.

Evaluate the held-out test set only after finalizing choices using validation:

```bash
srun --ntasks=1 --gpus=1 python -u -m baseline_vit.evaluate \
  --checkpoint baseline_vit/runs/baseline-01/best.pt --split test
```

Evaluation uses the configuration and normalization in the checkpoint. It refuses
to evaluate if the cache manifest or selected array sizes/mtimes have changed.
This fingerprint checks provenance metadata, not the full array contents.

Metrics printed as JSON include normalized MSE, physical MSE, aggregate relative
L2, persistence errors, MSE skill against persistence, and predicted/target mean
kinetic energy (unit density, velocity channels only). Positive skill means lower
MSE than persistence. The relative L2 is a global norm across evaluated examples,
not an average of per-snapshot relative errors.

This first implementation does not yet compute energy spectra, divergence, or
autoregressive rollouts. Add these before making claims about turbulence physics
or long-horizon stability. No pressure channel or fine-grid reconstruction is
part of this task.

## Synthetic tests

In an environment with PyTorch, NumPy, PyYAML, and Matplotlib:

```bash
python -m unittest baseline_vit.test_baseline -v
```

These test patch ordering, persistence initialization, gradient flow, temporal
pairing, train-only statistics, shared file endpoints, strict split separation,
persistence metrics, and physical-velocity plots along all three slice axes,
using small CPU tensors and temporary files.

## Visualize a forecast

Add `--plot` to checkpoint evaluation to save a PNG you can open in VS Code
or any image viewer. Matplotlib is required only for this option; rendering works
without a display server. The plot shows target, ViT prediction, and signed error
for each velocity component in physical values, with shared target/prediction
color limits per component.

From a GPU allocation:

```bash
srun --ntasks=1 --gpus=1 python -u -m baseline_vit.evaluate \
  --checkpoint baseline_vit/runs/baseline-01/best.pt --split val \
  --plot baseline_vit/runs/baseline-01/velocity-val.png \
  --sample-index 0 --slice-axis z --slice-index 32
```

The sample index selects a temporal pair within the chosen split (default 0).
The slice axis can be x, y, or z; the default is the central z plane.
Axes show voxel indices because the cache does not store physical spatial
coordinates. The image title records the source file and input/target times.
Evaluation still reports metrics over the entire selected split. Use validation
for development; reserve test evaluation for finalized choices. A trained
checkpoint is required; no example trained prediction is bundled.

## Matched rollout comparison

Full-gradient rollout training is available through `config-rollout.yaml`.
Both models share the same temporal dataset, objective, trainer, and evaluation.
See the [comparison guide](../forecasting/README.md) and submit
`sbatch rollout-comparison.sbatch` from the repository root to run both.
The existing one-step configuration and checkpoint evaluation remain supported.
