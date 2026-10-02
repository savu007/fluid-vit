# Yalamanchi-inspired 3D velocity forecasting baseline

An original local 3D SwinV2-UNet adaptation, **not an exact reproduction or
the authors' released code**. See the [source audit](docs/source-audit.md) and
[architecture guide for DNS practitioners](docs/architecture-for-dns.md).

The model predicts the next normalized 64³ velocity field from one field and
a relative timestep. It uses 4³ patches, widths 96/192/384, shifted 4³ attention
windows, two merging levels, convolutional additive skips, and periodic
reconstruction. Its zero-initialized increment head initially gives persistence.
The default model has 8,303,423 parameters.

Training and evaluation now use the shared `forecasting` package.
`config.yaml` retains one-step training; `config-rollout.yaml` enables fully
differentiated five-step training. Pushforward and warm-start support are removed.

For the matched two-model experiment:

```bash
sbatch rollout-comparison.sbatch
```

See the [comparison run guide](../forecasting/README.md) for settings, output
locations, metrics, tests, and individual commands. The existing
`yalamanchi-vit.sbatch` can still launch this model alone.

The package uses the same filenames as `baseline_vit`:

```text
config.yaml                 # one-step settings
config-rollout.yaml         # matched rollout settings
config.py                   # configuration validation
model.py                    # complete architecture and its building blocks
dataset.py                  # cache, normalization statistics, pairs, sequences
engine.py                   # shared training and metric functions
train.py                    # training command
evaluate.py                 # checkpoint evaluation command
visualize.py                # physical-velocity slice plots
test_baseline.py            # model and workflow tests
docs/architecture-for-dns.md # architecture explanation
```

Dataset, engine, and visualization modules expose shared implementations rather
than copying numerical code. The extra attention, block, and window files were
consolidated into `model.py`; test files were consolidated into `test_baseline.py`.
Checkpoint parameter names and architecture are unchanged by these moves.

Periodic windows have no artificial seam mask. Full-domain attention uses
minimum-image displacement with negative half-box ties. LayerNorm scales start
at one, convolutional skip scales at 1e-6, and only the output convolution is
zero initialized. Skip-refined features feed both deeper stages and decoder skips.
These are local choices, not verified author settings. No upstream code or
weights were vendored.

GPU runtime, memory, and turbulent-flow accuracy remain unmeasured.
The model has no exact incompressibility constraint. Spectra/divergence diagnostics,
multi-step plots, mixed precision, and distributed training are not implemented.

## Matched-specification experiment

`config-matched.yaml` defines the separate 10-block, 4³-patch experiment.
See the [specification and run guide](../forecasting/docs/matched-specs.md).

## Separate physical-loss experiment

See the [MSE plus energy-spectrum experiment](../forecasting/docs/physical-loss-experiment.md) for the new opt-in
training files, exact objective, checkpoints, and physical-loss.sbatch job.
