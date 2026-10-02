# Matched-specification architecture comparison

This experiment compares a flat, global-attention ViT with the hierarchical
Yalamanchi-inspired model. It matches selected capacity and training controls;
it does not make the architectures or computational costs identical.

## Chosen specifications

| Setting | Previous baseline | Matched baseline | Previous Yalamanchi | Matched Yalamanchi |
|---|---:|---:|---:|---:|
| Input/output grid | 64³ | 64³ | 64³ | 64³ |
| Velocity channels | 3 | 3 | 3 | 3 |
| Patch edge | 8 | **4** | 4 | **4** |
| Initial token count | 512 | **4,096** | 4,096 | **4,096** |
| Transformer blocks | 6 | **10** | 10 | **10** |
| Feature widths | 192 | **240** | 96/192/384 | **96/192/384** |
| Attention heads | 6 | **10** | 3/6/12 | **4/8/16** |
| Features per head | 32 | **24** | 32 | **24** |
| MLP expansion | 4 | **4** | 4 | **4** |
| Trainable parameters | 3,360,192 | **8,020,272** | 8,303,423 | **8,313,683** |

The parameter difference is **3.53% of the matched Yalamanchi count**.
The runner enforces a maximum difference of 5%. This is a declared tolerance,
not exact parameter matching.

The baseline uses one fixed width. Yalamanchi retains two downsampling levels,
a bottleneck, and two upsampling levels, with two transformer blocks per stage:
[2, 2, 2, 2, 2]. Decoder stages use 8 and 4 heads respectively. Its two convolutional
skip modules are additional architectural components, not counted as transformer
blocks; their parameters are included in the total.

The width/head choices bring capacity close while keeping a common 24-feature
attention head and a 4× MLP. Changing Yalamanchi's head count slightly changes
its position-bias network size, so this configuration also starts from scratch.

## What remains different

The baseline attends globally to all 4,096 tokens at every block. Yalamanchi
uses 64-token spatial windows and shifted windows, with a full-domain window
only at its coarse bottleneck. It also retains merging/expansion, convolutional
skips, cosine attention, relative-position bias, post-normalization, and periodic
output convolution. The baseline retains dot-product attention, absolute-position
embeddings, pre-normalization, and linear patch reconstruction.

Those differences are the architecture being tested. Equal transformer-block
counts do not imply equal numbers of all operations, receptive fields, or FLOPs.
Neither equal embedding width nor equal head count is imposed across different
hierarchical resolutions.

Matching parameters does not make memory use or runtime equal. The baseline now
has eight times as many tokens as its earlier configuration, and global attention
has 64 times as many token pairs per head per block. That is an attention-pair
count, not a measured runtime or memory multiplier: kernel implementation matters.
Full-size GPU memory and runtime for these settings remain untested.

## Data and objective

The experiment uses all 401 cached files at
/pscratch/sd/s/savu/fluid-vit/hit-coarse-64-131-531, including two gap files:

- Training: 131–449, 319 files, 1,595 five-step windows.
- Validation: 451–490, 40 files, 200 five-step windows.
- Test: 492–531, 40 files, 200 five-step windows.
- Gaps: 450 and 491, excluded from training and evaluation.

These chronological splits and the cache are identical for both architectures.
Results should be compared within this new run, not treated as an architecture-only
improvement over the older run, which used less data and different held-out periods.

The shared settings remain seed 42, 100 epochs, batch size 2, five-step
full-gradient rollout, AdamW learning rate 10^-4, weight decay 0.01, gradient
clipping at 1, zero dropout, and float32 arithmetic. Both start at persistence
from zero output heads and use training-only normalization.

There are 798 updates per epoch (the final batch has one example), 79,800 updates
over 100 epochs, and 797,500 training forward examples per model. Each checkpoint is selected independently using
mean validation normalized MSE across all five leads. Test data are not evaluated
by default. No new statistical or physics loss is introduced.
See the [loss-design reference](loss-design-for-dns.md).

This is a single-seed comparison with a small, temporally correlated dataset.
It supports a more controlled comparison than the previous settings, not an
unqualified claim about which architecture is universally better.

## Explicit timestamp-spacing decision

The recorded interval is 0.0299835205078125 before file 474. Near timestamp 128,
it changes through 0.03003692626953125 to 0.030059814453125: up to **0.25445%**
above the original interval. The original relative-spacing tolerance is 0.01%.
The cause has not been verified; it should not be declared roundoff or a genuine
DNS timestep change without source evidence.

The user approved a **0.3% relative-spacing tolerance** for both matched configs
(dt_rtol=0.003, unchanged dt_atol=1e-10). The original configurations remain at
0.01%. The model continues to receive each recorded interval divided by the
training reference interval. Timestamps are not resampled or overwritten.

The timing change falls in validation and test, while the training interval is
constant. Timestep conditioning exists, but this does not establish reliable
variable-timestep generalization. This small distribution difference is shared
by the two models and must remain part of the experiment's interpretation.

## Running and saved outputs

From the repository root:

~~~bash
sbatch matched-comparison.sbatch
~~~

The job requests two nodes, one GPU per node, and a 12-hour maximum.
It runs the models concurrently and does not use distributed training within
either model. This is an initial walltime request, not a performance estimate.

Configurations are [baseline](../../baseline_vit/config-matched.yaml) and
[Yalamanchi](../../yalamanchi_vit/config-matched.yaml). They have matching names
and are separate from the original experiment configurations.

Outputs live under runs/matched-comparison-JOBID/ in the project directory:

~~~text
matched-specs.json
launch.json
comparison.csv
comparison.json
baseline_vit/best.pt
yalamanchi_vit/best.pt
~~~

Each model also saves resolved configuration, training statistics, environment
information, epoch metrics, validation evaluation, and a one-step velocity plot.
Model stdout and error logs are retained in the comparison root. Existing
output directories are refused.

The --matched-specs runner flag checks patch size, total transformer depth, head
dimension, MLP expansion, parameter counts, and the existing shared data/training
requirements. It saves the verified capacity report before launching workers.
It does not claim equal FLOPs or bitwise numerical reproducibility.

If GPU memory requires lowering batch size, lower it for **both** models and
record the resulting change in optimizer-update count. Do not treat a batch-size
change for only one architecture as the same experiment.

The old rollout job and configurations remain available. No production job is
submitted by preparing these files.
