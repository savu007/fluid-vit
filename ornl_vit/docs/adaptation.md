# Four-frame MATEY: architecture and adaptation record

## 1. Scope and approved decisions

This adapts ORNL's public `TurbT` to our coarsened 64³ velocity fields. Source:
[ORNL/MATEY at the pinned commit](https://github.com/ORNL/MATEY/tree/631a4139295d4f51e573d474cff7d6586f3aa310).
Local model identifier: `ornl_matey_turbt_3d_history4_v1`.

The researcher approved four observed frames, five recursively predicted frames,
baseline normalized MSE, Matey-Small width 192 and 2³ patches, and the public
implementation's three levels, factor-two filtering and twelve blocks (four per
level). Where paper and code do not map unambiguously, the approved reference is
**the public implementation**. This document describes executable source behavior;
it does not assert exact reproduction of a published training experiment.

Existing baseline and Yalamanchi models, checkpoints and jobs are unchanged.
This model starts with random weights, not a pretrained MATEY checkpoint.

## 2. Exact specification

| Item | Adaptation | Basis / significance |
| --- | --- | --- |
| Fields | Full 64³ `u,v,w` volumes | Existing coarsened cache; no pressure |
| History | Four consecutive frames | Approved |
| One call | One next full 3D velocity volume | Native output |
| Sequence | Five calls with sliding four-frame history | Approved local task |
| Width | 192 at each level | Selected Matey-Small anchor |
| Patches | 2×2×2 at each level | Selected Matey-Small anchor |
| Heads | Three; 64 features per head | Retained configuration |
| Hierarchy | Three field grids: 16³, 32³, 64³ | Approved public-code hierarchy |
| Blocks | Four per level; twelve total | Approved |
| Upsampling | Learned `UpsampleinSpace` | Retained source path |
| Position encoding | `PositionAreaBias` | Retained source path |
| Lead conditioning | Next recorded interval / reference interval | Local timestamp mapping |
| Loss | Fixed training-channel-standardized MSE | Explicit user selection |
| Backend | Local PyTorch SDPA | Dependency adapter; no RingX execution |
| Parameters | 6,612,393 registered | Measured from the configured model |

The parameter count includes upstream auxiliary modules, including registered
2D upsamplers not executed by this 3D forward path. It is not exclusively the
active-3D parameter count or a parameter-matched comparison.

The prepared optimizer settings retain the preceding experiment's seed 42,
AdamW, learning rate 1e-4, weight decay 0.01, gradient clipping at norm 1, and
100 epochs. Batch size 1 is a conservative initial execution setting, not a
measured capacity or a matched-throughput choice. There is no scheduler,
gradient accumulation, distributed trainer, or global AMP wrapper. All settings
are visible in `config.yaml`; no production run has begun.

## 3. Input and output

```text
history: [batch, 4, 3, 64, 64, 64]
targets: [batch, 5, 3, 64, 64, 64]
times:   [batch, 9]
```

Fields enter in cached physical velocity units, channel order `u,v,w`, retaining
cache axis ordering. We do not split the problem into independent 2D slices.
These are coarsened fields, not the original 384³ DNS resolution; evaluation
cannot establish preservation of structures absent from the cache.

The wrapper transposes the first two axes for the upstream time-first interface
and supplies state labels `[0,1,2]`. No forcing, pressure, Reynolds-number or
boundary-condition inputs are supplied. Each call receives fresh forward options
because upstream mutates these during hierarchy recursion.

Native output is `[batch,3,64,64,64]`. Upstream restores velocity scale internally.
No extra last-frame residual is added by our wrapper.

## 4. Coarse-to-fine hierarchy

Each input history is filtered twice to produce 32³ and 16³ histories. Upstream
uses a normalized Gaussian-like 2³ kernel with stride two. This is neither an
arithmetic block average nor a Fourier cutoff: its weights are proportional to
`exp(-distance_squared/2)` about index `kernel_size//2`. With an even kernel,
this center is not the block's geometric center. This behavior is preserved.

The coarsest model predicts first. At each finer level, upstream combines a
learned correction with an upsampled coarser prediction:

```text
prediction[level] = U(prediction[level-1])
                  + correction[level] - U(F(correction[level]))
```

`F` is filtering/downsampling; `U` is the learned upsampler. Since `U` is nonlinear,
this is not an exact orthogonal Fourier-band decomposition. Each level has its
own tokenization, transformer, decoding and conditioning parameters. The same
whole model is reused at each of the five forecast calls.

The upsampler uses trilinear enlargement, convolutions, instance normalization
and GELU activations. Some convolutions use reflection padding. **This is not
an explicitly periodic model**, despite the periodic fluid problem. No
incompressibility projection or conservation constraint is added. These upstream
properties are retained rather than silently corrected.

## 5. Patchification and attention

A 2³ patch contains eight spatial cells at a given hierarchy level. Each frame
is patchified separately using the upstream hierarchical MLP stem: strided 3D
convolutions, instance normalization and GELU produce a 192-component vector per
patch. This stem is more than a single flattened-patch linear projection.

| Level | Field grid | Tokens per frame | Four-frame tokens | Attention grouping |
| --- | --- | --- | --- | --- |
| Coarse, 0 | 16³ | 8³ = 512 | 2,048 | One global sequence |
| Middle, 1 | 32³ | 16³ = 4,096 | 16,384 | 64 groups, 256 tokens each |
| Fine, 2 | 64³ | 32³ = 32,768 | 131,072 | 512 groups, 256 tokens each |

For this configuration, each finer-level group contains a 4³ spatial token block
across all four frames: 4 × 4³ = 256 tokens. There are four groups per spatial
axis at the middle level and eight at the finest level. Groups do not overlap
or shift between blocks. Coarse-level attention provides global communication.

The selected `SpaceTimeBlock_all2all` joins spatial and temporal tokens in one
attention sequence. Tokens from different observed times can attend to one
another. Generic upstream docstrings mentioning separate alternating spatial
and temporal processing do not describe this selected execution path.

Each block has instance normalization, learned query/key/value projections,
per-head query/key layer normalization, scaled dot-product attention, an output
projection and an MLP. Residual updates have learned scale factors initialized
to 1e-6. The first block of each level receives the lead-time embedding.

`PositionAreaBias` adds an embedding of frame position, spatial position and
patch extent to token features. It is not a physics term or a PDE constraint.
Spatial coordinates are normalized; history time coordinates are ordinal
`0,1,2,3`.

The constructor has `drop_path=0.2`, but upstream forms twelve linearly spaced
probabilities and reuses the first four at every level. Actual per-level rates
are approximately `0, 0.0181818, 0.0363636, 0.0545455`. Claiming the deepest
executed block drops 20% of updates would be wrong. This behavior is preserved.
Stochastic depth is disabled in evaluation. Forecast generation does not sample
random fields; hardware/backend numerical reproducibility is a separate issue.

## 6. Normalization and exact loss

Inside MATEY, each sample/channel is normalized using its mean and standard
deviation across the four history times and all spatial cells, separately at
each hierarchy level. Standard deviation is clamped to at least 1e-4. The final
output restores the finest history's scale and mean. Future targets do not
contribute to these input statistics.

For the training loss, we instead use a fixed standard deviation `sigma_c` per
channel, estimated over the selected training files by the existing baseline
`training_statistics` function:

```text
L = mean over batch, lead=1..5, channel and grid cells of
        ((prediction - target) / sigma_channel)^2
```

This equals MSE between fields standardized by the baseline's training mean and
standard deviation: means cancel in the difference. All five leads have equal
weight. There is no spectral, gradient, divergence, energy or other physics loss.

The public upstream trainer uses relative squared error normalized by each
sample/channel's target spatial mean square plus 1e-7. The user explicitly
selected baseline normalized MSE instead. Matching model source therefore does
not imply matching upstream loss or training protocol.

## 7. Five-step recursion and gradients

```text
[a,b,c,d]     -> p1
[b,c,d,p1]    -> p2
[c,d,p1,p2]   -> p3
[d,p1,p2,p3]  -> p4
[p1,p2,p3,p4] -> p5
```

Only observed history and earlier predictions enter the forecast. Targets are
not arguments to the rollout function. Predictions remain attached to autograd;
the five losses are averaged before one backward pass. This is rollout training,
not detached pushforward training, and requires five sequential model calls.

However, **the upstream model stops gradients internally**: `filterdata` runs
under `torch.no_grad()`, including input filtering and filtered corrections.
Normalization means/standard deviations are also calculated without gradients.
The direct unfiltered paths remain differentiable, and all three levels receive
parameter gradients in tests. “Full rollout gradients” here means no additional
detaches between calls; it does not mean removing upstream gradient stops.

## 8. Timestamps and data windows

All nine recorded timestamps are retained and checked for finiteness and strict
increase. Intervals must match the reference interval within `rtol=0.003`,
`atol=1e-10`, preserving the approved 0.3% tolerance for the cache's spacing change.
The lead-time MLP receives `(next_time - last_history_time) / dt_reference`.

**History position encoding remains ordinal**, not an encoding of the exact
small variations between history intervals. Recorded times affect checks and
next-lead conditioning and accompany saved forecasts, but this is not a general
irregular-time model.

| Split | File IDs | Window policy | Windows |
| --- | --- | --- | --- |
| Training | 131–237; 107 files | Within file | 214 |
| Epoch validation | 451–490; 40 files | Within file | 80 |
| Standalone test | 492–531; 40 files | Checked cross-file | 353 |

Cache: `/pscratch/sd/s/savu/fluid-vit/hit-coarse-64-131-531`. This retains the
preceding training-strategy experiment's subset, not all cached training files.
IDs 450 and 491 remain declared gaps.

Ten-frame files permit two starts for a nine-frame window. The one-input,
five-target baseline permitted five starts per file. Equal file selections
therefore do not give equal windows, optimizer updates, or input information.
This is not a purely architectural comparison.

Cross-file evaluation checks continuity and equality of repeated boundary
fields before deduplication, without crossing splits. Forty ten-frame files
with shared endpoints give 361 unique times and 361 - 9 + 1 = 353 windows.
Forecast origin is the fourth observed frame, not the first history timestamp.

## 9. Saved weights and evaluation

Run outputs are `config.json`, `normalization.json`, `environment.json`,
`metrics.jsonl`, `best.pt` and `last.pt`. Existing output directories are rejected.
Best weights minimize average five-step validation normalized MSE. Checkpoints
include configuration, normalization, epoch, validation report, reference
interval, cache fingerprint and source identifier. Evaluation checks the cache
fingerprint. Optimizer/RNG state is not stored for exact training resumption;
these are inference checkpoints, not a resume implementation.

Standalone evaluation writes `report.json` and `first-forecast.npz`. The NPZ has
four input, five predicted and five actual full velocity volumes, plus nine
timestamps. Report start records refer to the beginning of the history.

Per lead, metrics include physical MSE/MAE, normalized MSE, aggregate relative
L2, last-observed-field persistence MSE, skill versus persistence, and kinetic
energy. Relative L2 is sqrt(total squared error / total target squared magnitude),
not mean per-sample relative L2. Zero relative/skill denominators yield null.
Energy is `0.5 * mean_space(u²+v²+w²)`, averaged over windows; it includes any
mean flow and is not mean-subtracted turbulent kinetic energy.

Energy is a diagnostic, not a loss. Spectra, divergence, dissipation and structure
functions are not currently evaluated here. Evaluation covers five future steps;
no claim about 100-step accuracy or physical plausibility is supported yet.

## 10. Dependency adapter and source audit

Only three imports changed in the original model directory, in
`matey/models/attention_modules.py`. They redirect stochastic depth, local
attention and lazy RingX imports through `matey/compat.py`. Original forward
bodies and RingX code remain unchanged. Five supporting files were copied from
the pinned commit; SHA-256 values appear in `upstream-helpers.json`. Minimal
package initializers are local additions.

The selected backend is PyTorch SDPA: same Q/K/V layout, default inverse-square-
root head scaling, no attention dropout, and no causal mask. All four frames
are available history, so their mutual attention does not leak target fields.
Upstream bfloat16 Q/K/V casts remain even on CPU; remaining computation is not
globally converted to bfloat16. Kernels can differ in rounding; bitwise
FlashAttention equivalence has not been tested or claimed.

No sequence-parallel process group is created, so the adapter does not execute
RingX or distribute training across GPUs. Installing FlashAttention, enabling
RingX, changing boundaries, removing gradient stops or changing scientific
settings would be separate changes to discuss, not implicit parts of this task.

## 11. Verification and limits

Seven adaptation tests passed: rollout history/times and gradients; baseline
loss equivalence; invalid/nonfinite handling; SDPA layout versus explicit
attention; actual upstream five-step forward/backward on a small grid with
older-history influence and all hierarchy levels receiving gradients; dataset
boundary handling; and a tiny train/save/reload/evaluate cycle with cache
mismatch rejection. Tests use synthetic temporary data.

All 58 existing regression tests across baseline, Yalamanchi, forecasting and
coarsening passed. A random-weight full 64³ width-192 CPU forward produced finite
output of the correct shape in about 11 seconds in the tested environment.
This is not a GPU training estimate.

No full-size five-step backward pass, GPU memory/runtime benchmark, RingX test,
FlashAttention comparison, or production training was performed. No Slurm job
was submitted. A compute-allocation smoke test remains necessary to establish
whether a full-size rollout fits the intended GPU before budgeting production.
