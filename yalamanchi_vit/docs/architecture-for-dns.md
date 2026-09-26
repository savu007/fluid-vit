# A DNS practitioner's guide to the Yalamanchi-inspired 3D ViT

**Status: initial implementation available.** Model, trainer, evaluation, tests,
and a Slurm script are now present. See [current implementation](../README.md)
and the [running instructions](../README.md). GPU experiments remain pending.
This original plan was prepared on September 25, 2026 against the
[plain ViT baseline](../../baseline_vit/docs/architecture-for-dns.md).

**Current training scope:** one-step and full-gradient rollout only, shared with
the plain baseline. Pushforward and warm-start support have been removed.
See the [matched comparison guide](../../forecasting/README.md) for current commands.
The remaining milestone discussion records the original architecture plan.

The objective is to build a second, independently selectable baseline for
**deterministic next-snapshot prediction of coarse 3D velocity**. The existing
model remains the reference experiment.

We have not located a verified public implementation of the Yalamanchi paper.
Accordingly, the proposed model should be described as a **3D adaptation
inspired by Yalamanchi et al.**, not an exact reproduction or a port of their
source code. The [source audit](source-audit.md) records what was checked,
available building blocks, and unresolved details.

The intended reader understands DNS and attention but may not know neural
network engineering. This document explains the physical task, tensor layouts,
proposed architecture, training, evaluation, and ordered implementation work.

## Reading map

1. [Evidence and scope](#1-evidence-and-scope)
2. [The physical forecasting contract](#2-the-physical-forecasting-contract)
3. [Proposed defaults and tensor shapes](#3-proposed-defaults-and-tensor-shapes)
4. [Patch embedding and conditioning](#4-patch-embedding-and-conditioning)
5. [Window attention and periodicity](#5-window-attention-and-periodicity)
6. [SwinV2 block requirements](#6-swinv2-block-requirements)
7. [Encoder, decoder, and skip connections](#7-encoder-decoder-and-skip-connections)
8. [Output reconstruction and initialization](#8-output-reconstruction-and-initialization)
9. [Training regimes](#9-training-regimes)
10. [Data reuse and temporal windows](#10-data-reuse-and-temporal-windows)
11. [Evaluation and comparison](#11-evaluation-and-comparison)
12. [Implementation milestones](#12-implementation-milestones)
13. [Proposed code organization](#13-proposed-code-organization)
14. [Compute and reproducibility](#14-compute-and-reproducibility)
15. [Open decisions and acceptance criteria](#15-open-decisions-and-acceptance-criteria)
16. [Glossary](#16-glossary)

## 1. Evidence and scope

The paper describes a 2D SwinV2-UNet with patch embedding, shifted-window blocks,
patch merging/expansion, ConvNeXt-assisted additive skips, and field reconstruction.
Forecasting adds a predicted increment to the observed field. Time and data-source
embeddings condition the tokens. It studies single-step training, five-step
backpropagated rollouts, and pushforward training with a terminal loss and
detached intermediate steps. Its examples use jet-flow slices/projections.
These are the architectural and training ideas we will investigate.
[Paper, Sections 3 and 4.1](https://arxiv.org/html/2604.02483v1#S3)

Everything below is a **proposed implementation for this repository**, unless
explicitly identified as a verified upstream detail. Numerical choices are
starting hypotheses, not recovered author hyperparameters.

| Decision | Proposed treatment |
|---|---|
| Spatial dimensionality | Full 3D volumes, not independent 2D slices |
| Variables | Three coarse velocity components, in u/v/w order |
| Boundary topology | Periodic cube, explicitly handled in spatial operations |
| Input history per forward call | One snapshot |
| Output | One successor field through a learned increment |
| Dataset | Existing spectrally filtered cache and current file split |
| First experiment | One-step loss to isolate the architectural change |
| Current comparison | Matched five-step full-gradient training for both architectures |
| Multimodal field reconstruction | Outside this baseline's initial scope |
| Source conditioning | Disabled initially because there is one data source |

The 2D-to-3D change is substantial. Tensor dimensions, neighborhood grouping,
position features, window manipulation, and decoder rearrangements all change.
Replacing one convolution class alone would not be a correct adaptation.

There is no current requirement to generate ensembles or alternative futures.
Rollout training changes how a deterministic forecaster is fitted; it does not
turn it into a probabilistic generative model.

## 2. The physical forecasting contract

Let $U^n$ denote the cached physical velocity field and define

$$
X^n_c=(U^n_c-\mu_c)/\sigma_c,\qquad
\tau=\Delta t/\Delta t_{\rm ref}.
$$

The planned Python model interface is

~~~python
prediction_normalized = model(input_normalized, relative_dt)
# input and output: [B, 3, 64, 64, 64]
# relative_dt: [B]
~~~

The forward calculation is

$$
\widehat X^{n+1}=X^n+G_\theta(X^n,\tau).
$$

The caller restores physical values through
$\widehat U^{n+1}_c=\sigma_c\widehat X^{n+1}_c+\mu_c$.
The forward method includes the addition; callers must not add the input twice.

Keep the original training-only means, standard deviations, and reference
interval. Reuse the same preprocessing rather than generating a new cache.
Our input remains a filtered DNS field, not a separate coarse Navier–Stokes
solution or a fine-grid reconstruction task.

An embedding of $\Delta t$ is not a numerical time integrator. Initially, all
training intervals remain the same saved-time interval. No claim about arbitrary
lead times, zero-step identity after training, or CFL stability follows.

No pressure solve, divergence projection, or spectral penalty is added to the
first experiment. This keeps an architectural comparison interpretable.
Physical diagnostics will still be measured.

## 3. Proposed defaults and tensor shapes

The starter model will have two spatial coarsenings, a bottleneck, and two
matching refinements. This topology is an explicit local design decision;
the exact stage interpretation should be revisited if author code appears.

| Setting | Proposed value | Reason |
|---|---:|---|
| Physical grid | $64^3$ | Match existing cache |
| Physical channels | 3 | u, v, w |
| Patch side | 4 | Finer token grid than the original baseline |
| Base feature width | 96 | Initial 3D resource budget |
| Stage widths | 96, 192, 384 | More features at lower spatial resolution |
| Encoder blocks | 2 at each of the first two levels | One regular/shifted pair |
| Bottleneck blocks | 2 | Refine whole-domain coarse context |
| Decoder blocks | 2 at each returning level | Refine after skip fusion |
| Heads by resolution | 3, 6, 12 | 32 features per head |
| Window side | 4 tokens | 64 tokens per 3D attention window |
| Shift | Half a window, where multiple windows exist | Exchange information across window boundaries |
| MLP expansion | 4 | Initial feature-mixing capacity |
| Dropout / stochastic depth | 0 / 0 | Simplify baseline comparison and debugging |
| ConvNeXt-style skip kernel | $3^3$ | Proposed economical local 3D kernel |
| Initial training batch | 2, subject to measured memory | Match baseline if feasible |

These settings are **not the paper's claimed configuration**. We will calculate
parameter counts and measure memory after implementation; neither is inferred
from the paper's 2D model.

Use channel-last tokens internally, **[B, Nx, Ny, Nz, C]**, and channel-first
arrays for Conv3d. Convert explicitly at module boundaries. A consistent layout
reduces subtle errors where LayerNorm accidentally acts over a spatial axis.

| Stage | Token grid | Width | Token count |
|---|---:|---:|---:|
| Patch embedding | $16^3$ | 96 | 4096 |
| Encoder level 0 | $16^3$ | 96 | 4096 |
| Merge 0 / encoder level 1 | $8^3$ | 192 | 512 |
| Merge 1 / bottleneck | $4^3$ | 384 | 64 |
| Expand 1 / decoder level 1 | $8^3$ | 192 | 512 |
| Expand 0 / decoder level 0 | $16^3$ | 96 | 4096 |
| Unpatch to feature volume | $64^3$ | 96 | Not an attention token grid |
| Final convolution | $64^3$ | 3 | Physical-grid increment |

~~~mermaid
flowchart TD
    A["Normalized velocity: B × 3 × 64 × 64 × 64"] --> P["Patch embedding and time conditioning"]
    P --> E0["Encoder level 0: 16³ tokens, width 96"]
    E0 --> S0["ConvNeXt-style skip 0"]
    S0 --> M0["Merge eight neighbors: 8³ tokens, width 192"]
    M0 --> E1["Encoder level 1"]
    E1 --> S1["ConvNeXt-style skip 1"]
    S1 --> M1["Merge: 4³ tokens, width 384"]
    M1 --> B["Bottleneck"]
    B --> U1["Expand to 8³, width 192"]
    S1 --> F1["Add matching skip"]
    U1 --> F1
    F1 --> D1["Decoder level 1"]
    D1 --> U0["Expand to 16³, width 96"]
    S0 --> F0["Add matching skip"]
    U0 --> F0
    F0 --> D0["Decoder level 0"]
    D0 --> H["Unpatch features; periodic convolution to 3 channels"]
    H --> R["Add normalized input"]
    A --> R
    R --> O["Next normalized velocity field"]
~~~

The exact block ordering in this diagram is our proposed specification.
Encoder skips carry the post-ConvNeXt features; those features also feed the
next merge. Decoder refinement follows expansion and skip addition. Record any
later change rather than silently treating multiple layouts as equivalent.

## 4. Patch embedding and conditioning

A patch groups $4^3$ voxels and three components: 192 physical numbers.
A shared Conv3d with kernel and stride 4 maps them to 96 features.
LayerNorm then acts over the 96 features at each patch location.

A token is a learned feature vector, not a prescribed eddy, finite volume,
or Fourier mode. The learned projection is not an additional physical filter.
The original physical spectrum was already defined by cache preprocessing.

The timestep branch will map $\tau$ through an affine layer and LayerNorm to
96 features, then broadcast and add the result over the token grid.
We propose conditioning at the input only; the hierarchy propagates it afterward.

There is only one dataset modality in the initial comparison. A learned
constant source vector supplies no distinction between examples, so we propose
omitting the source branch rather than inventing eight categories.
If multiple sources are later introduced, add an explicit source vocabulary,
save it in checkpoints, and condition on verified metadata.

Use relative spatial information inside attention rather than the baseline's
learned absolute position table. This changes the positional inductive bias
but does not establish exact translation or rotation equivariance.

## 5. Window attention and periodicity

### Local windows

At the finest token grid, partition $16^3$ tokens into 64 windows of $4^3$
tokens. Attention within a window mixes its 64 tokens.

The next block shifts the grouping by two tokens along each spatial direction.
Tokens separated by a previous window edge can then communicate. This is an
indexing operation, not advection or translation of the physical field.

At the bottleneck, $4^3$ tokens fit in one window. Disable the shift there
because there is no neighboring window grouping to exchange.

The hierarchy expands physical context: a bottleneck token summarizes a larger
region than a fine-level token. The bottleneck's full-domain window enables
global information exchange through a compressed representation.

### Periodicity is a separate design choice

Standard finite-image shifted-window masking is not automatically a periodic
boundary condition. For our proposed periodic mode:

- Circularly shift the token grid and partition it into windows.
- Preserve wraparound interactions across true opposite faces.
- Do not apply a finite-image seam mask that would remove those neighbors.
- Require exact divisibility for the first implementation, avoiding padding.
- Use circular padding in spatial convolutions.
- Use a consistent relative-coordinate convention at wrapped boundaries.

For windows smaller than the domain, use the unwrapped local window coordinates:
a window crossing a periodic face still represents adjacent locations.
For a full-domain window, use minimum-image displacements with a fixed tie
convention for half-domain separations. This is an additional periodic adaptation
to document and test.

Keep a finite-domain masked reference mode for comparison against the upstream
2D building block on synthetic cases. It is a debugging reference, not the
intended physical boundary mode.

A periodic implementation can still depend on the chosen patch/window origin.
Do not assert exact equivariance under arbitrary grid translations. Test
boundary connectivity independently of that stronger property.

## 6. SwinV2 block requirements

The official Microsoft implementation provides the reference for the meaning
of **SwinV2**: scaled cosine attention, continuous relative-position bias from
transformed offsets, and residual post-normalization. The residual branches are
normalized after attention/MLP, before addition. These differ from the
pre-normalized dot-product blocks in our existing model.
[Official SwinV2 source](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py)

Our port checklist is:

1. Preserve those block semantics while changing offset coordinates to 3D.
2. Check Q/K normalization, trainable scale bounds, bias construction, and
   initialization against a pinned upstream revision.
3. Implement partition/reversal separately from the attention calculation.
4. Make normalization axes explicit after every layout conversion.
5. Validate a small 2D-equivalent case against the reference before using 3D.
6. Record any deliberate simplification as a variant, not as exact SwinV2.

For a DNS reader, normalization controls internal feature magnitudes; it is
unrelated to conserving physical energy. A relative-position bias says which
token separations affect matching; it is not a physical derivative stencil.

Avoid claiming a fast attention backend before verifying it supports the chosen
cosine scaling, relative biases, and masks correctly. A clear reference
implementation should precede optimization.

## 7. Encoder, decoder, and skip connections

### Patch merging

Our 3D merge groups eight neighboring tokens in a $2\times2\times2$ block.
Concatenate their features, then use a learned reduction:

$$
[B,n,n,n,C]
\longrightarrow
[B,n/2,n/2,n/2,8C]
\longrightarrow
[B,n/2,n/2,n/2,2C].
$$

This is learned feature coarsening, not conservative restriction of velocity.
The reduced grid lets later attention combine information over larger
physical regions at lower cost.

### ConvNeXt-style skips

A proposed skip module applies a depthwise $3^3$ periodic convolution,
feature normalization, pointwise expansion, a nonlinear activation, and
pointwise contraction, with a residual connection.

“Depthwise” means each feature channel is spatially convolved separately.
The pointwise layers then mix channels. This combines local spatial processing
with feature mixing at relatively modest cost.

The precise kernel, expansion, normalization, and residual scaling will be
written into the configuration. We will not present these provisional choices
as recovered paper hyperparameters.

The skip stores features before information is compressed further. It gives
the decoder access to fine-level structure without requiring the bottleneck
to preserve every detail.

### Patch expansion

To go from width $2C$ on a coarse grid to width $C$ on a grid twice as large
in each direction:

$$
[B,n,n,n,2C]
\longrightarrow
[B,n,n,n,8C]
\longrightarrow
[B,2n,2n,2n,C].
$$

The first map is learned; the second is rearrangement. In 3D, doubling each
axis creates eight children, so a 2D four-child expansion cannot be reused
unchanged.

The expanded features are added to the corresponding skip at the same shape,
then refined by decoder blocks. Addition requires equal feature widths;
any projection used to align them must be explicit.

An encoder–decoder skip is different from the outer velocity residual.
The former preserves learned features; the latter carries the observed field
directly into the forecast.

## 8. Output reconstruction and initialization

The proposed decoder returns a $16^3$ grid with width 96.
Project each token to $96\times4^3$ features and rearrange into a
96-channel $64^3$ feature volume. A periodic $3^3$ convolution then produces
three increment channels.

This final convolution can exchange information across reconstructed patch
edges. It still does not impose continuity, conservation, or incompressibility
as exact constraints.

For the first controlled comparison, initialize only the final increment
convolution to zero, yielding persistence initially. This is our baseline
initialization choice, not a verified detail of the authors' implementation.

As in the original model, a zero final map blocks upstream loss gradients
during the first update. Verify gradient flow after the output map becomes
nonzero. Keep small random output initialization as a later ablation if needed.

The full-resolution feature volume is a possible memory bottleneck:
$96\times64^3$ float32 values occupy about 96 MiB per example, before gradients
or other activations. Do not estimate memory from window attention alone.

## 9. Training regimes

Implement and compare single-step and full-gradient rollout configurations. Each forward call
still takes one field; unrolling does not introduce multi-frame attention.

### A. Single-step baseline

Start from a reference $X^n$ and minimize

$$
L_1=\operatorname{mean}[(F_\theta(X^n,\tau)-X^{n+1})^2].
$$

Initially reuse the existing optimizer settings, 100 epochs, and batch size 2
where memory allows. This is the first scientific comparison because it changes
architecture without simultaneously changing the training objective.

### B. Full-gradient five-step training

Starting from reference $X^n$, recursively predict five successors:

$$
\widehat X^{n+k}=F_\theta(\widehat X^{n+k-1},\tau),
\qquad
L_5=\frac15\sum_{k=1}^5
\operatorname{mean}[(\widehat X^{n+k}-X^{n+k})^2].
$$

Set $\widehat X^n=X^n$. Preserve the computation graph through all five updates.
Do not replace intermediate predictions with true frames; that would be
teacher forcing rather than this rollout objective.

Store states in the normalized convention throughout the recursion. Do not
normalize an already-normalized prediction again.

### Comparison fairness

Five-step windows create fewer examples per epoch and require more forward work
per example. Report epochs, optimizer updates, predicted transitions, runtime,
and memory. Equal epochs do not imply equal computational budgets.

The current comparison trains both architectures from scratch with the same
five-step objective and equal optimizer-update budgets. It does not warm-start
from one-step checkpoints.

## 10. Data reuse and temporal windows

Reuse the existing file membership and training-only normalization:

| Split | Files | One-step pairs | Five-step windows |
|---|---|---:|---:|
| Train | 131–150 | 180 | 100 |
| Validation | 152–155 | 36 | 20 |
| Test | 157–160 | 36 | 20 |

A ten-snapshot file provides $10-H$ windows of horizon $H$. A five-step window
contains six snapshots: one input and five successors. Permit starts 0–4.

Keep windows inside a single file for the first implementation. Neighboring
files share endpoint timestamps, and timestamp equality alone does not verify
identical endpoint values. No new cross-file stitching policy is needed for
the first five-step experiment.

Reuse **CacheIndex**, **training_statistics**, and **PairDataset** where their
contracts fit. Add a separate sequence dataset returning input, ordered targets,
and ordered relative intervals. Do not silently change the old pair dataset.

The current baseline configuration parser accepts an exact schema. A new model
needs its own parser; passing extra hierarchy keys into the old parser will fail.
Provide the existing cache loader with only the metadata fields it actually
requires, including model grid size and the data configuration.

Longer physical rollouts will require either verified file stitching or longer
contiguous reference sequences. Do not concatenate files by name and assume
continuity.

## 11. Evaluation and comparison

### Preserve the existing metrics

Use the same one-step normalized MSE, physical MSE, relative L2, persistence
skill, and kinetic energy calculations. Evaluate both models on the same
pairs, using the same saved normalization.

Add MAE as a supplementary metric with an explicit physical-value definition.
Do not replace the existing checkpoint-selection metric during the first
architectural comparison.

For rollout-trained models, choose and record a validation criterion before
running: proposed criterion is mean normalized MSE across all five horizons,
even if the training loss uses only the final horizon.

### Rollout evaluation

For each common starting window:

1. Supply the true initial field once.
2. Run both models recursively without later reference inputs.
3. Record metrics independently for horizons 1–5.
4. Compare with persistence using the initial field unchanged at every horizon.

Use the same 20 validation starts for all five-step comparisons. The original
36-pair one-step score is a separate measurement, not directly the same sample
population as horizon-one scores on those 20 starts.

Report aggregate errors as well as variation across starting states. Neighboring
windows are correlated; avoid treating them as independent experimental repeats.

### Physical diagnostics

Add diagnostics after the basic forecast implementation is verified:

- Resolved energy spectrum, with FFT normalization and shell counts documented.
- Volume-mean energy evolution.
- Divergence relative to a documented reference gradient scale.
- Velocity-increment distributions and temporal correlations.

For dimensional derivatives, verify box lengths/grid spacing from source
metadata first; the cache currently does not provide these to the baseline
plotter. Do not assume a $2\pi$ box without evidence.

Compare with equivalently filtered reference data, and account for its cubic
Fourier support when interpreting high-wavenumber shells. Fine-grid dissipation
statistics are not direct targets of a $64^3$ forecast.

Initially these are diagnostics, not extra loss terms. Spectral or divergence
penalties would constitute separate modeling experiments.

### Visual output

Reuse the baseline's physical-unit slice plotting where possible. Extend the
new evaluator's comparison to show input, target, prediction, predicted
increment, and error, with labeled scales. Add consistent-time rollout panels.

Plots should name the checkpoint, file, start time, lead time, component, and
slice. A spatially convincing image must not substitute for persistence skill
or physical diagnostics.

## 12. Implementation milestones

| Phase | Work | Completion evidence |
|---|---|---|
| 0: source and specification | Confirm source provenance, freeze adapted topology and defaults | Source audit and config schema agree; inherited versus new choices labeled |
| 1: spatial primitives | Window partition/reversal, shifts, 3D merge/expand, periodic convolutions | Ordering and boundary tests on labeled small tensors |
| 2: attention blocks | SwinV2 semantics, 3D relative position handling | Small reference comparisons; finite outputs/gradients |
| 3: complete model | Hierarchy, skips, conditioning, head, residual | Shape checks, parameter count, persistence initialization, gradient checks |
| 4: one-step pipeline | Config, shared cache adapter, trainer, checkpoint evaluator | Tiny synthetic integration run; real cache metadata validation |
| 5: GPU smoke test | Small allocated run and four-pair overfit diagnostic | Measured memory/time and substantial train-loss reduction |
| 6: first baseline | Full training and validation comparison | Saved configuration, checkpoint, metrics, physical-unit plots |
| 7: temporal objective | Shared sequence dataset and full-gradient rollout | Target alignment and full-gradient tests; common-start evaluation |
| 8: physical assessment | Spectra, divergence, temporal diagnostics | Analytical-field verification and reference comparisons |

GPU smoke tests and full experiments are future execution work. No job is
submitted by creating the implementation or job script.

### Tests that target likely errors

Use small synthetic tensors before any production computation.

- Partition then reverse must reproduce an indexed volume exactly.
- Shift then inverse shift must restore both values and positions.
- Verify 3D child ordering independently of learned projection coefficients.
- Test wraparound connectivity across x, y, and z faces.
- Compare a tiny attention calculation with an explicit reference.
- Check the bottleneck single-window/no-shift case.
- Reject incompatible grid, patch, window, and head dimensions.
- Confirm the whole model preserves input/output shapes and initial persistence.
- Confirm upstream gradients after moving the final map away from zero.
- Compare five-step targets with explicitly indexed source snapshots.
- Confirm full rollout gradients reach early steps.
- Use a toy evolution with a known exact update to test recursive evaluation.
- Save/reload a tiny checkpoint and reproduce the same prediction.
- Verify plots convert normalized outputs back to physical values exactly once.

These tests check numerical contracts; they are not evidence that turbulent
dynamics have been learned.

## 13. Code organization

The complete architecture, including window operations, attention, and blocks,
is implemented in `yalamanchi_vit/model.py`, matching the plain baseline layout.
Model-specific configuration remains in this package.
`train.py` and `evaluate.py` are thin entry points into the shared pipeline.

The `forecasting` package owns sequence windows, full-gradient rollout loss,
training, checkpoint handling, per-lead evaluation, and the matched comparison
runner. Both architectures use these same implementations for rollout experiments.
Existing `baseline_vit` helpers supply cache validation, training statistics,
one-step metrics, and physical-velocity slice plotting.

See the [comparison guide](../../forecasting/README.md) for current files and
commands. GPU experiments remain separate from implementation verification.

## 14. Compute and reproducibility

Local attention reduces token-pair work, but 3D activations, decoder features,
and five-step backpropagation can still be expensive.

At the finest proposed level, 4096 tokens each interact with 64 window tokens:
262,144 pairs per head. This happens to match the old baseline's $512^2$
global pair count, but it does **not** imply equal cost: widths, number of blocks,
other levels, convolutions, and activation storage differ.

Start with one GPU inside an allocation. Measure peak allocated memory and
time per update before choosing a production walltime. Use microbatch reduction,
gradient accumulation, or activation checkpointing only after profiling and
record the change.

Do not infer a new job duration from the old six-minute run. Multiple GPUs
require an explicit distributed-training implementation; requesting them alone
does not distribute this model.

Use separate run directories, fixed split membership, and saved normalization.
Record seed and software versions. Keep initial precision consistent with
the float32 baseline; introduce mixed precision as a measured later change.

Exact parameter matching is not guaranteed by the proposed width 96.
Publish actual parameter counts and, if necessary, run a width-adjusted control.
A larger architecture outperforming a smaller one does not isolate which
architectural component caused the improvement.

## 15. Open decisions and acceptance criteria

Before calling this implementation complete, resolve:

| Question | Current decision or required evidence |
|---|---|
| Is author code available? | Not located in inspected public sources; recheck if a repository is supplied |
| Exact original stage layout? | Our two-merge/two-expand layout is explicit; not claimed exact |
| Is the proposed width affordable? | Requires measured GPU memory and runtime |
| Are periodic relative offsets correct? | Requires synthetic seam tests and documented tie convention |
| Which optimizer schedule? | Start with the existing constant-rate protocol; tuning uses validation only |
| What defines best rollout checkpoint? | Mean five-horizon validation normalized MSE, saved in config |
| Does history enter one forward call? | No; adding it is a separate future experiment |
| Is the model physically constrained? | No exact projection/conservation constraint in the first baseline |
| Is it better than the plain ViT? | Must be established by common-data forecast and physical metrics |

A scientifically usable first version has a tested forward map, a complete
single-step training/evaluation path, documented source provenance, no silent
data-split changes, and reproducible artifacts. It need not beat the plain ViT
to be a valid baseline; a negative result should be reported as such.

Five-step objectives are a second milestone, with common-start comparisons
and verified detachment semantics. Claims about long-horizon turbulence require
a longer evaluation than five saved intervals.

## 16. Glossary

| Term | Meaning in this plan |
|---|---|
| Encoder | Builds coarser, wider feature representations |
| Decoder | Restores spatial resolution from those representations |
| Bottleneck | Coarsest representation between encoder and decoder |
| Window | Local group of tokens sharing an attention operation |
| Shifted window | Different grouping obtained by shifting partition origin |
| Patch merging | Learned reduction of neighboring token features |
| Patch expansion | Learned generation and rearrangement of child features |
| Skip connection | Route carrying features around compression stages |
| Depthwise convolution | Separate spatial convolution for each feature channel |
| Relative-position bias | Learned adjustment based on token separation |
| Teacher forcing | Supplying reference states instead of generated intermediate states |
| Full-gradient rollout | Backpropagating through the recursive forecast chain |
| Pushforward | Training a terminal step on detached model-generated inputs |
| Ablation | Controlled experiment changing one design choice |
| Provenance | Record of data, code, and configuration used for a result |
