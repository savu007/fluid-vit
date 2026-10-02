# A DNS practitioner's guide to the baseline 3D velocity ViT

For the verified current loss, physical-data assumptions, and open design decisions,
see the shared [loss-design reference](../../forecasting/docs/loss-design-for-dns.md).

This guide explains the implementation in this repository as of September 25,
2026. It assumes familiarity with velocity fields, spectral filtering, time
integration, and turbulence diagnostics. Machine-learning terms are introduced
as they appear. Numerical examples use the defaults in
[config.yaml](../config.yaml); changing those settings changes the dimensions.

The current objective is **deterministic, one-step forecasting of a coarse
velocity field**. The network returns one estimate of the next field. It does
not sample alternative futures. Forecast accuracy and physical fidelity are
complementary requirements: a useful predictor should improve on persistence
without destroying the resolved flow's physical structure.

## Reading map

1. [The physical prediction problem](#1-the-physical-prediction-problem)
2. [Where the input fields come from](#2-where-the-input-fields-come-from)
3. [Notation and the complete data path](#3-notation-and-the-complete-data-path)
4. [Normalization of physical velocities](#4-normalization-of-physical-velocities)
5. [Patch embedding](#5-patch-embedding)
6. [Position and timestep embeddings](#6-position-and-timestep-embeddings)
7. [The six transformer blocks](#7-the-six-transformer-blocks)
8. [The output head and reconstruction](#8-the-output-head-and-reconstruction)
9. [Residual forecasting and initialization](#9-residual-forecasting-and-initialization)
10. [How training changes the model](#10-how-training-changes-the-model)
11. [Dataset construction and separation](#11-dataset-construction-and-separation)
12. [Metrics and the validation image](#12-metrics-and-the-validation-image)
13. [Physical capabilities and limitations](#13-physical-capabilities-and-limitations)
14. [Configuration choices and computational cost](#14-configuration-choices-and-computational-cost)
15. [Code map and saved artifacts](#15-code-map-and-saved-artifacts)
16. [Glossary](#16-glossary)

## 1. The physical prediction problem

Let $\mathbf U^n=(u^n,v^n,w^n)$ be the **coarse**, three-component velocity
field at one saved time. The target is the next saved coarse field,
$\mathbf U^{n+1}$, separated by the dataset interval $\Delta t$.

The model's operation in physical values is

$$
\widehat{\mathbf U}^{n+1}
= \mathbf U^n
+ \boldsymbol{\sigma}\odot
G_\theta\left(
\frac{\mathbf U^n-\boldsymbol{\mu}}{\boldsymbol{\sigma}},
\frac{\Delta t}{\Delta t_{\rm ref}}
\right).
$$

Here:

- $G_\theta$ is the network branch predicting a normalized velocity increment.
- $\theta$ denotes all adjustable weights and biases.
- $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ contain one fixed mean and
  standard deviation per velocity component.
- $\odot$ means componentwise multiplication, broadcast across spatial points.
- $\Delta t_{\rm ref}$ is the median saved-time interval in the training files.
- A hat indicates a prediction rather than a stored reference field.

The Python object **BaselineViT** includes both $G_\theta$ and the addition
of the input. Its **forward** method returns the **next normalized field**, not
just the increment. Do not add the input a second time when calling it.

This resembles a time-integrator update, but its interpretation differs.
A DNS solver evaluates a discretized governing equation, pressure constraint,
and boundary conditions. The network learns a map from examples of input and
successor fields. It does not explicitly calculate convection, viscosity,
forcing, or a pressure projection.

The target interval is the interval between **saved snapshots**. It need not
equal the DNS solver's internal step. Do not infer seconds merely from the
variable name **dt**; physical time units must come from dataset metadata.

The output has the same grid and components as the input: three fields on
$64^3$ points. There is no pressure output or reconstruction onto the
original $384^3$ grid.

## 2. Where the input fields come from

Preprocessing is implemented in
[coarsening/coarsen.py](../../coarsening/coarsen.py), using
[coarsening/config.yaml](../../coarsening/config.yaml):

$$
\text{DNS velocity at }384^3
\longrightarrow
\text{sharp Fourier filtering and resampling}
\longrightarrow
\text{cached velocity at }64^3.
$$

Retained integer Fourier modes satisfy

$$
|k_x|,\ |k_y|,\ |k_z| \leq 31.
$$

This is a componentwise cubic cutoff, not a spherical cutoff in
$|\mathbf k|$. Coarse Nyquist planes are zero. FFT normalization is adjusted
when reconstructing on the smaller grid so retained amplitudes have the intended
scale. The operation uses float64; cached arrays use float32.

Each configured file contains ten snapshots, stored with shape
**[10, 3, 64, 64, 64]**. A manifest records file membership, channels, grids,
timestamps, and preprocessing metadata. Training reads these arrays rather
than repeatedly reopening the original NetCDF fields.

Two consequences are central for a DNS practitioner.

**The target is filtered DNS.** It is not automatically the result of solving
the Navier–Stokes equations independently on a $64^3$ mesh. Filtering nonlinear
dynamics introduces unresolved-scale effects. The network can absorb their
average influence into its learned update, but it has no explicit subgrid-stress
model or resolved-equation residual.

**Filtering and patch embedding serve different purposes.** Filtering defines
the physical scales included in the prediction problem. Patch embedding is a
learned representation inside the model. Smaller patches cannot recover
Fourier modes removed by preprocessing.

## 3. Notation and the complete data path

A **tensor** is a multidimensional array. A **batch** groups independent examples
processed together; it is not a temporal history supplied to the network.

| Symbol | Meaning | Default |
|---|---|---:|
| $B$ | Examples in a batch | 2 |
| $C$ | Physical velocity channels | 3 |
| $N$ | Grid points per spatial direction | 64 |
| $P$ | Patch side length | 8 |
| $L=(N/P)^3$ | Number of spatial tokens | 512 |
| $D$ | Features in each token | 192 |
| $H$ | Attention heads per block | 6 |
| $d_h=D/H$ | Features per head | 32 |

Field arrays are channel-first: **[B, C, X, Y, Z]**. After embedding, their
representation is **[B, L, D]**. The sequence axis $L$ enumerates spatial
patches in **one frame**, not time.

~~~mermaid
flowchart TD
    A["Physical velocity U(n): B × 3 × 64 × 64 × 64"] --> B["Fixed channel normalization"]
    B --> C["Normalized input X"]
    C --> D["Conv3d patch embedding: B × 192 × 8 × 8 × 8"]
    D --> E["Flatten spatial axes: B × 512 × 192"]
    P["Learned position vectors: 1 × 512 × 192"] --> F["Add position and timestep features"]
    T["dt / reference dt → Linear → LayerNorm"] --> F
    E --> F
    F --> G["Six transformer blocks"]
    G --> H["Final LayerNorm"]
    H --> I["Linear output head: B × 512 × 1536"]
    I --> J["Unpatchify: normalized increment"]
    C --> K["Add normalized input"]
    J --> K
    K --> L["Undo fixed normalization"]
    L --> M["Predicted physical velocity U(n+1)"]
~~~

| Operation | Output shape |
|---|---|
| Input or normalized input | [B, 3, 64, 64, 64] |
| Patch convolution | [B, 192, 8, 8, 8] |
| Flatten and transpose | [B, 512, 192] |
| Add embeddings | [B, 512, 192] |
| Each transformer block | [B, 512, 192] |
| Final LayerNorm | [B, 512, 192] |
| Linear output head | [B, 512, 1536] |
| Reassemble patches | [B, 3, 64, 64, 64] |
| Add normalized input | [B, 3, 64, 64, 64] |

The 1536 outputs per token equal $3\times8^3$: all components at every point
of that patch. With batch size 2, the input contains 1,572,864 scalar velocities.
The token representation contains 196,608 feature values. These are array sizes,
not counts of trainable parameters.

## 4. Normalization of physical velocities

Before entering the network, each component is transformed by

$$
X_c=\frac{U_c-\mu_c}{\sigma_c}.
$$

Statistics use **every training snapshot**, all spatial points, and float64
accumulation. The standard deviation is based on population variance. Once
computed, these statistics stay frozen during training and evaluation.

There are only three means and three standard deviations, not separate values
for every snapshot or voxel. A frame with unusually high energy therefore
remains unusually energetic relative to other frames after normalization.

Validation and test fields do not enter these statistics. Otherwise the training
pipeline would use information reserved for assessment. A standard deviation
below $10^{-12}$ is replaced by 1 to avoid near-zero division.

After prediction:

$$
\widehat U_c=\sigma_c\widehat X_c+\mu_c.
$$

“Physical values” means restoration of the dataset's original scale. If the DNS
data were nondimensional, restored values remain nondimensional.

Fixed physical normalization differs from **LayerNorm**, which operates on
internal features and has no direct velocity-unit interpretation.

## 5. Patch embedding

A patch is a non-overlapping $8\times8\times8$ block. With three velocity
components, each patch contains 1536 scalars.

The input layer is **Conv3d(3, 192, kernel_size=8, stride=8)**. It generates
192 features per patch. Because stride equals kernel size, patches do not
overlap. There are $8^3=512$ patches.

For patch $j$, this is an affine map:

$$
z_j=W_{\rm patch}\operatorname{vec}(X_j)+b_{\rm patch},
\qquad W_{\rm patch}\in\mathbb R^{192\times1536}.
$$

Here, “vec” arranges patch values into a consistent vector. The convolution is
an efficient way to apply the same matrix and bias at all patch locations.

### What a token contains

A **token** is the resulting 192-number feature vector. Its entries are learned
combinations of velocity components and positions within a patch. They are not
prescribed vorticity, strain, or Fourier coefficients. Learned features may
respond to those structures without each feature having a physical label.

This is not block averaging. Weights distinguish the positions inside a patch
and can represent local differences and patterns.

However, mapping 1536 numbers into 192 is a real bottleneck: arbitrary patches
cannot be uniquely reconstructed from this linear embedding. Training must
select features useful for predicting the increment.

The original field bypasses this compression through the outer residual
connection. Thus compression limits the information available to the learned
**correction**, while the unchanged part of the field passes through directly.

### Why flatten the patch grid?

The convolution produces **[B, 192, 8, 8, 8]**. Flattening the spatial axes and
transposing produces **[B, 512, 192]**.

Flattening changes indexing, not numerical resolution. Token ordering follows
x, y, z, with z varying fastest in the standard contiguous layout.
Reconstruction reverses this arrangement.

## 6. Position and timestep embeddings

### Position

Attention by itself does not identify absolute patch locations. The model adds
one learned vector $p_j\in\mathbb R^{192}$ at each token position:

$$
z_j\leftarrow z_j+p_j.
$$

These 512 vectors are initialized with small random values and optimized during
training. They are not explicitly physical coordinates or Fourier features.

Their array shape is **[1, 512, 192]**: the same position table is used for
all batch members.

For periodic homogeneous turbulence this choice deserves attention. Absolute
position vectors and fixed patch boundaries do not guarantee translation
equivariance. A one-grid-point shift of the input need not produce exactly the
same shift of the output. Periodic training data does not impose that symmetry
on the architecture.

### Timestep

For each example, the dataset supplies $\tau=\Delta t/\Delta t_{\rm ref}$.
A linear layer and LayerNorm map this scalar to 192 features:

$$
e_t=\operatorname{LN}(W_t\tau+b_t).
$$

The same vector is added to every token in that example. The transformer
therefore receives $z_j+p_j+e_t$.

This is a timestep embedding, not a history encoder. The model sees one
observed velocity field; there is no temporal sequence of past fields and no
embedding of absolute simulation time.

The loader currently requires approximately constant saved-time spacing.
Consequently, $\tau$ is approximately 1 throughout this experiment.
Time conditioning exists as an interface, but training at one interval does
not demonstrate accuracy at other intervals.

There is no explicit multiplication of the increment by $\Delta t$, nor a
constraint forcing zero increment at $\Delta t=0$. Treat this model as a
learned fixed-interval update, not an adaptive time integrator.

## 7. The six transformer blocks

All six blocks preserve shape **[B, 512, 192]**. Each block has its own
parameters; weights are not shared across the six blocks.

A block performs

$$
Z'=Z+\operatorname{Attention}(\operatorname{LN}_1(Z)),
$$

$$
Z_{\rm out}=Z'+\operatorname{MLP}(\operatorname{LN}_2(Z')).
$$

Attention exchanges information between patches. The MLP transforms each
patch's feature vector. Each branch adds its result to its input.

### 7.1 LayerNorm

For a token with $D$ features, LayerNorm calculates

$$
\operatorname{LN}(z)_a=
\gamma_a
\frac{z_a-\overline z}
{\sqrt{D^{-1}\sum_b(z_b-\overline z)^2+\epsilon}}
+\beta_a.
$$

Mean and variance are calculated across that token's features. Learned scale
and offset parameters, $\gamma$ and $\beta$, allow adjustment afterward.

This is not spatial averaging, normalization across batch examples, or an
energy constraint. Its purpose is to control internal feature scales during
optimization. It has no running training-set statistics.

Normalization precedes each learned branch, hence the name
**pre-normalization transformer block**.

### 7.2 Queries, keys, and values

An affine layer generates three feature vectors per token:

$$
Q=ZW_Q+b_Q,\qquad K=ZW_K+b_K,\qquad V=ZW_V+b_V.
$$

The code computes these jointly using **Linear(192, 576)** and then separates
them.

A query describes what the receiving token seeks; a key describes how another
token matches that request; a value carries information to be mixed into the
output. All are learned features. An attention **value** vector is not the
physical velocity vector.

### 7.3 Six attention heads

The 192 features are split into six heads of width 32. Within one head:

$$
A^{(h)}_{ij}
=
\frac{\exp(q^{(h)}_i\cdot k^{(h)}_j/\sqrt{32})}
{\sum_{\ell=1}^{512}\exp(q^{(h)}_i\cdot k^{(h)}_\ell/\sqrt{32})},
\qquad
o^{(h)}_i=\sum_{j=1}^{512}A^{(h)}_{ij}v^{(h)}_j.
$$

Softmax makes each row of attention weights sum to one. The square-root scale
controls dot-product magnitudes.

Different heads can learn different patterns of information exchange. They
are not assigned to particular components, directions, or spectral bands.

The six outputs are concatenated to width 192 and passed through another
learned **Linear(192, 192)** projection.

### 7.4 Global spatial attention

Every token can attend to every other token, including itself. There is no
restricted window or explicit nearest-neighbor stencil. One attention layer
can communicate across the whole domain.

The implementation calls PyTorch's **scaled_dot_product_attention** with
**is_causal=False**. This does not leak future data: the token sequence contains
only spatial patches of the observed frame. A causal mask would impose an
arbitrary order across spatial patches.

Q, K, and V have shape **[B, 6, 512, 32]**. Conceptually, attention weights have
shape **[B, 6, 512, 512]**. An optimized backend can avoid explicitly storing
this entire matrix; the source does not force a particular GPU kernel.

A connection between distant patches is not a pressure Poisson solve.
It permits learned nonlocal dependence without enforcing incompressibility
or a particular physical operator.

### 7.5 The MLP

MLP means **multilayer perceptron**. Here it is

$$
m(z)=W_2\operatorname{GELU}(W_1z+b_1)+b_2,
$$

with widths $192\rightarrow768\rightarrow192$. The factor of four is
**mlp_ratio**. The wider intermediate representation provides room to combine
features before returning to the standard width.

GELU is a smooth nonlinear activation, commonly expressed as $a\Phi(a)$,
where $\Phi$ is the standard normal cumulative distribution function.
Nonlinearity allows compositions of layers to represent more than an affine map.

The same MLP is applied independently at each token. It does not directly mix
token locations; attention supplies that communication.

Dropout can randomly suppress features during training as regularization.
The current probability is **0.0**, so both MLP dropout and attention dropout
are inactive.

### 7.6 Internal residual connections

The additions $Z+\cdots$ and $Z'+\cdots$ let branches learn corrections
to existing representations. They also provide direct paths for gradients
through a deeper model.

These feature-space residuals are separate from the final forecasting residual
$\mathbf U^n+\Delta\mathbf U$. None of the additions inherently conserves
momentum or kinetic energy.

## 8. The output head and reconstruction

A final LayerNorm follows the transformer stack. Then **Linear(192, 1536)**
maps each token to a patch of normalized velocity increments:

$$
d_j=W_{\rm head}\operatorname{LN}(z_j)+b_{\rm head}.
$$

The word **head** means the final output mapping here, distinct from an
attention head. Its weights are shared across spatial patch locations.

Each patch increment lies in the affine space spanned by the 192 columns of
the output matrix and its bias. These learned basis patterns need not be smooth,
but arbitrary 1536-dimensional increments cannot be independently represented.

### Unpatchifying

The **unpatchify** method rearranges **[B, 512, 1536]** into
**[B, 3, 64, 64, 64]**. It first interprets the values as

**[B, patch_x, patch_y, patch_z, channel, local_x, local_y, local_z]**

and then interleaves patch and within-patch coordinates.

This is reshaping and permutation, not interpolation. It reverses the patch
**layout**, not the lossy input embedding.

The output resolves individual voxels even though attention operates on patch
tokens. There is no overlapping-patch averaging, patch-boundary continuity
constraint, or divergence projection after reconstruction.

## 9. Residual forecasting and initialization

The normalized prediction is

$$
\widehat X^{n+1}=X^n+\Delta X.
$$

Undoing normalization gives
$\widehat U_c^{n+1}=U_c^n+\sigma_c\Delta X_c$.
The mean cancels from the increment.

The output head's weights and bias are initialized to zero. Thus the initial
increment is zero regardless of upstream features, and the initial prediction
is exactly **persistence in normalized space**. Conversion to and from physical
values can introduce ordinary floating-point roundoff.

This is a useful short-interval starting point, but also explains why an
untrained model can produce a visually convincing flow field.

During the first backward pass, gradients reach the output head, but its zero
weight matrix blocks loss gradients from reaching earlier layers. Once the
head weights move away from zero, gradients reach the upstream network.
AdamW's separate weight decay can still affect parameters with zero loss
gradients.

The residual formulation does not force permanent copying. Whether learned
changes help must be assessed against persistence, not just by inspecting
target and prediction images.

## 10. How training changes the model

### 10.1 Parameters and hyperparameters

A **parameter** is a coefficient learned from examples: a weight, bias, position
embedding, or LayerNorm scale. A **hyperparameter** is an experiment setting
such as depth, patch size, or learning rate.

The default model contains:

| Group | Trainable scalars |
|---|---:|
| Patch embedding | 295,104 |
| Position embeddings | 98,304 |
| Timestep linear layer and LayerNorm | 768 |
| Six transformer blocks | 2,669,184 |
| Final LayerNorm | 384 |
| Output head | 296,448 |
| **Total** | **3,360,192** |

Each block has 444,864 parameters. Fixed physical normalization statistics
are not trainable parameters.

### 10.2 Loss

For a batch, the objective is

$$
\mathcal L=
\frac{1}{BCN^3}
\sum_{b,c,i,j,k}
\left[
\frac{\widehat U_{b,c,i,j,k}^{n+1}-U_{b,c,i,j,k}^{n+1}}{\sigma_c}
\right]^2.
$$

This is normalized mean squared error (MSE). Voxels have equal weight.
Physical errors in different components are weighted by $1/\sigma_c^2$.

Comparing next states is equivalent to comparing increments here: the common
input cancels. There is no separate spectral, divergence, gradient, or
conservation penalty.

MSE is a useful first forecast objective, but large-scale errors can dominate
and uncertain small structures can favor smooth conditional-average predictions.
A low loss does not alone establish turbulence fidelity.

### 10.3 One optimizer update

1. Load two input–target pairs and their relative timesteps.
2. Move them to the device and apply frozen physical normalization.
3. Calculate predictions and the loss.
4. Calculate loss derivatives with respect to model parameters.
5. Clip the global gradient norm to at most 1.0.
6. Update parameters with AdamW.

**Backpropagation** is the chain rule through the network. These gradients are
derivatives with respect to model coefficients, not spatial derivatives such
as $\partial u/\partial x$.

AdamW keeps moving estimates of gradient moments to scale parameter updates.
The configured learning rate is $10^{-4}$, with weight decay 0.01.
The learning rate is not the physical time increment.

Gradient clipping limits an optimization update; it does not clip velocities
or guarantee rollout stability. Weight decay regularizes coefficients; it is
not a viscosity term.

### 10.4 Epochs and validation

An epoch is one pass through all training pairs. With 180 pairs and batch
size 2, each epoch has 90 optimizer updates. The default 100 epochs perform
9,000 updates using the same finite dataset repeatedly.

Pairs are shuffled, but each pair retains its input-before-target ordering.
Shuffling does not change the physical temporal relationship inside an example.

After each epoch, validation runs with frozen weights and no gradient
calculation. The checkpoint with the lowest validation normalized MSE becomes
**best.pt**. The test split is evaluated separately after model choices are fixed.

Recorded training loss averages losses from weights that change during the
epoch. Validation uses end-of-epoch weights on different data. Validation MSE
being below training MSE is therefore not automatically a bug.

### 10.5 Precision and reproducibility

The trainer uses float32 tensors and no mixed-precision training wrapper.
Statistics and evaluation reductions use float64 where specified in the code.
Float32 tensor dtype does not alone prescribe every backend's internal
GPU arithmetic mode.

Python, NumPy, and PyTorch seeds are set, and the data shuffler uses a seeded
generator. Strict deterministic GPU algorithms are not requested, so bitwise
reproducibility across software stacks and hardware is not guaranteed.

## 11. Dataset construction and separation

| Role | File IDs | Pairs with ten snapshots per file |
|---|---|---:|
| Training | 131–150 | 180 |
| Gap | 151 | 0 |
| Validation | 152–155 | 36 |
| Gap | 156 | 0 |
| Test | 157–160 | 36 |

Within a file, ten snapshots form nine pairs:
$(0,1),(1,2),\ldots,(8,9)$. No pair crosses a file boundary.

The loader checks channel ordering, array shapes and dtype, finite increasing
timestamps, compatible time attributes, and approximately constant within-file
saved-time intervals.

Neighboring files can share endpoint timestamps but cannot overlap in the
interiors of their time ranges. Training, validation, and test ranges must
remain strictly separated.

Repeated endpoints within training remain in both files and count once per
file when computing normalization. There is no deduplication. Timestamp
equality alone does not establish equality of the velocity arrays.

A gap file helps separate the splits but does not prove statistical independence.
That depends on correlation times and the physical duration sampled.

Arrays are read through memory maps with a small cache, and individual examples
are copied into tensors. Metadata and headers for all selected splits are
checked up front, but test field values do not enter training losses or statistics.

## 12. Metrics and the validation image

Here let $X$ denote the physical input, $Y$ the physical target,
$e=\widehat U-Y$, and $e_{\rm pers}=X-Y$. Sums and means cover all examples,
components, and voxels in the evaluated split.

| Metric | Definition | Interpretation |
|---|---|---|
| normalized_mse | $\operatorname{mean}[(e_c/\sigma_c)^2]$ | Training-scale error; checkpoint selection |
| physical_mse | $\operatorname{mean}[e^2]$ | Original velocity units squared |
| relative_l2 | $\sqrt{\sum e^2/\sum Y^2}$ | Global relative error magnitude |
| persistence_mse | $\operatorname{mean}[e_{\rm pers}^2]$ | Copying-input error |
| persistence_relative_l2 | $\sqrt{\sum e_{\rm pers}^2/\sum Y^2}$ | Relative error of copying |
| mse_skill_vs_persistence | $1-\sum e^2/\sum e_{\rm pers}^2$ | 0 equals copying, positive improves on it, 1 is perfect |
| predicted_kinetic_energy | $\frac12\langle\hat u^2+\hat v^2+\hat w^2\rangle$ | Predicted energy averaged over voxels and examples |
| target_kinetic_energy | $\frac12\langle u^2+v^2+w^2\rangle$ | Corresponding reference energy |
| samples | Number of evaluated pairs | 36 for validation |

Kinetic energy is per unit mass, or per volume if density is unity. It includes
the mean flow: the code does not subtract a mean velocity before calculating it.

Relative L2 is a global norm ratio, not an average of per-snapshot ratios or a
percentage of correct voxels. Ratios with zero denominators are returned as
**null**. Evaluation reductions use float64; predictions use float32 tensors.

Each line of **metrics.jsonl** also records **epoch**, **training_mse**, and
**seconds**. Seconds includes that epoch's training and validation, but not
initial setup or the checkpoint write following the record.

### Reading velocity-val.png

This optional plot shows one selected pair, not a validation-set average.
Rows correspond to u, v, and w; columns contain:

1. True successor field.
2. Predicted successor field.
3. Signed prediction-minus-target error.

The default is pair index 0 on the central z plane, index 32. For a z slice,
horizontal is x and vertical is y. Coordinates are voxel indices because the
plotter has no physical spatial-coordinate metadata.

Values are restored to the dataset's original scale. Target and prediction
share symmetric color limits within each row. Error has a separate scale;
different velocity rows may also have different limits. Read the color bars
before comparing apparent contrast.

The image omits the input and does not directly display the predicted increment.
It cannot alone establish improvement over persistence or stable long-term
evolution.

## 13. Physical capabilities and limitations

| DNS concept | Current implementation |
|---|---|
| Governing equations | Learned state-to-state map |
| Pressure projection | No explicit solve; pressure is not an input or output |
| Incompressibility | No divergence-free representation or projection |
| Periodicity | Assumed by preprocessing; no explicit periodic network boundary operator |
| Viscosity and forcing | No separate conditioning variables |
| Conservation | No exact momentum or energy constraint |
| Subgrid influence | Implicit in the learned update; no explicit closure |
| Time-step control | Fixed-interval experiment; no CFL or local-error controller |
| Translation and rotation symmetry | Not guaranteed |
| Uncertainty | One deterministic output; no calibrated uncertainty estimate |
| Long-horizon evolution | Full-gradient rollout training and evaluation are implemented in the shared forecasting package |

Attention alone does not enforce physical laws. Matching total energy does not
guarantee correct spatial organization or energy transfer across scales.

The coarse snapshot may not be a complete dynamical state. Different fine-scale
fields can share that snapshot and lead to different subsequent coarse evolution.
A deterministic single-frame map approximates these dynamics. Additional
history could be a future modeling choice but is absent here.

### One-step evaluation versus rollout

Current evaluation supplies a **true observed input for every pair**.
It never feeds one validation prediction into the next prediction.

An autoregressive rollout would instead calculate

$$
\widehat U^{n+1}=F_\theta(U^n),\qquad
\widehat U^{n+2}=F_\theta(\widehat U^{n+1}),\qquad\ldots
$$

Errors then modify later inputs and can accumulate. A model accurate on
reference inputs may encounter unfamiliar states in a rollout.

Useful future checks include energy spectra, divergence, energy evolution,
velocity-increment statistics, and temporal correlations. Compare them against
the **same coarse reference**, with matching filtering and sampling interval.
You cannot establish DNS dissipation-scale fidelity from fields whose
dissipative modes have been filtered out.

These are proposed evaluation extensions, not implemented features. They can
be added while retaining deterministic forecasting as the objective.

## 14. Configuration choices and computational cost

| Setting | Effect |
|---|---|
| grid_size | Changes physical array size and token grid; must match cache |
| patch_size | Changes compression and token count |
| embed_dim | Changes feature width and most parameter counts |
| depth | Changes number of transformer blocks |
| num_heads | Splits feature width into attention subspaces; must divide width |
| mlp_ratio | Changes intermediate MLP width |
| dropout | Controls feature suppression during training |
| batch_size | Changes examples per update and memory, not temporal history |
| learning_rate | Controls optimizer step scale |
| epochs | Controls repeated passes through training data |

Pairwise attention work scales as $L^2D$. With $L=512$, one head conceptually
has 262,144 token-pair interactions per example per block.

At fixed $64^3$ resolution, changing patch side from 8 to 4 increases tokens
from 512 to 4096. That is eight times as many tokens and 64 times as much
pairwise attention work at fixed feature width. Actual runtime also depends on
other operations and hardware.

Smaller patches offer a finer representation but do not guarantee improved
forecasts, recover filtered DNS modes, or impose conservation. Their value
must be assessed through accuracy and physical diagnostics.

This is a plain global-attention ViT. There is no hierarchical encoder–decoder,
shifted-window attention, multiscale branch, or sequence parallelism. It does
not implement SwinV2-UNet or Matey/RingX.

Training is single-process and single-device. Requesting extra GPUs without
modifying the implementation does not distribute the run. Scheduler allocation
size and the number of GPUs used by Python are separate facts.

## 15. Code map and saved artifacts

| File or symbol | Responsibility |
|---|---|
| [config.yaml](../config.yaml) | Experiment settings |
| [config.py](../config.py) | Validation of settings and path resolution |
| [dataset.py: CacheIndex](../dataset.py) | Metadata, temporal checks, provenance |
| [dataset.py: PairDataset](../dataset.py) | Pairs and relative timestep |
| [dataset.py: training_statistics](../dataset.py) | Fixed physical normalization |
| [model.py: BaselineViT](../model.py) | Full normalized state-to-state model |
| [model.py: Attention](../model.py) | Q/K/V, multi-head attention, output projection |
| [model.py: Block](../model.py) | Two pre-normalized residual branches |
| [train.py](../train.py) | Optimizer, validation, checkpoint selection |
| [engine.py](../engine.py) | Device, normalization tensors, metrics |
| [evaluate.py](../evaluate.py) | Checkpoint evaluation |
| [visualize.py](../visualize.py) | Velocity-slice comparisons |
| [test_baseline.py](../test_baseline.py) | Synthetic model, data, and plotting checks |

Baseline YAML paths are resolved relative to the YAML directory. Therefore
**training.output_dir: runs/baseline-01** means
**baseline_vit/runs/baseline-01**, not a directory relative to the current shell.

Run artifacts are:

- **config.json**: resolved settings.
- **normalization.json**: means, standard deviations, counts, and source file IDs.
- **metrics.jsonl**: one record per completed epoch.
- **best.pt**: weights, configuration, normalization, reference interval, selected
  epoch, validation metrics, and cache fingerprint.
- **velocity-val.png**: created only when plotting evaluation is requested.

The output directory must be new. Checkpoints are written via a temporary file
and replacement, but optimizer state is not saved and training resume is not
implemented.

Evaluation loads the saved configuration and statistics. It checks a fingerprint
of the manifest plus selected array sizes and modification times. That is a
metadata provenance check, not a full checksum of field values.

The loader recomputes the timestep reference from the checked manifest.
The checkpoint also records it.

For job commands and plotting options, see the [baseline README](../README.md).

## 16. Glossary

| Term | Meaning here |
|---|---|
| Architecture | Arrangement and connectivity of operations |
| Tensor | Multidimensional numerical array |
| Channel | Physical component in field arrays, or internal feature axis |
| Batch | Independent examples processed together |
| Patch | Local block of grid values |
| Token | Feature vector representing a patch |
| Embedding | Mapping to a learned feature representation |
| Feature | Internal scalar descriptor without a necessary physical label |
| Linear layer | Learned affine map $Wx+b$ |
| Activation | Nonlinear scalar function, here GELU |
| Attention head | Subspace for token-to-token interactions |
| Output head | Final map to patch increments |
| LayerNorm | Per-token normalization across features |
| Residual connection | Addition of a learned correction to an existing representation |
| Forward pass | Calculating a prediction |
| Loss | Scalar fitting objective |
| Gradient | Loss derivative with respect to a parameter |
| Backpropagation | Chain-rule calculation of gradients |
| Optimizer | Rule for parameter updates |
| Epoch | One pass through training examples |
| Inference | Prediction with fitted weights |
| Checkpoint | Saved weights and supporting metadata |
| Persistence | Next field is predicted to equal the current field |
| Autoregressive rollout | Repeated prediction using previous outputs as inputs |
| Generalization | Performance on data or conditions not used to fit weights |
