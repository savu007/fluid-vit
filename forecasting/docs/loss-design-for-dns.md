# Loss-design reference for coarse DNS velocity forecasting

Reviewed against the repository implementation on October 1, 2026.

This document is for a DNS practitioner designing training objectives for the
plain ViT and Yalamanchi-inspired model. It distinguishes **implemented behavior**,
**verified data properties**, **unverified metadata**, and **candidate objectives**.
It does not select or implement a new loss.

The task remains deterministic forecasting of a particular future coarse velocity
field. Statistical fidelity and physical consistency are additional requirements,
not substitutes for specifying which forecast is being predicted.

## 1. Where the current behavior is defined

| Concern | Source of truth |
|---|---|
| Filtering and cache construction | [coarsening/coarsen.py](../../coarsening/coarsen.py) |
| Large-cache selection | [config-131-531.yaml](../../coarsening/config-131-531.yaml) |
| Cache validation and training statistics | [baseline_vit/dataset.py](../../baseline_vit/dataset.py) |
| Temporal windows | [forecasting/dataset.py](../dataset.py) |
| Rollout loss and per-lead evaluation | [forecasting/rollout.py](../rollout.py) |
| Shared trainer and checkpoint selection | [forecasting/train.py](../train.py) |
| Legacy baseline one-step trainer | [baseline_vit/train.py](../../baseline_vit/train.py) |
| One-step metrics | [baseline_vit/engine.py](../../baseline_vit/engine.py) |
| Architecture explanations | [Plain ViT](../../baseline_vit/docs/architecture-for-dns.md), [Yalamanchi](../../yalamanchi_vit/docs/architecture-for-dns.md) |

The architecture guides contain historical design discussion. For current loss,
rollout, and checkpoint semantics, use this document and the code above.

No spectral, divergence, gradient, structure-function, momentum-residual, or
energy-budget penalty is currently implemented. Both models use the same
full-gradient rollout objective when configured for rollout training.

## 2. Data contract: what is established

| Property | Established behavior |
|---|---|
| Source and target grid | Source velocity at 384³, cached at 64³ |
| Variables | Three velocity components, in order u, v, w |
| Cached file | Float32 array [10, 3, 64, 64, 64] |
| Filter | Sharp Fourier truncation retaining integer modes -31 through 31 on each axis |
| Support | A cube in Fourier space, not a spherical cutoff |
| Coarse Nyquist planes | Set to zero during preprocessing |
| Arithmetic | Filtering in float64; stored output in float32 |
| Boundary assumption | Periodicity is explicitly required by the coarsening configuration |
| Time | Saved numerical coordinates are recorded in the manifest |
| Large cache | Files 131–531, at /pscratch/sd/s/savu/fluid-vit/hit-coarse-64-131-531 |
| Training selection | Defined separately by each experiment's configuration |

Cache completeness is not evidence that a training split has been selected or
validated. The existing config-rollout.yaml files still describe the original small-data
experiment. Separate config-matched.yaml files now prepare the larger-data
[matched-specification comparison](matched-specs.md), including its approved
timestamp-spacing tolerance. These preparations do not change the loss.

Targets are **filtered DNS**, not independently evolved solutions of the
unfiltered equations on a 64³ mesh. Nonlinear interactions with removed scales
can influence the retained scales. One coarse snapshot need not contain all
information needed to determine its exact successor.

The filter has no explicit counterpart at the model output. A prediction may
populate coarse Nyquist modes that are absent from the targets. The model also
does not guarantee solenoidality, momentum conservation, or an energy balance.

## 3. Physical metadata that remains unresolved

The inspected cache manifest records grids, channels, timestamps, source
provenance, and filter verification. It does **not** record the following:

| Required information | Needed for | Status/action |
|---|---|---|
| Box lengths Lx, Ly, Lz; grid spacing and coordinate convention | Physical wavenumbers and derivatives | Verify source coordinates and simulation documentation; do not assume a 2π box |
| Velocity and time units, reference scales | Dimensional interpretation and nondimensional losses | Not established; the inspected time attributes for file 131 are empty |
| Kinematic viscosity and any variation between files | Viscous term, dissipation, Reynolds-number interpretation | Not established |
| Forcing field or forcing rule | Momentum residual and energy injection | Not established |
| Pressure convention and availability | Pressure gradient in a momentum residual | Pressure is absent from this cache; source availability/convention needs verification |
| DNS differential operators and temporal sampling details | Consistent residual and divergence assessment | Saved-snapshot intervals are not necessarily solver timesteps |
| Subgrid-stress information or closure choice | Filtered momentum balance | Not present in the current velocity cache |
| Intended statistical averaging ensemble | Interpretation of statistics and uncertainty | Decide explicitly; nearby windows are correlated |

An absent manifest field does not imply the quantity is unavailable in the
original simulation. It means a loss must not silently assume its value.

“Physical velocity” in our code means the original dataset scale after undoing
normalization. It does not establish SI units: nondimensional DNS values remain
nondimensional.

## 4. Model output and normalization

Let U denote cached velocity and c a velocity component. The fixed transform is

\[
X_c=(U_c-\mu_c)/\sigma_c,\qquad
\widehat U_c=\sigma_c\widehat X_c+\mu_c.
\]

The three means and standard deviations are computed from all training snapshots
and all their grid points, using float64 accumulation and population variance.
They are not fitted separately to each frame. Values below 10^-12 for standard
deviation are replaced by one. Statistics are frozen and saved with the weights.

Repeated endpoint snapshots in adjacent files are counted as stored when fitting
statistics. The loader allows touching timestamps within a split, while requiring
strict temporal separation between training, validation, and test groups.

Both model interfaces are:

~~~python
prediction_normalized = model(input_normalized, relative_dt)
# input and prediction: [B, 3, 64, 64, 64]
# relative_dt: [B], equal to saved interval / training reference interval
~~~

The returned field already includes the residual addition:

\[
F_\theta(X,\tau)=X+G_\theta(X,\tau).
\]

G is a learned increment, **not** an explicitly parameterized time derivative.
It is not multiplied by the physical interval by the model. Do not add the input
a second time, and do not interpret G directly as acceleration.

For physical statistics or derivatives, undo normalization differentiably:

~~~python
physical_prediction = predicted_normalized * std + mean
# Keep this operation in PyTorch; no detach(), numpy(), or no_grad().
~~~

Componentwise scaling matters. Since the means are spatially constant,

\[
\nabla\cdot\widehat{\mathbf U}
=\sigma_u\,\partial_x\widehat X_u
+\sigma_v\,\partial_y\widehat X_v
+\sigma_w\,\partial_z\widehat X_w.
\]

Computing divergence directly from X generally penalizes a different quantity.
Likewise, energy of X is not physical kinetic energy. The fixed training mean
is also not necessarily the instantaneous spatial mean of an individual field.

## 5. Exact current training objective

For a window with H successors, initialize from truth once:

\[
\widehat X^{(0)}=X^{(0)},\qquad
\widehat X^{(h)}=F_\theta(\widehat X^{(h-1)},\tau_h).
\]

Let N=64, C=3, and B be the batch size. The implemented loss is

\[
L_{\mathrm{current}}=
\frac{1}{HBCN^3}
\sum_{h=1}^{H}\sum_{b,c,\mathbf j}
\left(\widehat X^{(h)}_{bc\mathbf j}-X^{(h)}_{bc\mathbf j}\right)^2.
\]

Equivalently, each original-scale component error is weighted by 1/σc².
All voxels, samples, and forecast leads receive equal weight. The default
rollout horizon is five; single-step training uses H=1.

The model is reused with shared weights at every step. All intermediate
predictions remain connected to the gradient computation. No reference future
is fed back into the model, and no step is detached. There is one optimizer update
per batch of complete windows, not one per predicted step.

Each ten-snapshot file supplies 10-H windows. Windows never cross file boundaries.
At H=5, each example contains one initial field and five reference successors.
Five leads denote five saved-snapshot intervals, not five DNS solver steps.
The reference interval is the median within-file training interval; the loader
checks spacing throughout the selected data with configured tolerances.

For one step from a common input, state error equals increment error. During a
rollout, the predicted predecessor differs from the true predecessor. Comparing
the model's increment to the true DNS increment is therefore a different loss
from the state-matching objective above.

Defaults are AdamW, learning rate 10^-4, weight decay 0.01, global gradient-norm
clipping at 1, batch size 2, and float32 model arithmetic. Weight decay acts in the
optimizer; it is not an explicit physics or statistical term in the reported loss.

## 6. Training, validation, and saved weights are separate choices

The shared trainer selects best.pt by the lowest **mean validation normalized
MSE across the configured forecast leads**. This remains true if someone changes
only forecast_loss. A new training objective would not automatically change
checkpoint selection.

Current evaluation reports normalized MSE, original-scale MSE, relative L2,
persistence error/skill, and kinetic energy. Rollout evaluation additionally
reports MAE. It uses original-scale errors with float64 accumulation.
Relative L2 is computed from aggregate squared sums, not an average of individual
sample-relative errors.

Kinetic energy is currently

\[
K=\frac12\langle u^2+v^2+w^2\rangle,
\]

including the spatial mean flow. It is not explicitly fluctuation kinetic energy.
It is averaged over voxels and evaluation examples for each lead.

Every rollout lead uses the same starting windows. Persistence holds the initial
field fixed. The separate one-step report uses all valid pairs, so its first-step
score need not equal the first-lead rollout score.

Physical diagnostics intended for training need separate differentiable functions:
the current evaluation functions use no-grad and scalar accumulation via item().
Reusing them directly as a loss would break gradient propagation.

Record the training objective, model-selection criterion, and reporting metrics
separately in any new experiment.

## 7. Candidate statistical losses: definitions to settle

These are design options, not implemented features or prescribed coefficients.

### 7.1 Energy spectrum

For a uniform periodic grid, one convenient proposed convention is a full complex
DFT normalized by the number of grid points:

\[
a_c(\mathbf m)=\frac{1}{N^3}
\sum_{\mathbf j} U_c(\mathbf j)
e^{-2\pi i\mathbf m\cdot\mathbf j/N}.
\]

Then Parseval gives

\[
K=\frac12\sum_{c,\mathbf m}|a_c(\mathbf m)|^2.
\]

With known box lengths, physical wavevectors are
k_i=2πm_i/L_i. Integer mode magnitudes can be reported before box lengths are known,
but must not be labeled physical wavenumbers. A cubic array does not by itself
prove equal physical box lengths.

Define shell energy by summing modal energies in explicitly defined bins.
A shell **sum** and a shell **average** are different quantities. If using a
one-sided real FFT, account for the omitted conjugate modes, without doubling
the zero or Nyquist plane on the truncated axis.

Two possible discrepancies are

\[
L_E=\sum_s w_s
\left[\frac{\widehat E_s-E_s}{E_*}\right]^2
\]

or

\[
L_{\log E}=\sum_s w_s
\left[\log\frac{\widehat E_s+\epsilon E_*}{E_s+\epsilon E_*}\right]^2.
\]

Specify the positive reference scale E*, dimensionless ε, bins, and weights.
Absolute error emphasizes energetic shells; a log discrepancy increases the
importance of weak-energy shells and becomes sensitive to the chosen floor.

Also decide:

- Total velocity spectrum versus fluctuation spectrum with the spatial mean removed.
- Per-snapshot matching versus matching a spectrum averaged over many snapshots.
- Which high-wavenumber shells to include: cubic truncation leaves outer shells
  only partially populated relative to a full spherical shell.
- Whether to exclude Nyquist modes, penalize their generated energy separately,
  or explicitly constrain output support. Excluding them alone leaves a blind spot.

A spectrum constrains amplitudes aggregated over modes; it does not determine
Fourier phases or the locations of coherent structures. An unweighted Fourier
coefficient squared error, with Parseval-consistent normalization, is just
physical-space squared error written in another basis—not a new statistic.

### 7.2 Energy and mean velocity

A dimensionless candidate is L_K=((Khat-K)/K*)², with a fixed positive energy scale.
This measures total amplitude, not scale distribution. Matching every spectral
shell already constrains their summed energy, so a separate energy term may
duplicate part of the objective.

Choose whether mean velocity is matched to the target, constrained to its initial
value, or controlled by another known rule. Neither conserved mean momentum nor
constant instantaneous kinetic energy should be assumed without checking forcing
and governing conditions.

### 7.3 Velocity increments and structure functions

For a separation vector r, form periodic increments
δrU(x)=U(x+r)-U(x). Candidate statistics include second-order increments and higher
moments of longitudinal or transverse components.

Record separation distances, directions, interpolation if needed, component
projections, averaging, and reference scaling. Fixed voxel separations alone are
not physical lengths without grid metadata. Near the filter scale, the target
statistics are those of the filtered field.

An increment-field error retains spatial alignment information. An error between
spatially averaged structure functions constrains statistics without matching
where structures occur. These are different objectives.

Hard histogram counts are generally unsuitable for direct backpropagation.
A PDF-based loss needs a differentiable estimator and documented bandwidth/binning.
Higher-order moments may be dominated by rare events; their stability must be
assessed on the available sample ensemble.

## 8. Physical constraints and their limits

### 8.1 Divergence

A candidate dimensionless penalty is

\[
L_{\mathrm{div}}=
\frac{\langle(\nabla\cdot\widehat{\mathbf U})^2\rangle}
{(U_*/L_*)^2}.
\]

Specify the velocity and length scales and the derivative operator. Fourier
derivatives on a periodic field and finite-difference derivatives need not yield
the same residual. Compare reference divergence using the same discrete operator.
A solenoidal spectral field need not have exactly zero finite-difference divergence.

Nyquist treatment must be explicit for real spectral derivatives. Predictions
can contain Nyquist energy even though reference fields have those planes zeroed.
Do not silently discard an imaginary component as a substitute for defining the
derivative convention.

### 8.2 Projection instead of a penalty

An alternative is to project each nonzero Fourier mode:

\[
\widehat{\mathbf U}_{\perp}(\mathbf k)=
\widehat{\mathbf U}(\mathbf k)
-\mathbf k\frac{\mathbf k\cdot\widehat{\mathbf U}(\mathbf k)}
{|\mathbf k|^2}.
\]

Preserve the zero mode separately, and define discrete/Nyquist handling.
This enforces solenoidality for the chosen operator; it does not enforce correct
dynamics. Applying it changes the model's output map. If used during training,
specify whether the projected state is also fed into the next rollout step.
A projection applied only at evaluation creates a different recursive predictor.

### 8.3 Momentum and energy-balance residuals

Filtering the nonlinear velocity products introduces unresolved-stress terms.
A filtered momentum residual needs a consistent pressure treatment, viscosity,
forcing, temporal derivative, and subgrid contribution or explicit closure
assumption. Velocity-only cached snapshots do not supply all of these.

Even for the DNS reference, a derivative estimated from saved snapshots need not
match the solver's discrete time update. Check any proposed residual on reference
fields before using it as a training target.

Resolved energy loss cannot be equated blindly with molecular viscous dissipation:
forcing, transfer to unresolved scales, and the chosen filter enter its balance.
Penalizing every energy decrease would impose an unjustified constraint.

These are metadata/modeling questions. Adding the words “Navier–Stokes residual”
to a loss does not resolve them.

## 9. Averaging, weighting, and gradients

For a statistic S, distinguish

\[
\frac1B\sum_b d(S(\widehat U_b),S(U_b))
\quad\text{from}\quad
d\left(\frac1B\sum_b S(\widehat U_b),\frac1B\sum_b S(U_b)\right).
\]

The second permits errors in different examples to cancel. With shuffled batches
of two, a batch statistic is also a noisy and changing proxy for an ensemble.
Likewise, averaging a statistic over forecast leads can hide excessive decay
at later leads.

A general design template is

\[
L=\sum_{h=1}^{H}\alpha_h
\left[\lambda_U L_U^{(h)}
+\lambda_E L_E^{(h)}
+\lambda_D L_{\mathrm{div}}^{(h)}
+\cdots\right],
\qquad \alpha_h\ge0,\quad\sum_h\alpha_h=1.
\]

This is a template, not a proposed final choice. The current loss has αh=1/H,
λU=1, and all other terms absent.

For each term specify its units, reduction axes, denominator/floor, and whether
normalizers are fixed from training data or vary with each target. Avoid choosing
weights solely because scalar losses have similar magnitudes: parameter-gradient
magnitudes may still differ substantially. Record each term separately and check
finite gradients through all rollout steps.

Choose weights and model-selection rules using training/validation data, not the
held-out test set. Reporting independent diagnostics that were not optimized helps
reveal compensation between loss terms.

## 10. Validation experiments before adopting a loss

Useful small tests include:

- Constant velocity: correct mean and energy; zero derivative.
- A single Fourier mode: correct amplitude, physical wavenumber, and shell energy.
- Parseval consistency, including real-FFT multiplicity.
- A constructed solenoidal field and a compressible field for divergence/projection.
- Target identical to prediction: zero discrepancy for matching losses.
- A periodic translation: unchanged global spectrum despite pointwise mismatch.
- A changed mode phase: unchanged modal energy with changed spatial structure.
- Channel rescaling: physical diagnostics agree after normalization is undone.
- Zero-energy and near-zero-energy cases: finite values and controlled gradients.
- A tiny recursive model: verify gradients reach early steps for the combined loss.

Then assess the reference data itself: divergence under the chosen operator,
spectral support, energy evolution, and statistical variability. This establishes
what “faithful” means for the coarse target and helps identify incompatible penalties.

These tests validate definitions and implementation. They do not establish
turbulent-flow skill or long-horizon stability.

## 11. Decisions to write down before changing code

1. What is the primary task: aligned short-horizon forecasting, long-time statistics,
   or an explicitly weighted combination?
2. Which original-scale or fluctuation quantities should be matched?
3. Which physical metadata has been verified, and from which source?
4. Which spectral/derivative conventions define each quantity?
5. Are statistics compared per field, per lead, or over a specified ensemble?
6. How are terms nondimensionalized, weighted, and stabilized near zero?
7. Are constraints penalties or output transformations, and what feeds the rollout?
8. What validation criterion selects the saved checkpoint?
9. Which independent diagnostics and uncertainty summaries assess success?

No particular spectral weighting, divergence coefficient, or physics residual
is selected by this reference.

Finally, loss comparisons and architecture comparisons are separate experiments.
The current plain ViT has six blocks, 8³ patches, and 3.36M parameters; the current
Yalamanchi-inspired model has ten blocks, 4³ patches, and 8.30M parameters. A separate
[matched-specification experiment](matched-specs.md) is now prepared with
10 blocks, 4³ patches, and parameter counts within 3.53%; it has not been run.
Changing a loss and those architectural settings together would confound their effects.
