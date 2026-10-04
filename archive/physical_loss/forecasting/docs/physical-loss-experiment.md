# MSE plus energy-spectrum training experiment

Both matched-specs architectures now have a separate, opt-in physical-loss
training path. This experiment trains a deterministic five-step forecast.
It is spectral regularization, not a Navier–Stokes residual or a guarantee
of physically admissible predictions. No job is submitted by preparing these files.

## Exact objective

Let z = (u - mean)/std be componentwise normalized velocity, using statistics
from training snapshots only. Recursively predict five fields, feeding each
prediction into the next step with the recorded relative time interval.
Gradients flow through all five steps.

For each sample and lead, the objective is

\[
L = \operatorname{mean}_{c,x,y,z}(\hat z-z)^2
 + 0.01\,\operatorname{mean}_{s\in S}
 \left[\log\left(\hat E_s/K_*+10^{-6}\right)
       -\log\left(E_s/K_*+10^{-6}\right)\right]^2.
\]

Average this loss equally over samples and the five forecast leads.
The natural logarithm is used. The MSE and spectrum terms are both dimensionless.

The spectrum uses velocity reconstructed in the original data units, not
normalized channels. Compute a full complex 3D FFT with forward normalization
(division by N³). For each integer mode vector m, assign shell
s = floor(sqrt(mx² + my² + mz²)). The shell energy is

\[
E_s=\tfrac12\sum_{m:\,\lfloor|m|\rfloor=s}\sum_{c=1}^3
|\mathcal F(u_c)_m|^2.
\]

These are shell **sums**, not averages over modes. Including shell zero, the
sum equals one-half the spatial mean of squared velocity (Parseval).
The loss excludes shell zero and weights every populated nonzero shell equally.
At N=64, these are shells 1–55. It includes the coarse Nyquist planes and
partially populated outer shells of the cubic Fourier grid. No spectral mask
is applied to predictions. The targets' preprocessing removed Nyquist planes.

K* = 0.5 sum_c(mean_c² + std_c²) is a fixed training-derived energy scale.
The existing normalization routine substitutes std=1 for a degenerate channel;
in that case K* is a normalization convention, not the exact training energy.
The additive floor corresponds to 10^-6 K* per shell and keeps logarithms finite.
Its value and the weight 0.01 are initial experimental choices, not calibrated
physical constants. A coefficient of 0.01 does not imply a 1% contribution:
inspect the logged weighted term and MSE to see their actual magnitudes.

## Why these two terms

MSE anchors the predicted field to the particular reference future, including
phase and mean velocity. The spectrum term discourages an incorrect distribution
of kinetic energy across scales; logs make relative spectral discrepancies matter
until the energy approaches the floor.

Equal spectra do not imply equal fields, correct vortex geometry, correct
component anisotropy, or correct higher-order statistics. Periodic translations
have identical spectra. The objective does not enforce divergence-free flow,
a momentum equation, energy budgets, or subgrid stresses. Those require further
design and metadata. Shells are defined in integer index space; interpreting
them as physical isotropic wavenumbers requires known domain lengths.
See the [DNS loss-design reference](loss-design-for-dns.md) for remaining gaps.

## Files and experimental controls

Both packages have matching new filenames:

- model_physical.py: PhysicalViT extends the existing model with the shared
  rollout-loss method. Forward architecture and parameter layout are unchanged.
- train_physical.py: entry point for the separate physical-loss trainer.
- config-physical.yaml: matched-specs settings plus the loss section.

The implementation is [physical_loss.py](../physical_loss.py), with the trainer
in [train_physical.py](../train_physical.py). Existing configuration loaders
accept the loss section only when explicitly opted in by the physical trainer.
An ordinary MSE training entry point rejects these configs.

The experiment retains the [matched-specs settings](matched-specs.md):
4³ patches, ten transformer blocks, 24 features per attention head, MLP ratio 4,
zero dropout, and about 8.0M versus 8.3M parameters. Global and hierarchical
attention remain different. Both start from scratch with seed 42.

The scratch cache is /pscratch/sd/s/savu/fluid-vit/hit-coarse-64-131-531.
Training uses files 131–449, validation 451–490, and test 492–531;
450 and 491 remain gaps. This uses the 401-file cache, not all 401 files for
training. The approved 0.3% spacing tolerance is retained, and recorded
intervals are supplied to the models. The job does not evaluate test values.

Both train for 100 epochs, batch size 2, five-step full-gradient rollouts,
float32, AdamW learning rate 1e-4, weight decay 0.01, and gradient clipping 1.
Training has 1,595 windows; validation has 200. The change is the objective,
not additional transformer capacity.

## Job and results

From the repository root, when ready to submit:

~~~bash
sbatch physical-loss.sbatch
~~~

The [job script](../../physical-loss.sbatch) requests two nodes and runs one
model per node concurrently, each using one GPU. It does not distribute an
individual model or launch any MSE-only comparison. The twelve-hour limit is
an initial request: full-size CUDA memory use and completion time have not
been benchmarked for this objective.

Output is runs/physical-loss-JOBID/ in the repository. Existing run directories
are refused. The root contains experiment.json and separate model logs.
Each model subdirectory contains:

- best.pt: weights selected by lowest mean validation combined loss.
- best-mse.pt: weights from the same physical-loss training, selected by lowest
  validation normalized MSE; this is not an MSE-only training control.
- metrics.jsonl: epoch training_loss, training_mse, training_spectral_loss,
  weighted_training_spectral_loss, validation terms per lead and their means,
  update counts, time and peak allocated CUDA memory.
- config.json, normalization.json, environment.json: resolved settings,
  training normalization, energy scale, shells, parameter count and device.
- evaluation-val.json: combined-loss-selected checkpoint's usual rollout
  metrics and physical-loss metrics.
- velocity-val.png: one-step reference/prediction/error visualization.

Checkpoints remain compatible with existing evaluation commands because model
parameters and architecture tags are unchanged. No optimizer-resume state is saved.
Both workers are waited for; either failure makes the batch job fail while retaining
the other worker's outputs.

## Verification

Synthetic tests cover Parseval including the zero and Nyquist modes, a known
Fourier mode, amplitude sensitivity, translation invariance of the spectral
term, normalization reversal, zero-energy finiteness, numerical gradient
checking, gradient flow across the rollout, config isolation, and training /
checkpoint loading / evaluation for both architectures. These tests do not
establish physical improvement or full-size GPU performance.
