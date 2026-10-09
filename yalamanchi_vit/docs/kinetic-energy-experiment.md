# Yalamanchi kinetic-energy loss experiment

## Question and approved setup

Does adding an error in volume-mean kinetic energy to five-step rollout MSE
improve the ten-step energy evolution shown in our validation slides?

The approved experiment trains Yalamanchi from scratch with **319 training
files**, matching the original slide checkpoint, and tests **lambda=1 and
lambda=10**. The existing 3D architecture is used without modification. This
is a separate experiment from the running MATEY/Yalamanchi comparison and the
retired spectral-loss experiment.

## Exact objective

For each sample and forecast lead h, compute kinetic energy from physical-valued
velocity after undoing the fixed training normalization:

```text
physical_prediction = normalized_prediction * channel_std + channel_mean
K = 0.5 * mean over spatial cells of (u² + v² + w²)
r_h = (K_predicted,h - K_actual,h) / max(K_actual,h, energy_floor)

L = mean over five leads of [normalized velocity MSE + lambda * mean_sample(r_h²)]
```

All five predictions remain connected through autograd, with no teacher forcing
or detached forecast steps. The added term uses the target energy at each lead;
it encourages the observed evolution, not constant energy. Each sample's energy
error is squared before batch averaging so opposite errors cannot cancel.

The denominator floor is `max(1e-8 * K_training, 1e-12)`, where
`K_training = 0.5 * sum(channel_std² + channel_mean²)` uses the same fixed training
statistics as the baseline. Its actual numeric value is saved in `loss.json`
and the checkpoint. For our nonzero-energy data the denominator is target energy.

K includes the spatial mean flow. It is volume-mean kinetic energy per unit mass
in cached velocity units squared, not mean-subtracted turbulent kinetic energy,
an integral multiplied by density, or kinetic energy of standardized fields.
Neither a density nor a physical box length is needed for this definition.

For a 10% relative energy error, the energy contribution is 0.01 at lambda=1
and 0.1 at lambda=10. The weights are experimental choices; neither is known
to be optimal. A scalar global-energy penalty does not specify the spectrum,
spatial phase, vorticity, dissipation, or divergence. Improved energy must be
assessed alongside velocity error and the forecast fields.

## Matched settings and historical control

Preparation reads the actual checkpoint recorded by
`runs/validation-slides-59353752/report.json` and verifies its SHA-256 hash. It
copies its model, data and training settings into separate run configurations.
Changing a source checkpoint or cache invalidates the comparison.

| Setting | Both new runs and original MSE reference |
| --- | --- |
| Training files | 131–449 inclusive: 319 files |
| Stored training snapshots | 3,190 entries; neighboring file endpoints repeat |
| Training windows | 1,595 per epoch, within-file |
| Input | One full 64³ velocity snapshot |
| Training horizon | Five recursive steps |
| Patch size / widths | 4³ / 96, 192, 384, 192, 96 |
| Blocks / heads | Two blocks per stage; heads 4, 8, 16 by resolution |
| Boundaries / dropout | Periodic / zero |
| Epochs / batch size / seed | 100 / 2 / 42 |
| Optimizer | AdamW, learning rate 1e-4, weight decay 0.01, clip norm 1 |
| Optimizer updates | 798 per epoch |
| Validation during training | Files 451–490, 200 within-file five-step windows |
| Final validation evaluation | Same files, 351 deduplicated cross-file ten-step origins |
| Slide trajectory | File 451, frame 0; t=121.72249603271484 to 122.02233123779297 |
| Slice | z=32 |

Fixed normalization is recomputed from training files and required to match
the reference checkpoint. Every run reseeds model initialization and the
training sampler. The lambda variants therefore use the same initialization
and training-window ordering, subject to normal numerical nondeterminism.

The original MSE-only checkpoint is
`runs/matched-comparison-59178955/yalamanchi_vit/best.pt`, selected at epoch 100.
It is reevaluated without training to obtain per-origin energy errors that were
not present in the old aggregate report. It is a historical control, not a newly
retrained third arm or a multi-seed study. No test split is used to choose a
weight or checkpoint in this experiment; these comparisons use validation.

## Checkpoint selection and weights

Each new run saves three inference checkpoints:

- `best.pt`: lowest mean five-step **validation normalized MSE**, retaining the
  reference's primary selection criterion. This checkpoint generates the default
  ten-step slides and comparison.
- `best-objective.pt`: lowest five-step **combined validation objective** for
  that run's lambda, saved for an explicitly labeled alternative analysis.
- `last.pt`: the last completed epoch.

Combined objective values from different lambdas are not directly comparable.
Use energy errors and velocity errors to assess the trade-off. Selecting only
the lowest-energy checkpoint and comparing it to an MSE-selected reference would
also change the selection rule; the default comparison avoids that confound.

The checkpoints retain the existing model architecture identifier and include
an `energy_loss` description and normalization. Their training objective is
explicitly labeled `rollout_mse_kinetic_energy`. Use the dedicated experiment
runner for training and labeled presentations. Optimizer/RNG state for exact
training resumption is not saved.

## Evaluation and presentation outputs

Each new run produces PNGs, a PDF and a self-contained HTML slideshow using the
existing renderer and the same selected validation trajectory as the old slides.
The context slide names the energy-loss weight. Each deck uses fixed color
scales across its own ten steps, selected from that deck's fields; color limits
are not forced to be identical across separate decks.

The experiment root contains two three-panel comparison figures:

- `selected_trajectory-comparison.png`: original slide trajectory, comparing
  actual energy, all three model energy curves, relative energy error, and
  velocity MSE across ten leads.
- `full_validation-comparison.png`: the same quantities over all 351 origins.

The aggregate energy error is `sqrt(mean_origin(r²))`. It is calculated per
forecast before aggregation, rather than as the error in the mean energy curve.
Reports also include relative energy bias and mean absolute relative energy
error, normalized/physical velocity MSE, relative L2 and persistence skill.
On the single example, energy RMSE reduces to absolute relative energy error.

The original full-validation mean at lead 10 was K_pred=2.454415 and
K_actual=2.561956, about 4.2% low. The selected slide trajectory had different
values; its mean-energy deficit is not interchangeable with the aggregate result.

Raw per-origin actual/predicted energies are retained in `energy-trajectories.npz`
with shape `[origin, actual_or_predicted, lead]`, index 0 actual and 1 predicted.
The selected full-volume forecast is stored in `selected-forecast.npz`. No new
spectral penalty or spectral claim is introduced by this experiment.

## Submit and inspect

From the project root:

```bash
sbatch jobs/yalamanchi-energy-loss.sbatch
```

This prepared job has not been submitted by the assistant. It requests one GPU
node, two independent single-GPU tasks, eight CPUs and 64 GB host memory per task,
and nine hours. Lambda=1 and lambda=10 train concurrently. The lambda=1 worker
also reevaluates the original MSE checkpoint after its own evaluation.

The old run recorded about 5.60 hours for 100 training/validation epochs. The
nine-hour limit covers preparation, new training and validation, presentations,
and reference reevaluation; new-loss GPU time has not yet been measured.

Output root: `runs/yalamanchi-energy-loss-JOBID/`.

```text
experiment.json, energy-1.json, energy-10.json
energy-1.log, energy-10.log
comparison.json, comparison.csv
selected_trajectory-comparison.png, full_validation-comparison.png
reference/                         reevaluated unchanged MSE checkpoint
energy-1/, energy-10/:
    best.pt, best-objective.pt, last.pt
    config.json, normalization.json, loss.json, environment.json, metrics.jsonl
    validation-slides/:
        forecast-slides.pdf, forecast-slides.html, numbered slide PNGs
        report.json, selected-forecast.npz, energy-trajectories.npz
```

The comparison CSV has 30 rows: three models times ten validation leads. Both
components and combined loss are logged separately during training/validation.
Weights are written atomically after completed epochs. A failure preserves
available results and creates a failure report when catchable; scheduler timeouts
may interrupt summarization. A new training run requires a new output directory.

An alternative checkpoint can be evaluated inside a GPU allocation without
retraining, into a new directory:

```bash
python -B -m yalamanchi_vit.energy_experiment evaluate \
  --output runs/yalamanchi-energy-loss-JOBID \
  --checkpoint runs/yalamanchi-energy-loss-JOBID/energy-10/best-objective.pt \
  --evaluation-output runs/yalamanchi-energy-loss-JOBID/energy-10/objective-selected-slides
```

The original models, shared training/evaluation code and existing jobs are not
edited by this experiment. Its additions are `energy_loss.py`,
`energy_experiment.py`, the dedicated tests, job, and this document.

## Verification

Seven tests cover physical energy and normalization, exact lambda-zero loss
agreement and gradient agreement with the existing MSE rollout, noncancellation
of opposite sample energy errors, an analytic five-step gradient, zero-energy
targets, invalid weights, and both actual-model synthetic training-to-slides
workflows. They check checkpoint labels, reference integrity and comparison
plots, including incomplete-run reporting. Production preparation was verified
against the real slide checkpoint/cache without training or job submission.
