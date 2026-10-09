# Fresh MATEY / Yalamanchi comparison and energy diagnostics

## Approved experiment

Train both models from scratch, using five-step recursive training with baseline
normalized MSE averaged across the five leads. Retain the preceding experiment's
107 training files and nine-hour, two-GPU allocation. Generate both total-energy
evolution pictures and predicted-versus-actual energy spectra.

The supplied script uses one Perlmutter GPU node with two independent workers,
one GPU per model, running concurrently. It does not distribute either model.
Each step requests explicit CPU and memory resources so one step does not reserve
the entire node and force the other to wait. This follows
[NERSC's single-GPU tasks in parallel example](https://docs.nersc.gov/systems/perlmutter/running-jobs/#single-gpu-tasks-in-parallel).
Host memory is 128 GB total, divided into 64 GB per worker; this is separate from
GPU memory. The GPU memory capacity is whatever the allocated node provides.

## Exact training and evaluation settings

| Setting | MATEY | Yalamanchi |
| --- | --- | --- |
| Model | Public-code 3D TurbT adaptation | Existing local 3D SwinV2-UNet |
| Observed frames per call | 4 | 1 |
| Grid | 64³, velocity channels `u,v,w` | Same |
| Patches | 2³ at each hierarchy level | 4³ |
| Widths | 192 at all three levels | 96, 192, 384, 192, 96 |
| Blocks | Four per level; twelve total | Two per stage; ten total |
| Heads | Three at each level | Four, eight, sixteen by resolution |
| Training horizon | Five recursive steps | Same |
| Loss | Fixed training-channel-standardized MSE averaged over five steps | Same |
| Epochs / seed | 100 / 42 | Same |
| AdamW learning rate / weight decay | 1e-4 / 0.01 | Same |
| Gradient norm clipping | 1 | Same |
| Training batch size | 1 | 2 |
| Training windows per epoch | 214 | 535 |
| Validation windows | 80 | 200 |
| Optimizer updates per epoch | 214 | 268 |
| Checkpoint selection | Lowest mean five-step validation normalized MSE on each model's within-file windows | Same rule |
| Test forecast horizon | Five steps | Same |
| Test origins | Same 353 origins with four available history frames and five future targets | Same |

Data come from `/pscratch/sd/s/savu/fluid-vit/hit-coarse-64-131-531`.
Training files are 131–237, validation 451–490, test 492–531; declared gaps are
450 and 491. Data/seed/optimizer settings are captured in `experiment.json` and
the per-worker resolved YAML files. Recorded intervals are retained with the
approved 0.3% spacing tolerance. This is a five-step comparison, not the earlier
100-step testing experiment.

The runner takes the existing Yalamanchi architecture from
`yalamanchi_vit/config-runtime.yaml` and the approved data subset/MATEY settings
from `ornl_vit/config.yaml`. It writes new run-specific configs rather than
changing either source config. The larger training-file list in the Yalamanchi
source is replaced in the resolved comparison config by the common 107-file list.

Training windows stay within files. Four observed frames plus five targets give
MATEY two starts per ten-frame file; one observed frame plus five targets gives
Yalamanchi five. Batch sizes and optimizer update counts also differ. Best-model
selection uses each model's own within-file validation windows. **This is a model
comparison, not a matched-capacity or architecture-only experiment.**

The existing MATEY implementation retains internal gradient stops, its history
normalization, reflection padding in some spatial layers, and bfloat16 Q/K/V
casts in the explicitly selected SDPA path. Yalamanchi retains its existing
fixed input standardization, periodic spatial treatment and arithmetic. Neither
architecture or loss is modified by this experiment. See the
[MATEY adaptation record](../../ornl_vit/docs/adaptation.md) for source differences.

## How the comparison is aligned

The common evaluator constructs nine-frame windows from the deduplicated test
trajectory. Frame four is the forecast origin. MATEY receives frames one through
four; Yalamanchi receives frame four alone. Both predict frames five through nine.
Yalamanchi therefore does not receive extra history covertly.

Recorded origin/target times, cache fingerprints and actual-field energy arrays
are checked across workers before a complete comparison is declared. Both use
the same fixed training mean/std for loss units and normalized-error reporting.
No test fields participate in training or checkpoint selection.

## Energy calculation and pictures

Energy is calculated from restored, physical-valued cached velocities:

```text
K = 0.5 * mean over spatial cells of (u² + v² + w²)
```

This is volume-mean kinetic energy per unit mass, expressed in cached velocity
units squared. It includes mean-flow energy; it is not mean-subtracted turbulent
kinetic energy or an integral multiplied by density/domain volume. Lead zero is
the actual observed origin and is shared by both predicted/actual curves.

Each model gets three PNGs:

- `energy-evolution.png`: mean actual and predicted K versus leads 0–5, averaging
  over the same 353 test origins. This is an average by forecast lead, not one
  continuous simulation trajectory.
- `energy-example.png`: the first test origin and its five actual/predicted
  future energies plotted against recorded simulation time.
- `energy-spectra.png`: six panels (observed origin and leads 1–5) showing actual
  and predicted spectra averaged over the common origins.

At the experiment root, `energy-comparison.png` overlays both models and the
actual mean energy. Plots display direct calculated energies; no smoothing or
rescaling to force agreement is applied. The example is chosen by chronological
index zero, rather than selecting a visually favorable forecast.

### Spectrum conventions

The periodic cubic velocity field is transformed using NumPy's 3D real FFT in
float64. Each mode contributes half its summed three-component squared Fourier
amplitude, normalized by N⁶ for an N³ field. Interior modes along the reduced
FFT axis are doubled to account for the omitted conjugates; zero and even-grid
Nyquist endpoints are not doubled.

Mode magnitudes are rounded into unit-width integer shells:

```text
shell m = floor(sqrt(mx² + my² + mz²) + 0.5)
```

The spectrum is shell-integrated energy, not divided by shell mode count. Summing
**all** saved shells, including shell zero, must reproduce K; the evaluator checks
this for every field. The horizontal axis is Fourier mode number, not physical
wavenumber in inverse length: no unavailable domain length is invented. If a
physical box length L is established, angular wavenumber is k = 2*pi*m/L.

CSV/NPZ data retain all shells. Logarithmic plots omit m=0 and shells beyond the
axial Nyquist radius N/2, where not all directions are represented. The displayed
spectral curves therefore need not sum to the displayed total K. Mean flow
remains in the saved zero shell and in total K. Spectra assume our periodic data
domain; predicted boundary mismatches can contribute spectral energy. Matching
energy or spectra alone does not establish matching phase, divergence or dynamics.

## Submission and outputs

From the repository root:

```bash
sbatch jobs/matey-yalamanchi-comparison.sbatch
```

The script is prepared but has not been submitted. It creates
`runs/matey-yalamanchi-comparison-JOBID/`. The directory must be new.

```text
experiment.json, matey.yaml, yalamanchi.yaml
matey.log, yalamanchi.log
matey-smoke.json, yalamanchi-smoke.json
comparison.json, comparison.csv, energy-comparison.png
matey/ and yalamanchi/:
    best.pt, last.pt
    config.json, normalization.json, environment.json, metrics.jsonl
    evaluation-test-5.json, first-forecast.npz
    energy-data.npz, energy-trajectories.csv, energy-spectra.csv
    energy-evolution.png, energy-example.png, energy-spectra.png
```

`comparison.csv` has ten rows: five test leads per model. It reports MSE, MAE,
normalized MSE, aggregate relative L2, persistence skill and energy, with selected
checkpoint epochs. There is no separate all-pair one-step evaluation; lead one
is evaluated on the common five-step test windows. `energy-trajectories.csv`
contains energies/time per forecast origin. `energy-data.npz` retains per-case
energy curves and mean spectra. `first-forecast.npz` retains full input, target
and predicted velocity volumes for the example.

## Runtime checks and failure behavior

Each worker first executes one configured five-step training batch, including
backward and an AdamW update, on its allocated device. This catches memory and
finite-gradient problems and records time/peak allocated GPU memory. Smoke-test
weights are discarded; real training reseeds and initializes afresh. No settings
are silently reduced after a failure.

The nine-hour limit covers preparation, smoke tests, training, test forecasting
and plots. It is an allocation budget, not a measured promise that 100 MATEY
epochs will fit. Full-size GPU timing remains unmeasured. Best/last weights are
saved after completed epochs, so a timeout preserves completed-epoch checkpoints,
but optimizer/RNG state for exact training resume is not stored.

Workers run independently. A caught failure creates `MODEL-failure.json`; the
other worker can finish. Missing reports produce an incomplete comparison and a
nonzero exit status, retaining available results. Slurm timeouts can terminate
the shell before summarization or failure JSON is written; inspect the scheduler
logs and model logs as well as checkpoint existence.

If training produced weights but evaluation was interrupted, inside a compute
allocation it can be repeated without retraining:

```bash
python -B -m forecasting.compare_matey evaluate --model matey --output runs/matey-yalamanchi-comparison-JOBID
python -B -m forecasting.compare_matey evaluate --model yalamanchi --output runs/matey-yalamanchi-comparison-JOBID
python -B -m forecasting.compare_matey summarize --output runs/matey-yalamanchi-comparison-JOBID
```

These explicit evaluation commands replace that run's evaluation artifacts, not
its trained weights. A new training run requires a new output directory.

## Verification

The synthetic CPU integration test trains both actual architectures from scratch,
saves/reloads weights, checks common origins and energy/metric agreement,
generates every PNG, and rejects differing actual evaluation data. Analytic tests
cover constant mean flow, a single sinusoidal mode, the Nyquist endpoint, and
Parseval equality on even and odd grids. Job shell syntax is checked locally.
These checks do not measure full-size GPU memory/runtime or forecast quality.
