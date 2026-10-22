# MATEY implementation audit and one-step experiment

Audit performed October 17, 2026. No production model, objective, data split, or
checkpoint was changed. No Slurm job was submitted by this audit.

## Conclusion

No discrepancy was found in the checked adapter, checkpointing, pushforward
history/gradient path, selected cache windows, or evaluation arithmetic.
This narrows the search; it does NOT prove that the implementation is bug-free,
that the GPU execution is equivalent, or that the published results are reproduced.
The source-code architecture may still be a poor fit for this training protocol.

## Executed checks and evidence

- Independently fetched all 13 non-initializer model Python files from ORNL's
  pinned commit `631a4139295d4f51e573d474cff7d6586f3aa310`. Twelve match byte for byte.
  `attention_modules.py` differs only in the three documented dependency imports.
  URLs, SHA256 hashes and exact diff: [source check](upstream-source-check.json).
- Independently instantiated the vendored TurbT and called it directly with the
  same weights, input, random seed, labels and lead conditioning as our adapter.
  Outputs and parameter gradients matched exactly (maximum difference zero).
  Both paths use our SDPA compatibility backend; this is not FlashAttention parity.
- Checkpointing on/off matched outputs, parameter gradients, and one AdamW update
  exactly, with stochastic depth active.
- An independently written list-based pushforward loop matched the production
  loop's output and parameter gradients exactly.
- Verified per-history normalization against an independent mean/std expression
  and reconstruction back to physical units. Checked independently expressed
  channel-standardized loss. The adapter adds neither a second normalization
  nor a last-frame residual. These tests do not prove this normalization is optimal.
- Traced first/middle/last windows from train/validation/test, plus a cross-file
  test window: each field equals its referenced cached frame and each timestamp
  matches the manifest. Counts: 214 train / 80 validation / 353 test windows.
  Frame IDs, times, channel names and dimension names: [audit.json](audit.json).
  Original NetCDF-to-cache coarsening was not independently revalidated.
- Independently accumulated five-step predictions and NumPy normalized MSE using
  a real tiny model; results match the production evaluator to 1e-12 relative tolerance.
- Saved population energy trajectories agree with reported test kinetic energy.
  This is an artifact-consistency check, not a rerun of every full-size test forecast.

Numerical equivalence checks ran on CPU, grid 16³, width 24, three levels,
12 blocks, configured drop_path .2. They use the production classes, but smaller
geometry changes attention grouping and does not exercise all production GPU paths.
343 parameter tensors have gradients; 104 do not in this test. The absent gradients
match upstream exactly and belong to unused masked-normalization branches,
2D upsamplers, and the unused level-zero upsampler. Lists are in audit.json.

## Important inherited details

The configuration value `drop_path=.2` does NOT mean the deepest executed block
has a .2 rate. Upstream reuses its first four schedule entries at every level:
0, .0181818, .0363636, .0545455. This was already correctly recorded in the
architecture document; earlier conversational shorthand was misleading.

Upstream computes normalization statistics without gradients. Its `filterdata`
also runs inside `torch.no_grad()`, including filtering a learned fine-level
correction before subtraction. Consequently autograd omits derivatives through
those filtering/statistics operations. Matching upstream gradients does not mean
we have the unrestricted mathematical derivative of every forward operation.
No attempt was made to change these inherited choices.

Reflection padding, non-shifting local attention groups, and per-history
normalization are retained. Their suitability for periodic DNS remains a
scientific question, independent of adapter correctness.

## Fixed-sequence fitting check

[Curves](fixed-sequence-fitting.png) | [numbers and scope](fixed-sequence-fitting.json)

Both objectives were trained for 50 updates on the SAME fixed training sequence,
with evaluation stochastic depth disabled. This uses width 24 and a stride-4
sample of the 64³ fields (16³), solely for inexpensive debugging. It is not a
new scientific dataset or evidence about production convergence.

One-step training reduced its supervised evaluation NMSE from 1.4152 to 1.1309.
Pushforward reduced fifth-step NMSE from 1.2776 to 1.1717. Both optimize something,
but neither approached zero in this short diagnostic. Thus a strong overfitting
sanity check remains inconclusive; these numbers do not prove adequate learning.

## Actual one-step production results (job 59880460)

Completed all 25 epochs / 5,350 updates in 55m52s. Best epoch 25.
Source: `/pscratch/sd/s/savu/fluid-vit/matey-memory/one-step-59880460/`.
Weights remain there (`best.pt`, `last.pt`); plots, reports, configuration,
learning metrics and first-forecast arrays have been copied here.

| Lead | One-step-trained MATEY test normalized MSE |
|---|---:|
| 1 | .0200934 |
| 2 | .0582781 |
| 3 | .0993236 |
| 4 | .1395777 |
| 5 | .1797581 |

[Learning curves](learning-curves.png) · [Speed contours](velocity-contours.png)

[Energy evolution](energy-evolution.png) · [Energy spectra](energy-spectra.png)

Contours show the first test forecast only: speed on the mid-z plane, common
color scale, and signed speed error. This is not a vector-error plot, and one
slice does not establish population-wide structure preservation.

## Still unresolved

- Full 64³/width-192 GPU output, gradient, optimizer and backend parity.
- Whether longer fixed-sequence training can approach near-zero fitting error.
- Original DNS-to-cache conversion and independent full-checkpoint test rerun.
- A controlled comparison with Yalamanchi on identical forecast origins,
  training exposures and objectives; no causal architecture ranking is established.

Reproduce the CPU audit into a NEW directory:

```bash
module load pytorch/2.8.0
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  .venv/matey-memory-torch28/bin/python -B -m ornl_vit.memory.audit \
  --output runs/matey-audit-repeat
```

The audit script reruns numerical/data/fitting checks and creates figures.
The pinned-source download comparison is separately recorded in
`upstream-source-check.json`; it is not redownloaded by that command.
