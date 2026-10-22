# ORNL MATEY adaptation for velocity forecasting

This directory contains the authors' model source plus a four-frame, five-step
forecasting pipeline. **This is a source-code adaptation, not an exact paper
reproduction.** Read [the adaptation record](docs/adaptation.md) before using its
results in a paper: it records tokenization, hierarchy, loss, gradient stops,
boundary handling, data selection, and verification limits.

- Upstream: [ORNL/MATEY](https://github.com/ORNL/MATEY).
- New isolated experiments: [checkpointed pushforward baseline and RingX jobs](memory/README.md).
- Pinned commit: `631a4139295d4f51e573d474cff7d6586f3aa310`.
- Original MIT license: [LICENSE](LICENSE).
- Settings: [config.yaml](config.yaml); wrapper: [model.py](model.py).
- Upstream hierarchy: [turbt.py](matey/models/turbt.py).
- Data: [dataset.py](dataset.py); forecast loop: [rollout.py](rollout.py).
- Loss, training, checkpoints and metrics: [engine.py](engine.py).

The configured model uses width 192, 2³ spatial patches, three attention heads,
and four transformer blocks at each of three hierarchy levels. Four full 64³
velocity frames produce one next volume; five recursive calls produce the
sequence. Training averages baseline channel-standardized MSE over five leads.
Registered parameter count: **6,612,393**, including upstream auxiliary modules.

## Source changes and dependencies

Only three imports changed in the existing model source, all in
`matey/models/attention_modules.py`, redirected through [compat.py](matey/compat.py).
The adapter provides stochastic depth without timm, lazy FlashAttention/RingX
imports, and an explicit PyTorch SDPA backend. Existing model forward bodies and
RingX source remain unchanged. Five required upstream helpers were added from
the pinned commit; their hashes are in [upstream-helpers.json](upstream-helpers.json).
Package initializers are minimal local exports.

Dependencies: PyTorch, NumPy, PyYAML, einops, torch-geometric and matplotlib (the
last two are imported by upstream helpers even for our regular-grid task).
`module load pytorch` supplies these in the tested Perlmutter environment.
No packages were installed. The selected backend is `sdpa`; optional `flash`
requires compatible FlashAttention and was not tested. No sequence-parallel
process group is created, so RingX is not executed.

## Running

From the repository root, on an appropriate compute allocation:

```bash
module load pytorch
python -B -m ornl_vit.train --config ornl_vit/config.yaml
```

This creates `runs/ornl-matey-history4-forecast5/`, including `best.pt`, `last.pt`,
configuration, normalization, environment information and epoch metrics. Existing
run directories are rejected. Settings currently specify CUDA, batch size 1 and
100 epochs. Full-size GPU memory/runtime remain unmeasured; no job was submitted.

After training:

```bash
python -B -m ornl_vit.evaluate --checkpoint runs/ornl-matey-history4-forecast5/best.pt --split test --device cuda --output runs/ornl-matey-history4-forecast5/test-forecast5
```

Evaluation writes `report.json` and full volumes for the first forecast in
`first-forecast.npz`. This evaluator covers five steps, not 100.

Small synthetic CPU tests:

```bash
OMP_NUM_THREADS=4 MPLBACKEND=Agg python -B -m unittest ornl_vit.test_baseline -q
```

Seven adaptation tests and 58 existing regression tests passed. A random-weight
64³ one-step CPU forward passed shape/finite checks. These checks do not establish
forecast quality, GPU capacity, or equivalence to published results.
