# ORNL MATEY model source

A reading copy of the authors' model implementation, including RingX attention.

- Upstream: https://github.com/ORNL/MATEY
- Pinned commit: `631a4139295d4f51e573d474cff7d6586f3aa310`
- Source directory: https://github.com/ORNL/MATEY/tree/631a4139295d4f51e573d474cff7d6586f3aa310/matey/models
- License: MIT; the original license is included in `LICENSE`.

All files under `matey/models/` are copied verbatim from that commit. Their Git
blob hashes were checked against the upstream tree. No model behavior has been
changed. Existing project models and experiments are untouched.

## Suggested reading order

1. `matey/models/turbt.py`: the hierarchical multiscale Turbulence Transformer.
2. `matey/models/basemodel.py`: shared model setup, tokenization, and output machinery.
3. `matey/models/spatial_modules.py`: spatial patch embedding, reconstruction,
   upsampling, and graph-related spatial operations.
4. `matey/models/spacetime_modules.py`: composition of spatial and temporal blocks.
5. `matey/models/attention_modules.py`: attention blocks and the calls to RingX.
6. `matey/models/ringX_attn.py`: distributed attention forward/backward operations,
   including the variable-length implementation used by the attention modules.

Additional files implement position bias, time/input conditioning, shared layers,
and upstream AViT, sViT, and ViT variants. They are retained to keep the upstream
model directory intact and help explain its shared components.

## Scope and dependencies

This is a source excerpt for reading, not a standalone installable model package.
The original imports into `matey.data_utils` and `matey.utils` are preserved;
those supporting packages are not included. Running these models would require
those upstream helpers and a compatible environment, including PyTorch,
FlashAttention, NumPy, einops, timm, and torch-geometric. RingX uses distributed
PyTorch and private FlashAttention interfaces, so compatibility needs runtime
validation before integration.

No training pipeline, dataset loaders, checkpoints, configurations, or Slurm
launchers are included. No dependencies were installed and no GPU execution was
performed. Python syntax was checked without importing or executing the code.
