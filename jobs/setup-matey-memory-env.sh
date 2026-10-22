#!/bin/bash
# Run once on a login node. Installs an official prebuilt wheel; no compilation.
set -euo pipefail
cd /global/u1/s/savu/projects/fluid-vit
module load pytorch/2.8.0
envdir="$PWD/.venv/matey-memory-torch28"
if [[ ! -x "$envdir/bin/python" ]]; then
  python -m venv --system-site-packages "$envdir"
fi
"$envdir/bin/python" - <<'PY'
import torch, sys
assert torch.__version__.startswith('2.8.0'), torch.__version__
assert sys.version_info[:2] == (3, 12), sys.version
assert torch._C._GLIBCXX_USE_CXX11_ABI
PY
"$envdir/bin/python" -m pip install --no-deps --no-cache-dir \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
CUDA_VISIBLE_DEVICES='' "$envdir/bin/python" -B - <<'PY'
import torch, flash_attn
from ornl_vit.matey.models.ringX_attn import ringX_varlen_attn_func
print('torch:', torch.__version__, 'flash_attn:', flash_attn.__version__)
print('Authors\' RingX function imported:', ringX_varlen_attn_func.__module__)
PY
