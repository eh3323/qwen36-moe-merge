#!/bin/bash
# Build the TRAINING env at /scratch/eh3323/apex/envs/train  (run manually on the LOGIN node; wheels only,
# nothing is compiled: torch/triton/fla-core/kernels are all prebuilt or pure-Python/Triton).
#   bash scripts/train_setup.sh
# Pinned lock (resolved with `uv pip compile` from configs/train/requirements.in, cu129 torch index):
#   configs/train/train.lock.txt          <- exact versions installed by `uv pip sync`
# After install, the frozen env is written to $APEX/envs/train.freeze.txt; copy it back into the repo with
#   scp -o ControlPath=~/.ssh/cm-torch torch:/scratch/eh3323/apex/envs/train.freeze.txt configs/train/train.freeze.txt
set -euo pipefail
source /scratch/eh3323/apex/env.sh
cd $APEX/repo
ENV=$APEX/envs/train

# 1) uv-managed CPython 3.12 venv
uv python install 3.12
uv venv --python 3.12 --seed $ENV
source $ENV/bin/activate

# 2) exact install from the lock (torch 2.12.1+cu129, transformers 5.19.0, peft 0.21.2, fla 0.5.2, kernels 0.17.2 ...)
uv pip sync configs/train/train.lock.txt \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
# (to re-resolve after editing requirements.in:
#  uv pip compile configs/train/requirements.in --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
#    --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu129 \
#    --index-strategy unsafe-best-match -o configs/train/train.lock.txt)

# 3) record exact env
uv pip freeze > $APEX/envs/train.freeze.txt
python - <<'PY'
import torch, transformers, peft, fla, triton
print("torch", torch.__version__, "cuda", torch.version.cuda, "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| fla", fla.__version__, "| triton", triton.__version__)
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M  # arch present
from fla.ops.gated_delta_rule import chunk_gated_delta_rule        # GDN kernel importable
from fla.modules.conv.causal_conv1d import causal_conv1d
print("grouped_mm:", hasattr(torch.nn.functional, "grouped_mm") or hasattr(torch, "_grouped_mm"))
PY

# 4) optional: pre-fetch prebuilt flash-attn varlen kernels from the HF kernels hub into HF_HUB_CACHE
#    (only used if they pass the numeric self-test at startup; otherwise exact per-sample SDPA is used)
hf download kernels-community/flash-attn3 --revision v1 --include "build/torch-stable-abi210-cu128-x86_64-linux/*" "build/torch-universal/*" >/dev/null 2>&1 \
  && echo "prefetched flash-attn3" || echo "flash-attn3 prefetch skipped"
hf download kernels-community/flash-attn2 --revision v2 --include "build/torch212-cxx11-cu126-x86_64-linux/*" >/dev/null 2>&1 \
  && echo "prefetched flash-attn2" || echo "flash-attn2 prefetch skipped"

echo "train env ready: $ENV   (lock: configs/train/train.lock.txt, freeze: $APEX/envs/train.freeze.txt)"
