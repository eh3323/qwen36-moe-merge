#!/bin/bash
# Create the data venv (uv-managed CPython 3.12). Light: run on the login node (~2-4 min, ~1.5 GB).
set -euo pipefail
source /scratch/eh3323/apex/env.sh
cd /scratch/eh3323/apex/envs
[[ -d data ]] || uv venv --python 3.12 data
export VIRTUAL_ENV=/scratch/eh3323/apex/envs/data
uv pip install "transformers==4.57.1" "tokenizers>=0.22" jinja2 pyyaml "pyarrow==21.0.0" "huggingface_hub>=0.35" \
    "datasets==4.1.1" numpy pandas zstandard xxhash orjson
uv pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    "torch==2.8.0" "sentence-transformers==5.1.1"
$VIRTUAL_ENV/bin/python -c "import transformers, torch, sentence_transformers, pyarrow; print('ok', transformers.__version__, torch.__version__)"
