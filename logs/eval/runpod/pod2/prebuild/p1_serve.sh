set -euo pipefail
source /scratch/eh3323/apex/env.sh; APEX=/scratch/eh3323/apex; REPO=$APEX/repo
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
date -u +"start %T"
[ -x $APEX/envs/serve/bin/vllm ] || { uv venv -q --python 3.12 $APEX/envs/serve; VIRTUAL_ENV=$APEX/envs/serve uv pip sync -q \
    $REPO/configs/serve.lock.txt --default-index "${UV_DEFAULT_INDEX:-https://pypi.org/simple}" \
    --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match; }
date -u +"done %T"
