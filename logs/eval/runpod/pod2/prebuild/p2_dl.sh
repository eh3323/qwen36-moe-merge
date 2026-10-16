set -euo pipefail
source /scratch/eh3323/apex/env.sh; APEX=/scratch/eh3323/apex
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
date -u +"start %T"
[ -x $APEX/envs/eval/bin/python ] || { uv venv -q --python 3.12 $APEX/envs/eval; VIRTUAL_ENV=$APEX/envs/eval uv pip install -q aiohttp "huggingface_hub[hf_xet]"; }
unset UV_DEFAULT_INDEX
date -u +"eval env %T"
hf=$APEX/envs/eval/bin/hf
( $hf download Qwen/Qwen3.6-35B-A3B --revision 995ad96e --local-dir $APEX/models/Qwen3.6-35B-A3B > $APEX/logs/prebuild/dl_base.log 2>&1; echo "base rc=$? $(date -u +%T)" ) &
( $hf download eh3323/Qwen3.6-35B-A3B-apex-mixed-folded --local-dir $APEX/models/merged/mixed_r32 > $APEX/logs/prebuild/dl_mixed.log 2>&1; echo "mixed rc=$? $(date -u +%T)" ) &
wait
date -u +"done %T"
