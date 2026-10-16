set -euo pipefail
source /scratch/eh3323/apex/env.sh; APEX=/scratch/eh3323/apex; REPO=$APEX/repo
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple USER=${USER:-root}
date -u +"start %T"
bash $REPO/scripts/eval_setup.sh asi opencode asi_tasks
date -u +"asi/opencode/tasks done %T"
cd $APEX/work/asi_runtime && $APEX/envs/asi/bin/python -I $APEX/logs/prebuild/prewarm_par.py $APEX/work/asi_runtime/tasks 8
date -u +"done %T"
