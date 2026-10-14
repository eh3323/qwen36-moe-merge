#!/bin/bash
#SBATCH --job-name=apex-tb2
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --output=/scratch/eh3323/apex/logs/eval/%x-%j.out
# Terminal-Bench 2.0 (local snapshot @2fd12b88) with Harbor terminus-2, ONE attempt per job
# (run k jobs with ATTEMPT=1..k in parallel; Base/merged k=5). CPU-only job; the model is
# served by slurm/serve_model.sbatch (endpoint file $APEX/work/endpoints/$RUN.env).
#
#   sbatch --job-name=tb2-base-a1 --export=ALL,RUN=base,ATTEMPT=1 slurm/eval_tb2.sbatch
#   smoke (2 tasks, net probe):  sbatch --time=6:00:00 --cpus-per-task=64 --mem=200G --export=ALL,RUN=base,SMOKE=1 slurm/eval_tb2.sbatch
#   oracle (no GPU needed; validates singularity+pasta+verifier on the cluster):
#                               sbatch --export=ALL,RUN=base,ORACLE=1,SMOKE=1 slurm/eval_tb2.sbatch
#
# Concurrency: per-trial override stays 32 CPU / 48 GB (nominal). Singularity cannot enforce CPU,
# so CPUs are NOT exclusively reserved; memory is capped per trial by Harbor's watchdog (48 GB).
# N_CONCURRENT defaults to RAM/48GB (max 10); set it explicitly to change.
RUN=${RUN:?} BENCH=tb2
source /scratch/eh3323/apex/repo/scripts/eval_common.sh
SMOKE=${SMOKE:-0}; ORACLE=${ORACLE:-0}; ATTEMPT=${ATTEMPT:-1}
TAG=a${ATTEMPT}; [ "$SMOKE" = 1 ] && TAG=smoke; [ "$ORACLE" = 1 ] && TAG=oracle
init_run "$TAG"
# default: RAM-limited, 48 GB nominal per trial (Harbor memory watchdog cap), at most 10
MEM_GB=$(( ${SLURM_MEM_PER_NODE:-500000} / 1024 ))
N_CONCURRENT=${N_CONCURRENT:-$(( MEM_GB / (TB2_MEMORY_MB / 1024) ))}
[ "$N_CONCURRENT" -gt 10 ] && [ -z "${N_CONCURRENT_FORCE:-}" ] && N_CONCURRENT=10; [ "$N_CONCURRENT" -lt 1 ] && N_CONCURRENT=1
export SIF_CACHE=$APEX/cache/sif HOSTPY=$APEX/envs/harbor-hostpy
export STAGING_ROOT=/tmp/$USER-hb-${SLURM_JOB_ID:-$$}-$RUN_ID; mkdir -p "$STAGING_ROOT"
# per-trial writable sandboxes (see patches/harbor_singularity_network_none.patch) on node-local disk
SANDBOX_ROOT=$(node_local_dir sbx-$RUN_ID) || { echo "no node-local dir"; exit 2; }; export SANDBOX_ROOT
export APPTAINER_TMPDIR=$SANDBOX_ROOT/apptainer-tmp; mkdir -p "$APPTAINER_TMPDIR"
export TB2_MAX_INPUT_TOKENS TB2_DIR TB2_AGENT TB2_CPUS TB2_MEMORY_MB TB2_TIMEOUT_SEC SERVED_MODEL_NAME MAX_MODEL_LEN MAX_TOKENS TEMPERATURE TOP_P TOP_K
trap 'rm -rf "$STAGING_ROOT"; chmod -R u+rwX "$SANDBOX_ROOT" 2>/dev/null; rm -rf "$SANDBOX_ROOT"; kill $PROXY_PID 2>/dev/null' EXIT
[ "$SMOKE" = 1 ] && net_probe

TASK_ARG=(); [ "$SMOKE" = 1 ] && TASK_ARG=(--tasks "$(echo $TB2_SMOKE_TASKS | tr ' ' ,)")
JOB=$RUN_ID; JOBS_DIR=$RAW_DIR/jobs
if [ "$ORACLE" = 1 ]; then
  $EVAL_PY $REPO/src/eval/harbor_job.py oracle --dataset tb2 --job-name "$JOB" --jobs-dir "$JOBS_DIR" \
    -n "$N_CONCURRENT" "${TASK_ARG[@]}" --out "$RAW_DIR/job_config.json" || exit 2
else
  wait_endpoint || exit 3
  start_proxy
  export PROXY_BASE HOSTED_VLLM_API_KEY=EMPTY
  $EVAL_PY $REPO/src/eval/harbor_job.py tb2 --job-name "$JOB" --jobs-dir "$JOBS_DIR" -k 1 \
    -n "$N_CONCURRENT" "${TASK_ARG[@]}" --out "$RAW_DIR/job_config.json" || exit 2
fi
write_run_meta "{\"attempt\": $ATTEMPT, \"n_concurrent\": $N_CONCURRENT, \"smoke\": $SMOKE, \"oracle\": $ORACLE}"

cd "$RAW_DIR"
if [ -f "$JOBS_DIR/$JOB/config.json" ]; then   # requeued: continue the same Harbor job
  harbor jobs resume -p "$JOBS_DIR/$JOB" 2>&1 | tee -a "$LOGDIR/harbor.log"
else
  harbor run -c "$RAW_DIR/job_config.json" -y 2>&1 | tee -a "$LOGDIR/harbor.log"
fi
$EVAL_PY $REPO/src/eval/normalize.py harbor --bench tb2 --job-dir "$JOBS_DIR/$JOB" --run-id "$RUN_ID" \
  --attempt "$ATTEMPT" --out "$NORM_DIR/$RUN_ID.jsonl"
echo "[eval] done: $NORM_DIR/$RUN_ID.jsonl"
