#!/bin/bash
#SBATCH --job-name=apex-swepro
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --output=/scratch/eh3323/apex/logs/eval/%x-%j.out
# SWE-bench Pro V2 HARD-51, locked protocol on Harbor/singularity (CPU-only job):
#   phase 1  host-driven mini-swe-agent (mini_toolcall.yaml, step_limit 250, 50-min budget);
#            sandbox started with `--net --network none` (patched singularity backend)
#   phase 2  official patch_replay re-grade of every model.patch on a pristine sandbox
#            (verifier keeps network, as in the protocol) -> authoritative score
#
#   sbatch --job-name=swe-base --export=ALL,RUN=base slurm/eval_swepro.sbatch
#   smoke (2 tasks + net probe):  sbatch --time=3:00:00 --export=ALL,RUN=base,SMOKE=1 slurm/eval_swepro.sbatch
#   oracle (no GPU; gold patches through the re-grade path):  ... --export=ALL,RUN=base,ORACLE=1,SMOKE=1
#   repeat runs: REP=2,3 (default 1); RETRY_ERRORS=1 re-runs RuntimeError trials of an existing REP
RUN=${RUN:?} BENCH=swepro
source /scratch/eh3323/apex/repo/scripts/eval_common.sh
SMOKE=${SMOKE:-0}; ORACLE=${ORACLE:-0}; REP=${REP:-1}
TAG=r${REP}; [ "$SMOKE" = 1 ] && TAG=smoke; [ "$ORACLE" = 1 ] && TAG=oracle
init_run "$TAG"
N_CONCURRENT=${N_CONCURRENT:-51}
export SIF_CACHE=$APEX/cache/sif HOSTPY=$APEX/envs/harbor-hostpy
export STAGING_ROOT=/tmp/$USER-hb-${SLURM_JOB_ID:-$$}-$RUN_ID; mkdir -p "$STAGING_ROOT"
# per-trial writable sandboxes (see patches/harbor_singularity_network_none.patch) on node-local disk
SANDBOX_ROOT=$(node_local_dir sbx-$RUN_ID) || { echo "no node-local dir"; exit 2; }; export SANDBOX_ROOT
export APPTAINER_TMPDIR=$SANDBOX_ROOT/apptainer-tmp; mkdir -p "$APPTAINER_TMPDIR"
export SWE_TASKS SWE_IDS SWE_CONFIG SWE_STEP_LIMIT SWE_BUDGET_SEC SERVED_MODEL_NAME
export PYTHONPATH=$REPO/src/eval:$SWE_REPO/v2/tooling${PYTHONPATH:+:$PYTHONPATH}
export MSWEA_CONFIGURED=true MSWEA_COST_TRACKING=ignore_errors MSWEA_GLOBAL_CONFIG_DIR=$APEX/cache/mswea
trap 'rm -rf "$STAGING_ROOT"; chmod -R u+rwX "$SANDBOX_ROOT" 2>/dev/null; rm -rf "$SANDBOX_ROOT"; kill $PROXY_PID 2>/dev/null' EXIT
[ "$SMOKE" = 1 ] && net_probe

TASK_ARG=(); [ "$SMOKE" = 1 ] && TASK_ARG=(--tasks "$(head -n $SWE_SMOKE_N $SWE_IDS | paste -sd,)")
JOBS_DIR=$RAW_DIR/jobs; AJOB=${RUN_ID}__agent; GJOB=${RUN_ID}__regrade
cd "$RAW_DIR"
# RETRY_ERRORS=1: on resume, drop trials that ended in a RuntimeError (environment failures such as an
# exec server that never came up) and run them again; finished trials are kept as they are. -f replaces
# harbor's default filter (CancelledError: trials cut by preemption / time limit), so it is passed again.
run_harbor() {  # <job_name> <config>
  if [ -f "$JOBS_DIR/$1/config.json" ]; then
    harbor jobs resume -p "$JOBS_DIR/$1" $( [ "${RETRY_ERRORS:-0}" = 1 ] && echo -f RuntimeError -f CancelledError )
  else harbor run -c "$2" -y; fi 2>&1 | tee -a "$LOGDIR/harbor_$1.log"
}

if [ "${REGRADE_ONLY:-0}" = 1 ]; then
  # Re-grade an existing agent job (no model needed). Runs before 2026-10-15 captured model.patch without its final
  # newline (exec-server stdout.strip()), so `git apply` failed on patches ending in added lines: restore the
  # newline (original kept as model.patch.stripped), keep the old regrade as <job>.nonl.<ts>, and re-grade.
  [ -d "$JOBS_DIR/$AJOB" ] || { echo "no agent job $JOBS_DIR/$AJOB"; exit 2; }
  $EVAL_PY -I - "$JOBS_DIR/$AJOB" <<'PY'
import pathlib, sys
fixed = 0
for p in pathlib.Path(sys.argv[1]).glob("*/agent/model.patch"):
    s = p.read_bytes()
    if s and not s.endswith(b"\n"):
        bak = p.with_name("model.patch.stripped")
        if not bak.exists():
            bak.write_bytes(s)
        p.write_bytes(s + b"\n")
        fixed += 1
print(f"[regrade] restored the final newline of {fixed} patches")
PY
  # all trials, not only the 11-23 per run where GNU patch then failed ("malformed patch"): in most others
  # `git apply` rejected the truncated patch and `patch --fuzz=3` applied it with fuzz, which can misplace hunks
  [ -d "$JOBS_DIR/$GJOB" ] && mv "$JOBS_DIR/$GJOB" "$JOBS_DIR/$GJOB.nonl.$(date +%Y%m%dT%H%M%S)"
  $EVAL_PY $REPO/src/eval/harbor_job.py swe_regrade --job-name "$GJOB" --jobs-dir "$JOBS_DIR" \
    --source-job "$JOBS_DIR/$AJOB" -n "$N_CONCURRENT" --out "$RAW_DIR/regrade_config.json" || exit 2
  run_harbor "$GJOB" "$RAW_DIR/regrade_config.json"
elif [ "$ORACLE" = 1 ]; then
  $EVAL_PY $REPO/src/eval/harbor_job.py oracle --dataset swe --job-name "$GJOB" --jobs-dir "$JOBS_DIR" \
    -n "$N_CONCURRENT" "${TASK_ARG[@]}" --out "$RAW_DIR/regrade_config.json" || exit 2
  write_run_meta "{\"oracle\": 1, \"smoke\": $SMOKE}"
  run_harbor "$GJOB" "$RAW_DIR/regrade_config.json"
else
  wait_endpoint || exit 3
  start_proxy
  export OPENAI_API_BASE=$PROXY_BASE OPENAI_BASE_URL=$PROXY_BASE OPENAI_API_KEY=EMPTY
  $EVAL_PY $REPO/src/eval/harbor_job.py swe --job-name "$AJOB" --jobs-dir "$JOBS_DIR" \
    -n "$N_CONCURRENT" "${TASK_ARG[@]}" --out "$RAW_DIR/agent_config.json" || exit 2
  write_run_meta "{\"rep\": $REP, \"smoke\": $SMOKE, \"n_concurrent\": $N_CONCURRENT}"
  run_harbor "$AJOB" "$RAW_DIR/agent_config.json"
  kill $PROXY_PID 2>/dev/null
  $EVAL_PY $REPO/src/eval/harbor_job.py swe_regrade --job-name "$GJOB" --jobs-dir "$JOBS_DIR" \
    --source-job "$JOBS_DIR/$AJOB" -n "$N_CONCURRENT" "${TASK_ARG[@]}" --out "$RAW_DIR/regrade_config.json" || exit 2
  run_harbor "$GJOB" "$RAW_DIR/regrade_config.json"
fi
$EVAL_PY $REPO/src/eval/normalize.py harbor --bench swepro --job-dir "$JOBS_DIR/$GJOB" \
  --agent-job-dir "$JOBS_DIR/$AJOB" --run-id "$RUN_ID" --attempt "$REP" --out "$NORM_DIR/$RUN_ID.jsonl"
echo "[eval] done: $NORM_DIR/$RUN_ID.jsonl"
