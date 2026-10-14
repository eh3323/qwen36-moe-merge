#!/bin/bash
#SBATCH --job-name=apex-asi
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --output=/scratch/eh3323/apex/logs/eval/%x-%j.out
# ASI-Bench seed31415, agent phase only (`asibench run`, produce-only), opencode_cli 1.17.15,
# --sandbox linux_ns, --timeout 10800. Scoring (needs the GPT-5.5 judge key) is a separate step:
# scripts/eval_asi_score.sh (login node, key only in your shell).
#
# Isolation (verified on the login node, see patches/asibench_linux_ns_hide.patch):
#  * the whole `asibench run` process tree lives in `unshare --user --map-root-user --mount --net`:
#    loopback only; the agent reaches the model solely via uds_bridge -> llm_proxy Unix socket;
#  * seed31415 (reference/) is covered by an empty tmpfs; each agent additionally sees ONLY its own
#    workspace under $APEX/results, and not the instance dirs / repo / other runs (locked mounts).
#
#   sbatch --job-name=asi-base-b3 --export=ALL,RUN=base,LEVELS=b3 slurm/eval_asi.sbatch     # one level per job
#   smoke (2 tasks, B3, net probe):  sbatch --time=8:00:00 --export=ALL,RUN=base,SMOKE=1 slurm/eval_asi.sbatch
RUN=${RUN:?} BENCH=asi
source /scratch/eh3323/apex/repo/scripts/eval_common.sh
SMOKE=${SMOKE:-0}; LEVELS=${LEVELS:-$ASI_LEVELS_DEFAULT}; LEVELS=${LEVELS//+/,}; PARALLEL=${PARALLEL:-16}
TAG=$(echo "$LEVELS" | tr -d ,); [ "$SMOKE" = 1 ] && { TAG=smoke; LEVELS=b3; PARALLEL=2; }
init_run "$TAG"
[ -d "$ASI_NOREF" ] || { echo "missing $ASI_NOREF (run scripts/eval_setup.sh step asi_noref)"; exit 2; }
[ -d "$ASI_TASKS_ROOT/tasks" ] || { echo "missing $ASI_TASKS_ROOT/tasks (run scripts/eval_setup.sh step asi_tasks)"; exit 2; }
[ "$SMOKE" = 1 ] && net_probe
wait_endpoint || exit 3
start_proxy
trap 'kill $PROXY_PID 2>/dev/null' EXIT
TASKS_ARG=(); [ "$SMOKE" = 1 ] && TASKS_ARG=(--tasks "$ASI_SMOKE_TASKS")
AGENT_CFG=$(printf '{"model":"%s","api_base":"http://127.0.0.1:18000/v1","api_key":"EMPTY","api_protocol":"openai"}' "$SERVED_MODEL_NAME")
write_run_meta "{\"levels\": \"$LEVELS\", \"parallel\": $PARALLEL, \"smoke\": $SMOKE, \"agent_config\": $AGENT_CFG, \"netlock\": true}"
unset OPENAI_API_KEY ANTHROPIC_API_KEY  # never visible to agents (linux_ns passes the full env)
export OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_MODELS_FETCH=1 OPENCODE_DISABLE_LSP_DOWNLOAD=1 OPENCODE_DISABLE_SHARE=1
export ASIBENCH_LINUXNS_HIDE="$ASI_FULL:$ASI_NOREF:$ASI_REPO:$ASI_TASKS_ROOT:$REPO:$APEX/work/research:$APEX/data/benchmarks"
export ASIBENCH_LINUXNS_ISOLATE_ROOT=$APEX/results
export PROXY_UDS ASI_FULL ASI_NOREF ASI_REPO ASI_TASKS_ROOT ASI_AGENT ASI_TIMEOUT_SEC LEVELS PARALLEL AGENT_CFG RAW_DIR EVAL_PY REPO
export ASI_TASKS_CSV="${TASKS_ARG[1]:-}"
cd "$RAW_DIR"
unshare --user --map-root-user --mount --net -- bash -c '
  set -e
  ip link set lo up
  mount -t tmpfs -o size=16k,mode=0755 asibench-hidden "$ASI_FULL"
  $EVAL_PY -I $REPO/src/eval/uds_bridge.py --listen 127.0.0.1:18000 --uds "$PROXY_UDS" > uds_bridge.log 2>&1 &
  sleep 1
  curl -sf -m 10 http://127.0.0.1:18000/v1/models >/dev/null && echo "[asi] model reachable via bridge"
  (curl -s -m 5 https://pypi.org >/dev/null && echo "[asi] WARNING: egress open") || echo "[asi] egress blocked (expected)"
  T=(); [ -n "$ASI_TASKS_CSV" ] && T=(--tasks "$ASI_TASKS_CSV")
  [ -d "$RAW_DIR/run" ] && T+=(--resume "$RAW_DIR/run")   # requeued job: skip finished instances
  exec $APEX/envs/asi/bin/asibench run --instances-dir "$ASI_NOREF" --tasks-dir "$ASI_TASKS_ROOT/tasks" \
    --agent "$ASI_AGENT" --agent-config "$AGENT_CFG" --sandbox linux_ns --timeout "$ASI_TIMEOUT_SEC" \
    --prompt-levels "$LEVELS" --parallel "$PARALLEL" --output-dir "$RAW_DIR/run" "${T[@]}"
' 2>&1 | tee -a "$LOGDIR/asibench_run.log"
echo "[eval] agent phase done: $RAW_DIR/run ; score with: RUN=$RUN RUN_ID=$RUN_ID scripts/eval_asi_score.sh"
