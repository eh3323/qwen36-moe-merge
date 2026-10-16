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
# ASI_SANDBOX=task: hosts that forbid user namespaces (a rented container, scripts/asi_cloud.sh): asibench's
# per-task uv venv (the same prewarmed task envs linux_ns uses) without the unshare wrapper, so no net lock and no
# filesystem hiding; agents talk to this run's proxy directly. Disclosed per run in run_meta.json ("netlock").
# (asibench's --sandbox os is its Docker mode, which a container host cannot run either.)
ASI_SANDBOX=${ASI_SANDBOX:-linux_ns}
API_BASE=http://127.0.0.1:18000/v1; [ "$ASI_SANDBOX" != linux_ns ] && API_BASE=$PROXY_BASE
AGENT_CFG=$(printf '{"model":"%s","api_base":"%s","api_key":"EMPTY","api_protocol":"openai"}' "$SERVED_MODEL_NAME" "$API_BASE")
NETLOCK=true; [ "$ASI_SANDBOX" != linux_ns ] && NETLOCK=false
write_run_meta "{\"levels\": \"$LEVELS\", \"parallel\": $PARALLEL, \"smoke\": $SMOKE, \"agent_config\": $AGENT_CFG, \"netlock\": $NETLOCK, \"sandbox\": \"$ASI_SANDBOX\"}"
unset OPENAI_API_KEY ANTHROPIC_API_KEY  # never visible to agents (linux_ns passes the full env)
export OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_MODELS_FETCH=1 OPENCODE_DISABLE_LSP_DOWNLOAD=1 OPENCODE_DISABLE_SHARE=1
export ASIBENCH_LINUXNS_HIDE="$ASI_FULL:$ASI_NOREF:$ASI_REPO:$ASI_TASKS_ROOT:$REPO:$APEX/work/research:$APEX/data/benchmarks"
export ASIBENCH_LINUXNS_ISOLATE_ROOT=$APEX/results
# asibench creates each agent workspace with tempfile.mkdtemp(): it must live under ISOLATE_ROOT, or the
# linux_ns wrapper refuses to start the agent (env.sh points TMPDIR at $APEX/cache/tmp; 2026-10-15 all instances
# failed with "workspace ... not under ..."). Inside the sandbox only the agent's own workspace is mounted back.
export TMPDIR=$RAW_DIR/tmp; mkdir -p "$TMPDIR"
# Per-agent opencode state: all agents shared one SQLite db under $XDG_DATA_HOME on NFS and concurrent starts
# failed with "database is locked" (2026-10-15 smoke run). The wrapper gives each opencode process its own
# XDG_DATA_HOME/XDG_STATE_HOME in its $TMPDIR (tmpfs inside the sandbox); XDG_CACHE_HOME (prewarmed provider
# packages, needed offline) stays shared. It lives under $APEX/tools, which the sandbox does not hide.
OC_WRAP=$APEX/tools/opencode-wrap/${SLURM_JOB_ID:-$$}-$RUN_ID; mkdir -p "$OC_WRAP"
cat > "$OC_WRAP/opencode" <<EOF
#!/bin/bash
d=\$(mktemp -d "\${TMPDIR:-/tmp}/opencode-state.XXXXXX")
export XDG_DATA_HOME="\$d/data" XDG_STATE_HOME="\$d/state"
exec $APEX/tools/bin/opencode "\$@"
EOF
chmod +x "$OC_WRAP/opencode"; export PATH=$OC_WRAP:$PATH
# --resume skips every instance that has a result file, failed or not: move aside the instances that never
# reached the agent (infrastructure failures), so a resubmission of the same RUN retries exactly those
if [ -d "$RAW_DIR/run" ]; then
  $EVAL_PY -I - "$RAW_DIR/run" <<'PY'
import json, pathlib, shutil, sys, time
run = pathlib.Path(sys.argv[1])
dst = run.parent / f"run.infra_failed.{time.strftime('%Y%m%dT%H%M%S')}"
MARKERS = ("Task sandbox setup failed", "asibench linux_ns:")
moved = 0
for f in run.rglob("*.json"):
    if f.name.endswith(".model_calls.json") or not f.exists():
        continue
    try:
        r = json.loads(f.read_text())
    except Exception:
        continue
    if not isinstance(r, dict) or "instance_id" not in r or r.get("status") != "failed":
        continue
    msg = str((r.get("agent_output") or {}).get("error_message") or "")
    if any(m in msg for m in MARKERS):
        for g in [f, f.with_name(f.name[:-5] + ".model_calls.json")]:
            if g.exists():
                (dst / g.relative_to(run)).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(g), str(dst / g.relative_to(run)))
        moved += 1
print(f"[asi] {moved} infrastructure-failed instances moved to {dst.name} (will be re-run)" if moved else "[asi] no infrastructure-failed instances")
PY
fi
export PROXY_UDS ASI_FULL ASI_NOREF ASI_REPO ASI_TASKS_ROOT ASI_AGENT ASI_TIMEOUT_SEC LEVELS PARALLEL AGENT_CFG RAW_DIR EVAL_PY REPO
export ASI_TASKS_CSV="${TASKS_ARG[1]:-}"
cd "$RAW_DIR"
if [ "$ASI_SANDBOX" != linux_ns ]; then
  T=(); [ -n "$ASI_TASKS_CSV" ] && T=(--tasks "$ASI_TASKS_CSV")
  [ -d "$RAW_DIR/run" ] && T+=(--resume "$RAW_DIR/run")
  # Without a PID namespace the agents can signal asibench itself: 2026-10-15 20:42 EDT an agent's `pkill -f python`
  # ended both runs on a pod. Run it under an interpreter alias without "python" in its name (as vLLM and the proxy,
  # slurm/serve_model.sbatch). asibench keys its task-env cache on sys.executable, so the task envs are rebuilt once
  # under the new key, from the local uv cache (these hosts have network; the linux_ns path below is unchanged).
  ASI_PY=$APEX/envs/asi/bin/asibench-rt
  [ -e "$ASI_PY" ] || ln -s "$(readlink -f "$APEX/envs/asi/bin/python")" "$ASI_PY"
  $ASI_PY $APEX/envs/asi/bin/asibench run --instances-dir "$ASI_NOREF" --tasks-dir "$ASI_TASKS_ROOT/tasks" \
    --agent "$ASI_AGENT" --agent-config "$AGENT_CFG" --sandbox "$ASI_SANDBOX" --timeout "$ASI_TIMEOUT_SEC" \
    --prompt-levels "$LEVELS" --parallel "$PARALLEL" --output-dir "$RAW_DIR/run" "${T[@]}" 2>&1 | tee -a "$LOGDIR/asibench_run.log"
  echo "[eval] agent phase done (sandbox=$ASI_SANDBOX): $RAW_DIR/run ; score with: RUN=$RUN RUN_ID=$RUN_ID scripts/eval_asi_score.sh"
  exit 0
fi
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
  # run the console script with bin/python, not its python3 shebang: asibench keys the task-env cache on
  # sys.executable, and scripts/eval_setup.sh asi_prewarm built the cache with bin/python (2026-10-15: with the
  # shebang every key missed, each task tried to build its venv offline and all 60 instances failed at setup)
  exec $APEX/envs/asi/bin/python $APEX/envs/asi/bin/asibench run --instances-dir "$ASI_NOREF" --tasks-dir "$ASI_TASKS_ROOT/tasks" \
    --agent "$ASI_AGENT" --agent-config "$AGENT_CFG" --sandbox linux_ns --timeout "$ASI_TIMEOUT_SEC" \
    --prompt-levels "$LEVELS" --parallel "$PARALLEL" --output-dir "$RAW_DIR/run" "${T[@]}"
' 2>&1 | tee -a "$LOGDIR/asibench_run.log"
echo "[eval] agent phase done: $RAW_DIR/run ; score with: RUN=$RUN RUN_ID=$RUN_ID scripts/eval_asi_score.sh"
