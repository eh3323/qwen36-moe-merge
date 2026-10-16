#!/bin/bash
# Shared helpers for the eval runners (sourced, not executed).
# Requires: RUN (base|e_swe|e_tb|e_asi|e_hle|mixed|merged_*), BENCH (tb2|swepro|asi|hle).
set -o pipefail
source /scratch/eh3323/apex/env.sh
REPO=$APEX/repo
source $REPO/configs/eval/harness.env
EVAL_PY=$APEX/envs/eval/bin/python
ASI_PY=$APEX/envs/asi/bin/python
HLE_PY=$APEX/envs/hle/bin/python
ENDPOINT_DIR=$APEX/work/endpoints
# ASI tasks copied OUTSIDE the asibench source checkout (eval_setup.sh step asi_tasks): with
# --tasks-dir inside the checkout, asibench switches to "source checkout" mode and looks for task
# venvs in $ASI_REPO/.ai4sci-bench (editable install, needs PyPI) instead of the prewarmed
# ~/.asibench/task-envs, and $ASI_REPO is hidden from the agent sandbox anyway.
ASI_TASKS_ROOT=$APEX/work/asi_runtime
EP_NAME=${ENDPOINT_NAME:-$RUN}   # eval_allinone uses a per-job name so each allocation talks to its own vLLM
export PATH=$APEX/envs/eval/bin:$APEX/tools/bin:$PATH
# The site profile (/share/apps/profile.d/nyu.sh, re-sourced by every non-interactive bash through
# BASH_ENV, including the apptainer wrapper itself) exports APPTAINER_BINDPATH=/scratch,/projects,...:
# every task container would see /scratch (gold patches, reference answers). Drop it for all runners.
unset BASH_ENV APPTAINER_BINDPATH SINGULARITY_BINDPATH APPTAINER_BIND SINGULARITY_BIND

# node_local_dir <name>: per-job dir on node-local disk (/state/partition1 if writable, else /tmp)
node_local_dir() {
  local base d
  for base in /state/partition1 /tmp; do
    d=$base/$USER-$1-${SLURM_JOB_ID:-$$}
    mkdir -p "$d" 2>/dev/null && [ -w "$d" ] && { echo "$d"; return 0; }
  done
  return 1
}

group_of() {  # RUN -> results group dir
  case "$1" in
    base|base_*) echo base ;; mixed) echo mixed ;; merged*) echo merged ;; e_*) echo experts ;;
    *) echo "unknown RUN '$1'" >&2; return 1 ;;
  esac
}

# init_run <run_tag>: sets RUN_ID, GROUP, RAW_DIR (scratch), NORM_DIR (repo results/), LOGDIR
init_run() {
  : "${RUN:?set RUN}" "${BENCH:?set BENCH}"
  GROUP=$(group_of "$RUN") || exit 2
  RUN_ID="${RUN}__${BENCH}__$1"
  RAW_DIR=$APEX/results/$GROUP/$BENCH/$RUN_ID
  NORM_DIR=$REPO/results/$GROUP/$BENCH
  LOGDIR=$APEX/logs/$RUN_ID
  mkdir -p "$RAW_DIR" "$NORM_DIR" "$LOGDIR"
  export RUN BENCH RUN_ID GROUP RAW_DIR NORM_DIR LOGDIR
  echo "[eval] RUN_ID=$RUN_ID RAW_DIR=$RAW_DIR host=$(hostname) job=${SLURM_JOB_ID:-none}"
}

# wait_endpoint: block until $ENDPOINT_DIR/$RUN.env exists and answers /health (max EP_WAIT_SEC).
wait_endpoint() {
  local f=$ENDPOINT_DIR/$EP_NAME.env t0=$SECONDS
  local max=${EP_WAIT_SEC:-86400}
  while true; do
    if [ -f "$f" ]; then
      source "$f"
      if curl -sf -m 10 "http://$ENDPOINT_HOST:$ENDPOINT_PORT/health" >/dev/null; then
        echo "[eval] endpoint up: $API_BASE (model_dir=$MODEL_DIR, serve job $SERVE_JOB_ID)"
        return 0
      fi
    fi
    (( SECONDS - t0 > max )) && { echo "[eval] endpoint $EP_NAME not up after ${max}s" >&2; return 1; }
    sleep 30
  done
}

# start_proxy: local OpenAI-compatible relay that (a) re-resolves the endpoint file per request
# (survives serve-job requeue) and (b) logs the sampling params + usage of every request.
# Sets PROXY_BASE (http://127.0.0.1:PORT/v1) and PROXY_UDS.
start_proxy() {
  # unique per (job, run_id): several runners may share one allocation (eval_allinone)
  local h=$(echo "${SLURM_JOB_ID:-$$}-$RUN_ID" | cksum | cut -d' ' -f1)
  local port=$(( 30000 + h % 20000 ))
  PROXY_UDS=/tmp/$USER-llmproxy-$h.sock
  # interpreter alias without "python" in its name, so agents' `pkill -f python3` cannot hit it (slurm/serve_model.sbatch)
  local py=$APEX/envs/eval/bin/llm-proxy
  [ -e "$py" ] || ln -s "$(readlink -f "$EVAL_PY")" "$py"
  $py -I $REPO/src/eval/llm_proxy.py --endpoint-file $ENDPOINT_DIR/$EP_NAME.env \
    --listen 127.0.0.1:$port --uds $PROXY_UDS --log $RAW_DIR/llm_requests.jsonl \
    > $LOGDIR/proxy.log 2>&1 &
  PROXY_PID=$!
  for _ in $(seq 1 30); do
    curl -sf -m 5 http://127.0.0.1:$port/v1/models >/dev/null && break; sleep 2
  done
  PROXY_BASE=http://127.0.0.1:$port/v1
  echo "[eval] proxy $PROXY_BASE (pid $PROXY_PID)"
}

# net_probe: record compute-node egress / local disk / container runtime facts (smoke runs).
net_probe() {
  local out=$LOGDIR/net_probe.txt
  {
    echo "host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-none}"
    for u in https://pypi.org/simple/pip/ https://files.pythonhosted.org https://astral.sh/uv/0.9.5/install.sh \
             https://registry-1.docker.io/v2/ https://ghcr.io/v2/ https://github.com https://proxy.golang.org \
             https://registry.npmjs.org/ https://huggingface.co https://api.openai.com/v1/models \
             http://archive.ubuntu.com/ubuntu/ http://deb.debian.org/debian/; do
      code=$(curl -s -o /dev/null -m 15 -w '%{http_code}' "$u" 2>/dev/null || echo 000)
      echo "egress $code $u"
    done
    [ -f "$ENDPOINT_DIR/$EP_NAME.env" ] && { source "$ENDPOINT_DIR/$EP_NAME.env";
      echo "endpoint $(curl -s -o /dev/null -m 10 -w '%{http_code}' http://$ENDPOINT_HOST:$ENDPOINT_PORT/health) $ENDPOINT_HOST:$ENDPOINT_PORT"; }
    echo "--- local disks"; df -h /tmp ${SLURM_TMPDIR:-} /state/partition1 2>/dev/null
    echo "--- sandbox root: ${SANDBOX_ROOT:-unset}  resolv.conf:"; grep -v '^#' /etc/resolv.conf
    echo "--- bind env: BASH_ENV=${BASH_ENV:-} APPTAINER_BINDPATH=${APPTAINER_BINDPATH:-}"
    echo "--- apptainer"; apptainer --version
    echo "--- pasta: $(command -v pasta)  unshare-userns: $(unshare --user --map-root-user true && echo ok)"
    echo "--- ip_unprivileged_port_start=$(sysctl -n net.ipv4.ip_unprivileged_port_start)"
  } > "$out" 2>&1
  echo "[eval] net probe -> $out"; grep -E "^egress|^endpoint" "$out"
}

# write_run_meta <json-extra>: provenance for this run (harness config, endpoint, git-less repo hash).
write_run_meta() {
  $EVAL_PY -I - "$RAW_DIR/run_meta.json" "$1" <<'PY'
import json, os, sys, hashlib, pathlib, time
out, extra = sys.argv[1], json.loads(sys.argv[2] or "{}")
repo = pathlib.Path(os.environ["REPO"])
harness = (repo / "configs/eval/harness.env").read_text()
ep = {}
f = pathlib.Path(os.environ["ENDPOINT_DIR"]) / f'{os.environ["EP_NAME"]}.env'
if f.exists():
    for line in f.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1); ep[k] = v.strip("'\"")
meta = {"run_id": os.environ["RUN_ID"], "run": os.environ["RUN"], "bench": os.environ["BENCH"],
        "group": os.environ["GROUP"], "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "host": os.uname().nodename, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "harness_env_sha256": hashlib.sha256(harness.encode()).hexdigest(), "harness_env": harness,
        "endpoint": ep, **extra}
p = pathlib.Path(out)
if p.exists():  # resumed run (e.g. HLE continued on another GPU type): keep where earlier segments ran
    try:
        old = json.loads(p.read_text())
        seg = {k: old.get(k) for k in ("slurm_job", "host", "started", "endpoint", "harness_env_sha256")}
        meta["previous_segments"] = old.get("previous_segments", []) + [seg]
    except Exception:
        pass
p.write_text(json.dumps(meta, indent=1))
PY
}

export REPO ENDPOINT_DIR EP_NAME
