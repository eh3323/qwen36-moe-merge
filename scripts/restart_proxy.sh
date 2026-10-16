#!/bin/bash
# restart_proxy.sh <RAW_DIR>: run INSIDE the eval job (srun --jobid=<job> --overlap --ntasks=1 bash restart_proxy.sh ...).
# Replaces that run's llm_proxy by a fresh process with identical arguments (same port, UDS and log file), which
# drops every request orphaned by killed harness processes (the old proxy kept them queued and generating; see
# handler_cancellation in src/eval/llm_proxy.py) and picks up the current llm_proxy.py. The new proxy runs in the
# foreground: the srun step (keep it alive, e.g. in tmux on the login node) must last as long as the run.
set -uo pipefail
RAW=${1:?RAW_DIR}
pid=$(pgrep -u "$USER" -f "llm_proxy.py .*--log $RAW/llm_requests.jsonl" | head -1)
[ -n "$pid" ] || { echo "no llm_proxy for $RAW on $(hostname)"; exit 1; }
mapfile -d '' -t CMD < /proc/$pid/cmdline
echo "$(date -Is) $(hostname) job=${SLURM_JOB_ID:-} replacing proxy pid $pid: ${CMD[*]}"
kill "$pid"
for i in $(seq 1 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
kill -9 "$pid" 2>/dev/null
exec "${CMD[@]}"
