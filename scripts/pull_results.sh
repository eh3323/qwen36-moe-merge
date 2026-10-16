#!/bin/bash
# Pull normalized per-task results from the cluster (read-only on the cluster side):
#   torch:/scratch/eh3323/apex/repo/results/<group>/<bench>/<run_id>.jsonl  ->  ./results/...
# Only *.jsonl and *.csv are copied (raw trajectories / judge files never leave the cluster).
# smoke/oracle files are pulled too; summarize.py drops them by run-id tag unless --include-smoke.
# No --delete by default (a local file is kept if it disappeared remotely); PULL_DELETE=1 mirrors deletions
# of *.jsonl/*.csv (e.g. after a run was renamed on the cluster). Excluded paths are never deleted.
set -euo pipefail
cd "$(dirname "$0")/.."
SSH="ssh -o ControlPath=$HOME/.ssh/cm-torch -o BatchMode=yes"
SRC="${RESULTS_SRC:-torch:/scratch/eh3323/apex/repo/results/}"
mkdir -p results
DEL=""; [ "${PULL_DELETE:-0}" = 1 ] && DEL=--delete
rsync -az -m $DEL -e "$SSH" \
  --exclude='/summary.csv' --exclude='trajectories/' --exclude='raw/' \
  --include='*/' --include='*.jsonl' --include='*.csv' --exclude='*' \
  "$SRC" results/
# (results/summary.csv is regenerated locally by analysis/make_tables.py, so a cluster copy is not pulled.)
find results -name '*.jsonl' | sort | while read -r f; do printf '%6d  %s\n' "$(wc -l < "$f")" "$f"; done
