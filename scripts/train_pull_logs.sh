#!/bin/bash
# Pull run logs (metrics.jsonl, metadata.json, throughput.json, lora_targets.txt) from the cluster into ./logs/<run_id>/
# (run logs live outside the synced repo dir because sync_to_cluster.sh uses rsync --delete).
set -euo pipefail
cd "$(dirname "$0")/.."
rsync -az -e "ssh -o ControlPath=~/.ssh/cm-torch" --include='*/' --include='logs/**' --exclude='*' \
  torch:/scratch/eh3323/apex/runs/ ./.runs_tmp/
for d in ./.runs_tmp/*/ ./.runs_tmp/_throughput/*/; do
  rid=$(basename "$d"); [ -d "$d/logs" ] || continue
  mkdir -p "logs/$rid"; rsync -a "$d/logs/" "logs/$rid/"
done
rm -rf ./.runs_tmp
rsync -az -e "ssh -o ControlPath=~/.ssh/cm-torch" torch:/scratch/eh3323/apex/logs/gpu_hours.csv logs/gpu_hours.csv 2>/dev/null || true
ls logs
