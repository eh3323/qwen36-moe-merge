#!/bin/bash
# ensure_merged.sh <out_dir> <merge config>: build the merged model with src.merge.merge_experts unless
# <out_dir>/merge_info.json already exists (write_folded_model writes <out_dir>.partial and renames it at the end,
# so the file only exists for a complete model). Called by slurm/serve_model.sbatch for late-bound eval jobs; several
# jobs may call it at once: an mkdir lock next to <out_dir> serialises them (a lock older than 45 min is stale).
set -uo pipefail
OUT=${1:?out_dir}; CFG=${2:?merge config}
source /scratch/eh3323/apex/env.sh
LOCK=$OUT.lock
for i in $(seq 1 180); do
  [ -f "$OUT/merge_info.json" ] && { echo "[ensure_merged] ready: $OUT"; exit 0; }
  if mkdir "$LOCK" 2>/dev/null; then
    echo "$(hostname) job=${SLURM_JOB_ID:-none} $(date -Is)" > "$LOCK/owner"
    trap 'rm -rf "$LOCK"' EXIT
    echo "[ensure_merged] building $OUT from $CFG on $(hostname)"
    (cd $APEX/repo && OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} \
      $APEX/envs/train/bin/python -u -m src.merge.merge_experts --config "$CFG")
    rc=$?
    rm -rf "$LOCK"; trap - EXIT
    [ $rc -eq 0 ] && [ -f "$OUT/merge_info.json" ] && { echo "[ensure_merged] built $OUT"; exit 0; }
    echo "[ensure_merged] merge failed (rc=$rc) or wrote a different out_dir than $OUT"; exit 1
  fi
  if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +45 2>/dev/null)" ]; then
    echo "[ensure_merged] stale lock $(cat "$LOCK/owner" 2>/dev/null); removing"; rm -rf "$LOCK"; continue
  fi
  sleep 30
done
echo "[ensure_merged] gave up waiting for $LOCK"; exit 1
