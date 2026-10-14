#!/bin/bash
#SBATCH --job-name=apex-prepull
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --output=/scratch/eh3323/apex/logs/eval/%x-%j.out
# Runs as a 1-GPU job (CPU partitions are jammed); the GPU only hosts the utilization keepalive
# (cluster cancels jobs with <60% GPU util over 2h). Any GPU type works (no --constraint).
# Resumable: finished SIFs are skipped, so just resubmit until "0 failed / all present".
# Pre-build SIFs for TB2 (89 images, ~49 GB) and SWE-Pro HARD-51 (51 images, ~89 GB) into
# $APEX/cache/sif with Harbor's exact naming, so trials never pull at run time.
# Estimate first (no pulls): python -I src/eval/prepull.py estimate --sets tb2,swe51
#   sbatch --export=ALL,SETS=tb2+swe51 slurm/eval_prepull.sbatch        (PAR=4 parallel pulls; '+' because
#   sbatch splits --export on commas)
#   chunked (<2h each, no keepalive needed): ... --time=02:00:00 --export=ALL,SETS=swe51,CHUNK=0/3 ...
# Docker Hub anonymous limit is 100 manifest pulls / 6 h per IP: 89 TB2 images can hit it. If
# failures say "toomanyrequests", resubmit later (existing SIFs are skipped) or log in first.
# Conversion temp space must be on local disk (APPTAINER_TMPDIR); /scratch is NFS.
set -uo pipefail
source /scratch/eh3323/apex/env.sh
SETS=${SETS:-tb2,swe51}; SETS=${SETS//+/,}; PAR=${PAR:-4}; CACHE=$APEX/cache/sif
LOGDIR=$APEX/logs/prepull-${SLURM_JOB_ID:-$$}; mkdir -p "$LOGDIR" "$CACHE"
for d in ${SLURM_TMPDIR:-} /state/partition1 /tmp; do
  [ -n "$d" ] && [ -d "$d" ] && [ -w "$d" ] && { LOCAL=$d; break; }
done
export APPTAINER_TMPDIR=$LOCAL/$USER-apptainer-${SLURM_JOB_ID:-$$}; mkdir -p "$APPTAINER_TMPDIR"
export APPTAINER_CACHEDIR=$LOCAL/$USER-apptcache-${SLURM_JOB_ID:-$$}; mkdir -p "$APPTAINER_CACHEDIR"  # layer blobs: local, discarded
echo "local tmp: $LOCAL"; df -h "$LOCAL"
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  RLLM_GPU_SHARE_DEVICES=all $APEX/envs/serve/bin/python $APEX/work/gpu_keepalive_shared.py > "$LOGDIR/keepalive.log" 2>&1 &
  KEEP_PID=$!
fi
trap 'kill ${KEEP_PID:-} 2>/dev/null; rm -rf "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"' EXIT
PY=$APEX/cache/uv-python/cpython-3.12-linux-x86_64-gnu/bin/python3
$PY -I $APEX/repo/src/eval/prepull.py list --sets "$SETS" --cache "$CACHE" > "$LOGDIR/all.tsv"
if [ -n "${CHUNK:-}" ]; then  # CHUNK=i/n -> every n-th image starting at i
  awk -v i="${CHUNK%/*}" -v n="${CHUNK#*/}" '(NR-1)%n==i' "$LOGDIR/all.tsv" > "$LOGDIR/list.tsv"
else cp "$LOGDIR/all.tsv" "$LOGDIR/list.tsv"; fi
echo "$(wc -l < "$LOGDIR/list.tsv") images; $(cut -f2 "$LOGDIR/list.tsv" | xargs -r ls 2>/dev/null | wc -l) already present"

pull_one() {
  img=$1; sif=$2
  [ -s "$sif" ] && return 0
  (
    flock 9   # same lock file Harbor uses (<sif>.lock)
    [ -s "$sif" ] && exit 0
    tmp="$sif.tmp.prepull.$$"
    t0=$(date +%s)
    if apptainer pull --disable-cache "$tmp" "docker://$img" > "$LOGDIR/$(basename "$sif").log" 2>&1; then
      mv "$tmp" "$sif"; echo "OK   $img $(( $(date +%s) - t0 ))s $(du -h "$sif" | cut -f1)"
    else
      rm -f "$tmp"; echo "FAIL $img (see $LOGDIR/$(basename "$sif").log)"
    fi
  ) 9>"$sif.lock"
}
export -f pull_one; export LOGDIR
# biggest first is not known without the estimate; plain order, PAR in parallel
tr '\t' ' ' < "$LOGDIR/list.tsv" | xargs -P "$PAR" -n 2 bash -c 'pull_one "$0" "$1"' | tee "$LOGDIR/summary.txt"
echo "done: $(grep -c ^OK "$LOGDIR/summary.txt") ok, $(grep -c ^FAIL "$LOGDIR/summary.txt") failed; cache size $(du -sh "$CACHE" | cut -f1)"
