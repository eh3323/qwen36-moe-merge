#!/bin/bash
# Submit merge jobs (cluster login node).
#   scripts/merge_submit.sh analysis                                   # conflict analysis of the 4 experts
#   scripts/merge_submit.sh single  <adapter_dir> <out_dir>            # fold one LoRA (experts / mixed) for vLLM
#   scripts/merge_submit.sh experts configs/merge/ties.yaml [--set global_scale=0.5 ...]
#   scripts/merge_submit.sh select  [configs/merge/select.yaml]          # val-loss selection of merge candidates
# CONSTRAINT overrides the GPU type (default h200|b200). Folding/merging only does small per-module
# matmuls (~67 GB disk I/O), so any GPU works, e.g. CONSTRAINT="h200|b200|l40s|rtx6000".
set -euo pipefail
cd /scratch/eh3323/apex/repo; mkdir -p /scratch/eh3323/apex/logs/merge
MODE=${1:?mode}; shift
case $MODE in
  analysis) export MODE CONFIG=${1:-configs/merge/analysis.yaml} EXTRA_ARGS="" ;;
  single)   export MODE ADAPTER=${1:?adapter} OUT=${2:?out} EXTRA_ARGS="" ;;
  experts)  export MODE CONFIG=${1:?config}; shift; export EXTRA_ARGS="$*" ;;
  select)   export MODE CONFIG=${1:-configs/merge/select.yaml}; [ $# -gt 0 ] && shift; export EXTRA_ARGS="$*" ;;
esac
sbatch ${CONSTRAINT:+--constraint=$CONSTRAINT} --job-name=merge-$MODE --export=ALL slurm/merge_job.sbatch
