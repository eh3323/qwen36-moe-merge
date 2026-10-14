#!/bin/bash
#SBATCH --job-name=apex-data-gpunode
#SBATCH --account=torch_pr_147_courant
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:55:00
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --requeue
#SBATCH --output=/scratch/eh3323/apex/logs/data_build_gpunode_%j.log
# CPU 队列严重拥堵时的备选：借 1 卡 GPU 作业附带的 16 CPU 并行构建 tb/asi/hle（不使用 GPU）。
# 时限 < 2h，处于 Torch 低 GPU 利用率自动取消的宽限期内。等 SWE_JOB 结束后再执行 finalize（mix/verify/summarize）。
# usage: sbatch --export=ALL,EXPERTS="tb asi hle",SWE_JOB=<jobid> slurm/data_build_gpu.sbatch
set -uo pipefail
source /scratch/eh3323/apex/env.sh
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
cd /scratch/eh3323/apex/repo
PY=/scratch/eh3323/apex/envs/data/bin/python
EXPERTS=${EXPERTS:-"tb asi hle"}
pids=()
for e in $EXPERTS; do
  $PY -m src.data.build --expert "$e" --workers 5 > /scratch/eh3323/apex/logs/data_build_${e}_gpunode_${SLURM_JOB_ID}.log 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
echo "builds [$EXPERTS] rc=$rc"; [ $rc -eq 0 ] || exit 1
if [ -n "${SWE_JOB:-}" ]; then
  while squeue -h -j "$SWE_JOB" 2>/dev/null | grep -q .; do sleep 60; done
  sacct -n -X -j "$SWE_JOB" -o State | grep -q COMPLETED || { echo "SWE build $SWE_JOB not COMPLETED"; exit 1; }
fi
$PY -m src.data.mix && $PY -m src.data.verify --recount 100 && $PY -m src.data.summarize
