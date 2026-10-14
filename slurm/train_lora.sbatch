#!/bin/bash
# LoRA SFT job (single node, NGPUS data-parallel ranks). Submit via scripts/train_submit.sh CONFIG NGPUS [TIME]
#   which sets --gres/--ntasks/--cpus/--mem/--time to match NGPUS and exports CONFIG / NGPUS / EXTRA_ARGS.
# Resumable: periodic checkpoints in $APEX/runs/<run_id>/checkpoints; on SIGTERM (preemption) or SIGUSR1
# (15 min before the time limit) every rank finishes its current step, rank 0 checkpoints, and this script
# requeues the job, which resumes from the latest checkpoint.
#SBATCH --job-name=apex-lora
#SBATCH --account=torch_pr_147_courant
#SBATCH --constraint=h200|b200
#SBATCH --nodes=1
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --requeue
#SBATCH --signal=USR1@900
#SBATCH --open-mode=append
#SBATCH --output=/scratch/eh3323/apex/logs/train/%x-%j.out

set -uo pipefail
source /scratch/eh3323/apex/env.sh
source $APEX/envs/train/bin/activate
cd $APEX/repo

CONFIG=${CONFIG:?set CONFIG=configs/train/<run>.yaml}
NGPUS=${NGPUS:-${SLURM_NTASKS:-1}}
EXTRA_ARGS=${EXTRA_ARGS:-}          # e.g. "--set train.epochs=1 --set train.lr=5e-5"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000))
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/$USER-triton-${SLURM_JOB_ID:-x}   # node-local JIT cache (Triton/FLA kernels)
export NCCL_DEBUG=WARN
mkdir -p $APEX/logs/train

START_TS=$(date +%s); START_ISO=$(date -Iseconds)
RUN_DIR=$(python -m src.train.sft_lora --config "$CONFIG" $EXTRA_ARGS --print-run-dir)
RUN_ID=$(basename $RUN_DIR)
echo "[sbatch] job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} host=$(hostname) config=$CONFIG run=$RUN_ID ngpus=$NGPUS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n $NGPUS
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')

# GPU-hour accounting (one CSV line per job segment; also written if the batch shell gets SIGTERM on preemption)
ACCOUNTED=0
account_gpu_hours () {
  [ $ACCOUNTED -eq 1 ] && return; ACCOUNTED=1
  local END_TS=$(date +%s)
  local GPUH=$(python -c "print(round(($END_TS-$START_TS)*$NGPUS/3600,4))")
  [ -f $APEX/logs/gpu_hours.csv ] || echo "job_id,restart,run_id,config,ngpus,gpu_type,start,end,elapsed_s,gpu_hours,rc,status" > $APEX/logs/gpu_hours.csv
  echo "$SLURM_JOB_ID,${SLURM_RESTART_COUNT:-0},$RUN_ID,$CONFIG,$NGPUS,$GPU_TYPE,$START_ISO,$(date -Iseconds),$((END_TS-START_TS)),$GPUH,${RC:-NA},$1" >> $APEX/logs/gpu_hours.csv
  echo "[sbatch] gpu_hours(this segment)=$GPUH"
}
GOT_TERM=0
trap 'GOT_TERM=1; account_gpu_hours preempted_or_cancelled' TERM

# 1) tokenize + cache (single CPU process pool; no-op if cached)
python -m src.train.sft_lora --config "$CONFIG" $EXTRA_ARGS --prepare-only | tail -n 40
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "[sbatch] prepare failed"; exit 1; }

# 2) train: one task per GPU; each task receives Slurm signals directly (no torchrun agent in between)
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -l 30 \
  > $APEX/logs/train/gpuutil-$SLURM_JOB_ID.csv 2>/dev/null &
SMI_PID=$!
srun --ntasks=$NGPUS --cpus-per-task=$(( ${SLURM_CPUS_ON_NODE:-16} / NGPUS )) --cpu-bind=none \
     --kill-on-bad-exit=1 --export=ALL \
     bash -c 'export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS; \
              exec python -u -m src.train.sft_lora --config "'"$CONFIG"'" '"$EXTRA_ARGS"
RC=$?
kill $SMI_PID 2>/dev/null

STATUS=$(cat $RUN_DIR/EXIT_STATUS 2>/dev/null || echo unknown)
account_gpu_hours "$STATUS"
echo "[sbatch] rc=$RC status=$STATUS"

# 3) requeue when interrupted (time limit / preemption) or crashed (max 3 restarts for crashes)
#    SIGTERM to the batch shell = preemption (Slurm requeues by itself) or user scancel (must NOT requeue)
if [ "$STATUS" = "done" ] || [ $GOT_TERM -eq 1 ]; then
  exit 0
elif [ "$STATUS" = "interrupted" ] || { [ $RC -ne 0 ] && [ ${SLURM_RESTART_COUNT:-0} -lt 3 ]; }; then
  echo "[sbatch] requeueing $SLURM_JOB_ID"
  scontrol requeue $SLURM_JOB_ID
fi
exit $RC
