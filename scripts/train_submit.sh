#!/bin/bash
# Submit a LoRA SFT run (run on the cluster login node, from $APEX/repo).
# usage: scripts/train_submit.sh CONFIG [NGPUS=4] [TIME=12:00:00] [--set key=value ...]
#   scripts/train_submit.sh configs/train/e_tb.yaml 4
#   scripts/train_submit.sh configs/train/e_swe.yaml 4 16:00:00 --set train.epochs=1
set -euo pipefail
CONFIG=${1:?config}; NGPUS=${2:-4}; TIME=${3:-12:00:00}
shift $(( $# < 3 ? $# : 3 ))
export CONFIG NGPUS EXTRA_ARGS="$*"
cd /scratch/eh3323/apex/repo
mkdir -p /scratch/eh3323/apex/logs/train
# CONSTRAINT 可覆盖卡型：32K 训练峰值 83GB，RTX Pro 6000(96GB) 也放得下；64K(E-SWE) 需 h200|b200
sbatch ${CONSTRAINT:+--constraint=$CONSTRAINT} --gres=gpu:$NGPUS --ntasks=$NGPUS --gpus-per-task=1 --cpus-per-task=16 --mem=$((NGPUS * 160))G --time=$TIME \
  --job-name=lora-$(basename "$CONFIG" .yaml) --export=ALL slurm/train_lora.sbatch
