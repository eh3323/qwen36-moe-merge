#!/bin/bash
# Merge job (1 GPU used only for fast dense ΔW math; mostly disk-bound: ~67 GB written per merged model,
# minus symlinked untouched shards). Submit via scripts/merge_submit.sh.
#   MODE=experts CONFIG=configs/merge/ties.yaml EXTRA_ARGS="--set global_scale=0.5"   (expert merge)
#   MODE=analysis CONFIG=configs/merge/analysis.yaml                                  (conflict analysis only)
#   MODE=single ADAPTER=$APEX/runs/e_tb_r32/adapter_final OUT=$APEX/models/merged/e_tb_r32  (fold one LoRA)
#   MODE=select CONFIG=configs/merge/select.yaml   (score merge candidates on val loss in memory, nothing written)
#SBATCH --job-name=apex-merge
#SBATCH --account=torch_pr_147_courant
#SBATCH --constraint=h200|b200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --requeue
#SBATCH --output=/scratch/eh3323/apex/logs/merge/%x-%j.out
set -euo pipefail
source /scratch/eh3323/apex/env.sh
source $APEX/envs/train/bin/activate
cd $APEX/repo
export OMP_NUM_THREADS=16
T0=$(date +%s)
case "${MODE:?MODE=experts|analysis|single|select}" in
  experts)  python -u -m src.merge.merge_experts --config "$CONFIG" ${EXTRA_ARGS:-} ;;
  analysis) python -u -m src.merge.merge_experts --config "$CONFIG" --analyze-only ${EXTRA_ARGS:-} ;;
  single)   python -u -m src.merge.merge_lora_to_full --adapter "$ADAPTER" --out "$OUT" ${EXTRA_ARGS:-} ;;
  select)   python -u -m src.merge.merge_select --config "$CONFIG" ${EXTRA_ARGS:-} ;;
esac
echo "$SLURM_JOB_ID,merge-$MODE,${CONFIG:-$ADAPTER},1,$(( $(date +%s) - T0 ))s,$(python -c "print(round(($(date +%s)-$T0)/3600,4))")" >> $APEX/logs/gpu_hours_merge.csv
