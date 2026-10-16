#!/bin/bash
#SBATCH --job-name=swe-regrade
#SBATCH --account=torch_pr_147_courant
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200|b200
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --output=/scratch/eh3323/apex/logs/eval/%x-%j.out
# Re-grade finished SWE-bench Pro runs with the final newline of model.patch restored (REGRADE_ONLY mode of
# slurm/eval_swepro.sbatch; no model is served, the GPU is only how this cluster schedules 16 CPUs quickly).
#   sbatch --export=ALL,RUNS="base:1 base:2 base:3 merged_v4:1 ..." slurm/swe_regrade.sbatch
set -uo pipefail
: "${RUNS:?RUNS=\"<run>:<rep> ...\"}"
for spec in $RUNS; do
  echo "=== $(date -Is) regrade ${spec}"
  RUN=${spec%%:*} REP=${spec#*:} REGRADE_ONLY=1 N_CONCURRENT=${N_CONCURRENT:-16} \
    bash /scratch/eh3323/apex/repo/slurm/eval_swepro.sbatch
done
echo "=== $(date -Is) all done"
