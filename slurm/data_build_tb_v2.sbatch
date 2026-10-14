#!/bin/bash
#SBATCH --job-name=data-tb-v2
#SBATCH --account=torch_pr_147_courant
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --requeue
#SBATCH --output=/scratch/eh3323/apex/logs/data_build_tb_v2_%j.log
# E-TB-v2 side experiment data (DATA ONLY; training is a separate, not-yet-approved step):
# download TermGrade (pinned) -> filter/convert/select (src/data/build_tb_v2.py) -> verify -> render check.
#   sbatch slurm/data_build_tb_v2.sbatch          (STEPS=download+build+verify by default)
set -euo pipefail
source /scratch/eh3323/apex/env.sh
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 HF_HUB_ENABLE_HF_TRANSFER=0
cd /scratch/eh3323/apex/repo
PY=/scratch/eh3323/apex/envs/data/bin/python
STEPS=${STEPS:-download build verify}; STEPS=${STEPS//+/ }   # sbatch --export: use + between steps
for s in $STEPS; do
  echo "=== $s $(date -Is)"
  case $s in
    download) $PY -m src.data.build_tb_v2 download ;;
    build)    $PY -m src.data.build_tb_v2 build --workers 15 ;;
    verify)   $PY -m src.data.verify --experts tb_v2 --recount 50 ;;
    render)   # non-fatal: shows exactly what the trainer would see for one sample
      source /scratch/eh3323/apex/envs/train/bin/activate
      python -m src.train.sft_lora --config configs/train/e_tb_v2.yaml --dry-run-render 1 | head -c 4000 || true ;;
    # per-turn variant (eval-identical thinking context): build -> verify -> byte-exact check -> token cache
    turns)        $PY -m src.data.build_tb_v2 build --per-turn --workers 15 ;;
    verify_turns) $PY -m src.data.verify --experts tb_v2_turns --recount 50 ;;
    check_turns)  /scratch/eh3323/apex/envs/train/bin/python -m src.data.build_tb_v2 check-turns --n-check 40 ;;
    prepare)      source /scratch/eh3323/apex/envs/train/bin/activate
                  python -m src.train.sft_lora --config configs/train/e_tb_v2.yaml --prepare-only 2>&1 | tail -n 25 ;;
  esac
done
echo "=== done $(date -Is)"
