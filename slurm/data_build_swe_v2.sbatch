#!/bin/bash
#SBATCH --job-name=data-swe-v2
#SBATCH --account=torch_pr_147_courant
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --requeue
#SBATCH --output=/scratch/eh3323/apex/logs/data_build_swe_v2_%j.log
# E-SWE-v2 side experiment data: download nvidia/Open-SWE-Traces (pinned) -> filter/convert/select
# (src/data/build_swe_v2.py) -> verify -> render check -> pre-build the training token cache.
#   sbatch slurm/data_build_swe_v2.sbatch          (STEPS=download+build+verify+prepare by default)
set -euo pipefail
source /scratch/eh3323/apex/env.sh
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 HF_HUB_ENABLE_HF_TRANSFER=0
cd /scratch/eh3323/apex/repo
PY=/scratch/eh3323/apex/envs/data/bin/python
STEPS=${STEPS:-download build verify prepare}; STEPS=${STEPS//+/ }   # sbatch --export: use + between steps
for s in $STEPS; do
  echo "=== $s $(date -Is)"
  case $s in
    download) $PY -m src.data.build_swe_v2 download ;;
    build)    $PY -m src.data.build_swe_v2 build --workers 15 ;;
    verify)   $PY -m src.data.verify --experts swe_v2 --recount 50 ;;
    prepare)  # non-fatal: the training job re-runs --prepare-only anyway (cache key = file sha256)
      source /scratch/eh3323/apex/envs/train/bin/activate
      python -m src.train.sft_lora --config configs/train/e_swe_v2.yaml --dry-run-render 1 | head -c 4000 || true
      python -m src.train.sft_lora --config configs/train/e_swe_v2.yaml --prepare-only 2>&1 | tail -n 25 || true ;;
  esac
done
echo "=== done $(date -Is)"
