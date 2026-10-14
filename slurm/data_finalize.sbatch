#!/bin/bash
#SBATCH --job-name=apex-data-final
#SBATCH --account=torch_pr_147_courant
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --requeue
#SBATCH --output=/scratch/eh3323/apex/logs/data_final_%j.log
# Mixed control (union of the 4 train sets, same ids) + verification of all outputs.
set -euo pipefail
source /scratch/eh3323/apex/env.sh
cd /scratch/eh3323/apex/repo
PY=/scratch/eh3323/apex/envs/data/bin/python
$PY -m src.data.mix
$PY -m src.data.verify --recount 100
$PY -m src.data.summarize
