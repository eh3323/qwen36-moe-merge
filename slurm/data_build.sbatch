#!/bin/bash
#SBATCH --job-name=apex-data-build
#SBATCH --account=torch_pr_147_courant
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --requeue
#SBATCH --output=/scratch/eh3323/apex/logs/data_build_%x_%j.log
# usage: sbatch --export=ALL,EXPERT=swe slurm/data_build.sbatch     (EXPERT in swe|tb|asi|hle)
# filter -> decontaminate -> convert -> exact token count -> stratified token-budgeted selection -> split.
# E-HLE streams Nemotron-Math-v2 / OpenScienceReasoning-2 from the HF Hub (needs outbound HTTPS).
set -euo pipefail
: "${EXPERT:?set EXPERT=swe|tb|asi|hle}"
source /scratch/eh3323/apex/env.sh
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd /scratch/eh3323/apex/repo
/scratch/eh3323/apex/envs/data/bin/python -m src.data.build --expert "$EXPERT" --workers 15
