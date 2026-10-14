#!/bin/bash
#SBATCH --job-name=apex-data-prep
#SBATCH --account=torch_pr_147_courant
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --requeue
#SBATCH --output=/scratch/eh3323/apex/logs/data_prep_%j.log
# Download pinned raw sources (~15 GB; Nemotron-Math-v2 / OSR-2 are streamed later, not downloaded)
# and write the dev-set manifests (must precede the builds: E-SWE/E-TB exclude dev repos/tasks).
set -euo pipefail
source /scratch/eh3323/apex/env.sh
cd /scratch/eh3323/apex/repo
PY=/scratch/eh3323/apex/envs/data/bin/python
$PY -m src.data.download --workers 8          # idempotent: re-running resumes / skips finished files
$PY -m src.data.dev_sets
