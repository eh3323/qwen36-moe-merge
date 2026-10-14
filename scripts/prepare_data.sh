#!/bin/bash
# Submit the data pipeline as a Slurm dependency chain (run on the cluster login node):
#   prep (download + dev manifests) -> 4 parallel expert builds -> finalize (mixed control + verify)
# usage: bash scripts/prepare_data.sh [--skip-prep]
set -euo pipefail
cd /scratch/eh3323/apex/repo
mkdir -p /scratch/eh3323/apex/logs
dep=""
if [[ "${1:-}" != "--skip-prep" ]]; then
  prep=$(sbatch --parsable slurm/data_prep.sbatch); echo "prep: $prep"; dep="--dependency=afterok:$prep"
fi
ids=()
for e in swe tb asi hle; do
  j=$(sbatch --parsable $dep --job-name=apex-data-$e --export=ALL,EXPERT=$e slurm/data_build.sbatch)
  echo "build $e: $j"; ids+=("$j")
done
fin=$(sbatch --parsable --dependency=afterok:$(IFS=:; echo "${ids[*]}") slurm/data_finalize.sbatch)
echo "finalize: $fin"
echo "monitor: squeue -u \$USER ; tail -f /scratch/eh3323/apex/logs/data_*.log"
