#!/bin/bash
# Coefficient ablation sweep (each variant writes a ~67 GB model dir under $APEX/models/merged/<name>;
# untouched shards are symlinked). Edit the lists to taste; then evaluate each dir with the eval track.
set -euo pipefail

cd /scratch/eh3323/apex/repo/scripts
for L in 0.25 0.5 0.75 1.0; do ./merge_submit.sh experts configs/merge/linear.yaml --set global_scale=$L; done
for L in 0.5 1.0 1.5;       do ./merge_submit.sh experts configs/merge/ties.yaml --set global_scale=$L; done
for D in 0.1 0.5;           do ./merge_submit.sh experts configs/merge/ties.yaml --set density=$D; done
for L in 0.5 1.0;           do ./merge_submit.sh experts configs/merge/dare_ties.yaml --set global_scale=$L; done
./merge_submit.sh experts configs/merge/dare_linear.yaml
