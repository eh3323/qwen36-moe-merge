#!/bin/bash
# Pull data manifests (ids, sha256, counts, drop stats) from the cluster into data/manifests/ for git.
# Manifests are written on the cluster to $APEX/data/manifests (outside the synced repo, so
# sync_to_cluster.sh --delete can never wipe them).
set -euo pipefail
rsync -az -e "ssh -o ControlPath=~/.ssh/cm-torch" \
  torch:/scratch/eh3323/apex/data/manifests/ "$(dirname "$0")/../data/manifests/"
rsync -az -e "ssh -o ControlPath=~/.ssh/cm-torch" \
  torch:/scratch/eh3323/apex/data/raw/download_record.json "$(dirname "$0")/../data/manifests/download_record.json"
ls -la "$(dirname "$0")/../data/manifests/"
