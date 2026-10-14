#!/bin/bash
# Pull conflict-analysis outputs ($APEX/analysis on the cluster) into ./analysis/
cd "$(dirname "$0")/.."
rsync -az -e "ssh -o ControlPath=~/.ssh/cm-torch" torch:/scratch/eh3323/apex/analysis/ ./analysis/
