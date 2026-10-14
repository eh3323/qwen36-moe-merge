#!/bin/bash
# 将本地仓库同步到集群（不含 .git）
# results/ 由集群上的评测作业写入（NORM_DIR=$REPO/results/...），方向是集群 -> 本地，
# 因此排除在外：否则 --delete 会删掉作业生成的归一化结果。
rsync -az --delete --exclude .git --exclude /results/ -e "ssh -o ControlPath=~/.ssh/cm-torch" "$(dirname "$0")/../" torch:/scratch/eh3323/apex/repo/
