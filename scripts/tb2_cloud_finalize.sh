#!/bin/bash
# Final collection on the TB2 VM (scripts/tb2_cloud.sh), after every attempt has finished:
#  1. normalize each attempt to results/<group>/tb2/<run>__tb2__a<k>.jsonl in the VM's repo copy;
#     the final-merge attempt a1 is the union of its two harbor jobs: merged_v4_cloud__tb2__a1 for the tasks it had
#     started by 02:05 EDT 10-16, merged_v4_cloud__tb2__a1b for the other 15 (a1's later duplicates of those are
#     dropped; README §11.21). merged_v4_cloud__tb2__a2 was stopped at 14/89 and is not normalized.
#  2. archive the raw job dirs (trajectories, verifier output, result.json, llm_requests.jsonl, run_meta), the logs
#     and endpoint files to ~/apex/cloud_tb2_raw.tgz; single files > 200 MB (runaway terminal recordings: one trial
#     wrote a 99 GB recording.cast) are left out and listed in cloud_tb2_raw_excluded.txt with their sizes.
#
#   bash scripts/tb2_cloud_finalize.sh normalize
#   bash scripts/tb2_cloud_finalize.sh archive
set -euo pipefail
W=${W:-$HOME/apex}; REPO=$W/repo; PY=$W/envs/eval/bin/python; RAW=$W/results/raw

normalize() {
  for rid in base_cloud__tb2__a1 base_cloud__tb2__a2 e_tb_v2_cloud__tb2__a1 mixed_cloud__tb2__a1; do
    run=${rid%%__*}; att=${rid##*__a}
    case $run in base*) g=base ;; mixed*) g=mixed ;; e_*) g=experts ;; esac
    mkdir -p "$REPO/results/$g/tb2"
    $PY "$REPO/src/eval/normalize.py" harbor --bench tb2 --job-dir "$RAW/$rid/jobs/$rid" --run-id "$rid" \
      --attempt "$att" --out "$REPO/results/$g/tb2/$rid.jsonl"
  done
  local rid=merged_v4_cloud__tb2__a1 tmp; tmp=$(mktemp -d)
  $PY "$REPO/src/eval/normalize.py" harbor --bench tb2 --job-dir "$RAW/$rid/jobs/$rid" --run-id $rid --attempt 1 \
    --out "$tmp/a1.jsonl"
  $PY "$REPO/src/eval/normalize.py" harbor --bench tb2 --job-dir "$RAW/${rid}b/jobs/${rid}b" --run-id $rid --attempt 1 \
    --out "$tmp/a1b.jsonl"
  mkdir -p "$REPO/results/merged/tb2"
  $PY - "$tmp/a1.jsonl" "$tmp/a1b.jsonl" "$W/v4_a1b_tasks.txt" "$REPO/results/merged/tb2/$rid.jsonl" <<'PY'
import json, sys
a1, a1b, tasks_b, out = sys.argv[1:]
split = set(open(tasks_b).read().strip().split(","))
rows = [json.loads(l) for l in open(a1) if json.loads(l)["task_id"] not in split]
rows_b = [json.loads(l) for l in open(a1b)]
assert {r["task_id"] for r in rows_b} == split, "a1b must cover exactly the 15 split tasks"
for r in rows:
    r["extra"]["harbor_job"] = "merged_v4_cloud__tb2__a1"
for r in rows_b:
    r["extra"]["harbor_job"] = "merged_v4_cloud__tb2__a1b"
allr = rows + rows_b
assert len({r["task_id"] for r in allr}) == len(allr) == 89, (len(allr), len({r["task_id"] for r in allr}))
with open(out, "w") as f:
    for r in sorted(allr, key=lambda r: r["task_id"]):
        f.write(json.dumps(r) + "\n")
print(f"{out}: {len(rows)} tasks from a1 + {len(rows_b)} from a1b")
PY
  rm -rf "$tmp"
}

archive() {
  cd "$W"
  find results/raw -type f -size +200M -printf '%s\t%p\n' | sort -rn > cloud_tb2_raw_excluded.txt
  cut -f2 cloud_tb2_raw_excluded.txt > "$W/.tb2_exclude_list"
  tar czf cloud_tb2_raw.tgz --exclude-from="$W/.tb2_exclude_list" results/raw logs endpoints \
    v4_a1b_tasks.txt v4_a1_original_trials.txt cloud_tb2_raw_excluded.txt
  ls -la cloud_tb2_raw.tgz; sha256sum cloud_tb2_raw.tgz
  echo "excluded (> 200 MB):"; awk -F'\t' '{printf "  %.1f GB  %s\n", $1/1e9, $2}' cloud_tb2_raw_excluded.txt
}

"${1:?normalize|archive}"
