#!/usr/bin/env python3
"""Failed-task list: every (task, run) whose status is not `success`, from the normalized per-task results.

    python3 analysis/make_failed_tasks.py

Reads results/{base,experts,merged,mixed}/<bench>/<run_id>.jsonl (written by src/eval/normalize.py) and writes
  results/failed_tasks.csv   one row per non-success (task, run): run_id, group, bench, task_id, attempt, status,
                             score, wall_s, error (first 200 characters)
  results/failed_tasks_summary.csv   counts per run_id and status (fail / timeout / env_error / pending)
Status values are those of normalize.py: `fail` = finished but not solved (for ASI: score 0, e.g. a hard gate not
passed; partial ASI scores count as `success`), `timeout` = agent timeout, `env_error` = environment / harness failure.
Smoke, oracle and `.failed_*` runs are skipped.
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    rows, counts = [], collections.Counter()
    for path in sorted(glob.glob(os.path.join(REPO, "results", "*", "*", "*.jsonl"))):
        run_id = os.path.basename(path)[: -len(".jsonl")]
        if any(t in run_id for t in ("smoke", "oracle", ".failed")):
            continue
        group = path.split(os.sep)[-3]
        for line in open(path):
            r = json.loads(line)
            st = r.get("status")
            counts[(run_id, group, r.get("bench"), st)] += 1
            if st == "success":
                continue
            err = (r.get("error") or "").replace("\n", " ")[:200]
            rows.append([run_id, group, r.get("bench"), r.get("task_id"), r.get("attempt"), st, r.get("score"),
                         r.get("wall_s"), err])
    out = os.path.join(REPO, "results", "failed_tasks.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "group", "bench", "task_id", "attempt", "status", "score", "wall_s", "error"])
        w.writerows(rows)
    runs = sorted({k[:3] for k in counts})
    statuses = ["success", "fail", "timeout", "env_error", "pending"]
    out2 = os.path.join(REPO, "results", "failed_tasks_summary.csv")
    with open(out2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "group", "bench", "n"] + statuses)
        for run in runs:
            c = [counts[run + (s,)] for s in statuses]
            w.writerow(list(run) + [sum(c)] + c)
    print(f"{out}: {len(rows)} rows; {out2}: {len(runs)} runs")


if __name__ == "__main__":
    main()
