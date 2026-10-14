#!/usr/bin/env python3
"""Pre-build ASI-Bench per-task uv environments (needs network; run once in envs/asi).

The agent phase runs network-locked (no PyPI), so every task venv that
`--sandbox linux_ns` / scoring will ask for must already be in the cache
(~/.asibench/task-envs -> symlinked to /scratch by scripts/eval_setup.sh).
Uses ASI-Bench's own TaskLoader + TaskEnvironmentManager and resolves the runtime root exactly as
`asibench run` does (orchestrator: resolve_runtime_root(tasks_dir)), so cache keys and paths match.
--tasks-dir must be the copy outside the source checkout ($APEX/work/asi_runtime/tasks): inside a
checkout asibench would use <checkout>/.ai4sci-bench/task-envs with an editable install instead.

    $APEX/envs/asi/bin/python -I src/eval/asi_prewarm.py --tasks-dir $APEX/work/asi_runtime/tasks [--only id1,id2]
"""
import argparse
import sys
from pathlib import Path

from ai4sci_bench.core.task import TaskLoader
from ai4sci_bench.runner.runtime_root import resolve_runtime_root
from ai4sci_bench.runner.task_env import TaskEnvironmentManager

ap = argparse.ArgumentParser()
ap.add_argument("--tasks-dir", required=True)
ap.add_argument("--only", default=None)
a = ap.parse_args()
tasks = TaskLoader(Path(a.tasks_dir)).discover_tasks()
if a.only:
    keep = set(a.only.split(","))
    tasks = [t for t in tasks if t.get("id") in keep]
mgr = TaskEnvironmentManager(resolve_runtime_root(Path(a.tasks_dir)))
print(f"cache_root={mgr.cache_root}  source_checkout={mgr.source_checkout}  tasks={len(tasks)}")
if mgr.source_checkout:
    sys.exit(f"--tasks-dir {a.tasks_dir} is inside an asibench source checkout; use the asi_tasks copy")
bad = 0
for t in tasks:
    try:
        env = mgr.ensure_env(t)
        print(f"ok   {t.get('id'):<55} hit={env.cache_hit} key={env.cache_key}")
    except Exception as e:
        bad += 1
        print(f"FAIL {t.get('id'):<55} {type(e).__name__}: {str(e)[:300]}")
sys.exit(1 if bad else 0)
