#!/usr/bin/env python3
"""Parallel variant of src/eval/asi_prewarm.py: one build per unique cache key, N workers (threads; builds are uv subprocesses)."""
import sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from ai4sci_bench.core.task import TaskLoader
from ai4sci_bench.runner.runtime_root import resolve_runtime_root
from ai4sci_bench.runner.task_env import TaskEnvironmentManager

tasks_dir = Path(sys.argv[1]); workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
tasks = TaskLoader(tasks_dir).discover_tasks()
mgr = TaskEnvironmentManager(resolve_runtime_root(tasks_dir))
assert not mgr.source_checkout
by_key = {}
for t in tasks:
    by_key.setdefault(mgr.compute_cache_key(t), t)
print(f"cache_root={mgr.cache_root} tasks={len(tasks)} keys={len(by_key)} workers={workers}", flush=True)

def build(k, t):
    t0 = time.time(); env = mgr.ensure_env(t)
    return k, t.get("id"), env.cache_hit, time.time() - t0

bad = 0
with ThreadPoolExecutor(workers) as ex:
    futs = {ex.submit(build, k, t): (k, t.get("id")) for k, t in by_key.items()}
    for f in as_completed(futs):
        k, tid = futs[f]
        try:
            _, _, hit, dt = f.result()
            print(f"ok   {tid:<55} key={k} hit={hit} {dt:.0f}s", flush=True)
        except Exception as e:
            bad += 1
            print(f"FAIL {tid:<55} key={k} {type(e).__name__}: {str(e)[:600]}", flush=True)
sys.exit(1 if bad else 0)
