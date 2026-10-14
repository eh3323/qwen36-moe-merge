#!/usr/bin/env python3
"""Aggregate normalized per-task JSONL (results/<group>/<bench>/<run_id>.jsonl) into tables.

  summarize.py table   [--results results] [--out results/summary.csv] [--include-smoke]
      one row per (model RUN, benchmark[/level]):
        score       0-100, env_error counted as 0 (primary, official-style)
        score_std   std over attempts (TB2: k runs -> mean +- std, ddof=1); per-attempt scores listed
        score_excl_env  0-100 over non-env-error tasks (sensitivity)
        success/fail/timeout/env_error/pending rates, tokens, wall time
      ASI rows per level (asi_b1..asi_b4) plus asi_overall = mean of the levels present.
      HLE also gets the official RMS calibration error (beta=100, last-bin quirk kept).

  summarize.py compare --a RUN_A --b RUN_B --bench tb2 [--level b3] [--env-policy zero|drop]
      paired bootstrap over tasks shared by both runs (per-task score = mean over attempts):
      mean diff (B-A, points), 95% percentile CI, two-sided bootstrap p, n tasks, env errors.

Run ids follow <RUN>__<bench>__<tag>; tags 'smoke'/'oracle' are excluded unless --include-smoke.
Stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import statistics as st
from collections import Counter, defaultdict

STATUSES = ("success", "fail", "timeout", "env_error", "pending")


def load(results: str, include_smoke: bool) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(os.path.join(results, "*", "*", "*.jsonl"))):
        run_id = os.path.basename(f)[:-6]
        parts = run_id.split("__")
        if len(parts) != 3:
            continue
        run, bench, tag = parts
        if not include_smoke and tag in ("smoke", "oracle"):
            continue
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                r.update(_run=run, _bench=bench, _tag=tag, _group=f.split(os.sep)[-3])
                rows.append(r)
    return rows


def bench_key(r: dict) -> str:
    if r["_bench"] == "asi":
        return f"asi_{(r.get('extra') or {}).get('level') or r['task_id'].split('@')[-1]}"
    return r["_bench"]


def calib_err(conf: list[float], corr: list[float], beta: int = 100) -> float | None:
    """Official hle_eval calib_err(p='2'), including its skip-the-last-bin loop."""
    if len(conf) < beta:
        return None
    idx = sorted(range(len(conf)), key=lambda i: conf[i])
    c = [conf[i] for i in idx]
    k = [corr[i] for i in idx]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(c) // beta)]
    bins[-1] = [bins[-1][0], len(c)]
    err = 0.0
    for lo, hi in bins[:-1]:
        n = hi - lo
        err += n / len(c) * (st.fmean(c[lo:hi]) - st.fmean(k[lo:hi])) ** 2
    return 100 * math.sqrt(err)


def table(a) -> None:
    rows = load(a.results, a.include_smoke)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["_run"], bench_key(r))].append(r)
    out = []
    for (run, bench), rs in sorted(groups.items()):
        by_att: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            by_att[f'{r["_tag"]}#{r.get("attempt", 1)}'].append(r)
        att_scores = []
        for _, ar in sorted(by_att.items()):
            graded = [r for r in ar if r["status"] != "pending"]
            if graded:
                att_scores.append(100 * st.fmean(float(r["score"] or 0) for r in graded))
        n = len(rs)
        cnt = Counter(r["status"] for r in rs)
        non_env = [r for r in rs if r["status"] not in ("env_error", "pending")]
        row = {
            "run": run, "group": rs[0]["_group"], "bench": bench,
            "n_attempts": len(by_att), "n_task_attempts": n,
            "n_tasks": len({r["task_id"] for r in rs}),
            "score": round(st.fmean(att_scores), 2) if att_scores else None,
            "score_std": round(st.stdev(att_scores), 2) if len(att_scores) > 1 else None,
            "attempt_scores": ";".join(f"{s:.2f}" for s in att_scores),
            "score_excl_env": round(100 * st.fmean(float(r["score"] or 0) for r in non_env), 2) if non_env else None,
            **{f"{s}_rate": round(cnt.get(s, 0) / n, 4) for s in STATUSES},
            "tokens_in": sum(r.get("tokens_in") or 0 for r in rs),
            "tokens_out": sum(r.get("tokens_out") or 0 for r in rs),
            "agent_wall_h": round(sum(r.get("wall_s") or 0 for r in rs) / 3600, 2),
            "calibration_error": None,
        }
        if bench == "hle":
            judged = [r for r in rs if r["status"] != "pending" and (r.get("extra") or {}).get("confidence") is not None]
            row["calibration_error"] = calib_err([r["extra"]["confidence"] / 100 for r in judged],
                                                 [float(r["score"]) for r in judged])
        out.append(row)
    # ASI overall = mean over the levels present for that run
    for run in sorted({r["run"] for r in out}):
        lv = [r for r in out if r["run"] == run and r["bench"].startswith("asi_b")]
        if lv:
            out.append({"run": run, "group": lv[0]["group"], "bench": "asi_overall",
                        "n_attempts": None, "n_task_attempts": sum(r["n_task_attempts"] for r in lv),
                        "n_tasks": sum(r["n_tasks"] for r in lv),
                        "score": round(st.fmean(r["score"] for r in lv if r["score"] is not None), 2),
                        "attempt_scores": "levels=" + ",".join(r["bench"][4:] for r in lv)})
    fields = ["run", "group", "bench", "n_attempts", "n_task_attempts", "n_tasks", "score", "score_std",
              "attempt_scores", "score_excl_env", *[f"{s}_rate" for s in STATUSES], "tokens_in", "tokens_out",
              "agent_wall_h", "calibration_error"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow(r)
    for r in out:
        sd = f" +- {r['score_std']}" if r.get("score_std") else ""
        print(f"{r['run']:<12} {r['bench']:<12} score={r['score']}{sd}  env_err={r.get('env_error_rate')}  "
              f"n={r['n_task_attempts']}")
    print(f"-> {a.out}")


def per_task(rows: list[dict], run: str, bench: str, level: str | None, env_policy: str) -> dict:
    acc: dict[str, list[float]] = defaultdict(list)
    env: set[str] = set()
    for r in rows:
        if r["_run"] != run or r["_bench"] != bench or r["status"] == "pending":
            continue
        if level and not r["task_id"].endswith("@" + level):
            continue
        if r["status"] == "env_error":
            env.add(r["task_id"])
            if env_policy == "drop":
                continue
        acc[r["task_id"]].append(float(r["score"] or 0))
    return {t: st.fmean(v) for t, v in acc.items()}, env


def compare(a) -> None:
    rows = load(a.results, a.include_smoke)
    A, envA = per_task(rows, a.a, a.bench, a.level, a.env_policy)
    B, envB = per_task(rows, a.b, a.bench, a.level, a.env_policy)
    tasks = sorted(set(A) & set(B))
    if a.env_policy == "drop":
        tasks = [t for t in tasks if t not in envA and t not in envB]
    if not tasks:
        raise SystemExit("no shared tasks")
    d = [100 * (B[t] - A[t]) for t in tasks]
    rng = random.Random(a.seed)
    n = len(d)
    boots = sorted(st.fmean(d[rng.randrange(n)] for _ in range(n)) for _ in range(a.n_boot))
    lo, hi = boots[int(0.025 * a.n_boot)], boots[int(0.975 * a.n_boot) - 1]
    p = 2 * min(sum(x <= 0 for x in boots), sum(x >= 0 for x in boots)) / a.n_boot
    res = {"bench": a.bench + (f"@{a.level}" if a.level else ""), "A": a.a, "B": a.b, "n_tasks": n,
           "mean_A": round(100 * st.fmean(A[t] for t in tasks), 2),
           "mean_B": round(100 * st.fmean(B[t] for t in tasks), 2),
           "diff_B_minus_A": round(st.fmean(d), 2), "ci95": [round(lo, 2), round(hi, 2)],
           "p_two_sided": round(min(p, 1.0), 4), "env_policy": a.env_policy,
           "env_errors_A": len(envA), "env_errors_B": len(envB), "n_boot": a.n_boot}
    print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("table")
    t.add_argument("--out", default="results/summary.csv")
    c = sp.add_parser("compare")
    c.add_argument("--a", required=True, help="RUN tag, e.g. base")
    c.add_argument("--b", required=True, help="RUN tag, e.g. merged_ties")
    c.add_argument("--bench", required=True, choices=["tb2", "swepro", "asi", "hle"])
    c.add_argument("--level", default=None, help="ASI level, e.g. b3")
    c.add_argument("--env-policy", choices=["zero", "drop"], default="zero")
    c.add_argument("--n-boot", type=int, default=10000)
    c.add_argument("--seed", type=int, default=0)
    for p in (t, c):
        p.add_argument("--results", default="results")
        p.add_argument("--include-smoke", action="store_true")
    a = ap.parse_args()
    table(a) if a.cmd == "table" else compare(a)


if __name__ == "__main__":
    main()
