#!/usr/bin/env python3
"""End-of-project reporting: summary.csv, bootstrap CIs, paste-ready markdown tables and the Delta-CI figure.

    bash scripts/pull_results.sh && python3 analysis/make_tables.py

Inputs : results/<group>/<bench>/<RUN>__<bench>__<tag>.jsonl   (normalized per-task outcomes)
Outputs: results/summary.csv            src/eval/summarize.py table (complete runs only, see below)
         analysis/ci.json               paired bootstrap (compare) and macro bootstrap (macro) results,
                                        env-policy zero (primary) and drop (sensitivity)
         analysis/results_tables.md     score matrix, Delta vs Base, all comparisons, failure rates, per-run scores
         analysis/fig_delta_ci.png      v4-Base and Mixed-Base Delta with 95% CI per benchmark + macro

All statistics come from src/eval/summarize.py (shelled out, several in parallel); this script only decides
which runs are complete, which comparisons are possible, and formats the output.

Incomplete runs: a run file is held back as 'pending' (left out of summary.csv, tables and CIs) when it has
fewer tasks than the benchmark has (tb2 89, swepro 51, hle 2500, asi 60 per level), more than
--max-pending of its rows still pending (e.g. HLE generated but not judged), or more than --max-env of its
rows env_error (e.g. HLE normalized while generation was still running: missing predictions are env_error).
--keep-incomplete disables this. Missing comparisons are recorded as pending, never fail the script.

Stdlib + matplotlib only (matplotlib only for the figure; the rest runs without it).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import glob
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARIZE = os.path.join(REPO, "src", "eval", "summarize.py")
_spec = importlib.util.spec_from_file_location("summarize", SUMMARIZE)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)  # reuse load() / bench_key() so run parsing matches summarize.py exactly

# (summary.csv bench key, summarize.py spec, short label, long label)
BENCHES = [
    ("tb2", "tb2", "TB2", "Terminal-Bench 2.0"),
    ("swepro", "swepro", "SWE-Pro", "SWE-bench Pro HARD-51"),
    ("hle", "hle", "HLE", "HLE"),
    ("asi_b3", "asi@b3", "ASI B3", "ASI-Bench B3"),
]
BKEYS = [b[0] for b in BENCHES]
SPEC = {b[0]: b[1] for b in BENCHES}
SHORT = {b[0]: b[2] for b in BENCHES}
LONG = {b[0]: b[3] for b in BENCHES}
KEY_OF_SPEC = {b[1]: b[0] for b in BENCHES}

MODELS = [("base", "Base"), ("e_swe_v2", "E-SWE v2"), ("e_tb_v2", "E-TB v2"), ("e_asi", "E-ASI"),
          ("e_hle", "E-HLE"), ("mixed", "Mixed"), ("merged_v4", "v4")]
HISTORICAL = [("e_swe", "E-SWE v1 (hist.)"), ("e_tb", "E-TB v1 (hist.)"), ("merged", "v3 (hist.)"),
              ("base_cluster", "Base (cluster TB2)"), ("merged_v4_cluster", "v4 (cluster TB2)"),
              ("base_all", "Base (all TB2 runs)"), ("merged_v4_all", "v4 (all TB2 runs)")]
LABEL = dict(MODELS + HISTORICAL)
ALL4 = set(BKEYS)
PLANNED = {"base": ALL4, "mixed": ALL4, "merged_v4": ALL4, "e_swe_v2": {"swepro"}, "e_tb_v2": {"tb2"},
           "e_asi": {"asi_b3"}, "e_hle": {"hle"}}
EXPECTED_TASKS = {"tb2": 89, "swepro": 51, "hle": 2500, "asi": 60}  # ASI: per prompt level
RUN_TAGS = {"tb2": ["a1", "a2", "a3", "a4", "a5"], "swepro": ["r1", "r2", "r3"]}
# runs planned per (model, bench) for the per-run tables; any other planned model gets one run (a1 / r1)
PLANNED_RUNS = {("base", "tb2"): 2, ("merged_v4", "tb2"): 1, ("base", "swepro"): 3, ("merged_v4", "swepro"): 3}
# TB2 ran in two environments (README §11.21): the cluster (apptainer; Base a1-a2, v4 a1, E-TB v1) and one Nebius VM
# (Docker; RUN names *_cloud: Base x2, v4 x1 (a2 stopped at 14/89), Mixed, E-TB v2). The primary TB2 cells and comparisons use only the
# VM runs, where every model shares node and setup; the cluster runs are kept as <run>_cluster and everything is
# pooled into <run>_all (mean +- std over all runs) for Base and v4.
TB2_POOLED = ("base", "merged_v4")


def staged_names(run: str, bench: str, tag: str) -> list[tuple[str, str]]:
    """(run, tag) names a result file is staged under for summarize.py."""
    if bench != "tb2":
        return [(run, tag)]
    if run.endswith("_cloud"):
        m = run[:-len("_cloud")]
        return [(m, tag)] + ([(m + "_all", "c" + tag)] if m in TB2_POOLED else [])
    if run in TB2_POOLED:
        return [(run + "_cluster", tag), (run + "_all", tag)]
    return [(run, tag)]

# (id, A, B, benches, macro?)  -- every Delta is B - A, i.e. "<B> vs <A>"
COMPARISONS = [
    ("v4_vs_base", "base", "merged_v4", BKEYS, True),
    ("mixed_vs_base", "base", "mixed", BKEYS, True),
    ("v4_vs_mixed", "mixed", "merged_v4", BKEYS, True),
    ("e_swe_v2_vs_base", "base", "e_swe_v2", ["swepro"], False),
    ("e_tb_v2_vs_base", "base", "e_tb_v2", ["tb2"], False),
    ("e_asi_vs_base", "base", "e_asi", ["asi_b3"], False),
    ("e_hle_vs_base", "base", "e_hle", ["hle"], False),
    ("e_swe_v1_vs_base", "base", "e_swe", ["swepro"], False),
    ("e_tb_v1_vs_base", "base_cluster", "e_tb", ["tb2"], False),   # same environment (cluster)
    ("v4_vs_base_tb2_all", "base_all", "merged_v4_all", ["tb2"], False),   # pooled over both TB2 environments
    ("v3_vs_base", "base", "merged", ["hle"], False),
    ("v3_vs_v4", "merged_v4", "merged", ["hle"], False),
]
POLICIES = ("zero", "drop")


# ----------------------------------------------------------------------------------------------- run health
def scan(results: str, a) -> tuple[list[dict], list[dict]]:
    """One record per run file (smoke/oracle skipped); returns (complete, incomplete)."""
    complete, incomplete = [], []
    for f in sorted(glob.glob(os.path.join(results, "*", "*", "*.jsonl"))):
        parts = os.path.basename(f)[:-6].split("__")
        if len(parts) != 3 or parts[2] in ("smoke", "oracle"):
            continue
        run, bench, tag = parts
        rows = [json.loads(line) for line in open(f) if line.strip()]
        for r in rows:
            r.update(_run=run, _bench=bench, _tag=tag)
        cnt = Counter(r["status"] for r in rows)
        n = len(rows)
        per_key: dict[str, set] = defaultdict(set)
        for r in rows:
            per_key[S.bench_key(r)].add(r["task_id"])
        reasons = []
        if n == 0:
            reasons.append("empty file")
        for k, ids in sorted(per_key.items()):
            exp = EXPECTED_TASKS.get(bench)
            if exp and len(ids) < exp:
                reasons.append(f"{k}: {len(ids)}/{exp} tasks")
        if n and cnt.get("pending", 0) / n > a.max_pending:
            reasons.append(f"{cnt['pending']}/{n} rows pending (not graded yet)")
        if n and cnt.get("env_error", 0) / n > a.max_env:
            reasons.append(f"{cnt['env_error']}/{n} rows env_error (run unfinished or broken)")
        rec = {"file": os.path.join("results", os.path.relpath(f, results)), "path": f, "run": run, "bench": bench, "tag": tag,
               "bench_keys": sorted(per_key), "n_rows": n, "status_counts": dict(cnt),
               "mtime": dt.datetime.fromtimestamp(os.path.getmtime(f)).isoformat(timespec="seconds")}
        if reasons and not a.keep_incomplete:
            rec["reasons"] = reasons
            incomplete.append(rec)
        else:
            if reasons:
                rec["warnings"] = reasons
            complete.append(rec)
    return complete, incomplete


def stage(complete: list[dict], results: str, tmp: str) -> str:
    """Symlink the complete run files into tmp/<group>/<bench>/ (under their staged names, see staged_names) so
    summarize.py sees only those."""
    for rec in complete:
        d = os.path.join(tmp, os.path.dirname(os.path.relpath(rec["path"], results)))
        os.makedirs(d, exist_ok=True)
        for run, tag in staged_names(rec["run"], rec["bench"], rec["tag"]):
            os.symlink(os.path.abspath(rec["path"]), os.path.join(d, f"{run}__{rec['bench']}__{tag}.jsonl"))
    return tmp


# ----------------------------------------------------------------------------------------------- summarize.py
def run_summarize(args: list[str]) -> dict:
    p = subprocess.run([sys.executable, SUMMARIZE, *args], cwd=REPO, capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": (p.stderr.strip() or p.stdout.strip() or f"exit {p.returncode}").splitlines()[-1]}
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"error": "unparseable output: " + p.stdout[:200]}


def read_summary(path: str) -> dict:
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                if v == "":
                    r[k] = None
                else:
                    try:
                        r[k] = float(v) if k not in ("run", "group", "bench", "attempt_scores") else v
                    except ValueError:
                        pass
            out[(r["run"], r["bench"])] = r
    return out


def per_run_scores(staged: str) -> dict:
    """(run, bench_key) -> [(tag, score, n_env, n_timeout, n_pending)]; same per-attempt score as summarize table."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in S.load(staged, False):
        groups[(r["_run"], S.bench_key(r), r["_tag"], r.get("attempt", 1))].append(r)
    multi = Counter((run, bk, tag) for run, bk, tag, _ in groups)
    out: dict[tuple, list] = defaultdict(list)
    for (run, bk, tag, att), rs in sorted(groups.items()):
        graded = [r for r in rs if r["status"] != "pending"]
        c = Counter(r["status"] for r in rs)
        score = 100 * sum(float(r["score"] or 0) for r in graded) / len(graded) if graded else None
        label = tag if multi[(run, bk, tag)] == 1 else f"{tag}#{att}"
        out[(run, bk)].append((label, score, c.get("env_error", 0), c.get("timeout", 0), c.get("pending", 0)))
    return out


def build_jobs(avail: set, a) -> tuple[dict, list]:
    """ci skeleton + list of (path-in-ci, summarize args) jobs for the comparisons that are possible now."""
    ci, jobs = {}, []
    common = ["--n-boot", str(a.n_boot), "--seed", str(a.seed)]
    for cid, A, B, benches, do_macro in COMPARISONS:
        missing = {}
        for bk in benches:
            miss = [r for r in (A, B) if (r, bk) not in avail]
            if miss:
                missing[bk] = miss
        used = [bk for bk in benches if bk not in missing]
        entry = {"A": A, "B": B, "label": f"{LABEL.get(B, B)} vs {LABEL.get(A, A)}", "diff": "B - A (points, 0-100)",
                 "benches_requested": [SPEC[b] for b in benches], "benches_used": [SPEC[b] for b in used],
                 "benches_missing": {SPEC[b]: "pending: no complete run for " + ", ".join(m)
                                     for b, m in missing.items()}}
        for pol in POLICIES:
            entry[pol] = {"per_bench": {SPEC[b]: None for b in benches}}
            if do_macro:
                entry[pol]["macro"] = None
            for bk in used:
                bench, _, level = SPEC[bk].partition("@")
                args = ["compare", "--a", A, "--b", B, "--bench", bench, "--env-policy", pol, *common]
                if level:
                    args += ["--level", level]
                jobs.append(((cid, pol, "per_bench", SPEC[bk]), args))
            if do_macro and used:
                jobs.append(((cid, pol, "macro", None),
                             ["macro", "--a", A, "--b", B, "--benches", ",".join(SPEC[b] for b in used),
                              "--env-policy", pol, *common]))
        ci[cid] = entry
    return ci, jobs


# ----------------------------------------------------------------------------------------------- formatting
def f2(x, sign=False):
    if x is None:
        return "—"
    s = f"{x:+.2f}" if sign else f"{x:.2f}"
    return s.replace("-", "−")


def fp(p, n_boot):
    if p is None:
        return "—"
    return f"<{2 / n_boot:.4f}" if p == 0 else f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


def delta_cell(r: dict | None, n_boot: int, with_n=True) -> str:
    if r is None:
        return "pending"
    if "error" in r:
        return "error: " + r["error"]
    d = r.get("diff_B_minus_A", r.get("macro_diff_B_minus_A", r.get("diff")))
    ci = r.get("ci95", r.get("macro_ci95"))
    p = r.get("p_two_sided", r.get("macro_p_two_sided"))
    ps = fp(p, n_boot)
    s = f"{f2(d, True)} [{f2(ci[0], True)}, {f2(ci[1], True)}], p{ps if ps.startswith('<') else '=' + ps}"
    if with_n and "n_tasks" in r:
        s += f" (n={r['n_tasks']})"
    return s


def score_cell(row: dict | None) -> str:
    if not row or row.get("score") is None:
        return None
    k = int(row["n_attempts"] or 1)
    if k > 1 and row.get("score_std") is not None:
        return f"{row['score']:.2f} ± {row['score_std']:.2f} (n={k})"
    return f"{row['score']:.2f} (n={k})"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


# ----------------------------------------------------------------------------------------------- markdown
def write_md(path, summ, runs, ci, incomplete, complete, a, generated):
    nb = a.n_boot
    pending_files = {(n, k) for r in incomplete for k in r["bench_keys"]
                     for n, _ in staged_names(r["run"], r["bench"], r["tag"])}
    pending_tags = {(n, k, t) for r in incomplete for k in r["bench_keys"]
                    for n, t in staged_names(r["run"], r["bench"], r["tag"])}
    L = [f"<!-- generated by analysis/make_tables.py at {generated}; do not edit by hand -->",
         "# Results tables", "",
         f"Scores are 0–100 (primary metric: env_error counted as 0). Generated {generated} from "
         f"{len(complete)} complete run files; {len(incomplete)} incomplete file(s) held back as pending "
         f"(listed at the end). CIs: paired bootstrap over tasks, {nb} resamples, seed {a.seed}, 95% percentile; "
         "p is the two-sided bootstrap p. Δ is always row-model − reference.", ""]

    # (a) score matrix
    L += ["## (a) Score matrix", "",
          "mean ± std over runs (ddof=1) with the number of runs; SWE = 3 runs (r1–r3) where available. TB2 = the "
          "runs on the Nebius VM only (same node and setup for every model; Base 2 runs, v4 1); all TB2 runs "
          "including the cluster ones are pooled in (d). “—” = not evaluated by design; “pending” = planned, no "
          "complete run yet.", ""]
    hdr = ["Benchmark", *[m[1] for m in MODELS]]
    rows = []
    for bk in BKEYS:
        row = [LONG[bk]]
        for run, _ in MODELS:
            c = score_cell(summ.get((run, bk)))
            if c is None:
                planned = bk in PLANNED.get(run, ())
                c = ("pending (partial run)" if (run, bk) in pending_files else "pending") if planned else "—"
            row.append(c)
        rows.append(row)
    macro_row = ["**Macro average (4 benchmarks)**"]
    for run, _ in MODELS:
        if PLANNED.get(run) != ALL4:
            macro_row.append("—")
            continue
        vals = [summ.get((run, bk), {}).get("score") for bk in BKEYS]
        miss = [SHORT[bk] for bk, v in zip(BKEYS, vals) if v is None]
        macro_row.append(f"**{sum(vals) / 4:.2f}**" if not miss else "pending (missing " + ", ".join(miss) + ")")
    rows.append(macro_row)
    L += [md_table(hdr, rows), ""]
    hist = [f"{lbl}: {SHORT[bk]} {score_cell(summ.get((run, bk)))}" for run, lbl in HISTORICAL for bk in BKEYS
            if score_cell(summ.get((run, bk)))]
    if hist:
        L += ["Historical runs (not in the matrix): " + "; ".join(hist) + ".", ""]
    cal = [f"{LABEL.get(run, run)} {r['calibration_error']:.1f}" for (run, bk), r in sorted(summ.items())
           if bk == "hle" and r.get("calibration_error") is not None]
    if cal:
        L += ["HLE RMS calibration error (official hle_eval, β=100): " + ", ".join(cal) + ".", ""]

    # (b) Delta vs Base
    def bsec(pol, title):
        out = [title, ""]
        cols = [("v4_vs_base", "v4 − Base"), ("mixed_vs_base", "Mixed − Base"), ("v4_vs_mixed", "v4 − Mixed")]
        rows = []
        for bk in BKEYS:
            rows.append([LONG[bk], *[delta_cell(ci[c][pol]["per_bench"][SPEC[bk]], nb) for c, _ in cols]])
        mrow = ["**Macro**"]
        for c, _ in cols:
            e = ci[c]
            cell = delta_cell(e[pol]["macro"], nb, with_n=False)
            used = [SHORT[KEY_OF_SPEC[s]] for s in e["benches_used"]]
            if e[pol]["macro"] is not None and len(used) < len(BKEYS):
                cell += f" — partial: over {', '.join(used)} only"
            mrow.append(cell)
        rows.append(mrow)
        out += [md_table(["Benchmark", *[c[1] for c in cols]], rows), ""]
        return out
    L += bsec("zero", "## (b) Δ vs Base, 95% CI and p (env_error = 0, primary)")
    L += ["Per-benchmark cells: `summarize.py compare` (paired over shared tasks; per-task score = mean over that "
          "model's runs). Macro: `summarize.py macro` (equal weight per benchmark; tasks resampled within each "
          "benchmark). The macro uses only benchmarks where both models have a complete run.", ""]
    L += bsec("drop", "### (b′) Sensitivity: tasks with an env_error in either model dropped")

    # (b2) all comparisons
    L += ["### (b2) All comparisons (per benchmark)", "",
          "McNemar = exact test on discordant pairs, only when both sides have one binary outcome per task "
          "(single runs); otherwise —.", ""]
    rows = []
    for cid, A, B, benches, _ in COMPARISONS:
        e = ci[cid]
        for bk in benches:
            r = e["zero"]["per_bench"][SPEC[bk]]
            rd = e["drop"]["per_bench"][SPEC[bk]]
            if r is None or "error" in r:
                rows.append([e["label"], LONG[bk], "—", "—", "—",
                             "pending" if r is None else "error: " + r["error"], "—", "—", "—"])
                continue
            mc = r.get("mcnemar")
            pm = None if not mc else ("<1e-6" if mc["p_exact"] == 0 else f"{mc['p_exact']:.1e}"
                                      if mc["p_exact"] < 0.001 else f"{mc['p_exact']:.3f}")
            rows.append([e["label"], LONG[bk], r["n_tasks"], f2(r["mean_A"]), f2(r["mean_B"]),
                         delta_cell(r, nb, with_n=False),
                         f"{mc['A_only']}:{mc['B_only']}, p{pm if pm.startswith('<') else '=' + pm}" if mc else "—",
                         f"{r['env_errors_A']} / {r['env_errors_B']}",
                         delta_cell(rd, nb, with_n=False) if rd and "error" not in rd else "—"])
    L += [md_table(["Comparison (B vs A)", "Benchmark", "n tasks", "A", "B", "Δ = B − A [95% CI], p",
                    "McNemar (A-only:B-only)", "env_err A / B", "Δ, env_error dropped"], rows), ""]

    # (c) failure rates
    L += ["## (c) Failure rates and cost per model × benchmark", "",
          "From results/summary.csv. Rates are over all task-attempts of all runs; “excl. env” rescores without "
          "env_error tasks.", ""]
    rows = []
    for run, lbl in MODELS + HISTORICAL:
        for bk in BKEYS:
            r = summ.get((run, bk))
            if not r:
                continue
            nt = r["n_task_attempts"] or 0
            tin, tout = r.get("tokens_in") or 0, r.get("tokens_out") or 0
            tok = f"{tin / nt / 1e3:.1f}k / {tout / nt / 1e3:.1f}k" if nt and (tin or tout) else "—"
            rows.append([lbl, LONG[bk], int(r["n_attempts"]), int(nt), f2(r["score"]), f2(r["score_excl_env"]),
                         pct(r["env_error_rate"]), pct(r["timeout_rate"]), pct(r["fail_rate"]),
                         pct(r["pending_rate"]), tok, f2(r.get("agent_wall_h")) if r.get("agent_wall_h") else "—"])
    L += [md_table(["Model", "Benchmark", "runs", "task-attempts", "score", "score excl. env", "env_error",
                    "timeout", "fail", "pending", "tokens in / out per task", "agent wall (h)"], rows), ""]

    # (d) per-run tables
    for bk, title in (("tb2", "## (d) Per-run scores: Terminal-Bench 2.0 (89 tasks)"),
                      ("swepro", "### (d′) Per-run scores: SWE-bench Pro HARD-51 (51 tasks)")):
        tags = list(RUN_TAGS[bk])
        for (run, k), lst in runs.items():
            if k == bk:
                tags += [t for t, *_ in lst if t not in tags]
        rows = []
        for run, lbl in MODELS + HISTORICAL:
            lst = runs.get((run, bk))
            if not lst and bk not in PLANNED.get(run, ()):
                continue
            d = {t: x for t, *x in (lst or [])}
            r = summ.get((run, bk))
            planned = RUN_TAGS[bk][:PLANNED_RUNS.get((run, bk), 1)] if bk in PLANNED.get(run, ()) else []
            cells = []
            for t in tags:
                if t in d:
                    cells.append(f2(d[t][0]))
                elif (run, bk, t) in pending_tags:
                    cells.append("pending (partial)")
                else:
                    cells.append("pending" if t in planned else "—")
            mean = score_cell(r) if r else ("pending (partial run)" if (run, bk) in pending_files else "pending")
            envs = " / ".join(str(d[t][1]) for t in tags if t in d) or "—"
            tos = " / ".join(str(d[t][2]) for t in tags if t in d) or "—"
            rows.append([lbl, *cells, mean, envs, tos])
        L += [title, "", md_table(["Model", *tags, "mean ± std (n)", "env_error per run", "timeout per run"], rows),
              ""]

    # pending / incomplete
    L += ["## Pending and held-back runs", ""]
    miss = [f"{LABEL[run]} × {LONG[bk]}" for run, _ in MODELS for bk in BKEYS
            if bk in PLANNED.get(run, ()) and (run, bk) not in summ]
    L += ["- Planned cells without a complete run: " + (", ".join(miss) if miss else "none") + "."]
    for rec in incomplete:
        L += [f"- `{rec['file']}` (modified {rec['mtime']}) held back: " + "; ".join(rec["reasons"]) + "."]
    for rec in complete:
        if rec.get("warnings"):
            L += [f"- `{rec['file']}` kept despite: " + "; ".join(rec["warnings"]) + "."]
    pend = [f"`{rec['file']}` ({rec['status_counts']['pending']} row(s))" for rec in complete
            if rec["status_counts"].get("pending")]
    if pend:
        L += ["- Complete runs with a few ungraded (pending) rows: " + ", ".join(pend) + ". summarize.py leaves "
              "pending rows out of that run's score denominator and out of the paired comparisons."]
    L += [""]
    with open(path, "w") as f:
        f.write("\n".join(L))


# ----------------------------------------------------------------------------------------------- figure
def write_fig(path, ci):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#d9d8d4"
    series = [("v4_vs_base", "v4 − Base", "#2a78d6", "o", 0.17),
              ("mixed_vs_base", "Mixed − Base", "#eb6834", "s", -0.17)]
    labels = [LONG[bk] for bk in BKEYS] + ["Macro"]
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": ink, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(6.5, 2.2), dpi=200)
    n = len(labels)
    xs = []
    for cid, name, color, mk, off in series:
        e = ci[cid]["zero"]
        for i, bk in enumerate(BKEYS + ["macro"]):
            r = e["macro"] if bk == "macro" else e["per_bench"][SPEC[bk]]
            y = n - 1 - i + off
            if r is None or "error" in r:
                ax.text(0, y, f"{name}: pending", color=muted, fontsize=5.5, va="center", ha="center",
                        style="italic", zorder=4, bbox=dict(fc="white", ec="none", pad=0.6))
                continue
            d = r.get("diff_B_minus_A", r.get("macro_diff_B_minus_A"))
            lo, hi = r.get("ci95", r.get("macro_ci95"))
            xs += [lo, hi]
            ax.plot([lo, hi], [y, y], color=color, lw=1.4, solid_capstyle="round", zorder=2)
            ax.plot([d], [y], marker=mk, ms=4.5, color=color, mec="white", mew=0.8, zorder=3, ls="none")
            p = r.get("p_two_sided", r.get("macro_p_two_sided"))
            note = ""
            if bk == "macro" and len(ci[cid]["benches_used"]) < len(BKEYS):
                note = "  (" + ", ".join(SHORT[KEY_OF_SPEC[s]] for s in ci[cid]["benches_used"]) + " only)"
            ax.annotate(f"{d:+.1f}".replace("-", "\u2212") + ("*" if p is not None and p < 0.05 else "") + note,
                        (hi, y), xytext=(4, 0), textcoords="offset points", color=ink, fontsize=5.5, va="center",
                        ha="left", zorder=4, bbox=dict(fc="white", ec="none", pad=0.2, alpha=0.85))
    lim = max([abs(x) for x in xs] + [5]) * 1.35
    ax.set_xlim(-lim, lim)
    ax.axvline(0, color=muted, lw=0.8, zorder=1)
    ax.set_yticks(range(n))
    partial = any(len(ci[c]["benches_used"]) < len(BKEYS) for c, *_ in series)
    mlabel = "Macro (partial)" if partial else "Macro (4 benchmarks)"
    ax.set_yticklabels(list(reversed(labels[:-1] + [mlabel])))
    ax.set_ylim(-0.6, n - 0.4)
    ax.xaxis.grid(True, color=grid, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Δ vs Base (pp)")
    ax.set_title("Final merge (v4) and Mixed vs Base", fontsize=8, color=ink, loc="left")
    handles = [plt.Line2D([], [], color=c, marker=m, lw=1.4, ms=4.5, mec="white", label=nm)
               for _, nm, c, m, _ in series]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.16), ncol=2, frameon=False,
              fontsize=6.5, handlelength=1.6)
    fig.text(0.995, 0.01, "95% paired-bootstrap CI; * p<0.05", ha="right", va="bottom", fontsize=5.5,
             color=muted)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(REPO, "results"))
    ap.add_argument("--summary", default=None, help="default: <results>/summary.csv")
    ap.add_argument("--outdir", default=os.path.join(REPO, "analysis"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--max-pending", type=float, default=0.01, help="max fraction of pending rows in a run file")
    ap.add_argument("--max-env", type=float, default=0.5, help="max fraction of env_error rows in a run file")
    ap.add_argument("--keep-incomplete", action="store_true", help="use every run file as is")
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    results = os.path.abspath(a.results)
    summary = a.summary or os.path.join(results, "summary.csv")
    generated = dt.datetime.now().isoformat(timespec="minutes")

    complete, incomplete = scan(results, a)
    for rec in incomplete:
        print(f"pending (held back): {rec['file']}: " + "; ".join(rec["reasons"]), file=sys.stderr)

    with tempfile.TemporaryDirectory(prefix="apex_results_") as tmp:
        staged = stage(complete, results, tmp)
        p = subprocess.run([sys.executable, SUMMARIZE, "table", "--results", staged, "--out", summary],
                           cwd=REPO, capture_output=True, text=True)
        if p.returncode != 0:
            raise SystemExit("summarize.py table failed:\n" + p.stderr)
        print(p.stdout.rstrip())
        summ = read_summary(summary)
        runs = per_run_scores(staged)
        avail = {(run, bk) for (run, bk), r in summ.items() if r.get("score") is not None}
        ci, jobs = build_jobs(avail, a)
        for j, (key, _) in enumerate(jobs):
            jobs[j] = (key, jobs[j][1][:1] + ["--results", staged] + jobs[j][1][1:])
        print(f"running {len(jobs)} bootstrap jobs ({a.n_boot} resamples, {a.jobs} in parallel) ...", file=sys.stderr)
        with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
            futs = {ex.submit(run_summarize, args): key for key, args in jobs}
            for fu in cf.as_completed(futs):
                cid, pol, kind, spec = futs[fu]
                res = fu.result()
                if kind == "macro":
                    ci[cid][pol]["macro"] = res
                else:
                    ci[cid][pol]["per_bench"][spec] = res

    out = {"generated": generated, "results_dir": results if os.path.relpath(results, REPO).startswith("..")
           else os.path.relpath(results, REPO), "n_boot": a.n_boot,
           "seed": a.seed, "primary_env_policy": "zero",
           "notes": "diff = B - A in points (0-100). per_bench: summarize.py compare; macro: summarize.py macro over "
                    "benches_used (benchmarks where both runs are complete). null = pending.",
           "runs_used": [{k: rec[k] for k in ("file", "n_rows", "status_counts", "mtime")} for rec in complete],
           "runs_held_back": [{k: rec[k] for k in ("file", "n_rows", "status_counts", "mtime", "reasons")}
                              for rec in incomplete],
           "comparisons": ci}
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "ci.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    write_md(os.path.join(a.outdir, "results_tables.md"), summ, runs, ci, incomplete, complete, a, generated)
    if not a.no_fig:
        write_fig(os.path.join(a.outdir, "fig_delta_ci.png"), ci)
    errs = [(c, pol, k) for c, e in ci.items() for pol in POLICIES for k, v in e[pol]["per_bench"].items()
            if v and "error" in v]
    for e in errs:
        print("comparison error:", e, file=sys.stderr)
    def show(x):
        rel = os.path.relpath(x, REPO)
        return x if rel.startswith("..") else rel
    print("-> " + ", ".join(show(x) for x in
                            [summary] + [os.path.join(a.outdir, n) for n in ("ci.json", "results_tables.md")]
                            + ([] if a.no_fig else [os.path.join(a.outdir, "fig_delta_ci.png")])))


if __name__ == "__main__":
    main()
