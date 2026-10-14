"""Independent dev sets (checkpoint selection only; never the target benchmarks).

  SWE : 50 SWE-rebench-V2 tasks, language mix ~ HARD-51 (20 go / 20 python / 10 ts+js); their repos are
        excluded from E-SWE training (build.py reads dev_swe.csv). Prefers repos absent from the
        trajectory corpora so the hold-out costs little training data.
  TB  : OpenThoughts-TBLite (100) minus grpc-plant-position-server (near-dup of TB2 kv-store-grpc) and any
        name in the TB1-unsafe/TB2 list.
  ASI : 35 ScienceAgentBench tasks, round-robin over domains.
  HLE : GPQA-Diamond (198; gated, only ids + sha256 of the question go in git) + 300 held-out items written
        by build.py --expert hle (dev_hle_heldout.csv).
Run BEFORE build.py (SWE and TB builds read these manifests).
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

from .common import APEX, CFG, guard_path, log, raw_dir, sha256_text, stable_hash01
from .decontam import swe_repo_verdict, tb_unsafe_names

MANI = APEX / "data" / "manifests"


def _w(name, header, rows):
    MANI.mkdir(parents=True, exist_ok=True)
    with open(MANI / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    log(f"[dev] {name}: {len(rows)} rows")


def train_repo_counts() -> Counter:
    import pyarrow.parquet as pq
    c = Counter()
    for src, pat in (("swe_rebench_oh", "trajectories.parquet"), ("swe_hero_oh", "data/*.parquet")):
        for p in sorted(raw_dir(src).glob(pat)):
            c.update(x.lower() for x in pq.read_table(guard_path(p), columns=["repo"]).column(0).to_pylist())
    return c


def swe_dev(n_go=20, n_py=20, n_ts=10):
    import ast
    import pyarrow.parquet as pq
    t = pq.read_table(guard_path(raw_dir("swe_rebench_v2") / "data" / "train-00000-of-00001.parquet"),
                      columns=["instance_id", "repo", "language", "image_name", "meta", "problem_statement"])
    rows = t.to_pylist()
    used = train_repo_counts()
    pools = defaultdict(list)
    for r in rows:
        if swe_repo_verdict(r["repo"]):
            continue
        lang = (r["language"] or "").lower()
        key = {"go": "go", "python": "py", "ts": "ts", "typescript": "ts", "js": "ts", "javascript": "ts"}.get(lang)
        if not key:
            continue
        meta = r["meta"]
        if isinstance(meta, str):
            try:
                meta = ast.literal_eval(meta)
            except Exception:
                meta = {}
        lm = (meta or {}).get("llm_metadata") or {}
        if lm.get("difficulty") not in ("medium", "hard") or float(lm.get("confidence") or 0) < 0.9:
            continue
        if any((lm.get("detected_issues") or {}).values()):
            continue
        pools[key].append(r)
    chosen = []
    for key, n in (("go", n_go), ("py", n_py), ("ts", n_ts)):
        cand = sorted(pools[key], key=lambda r: (used[r["repo"].lower()], stable_hash01(r["instance_id"], "swe-dev")))
        repos = set()
        for r in cand:
            if r["repo"].lower() in repos:      # one task per repo -> diverse dev set
                continue
            repos.add(r["repo"].lower())
            chosen.append((key, r))
            if sum(1 for k, _ in chosen if k == key) >= n:
                break
    _w("dev_swe.csv", ["instance_id", "repo", "language", "image_name", "train_trajectories_in_repo",
                       "sha256_problem"],
       [[r["instance_id"], r["repo"], k, r["image_name"], used[r["repo"].lower()],
         sha256_text(r["problem_statement"] or "")] for k, r in chosen])


def tb_dev():
    root = raw_dir("tblite")
    unsafe = tb_unsafe_names()
    rows = []
    for d in sorted(p for p in root.iterdir() if (p / "instruction.md").exists()):
        if d.name == "grpc-plant-position-server" or d.name in unsafe:
            continue
        rows.append([d.name, sha256_text((d / "instruction.md").read_text())])
    _w("dev_tb.csv", ["task", "sha256_instruction"], rows)


def asi_dev(n=35):
    import pyarrow.parquet as pq
    p = sorted((raw_dir("scienceagentbench") / "data").glob("*.parquet"))[0]
    rows = pq.read_table(guard_path(p)).to_pylist()
    by = defaultdict(list)
    for r in rows:
        by[r["domain"]].append(r)
    for k in by:
        by[k].sort(key=lambda r: stable_hash01(str(r["instance_id"]), "asi-dev"))
    chosen = []
    while len(chosen) < n and any(by.values()):
        for k in sorted(by):
            if by[k] and len(chosen) < n:
                chosen.append(by[k].pop(0))
    _w("dev_asi.csv", ["instance_id", "domain", "github_name", "sha256_task_inst"],
       [[r["instance_id"], r["domain"], r["github_name"], sha256_text(r["task_inst"])] for r in chosen])


def gpqa_dev():
    p = raw_dir("gpqa") / "gpqa_diamond.csv"
    if not p.exists():
        log("[dev] GPQA-Diamond not downloaded (gated dataset: accept terms on HF, then re-run download)")
        return
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    _w("dev_hle_gpqa_diamond.csv", ["record_id", "subdomain", "sha256_question"],
       [[r.get("Record ID"), r.get("Subdomain"), sha256_text(r.get("Question") or "")] for r in rows])


def dev_tb_names() -> set[str]:
    p = MANI / "dev_tb.csv"
    if not p.exists():
        return set()
    with open(p) as f:
        return {r["task"] for r in csv.DictReader(f)}


def dev_swe_repos() -> set[str]:
    p = MANI / "dev_swe.csv"
    if not p.exists():
        return set()
    with open(p) as f:
        return {r["repo"].lower() for r in csv.DictReader(f)}


def main():
    swe_dev()
    tb_dev()
    asi_dev()
    gpqa_dev()
    (MANI / "dev_sets_info.json").write_text(json.dumps(
        {"swe": "SWE-rebench-V2 " + CFG["sources"]["swe_rebench_v2"]["revision"],
         "tb": "OpenThoughts-TBLite " + CFG["sources"]["tblite"]["revision"],
         "asi": "ScienceAgentBench " + CFG["sources"]["scienceagentbench"]["revision"],
         "hle": "GPQA-Diamond " + CFG["sources"]["gpqa"]["revision"] + " + 300 held-out (build.py hle)"}, indent=1))


if __name__ == "__main__":
    main()
