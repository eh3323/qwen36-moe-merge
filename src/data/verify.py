"""Post-build verification. Exit code != 0 on any hard failure.

Checks per expert: schema, unique ids, train/val instance disjointness, manifest sha256 == file content,
n_tokens re-count on a random subset, max length, no HLE canary / SWE-Pro repo / dev-holdout leakage,
format invariants (mini-swe submit command, Harbor template prefix, opencode bash-only tools, HLE answer
format). Writes one rendered example per expert to $APEX/data/interim/<expert>/example_rendered.txt.
usage: python -m src.data.verify [--experts swe,tb,asi,hle,mixed] [--recount 50]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys

from .common import (APEX, CFG, count_tokens, interim_dir, processed_dir, read_jsonl, render, sample_sha,
                     validate_sample)
from .decontam import HLE_CANARY, swe_repo_verdict

MANI = APEX / "data" / "manifests"


def check_expert(e: str, recount: int) -> list[str]:
    fails = []
    rows = {s: list(read_jsonl(processed_dir(e) / f"{s}.jsonl")) for s in ("train", "val")}
    allr = rows["train"] + rows["val"]
    ids = [r["id"] for r in allr]
    if len(ids) != len(set(ids)):
        fails.append("duplicate ids")
    if e != "mixed":
        ti = {r["instance_id"] for r in rows["train"]}
        vi = {r["instance_id"] for r in rows["val"]}
        if ti & vi:
            fails.append(f"{len(ti & vi)} instance ids in both train and val")
    for r in allr:
        errs = validate_sample(r)
        if errs:
            fails.append(f"{r['id']}: {errs[:2]}")
            break
    for s in ("train", "val"):
        with open(MANI / f"{e}_{s}.csv") as f:
            man = {x["id"]: x["sha256"] for x in csv.DictReader(f)}
        got = {r["id"]: sample_sha(r) for r in rows[s]}
        if man != got:
            fails.append(f"manifest {e}_{s}.csv mismatch ({len(set(man) ^ set(got))} id diffs)")
    blob_canary = sum(1 for r in allr if HLE_CANARY in json.dumps(r))
    if blob_canary:
        fails.append(f"{blob_canary} samples contain the HLE canary")
    if any("seed31415" in json.dumps(r) for r in allr):
        fails.append("seed31415 string present")
    rng = random.Random(0)
    for r in rng.sample(allr, min(recount, len(allr))):
        n = count_tokens(r["messages"], r.get("tools"))
        if n != r["n_tokens"]:
            fails.append(f"{r['id']}: n_tokens {r['n_tokens']} != recount {n}")
            break
    if e in CFG["experts"]:
        mx = CFG["experts"][e]["max_tokens"]
        over = [r["id"] for r in allr if r["n_tokens"] > mx]
        if over:
            fails.append(f"{len(over)} samples over {mx} tokens")
    # format invariants
    for r in allr:
        m = r["messages"]
        ex = r.get("meta", {}).get("expert", e)
        if ex == "swe":
            last = m[-1]["tool_calls"][0]["function"]["arguments"]["command"]
            if last != "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT":
                fails.append(f"{r['id']}: last command is not the submit command"); break
            if swe_repo_verdict(r["meta"].get("repo", "")):
                fails.append(f"{r['id']}: SWE-Pro repo leak"); break
            if [t["function"]["name"] for t in r["tools"]] != ["bash"]:
                fails.append(f"{r['id']}: tools != [bash]"); break
        elif ex == "tb":
            if not m[0]["content"].startswith("You are an AI assistant tasked with solving command-line tasks"):
                fails.append(f"{r['id']}: not Harbor template"); break
            if "Most bash commands should end with a newline" in m[0]["content"]:
                fails.append(f"{r['id']}: old Terminus template"); break
        elif ex == "asi":
            if [t["function"]["name"] for t in r["tools"]] != ["bash"] or not m[0]["content"].startswith("You are opencode"):
                fails.append(f"{r['id']}: not opencode/bash format"); break
            if m[-1].get("tool_calls"):
                fails.append(f"{r['id']}: ASI sample does not end with a final message"); break
        elif ex == "hle":
            if not re.fullmatch(r"Explanation: [\s\S]+\nAnswer: .+\nConfidence: \d{1,3}%", m[-1]["content"]):
                fails.append(f"{r['id']}: HLE answer format"); break
        if ex in ("swe", "tb", "asi", "hle") and any(x["role"] == "assistant" and not (x.get("reasoning_content") or "").strip()
                                                    for x in m):
            r.setdefault("_empty_think", True)
    n_empty = sum(1 for r in allr if r.get("_empty_think"))
    if e != "mixed" and allr:
        ex = allr[0]
        (interim_dir(e) / "example_rendered.txt").write_text(render(ex["messages"], ex.get("tools")))
    print(json.dumps({"expert": e, "train": len(rows["train"]), "val": len(rows["val"]),
                      "train_tokens": sum(r["n_tokens"] for r in rows["train"]),
                      "samples_with_some_empty_think": n_empty, "fails": fails[:10]}))
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", default="swe,tb,asi,hle,mixed")
    ap.add_argument("--recount", type=int, default=50)
    a = ap.parse_args()
    bad = 0
    for e in a.experts.split(","):
        try:
            bad += len(check_expert(e, a.recount))
        except FileNotFoundError as ex:
            print(json.dumps({"expert": e, "fails": [f"missing: {ex}"]}))
            bad += 1
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
