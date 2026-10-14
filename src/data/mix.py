"""Mixed-data control = union of the four expert train sets with exactly the same sample ids.

Writes $APEX/data/processed/mixed/{train,val}.jsonl (deterministic shuffle) and
$APEX/data/manifests/mixed_{train,val}.csv (id, expert, n_tokens, sha256) + mixed_stats.json.
"""
from __future__ import annotations

import csv
import json

from .common import APEX, log, processed_dir, read_jsonl, sample_sha, stable_hash01, write_jsonl

EXPERTS = ["swe", "tb", "asi", "hle"]
MANI = APEX / "data" / "manifests"


def main():
    stats = {}
    for split in ("train", "val"):
        rows = []
        for e in EXPERTS:
            p = processed_dir(e) / f"{split}.jsonl"
            if not p.exists():
                raise SystemExit(f"missing {p}; build all experts first")
            for r in read_jsonl(p):
                r["meta"] = {**(r.get("meta") or {}), "expert": e}
                rows.append(r)
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids)), "duplicate ids across experts"
        rows.sort(key=lambda r: stable_hash01(r["id"], "mixed-shuffle"))
        write_jsonl(processed_dir("mixed") / f"{split}.jsonl", rows)
        with open(MANI / f"mixed_{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "expert", "n_tokens", "sha256"])
            for r in rows:
                w.writerow([r["id"], r["meta"]["expert"], r["n_tokens"], sample_sha(r)])
        # id-set equality with the per-expert manifests
        want = set()
        for e in EXPERTS:
            with open(MANI / f"{e}_{split}.csv") as f:
                want |= {x["id"] for x in csv.DictReader(f)}
        assert want == set(ids), f"mixed {split} ids != union of expert manifests"
        stats[split] = {"n": len(rows), "tokens": sum(r["n_tokens"] for r in rows),
                        "by_expert": {e: sum(1 for r in rows if r["meta"]["expert"] == e) for e in EXPERTS},
                        "tokens_by_expert": {e: sum(r["n_tokens"] for r in rows if r["meta"]["expert"] == e)
                                             for e in EXPERTS}}
        log(f"[mixed] {split}: {stats[split]}")
    (MANI / "mixed_stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
