"""One-off repair of the already-built E-HLE set (2026-10-13).

nemotron_science_rqa answers were written from the raw \\boxed{} content, so multi-line boxed answers
(aligned/array blocks; 138/1838 train, 3/28 val) broke the single-line "Answer: ..." line that verify.py checks.
convert_hle.from_nemotron_science now applies clean_answer(); this script applies the same transformation to the
built files instead of re-running the ~46 min build. Selection is unchanged; n_tokens, manifests and stats are
recomputed. Originals are kept in $APEX/data/interim/hle/pre_repair/.

usage: python -m src.data.repair_hle
       python -m src.data.mix && python -m src.data.verify --recount 100 && python -m src.data.summarize
"""
from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter

from .build import MANI
from .common import count_tokens, interim_dir, log, processed_dir, read_jsonl, sample_sha, write_jsonl
from .convert_hle import clean_answer

FMT = re.compile(r"Explanation: [\s\S]+\nAnswer: .+\nConfidence: \d{1,3}%")
TAIL = re.compile(r"\nConfidence: (\d{1,3})%$")


def repair(row: dict) -> bool:
    m = row["messages"][-1]
    c = m["content"]
    if FMT.fullmatch(c):
        return False
    t = TAIL.search(c)
    k = c.rfind("\nAnswer: ", 0, t.start() if t else len(c))
    if row["source"] != "nemotron_science_rqa" or not t or k < 0:
        raise ValueError(f"{row['id']}: unexpected HLE content layout")
    ans = clean_answer(c[k + len("\nAnswer: "):t.start()])
    m["content"] = f"{c[:k]}\nAnswer: {ans}\nConfidence: {t.group(1)}%"
    if not FMT.fullmatch(m["content"]):
        raise ValueError(f"{row['id']}: still malformed after repair")
    row["n_tokens"] = count_tokens(row["messages"], row.get("tools"))
    return True


def main():
    bak = interim_dir("hle") / "pre_repair"
    bak.mkdir(parents=True, exist_ok=True)
    files = {"train": processed_dir("hle") / "train.jsonl", "val": processed_dir("hle") / "val.jsonl",
             "heldout": processed_dir("dev") / "hle_heldout.jsonl"}
    rows, fixed = {}, Counter()
    for name, p in files.items():
        if not (bak / p.name).exists():
            shutil.copy2(p, bak / p.name)
        rows[name] = list(read_jsonl(p))
        fixed[name] = sum(repair(r) for r in rows[name])
        write_jsonl(p, rows[name])
    for name in ("train", "val"):
        with open(MANI / f"hle_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "source", "instance_id", "n_tokens", "sha256"])
            for c in sorted(rows[name], key=lambda c: c["id"]):
                w.writerow([c["id"], c["source"], c["instance_id"], c["n_tokens"], sample_sha(c)])
    with open(MANI / "dev_hle_heldout.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "instance_id", "sha256"])
        for s in sorted(rows["heldout"], key=lambda s: s["id"]):
            w.writerow([s["id"], s["source"], s["instance_id"], sample_sha(s)])
    sp = MANI / "hle_stats.json"
    st = json.loads(sp.read_text())
    train, val = rows["train"], rows["val"]
    st.update({"n_train": len(train), "n_val": len(val),
               "train_tokens": sum(c["n_tokens"] for c in train), "val_tokens": sum(c["n_tokens"] for c in val),
               "train_tokens_by_source": {s: sum(c["n_tokens"] for c in train if c["source"] == s)
                                          for s in sorted({c["source"] for c in train})},
               "mean_tokens": round(sum(c["n_tokens"] for c in train) / max(1, len(train))),
               "repair": {"what": "rqa multi-line Answer -> clean_answer()", "fixed": dict(fixed)}})
    sp.write_text(json.dumps(st, indent=1, default=str))
    log(f"[repair_hle] fixed {dict(fixed)}; train_tokens={st['train_tokens']/1e6:.2f}M")


if __name__ == "__main__":
    main()
