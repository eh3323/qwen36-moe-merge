"""Small-sample self-test of every converter + tokenisation (login-node safe: a few hundred rows, 1 proc).

usage: python -m src.data.selftest --samples DIR [--out DIR]
DIR holds jsonl dumps of raw rows: nebius.jsonl hero.jsonl nemo_debug_med.jsonl ota.jsonl nemo_sci_med.jsonl
jupyter.jsonl nsci_mcq.jsonl nsci_rqa.jsonl math.jsonl osr2.jsonl (missing files are skipped).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .common import count_tokens, read_jsonl, render, validate_sample


def run(name, path, fn, out: Path, limit=200):
    if not path.exists():
        print(f"== {name}: (no sample file)")
        return
    ok, why, toks, ex = 0, Counter(), [], None
    for i, r in enumerate(read_jsonl(path)):
        if i >= limit:
            break
        try:
            res = fn(r)
        except Exception as e:
            why[f"{type(e).__name__}:{str(e)[:60]}"] += 1
            continue
        if res is None:
            why["filtered"] += 1
            continue
        msgs, tools = res
        errs = validate_sample({"id": "x", "source": name, "messages": msgs, "n_tokens": 0})
        if errs:
            why["schema:" + errs[0][:50]] += 1
            continue
        n = count_tokens(msgs, tools)
        toks.append(n)
        ok += 1
        if ex is None or (len(msgs) > 6 and len(ex[0]) <= 6):
            ex = (msgs, tools)
    toks.sort()
    q = (lambda p: toks[int(p * (len(toks) - 1))] if toks else 0)
    print(f"== {name}: ok={ok} tokens p10/p50/p90/max={q(.1)}/{q(.5)}/{q(.9)}/{q(1)} drops={dict(why.most_common(12))}")
    if ex:
        (out / f"{name}.rendered.txt").write_text(render(*ex))
        (out / f"{name}.messages.json").write_text(json.dumps(ex[0], indent=1, ensure_ascii=False)[:400000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    S = Path(a.samples)
    out = Path(a.out or S / "selftest_out")
    out.mkdir(parents=True, exist_ok=True)

    from . import convert_hle as H
    from .convert_asi import bash_tool as oc_tool, jupyter_to_opencode, t2_to_opencode
    from .convert_swe import bash_tool, convert_openhands
    from .convert_terminus import normalize_t2
    from .decontam import (asi_blacklist_hit, asi_ngram_index, hle_verdict, swe_pro_ngram_index,
                           swe_repo_verdict, tb2_ngram_index, tb_name_verdict, tb_unsafe_names)

    print("refs:", len(swe_pro_ngram_index()), len(tb2_ngram_index()), len(tb_unsafe_names()), len(asi_ngram_index()))

    def swe(r):
        if swe_repo_verdict(r["repo"], r.get("model_patch") or ""):
            return None
        if "resolved" in r and int(r["resolved"]) != 1:
            return None
        return convert_openhands(r["trajectory"])[0], [bash_tool()]

    def tb(r):
        msgs, st = normalize_t2(r["conversations"])
        if tb_name_verdict(r.get("task", ""), st["instruction"]) or tb2_ngram_index().hits(st["instruction"])[0] >= 3:
            return None
        return msgs, None

    def asi_t2(r):
        msgs, _ = t2_to_opencode(r["conversations"])
        if asi_blacklist_hit(msgs[1]["content"]) or asi_ngram_index().hits(msgs[1]["content"])[0] >= 3:
            return None
        return msgs, [oc_tool()]

    def asi_jup(r):
        return jupyter_to_opencode(r)[0], [oc_tool()]

    def hle(fn):
        def f(r):
            q, msgs, _ = fn(r)
            if hle_verdict(q):
                return None
            return msgs, None
        return f

    run("swe_rebench_oh", S / "nebius.jsonl", swe, out)
    run("swe_hero_oh", S / "hero.jsonl", swe, out)
    run("tb_nemotron", S / "nemo_debug_med.jsonl", tb, out)
    run("tb_ot_agent", S / "ota.jsonl", tb, out)
    run("asi_nemotron", S / "nemo_sci_med.jsonl", asi_t2, out)
    run("asi_jupyter", S / "jupyter.jsonl", asi_jup, out)
    run("hle_nsci_mcq", S / "nsci_mcq.jsonl", hle(lambda r: H.from_nemotron_science(r, "mcq")), out)
    run("hle_nsci_rqa", S / "nsci_rqa.jsonl", hle(lambda r: H.from_nemotron_science(r, "rqa")), out)
    run("hle_math", S / "math.jsonl", hle(H.from_nemotron_math), out)
    run("hle_osr2", S / "osr2.jsonl", hle(H.from_osr2), out)


if __name__ == "__main__":
    main()
