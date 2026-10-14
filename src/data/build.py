"""Build one expert's SFT set.

usage: python -m src.data.build --expert {swe,tb,asi,hle} [--workers 16] [--max-rows N] [--dry]
Outputs:
  $APEX/data/processed/<expert>/{train,val}.jsonl      unified schema (see common.py)
  $APEX/data/manifests/<expert>_{train,val}.csv        id,source,instance_id,n_tokens,sha256
  $APEX/data/manifests/<expert>_stats.json             drop counts per reason, strata, token totals
  $APEX/data/interim/<expert>/candidates.jsonl         all converted+counted candidates (for audit)
"""
from __future__ import annotations

import argparse
import csv
import os
import io
import json
import multiprocessing as mp
import re
import time
from collections import Counter, defaultdict

from .common import (APEX, CFG, DropCounter, count_tokens, est_tokens, get_tokenizer, guard_path, interim_dir, log,
                     processed_dir, raw_dir, read_jsonl, sample_sha, sha256_text, stable_hash01, validate_sample,
                     write_jsonl)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MANI = APEX / "data" / "manifests"


# ============================================================ generic helpers
def iter_parquet(path, columns=None, batch_size=64, row_filter=None):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(guard_path(path))
    for b in pf.iter_batches(batch_size=batch_size, columns=columns):
        for r in b.to_pylist():
            if row_filter is None or row_filter(r):
                yield r


def _finish_sample(s: dict, max_tokens: int):
    """Token count (exact, Qwen3.6 template, preserve_thinking) + validation. Returns (sample|None, reason)."""
    if est_tokens(s["messages"], s.get("tools")) > 1.6 * max_tokens:
        return None, "too_long_est"
    errs = validate_sample({**s, "n_tokens": 0})
    if errs:
        return None, "schema:" + errs[0].split(":")[-1].strip()[:40]
    n = count_tokens(s["messages"], s.get("tools"))
    if n > max_tokens:
        return None, "too_long"
    s["n_tokens"] = n
    return s, None


def run_pool(fn, rows, workers, batch=None):
    """Bounded parallel map: Pool.imap would drain the row generator eagerly (whole dataset in RAM),
    so rows are fed in batches of `batch`."""
    if workers <= 1:
        for r in rows:
            yield fn(r)
        return
    batch = batch or workers * 16
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, maxtasksperchild=2000) as pool:
        buf = []
        for r in rows:
            buf.append(r)
            if len(buf) >= batch:
                yield from pool.imap_unordered(fn, buf, chunksize=2)
                buf = []
        if buf:
            yield from pool.imap_unordered(fn, buf, chunksize=2)


def select(cands, strata_frac: dict, key_stratum, budget: int, max_count: int, constraints) -> list:
    """Greedy count-balanced stratified selection until the token budget (train+val) is reached."""
    by = defaultdict(list)
    for c in cands:
        by[key_stratum(c)].append(c)
    for k in by:
        by[k].sort(key=lambda c: stable_hash01(c["id"], "select"))
    ptr = {k: 0 for k in by}
    chosen, cnt, tok = [], Counter(), 0
    state = defaultdict(Counter)
    live = {k for k in by if strata_frac.get(k, 0) > 0}
    while live and tok < budget and len(chosen) < max_count:
        k = min(live, key=lambda s: cnt[s] / strata_frac[s])
        lst = by[k]
        while ptr[k] < len(lst) and not constraints(lst[ptr[k]], state):
            ptr[k] += 1
        if ptr[k] >= len(lst):
            live.discard(k)
            continue
        c = lst[ptr[k]]
        ptr[k] += 1
        constraints(c, state, commit=True)
        chosen.append(c)
        cnt[k] += 1
        tok += c["n_tokens"]
    return chosen


def _budget(c: dict) -> int:
    """train-token budget (per-expert override or global), grossed up for the val split taken afterwards."""
    return int((c.get("token_budget") or CFG["token_budget_per_expert"]) / (1 - CFG["val_fraction"]))


def make_constraints(max_per_instance=2, max_per_repo=None):
    def f(c, state, commit=False):
        inst, repo = c["instance_id"], c.get("meta", {}).get("repo")
        if state["inst"][inst] >= max_per_instance:
            return False
        if max_per_repo and repo and state["repo"][repo] >= max_per_repo:
            return False
        if commit:
            state["inst"][inst] += 1
            if repo:
                state["repo"][repo] += 1
        return True
    return f


def write_outputs(expert: str, chosen: list, drops: dict, extra: dict):
    vf = CFG["val_fraction"]
    for c in chosen:
        c["split"] = "val" if stable_hash01(c["instance_id"], expert + "|val") < vf else "train"
    train = [c for c in chosen if c["split"] == "train"]
    val = [c for c in chosen if c["split"] == "val"]
    if not val and len(train) > 50:   # guarantee a non-empty val split
        val = [train.pop()]
    out = processed_dir(expert)
    keep = ("id", "source", "instance_id", "messages", "tools", "n_tokens", "meta")
    write_jsonl(out / "train.jsonl", ({k: c.get(k) for k in keep} for c in train))
    write_jsonl(out / "val.jsonl", ({k: c.get(k) for k in keep} for c in val))
    MANI.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with open(MANI / f"{expert}_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "source", "instance_id", "n_tokens", "sha256"])
            for c in sorted(rows, key=lambda c: c["id"]):
                w.writerow([c["id"], c["source"], c["instance_id"], c["n_tokens"], sample_sha(c)])
    st = {"expert": expert, "n_train": len(train), "n_val": len(val),
          "train_tokens": sum(c["n_tokens"] for c in train), "val_tokens": sum(c["n_tokens"] for c in val),
          "train_by_source": Counter(c["source"] for c in train),
          "train_tokens_by_source": dict(Counter({s: sum(c["n_tokens"] for c in train if c["source"] == s)
                                                 for s in {c["source"] for c in train}})),
          "train_by_stratum": Counter(c.get("meta", {}).get("stratum") for c in train),
          "mean_tokens": round(sum(c["n_tokens"] for c in train) / max(1, len(train))),
          "drops": drops, **extra, "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (MANI / f"{expert}_stats.json").write_text(json.dumps(st, indent=1, default=str))
    log(f"[{expert}] train={len(train)} ({st['train_tokens']/1e6:.2f}M tok) val={len(val)}")
    return st


def dump_candidates(expert, cands):
    write_jsonl(interim_dir(expert) / "candidates.jsonl",
                ({k: c[k] for k in ("id", "source", "instance_id", "n_tokens", "meta")} for c in cands))


# ============================================================ E-SWE
def _swe_worker(args):
    from .convert_swe import Unconvertible, convert_openhands, bash_tool, patch_files
    src, r = args
    try:
        msgs, st = convert_openhands(r["trajectory"])
    except Unconvertible as e:
        return None, src, f"convert:{e}"
    except Exception as e:  # noqa
        return None, src, f"convert_crash:{type(e).__name__}"
    core, allf = patch_files(r.get("model_patch") or "")
    if core == 0:
        return None, src, "patch_no_source_change"
    turns = sum(1 for m in msgs if m["role"] == "assistant")
    hard = core >= 2 or turns >= 40
    tid = r.get("trajectory_id") or sha256_text(json.dumps(r["trajectory"], default=str))[:16]
    s = {"id": f"swe/{src}/{tid}", "source": src, "instance_id": r["instance_id"], "messages": msgs,
         "tools": [bash_tool()],
         "meta": {"repo": r["repo"].lower(), "files_changed": core, "turns": turns, "stratum": "hard" if hard else "other",
                  **st}}
    s, why = _finish_sample(s, CFG["experts"]["swe"]["max_tokens"])
    return s, src, why


def build_swe(a):
    from .convert_swe import mini_cfg, swe_pro_wrapper
    from .decontam import swe_pro_ngram_index, swe_repo_verdict
    from .dev_sets import dev_swe_repos
    c = CFG["experts"]["swe"]
    drops = DropCounter()
    idx = swe_pro_ngram_index()
    dev_repos = dev_swe_repos()
    log(f"[swe] holding out {len(dev_repos)} dev repos")
    mini_cfg(); swe_pro_wrapper(); get_tokenizer()          # build caches before forking workers

    def prefilter(src, r):
        if src == "swe_rebench_oh" and int(r.get("resolved") or 0) != 1:
            drops.drop(f"{src}:not_resolved"); return False
        if src == "swe_hero_oh" and "R2E-Gym" not in (r.get("dataset") or ""):
            drops.drop(f"{src}:not_r2e_gym(execution-verified)"); return False
        if (r.get("repo") or "").lower() in dev_repos:
            drops.drop(f"{src}:dev_repo_holdout"); return False
        v = swe_repo_verdict(r.get("repo", ""), r.get("model_patch") or "")
        if v:
            drops.drop(f"{src}:decontam:{v}"); return False
        user = next((m["content"] for m in r["trajectory"] if m["role"] == "user"), "")
        k, _ = idx.hits(user)
        if k >= 1:
            drops.drop(f"{src}:decontam:swe_pro_13gram"); return False
        if r.get("exit_status") and r["exit_status"] != "submit":
            drops.drop(f"{src}:exit_not_submit"); return False
        return True

    def rows():
        n = 0
        for src, pat in (("swe_rebench_oh", "trajectories.parquet"), ("swe_hero_oh", "data/*.parquet")):
            for p in sorted(raw_dir(src).glob(pat)):
                for r in iter_parquet(p, batch_size=32):
                    n += 1
                    if a.max_rows and n > a.max_rows:
                        return
                    if prefilter(src, r):
                        yield src, r
    cands = []
    for s, src, why in run_pool(_swe_worker, rows(), a.workers):
        if s is None:
            drops.drop(f"{src}:{why}")
        else:
            cands.append(s)
    # dedup identical content
    seen, uniq = set(), []
    for s in cands:
        h = sample_sha(s)
        if h in seen:
            drops.drop(f"{s['source']}:dup_content"); continue
        seen.add(h); uniq.append(s)
    dump_candidates("swe", uniq)
    budget = _budget(c)
    sec = c["secondary_fraction"]
    hf = c["hard_fraction"]
    frac = {("swe_rebench_oh", "hard"): (1 - sec) * hf, ("swe_rebench_oh", "other"): (1 - sec) * (1 - hf),
            ("swe_hero_oh", "hard"): sec * hf, ("swe_hero_oh", "other"): sec * (1 - hf)}
    max_repo = max(1, int(c["max_repo_fraction"] * c["target_count"]))
    chosen = select(uniq, frac, lambda s: (s["source"], s["meta"]["stratum"]), budget,
                    c["target_count"], make_constraints(c["max_per_instance"], max_repo))
    return write_outputs("swe", chosen, dict(drops), {"n_candidates": len(uniq),
                         "candidates_by_stratum": Counter(f"{s['source']}/{s['meta']['stratum']}" for s in uniq)})


# ============================================================ E-TB
def _tb_worker(args):
    from .convert_terminus import Unconvertible, normalize_t2
    from .decontam import tb2_ngram_index, tb_name_verdict
    src, r = args
    try:
        msgs, st = normalize_t2(r["conversations"], max_parse_errors=0)
    except Unconvertible as e:
        return None, src, f"convert:{e}"
    except Exception as e:  # noqa
        return None, src, f"convert_crash:{type(e).__name__}"
    v = tb_name_verdict(r.get("task", ""), st["instruction"])
    if v:
        return None, src, f"decontam:{v}"
    if re.sub(r"_copy\d+$", "", r.get("task") or "") in _TB_DEV:
        return None, src, "dev_task_holdout"
    k, _ = tb2_ngram_index().hits(st["instruction"])
    if k >= 3:
        return None, src, "decontam:tb2_13gram"
    task = r.get("task") or r.get("trial_name")
    inst = re.sub(r"_copy\d+$", "", task)
    level = r.get("_level", "")
    stratum = "complete" if st["complete"] else "incomplete"
    s = {"id": f"tb/{src}/{r.get('trial_name') or task}", "source": src, "instance_id": f"{src}:{inst}",
         "messages": msgs, "tools": None,
         "meta": {"task": task, "category": r.get("_category", ""), "level": level, "turns": st["n_turns"],
                  "complete": st["complete"], "stratum": stratum}}
    s, why = _finish_sample(s, CFG["experts"]["tb"]["max_tokens"])
    return s, src, why


def _nemotron_files(subsets_include=None, exclude_cats=(), levels=("medium", "mixed")):
    root = raw_dir("nemotron_terminal") / "synthetic_tasks" / "skill_based"
    out = []
    for lv in levels:
        for d in sorted((root / lv).glob("*")):
            sub = f"{lv}/{d.name}"
            if subsets_include is not None and sub not in subsets_include:
                continue
            if d.name in exclude_cats:
                continue
            out += [(lv, d.name, p) for p in sorted(d.glob("*.parquet"))]
    return out


def _pool_rows(files_with_tags, pool_size, salt, extra_filter=None, drops=None, src="", max_rows=0):
    """Hash-sample ~pool_size rows across files (deterministic)."""
    import pyarrow.parquet as pq
    total = sum(pq.ParquetFile(guard_path(p)).metadata.num_rows for *_, p in files_with_tags)
    p_keep = min(1.0, 1.6 * pool_size / max(1, total))
    log(f"[pool] {src}: {total} rows, keep prob {p_keep:.3f}")
    n = 0
    for tags in files_with_tags:
        *meta, p = tags
        for r in iter_parquet(p, batch_size=64):
            n += 1
            if max_rows and n > max_rows:
                return
            key = r.get("trial_name") or r.get("id") or json.dumps(r)[:200]
            if stable_hash01(key, salt) >= p_keep:
                continue
            if extra_filter and not extra_filter(r):
                if drops is not None:
                    drops.drop(f"{src}:source_filter")
                continue
            if meta:
                r["_level"], r["_category"] = meta[0], meta[1]
            yield r


_TB_DEV: set = set()


def build_tb(a):
    global _TB_DEV
    from .convert_terminus import harbor_template
    from .decontam import tb2_ngram_index, tb_unsafe_names
    from .dev_sets import dev_tb_names
    c = CFG["experts"]["tb"]
    drops = DropCounter()
    _TB_DEV = dev_tb_names()
    tb2_ngram_index(); tb_unsafe_names(); harbor_template(); get_tokenizer()
    nem_files = _nemotron_files(exclude_cats=set(c["nemotron_categories_exclude"]), levels=c["nemotron_levels"])
    ot_files = [("", "", p) for p in sorted((raw_dir("ot_agent_100k") / "data").glob("*.parquet"))]
    excl = tuple(c["ot_exclude_task_prefixes"])

    def rows():
        for r in _pool_rows(nem_files, c["candidate_pool_per_source"], "tb-nem", drops=drops,
                            src="nemotron_terminal", max_rows=a.max_rows):
            yield "nemotron_terminal", r
        for r in _pool_rows(ot_files, c["candidate_pool_per_source"], "tb-ot",
                            extra_filter=lambda r: not (r.get("task") or "").startswith(excl)
                            and (r.get("result") in (None, "None", "")),
                            drops=drops, src="ot_agent_100k", max_rows=a.max_rows):
            yield "ot_agent_100k", r
    cands = []
    for s, src, why in run_pool(_tb_worker, rows(), a.workers):
        if s is None:
            drops.drop(f"{src}:{why}")
        else:
            cands.append(s)
    dump_candidates("tb", cands)
    budget = _budget(c)
    nf, inc = c["nemotron_fraction"], c["max_incomplete_fraction"]
    frac = {("nemotron_terminal", "complete"): nf * (1 - inc), ("nemotron_terminal", "incomplete"): nf * inc,
            ("ot_agent_100k", "complete"): (1 - nf) * (1 - inc), ("ot_agent_100k", "incomplete"): (1 - nf) * inc}
    chosen = select(cands, frac, lambda s: (s["source"], s["meta"]["stratum"]), budget,
                    c["target_count"], make_constraints(2))
    return write_outputs("tb", chosen, dict(drops), {"n_candidates": len(cands),
                         "candidates_by_stratum": Counter(f"{s['source']}/{s['meta']['stratum']}" for s in cands)})


# ============================================================ E-ASI
def _asi_worker(args):
    from .convert_asi import bash_tool, jupyter_to_opencode, t2_to_opencode
    from .convert_terminus import Unconvertible
    from .decontam import asi_blacklist_hit, asi_ngram_index
    src, r = args
    try:
        if src == "jupyter_agent":
            msgs, st = jupyter_to_opencode(r)
            task = r["id"]
        else:
            msgs, st = t2_to_opencode(r["conversations"])
            task = r.get("task") or r.get("trial_name")
    except Unconvertible as e:
        return None, src, f"convert:{e}"
    except Exception as e:  # noqa
        return None, src, f"convert_crash:{type(e).__name__}"
    prompt = msgs[1]["content"]
    k, _ = asi_ngram_index().hits(prompt)
    if k >= 3:
        return None, src, "decontam:asi_13gram"
    b = asi_blacklist_hit(prompt)
    if b:
        return None, src, "decontam:method_blacklist"
    turns = st["n_turns"]
    stratum = "long" if turns >= 12 else "short"
    s = {"id": f"asi/{src}/{r.get('trial_name') or task}", "source": src, "instance_id": f"{src}:{task}",
         "messages": msgs, "tools": [bash_tool()],
         "meta": {"task": task, "subset": f"{r.get('_level', '')}/{r.get('_category', '')}", "turns": turns,
                  "stratum": stratum, "blacklist_term": None}}
    s, why = _finish_sample(s, CFG["experts"]["asi"]["max_tokens"])
    return s, src, why


def build_asi(a):
    from .convert_asi import bash_tool, system_prompt
    from .decontam import asi_embed_refs, asi_ngram_index, max_cosine
    c = CFG["experts"]["asi"]
    drops = DropCounter()
    asi_ngram_index(); bash_tool(); system_prompt(); get_tokenizer()
    subs = set(c["nemotron_subsets"])
    nem_files = _nemotron_files(subsets_include=subs, levels=("medium", "mixed"))
    jup_files = [("", "", p) for p in sorted((raw_dir("jupyter_agent") / "data").glob("thinking-*.parquet"))]

    def rows():
        for r in _pool_rows(nem_files, c["candidate_pool"], "asi-nem", drops=drops, src="nemotron_terminal",
                            max_rows=a.max_rows):
            yield "nemotron_terminal", r
        for r in _pool_rows(jup_files, int(c["candidate_pool"] * c["jupyter_fraction"] * 2), "asi-jup",
                            drops=drops, src="jupyter_agent", max_rows=a.max_rows):
            yield "jupyter_agent", r
    cands = []
    for s, src, why in run_pool(_asi_worker, rows(), a.workers):
        if s is None:
            drops.drop(f"{src}:{why}")
        else:
            cands.append(s)
    if cands and not a.no_embed:
        sims = max_cosine([s["messages"][1]["content"][:4000] for s in cands], asi_embed_refs())
        keep = []
        for s, (sim, j) in zip(cands, sims):
            s["meta"]["asi_max_cos"] = round(sim, 4)
            if sim >= c["embed_threshold"]:
                drops.drop(f"{s['source']}:decontam:embed_cos>={c['embed_threshold']}")
            else:
                keep.append(s)
        cands = keep
    dump_candidates("asi", cands)
    budget = _budget(c)
    jf = c["jupyter_fraction"]
    frac = {("nemotron_terminal", "long"): (1 - jf) * 0.6, ("nemotron_terminal", "short"): (1 - jf) * 0.4,
            ("jupyter_agent", "long"): jf * 0.5, ("jupyter_agent", "short"): jf * 0.5}
    chosen = select(cands, frac, lambda s: (s["source"], s["meta"]["stratum"]), budget,
                    c["target_count"], make_constraints(1))
    return write_outputs("asi", chosen, dict(drops), {"n_candidates": len(cands),
                         "candidates_by_stratum": Counter(f"{s['source']}/{s['meta']['stratum']}" for s in cands)})


# ============================================================ E-HLE
def _hle_worker(args):
    from . import convert_hle as H
    from .decontam import hle_verdict
    src, r = args
    try:
        if src == "nemotron_science_mcq":
            q, msgs, st = H.from_nemotron_science(r, "mcq")
        elif src == "nemotron_science_rqa":
            q, msgs, st = H.from_nemotron_science(r, "rqa")
        elif src == "nemotron_math_aops":
            q, msgs, st = H.from_nemotron_math(r)
        else:
            q, msgs, st = H.from_osr2(r)
    except H.Unconvertible as e:
        return None, src, f"convert:{e}"
    except Exception as e:  # noqa
        return None, src, f"convert_crash:{type(e).__name__}"
    v = hle_verdict(q, json.dumps(msgs, ensure_ascii=False))
    if v:
        return None, src, f"decontam:{v}"
    uid = r.get("uuid") or ""
    if not uid:
        uid = f"{stable_hash01(q, 'hle-id'):.12f}"[2:]
    s = {"id": f"hle/{src}/{uid}", "source": src, "instance_id": "q:" + sha256_text(q.strip().lower())[:16],
         "messages": msgs, "tools": None,
         "meta": {"confidence": st["confidence"], "stratum": src, "q_chars": len(q),
                  "reasoning_chars": len(msgs[-1]["reasoning_content"])}}
    s, why = _finish_sample(s, CFG["experts"]["hle"]["max_tokens"])
    return s, src, why


def _stream_jsonl(repo, rev, path, max_rows):
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    with fs.open(f"datasets/{repo}@{rev}/{path}", "rb", block_size=2 ** 24) as fh:
        for i, line in enumerate(io.TextIOWrapper(fh, encoding="utf-8")):
            if i >= max_rows:
                break
            yield json.loads(line)


def _stream_parquet(repo, rev, path, max_full_rgs, filter_col, pred, want):
    """Two-pass row-group streaming for a huge single-file parquet (OSR-2: 9 RGs x 200k rows, ~3.8 GB
    uncompressed `output` per RG): pass 1 reads only `filter_col` of every RG and evaluates `pred`;
    pass 2 reads full rows only from the <= max_full_rgs RGs with the most hits (until `want` rows)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    with fs.open(f"datasets/{repo}@{rev}/{path}", "rb", block_size=2 ** 24) as fh:
        pf = pq.ParquetFile(fh)
        hits = {}
        for g in range(pf.num_row_groups):
            col = pf.read_row_group(g, columns=[filter_col]).column(0).to_pylist()
            hits[g] = [i for i, v in enumerate(col) if pred(v)]
            log(f"[osr2] row group {g}: {len(hits[g])}/{len(col)} hits")
            del col
        n = 0
        for g in sorted(hits, key=lambda g: -len(hits[g]))[:max_full_rgs]:
            idx = sorted(hits[g], key=lambda i: stable_hash01(f"{g}:{i}", "osr2"))[: max(0, want - n)]
            if not idx:
                break
            t = pf.read_row_group(g)
            # 逐行零拷贝 slice：Table.take 会把多个 chunk 的 string 列拼接，超过 2GB 时 offset 溢出（ArrowInvalid）
            for i in sorted(idx):
                yield t.slice(i, 1).to_pylist()[0]
                n += 1
            del t


def build_hle(a):
    from . import convert_hle as H
    from .decontam import hle_exact_set, hle_ngram_index, hle_rows, max_cosine
    c = CFG["experts"]["hle"]
    drops = DropCounter()
    pool = c["candidate_pool_per_source"] if not a.max_rows else a.max_rows
    S = CFG["sources"]
    hle_ngram_index(); hle_exact_set(); get_tokenizer()

    def rows():
        for kind, fn in (("mcq", "MCQ.jsonl"), ("rqa", "RQA.jsonl")):
            src = f"nemotron_science_{kind}"
            p = raw_dir("nemotron_science") / "data" / fn
            cnt = 0
            for r in read_jsonl(p):
                a_ = next((m for m in r["messages"] if m["role"] == "assistant"), None)
                if not a_ or len(a_.get("reasoning_content") or "") < 3000:   # hardness proxy: long traces
                    drops.drop(f"{src}:short_reasoning"); continue
                if stable_hash01(r["uuid"], "hle-sci") > 0.35:
                    continue
                yield src, r
                cnt += 1
                if cnt >= pool:
                    break
        if a.skip_streams:
            return
        m = S["nemotron_math_v2"]

        def math_ok(r):
            if r.get("data_source") != "aops" or "cc-by-sa" in (r.get("license") or "").lower():
                return False
            meta = r.get("metadata") or {}
            hi = (meta.get("reason_high_no_tool") or {}).get("accuracy")
            lo = (meta.get("reason_low_no_tool") or {}).get("accuracy")
            return hi is not None and 0.125 <= hi <= 0.75 and (lo is None or lo <= 0.5)
        cnt = 0
        for r in _stream_jsonl(m["repo"], m["revision"], m["stream_file"], m["stream_max_rows"]):
            if not math_ok(r):
                drops.drop("nemotron_math_aops:not_aops_hard"); continue
            yield "nemotron_math_aops", r
            cnt += 1
            if cnt >= pool:
                break
        o = S["osr2"]
        cnt = 0
        for r in _stream_parquet(o["repo"], o["revision"], o["stream_file"], o["max_full_row_groups"], "input",
                                 H.is_nonstem, max(1, pool // 2)):
            yield "osr2_nonstem", r
            cnt += 1
            if cnt >= max(1, pool // 2):
                break
    cands = []
    for s, src, why in run_pool(_hle_worker, rows(), a.workers):
        if s is None:
            drops.drop(f"{src}:{why}")
        else:
            cands.append(s)
    # exact dedup on question
    seen, uniq = set(), []
    for s in cands:
        if s["instance_id"] in seen:
            drops.drop(f"{s['source']}:dup_question"); continue
        seen.add(s["instance_id"]); uniq.append(s)
    cands = uniq
    if cands and not a.no_embed:
        sims = max_cosine([s["messages"][1]["content"][:2000] for s in cands], [r["question"][:2000] for r in hle_rows()])
        keep = []
        for s, (sim, j) in zip(cands, sims):
            s["meta"]["hle_max_cos"] = round(sim, 4)
            if sim >= c["embed_threshold"]:
                drops.drop(f"{s['source']}:decontam:embed_cos>={c['embed_threshold']}")
            else:
                keep.append(s)
        cands = keep
    # held-out dev items (removed before selection): hardest by reasoning length, per source proportional
    cands.sort(key=lambda s: stable_hash01(s["id"], "heldout"))
    n_hold = c["heldout_count"]
    held = []
    for src, f in c["mix"].items():
        pool_src = sorted([s for s in cands if s["source"] == src], key=lambda s: -s["meta"]["reasoning_chars"])
        held += pool_src[: int(round(n_hold * f))]
    held_ids = {s["id"] for s in held}
    cands = [s for s in cands if s["id"] not in held_ids]
    write_jsonl(processed_dir("dev") / "hle_heldout.jsonl",
                ({k: s[k] for k in ("id", "source", "instance_id", "messages", "n_tokens", "meta")} for s in held))
    MANI.mkdir(parents=True, exist_ok=True)
    with open(MANI / "dev_hle_heldout.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "instance_id", "sha256"])
        for s in sorted(held, key=lambda s: s["id"]):
            w.writerow([s["id"], s["source"], s["instance_id"], sample_sha(s)])
    dump_candidates("hle", cands)
    budget = _budget(c)
    chosen = select(cands, dict(c["mix"]), lambda s: s["source"], budget, c["target_count"],
                    make_constraints(1))
    return write_outputs("hle", chosen, dict(drops), {"n_candidates": len(cands), "n_heldout_dev": len(held),
                         "candidates_by_source": Counter(s["source"] for s in cands),
                         "confidence_hist": Counter(s["meta"]["confidence"] for s in chosen)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True, choices=["swe", "tb", "asi", "hle"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=0, help="debug: cap rows read per source")
    ap.add_argument("--no-embed", action="store_true", help="debug: skip embedding filter")
    ap.add_argument("--skip-streams", action="store_true", help="debug: skip HF-streamed sources (Math-v2, OSR-2)")
    a = ap.parse_args()
    t0 = time.time()
    st = {"swe": build_swe, "tb": build_tb, "asi": build_asi, "hle": build_hle}[a.expert](a)
    log(f"[{a.expert}] done in {time.time() - t0:.0f}s: {json.dumps({k: st[k] for k in ('n_train', 'n_val', 'train_tokens')})}")


if __name__ == "__main__":
    main()
