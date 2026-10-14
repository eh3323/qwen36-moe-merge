"""Chat rendering + assistant-only loss masking + token cache + packing.

Rendering: the model's own chat_template.jinja via tokenizer.apply_chat_template(messages, tools=...,
preserve_thinking=True) so every assistant turn keeps `<think>reasoning</think>` (the stock template drops
reasoning for assistant turns before the last real user query unless preserve_thinking is set).

Masking (token level, robust to string quirks): a token is a *target* iff it lies inside an assistant turn,
i.e. after the header tokens `<|im_start|>assistant\\n` and up to and including the closing `<|im_end|>`.
That covers thinking, visible content and tool calls; system/user/tool-response tokens are context only.

Optional per-record overrides (absent in every original data set, whose behaviour is unchanged):
  "render": {"preserve_thinking": false}  render this record with the template default (earlier turns'
                                          reasoning not replayed), i.e. how the eval server renders
                                          (chat_template_kwargs {"enable_thinking": true, "preserve_thinking": false});
  "loss": "last_assistant"                only the final assistant turn is a target, and only the tokens the
                                          model generates at eval: after `<|im_start|>assistant\\n<think>\\n`
                                          (the generation prompt) through the final `<|im_end|>`. Prompt and
                                          generated part are tokenized separately (as at eval).
  Used by the per-turn E-TB-v2 set (src/data/build_tb_v2.py --per-turn).

Cache: per data file, arrays ids(int32) / mask(uint8) / offsets(int64) as .npy (mmap-able) + meta json.
Packing: window-shuffled first-fit-decreasing into bins of max_seq_len; each sample keeps its own
position_ids (0..len-1) and cu_seqlens boundaries, so attention / GDN recurrence / conv never cross samples.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CACHE_VERSION = "v1"
_TOK = None  # per-process tokenizer
_TOK_CFG: dict = {}


# ----------------------------------------------------------------------------- hashing
def sha256_file(path: str | os.PathLike, bufsize: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tokenizer_fingerprint(model_path: str) -> str:
    h = hashlib.sha256()
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        p = Path(model_path) / name
        if p.exists():
            h.update(name.encode())
            h.update(sha256_file(p).encode())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------- rendering
def normalize_messages(messages: list[dict]) -> list[dict]:
    """Make records template-safe: tool_call arguments must be dicts (template iterates |items)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("content") is None:
            m["content"] = ""
        tcs = m.get("tool_calls")
        if tcs:
            new_tcs = []
            for tc in tcs:
                tc = dict(tc)
                fn = dict(tc.get("function", tc))
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args) if args.strip() else {}
                        fn["arguments"] = parsed if isinstance(parsed, dict) else {"input": parsed}
                    except json.JSONDecodeError:
                        fn["arguments"] = {"input": args}
                elif args is None:
                    fn["arguments"] = {}
                if "function" in tc:
                    tc["function"] = fn
                else:
                    tc = fn
                new_tcs.append(tc)
            m["tool_calls"] = new_tcs
        elif "tool_calls" in m:
            m.pop("tool_calls")
        out.append(m)
    return out


def _init_worker(model_path: str, preserve_thinking: bool):
    global _TOK, _TOK_CFG
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer

    _TOK = AutoTokenizer.from_pretrained(model_path)
    hdr = _TOK.encode("<|im_start|>assistant\n", add_special_tokens=False)
    _TOK_CFG = {
        "preserve_thinking": preserve_thinking,
        "hdr": hdr,
        "im_start": _TOK.convert_tokens_to_ids("<|im_start|>"),
        "im_end": _TOK.convert_tokens_to_ids("<|im_end|>"),
    }


def render_text(rec: dict, tok=None, preserve_thinking: bool = True) -> str:
    tok = tok or _TOK
    msgs = normalize_messages(rec["messages"])
    kwargs = {}
    if preserve_thinking:
        kwargs["preserve_thinking"] = True
    return tok.apply_chat_template(
        msgs, tools=rec.get("tools") or None, tokenize=False, add_generation_prompt=False, **kwargs
    )


def assistant_mask(ids: list[int] | np.ndarray, hdr: list[int], im_end: int) -> tuple[np.ndarray, int]:
    """1 for tokens inside assistant turns (after header, through <|im_end|> inclusive)."""
    ids = np.asarray(ids)
    n = len(ids)
    mask = np.zeros(n, dtype=np.uint8)
    h0, hl = hdr[0], len(hdr)
    starts = np.nonzero(ids == h0)[0]
    ends = np.nonzero(ids == im_end)[0]
    n_turns = 0
    for s in starts:
        if s + hl <= n and np.array_equal(ids[s : s + hl], hdr):
            b = s + hl
            j = np.searchsorted(ends, b)
            e = ends[j] + 1 if j < len(ends) else n  # unterminated turn: train to end
            mask[b:e] = 1
            n_turns += 1
    return mask, n_turns


def render_prompt(rec: dict, tok=None, preserve_thinking: bool = True) -> str:
    """Eval-server prompt for the record's final assistant turn: everything before it + generation prompt."""
    tok = tok or _TOK
    msgs = normalize_messages(rec["messages"])[:-1]
    kwargs = {"preserve_thinking": True} if preserve_thinking else {}
    return tok.apply_chat_template(
        msgs, tools=rec.get("tools") or None, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        **kwargs
    )


def tokenize_record(line: str) -> dict:
    rec = json.loads(line)
    ropt = rec.get("render") or {}
    preserve = bool(ropt.get("preserve_thinking", _TOK_CFG["preserve_thinking"]))
    last_only = rec.get("loss") == "last_assistant"
    try:
        text = render_text(rec, preserve_thinking=preserve)
        prompt = render_prompt(rec, preserve_thinking=preserve) if last_only else None
        if prompt is not None and not text.startswith(prompt):
            raise ValueError("eval prompt is not a prefix of the rendered record")
    except Exception as e:  # template errors -> drop sample, report
        return {"id": rec.get("id"), "error": f"{type(e).__name__}: {e}"[:300]}
    if prompt is None:
        ids = _TOK.encode(text, add_special_tokens=False)
        mask, n_turns = assistant_mask(ids, _TOK_CFG["hdr"], _TOK_CFG["im_end"])
    else:
        # prompt and generated part tokenized separately, exactly as at eval (the server tokenizes the
        # prompt; the model generates the rest), so boundary merges like "<think>\n" + "\n</think>" cannot occur
        p_ids = _TOK.encode(prompt, add_special_tokens=False)
        g_ids = _TOK.encode(text[len(prompt):], add_special_tokens=False)
        ids = p_ids + g_ids
        _, n_turns = assistant_mask(ids, _TOK_CFG["hdr"], _TOK_CFG["im_end"])
        mask = np.zeros(len(ids), dtype=np.uint8)
        ends = [i for i, t in enumerate(g_ids) if t == _TOK_CFG["im_end"]]
        e = ends[0] + 1 if ends else len(g_ids)
        mask[len(p_ids) : len(p_ids) + e] = 1
    n_asst = sum(1 for m in rec["messages"] if m.get("role") == "assistant")
    return {
        "id": rec.get("id"),
        "source": rec.get("source"),
        "ids": np.asarray(ids, dtype=np.int32),
        "mask": mask,
        "n_turns": n_turns,
        "n_asst_msgs": n_asst,
    }


# ----------------------------------------------------------------------------- cache
@dataclass
class TokenizedSet:
    name: str
    ids: np.ndarray  # int32 concat
    mask: np.ndarray  # uint8 concat
    offsets: np.ndarray  # int64 [n+1]
    meta: dict

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def __len__(self):
        return len(self.offsets) - 1

    def label_tokens(self, i: int, L: int) -> int:
        """# supervised targets when sample i is truncated to L tokens (targets are tokens 1..L-1)."""
        s = self.offsets[i]
        return int(self.mask[s + 1 : s + L].sum())


def _cache_paths(cache_dir: Path, key: str) -> dict:
    d = cache_dir / key
    return {"dir": d, "ids": d / "ids.npy", "mask": d / "mask.npy", "offsets": d / "offsets.npy",
            "meta": d / "meta.json"}


def build_or_load_cache(
    name: str,
    jsonl_path: str,
    model_path: str,
    cache_dir: str,
    preserve_thinking: bool = True,
    num_proc: int = 8,
    build: bool = True,
    wait_timeout_s: int = 7200,
    log=print,
) -> TokenizedSet:
    jsonl_path = str(jsonl_path)
    fsha = sha256_file(jsonl_path)
    key_src = json.dumps([CACHE_VERSION, fsha, tokenizer_fingerprint(model_path), bool(preserve_thinking)])
    key = f"{Path(jsonl_path).parent.name}_{Path(jsonl_path).stem}_{hashlib.sha256(key_src.encode()).hexdigest()[:16]}"
    cp = _cache_paths(Path(cache_dir), key)

    if not cp["meta"].exists():
        if not build:
            t0 = time.time()
            log(f"[data] waiting for cache {cp['dir']} (built by rank 0)")
            while not cp["meta"].exists():
                if time.time() - t0 > wait_timeout_s:
                    raise TimeoutError(f"cache {cp['dir']} not built in time")
                time.sleep(5)
        else:
            _build_cache(jsonl_path, fsha, model_path, cp, preserve_thinking, num_proc, log)
    meta = json.loads(cp["meta"].read_text())
    return TokenizedSet(
        name=name,
        ids=np.load(cp["ids"], mmap_mode="r"),
        mask=np.load(cp["mask"], mmap_mode="r"),
        offsets=np.load(cp["offsets"]),
        meta=meta,
    )


def _build_cache(jsonl_path, fsha, model_path, cp, preserve_thinking, num_proc, log):
    import multiprocessing as mp

    t0 = time.time()
    with open(jsonl_path) as f:
        lines = [ln for ln in f if ln.strip()]
    log(f"[data] tokenizing {len(lines)} records from {jsonl_path} with {num_proc} procs")
    ctx = mp.get_context("spawn")
    if num_proc > 1:
        with ctx.Pool(num_proc, initializer=_init_worker, initargs=(model_path, preserve_thinking)) as pool:
            results = pool.map(tokenize_record, lines, chunksize=max(1, len(lines) // (num_proc * 8) or 1))
    else:
        _init_worker(model_path, preserve_thinking)
        results = [tokenize_record(ln) for ln in lines]

    errors = [r for r in results if "error" in r]
    good = [r for r in results if "error" not in r and r["mask"][1:].sum() > 0]
    no_label = len(results) - len(errors) - len(good)
    turn_mismatch = sum(1 for r in good if r["n_turns"] != r["n_asst_msgs"])
    lens = np.array([len(r["ids"]) for r in good], dtype=np.int64)
    offsets = np.zeros(len(good) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lens)
    ids = np.concatenate([r["ids"] for r in good]) if good else np.zeros(0, np.int32)
    mask = np.concatenate([r["mask"] for r in good]) if good else np.zeros(0, np.uint8)

    tmp = cp["dir"].with_name(cp["dir"].name + f".tmp{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "ids.npy", ids)
    np.save(tmp / "mask.npy", mask)
    np.save(tmp / "offsets.npy", offsets)
    meta = {
        "cache_version": CACHE_VERSION,
        "source_file": jsonl_path,
        "source_sha256": fsha,
        "tokenizer_fingerprint": tokenizer_fingerprint(model_path),
        "preserve_thinking": preserve_thinking,
        "n_records": len(results),
        "n_kept": len(good),
        "n_template_errors": len(errors),
        "n_no_label": no_label,
        "n_turn_count_mismatch": turn_mismatch,
        "error_examples": errors[:5],
        "total_tokens": int(lens.sum()) if len(lens) else 0,
        "label_tokens": int(mask.sum()),
        "len_p50": float(np.percentile(lens, 50)) if len(lens) else 0,
        "len_p90": float(np.percentile(lens, 90)) if len(lens) else 0,
        "len_max": int(lens.max()) if len(lens) else 0,
        "ids_list": [r["id"] for r in good],
        "sources": [r["source"] for r in good],
        "build_seconds": round(time.time() - t0, 1),
    }
    (tmp / "meta.json").write_text(json.dumps(meta))
    try:
        os.replace(tmp, cp["dir"])
    except OSError:  # another process won the race
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    log(f"[data] cached {len(good)} samples, {meta['total_tokens']/1e6:.2f}M tokens "
        f"({meta['label_tokens']/1e6:.2f}M label) errors={len(errors)} no_label={no_label} "
        f"turn_mismatch={turn_mismatch} in {meta['build_seconds']}s -> {cp['dir']}")


# ----------------------------------------------------------------------------- synthetic
def synthetic_set(n_samples: int, min_len: int, max_len: int, vocab: int = 150000, seed: int = 0,
                  label_frac: float = 0.7) -> TokenizedSet:
    rng = np.random.default_rng(seed)
    lens = rng.integers(min_len, max_len + 1, size=n_samples)
    offsets = np.zeros(n_samples + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lens)
    ids = rng.integers(1000, vocab, size=int(offsets[-1]), dtype=np.int32)
    mask = (rng.random(int(offsets[-1])) < label_frac).astype(np.uint8)
    return TokenizedSet("synthetic", ids, mask, offsets, {"synthetic": True, "total_tokens": int(offsets[-1])})


# ----------------------------------------------------------------------------- packing
# A "piece" = (set_idx, sample_idx, length_used). A "pack" = list of pieces with total length <= max_len.
def pack_pieces(pieces: list[tuple[int, int, int]], max_len: int, rng: random.Random | None,
                window: int = 512) -> list[list[tuple[int, int, int]]]:
    order = list(pieces)
    if rng is not None:
        rng.shuffle(order)
    packs: list[list] = []
    for w0 in range(0, len(order), window):
        win = sorted(order[w0 : w0 + window], key=lambda p: -p[2])
        bins: list[list] = []
        free: list[int] = []
        for p in win:
            placed = False
            for bi, fr in enumerate(free):
                if fr >= p[2]:
                    bins[bi].append(p)
                    free[bi] -= p[2]
                    placed = True
                    break
            if not placed:
                bins.append([p])
                free.append(max_len - p[2])
        packs.extend(bins)
    if rng is not None:
        rng.shuffle(packs)
    return packs


def make_pieces(sets: list[TokenizedSet], max_len: int, overlong: str = "truncate",
                max_tokens: int | None = None) -> tuple[list[tuple[int, int, int]], dict]:
    pieces, n_trunc, n_drop, tot = [], 0, 0, 0
    for si, s in enumerate(sets):
        for i, L in enumerate(s.lengths.tolist()):
            if L > max_len:
                if overlong == "drop":
                    n_drop += 1
                    continue
                n_trunc += 1
                L = max_len
            if L < 2:
                continue
            if max_tokens is not None and tot + L > max_tokens and pieces:
                break
            pieces.append((si, i, L))
            tot += L
    return pieces, {"n_pieces": len(pieces), "n_truncated": n_trunc, "n_dropped": n_drop, "tokens": tot}


def build_pack_arrays(pack, sets: list[TokenizedSet]):
    """-> input_ids, labels (next-token targets, -100 = ignore), position_ids, cu_seqlens (np arrays)."""
    ids_l, lab_l, pos_l, cu = [], [], [], [0]
    for si, i, L in pack:
        s = sets[si]
        o = int(s.offsets[i])
        ids = np.asarray(s.ids[o : o + L], dtype=np.int64)
        m = np.asarray(s.mask[o : o + L])
        lab = np.full(L, -100, dtype=np.int64)
        tgt_mask = m[1:].astype(bool)
        lab[:-1][tgt_mask] = ids[1:][tgt_mask]
        ids_l.append(ids)
        lab_l.append(lab)
        pos_l.append(np.arange(L, dtype=np.int64))
        cu.append(cu[-1] + L)
    return (np.concatenate(ids_l), np.concatenate(lab_l), np.concatenate(pos_l), np.asarray(cu, dtype=np.int32))


def pack_label_tokens(pack, sets: list[TokenizedSet]) -> int:
    return sum(sets[si].label_tokens(i, L) for si, i, L in pack)


def pack_tokens(pack) -> int:
    return sum(L for _, _, L in pack)


def epoch_plan(pieces, max_len: int, seed: int, epoch: int, window: int) -> list[list[tuple[int, int, int]]]:
    return pack_pieces(pieces, max_len, random.Random(seed * 1000 + epoch), window)


def steps_from_packs(n_packs: int, packs_per_step: int) -> int:
    return math.ceil(n_packs / packs_per_step)
