"""Shared utilities for the data pipeline: config, paths, IO, hashing, n-grams, tokenisation.

Unified sample schema (one JSON object per line):
  {"id": str, "source": str, "instance_id": str, "messages": [...], "tools": [...] | None,
   "n_tokens": int, "meta": {...}}
  message = {"role": system|user|assistant|tool, "content": str,
             "reasoning_content"?: str, "tool_calls"?: [{"id","type":"function",
             "function":{"name","arguments": dict}}], "tool_call_id"?: str}
Tool-call arguments are stored as a JSON *object* (dict), which is what the Qwen3.6 chat
template iterates over (`tool_call.arguments|items`); OpenAI wire format would be a string.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import yaml

REPO = Path(__file__).resolve().parents[2]
APEX = Path(os.environ.get("APEX", "/scratch/eh3323/apex"))
CFG_PATH = REPO / "configs" / "data" / "sources.yaml"

FORBIDDEN_SUBSTR = "seed31415"   # ASI-Bench split with reference answers: never an input


def load_cfg() -> dict:
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


CFG = load_cfg()


def apex_path(key: str) -> Path:
    return APEX / CFG["paths"][key]


def guard_path(p) -> Path:
    """Every input path goes through this: hard-fail if it touches the answer-bearing split."""
    s = str(p)
    assert FORBIDDEN_SUBSTR not in s.lower(), f"refusing to read {s}: contains {FORBIDDEN_SUBSTR}"
    return Path(s)


def raw_dir(source_key: str) -> Path:
    return guard_path(apex_path("raw") / source_key)


def interim_dir(expert: str) -> Path:
    d = apex_path("interim") / expert
    d.mkdir(parents=True, exist_ok=True)
    return d


def processed_dir(expert: str) -> Path:
    d = apex_path("processed") / expert
    d.mkdir(parents=True, exist_ok=True)
    return d


def bench(*parts) -> Path:
    return guard_path(apex_path("benchmarks").joinpath(*parts))


# ---------------------------------------------------------------- IO
def read_jsonl(path) -> Iterator[dict]:
    with open(guard_path(path)) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            # keep key insertion order: tool schemas / arguments are rendered by the chat template with
            # `tojson`, so re-ordering keys would change the training text (canonical_json is for hashing only)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sample_sha(row: dict) -> str:
    """Content hash of a sample (messages + tools), independent of id/meta/n_tokens."""
    return sha256_text(canonical_json({"messages": row["messages"], "tools": row.get("tools")}))


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash01(s: str, salt: str = "") -> float:
    """Deterministic pseudo-random number in [0,1) from a string (used for splits/sampling)."""
    h = hashlib.sha256((salt + "|" + s).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def rng(salt: str = "") -> random.Random:
    return random.Random(f"{CFG['seed']}|{salt}")


# ---------------------------------------------------------------- text normalisation / n-grams
_WORD = re.compile(r"[a-z0-9]+")
_LATEXISH = re.compile(r"^(?:\d+|[a-z])$")   # tokens that are pure digits or single letters


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"\\(?:left|right|,|;|!|quad|qquad|displaystyle|text|mathrm|mathbf)\b", " ", s)
    s = re.sub(r"[\s$]+", " ", s)
    return s.strip()


def words(s: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFKC", s or "").lower())


def ngrams(s: str, n: int = 13, skip_trivial: bool = False) -> set[str]:
    w = words(s)
    out = set()
    for i in range(len(w) - n + 1):
        g = w[i:i + n]
        if skip_trivial and sum(1 for t in g if not _LATEXISH.match(t)) < 4:
            # gram made (almost) only of digits / single-letter LaTeX symbols (matrix literals etc.)
            continue
        out.add(" ".join(g))
    return out


class NgramIndex:
    """gram -> set(ref ids); drops boilerplate grams shared by >= max_refs references."""

    def __init__(self, n: int = 13, max_refs: int = 5, skip_trivial: bool = False):
        self.n, self.max_refs, self.skip_trivial = n, max_refs, skip_trivial
        self.idx: dict[str, set] = {}

    def add(self, ref_id: str, text: str):
        for g in ngrams(text, self.n, self.skip_trivial):
            self.idx.setdefault(g, set()).add(ref_id)

    def finalize(self):
        if self.max_refs:
            self.idx = {g: s for g, s in self.idx.items() if len(s) < self.max_refs}
        return self

    def hits(self, text: str) -> tuple[int, set]:
        refs, k = set(), 0
        for g in ngrams(text, self.n, self.skip_trivial):
            r = self.idx.get(g)
            if r:
                k += 1
                refs |= r
        return k, refs

    def __len__(self):
        return len(self.idx)


# ---------------------------------------------------------------- think-tag splitting
_THINK = re.compile(r"^\s*<think>(.*?)</think>\s*", re.S)


def split_think(content: str) -> tuple[str | None, str]:
    """'<think>R</think>C' -> (R, C). Content without think block -> (None, content)."""
    if content is None:
        return None, ""
    m = _THINK.match(content)
    if m:
        return m.group(1).strip(), content[m.end():].strip()
    if "</think>" in content and "<think>" not in content.split("</think>")[0][:20]:
        r, c = content.split("</think>", 1)
        return r.replace("<think>", "").strip(), c.strip()
    return None, content.strip()


# ---------------------------------------------------------------- tokenizer / chat template
_TOK = None
_TMPL = None


def get_tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(str(apex_path("model_dir")))
    return _TOK


def get_template() -> str:
    global _TMPL
    if _TMPL is None:
        _TMPL = (apex_path("model_dir") / "chat_template.jinja").read_text()
    return _TMPL


def render(messages: list[dict], tools: list | None) -> str:
    """Render with the Qwen3.6 chat template, preserve_thinking=True (thinking kept in every
    assistant turn, matching how we train; the template default strips thinking before the
    last user query)."""
    tok = get_tokenizer()
    return tok.apply_chat_template(
        messages, tools=tools or None, chat_template=get_template(), tokenize=False,
        add_generation_prompt=False, preserve_thinking=True)


def count_tokens(messages: list[dict], tools: list | None) -> int:
    text = render(messages, tools)
    return len(get_tokenizer()(text, add_special_tokens=False)["input_ids"])


def est_tokens(messages: list[dict], tools: list | None) -> int:
    """Cheap upper-ish estimate used to skip rendering obviously-too-long samples."""
    n = sum(len(m.get("content") or "") + len(m.get("reasoning_content") or "")
            + len(json.dumps(m.get("tool_calls") or "")) for m in messages)
    n += len(json.dumps(tools)) if tools else 0
    return int(n / 3.6)


# ---------------------------------------------------------------- validation
ROLES = {"system", "user", "assistant", "tool"}


def validate_sample(row: dict) -> list[str]:
    errs = []
    for k in ("id", "source", "messages", "n_tokens"):
        if k not in row:
            errs.append(f"missing {k}")
    msgs = row.get("messages") or []
    if not msgs:
        errs.append("no messages")
    pending = set()
    for i, m in enumerate(msgs):
        r = m.get("role")
        if r not in ROLES:
            errs.append(f"msg{i}: bad role {r}")
        if not isinstance(m.get("content", ""), str):
            errs.append(f"msg{i}: content not str")
        if r == "system" and i != 0:
            errs.append(f"msg{i}: system not first")
        if r == "assistant":
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc.get("function", {}).get("arguments"), dict):
                    errs.append(f"msg{i}: tool args not dict")
                pending.add(tc.get("id"))
        if r == "tool":
            if m.get("tool_call_id") not in pending:
                errs.append(f"msg{i}: tool_call_id {m.get('tool_call_id')} without call")
    if msgs and msgs[-1]["role"] != "assistant":
        errs.append("last message is not assistant")
    if not any(m["role"] == "user" for m in msgs):
        errs.append("no user message")
    return errs


class DropCounter(Counter):
    def drop(self, reason: str, n: int = 1):
        self[reason] += n


def log(*a):
    print(*a, file=sys.stderr, flush=True)
