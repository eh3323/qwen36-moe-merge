"""Text QA -> official HLE answer format.

  system    = HLE SYSTEM_PROMPT (centerforaisafety/hle hle_eval/run_model_predictions.py L11, verbatim)
  user      = question (MCQ rewritten to HLE style: "...\\n\\nAnswer Choices:\\nA. ...\\nB. ...")
  assistant = reasoning_content: teacher reasoning trace (kept)
              content: "Explanation: <teacher's visible solution, answer wrapper removed>\\n"
                       "Answer: <final answer>\\nConfidence: <c>%"

Confidence heuristic (documented in data/README.md):
  * Nemotron-Math-v2 rows carry teacher pass statistics (gpt-oss-120b, 8 samples per reasoning mode).
    c = Laplace-smoothed pass rate of the *high, no-tool* mode = (pass+1)/(count+2), i.e. an empirical
    estimate of how often this kind of reasoning reaches the reference answer.
  * Other sources have no pass statistics: c = source prior (MCQ 75, RQA 65, OSR-2 non-STEM 70)
    - 5 x min(4, hedges per 1k reasoning words) - 10 if reasoning > 3k words - 10 more if > 6k words.
  * clipped to [10, 95] and rounded to a multiple of 5.
"""
from __future__ import annotations

import re

HLE_SYSTEM_PROMPT = ("Your response should be in the following format:\nExplanation: {your explanation for your "
                     "answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between "
                     "0% and 100% for your answer}")

_BOXED_INSTR = re.compile(r"^\s*(Solve the following (?:math )?problem\.?\s*)?Make sure to put the answer "
                          r"\(and only answer\) inside \\boxed\{\}\.?\s*", re.I)
_OPT_LINE = re.compile(r"^\s*\(?([A-J])[\)\.:]\s+(.*)$")


class Unconvertible(Exception):
    pass


def strip_instructions(q: str) -> str:
    q = q.replace("\x08oxed", "\\boxed")        # upstream escaping bug: "\boxed" stored as backspace+"oxed"
    q = _BOXED_INSTR.sub("", q.strip())
    q = re.sub(r"\n+\s*(Please )?(put|return|provide) (your|the) (final )?answer[^\n]*\\boxed\{\}[^\n]*$", "", q,
               flags=re.I)
    return q.strip()


def to_hle_mcq(q: str) -> tuple[str, list[str]]:
    """Rewrite 'A) foo' / 'A: foo' option lines into HLE 'Answer Choices:\\nA. foo'."""
    lines = q.split("\n")
    first = next((i for i, ln in enumerate(lines) if _OPT_LINE.match(ln) and _OPT_LINE.match(ln).group(1) == "A"),
                 None)
    if first is None:
        raise Unconvertible("mcq_no_options")
    stem = "\n".join(lines[:first]).rstrip()
    opts, letters = [], []
    for ln in lines[first:]:
        m = _OPT_LINE.match(ln)
        if m:
            opts.append(f"{m.group(1)}. {m.group(2).strip()}")
            letters.append(m.group(1))
        elif ln.strip() and opts:
            opts[-1] += " " + ln.strip()
    if len(opts) < 2 or letters != [chr(65 + i) for i in range(len(letters))]:
        raise Unconvertible("mcq_bad_options")
    return f"{stem}\n\nAnswer Choices:\n" + "\n".join(opts), letters


def last_boxed(s: str) -> str | None:
    k = s.rfind("\\boxed{")
    if k < 0:
        return None
    i, depth, out = k + len("\\boxed{"), 1, []
    while i < len(s):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(ch)
        i += 1
    return None


def mcq_letter(content: str, letters: list[str]) -> str | None:
    b = last_boxed(content)
    cands = []
    if b:
        cands.append(b)
    cands += re.findall(r"(?:[Aa]nswer|correct (?:answer|option|choice))\s*(?:is|:)?\s*\**\(?([A-J])\b", content)
    cands += re.findall(r"\*\*\(?([A-J])[\).:]", content)
    for c in cands:
        c = re.sub(r"[^A-J]", "", c.strip().upper().replace("TEXT", ""))
        if len(c) == 1 and c in letters:
            return c
    return None


def norm_answer(a: str) -> str:
    """Lenient canonical form for answer matching (LaTeX spacing/sizing, braces, $, \\( \\) removed)."""
    a = (a or "").strip()
    a = re.sub(r"\\(?:displaystyle|,|;|!|quad|qquad)", " ", a).strip()
    a = re.sub(r"^\$+|\$+$", "", a)
    a = a.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    a = re.sub(r"\\(?:displaystyle|left|right|big|Big|bigl|bigr|,|;|!|:|quad|qquad|mathrm|text|operatorname)", "", a)
    a = re.sub(r"\\[dt]frac", r"\\frac", a)
    a = re.sub(r"^[a-zA-Z]\w*(?:\([^)]*\))?'?(?:\([^)]*\))?\s*=\s*", "", a)   # leading "f'(x) =" / "x ="
    a = re.sub(r"[{}\s]", "", a)
    return a.strip(".").lower()


def answers_match(got: str, exp: str) -> bool:
    g, e = norm_answer(got), norm_answer(exp)
    if not g or not e:
        return False
    if g == e:
        return True
    # verbose boxed answers ("the largest n ... is 5a^2+30a-36"): accept containment of a non-trivial reference
    return len(e) >= 4 and (g.endswith(e) or ("=" + e) in g or ("is" + e) in g)


def clean_answer(a: str) -> str:
    a = (a or "").strip()
    a = re.sub(r"^\$+|\$+$", "", a)
    a = re.sub(r"^\\\(|\\\)$", "", a)
    a = re.sub(r"^(\\[,;!]|\s)+|(\\[,;!]|\s)+$", "", a)
    return re.sub(r"\s*\n\s*", " ", a).strip()


def explanation(content: str) -> str:
    c = content.strip()
    c = re.sub(r"\\\[\s*\\boxed\{", r"\\[{", c)
    c = re.sub(r"\\boxed\{", "{", c)
    c = re.sub(r"\n+\s*(\*\*)?(Final )?Answer:?(\*\*)?[^\n]*$", "", c, flags=re.I)
    c = re.sub(r"\n{3,}", "\n\n", c)
    return c.strip()


_HEDGES = re.compile(r"\b(not sure|maybe|perhaps|might be|i think|guess|unclear|ambiguous|hmm|wait|"
                     r"alternatively|double-check|uncertain|probably|seems)\b", re.I)
PRIORS = {"nemotron_science_mcq": 75, "nemotron_science_rqa": 65, "osr2_nonstem": 70}


def confidence(source: str, reasoning: str, math_meta: dict | None = None) -> int:
    if math_meta:
        m = math_meta.get("reason_high_no_tool") or {}
        p = (int(m.get("pass", 0)) + 1) / (int(m.get("count", 0)) + 2)
        c = 100 * p
    else:
        w = max(1, len(reasoning.split()))
        hedge_rate = len(_HEDGES.findall(reasoning)) * 1000 / w
        c = PRIORS.get(source, 65) - 5 * min(4, int(hedge_rate)) - (10 if w > 3000 else 0) - (10 if w > 6000 else 0)
    c = min(95, max(10, c))
    return int(5 * round(c / 5))


def hle_messages(question: str, reasoning: str, expl: str, answer: str, conf: int) -> list[dict]:
    content = f"Explanation: {expl}\nAnswer: {answer}\nConfidence: {conf}%"
    return [{"role": "system", "content": HLE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": content, "reasoning_content": reasoning.strip()}]


# ------------------------------------------------------------------ per-source adapters
def from_nemotron_science(row: dict, kind: str) -> tuple[str, list[dict], dict]:
    msgs = row["messages"]
    u = next(m for m in msgs if m["role"] == "user")["content"]
    a = next(m for m in msgs if m["role"] == "assistant")
    r = a.get("reasoning_content") or ""
    if not r or r == "None":
        raise Unconvertible("no_reasoning")
    q = strip_instructions(u)
    if kind == "mcq":
        q, letters = to_hle_mcq(q)
        ans = mcq_letter(a["content"], letters)
        if not ans:
            raise Unconvertible("mcq_no_answer")
        src = "nemotron_science_mcq"
    else:
        ans = clean_answer(last_boxed(a["content"]) or "")   # multi-line boxed (aligned/array) -> one line
        if not ans:
            raise Unconvertible("rqa_no_boxed")
        src = "nemotron_science_rqa"
    conf = confidence(src, r)
    return q, hle_messages(q, r, explanation(a["content"]), ans, conf), {"confidence": conf, "answer": ans}


def from_nemotron_math(row: dict) -> tuple[str, list[dict], dict]:
    msgs = row["messages"]
    if any(m.get("tool_calls") for m in msgs) or row.get("tools"):
        raise Unconvertible("math_tool_use")
    a = next(m for m in msgs if m["role"] == "assistant")
    r = a.get("reasoning_content") or ""
    if not r:
        raise Unconvertible("no_reasoning")
    got = last_boxed(a["content"] or "")
    exp = row.get("expected_answer") or ""
    if not got or not answers_match(got, exp):
        raise Unconvertible("math_answer_mismatch")
    got = clean_answer(exp)        # concise reference form for the "Answer:" line
    q = strip_instructions(row.get("problem") or next(m for m in msgs if m["role"] == "user")["content"])
    meta = row.get("metadata")
    if isinstance(meta, str):
        import ast
        meta = ast.literal_eval(meta) if meta.strip().startswith("{'") else __import__("json").loads(meta)
    conf = confidence("nemotron_math_aops", r, meta or {})
    return q, hle_messages(q, r, explanation(a["content"]), got, conf), {"confidence": conf, "answer": got}


def from_osr2(row: dict) -> tuple[str, list[dict], dict]:
    r, body = None, row.get("output") or ""
    m = re.match(r"\s*<think>(.*?)</think>\s*(.*)$", body, re.S)
    if not m:
        raise Unconvertible("osr_no_think")
    r, body = m.group(1).strip(), m.group(2).strip()
    exp = (row.get("expected_answer") or "").strip()
    q = strip_instructions(row.get("input") or "")
    got = last_boxed(body)
    if re.fullmatch(r"[A-J]", exp):
        q, letters = to_hle_mcq(q)
        got = mcq_letter(body, letters)
        if got != exp:
            raise Unconvertible("osr_answer_mismatch")
    elif not got or not answers_match(got, exp):
        raise Unconvertible("osr_answer_mismatch")
    else:
        got = clean_answer(exp)
    conf = confidence("osr2_nonstem", r)
    return q, hle_messages(q, r, explanation(body), got, conf), {"confidence": conf, "answer": got}


_NONSTEM = re.compile(r"\b(law|legal|court|statute|plaintiff|defendant|constitution(al)?|contract|tort|"
                      r"econom\w*|market|monetary|fiscal|gdp|inflation|philosoph\w*|ethic\w*|moral|kant|"
                      r"history|historical|century|empire|war|treaty|dynasty|literature|novel|poem|poet|"
                      r"author|linguistic\w*|language|grammar|religio\w*|theolog\w*|art|painting|music\w*|"
                      r"sociolog\w*|anthropolog\w*|political|politics|psycholog\w*|archaeolog\w*)\b", re.I)
_STEMY = re.compile(r"\b(equation|integral|derivative|molecule|enzyme|protein|electron|quantum|velocity|"
                    r"compute|calculate|algorithm|matrix|reaction|orbital|voltage|theorem)\b", re.I)


def is_nonstem(text: str) -> bool:
    return len(_NONSTEM.findall(text)) >= 2 and not _STEMY.search(text)
