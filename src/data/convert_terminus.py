"""Terminus-2 (Harbor) trajectories: parsing + normalisation to the *current* Harbor template.

Corpora (Nemotron-Terminal-Corpus, OpenThoughts-Agent-SFT-100K) are Harbor `traces export` rows:
conversations = [user(prompt template + task), assistant, user(observation), ...].

Template check (2026-10-13, harbor_repo @4778ceba, templates/terminus-json-plain.txt):
  * OpenThoughts-Agent-SFT-100K first user message == current template (byte-identical prefix).
  * Nemotron-Terminal-Corpus uses an older Terminus-2 template revision ("Most bash commands should end
    with a newline (\\n) to cause them to execute" instead of "You must end every command with a newline
    (\\n) or it will not execute.").  => we ALWAYS re-render the first user message with the exact current
    Harbor template: template.format(instruction=<task>, terminal_state=<initial screen>), which is what
    Terminus-2 does at run time (terminus_2.py ~L1725). Later observation strings ("New Terminal Output:",
    "Current Terminal Screen:", completion confirmation, parse-error feedback) already match Harbor.
  * Thinking: `<think>..</think>` prefix of assistant content -> reasoning_content (vLLM reasoning parser
    behaviour at eval). OT-Agent's teacher emitted the think block inline, which made Harbor add
    "WARNINGS: - Extra text detected before JSON object" to the next observation; GLM also often wrapped
    the JSON in ```json fences. We strip the fences and remove the "Extra text detected before/after JSON
    object" warning items (every kept assistant turn is enforced to be a bare, valid JSON object, so such
    warnings could not occur for the normalised turns).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from .common import apex_path, guard_path, split_think


class Unconvertible(Exception):
    pass


@lru_cache(None)
def harbor_template() -> str:
    return guard_path(apex_path("terminus_template")).read_text()


def render_first_user(instruction: str, terminal_state: str) -> str:
    return harbor_template().format(instruction=instruction, terminal_state=terminal_state)


_SUMMARY_MARKERS = ("[SYSTEM WARNING]", "You are picking up work from a previous",
                    "Here are the answers the other agent provided")


def parse_first_user(text: str) -> tuple[str, str]:
    k = text.find("Task Description:\n")
    j = text.rfind("\n\nCurrent terminal state:\n")
    if k < 0 or j < 0 or j < k:
        raise Unconvertible("first_user_unparseable")
    return text[k + len("Task Description:\n"):j], text[j + len("\n\nCurrent terminal state:\n"):]


_WARN = re.compile(r"^Previous response had warnings:\nWARNINGS: (.*?)\n\n", re.S)
_EXTRA = {"Extra text detected before JSON object", "Extra text detected after JSON object"}
_FENCE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.S)


def clean_observation(text: str) -> str:
    m = _WARN.match(text)
    if not m:
        return text
    items = [ln[2:] if ln.startswith("- ") else ln for ln in m.group(1).split("\n")]
    items = [x for x in items if x.strip() and x.strip() not in _EXTRA]
    rest = text[m.end():]
    if not items:
        return rest
    return "Previous response had warnings:\nWARNINGS: " + "\n".join("- " + x for x in items) + "\n\n" + rest


def parse_t2_json(content: str) -> dict:
    try:
        d = json.loads(content)
    except Exception:
        raise Unconvertible("assistant_not_pure_json")
    if not isinstance(d, dict) or not {"analysis", "plan", "commands"} <= set(d):
        raise Unconvertible("assistant_missing_fields")
    if not isinstance(d["commands"], list):
        raise Unconvertible("assistant_bad_commands")
    return d


def normalize_t2(convs: list[dict], max_parse_errors: int = 0) -> tuple[list[dict], dict]:
    """Returns (messages in unified schema, stats). Keeps the native Terminus-2 text protocol."""
    if isinstance(convs, str):
        convs = json.loads(convs)
    if not convs or convs[0]["role"] != "user":
        raise Unconvertible("bad_first_role")
    if convs[-1]["role"] != "assistant":
        convs = convs[:-1]
    instr, term = parse_first_user(convs[0]["content"])
    msgs = [{"role": "user", "content": render_first_user(instr, term)}]
    parse_err = 0
    turns = []
    for i, m in enumerate(convs[1:], start=1):
        want = "assistant" if i % 2 == 1 else "user"
        if m["role"] != want:
            raise Unconvertible("role_alternation")
        c = m["content"] or ""
        if want == "user":
            if any(s in c for s in _SUMMARY_MARKERS):
                raise Unconvertible("summarization_or_system_warning")
            if c.startswith("Previous response had parsing errors"):
                parse_err += 1
            msgs.append({"role": "user", "content": clean_observation(c)})
        else:
            r, body = split_think(c)
            if r is None:
                raise Unconvertible("no_think_block")
            fm = _FENCE.match(body)
            if fm:                       # ```json fenced reply (GLM habit) -> bare JSON object
                body = fm.group(1).strip()
            msgs.append({"role": "assistant", "content": body, "reasoning_content": r})
            turns.append(body)
    if parse_err > max_parse_errors:
        raise Unconvertible("parse_errors")
    parsed = []
    for b in turns:
        parsed.append(parse_t2_json(b))
    complete = bool(parsed) and parsed[-1].get("task_complete") is True
    return msgs, {"instruction": instr, "terminal_state": term, "n_turns": len(turns),
                  "complete": complete, "parsed": parsed}
