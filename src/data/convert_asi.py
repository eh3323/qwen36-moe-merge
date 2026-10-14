"""E-ASI conversion -> opencode (v1.17.15) native tool calling with ONLY the `bash` tool.

Target harness: ASI-Bench `opencode_cli` adapter (pinned opencode 1.17.15) talking OpenAI Chat
Completions to vLLM. Reproduced pieces (vendored in configs/data/opencode/, fetched from
github.com/anomalyco/opencode tag v1.17.15 = 5fb0d1cd):
  * system prompt = session/prompt/default.txt (provider prompt chosen for a non-GPT/Claude/Gemini model id)
    + "\\n" + environment block from session/system.ts (model id, <env> working dir / git / platform / date).
    APPROXIMATION: we omit the skills block and any AGENTS.md instructions (none exist in an ASI workspace;
    the skills block depends on locally installed skills) — capture a real request to verify.
  * bash tool = tool/shell/prompt.ts rendered for shell "bash", platform linux, 2000 lines / 51200 bytes,
    120000 ms default timeout (render_bash.js executes the original template code); parameters
    {command, timeout?, workdir?} from parameterSchema() (JSON-schema rendering of the effect Schema is
    approximated: integer timeout with exclusiveMinimum 0, additionalProperties false).
  * tool result text = raw combined output; "(no output)" when empty; > 2000 lines / 51200 bytes -> tail +
    "...output truncated...\\n\\nFull output saved to: <file>\\n\\n" prefix (tool/shell.ts). No exit code shown.
  * end of episode = an assistant message without tool calls (opencode run exits).
  User decision: only the bash tool is exposed in training (opencode at eval also offers read/write/edit/...).

Sources:
  Nemotron-Terminal (Terminus-2 JSON) -> per assistant turn: reasoning_content = <think>, content = plan,
    one bash call whose command = the turn's keystrokes joined by newlines; `workdir` set when the tmux
    prompt shows a cwd other than /app. Observation = tmux output with shell prompts / echoed commands /
    heredoc continuation lines removed. Final message = analysis of the completing turn.
  jupyter-agent (thinking) -> cumulative script: every cell is appended to /app/analysis.py and the script
    is re-run (python3 /app/analysis.py); observation = concatenated outputs of cells 1..k (deterministic
    re-run semantics). Last bare expression of a cell is wrapped in print(). Final message = final_answer.
"""
from __future__ import annotations

import ast
import json
import re
from functools import lru_cache

from .common import CFG, REPO, split_think
from .convert_terminus import Unconvertible, normalize_t2

OC_DIR = REPO / "configs" / "data" / "opencode"
WORKDIR = CFG["experts"]["asi"]["opencode"]["workdir"]


@lru_cache(None)
def bash_tool() -> dict:
    return json.loads((OC_DIR / "bash_tool.json").read_text())


@lru_cache(None)
def system_prompt() -> str:
    oc = CFG["experts"]["asi"]["opencode"]
    env = "\n".join([
        f"You are powered by the model named {oc['model_api_id']}. The exact model ID is "
        f"{oc['provider_id']}/{oc['model_api_id']}",
        "Here is some useful information about the environment you are running in:",
        "<env>",
        f"  Working directory: {WORKDIR}",
        f"  Workspace root folder: {WORKDIR}",
        "  Is directory a git repo: no",
        "  Platform: linux",
        f"  Today's date: {oc['date']}",
        "</env>"])
    return "\n".join([(OC_DIR / "default.txt").read_text(), env])


MAX_LINES, MAX_BYTES = 2000, 50 * 1024


def oc_output(text: str, call_idx: int) -> str:
    text = text if text.strip() else ""
    if not text:
        return "(no output)"
    lines = text.split("\n")
    if len(lines) <= MAX_LINES and len(text.encode()) <= MAX_BYTES:
        return text
    out, size = [], 0
    for ln in reversed(lines):
        b = len(ln.encode()) + 1
        if len(out) >= MAX_LINES or size + b > MAX_BYTES:
            break
        out.append(ln)
        size += b
    return (f"...output truncated...\n\nFull output saved to: /tmp/opencode/tool_output_{call_idx:04d}.txt\n\n"
            + "\n".join(reversed(out)))


# ------------------------------------------------------------------ Terminus-2 -> opencode
_PROMPT = re.compile(r"(?:\([\w.-]+\) )?root@[\w.-]+:([^#\n]*)# ?")
_INTERACTIVE = re.compile(r"^\s*(vim?|nano|less|more|top|htop|man|ipython|python3?|bash|sh|tmux|watch|ssh)\s*$")
_SPECIAL_KEY = re.compile(r"^(C-[a-z]|M-[a-z]|Escape|Enter|Up|Down|Left|Right|Tab|q)$")


def _strip_terminal(obs: str) -> tuple[str, str | None]:
    """Remove tmux prompts, echoed commands and heredoc continuations; return (output, last cwd)."""
    for p in ("New Terminal Output:\n", "Current Terminal Screen:\n"):
        if obs.startswith(p):
            obs = obs[len(p):]
    out, cwd, in_heredoc = [], None, False
    for ln in obs.split("\n"):
        m = _PROMPT.search(ln)
        if m:
            pre = ln[:m.start()]
            if pre.strip():
                out.append(pre)
            cwd = m.group(1).strip() or cwd
            cmd = ln[m.end():]
            in_heredoc = "<<" in cmd
            continue
        if in_heredoc and (ln.startswith("> ") or ln == ">"):
            continue
        in_heredoc = False
        out.append(ln)
    text = "\n".join(out).strip("\n")
    return (text + "\n" if text else ""), cwd


def _cwd_abs(cwd: str | None) -> str | None:
    if not cwd:
        return None
    if cwd.startswith("~"):
        return "/root" + cwd[1:]
    return cwd


def t2_to_opencode(convs) -> tuple[list[dict], dict]:
    msgs, st = normalize_t2(convs, max_parse_errors=0)
    if not st["complete"]:
        raise Unconvertible("not_complete")
    parsed = st["parsed"]
    out = [{"role": "system", "content": system_prompt()},
           {"role": "user", "content": st["instruction"].strip()}]
    # msgs: [user0, a1, u1, a2, u2, ...]
    cwd = WORKDIR
    k = 0
    for t, d in enumerate(parsed):
        a_msg = msgs[1 + 2 * t]
        obs = msgs[2 + 2 * t]["content"] if 2 + 2 * t < len(msgs) else None
        if obs is not None and obs.startswith("Previous response had warnings"):
            obs = obs.split("\n\n", 1)[1] if "\n\n" in obs else ""
        if obs is not None and ("timed out after" in obs[:300]):
            raise Unconvertible("command_timeout")
        keys = [c.get("keystrokes", "") for c in d["commands"] if isinstance(c, dict)]
        for ks in keys:
            if _SPECIAL_KEY.match(ks.strip()) or (ks.strip() and not ks.endswith("\n")):
                raise Unconvertible("interactive_keystrokes")
            if _INTERACTIVE.match(ks.strip()):
                raise Unconvertible("interactive_program")
        cmd = "\n".join(ks.rstrip("\n") for ks in keys if ks.strip())
        done = d.get("task_complete") is True
        if not cmd and not done:
            raise Unconvertible("wait_turn")
        if cmd:
            args = {"command": cmd}
            if cwd != WORKDIR:
                args["workdir"] = cwd
            cid = f"call_{k}"
            out.append({"role": "assistant", "content": d.get("plan", "").strip(),
                        "reasoning_content": a_msg["reasoning_content"],
                        "tool_calls": [{"id": cid, "type": "function",
                                        "function": {"name": "bash", "arguments": args}}]})
            if obs is None:
                raise Unconvertible("missing_observation")
            body = obs
            if body.startswith("Current terminal state:\n"):    # completion confirmation wrapper
                body = body[len("Current terminal state:\n"):].split("\n\nAre you sure you want to mark", 1)[0]
            text, new_cwd = _strip_terminal(body)
            cwd = _cwd_abs(new_cwd) or cwd
            out.append({"role": "tool", "tool_call_id": cid, "content": oc_output(text, k)})
            k += 1
        if done:
            if cmd:   # need a final no-tool message: use the confirming turn if present
                nxt = parsed[t + 1] if t + 1 < len(parsed) else None
                r = msgs[1 + 2 * (t + 1)]["reasoning_content"] if nxt else a_msg["reasoning_content"]
                final = (nxt or d).get("analysis", "").strip()
            else:
                r, final = a_msg["reasoning_content"], d.get("analysis", "").strip()
            out.append({"role": "assistant", "content": final or "Task complete.", "reasoning_content": r})
            break
    if k == 0:
        raise Unconvertible("no_actions")
    return out, {"n_turns": k + 1, "n_tool_calls": k}


# ------------------------------------------------------------------ jupyter-agent -> opencode
_MAGIC = re.compile(r"^\s*[!%]", re.M)


def _printify(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        raise Unconvertible("cell_syntax_error")
    if not tree.body:
        return code
    last = tree.body[-1]
    if isinstance(last, ast.Expr):
        v = last.value
        if isinstance(v, ast.Call) and getattr(v.func, "id", "") in ("print", "display"):
            pass
        elif isinstance(v, ast.Call) and getattr(v.func, "attr", "") in ("show", "savefig", "plot", "hist",
                                                                           "bar", "scatter", "boxplot"):
            pass
        else:
            lines = code.split("\n")
            seg = "\n".join(lines[last.lineno - 1:last.end_lineno])
            head = "\n".join(lines[:last.lineno - 1])
            code = (head + "\n" if head else "") + f"print({seg.strip()})"
    return re.sub(r"(?<![\w.])display\(", "print(", code)


def jupyter_to_opencode(row: dict) -> tuple[list[dict], dict]:
    m = row["messages"]
    m = json.loads(m) if isinstance(m, str) else m
    files = row.get("files_used") or []
    names = [f.rsplit("/", 1)[-1] for f in files]
    q = (row.get("question") or "").strip()
    if not q or not names:
        raise Unconvertible("jupyter_no_question_or_files")
    user = (f"{q}\n\nThe input data files are in `data/`: " + ", ".join(f"`data/{n}`" for n in names) +
            ". Write your analysis code to `analysis.py` and report the final answer.")
    out = [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}]
    outputs, k, final, carry = [], 0, None, ""
    i = 1
    while i < len(m):
        a = m[i]
        if a["role"] != "assistant":
            i += 1
            continue
        r, _ = split_think(a.get("content") or "")
        tcs = a.get("tool_calls")
        if isinstance(tcs, str):
            tcs = ast.literal_eval(tcs) if tcs not in ("None", "") else None
        if not tcs:
            raise Unconvertible("jupyter_no_tool_call")
        f = tcs[0]["function"]
        args = f["arguments"] if isinstance(f["arguments"], dict) else json.loads(f["arguments"])
        if f["name"] == "final_answer":
            final = (args.get("answer") or "").strip()
            out.append({"role": "assistant", "content": final, "reasoning_content": r or ""})
            break
        code = args.get("code") or ""
        if _MAGIC.search(code):
            if all(_MAGIC.match(ln) or not ln.strip() or ln.strip().startswith("#") for ln in code.split("\n")):
                # pure `!pip install` / `%matplotlib` cell: no shell-script equivalent needed -> skip turn,
                # keep its reasoning for the next action
                carry = (carry + "\n\n" if carry else "") + (r or "")
                i += 2
                continue
            raise Unconvertible("jupyter_magic")
        if carry:
            r, carry = (carry + "\n\n" + (r or "")).strip(), ""
        code = code.replace("/home/user/input/", "data/").replace("/kaggle/input/", "data/")
        code = _printify(code)
        obs = (m[i + 1].get("content") or "") if i + 1 < len(m) else ""
        if "Traceback" in obs or "Error" in obs[:200]:
            raise Unconvertible("jupyter_cell_error")
        # E2B display truncation marker: keep the visible lines, drop the marker (documented approximation)
        obs = obs.replace("[Output is truncated as it is more than 10 lines]", "").rstrip()
        if "<Figure" in obs or "image/png" in obs:
            obs = re.sub(r"<Figure size [^>]*>", "", obs)
        outputs.append(obs.rstrip("\n"))
        verb = ">" if k == 0 else ">>"
        cmd = f"cat {verb} analysis.py <<'EOF'\n{code}\nEOF\npython3 analysis.py"
        cid = f"call_{k}"
        out.append({"role": "assistant", "content": "", "reasoning_content": r or "",
                    "tool_calls": [{"id": cid, "type": "function",
                                    "function": {"name": "bash", "arguments": {"command": cmd}}}]})
        out.append({"role": "tool", "tool_call_id": cid,
                    "content": oc_output("\n".join(x for x in outputs if x) + "\n", k)})
        k += 1
        i += 2
    if final is None or k == 0:
        raise Unconvertible("jupyter_no_final_answer")
    gold = (row.get("answer") or "").strip().lower()
    if gold and gold not in final.lower():
        raise Unconvertible("jupyter_answer_mismatch")
    if any(not x.get("reasoning_content") for x in out if x["role"] == "assistant"):
        raise Unconvertible("jupyter_missing_thinking")
    return out, {"n_turns": k + 1, "n_tool_calls": k}
