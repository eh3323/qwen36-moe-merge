"""OpenHands (CodeAct) trajectories -> mini-swe-agent `mini_toolcall.yaml` format.

Target (SWE-bench Pro locked protocol, mini_toolcall.yaml copied verbatim at build time):
  system  = agent.system_template
  user    = agent.instance_template rendered with task=<SWE-Pro-style instruction>, system info
  assistant turns: reasoning_content + exactly one `bash` tool call {"command": ...}
  tool turns: model.observation_template rendered with jinja2 (same engine/filters as
              mini-swe-agent -> identical JSON escaping and the 5000 head + 5000 tail truncation)
  last turn: bash `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` (no observation; agent exits)

Conversion of OpenHands actions:
  execute_bash(command)          -> same command; prefixed `cd <cwd> && ` when OpenHands' persistent
                                    cwd differs from /app (mini-swe runs every command in a fresh
                                    subshell); OH status trailer lines stripped, exit code -> returncode
  str_replace_editor view file   -> nl -ba <path> [| sed -n 'a,bp']   (identical `cat -n` format)
  str_replace_editor view dir    -> find <path> -maxdepth 2 -not -path '*/.*' | sort
  str_replace_editor create      -> cat <<'EOF' > <path> ... EOF
  str_replace_editor str_replace -> python3 heredoc doing an exact unique replacement and printing the
                                    same numbered +-4-line snippet OpenHands showed
  str_replace_editor insert      -> python3 heredoc inserting after line N (+ snippet)
  think(thought)                 -> folded into reasoning_content of the next action turn
  finish(message)                -> echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT (message kept as content)
  task_tracker(task_list)        -> plan rendered as '- [status] title' lines, folded into reasoning
  failed editor calls without a faithful shell equivalent (view_range past EOF, create on existing file,
  ...) -> the action/observation pair is removed and its reasoning folded forward (<=3 per trajectory).
Anything else (undo_edit, interactive is_input / soft-timeout, stateful `export`/`source`)
=> trajectory dropped (reason counted).

Thinking channel: the teacher (Qwen3-Coder-480B) is a non-thinking model whose visible text before a
tool call is its reasoning. We move that text (plus any `think` tool thoughts) into
`reasoning_content` and leave `content` empty, so every training turn has a non-empty <think> block
(Qwen3.6 is evaluated with thinking on). mini-swe-agent accepts tool-call-only responses.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import jinja2
import yaml

from .common import CFG, apex_path, bench, guard_path

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
APP = "/app"


class Unconvertible(Exception):
    pass


class SkipTurn(Exception):
    pass


_WS = re.compile(r"(?<![\w/.])/workspace\b")


@lru_cache(None)
def mini_cfg() -> dict:
    return yaml.safe_load(guard_path(apex_path("mini_toolcall_yaml")).read_text())


@lru_cache(None)
def bash_tool() -> dict:
    # mini-swe-agent 2.4.6 models/utils/actions_toolcall.py BASH_TOOL (verbatim)
    return {"type": "function", "function": {
        "name": "bash", "description": "Execute a bash command",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The bash command to execute"}},
            "required": ["command"]}}}


@lru_cache(None)
def swe_pro_wrapper() -> tuple[str, str]:
    """Prefix/suffix around the problem text in SWE-bench Pro V2 instruction.md (= {{task}})."""
    root = bench("swebench_pro_repo", "v2", "tasks")
    d = sorted(root.iterdir())[0]
    txt = (d / "instruction.md").read_text()
    pre, rest = txt.split("<pr_description>", 1)
    _, post = rest.split("</pr_description>", 1)
    return pre + "<pr_description>\n", "\n</pr_description>" + post


def _jinja(template: str, **kw) -> str:
    return jinja2.Template(template, undefined=jinja2.StrictUndefined).render(**kw)


def render_observation(returncode: int, output: str, exception_info: str = "") -> str:
    return _jinja(mini_cfg()["model"]["observation_template"],
                  output={"returncode": returncode, "output": output, "exception_info": exception_info})


def system_and_instance(task_text: str) -> tuple[str, str]:
    c = mini_cfg()["agent"]
    parts = CFG["experts"]["swe"]["system_info"].split(" ")
    system, release, machine = parts[0], parts[1], parts[-1]
    version = " ".join(parts[2:-1])
    sys_msg = _jinja(c["system_template"])
    inst = _jinja(c["instance_template"], task=task_text, system=system, release=release,
                  version=version, machine=machine)
    return sys_msg, inst


# ----------------------------------------------------------------------------- helpers
_TRAILER = re.compile(r"\n?\[(?:The command completed with exit code (-?\d+)\.(?: CTRL\+C was sent\.)?|"
                      r"Current working directory: ([^\]\n]*)|Python interpreter: [^\]\n]*|"
                      r"Command finished with exit code (-?\d+))\]")
_STATEFUL = re.compile(r"^\s*(export\s|source\s|\.\s|conda\s+activate|alias\s|pushd\s|popd\b|set\s+-|"
                       r"ulimit\s|unset\s|deactivate\b)")


def _parse_bash_obs(obs: str) -> tuple[str, int, str | None]:
    if "no new output after" in obs or "[Your command" in obs or "timed out" in obs.split("\n")[-1]:
        raise Unconvertible("bash_soft_timeout")
    rc, cwd = None, None
    for m in _TRAILER.finditer(obs):
        if m.group(1) is not None:
            rc = int(m.group(1))
        if m.group(3) is not None:
            rc = int(m.group(3))
        if m.group(2) is not None:
            cwd = m.group(2).strip()
    if rc is None:
        raise Unconvertible("bash_no_exit_code")
    if rc < 0:
        raise Unconvertible("bash_negative_exit")
    out = _TRAILER.sub("", obs)
    if out and not out.endswith("\n"):
        out += "\n"
    return out, rc, cwd


def _strip_echo(out: str, cmd: str) -> str:
    """OpenHands' tmux shell sometimes echoes the command before its output; mini-swe does not."""
    cl = [ln.rstrip() for ln in cmd.split("\n") if ln.strip()]
    ol = out.split("\n")
    k = 0
    for c in cl:
        while k < len(ol) and not ol[k].strip():
            k += 1
        if k < len(ol) and ol[k].rstrip() == c:
            k += 1
        else:
            return out
    rest = "\n".join(ol[k:]).lstrip("\n")
    return rest


def _heredoc_delim(text: str, base: str) -> str:
    d, k = base, 0
    lines = set(text.split("\n"))
    while d in lines:
        k += 1
        d = f"{base}{k}"
    return d


def _pystr(s: str) -> str:
    if "'''" not in s and not s.endswith("\\") and "\r" not in s:
        return "r'''" + s + "'''"
    return repr(s)


def _snippet_from_obs(obs: str) -> str:
    """Numbered snippet OpenHands printed after an edit (between header and 'Review the changes')."""
    body = obs.split(":\n", 1)[1] if ":\n" in obs else ""
    body = body.split("\nReview the changes and make sure", 1)[0]
    return body + "\n" if body else ""


_EDIT_PRINT = """lines = s2.split('\\n')
a = max(0, line0 - 4); b = min(len(lines), line0 + n_new + 5)
print('\\n'.join(f'{i + 1:6}\\t{lines[i]}' for i in range(a, b)))"""


def _str_replace_cmd(path: str, old: str, new: str) -> str:
    code = (f"from pathlib import Path\np = Path('{path}')\ns = p.read_text()\nold = {_pystr(old)}\n"
            f"new = {_pystr(new)}\nif old == new:\n    raise SystemExit('Error: old and new strings are identical')\n"
            f"n = s.count(old)\nif n != 1:\n    raise SystemExit(f'Error: expected exactly one occurrence of the old string in {{p}}, found {{n}}')\n"
            f"line0 = s[:s.index(old)].count('\\n')\nn_new = new.count('\\n')\ns2 = s.replace(old, new, 1)\n"
            f"p.write_text(s2)\n{_EDIT_PRINT}\n")
    d = _heredoc_delim(code, "PYEOF")
    return f"python3 - <<'{d}'\n{code}{d}"


def _insert_cmd(path: str, line: int, new: str) -> str:
    code = (f"from pathlib import Path\np = Path('{path}')\nlines = p.read_text().split('\\n')\n"
            f"new = {_pystr(new)}\nline0 = {int(line)}\nn_new = new.count('\\n')\n"
            f"s2 = '\\n'.join(lines[:line0] + new.split('\\n') + lines[line0:])\np.write_text(s2)\n{_EDIT_PRINT}\n")
    d = _heredoc_delim(code, "PYEOF")
    return f"python3 - <<'{d}'\n{code}{d}"


def _sq(path: str) -> str:
    return path if re.fullmatch(r"[\w./@+-]+", path) else "'" + path.replace("'", "'\\''") + "'"


# ----------------------------------------------------------------------------- core
def _editor_action(a: dict, obs: str, repo_root: str) -> tuple[str, str, int]:
    cmd = a.get("command")
    path = (a.get("path") or "").replace(repo_root, APP)
    if path == "/workspace":
        path = APP
    is_err = obs.startswith("ERROR:")
    if cmd == "view":
        if "files and directories up to 2 levels deep" in obs:
            if is_err:
                raise SkipTurn()
            lines = [ln.rstrip("/") for ln in obs.split("\n")[1:]
                     if ln.startswith(APP) and "hidden files/directories" not in ln]
            return (f"find {_sq(path)} -maxdepth 2 -not -path '*/.*' | sort",
                    "\n".join(sorted(set(lines))) + "\n", 0)
        if is_err:
            if "does not exist" in obs:
                return f"nl -ba {_sq(path)}", f"nl: {path}: No such file or directory\n", 1
            raise SkipTurn()
        vr = a.get("view_range")
        c = f"nl -ba {_sq(path)}"
        if vr:
            lo, hi = int(vr[0]), int(vr[1])
            c += f" | sed -n '{lo},{'$' if hi == -1 else hi}p'"
        body = obs.split("\n", 1)[1] if "\n" in obs else ""
        body = re.sub(r"<response clipped><NOTE>.*?</NOTE>\s*$", "", body, flags=re.S)
        if body and not body.endswith("\n"):
            body += "\n"
        return c, body, 0
    if cmd == "create":
        text = a.get("file_text")
        if text is None:
            raise Unconvertible("editor_create_no_text")
        d = _heredoc_delim(text, "EOF")
        c = f"cat <<'{d}' > {_sq(path)}\n{text}\n{d}"
        if is_err:
            if "No such file or directory" in obs:
                return c, f"bash: line 1: {path}: No such file or directory\n", 1
            raise SkipTurn()   # e.g. 'already exists' (cat would overwrite)
        return c, "", 0
    if cmd == "str_replace":
        old, new = a.get("old_str"), a.get("new_str") or ""
        if old is None:
            raise Unconvertible("editor_str_replace_no_old")
        c = _str_replace_cmd(path, old, new)
        if is_err:
            if "must be different" in obs:
                return c, "Error: old and new strings are identical\n", 1
            if "did not appear verbatim" in obs:
                return c, f"Error: expected exactly one occurrence of the old string in {path}, found 0\n", 1
            m = re.search(r"Multiple occurrences of old_str .*? in lines \[([\d, ]+)\]", obs, re.S)
            if m:
                k = len(m.group(1).split(","))
                return c, f"Error: expected exactly one occurrence of the old string in {path}, found {k}\n", 1
            raise SkipTurn()
        return c, _snippet_from_obs(obs), 0
    if cmd == "insert":
        if is_err or a.get("insert_line") is None:
            raise SkipTurn()
        return _insert_cmd(path, int(a["insert_line"]), a.get("new_str") or ""), _snippet_from_obs(obs), 0
    raise Unconvertible(f"editor_{cmd}")


def _args(tc) -> dict:
    a = tc["function"]["arguments"]
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            raise Unconvertible("bad_tool_json")
    return a


def convert_openhands(traj: list[dict]) -> tuple[list[dict], dict]:
    """Returns (messages, stats). Raises Unconvertible."""
    if not traj or traj[0]["role"] != "system" or traj[1]["role"] != "user":
        raise Unconvertible("bad_prefix")
    user = traj[1]["content"]
    m = re.search(r"<uploaded_files>\s*(/workspace/[^\s<]+)\s*</uploaded_files>", user)
    m2 = re.search(r"<issue_description>\n?(.*?)\n?</issue_description>", user, re.S)
    if not m or not m2:
        raise Unconvertible("no_issue_or_repo")
    repo_root = m.group(1).rstrip("/")
    issue = m2.group(1).strip()
    pre, post = swe_pro_wrapper()
    sys_msg, inst = system_and_instance(pre + issue + post)
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": inst}]

    def fix(s: str) -> str:
        return _WS.sub(APP, (s or "").replace(repo_root, APP))

    cwd = APP
    pending_reason: list[str] = []
    n_actions = 0
    skipped = 0
    finished = False
    i = 2
    while i < len(traj):
        mm = traj[i]
        if mm["role"] != "assistant":
            raise Unconvertible("unexpected_role_order")
        tcs = mm.get("tool_calls")
        tcs = tcs if isinstance(tcs, list) else []
        text = fix(mm.get("content") or "").strip()
        if len(tcs) != 1:
            raise Unconvertible("not_one_tool_call")
        tc = tcs[0]
        name = tc["function"]["name"]
        a = _args(tc)
        obs_msg = traj[i + 1] if i + 1 < len(traj) else None
        obs = (obs_msg or {}).get("content") or ""
        if obs_msg is not None and obs_msg["role"] != "tool":
            raise Unconvertible("missing_observation")
        if name == "think":
            pending_reason += [t for t in (text, fix(a.get("thought", "")).strip()) if t]
            i += 2
            continue
        if name == "task_tracker":
            items = a.get("task_list") or []
            plan = "\n".join(f"- [{t.get('status', 'todo')}] {t.get('title', '')}" for t in items
                             if isinstance(t, dict))
            pending_reason += [t for t in (text, fix(plan)) if t]
            i += 2
            continue
        reasoning = "\n\n".join(pending_reason + ([text] if text else []))
        pending_reason = []
        call_id = f"call_{n_actions}"
        if name == "finish":
            msgs.append({"role": "assistant", "content": fix(a.get("message", "")).strip(),
                         "reasoning_content": reasoning,
                         "tool_calls": [{"id": call_id, "type": "function",
                                         "function": {"name": "bash", "arguments": {"command": SUBMIT}}}]})
            finished = True
            break
        if name == "execute_bash":
            if str(a.get("is_input", "false")).lower() == "true":
                raise Unconvertible("bash_is_input")
            cmd = fix(a.get("command", "")).strip()
            if not cmd:
                raise Unconvertible("bash_empty")
            if _STATEFUL.match(cmd):
                raise Unconvertible("bash_stateful")
            out, rc, new_cwd = _parse_bash_obs(fix(obs))
            out = _strip_echo(out, cmd)
            if cwd != APP and not cmd.startswith("cd /"):
                cmd = f"cd {_sq(cwd)} && {cmd}"
            if new_cwd:
                cwd = new_cwd
        elif name == "str_replace_editor":
            try:
                fa = {k: (fix(v) if isinstance(v, str) else v) for k, v in a.items()}
                cmd, out, rc = _editor_action(fa, fix(obs), APP)
            except SkipTurn:
                # failed editor call with no faithful shell equivalent (e.g. view_range past EOF, create on
                # an existing file): drop the action + its error observation, keep the reasoning.
                skipped += 1
                if skipped > 3:
                    raise Unconvertible("too_many_skipped_editor_errors")
                if reasoning:
                    pending_reason.append(reasoning)
                i += 2
                continue
        else:
            raise Unconvertible(f"tool_{name}")
        if not reasoning:
            reasoning = ""
        msgs.append({"role": "assistant", "content": "", "reasoning_content": reasoning,
                     "tool_calls": [{"id": call_id, "type": "function",
                                     "function": {"name": "bash", "arguments": {"command": cmd}}}]})
        msgs.append({"role": "tool", "tool_call_id": call_id, "content": render_observation(rc, out)})
        n_actions += 1
        i += 2
    if not finished:
        raise Unconvertible("no_finish")
    if _WS.search(json.dumps(msgs)):
        raise Unconvertible("residual_workspace_path")
    empty_reason = sum(1 for x in msgs if x["role"] == "assistant" and not x["reasoning_content"])
    return msgs, {"n_actions": n_actions + 1, "empty_reasoning_turns": empty_reason, "skipped_error_turns": skipped}


# ----------------------------------------------------------------------------- patch features
_TESTY = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.(py|go)$|\.(test|spec)\.[jt]sx?$")
_SCRATCH = re.compile(r"(^|/)(reproduce|repro|debug|check|verify|final_|comprehensive|test_fix|simple_test|"
                      r"edge_case|demo|test_edge)[^/]*\.py$")


def patch_files(patch: str) -> tuple[int, int]:
    """(#non-test, non-scratch source files changed, #all files)."""
    files = re.findall(r"^diff --git a/(\S+) b/", patch or "", re.M)
    core = [f for f in files if not _TESTY.search(f) and not _SCRATCH.search(f)]
    return len(core), len(files)
