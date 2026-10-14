"""E-SWE-v2 side experiment: nvidia/Open-SWE-Traces (native mini-swe-agent) -> E-SWE training format.

Why: E-SWE (OpenHands-converted, 100% Python, Qwen3-Coder-480B teacher) did not beat Base on HARD-51 and
picked up OpenHands habits (repro scripts and test edits left in the repo). This set keeps the E-SWE recipe
(same schema, same locked mini_toolcall.yaml rendering, ~15.05M train tokens, 64K max) but uses:
  * nvidia/Open-SWE-Traces @ REVISION, data/minisweagent/qwen38_27b/swe-rebench-v2 (teacher Qwen3.8-27B,
    thinking, Apache-2.0; dataset CC-BY-4.0), resolved == 1 only;
  * a language quota close to HARD-51 (Go 40% / TS+JS 30% / Python 30%), harder tasks preferred;
  * trajectories whose final repository state would submit test edits or scratch files under our protocol
    dropped (see `leftover_files`).

Conversion (exactly what the locked eval sees):
  system/user  re-rendered with mini_toolcall.yaml (convert_swe.system_and_instance), task = text inside
               <pr_description>, wrapped like a SWE-bench Pro V2 instruction (convert_swe.swe_pro_wrapper);
  paths        /testbed -> /app in commands, observations, content and reasoning;
  observations the teacher used the same observation_template (same JSON layout + tojson escaping), so the
               JSON is parsed, paths rewritten and re-serialised in that layout;
  patch.txt    the teacher's protocol writes `git diff -- <files> > patch.txt && cat patch.txt` and submits
               `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`. Ours captures
               `git add -A && git diff --cached` after a bare submit, so the first becomes `git diff -- <files>`
               (identical output) and the submit becomes the bare command. Any other use of patch.txt -> drop.
  parallel tool calls in one turn are kept (mini-swe-agent 2.4.6 executes them in order).

usage (cluster, CPU job: slurm/data_build_swe_v2.sbatch):
  python -m src.data.build_swe_v2 download
  python -m src.data.build_swe_v2 build --workers 15
  python -m src.data.verify --experts swe_v2 --recount 50
Outputs: $APEX/data/processed/swe_v2/{train,val}.jsonl, $APEX/data/manifests/swe_v2_{train,val}.csv,
         $APEX/data/manifests/swe_v2_stats.json, $APEX/data/interim/swe_v2/candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter

from .common import DropCounter, apex_path, get_tokenizer, log, raw_dir

REPO_ID = "nvidia/Open-SWE-Traces"
REVISION = "1db674173119d71df45c22007af99e214600711a"
SUBDIRS = ["data/minisweagent/qwen38_27b/swe-rebench-v2"]
RAW_KEY = "open_swe_traces"
EXPERT = "swe_v2"
MAX_TOKENS = 65536
MIN_TURNS, MAX_TURNS = 30, 150     # min on executed actions (tool calls), max on assistant turns
LANG_FRAC = {"go": 0.40, "tsjs": 0.30, "python": 0.30}
HARD_FRAC = 0.75                     # preferred share of hard tasks inside each language bucket
TARGET_COUNT = 1200                  # hard cap; the token budget normally stops selection first
MAX_REPO_FRACTION = 0.03
LANG_BUCKET = {"go": "go", "golang": "go", "typescript": "tsjs", "javascript": "tsjs", "ts": "tsjs",
               "js": "tsjs", "python": "python", "py": "python"}

SUBMIT_MARK = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
_TB = re.compile(r"(?<![\w/.])/testbed\b")
_PATCH_TXT_DIFF = re.compile(r"\s*>\s*patch\.txt\s*(?:&&|;)\s*cat\s+patch\.txt\b")
_GIT_HISTORY = re.compile(r"git\s+(?:log|rev-list)\b[^|;&]*--all|git\s+reflog\b|git\s+fetch\b|"
                          r"git\s+(?:show|checkout|diff|log|cherry-pick|reset)\s+[^|;&]*\b[0-9a-f]{7,40}\b|"
                          r"git\s+log\b[^|;&]*\s-p\b")
_TESTY = re.compile(r"(^|/)(tests?|testing|__tests__|spec|e2e)/|(^|/)test_[^/]*\.py$|_test\.(py|go)$|"
                    r"\.(test|spec)\.[cm]?[jt]sx?$|(^|/)conftest\.py$|_spec\.rb$|Test\.java$")
_SCRATCH = re.compile(r"(^|/)(reproduce|repro|debug|scratch|tmp_|temp_|check_|verify_|demo_|final_|"
                      r"comprehensive|simple_test|edge_case)[^/]*$")


class Unconvertible(Exception):
    pass


# ----------------------------------------------------------------------------- download
def download(a):
    from huggingface_hub import snapshot_download
    dest = raw_dir(RAW_KEY)
    t0 = time.time()
    allow = [f"{d}/*.parquet" for d in SUBDIRS] + ["README.md"]
    if a.max_shards:
        allow = [f"{d}/train-0000{i}-*.parquet" for d in SUBDIRS for i in range(a.max_shards)] + ["README.md"]
    log(f"[{EXPERT}] downloading {REPO_ID}@{REVISION[:10]} {allow}")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", revision=REVISION, allow_patterns=allow,
                      local_dir=str(dest), max_workers=8)
    files = sorted(p for p in dest.rglob("*.parquet"))
    rec = {"repo": REPO_ID, "revision": REVISION, "subdirs": SUBDIRS, "seconds": round(time.time() - t0, 1),
           "files": {str(p.relative_to(dest)): p.stat().st_size for p in files}}
    (dest / "_download_record.json").write_text(json.dumps(rec, indent=1))
    log(f"[{EXPERT}] {len(files)} parquet files, {sum(rec['files'].values()) / 1e9:.2f} GB")


# ----------------------------------------------------------------------------- conversion helpers
def _fix(s: str) -> str:
    return _TB.sub("/app", s or "")


def _loads(x):
    return json.loads(x) if isinstance(x, str) else x


def task_text(user: str) -> str:
    m = re.search(r"<pr_description>\s*(.*?)\s*</pr_description>", user or "", re.S)
    if not m:
        raise Unconvertible("no_pr_description")
    t = re.sub(r"^\s*Consider the following PR description:\s*\n", "", m.group(1)).strip()
    if len(t) < 20:
        raise Unconvertible("short_task")
    return _fix(t)


_TOJSON = None


def _tojson(s) -> str:
    """jinja2 `tojson` (htmlsafe escaping), exactly as the observation_template renders strings."""
    global _TOJSON
    if _TOJSON is None:
        import jinja2
        _TOJSON = jinja2.Template("{{ x | tojson }}")
    return _TOJSON.render(x=s)


def fix_observation(content: str) -> tuple[str, dict]:
    """Rewrite paths inside the teacher's rendered observation, keeping the observation_template layout."""
    from .convert_swe import render_observation
    try:
        o = json.loads(content)
    except Exception:
        raise Unconvertible("obs_not_json")
    rc = o.get("returncode")
    if not isinstance(rc, int):
        raise Unconvertible("obs_no_returncode")
    exc = _fix(o.get("exception_info") or "")
    # under our protocol patch.txt is never created: drop its porcelain `git status` line
    for key in ("output", "output_head", "output_tail"):
        if isinstance(o.get(key), str) and "?? patch.txt" in o[key]:
            o[key] = re.sub(r"(?m)^\?\? patch\.txt\n?", "", o[key])
    if "output" in o:
        out = _fix(o["output"])
        if len(out) >= 10000:       # cannot happen when only shortening paths; guard anyway
            raise Unconvertible("obs_len_changed_bucket")
        return render_observation(rc, out, exc), o
    if "output_head" in o and "output_tail" in o:
        txt = ("{\n" f'  "returncode": {rc},\n' f'  "output_head": {_tojson(_fix(o["output_head"]))},\n'
               f'  "output_tail": {_tojson(_fix(o["output_tail"]))},\n'
               f'  "elided_chars": {int(o.get("elided_chars", 0))},\n' '  "warning": "Output too long."')
        if exc:
            txt += f', "exception_info": {_tojson(exc)}'
        return txt + "\n}", o
    raise Unconvertible("obs_unknown_layout")


def _norm_path(p: str, cwd: str = "/app") -> str | None:
    """Repo-relative path if p lies inside /app, else None."""
    p = p.strip().strip("'\"")
    if not p or p.startswith(("$", "&", "-")) or p in ("/dev/null",):
        return None
    if not p.startswith("/"):
        p = cwd.rstrip("/") + "/" + p
    parts = []
    for seg in p.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    if not parts or parts[0] != "app":
        return None
    rel = "/".join(parts[1:])
    return rel or None


_P = r"([^\s;&|<>()'\"`]+)"
_WRITE_PATTERNS = [re.compile(r"(?<![<0-9&>])>>?\s*" + _P), re.compile(r"\btee\s+(?:-a\s+)?" + _P),
                   re.compile(r"\btouch\s+" + _P), re.compile(r"\b(?:cp|mv)\s+(?:-\w+\s+)*\S+\s+" + _P),
                   re.compile(r"open\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"][wa]"),
                   re.compile(r"Path\(\s*['\"]([^'\"]+)['\"]\s*\)\.write_(?:text|bytes)")]
_CD = re.compile(r"\bcd\s+" + _P)
_RM = re.compile(r"\brm\s+(?:-\w+\s+)*([^;&|<>()`]+)")
_STATUS_LINE = re.compile(r"^\s?([MADRCU?]{1,2})\s+(\S+)(?:\s+->\s+(\S+))?$")


def _cwd_at(cmd: str, pos: int) -> str:
    cwd = "/app"
    for m in _CD.finditer(cmd[:pos]):
        d = m.group(1)
        cwd = d if d.startswith("/") else cwd.rstrip("/") + "/" + d
    return cwd


_HEREDOC = re.compile(r"(<<-?\s*['\"]?(\w+)['\"]?)([^\n]*)\n(.*?)\n[ \t]*\2[ \t]*(?=\n|$)", re.S)
_QUOTED = re.compile(r"'[^'\n]*'|\"[^\"\n]*\"")


def _split_heredocs(cmd: str) -> tuple[str, list[str]]:
    """Shell text without heredoc bodies (the `<<EOF > path` line is kept) + the bodies."""
    bodies = []

    def rep(m):
        bodies.append(m.group(4))
        return m.group(1) + m.group(3)
    return _HEREDOC.sub(rep, cmd), bodies


def leftover_files(steps: list[tuple[str, str]], patch_files: set[str]) -> set[str]:
    """Approximate the extra files our `git add -A && git diff --cached` would capture: files written in
    /app by commands (redirections, tee, touch, cp/mv, python open/write_text in heredocs) or reported
    modified/untracked by the LAST `git status`, minus the teacher's own patch files, minus files removed or
    reverted afterwards (rm, git checkout/restore -- <path>, git clean -f)."""
    last: dict[str, int] = {}      # candidate -> step index it was (last) written / reported
    removed: dict[str, int] = {}
    for i, (cmd, obs) in enumerate(steps):
        shell, bodies = _split_heredocs(cmd)
        body = _QUOTED.sub("''", shell)
        for pat in _WRITE_PATTERNS[:4]:
            for m in pat.finditer(body):
                rel = _norm_path(m.group(1), _cwd_at(body, m.start()))
                if rel:
                    last[rel] = i
        for hd in bodies:
            for pat in _WRITE_PATTERNS[4:]:
                for m in pat.finditer(hd):
                    rel = _norm_path(m.group(1), _cwd_at(body, 0))
                    if rel:
                        last[rel] = i
        for m in _RM.finditer(body):
            for tok in m.group(1).split():
                rel = _norm_path(tok, _cwd_at(body, m.start()))
                if rel:
                    removed[rel] = i
        if re.search(r"\bgit\s+clean\s+-\w*f", body):
            for rel in list(last):
                removed[rel] = i
        if re.search(r"\bgit\s+(?:checkout|restore)\b", body):
            for rel in list(last):
                if rel in shell:
                    removed[rel] = i
        if re.search(r"\bgit\s+status\b", body):
            try:
                out = json.loads(obs).get("output") or ""
            except Exception:
                out = ""
            for ln in out.split("\n"):
                mm = _STATUS_LINE.match(ln)
                if mm:
                    last[(mm.group(3) or mm.group(2)).rstrip("/")] = i
                mm = re.match(r"^\s+(?:modified|new file|deleted):\s+(\S+)$", ln)
                if mm:
                    last[mm.group(1)] = i
            if "Untracked files:" in out:
                for ln in out.split("Untracked files:", 1)[1].split("\n")[1:]:
                    if ln.startswith("\t"):
                        last[ln.strip().rstrip("/")] = i
                    elif ln.strip() and not ln.lstrip().startswith("("):
                        break
    ok = set(patch_files) | {"patch.txt"}
    return {p for p, i in last.items() if removed.get(p, -1) < i and p not in ok}


def _patch_text(p) -> str:
    """metadata.{model,reference}_patch is {'patch', 'num_modified_files', 'num_modified_lines'}."""
    if isinstance(p, dict):
        return p.get("patch") or ""
    return p or ""


def _patch_paths(patch: str) -> list[str]:
    return re.findall(r"^diff --git a/(\S+) b/", patch or "", re.M)


def _patch_lines(patch: str) -> int:
    return sum(1 for ln in (patch or "").split("\n")
               if (ln.startswith("+") or ln.startswith("-")) and not ln.startswith(("+++", "---")))


def convert_mini(messages: list[dict]) -> tuple[list[dict], dict]:
    from .convert_swe import SUBMIT, swe_pro_wrapper, system_and_instance
    if len(messages) < 4 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
        raise Unconvertible("bad_prefix")
    pre, post = swe_pro_wrapper()
    sys_msg, inst = system_and_instance(pre + task_text(messages[1]["content"]) + post)
    out = [{"role": "system", "content": sys_msg}, {"role": "user", "content": inst}]
    steps: list[tuple[str, str]] = []
    k, i, finished, n_parallel, n_patch_rewrites = 0, 2, False, 0, 0
    while i < len(messages):
        m = messages[i]
        if m["role"] != "assistant":
            raise Unconvertible(f"unexpected_{m['role']}_turn")
        tcs = m.get("tool_calls") or []
        if not tcs:
            raise Unconvertible("assistant_without_tool_call")
        content, reasoning = _fix(m.get("content") or "").strip(), _fix(m.get("reasoning_content") or "").strip()
        cmds = []
        for tc in tcs:
            if tc["function"]["name"] != "bash":
                raise Unconvertible(f"tool_{tc['function']['name']}")
            try:
                args = _loads(tc["function"]["arguments"])
                cmd = _fix(args["command"]).strip()
            except Exception:
                raise Unconvertible("bad_tool_json")
            cmds.append(cmd)
        if any(SUBMIT_MARK in c for c in cmds):
            if len(cmds) != 1:
                raise Unconvertible("submit_with_other_calls")
            out.append({"role": "assistant", "content": content, "reasoning_content": reasoning,
                        "tool_calls": [{"id": f"call_{k}", "type": "function",
                                        "function": {"name": "bash", "arguments": {"command": SUBMIT}}}]})
            finished = True
            break
        new_cmds = []
        for c in cmds:
            c2 = _PATCH_TXT_DIFF.sub("", c)
            n_patch_rewrites += c2 != c
            if "patch.txt" in c2:
                raise Unconvertible("patch_txt_other_use")
            if _GIT_HISTORY.search(c2):
                raise Unconvertible("git_history")
            new_cmds.append(c2)
        obs_msgs = messages[i + 1:i + 1 + len(cmds)]
        if len(obs_msgs) != len(cmds) or any(o["role"] != "tool" for o in obs_msgs):
            raise Unconvertible("missing_observation")
        calls, tools = [], []
        for c, o in zip(new_cmds, obs_msgs):
            obs_txt, _ = fix_observation(o.get("content") or "")
            if "patch.txt" in obs_txt:
                raise Unconvertible("patch_txt_in_observation")
            cid = f"call_{k}"
            k += 1
            calls.append({"id": cid, "type": "function", "function": {"name": "bash", "arguments": {"command": c}}})
            tools.append({"role": "tool", "tool_call_id": cid, "content": obs_txt})
            steps.append((c, obs_txt))
        n_parallel += len(calls) > 1
        out.append({"role": "assistant", "content": content, "reasoning_content": reasoning, "tool_calls": calls})
        out.extend(tools)
        i += 1 + len(cmds)
    if not finished:
        raise Unconvertible("no_submit")
    blob = json.dumps(out)
    if "/testbed" in blob:
        raise Unconvertible("residual_testbed_path")
    return out, {"steps": steps, "n_parallel_turns": n_parallel, "n_patch_txt_rewrites": n_patch_rewrites,
                 "empty_reasoning_turns": sum(1 for x in out if x["role"] == "assistant" and not x["reasoning_content"])}


# ----------------------------------------------------------------------------- worker / build
def _worker(r: dict):
    from .build import _finish_sample
    from .convert_swe import bash_tool
    lang = LANG_BUCKET.get((r.get("language") or "").lower())
    md = _loads(r.get("metadata")) or {}
    try:
        msgs, st = convert_mini(_loads(r["messages"]))
    except Unconvertible as e:
        return None, f"convert:{e}"
    except Exception as e:  # noqa
        return None, f"convert_crash:{type(e).__name__}"
    if any("patch.txt" in (m.get("content") or "") + (m.get("reasoning_content") or "")
           for m in msgs if m["role"] == "assistant"):
        return None, "patch_txt_in_text"    # thought/action mismatch after the rewrite; could make the model
        #                                     create patch.txt, which `git add -A` would then submit
    turns = sum(1 for m in msgs if m["role"] == "assistant")
    actions = sum(len(m.get("tool_calls") or []) for m in msgs if m["role"] == "assistant")
    if actions < MIN_TURNS:          # the teacher often issues parallel calls: length = executed actions
        return None, "actions<30"
    if turns > MAX_TURNS:
        return None, "turns>150"
    mpatch = _patch_text(md.get("model_patch"))
    mfiles = _patch_paths(mpatch)
    core = [f for f in mfiles if not _TESTY.search(f) and not _SCRATCH.search(f)]
    if not core:
        return None, "patch_no_source_change"
    if len(core) != len(mfiles):
        return None, "patch_edits_tests_or_scratch"
    extra = leftover_files(st.pop("steps"), set(mfiles))
    if extra:
        return None, "leftover_repo_files"
    ref = _patch_text(md.get("reference_patch"))
    rfiles, rlines = len(_patch_paths(ref)), _patch_lines(ref)
    hard = rfiles >= 2 or rlines >= 30
    tid = r.get("trajectory_id") or r["instance_id"]
    s = {"id": f"swe_v2/{tid}", "source": "open_swe_traces", "instance_id": r["instance_id"], "messages": msgs,
         "tools": [bash_tool()],
         "meta": {"expert": "swe", "repo": (r.get("repo") or "").lower(), "language": lang,
                  "language_raw": r.get("language"), "files_changed": len(core), "turns": turns, "actions": actions,
                  "ref_files": rfiles, "ref_lines": rlines, "stratum": f"{lang}/{'hard' if hard else 'other'}",
                  "teacher": (md.get("teacher_model") or {}).get("name") if isinstance(md.get("teacher_model"), dict)
                  else md.get("teacher_model"), "category": md.get("category"),
                  "hf_dataset_name": r.get("hf_dataset_name"), **st}}
    s, why = _finish_sample(s, MAX_TOKENS)
    return s, why


def build(a):
    from .build import MANI, dump_candidates, iter_parquet, make_constraints, run_pool, select, write_outputs
    from .common import CFG, sample_sha
    from .convert_swe import mini_cfg, swe_pro_wrapper
    from .decontam import swe_pro_ngram_index, swe_repo_verdict
    from .dev_sets import dev_swe_repos
    drops = DropCounter()
    idx = swe_pro_ngram_index()
    dev_repos = dev_swe_repos()
    mini_cfg(); swe_pro_wrapper(); get_tokenizer()          # warm caches before forking workers
    files = sorted(raw_dir(RAW_KEY).rglob("*.parquet"))
    if not files:
        raise SystemExit("no parquet files: run `download` first")
    log(f"[{EXPERT}] {len(files)} parquet files; holding out {len(dev_repos)} dev repos")
    langs_seen = Counter()

    def prefilter(r):
        langs_seen[(r.get("language") or "?").lower()] += 1
        if int(r.get("resolved") or 0) != 1:
            drops.drop("not_resolved"); return False
        if LANG_BUCKET.get((r.get("language") or "").lower()) is None:
            drops.drop("language_not_in_quota"); return False
        repo = (r.get("repo") or "").lower()
        if repo in dev_repos:
            drops.drop("dev_repo_holdout"); return False
        md = _loads(r.get("metadata")) or {}
        v = swe_repo_verdict(repo, _patch_text(md.get("model_patch")))
        if v:
            drops.drop(f"decontam:{v}"); return False
        msgs = _loads(r["messages"])
        user = msgs[1]["content"] if len(msgs) > 1 else ""
        if idx.hits(user)[0] >= 1:
            drops.drop("decontam:swe_pro_13gram"); return False
        return True

    def rows():
        n = 0
        for p in files:
            for r in iter_parquet(p, batch_size=16):
                n += 1
                if a.max_rows and n > a.max_rows:
                    return
                if prefilter(r):
                    yield r

    cands = []
    for s, why in run_pool(_worker, rows(), a.workers):
        if s is None:
            drops.drop(why)
        else:
            cands.append(s)
    seen, uniq = set(), []
    for s in cands:
        h = sample_sha(s)
        if h in seen:
            drops.drop("dup_content"); continue
        seen.add(h); uniq.append(s)
    dump_candidates(EXPERT, uniq)
    # token budget: exactly E-SWE's train tokens (token-matched ablation), grossed up for the val split
    e_swe_tokens = json.loads((MANI / "swe_stats.json").read_text())["train_tokens"]
    budget = int(e_swe_tokens / (1 - CFG["val_fraction"]))
    frac = {}
    for lang, f in LANG_FRAC.items():
        frac[f"{lang}/hard"] = f * HARD_FRAC
        frac[f"{lang}/other"] = f * (1 - HARD_FRAC)
    max_repo = max(1, int(MAX_REPO_FRACTION * TARGET_COUNT))
    chosen = select(uniq, frac, lambda s: s["meta"]["stratum"], budget, TARGET_COUNT,
                    make_constraints(1, max_repo))
    st = write_outputs(EXPERT, chosen, dict(drops), {
        "source": {"repo": REPO_ID, "revision": REVISION, "subdirs": SUBDIRS, "license": "CC-BY-4.0",
                   "teacher": "Qwen3.8-27B (thinking, Apache-2.0)"},
        "filters": {"resolved": 1, "actions_min": MIN_TURNS, "assistant_turns_max": MAX_TURNS, "max_tokens": MAX_TOKENS,
                    "lang_frac": LANG_FRAC, "hard_frac": HARD_FRAC, "hard": "reference_patch >=2 files or >=30 lines",
                    "max_per_instance": 1, "max_per_repo": max_repo,
                    "dropped": "test/scratch files in model_patch, leftover repo files, git-history lookups, "
                               "other patch.txt uses (commands, observations, assistant text), SWE-Pro repos/forks, "
                               "dev repos, SWE-Pro 13-grams"},
        "token_budget": {"e_swe_train_tokens": e_swe_tokens, "budget_incl_val": budget},
        "n_candidates": len(uniq), "rows_seen_by_language": dict(langs_seen),
        "candidates_by_stratum": Counter(s["meta"]["stratum"] for s in uniq),
        "train_by_language_tokens": dict(Counter({l: sum(c["n_tokens"] for c in chosen if c["meta"]["language"] == l)
                                                  for l in LANG_FRAC})),
        "train_by_language_count": dict(Counter(c["meta"]["language"] for c in chosen)),
        "parallel_turn_samples": sum(1 for c in chosen if c["meta"].get("n_parallel_turns")),
    })
    log(json.dumps({k: st[k] for k in ("n_train", "n_val", "train_tokens", "n_candidates")}))
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "build"])
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--max-shards", type=int, default=0)
    a = ap.parse_args()
    {"download": download, "build": build}[a.cmd](a)


if __name__ == "__main__":
    main()
