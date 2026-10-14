#!/usr/bin/env python3
"""Convert raw harness outputs into the normalized per-task JSONL used by summarize.py.

One JSON object per (task, attempt):
  run_id, bench, task_id, attempt,
  status   success | fail | timeout | env_error | pending
           precedence: env_error > success (score>0) > timeout > fail;
           pending = generated but not graded yet (e.g. HLE before judging)
  score    float in [0,1] (binary benchmarks: 0/1; ASI: final_score/max_score), null if pending
  tokens_in, tokens_out, tokens_cached, wall_s, error, extra{}

Primary metrics count env_error as score 0 (as Harbor/HLE official do); summarize.py also
reports the env-error rate and the score excluding env errors.

Subcommands:  harbor (tb2 | swepro), asi, hle   (stdlib only)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path

# Harbor exception types that are infrastructure failures, not model failures.
ENV_EXC = {
    "EnvironmentStartTimeoutError", "EnvironmentBuildTimeoutError", "AgentSetupTimeoutError",
    "RewardFileNotFoundError", "RewardFileEmptyError", "VerifierOutputParseError",
    "DockerError", "ConflictError", "APIConnectionError", "ServiceUnavailableError",
    "InternalServerError", "BadGatewayError", "Timeout", "ConnectError", "ConnectionError",
    "RuntimeError", "FileNotFoundError", "AgentAuthenticationError", "RemoteProtocolError",
}
TIMEOUT_EXC = {"AgentTimeoutError", "TimeoutError"}
ENV_HINTS = ("Failed to convert Docker image", "Singularity FastAPI server", "llm endpoint unavailable",
             "Connection refused", "Server process died", "Failed to start", "No space left")


def _dt(s):
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _secs(a, b):
    a, b = _dt(a), _dt(b)
    return round((b - a).total_seconds(), 1) if a and b else None


def _reward(tr: dict):
    vr = tr.get("verifier_result") or {}
    rw = vr.get("rewards") or {}
    if not rw:
        return None
    v = rw.get("reward", next(iter(rw.values())))
    try:
        return float(v)
    except Exception:
        return None


def _exc(tr: dict):
    ei = tr.get("exception_info") or {}
    return ei.get("exception_type"), (ei.get("exception_message") or "")[:400]


def classify(score, exc_type, exc_msg, timed_out=False):
    if exc_type and (exc_type in ENV_EXC or any(h in exc_msg for h in ENV_HINTS)) and not score:
        return "env_error"
    if score is not None and score > 0:
        return "success"
    if timed_out or exc_type in TIMEOUT_EXC:
        return "timeout"
    if score is None and exc_type:
        return "env_error"  # no verdict at all and an exception: infrastructure
    return "fail"


def load_trials(job_dir: Path) -> dict[str, dict]:
    out = {}
    for rj in sorted(glob.glob(str(job_dir / "*" / "result.json"))):
        tr = json.load(open(rj))
        if "task_name" not in tr:
            continue  # job-level result.json
        tid = tr["task_name"].split("/")[-1]
        tr["_trial_dir"] = str(Path(rj).parent)
        out.setdefault(tid, tr)  # one trial per task per job (k handled as separate jobs)
    return out


def harbor(a) -> list[dict]:
    trials = load_trials(Path(a.job_dir))
    agent_trials = load_trials(Path(a.agent_job_dir)) if a.agent_job_dir else {}
    rows = []
    for tid, tr in trials.items():
        score = _reward(tr)
        et, em = _exc(tr)
        src = agent_trials.get(tid, tr)  # SWE: tokens/time/agent status from the agent job
        ar = src.get("agent_result") or {}
        timed_out = False
        extra = {"trial_dir": tr["_trial_dir"]}
        if agent_trials:
            aet, aem = _exc(src)
            extra.update(agent_exception=aet, agent_trial_dir=src.get("_trial_dir"))
            try:
                r = json.load(open(Path(src["_trial_dir"]) / "agent" / "result.json"))
                extra.update(agent_exit_status=r.get("exit_status"), n_calls=r.get("n_calls"))
                timed_out = bool(r.get("budget_exhausted"))
                if r.get("exit_status") in ENV_EXC and not et:  # e.g. endpoint down mid-run
                    et, em = r["exit_status"], (r.get("error") or "")[:400]
            except Exception:
                pass
            try:
                extra["patch_bytes"] = os.path.getsize(Path(src["_trial_dir"]) / "agent" / "model.patch")
            except OSError:
                extra["patch_bytes"] = None
            if aet in TIMEOUT_EXC:
                timed_out = True
            if aet and (aet in ENV_EXC or any(h in aem for h in ENV_HINTS)) and not et:
                et, em = aet, aem  # agent-side infra failure (e.g. endpoint down)
        status = classify(score, et, em, timed_out)
        ae = src.get("agent_execution") or {}
        rows.append({
            "run_id": a.run_id, "bench": a.bench, "task_id": tid, "attempt": a.attempt,
            "status": status, "score": score if score is not None else (0.0 if status != "pending" else None),
            "tokens_in": ar.get("n_input_tokens"), "tokens_out": ar.get("n_output_tokens"),
            "tokens_cached": ar.get("n_cache_tokens"),
            "wall_s": _secs(ae.get("started_at"), ae.get("finished_at")),
            "trial_wall_s": _secs(src.get("started_at"), src.get("finished_at")),
            "error": f"{et}: {em}" if et else None, "extra": extra,
        })
    return rows


def asi(a) -> list[dict]:
    report = json.load(open(a.score))
    rows = []
    for r in report.get("results", []):
        src = {}
        try:
            src = json.load(open(r["source_result"]))
        except Exception:
            pass
        ao = src.get("agent_output") or {}
        fs, ms = r.get("final_score"), r.get("max_score") or 100.0
        score = (float(fs) / float(ms)) if fs is not None and ms else None
        err = (ao.get("error_message") or "")[:400] or None
        agent_status = (ao.get("status") or src.get("status") or "").lower()
        timed_out = "timeout" in agent_status or bool(ao.get("timeout_phase"))
        if r.get("evaluation_status") == "evaluation_invalid":
            status, score = "env_error", 0.0
            err = err or f"scorer: {r.get('failure_kind')}"
        elif err and any(h.lower() in err.lower() for h in ("connection refused", "econnrefused", "502",
                                                             "endpoint unavailable", "fetch failed")) and not score:
            status = "env_error"
        elif score and score > 0:
            status = "success"
        elif timed_out:
            status = "timeout"
        else:
            status = "fail"
        rows.append({
            "run_id": a.run_id, "bench": "asi", "task_id": f'{r["task_id"]}@{r["prompt_level"]}',
            "attempt": r.get("attempt", 1), "status": status, "score": score if score is not None else 0.0,
            "tokens_in": None, "tokens_out": None, "tokens_cached": None,
            "wall_s": src.get("execution_time_seconds"), "error": err,
            "extra": {"level": r["prompt_level"], "agent_status": agent_status,
                      "hard_gates_passed": r.get("hard_gates_passed"), "raw_final_score": fs,
                      "source_result": r.get("source_result")},
        })
    return rows


def hle(a) -> list[dict]:
    import pandas as pd  # available in envs/eval (via harbor deps) and envs/hle
    df = pd.read_parquet(glob.glob(os.path.join(a.dataset, "data", "test-*.parquet"))[0],
                         columns=["id", "answer_type", "category", "image"])
    preds = json.load(open(a.pred)) if os.path.exists(a.pred) else {}
    judged = json.load(open(a.judged)) if a.judged and os.path.exists(a.judged) else {}
    ids = None
    if a.ids:
        ids = set(a.ids.split(","))
    elif a.pred_only_ids:
        ids = set(preds)
    rows = []
    for rec in df.itertuples():
        if ids is not None and rec.id not in ids:
            continue
        p, j = preds.get(rec.id), judged.get(rec.id, {}).get("judge_response")
        usage = (p or {}).get("usage") or {}
        if p is None:
            status, score, err = "env_error", 0.0, "no prediction (API call failed in all passes)"
        elif j is None:
            status, score, err = "pending", None, None
        else:
            score = 1.0 if "yes" in j.get("correct", "") else 0.0
            status = "success" if score else ("timeout" if p.get("finish_reason") == "length" else "fail")
            err = None
        rows.append({
            "run_id": a.run_id, "bench": "hle", "task_id": rec.id, "attempt": 1, "status": status,
            "score": score, "tokens_in": usage.get("prompt_tokens"), "tokens_out": usage.get("completion_tokens"),
            "tokens_cached": None, "wall_s": None, "error": err,
            "extra": {"answer_type": rec.answer_type, "category": rec.category,
                      "has_image": bool(rec.image), "finish_reason": (p or {}).get("finish_reason"),
                      "confidence": (j or {}).get("confidence")},
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    h = sp.add_parser("harbor")
    h.add_argument("--bench", required=True, choices=["tb2", "swepro"])
    h.add_argument("--job-dir", required=True)
    h.add_argument("--agent-job-dir", default=None)
    h.add_argument("--attempt", type=int, default=1)
    s = sp.add_parser("asi")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--score", required=True)
    q = sp.add_parser("hle")
    q.add_argument("--pred", required=True)
    q.add_argument("--judged", default=None)
    q.add_argument("--dataset", required=True)
    q.add_argument("--ids", default=None)
    q.add_argument("--pred-only-ids", action="store_true")
    for p in (h, s, q):
        p.add_argument("--run-id", required=True)
        p.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = {"harbor": harbor, "asi": asi, "hle": hle}[a.cmd](a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    from collections import Counter
    print(f"{a.out}: {len(rows)} rows, status={dict(Counter(r['status'] for r in rows))}")


if __name__ == "__main__":
    main()
