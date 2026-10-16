#!/usr/bin/env python3
"""Build (and validate against Harbor's own JobConfig) the locked Harbor job configs.

Kinds:
  tb2         Terminus-2 on TB2 @2fd12b88 (model card setup: 3 h, 32 CPU/48 GB, temp 1.0,
              top_p 0.95, top_k 20, max_tokens 81920, 256K ctx; raw trajectories)
  swe         SWE-Pro HARD-51, host-driven mini-swe-agent (mini_toolcall.yaml, step_limit 250,
              50-min budget), sandbox fully offline (singularity_network=none)
  swe_regrade patch_replay of a finished `swe` job on pristine sandboxes (verifier keeps
              network as in the official protocol -> singularity_network=pasta)
  oracle      `-a oracle` sanity run (tb2 or swe tasks), no LLM

All values come from configs/eval/harness.env via environment variables (see the sbatch files).
Writes JSON (valid YAML) for `harbor run -c <file>`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

E = os.environ


def env_kwargs(net: str, ignore_policy: bool = False) -> dict:
    kw = {
        "singularity_image_cache_dir": E["SIF_CACHE"],
        "singularity_network": net,
        "singularity_server_python": E["HOSTPY"],
        "singularity_staging_root": E["STAGING_ROOT"],
        "singularity_server_start_timeout_sec": 300,
    }
    if E.get("SANDBOX_ROOT"):  # per-trial writable sandbox (no subuid range for --fakeroot)
        kw["singularity_sandbox_root"] = E["SANDBOX_ROOT"]
    if ignore_policy:
        kw["singularity_ignore_network_policy"] = True
    return kw


def build(a: argparse.Namespace) -> dict:
    tasks = a.tasks.split(",") if a.tasks else None
    cfg: dict = {
        "job_name": a.job_name,
        "jobs_dir": a.jobs_dir,
        "n_attempts": a.k,
        "n_concurrent_trials": a.n_concurrent,
        "quiet": True,
        "retry": {"max_retries": 0},
    }
    if a.kind == "tb2" or (a.kind == "oracle" and a.dataset == "tb2"):
        ds = {"path": E["TB2_DIR"]}
        if E.get("HARBOR_ENV") == "docker":  # rented node with Docker (scripts/tb2_cloud.sh): Harbor's native backend
            env = {"type": "docker", "delete": True,
                   "override_cpus": int(E["TB2_CPUS"]), "override_memory_mb": int(E["TB2_MEMORY_MB"])}
        else:
            env = {"type": "singularity", "kwargs": env_kwargs("pasta"), "delete": True,
                   "override_cpus": int(E["TB2_CPUS"]), "override_memory_mb": int(E["TB2_MEMORY_MB"])}
    else:
        ds = {"path": E["SWE_TASKS"]}
        ids = Path(E["SWE_IDS"]).read_text().split()
        ds["task_names"] = ids
        if a.kind == "swe":
            env = {"type": "singularity", "kwargs": env_kwargs("none"), "delete": True}
            # in-sandbox verifier would run offline (the sandbox never has network) and its
            # score is not reported: the authoritative score is the patch_replay re-grade.
            cfg["verifier"] = {"disable": True}
        else:  # swe_regrade / oracle on swe: deterministic agent; verifier needs network (protocol)
            env = {"type": "singularity", "kwargs": env_kwargs("pasta", ignore_policy=True),
                   "delete": True}
    if tasks:
        ds["task_names"] = tasks
    cfg["datasets"] = [ds]
    cfg["environment"] = env

    api = E.get("PROXY_BASE", "")
    if a.kind == "tb2":
        cfg["agents"] = [{
            "name": E["TB2_AGENT"],
            "model_name": f"hosted_vllm/{E['SERVED_MODEL_NAME']}",
            "override_timeout_sec": float(E["TB2_TIMEOUT_SEC"]),
            "kwargs": {
                "api_base": api,
                "model_info": {"max_input_tokens": int(E["TB2_MAX_INPUT_TOKENS"]),
                               "max_output_tokens": int(E["MAX_TOKENS"]),
                               "input_cost_per_token": 0.0, "output_cost_per_token": 0.0},
                "temperature": float(E["TEMPERATURE"]),
                "llm_kwargs": {"top_p": float(E["TOP_P"]), "max_tokens": int(E["MAX_TOKENS"])},
                "llm_call_kwargs": {"extra_body": {"top_k": int(E["TOP_K"])}},
                "trajectory_config": {"raw_content": True},
                "store_all_messages": True,
            },
        }]
    elif a.kind == "swe":
        cfg["agents"] = [{
            "import_path": "swepro_host_agent:HostMiniSweAgentBudget",
            "model_name": f"openai/{E['SERVED_MODEL_NAME']}",
            "kwargs": {"config_file": E["SWE_CONFIG"], "step_limit": int(E["SWE_STEP_LIMIT"]),
                       "budget_sec": int(E["SWE_BUDGET_SEC"]), "margin_sec": 60},
        }]
    elif a.kind == "swe_regrade":
        cfg["agents"] = [{"import_path": "patch_replay:PatchReplayAgent", "model_name": "replay",
                          "kwargs": {"source_job": a.source_job}}]
    else:
        cfg["agents"] = [{"name": "oracle"}]
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["tb2", "swe", "swe_regrade", "oracle"])
    ap.add_argument("--dataset", choices=["tb2", "swe"], default="tb2", help="for oracle")
    ap.add_argument("--job-name", required=True)
    ap.add_argument("--jobs-dir", required=True)
    ap.add_argument("-k", type=int, default=1)
    ap.add_argument("-n", "--n-concurrent", type=int, default=4)
    ap.add_argument("--tasks", default=None, help="comma list of task names (smoke)")
    ap.add_argument("--source-job", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = build(a)
    try:  # validate with the installed (patched) harbor
        from harbor.models.job.config import JobConfig
        JobConfig.model_validate(cfg)
    except ImportError:
        print("warning: harbor not importable, config not validated", file=sys.stderr)
    Path(a.out).write_text(json.dumps(cfg, indent=1))
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
