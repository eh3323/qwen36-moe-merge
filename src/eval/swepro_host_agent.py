"""SWE-bench Pro V2 host-driven mini-swe-agent with the locked 50-minute budget.

Thin subclass of the locked tooling's ``host_mini_swe:HostMiniSweAgent``
(swebench_pro_repo @66f9276, v2/tooling/host_mini_swe.py). Unchanged: config
(mini_toolcall.yaml), prompt/tool format, LLM loop on the host, bash exec'd into
the sandbox, model.patch capture via ``git add -A && git diff --cached``.

Added (protocol rule 3, README: "50-minute budget per task (in-sandbox timeout,
partial patch still captured/graded)"): the upstream host agent relies on
Harbor's agent timeout, which cancels ``run()`` *before* the patch is captured,
so a timed-out task would be graded on an empty patch. Here the agent loop gets
a soft deadline ``budget_sec - 60`` (same margin as locked_mini_swe's
``timeout -k 30 (3000-60)``): at the deadline every further model query /
command raises, the loop exits, and the partial diff is captured as usual.

Usage: PYTHONPATH=$REPO/src/eval:$SWE_REPO/v2/tooling harbor run ... \
         --agent-import-path swepro_host_agent:HostMiniSweAgentBudget  (or job config import_path)
"""
from __future__ import annotations

import asyncio
import json
import time

import yaml
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from host_mini_swe import HarborSandboxEnvironment, HostMiniSweAgent


class HostMiniSweAgentBudget(HostMiniSweAgent):
    def __init__(self, *args, budget_sec: int | str = 3000, margin_sec: int | str = 60, **kwargs):
        super().__init__(*args, **kwargs)
        self._budget = int(budget_sec)
        self._margin = int(margin_sec)

    @staticmethod
    def name() -> str:
        return "host-mini-swe-agent-budget"

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.config import get_config_path
        from minisweagent.exceptions import TimeExceeded

        t_end = time.monotonic() + self._budget - self._margin
        config = yaml.safe_load(get_config_path(self._config_file).read_text())
        agent_cfg: dict = dict(config.get("agent", {}))
        if self._step_limit is not None:
            agent_cfg["step_limit"] = int(self._step_limit)
        if self._cost_limit is not None:
            agent_cfg["cost_limit"] = float(self._cost_limit)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        agent_cfg["output_path"] = self.logs_dir / "mini-swe-agent.trajectory.json"

        cwd = self._cwd or await self._detect_cwd(environment)
        env_cfg = config.get("environment", {})
        sandbox = HarborSandboxEnvironment(
            environment, asyncio.get_running_loop(), cwd=cwd, env=env_cfg.get("env", {}),
            timeout=int(self._command_timeout or env_cfg.get("timeout", 60)),
        )
        model = self._build_model(config.get("model", {}))

        # deadline: native mini-swe wall clock check before each query, plus a guard on
        # commands; an in-flight model call at the deadline is abandoned (thread left to
        # finish on its own) so the capture below always runs before Harbor's timeout.
        agent_cfg["wall_time_limit_seconds"] = max(self._budget - self._margin, 1)
        orig_exec = sandbox.execute

        def execute(action, cwd="", *, timeout=None):
            remaining = int(t_end - time.monotonic())
            if remaining <= 0:
                raise TimeExceeded({"role": "exit", "content": "TimeExceeded",
                                    "extra": {"exit_status": "TimeExceeded", "submission": ""}})
            return orig_exec(action, cwd, timeout=min(timeout or sandbox.timeout, max(remaining, 1)))

        sandbox.execute = execute
        agent = DefaultAgent(model, sandbox, **agent_cfg)
        self.logger.info(f"host-mini-swe-agent-budget: cwd={cwd} model={self.model_name} "
                         f"config={self._config_file} budget={self._budget}s")
        t0 = time.monotonic()
        exit_status, err = None, None
        fut = asyncio.ensure_future(asyncio.to_thread(agent.run, instruction))
        try:
            result = await asyncio.wait_for(asyncio.shield(fut), timeout=max(t_end - time.monotonic(), 1))
            exit_status = result.get("exit_status")
        except asyncio.TimeoutError:
            exit_status = "BudgetExhausted(in-flight call abandoned)"
        except Exception as e:
            exit_status, err = type(e).__name__, str(e)[:2000]
        budget_hit = time.monotonic() > t_end
        (self.logs_dir / "result.json").write_text(json.dumps(
            {"exit_status": exit_status, "n_calls": agent.n_calls, "cost": agent.cost,
             "budget_exhausted": budget_hit, "agent_wall_sec": round(time.monotonic() - t0, 1),
             "error": err}, indent=1))
        r = await environment.exec(command=f"cd {cwd} && git add -A && git diff --cached", timeout_sec=120)
        # the singularity exec server strips stdout; `git diff` output always ends with a newline, and without it
        # `git apply` rejects the patch ("corrupt patch") whenever its last hunk ends in added lines (2026-10-15)
        patch = r.stdout or ""
        if patch and not patch.endswith("\n"):
            patch += "\n"
        (self.logs_dir / "model.patch").write_text(patch)
        await environment.exec(command=f"cd {cwd} && git reset -q", timeout_sec=60)
        try:
            context.n_input_tokens = getattr(model, "n_input_tokens", None) or context.n_input_tokens
        except Exception:
            pass
