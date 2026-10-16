#!/bin/bash
# Smoke test for the musl (Alpine) sandbox fix in patches/harbor_singularity_network_none.patch.
# Runs Harbor's real SingularityEnvironment (start/exec/stop, same kwargs as src/eval/harbor_job.py) on
#   - one musl task image   (default vuls-e1fab805, one of the 10 HARD-51 tasks that never started),
#     once with singularity_network=none  (agent phase) and once with pasta (patch_replay re-grade),
#   - one glibc control     (default ansible-c1f2df47, smallest HARD-51 image; must behave as before),
# and checks: start-up flags, uid 0, writable /app, git + toolchain, no egress in "none" mode
# (egress in pasta mode), /scratch invisible, server deps not leaked into the image, sandbox removed.
#
# Run on a compute node inside an existing allocation (needs /state/partition1 or /tmp space, ~5 GB):
#   srun --jobid=<job> --overlap --ntasks=1 bash /scratch/eh3323/apex/repo/scripts/test_musl_sandbox.sh
# It does not modify the installed harbor: by default the REPO patch is applied to a pristine copy of
# the singularity backend (harbor @4778ceba) in a node-local overlay (HARBOR_PKG=installed tests the
# installed package instead). If $APEX/envs/harbor-hostpy-pylib does not exist yet (scripts/eval_setup.sh
# pylib), a temporary copy is built in the work dir (needs PyPI egress from the node).
# Everything is written under one node-local work dir, removed at exit (KEEP=1 keeps it).
# Overrides: MUSL_TASK, GLIBC_TASK (instance id prefixes), SKIP_PASTA=1, CASE_TIMEOUT (s, default 540).
# E.g. MUSL_TASK=instance_gravitational__teleport-2b15263e: Alpine + gcompat (ld-linux shim), the case a
# loader-file heuristic misclassifies as glibc.
# Expected runtime ~6-12 min.
set -o pipefail
RUN=musltest BENCH=swepro
source /scratch/eh3323/apex/repo/scripts/eval_common.sh   # env.sh, harness.env, unset BINDPATH/BASH_ENV, PATH
set -u
HB_COMMIT=4778ceba98314a3b0fe119249f7ac84fc0af08b7
MUSL_TASK=${MUSL_TASK:-instance_future-architect__vuls-e1fab805}
GLIBC_TASK=${GLIBC_TASK:-instance_ansible__ansible-c1f2df47}
HARBOR_PKG=${HARBOR_PKG:-overlay}
HOSTPY=$APEX/envs/harbor-hostpy; PYLIB_DEFAULT=$HOSTPY-pylib

WORK=$(node_local_dir musltest) || { echo "FAIL: no node-local work dir"; exit 2; }
STAGING_ROOT=/tmp/$USER-musltest-stg-${SLURM_JOB_ID:-$$}; mkdir -p "$STAGING_ROOT"
cleanup() {
  [ "${KEEP:-0}" = 1 ] && { echo "kept $WORK"; return; }
  chmod -R u+rwX "$WORK" "$STAGING_ROOT" 2>/dev/null; rm -rf "$WORK" "$STAGING_ROOT"
}
trap cleanup EXIT
export APPTAINER_TMPDIR=$WORK/apptainer-tmp; mkdir -p "$APPTAINER_TMPDIR" "$WORK/sbx"

resolve_task() {  # <id prefix> -> full instance id from the HARD-51 list
  local t; t=$(grep -m1 "^$1" "$SWE_IDS") || { echo "task $1 not in $SWE_IDS" >&2; return 1; }; echo "$t"
}
MUSL_ID=$(resolve_task "$MUSL_TASK") && GLIBC_ID=$(resolve_task "$GLIBC_TASK") || exit 2

echo "== host=$(hostname) job=${SLURM_JOB_ID:-none} work=$WORK ($(df -h --output=avail "$WORK" | tail -1 | tr -d ' ') free)"
echo "== $(apptainer --version); pasta: $(command -v pasta || echo MISSING)"
HOST_EGRESS=$(curl -s -o /dev/null -m 10 -w '%{http_code}' https://pypi.org/simple/ 2>/dev/null || echo 000)
echo "== host egress to pypi.org: HTTP $HOST_EGRESS (container 'no egress' checks are only meaningful if the host has it)"

# --- harbor package: installed, or overlay = installed package + singularity backend rebuilt from the repo patch
SITE_PKG=$($EVAL_PY -c 'import harbor, os; print(os.path.dirname(harbor.__file__))') || exit 2
PKG_PATH=""
if [ "$HARBOR_PKG" = overlay ]; then
  OVL=$WORK/overlay; SD=src/harbor/environments/singularity
  mkdir -p "$OVL/$SD" "$WORK/pkg"
  for f in bootstrap.sh server.py singularity.py; do
    git -C "$APEX/work/src/harbor" show "$HB_COMMIT:$SD/$f" > "$OVL/$SD/$f" || { echo "FAIL: git show $f"; exit 2; }
  done
  chmod 755 "$OVL/$SD/bootstrap.sh"
  (cd "$OVL" && git apply "$REPO/patches/harbor_singularity_network_none.patch") || { echo "FAIL: repo patch does not apply"; exit 2; }
  cp -r "$SITE_PKG" "$WORK/pkg/harbor" && cp "$OVL/$SD/"* "$WORK/pkg/harbor/environments/singularity/"
  PKG_PATH=$WORK/pkg
  if cmp -s "$OVL/$SD/singularity.py" "$SITE_PKG/environments/singularity/singularity.py" \
     && cmp -s "$OVL/$SD/bootstrap.sh" "$SITE_PKG/environments/singularity/bootstrap.sh"; then
    echo "== harbor: repo patch == installed package"
  else
    echo "== harbor: testing the REPO patch via overlay (installed package differs: run scripts/eval_setup.sh harbor)"
  fi
else
  echo "== harbor: installed package $SITE_PKG"
fi

# --- pure-Python server deps for musl images
PYLIB_KW=""
if [ -d "$PYLIB_DEFAULT" ]; then
  echo "== pylib: $PYLIB_DEFAULT (picked up by the default <hostpy>-pylib lookup)"
else
  PYLIB_KW=$WORK/pylib
  echo "== pylib: $PYLIB_DEFAULT missing -> temporary build in $PYLIB_KW (run scripts/eval_setup.sh pylib for real runs)"
  "$HOSTPY/bin/python3" -m pip install -q --no-compile --disable-pip-version-check --target "$PYLIB_KW" \
    --only-binary=:all: --platform any --implementation py --abi none --python-version 3.10 \
    fastapi==0.115.14 pydantic==1.10.26 uvicorn==0.34.3 starlette==0.46.2 anyio==4.15.1 \
    exceptiongroup==1.3.1 click==8.5.0 h11==0.16.0 idna==3.20 typing_extensions==4.16.0 \
    || { echo "FAIL: could not build the temporary pylib"; exit 2; }
fi

CASES="[{\"name\":\"musl-none\",\"task\":\"$MUSL_ID\",\"net\":\"none\",\"musl\":true}"
[ "${SKIP_PASTA:-0}" = 1 ] || CASES="$CASES,{\"name\":\"musl-pasta\",\"task\":\"$MUSL_ID\",\"net\":\"pasta\",\"musl\":true}"
CASES="$CASES,{\"name\":\"glibc-none\",\"task\":\"$GLIBC_ID\",\"net\":\"none\",\"musl\":false}]"

cat > "$WORK/driver.py" <<'PY'
"""Start each case with Harbor's SingularityEnvironment exactly as a trial does, run checks, stop."""
import asyncio, json, logging, os, re, sys, time, uuid
from pathlib import Path

cases, work = json.loads(sys.argv[1]), Path(sys.argv[2])
E = os.environ
from harbor.environments.singularity import singularity as sing
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths

print(f"harbor singularity backend: {sing.__file__}", flush=True)
captured: list[list[str]] = []
_orig = asyncio.create_subprocess_exec

async def _spy(*a, **k):  # record the container start command lines built by the backend
    if a and a[0] in ("singularity", "pasta") and "exec" in a:
        captured.append([str(x) for x in a])
    return await _orig(*a, **k)

asyncio.create_subprocess_exec = _spy
results: list[tuple[str, str, str, str]] = []
# plain TCP connects through bash /dev/tcp: no dependence on curl/wget/TLS support in the image
NET = ("timeout 8 bash -c 'exec 3<>/dev/tcp/pypi.org/443' 2>/dev/null && echo dns=OPEN || echo dns=BLOCKED; "
       "timeout 8 bash -c 'exec 3<>/dev/tcp/1.1.1.1/80' 2>/dev/null && echo ip=OPEN || echo ip=BLOCKED; "
       "echo ifaces=$(awk 'NR>2{sub(\":\",\"\",$1); printf \"%s,\", $1}' /proc/net/dev)")


async def run_case(c: dict) -> None:
    name, musl = c["name"], c["musl"]

    def rec(check: str, ok: bool, detail="", soft=False) -> None:
        st = "PASS" if ok else ("WARN" if soft else "FAIL")
        detail = " ".join(str(detail).split())[:170]
        results.append((name, check, st, detail))
        print(f"[{st}] {name:<10} {check:<22} {detail}", flush=True)

    task_dir = Path(E["SWE_TASKS"]) / c["task"]
    task = Task(task_dir)
    tp = TrialPaths(work / "trials" / name)
    tp.mkdir()
    logger = logging.getLogger(f"musltest.{name}")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(work / f"{name}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    kw = {  # == src/eval/harbor_job.py env_kwargs()
        "singularity_image_cache_dir": E["SIF_CACHE"],
        "singularity_network": c["net"],
        "singularity_server_python": E["HOSTPY"],
        "singularity_staging_root": E["STAGING_ROOT"],
        "singularity_server_start_timeout_sec": 300,
        "singularity_sandbox_root": E["SANDBOX_ROOT"],
    }
    if c["net"] == "pasta":
        kw["singularity_ignore_network_policy"] = True
    if E.get("PYLIB_KW"):
        kw["singularity_server_pylib"] = E["PYLIB_KW"]
    mounts = [  # what a trial binds into the agent env
        {"type": "bind", "source": str(tp.verifier_dir.resolve()), "target": "/logs/verifier"},
        {"type": "bind", "source": str(tp.agent_dir.resolve()), "target": "/logs/agent"},
    ]
    env = sing.SingularityEnvironment(
        environment_dir=task.paths.environment_dir, environment_name=task.short_name,
        session_id=f"musltest-{name}-{uuid.uuid4().hex[:6]}", trial_paths=tp,
        task_env_config=task.config.environment, logger=logger, mounts=mounts, **kw)
    n0, t0 = len(captured), time.monotonic()
    try:
        await env.start(force_build=False)
    except Exception as e:
        rec("start", False, f"{type(e).__name__}: {e}")
        tail = (work / f"{name}.log").read_text(errors="replace").splitlines()[-15:]
        print("   log tail:\n   " + "\n   ".join(tail), flush=True)
        try:
            await env.stop(delete=True)
        except Exception:
            pass
        return
    rec("start", True, f"{time.monotonic() - t0:.0f}s, {len(captured) - n0} start attempt(s)")
    argv = captured[-1] if len(captured) > n0 else []
    a = " ".join(argv)
    logger.info(f"start command: {a}")
    flags = [x for i, x in enumerate(argv) if x.startswith("--") or (i and argv[i - 1] in ("-B", "--env", "--network"))]
    print("   flags: " + " ".join(f.split(":")[1] if f.count(":") >= 1 and f.startswith("/") else f
                                 for f in flags)[:400], flush=True)
    rec("detected libc", getattr(env, "_musl_rootfs", None) == musl, f"_musl_rootfs={getattr(env, '_musl_rootfs', None)}")
    want = {"--fakeroot": True, "--containall": True, "--writable": True,
            "--ignore-fakeroot-command": musl, "/opt/harbor-pylib": musl, "/opt/harbor-hostpy": not musl,
            "--network none": c["net"] == "none", "pasta --config-net": c["net"] == "pasta"}
    bad = [f"{k}:{'missing' if v else 'unexpected'}" for k, v in want.items() if (k in a) != v]
    rec("start flags", not bad, ", ".join(bad) or "fakeroot/net/bind flags as expected")

    async def sh(cmd: str, t: int = 120) -> tuple[int, str]:
        r = await env.exec(command=cmd, cwd="/app", timeout_sec=t)
        return r.return_code, (r.stdout or "").strip()

    try:
        rc, out = await sh("id -u; whoami; echo HOME=$HOME")
        rec("uid 0", rc == 0 and out.split()[:2] == ["0", "root"], out)
        rc, out = await sh("echo probe-$$ > /app/.harbor_probe && cat /app/.harbor_probe && rm -f /app/.harbor_probe "
                           "&& touch /root/.harbor_probe && rm -f /root/.harbor_probe && echo ok")
        rec("write /app + /root", rc == 0 and out.endswith("ok"), out)
        rc, out = await sh("git status --porcelain | wc -l; git rev-parse --short HEAD")
        rec("git", rc == 0, out)
        rc, out = await sh("if [ -f go.mod ]; then go version; elif [ -f package.json ]; then "
                           "node --version && (yarn --version || npm --version); else python3 --version; fi")
        rec("toolchain", rc == 0, out)
        rc, out = await sh(NET, 60)
        if c["net"] == "none":
            rec("network isolated", "dns=BLOCKED" in out and "ip=BLOCKED" in out and "ifaces=lo," in out, out)
        else:
            rec("egress (pasta)", "dns=OPEN" in out, out, soft=E.get("HOST_EGRESS", "000") == "000")
        rc, out = await sh("for d in /scratch /scratch/eh3323 /home/eh3323/.ssh; do [ -e $d ] && echo VISIBLE:$d; done; "
                           "echo checked")
        rec("/scratch hidden", "VISIBLE:/scratch/eh3323" not in out and "VISIBLE:/home/eh3323/.ssh" not in out, out)
        rc, out = await sh("for p in /proc/[0-9]*; do tr '\\0' ' ' < $p/cmdline 2>/dev/null; echo; done "
                           "| grep -E '^/[^ ]*python[0-9.]* .*_hbexec[.]py' | head -1 | cut -c1-90")
        exp = "/usr/bin/python3 -E -c" if musl else "/opt/harbor-hostpy/bin/python3"
        rec("server interpreter", out.startswith(exp), out)
        rc, out = await sh("python3 -c 'import fastapi' >/dev/null 2>&1 && echo LEAK || echo clean; "
                           "echo PYTHONPATH=${PYTHONPATH-}; [ -e /opt/harbor-server ] && echo venv-created || true")
        rec("image untouched", "LEAK" not in out and "venv-created" not in out, out, soft=not musl)
        # cheap task-level check (soft): compile the first Go package of the official test command offline
        run_script = (task_dir / "tests" / "run_script.sh")
        if c["net"] == "pasta":
            return  # cheap task check only once per image (in the agent-phase case)
        m = re.search(r"go test[^\n]*?(\./[\w./-]+)", run_script.read_text()) if run_script.exists() else None
        if m:
            rc, out = await sh(f"timeout 180 go test -count=1 -run '^$' {m.group(1)} 2>&1 | tail -2; "
                               "exit ${PIPESTATUS[0]}", 200)
            rec(f"go test -run ^$ {m.group(1)}", rc == 0, out, soft=True)
        elif (await sh("[ -f package.json ]"))[0] != 0:
            rc, out = await sh("timeout 120 python3 -m pytest --version 2>&1 | tail -1", 150)
            rec("pytest --version", rc == 0, out, soft=True)
    finally:
        sbx = env._sandbox_dir
        await env.stop(delete=True)
        rec("sandbox removed", sbx is None or not sbx.exists(), str(sbx))


async def main() -> int:
    for c in cases:
        print(f"--- {c['name']}: {c['task']} (singularity_network={c['net']})", flush=True)
        try:
            await asyncio.wait_for(run_case(c), timeout=int(E.get("CASE_TIMEOUT", "540")))
        except asyncio.TimeoutError:
            results.append((c["name"], "case timeout", "FAIL", E.get("CASE_TIMEOUT", "540")))
            print(f"[FAIL] {c['name']} case timeout", flush=True)
    left = [p.name for p in Path(E["SANDBOX_ROOT"]).glob("sandbox_*")]
    results.append(("all", "no leftover sandboxes", "PASS" if not left else "FAIL", ",".join(left)))
    print("\n================ SUMMARY ================")
    for st in ("FAIL", "WARN"):
        for r in results:
            if r[2] == st:
                print(f"  {st}  {r[0]:<10} {r[1]:<22} {r[3]}")
    n = {s: sum(r[2] == s for r in results) for s in ("PASS", "WARN", "FAIL")}
    verdict = "PASS" if n["FAIL"] == 0 else "FAIL"
    print(f"  {n['PASS']} passed, {n['WARN']} warnings, {n['FAIL']} failed  ->  OVERALL {verdict}")
    return 0 if verdict == "PASS" else 1


sys.exit(asyncio.run(main()))
PY

export SIF_CACHE=$APEX/cache/sif HOSTPY STAGING_ROOT SANDBOX_ROOT=$WORK/sbx \
  PYLIB_KW HOST_EGRESS SWE_TASKS CASE_TIMEOUT=${CASE_TIMEOUT:-540}
T0=$SECONDS
PYTHONPATH=$PKG_PATH $EVAL_PY "$WORK/driver.py" "$CASES" "$WORK"
rc=$?
echo "== finished in $(( (SECONDS - T0) / 60 )) min (logs: $WORK/*.log, removed at exit unless KEEP=1)"
exit $rc
