#!/bin/bash
# One-time setup of the evaluation tooling on torch (run by hand on the LOGIN node; every step is
# idempotent and only downloads/unpacks: no compilation). Usage:
#   bash scripts/eval_setup.sh all            # or a subset, in this order:
#   bash scripts/eval_setup.sh harbor hostpy pylib hle asi opencode pasta asi_noref asi_tasks asi_prewarm dirs verify
# Locked versions: harbor @4778ceba (+patch), mini-swe-agent 2.4.6, asibench @8859f9b (+patch),
# opencode 1.17.15 (standalone binary from the npm registry, sha512-verified; no Node.js needed),
# hle_eval @22ed3074 deps (openai 1.57.0, datasets 3.2.0, numpy 1.26.4).
set -euo pipefail
source /scratch/eh3323/apex/env.sh
REPO=$APEX/repo; source $REPO/configs/eval/harness.env
SRC=$APEX/work/src; mkdir -p $SRC $APEX/tools/bin $APEX/cache/sif $APEX/work/endpoints $APEX/logs/eval $APEX/logs/serve
PY312=3.12
STEPS=("$@"); [ "${#STEPS[@]}" -eq 0 ] && { sed -n 2,9p "$0"; exit 1; }
[ "${STEPS[0]}" = all ] && STEPS=(harbor hostpy pylib hle asi opencode pasta asi_noref asi_tasks asi_prewarm dirs verify)

checkout() {  # <src repo> <commit> <dest> <patch|-> : pristine local clone at the locked commit, then patch
  local src=$1 commit=$2 dest=$3 patch=$4
  if [ ! -d "$dest/.git" ]; then
    git clone -q --no-checkout "$src" "$dest" && git -C "$dest" checkout -q "$commit"
  fi
  [ "$(git -C "$dest" rev-parse HEAD)" = "$commit" ] || { echo "$dest not at $commit"; exit 1; }
  if [ "$patch" != - ]; then
    if git -C "$dest" apply --check -R "$patch" 2>/dev/null; then echo "already patched: $dest"; return; fi
    # an older revision of the patch may be applied: start again from the pristine commit
    git -C "$dest" apply --check "$patch" 2>/dev/null || git -C "$dest" reset -q --hard "$commit"
    if git -C "$dest" apply --check "$patch" 2>/dev/null; then git -C "$dest" apply "$patch"; echo "patched $dest"
    else echo "patch does not apply cleanly to $dest"; exit 1; fi
  fi
}

step_harbor() {  # envs/eval: patched Harbor + mini-swe-agent 2.4.6 (host_mini_swe runs in-process)
  checkout $APEX/data/benchmarks/harbor_repo 4778ceba98314a3b0fe119249f7ac84fc0af08b7 $SRC/harbor \
    $REPO/patches/harbor_singularity_network_none.patch
  (cd $SRC/harbor && uv export --frozen --no-dev --no-hashes --no-emit-workspace --no-header -q -o $APEX/work/harbor_constraints.txt)
  [ -x $APEX/envs/eval/bin/python ] || uv venv -q --python $PY312 $APEX/envs/eval
  # --reinstall-package: same version string, so uv would otherwise keep the previously patched build
  VIRTUAL_ENV=$APEX/envs/eval uv pip install -q -c $APEX/work/harbor_constraints.txt --reinstall-package harbor \
    "$SRC/harbor" "mini-swe-agent==2.4.6" pyyaml aiohttp
  $APEX/envs/eval/bin/python -c "import inspect, harbor.environments.singularity.singularity as s; print('harbor patch: sandbox', 'singularity_sandbox_root' in inspect.getsource(s))"
  VIRTUAL_ENV=$APEX/envs/eval uv pip freeze > $APEX/work/envs_eval_freeze.txt
}

step_hostpy() {  # relocatable python + fastapi/uvicorn, bound into containers for Harbor's exec server
  local H=$APEX/envs/harbor-hostpy
  if [ ! -x $H/bin/python3 ]; then
    uv python install -q $PY312
    cp -a "$(dirname "$(dirname "$(uv python find $PY312)")")" $H
    rm -f $H/lib/python3.12/EXTERNALLY-MANAGED
  fi
  uv pip install -q --python $H/bin/python3 --system --break-system-packages \
    "fastapi==0.136.3" "uvicorn==0.54.0" "pydantic==2.14.1"
  $H/bin/python3 -c "import fastapi, uvicorn; print('hostpy ok', fastapi.__version__, uvicorn.__version__)"
}

step_pylib() {  # exec-server deps for musl (Alpine) task images (10 of HARD-51), where the glibc hostpy
  # cannot run: harbor binds <hostpy>-pylib at /opt/harbor-pylib and runs the server on the image's own
  # python3 (3.10-3.12), see patches/harbor_singularity_network_none.patch. py3-none-any wheels only (no
  # compiled code: any libc / CPython >= 3.10), hence pydantic v1 + fastapi 0.115. All pins explicit:
  # pip evaluates markers for the host's 3.12, so exceptiongroup (anyio on 3.10) would otherwise be missing.
  local P=$APEX/envs/harbor-hostpy-pylib T
  T=$(mktemp -d $APEX/envs/.pylib.XXXXXX)
  $APEX/envs/harbor-hostpy/bin/python3 -m pip install -q --no-compile --disable-pip-version-check --target $T \
    --only-binary=:all: --platform any --implementation py --abi none --python-version 3.10 \
    fastapi==0.115.14 pydantic==1.10.26 uvicorn==0.34.3 starlette==0.46.2 anyio==4.15.1 \
    exceptiongroup==1.3.1 click==8.5.0 h11==0.16.0 idna==3.20 typing_extensions==4.16.0
  if find $T -name '*.so' | grep -q .; then echo "compiled files in $T"; exit 1; fi
  rm -rf $P.old; if [ -d $P ]; then mv $P $P.old; fi; mv $T $P; rm -rf $P.old; chmod -R a+rX $P
  $APEX/envs/harbor-hostpy/bin/python3 -I -c "import sys; sys.path.insert(0, '$P'); import fastapi, uvicorn, pydantic, exceptiongroup; print('pylib ok', fastapi.__version__, uvicorn.__version__, pydantic.VERSION)"
}

step_hle() {  # envs/hle + patched copy of hle_eval
  [ -x $APEX/envs/hle/bin/python ] || uv venv -q --python $PY312 $APEX/envs/hle
  VIRTUAL_ENV=$APEX/envs/hle uv pip install -q "openai==1.57.0" "datasets==3.2.0" "numpy==1.26.4" pandas "pyarrow==17.0.0" tqdm  # pyarrow>=18 需要 numpy 2
  mkdir -p $APEX/work/hle_eval
  cp $REPO/src/eval/hle/upstream/run_model_predictions.py $REPO/src/eval/hle/upstream/run_judge_results.py $APEX/work/hle_eval/
  (cd $APEX/work/hle_eval && git apply -p1 $REPO/patches/hle_eval_apex.patch)  # 登录节点没有 patch 命令
  echo "hle_eval patched at $APEX/work/hle_eval"
}

step_asi() {  # envs/asi: patched asibench[full] (non-editable; caches redirected off $HOME)
  checkout $APEX/data/benchmarks/asibench_repo 8859f9b44ada4a98d82d6eb328b94a704cd84e80 $SRC/asibench \
    $REPO/patches/asibench_linux_ns_hide.patch
  (cd $SRC/asibench && uv export --frozen --no-dev --no-hashes --no-emit-project --no-header --extra full -q -o $APEX/work/asi_constraints.txt) || true
  [ -x $APEX/envs/asi/bin/python ] || uv venv -q --python $PY312 $APEX/envs/asi
  local C=(); [ -s $APEX/work/asi_constraints.txt ] && C=(-c $APEX/work/asi_constraints.txt)
  VIRTUAL_ENV=$APEX/envs/asi uv pip install -q "${C[@]}" "$SRC/asibench[full]"
  # asibench keeps task venvs/uv cache in ~/.asibench; $HOME has a 30k-file quota -> move to /scratch
  mkdir -p $APEX/cache/asibench
  if [ -e ~/.asibench ] && [ ! -L ~/.asibench ]; then echo "~/.asibench exists (not a symlink): move it manually"; exit 1; fi
  [ -L ~/.asibench ] || ln -s $APEX/cache/asibench ~/.asibench
  VIRTUAL_ENV=$APEX/envs/asi uv pip freeze > $APEX/work/envs_asi_freeze.txt
}

step_opencode() {  # standalone Bun-compiled binary, no Node.js needed
  local v=$OPENCODE_VERSION tgz=$APEX/work/opencode-linux-x64-$OPENCODE_VERSION.tgz
  local want='sha512-qFVYvLYGW4RElqrJ0mfhQJWFu4lYS/hwT/Izag+r4Q5EbmlLZcmnXOeA+tXkDjA6R2MCxL58NkNgs1QDM2bTqw=='
  [ -f $tgz ] || curl -fsSL -o $tgz https://registry.npmjs.org/opencode-linux-x64/-/opencode-linux-x64-$v.tgz
  local got="sha512-$(openssl dgst -sha512 -binary $tgz | base64 -w0)"
  [ "$got" = "$want" ] || { echo "opencode tarball integrity mismatch"; exit 1; }
  tar -xzf $tgz -C $APEX/work package/bin/opencode
  install -m 755 $APEX/work/package/bin/opencode $APEX/tools/bin/opencode
  $APEX/tools/bin/opencode --version
}

step_pasta() {  # passt/pasta is installed on the login nodes only; TB2 + SWE re-grade sandboxes need it on
  # compute nodes (private netns with NAT egress). It links only libc, so copy the login node's binary
  # (same RHEL 10.2 image) into $APEX/tools/bin, which every eval job puts on PATH.
  [ -x /usr/bin/passt ] || { echo "/usr/bin/passt not found on $(hostname)"; exit 1; }
  install -m 755 /usr/bin/passt $APEX/tools/bin/pasta   # same binary; argv[0] "pasta" selects pasta mode
  # the generic build re-execs "<argv0>.avx2" when the CPU has AVX2 (else warns and stays generic)
  [ -x /usr/bin/passt.avx2 ] && install -m 755 /usr/bin/passt.avx2 $APEX/tools/bin/pasta.avx2
  echo "pasta from $(rpm -q passt 2>/dev/null || echo passt) sha256=$(sha256sum $APEX/tools/bin/pasta | cut -c1-16)"
  $APEX/tools/bin/pasta --version | head -1
}

step_asi_noref() {  # agent-visible instances: no reference/, no solver.py, instance_meta -> params_used only
  local D=$ASI_NOREF
  mkdir -p $D
  rsync -a --delete --exclude 'reference/' --exclude 'solver.py' $ASI_FULL/tasks/ $D/
  $APEX/envs/asi/bin/python -I - "$D" <<'PY'
import json, sys, pathlib
for m in pathlib.Path(sys.argv[1]).glob("*/instance_meta.json"):
    meta = json.loads(m.read_text())
    m.write_text(json.dumps({"params_used": meta.get("params_used", {})}, indent=2))
PY
  echo "instances: $(ls $D | wc -l) dirs; reference dirs left: $(find $D -name reference | wc -l)"
}

ASI_TASKS_ROOT=$APEX/work/asi_runtime   # same path as scripts/eval_common.sh
step_asi_tasks() {  # tasks/ outside the source checkout -> asibench uses the ~/.asibench task-env cache
  mkdir -p $ASI_TASKS_ROOT/tasks
  rsync -a --delete $ASI_REPO/tasks/ $ASI_TASKS_ROOT/tasks/
  echo "asi tasks: $(find $ASI_TASKS_ROOT/tasks -name task_meta.yaml | wc -l) task dirs in $ASI_TASKS_ROOT/tasks"
}

step_asi_prewarm() {  # build every task venv while network is available (agent phase is offline)
  (cd $ASI_TASKS_ROOT && $APEX/envs/asi/bin/python -I $REPO/src/eval/asi_prewarm.py --tasks-dir $ASI_TASKS_ROOT/tasks)
}

step_dirs() {
  mkdir -p $APEX/cache/sif $APEX/cache/mswea $APEX/results $APEX/work/endpoints
  echo "dirs ok"
}

step_verify() {
  set +e
  $APEX/envs/eval/bin/harbor --version
  $APEX/envs/eval/bin/python -c "import inspect, harbor.environments.singularity.singularity as s; print('harbor patch:', 'singularity_network' in inspect.getsource(s))"
  $APEX/envs/eval/bin/python -c "import minisweagent; print('mini-swe-agent', minisweagent.__version__)"
  $APEX/envs/asi/bin/asibench --help >/dev/null && echo "asibench ok"
  $APEX/envs/asi/bin/python -c "import inspect, ai4sci_bench.runner.linux_ns_sandbox as m; print('asibench patch:', '_HIDE_SCRIPT' in inspect.getsource(m))"
  $APEX/tools/bin/opencode --version
  $APEX/tools/bin/pasta --version | head -1
  $APEX/envs/hle/bin/python -c "import openai, datasets; print('hle deps', openai.__version__, datasets.__version__)"
  $APEX/envs/harbor-hostpy/bin/python3 -c "import fastapi, uvicorn; print('hostpy ok')"
  $APEX/envs/harbor-hostpy/bin/python3 -I -c "import sys; sys.path.insert(0, '$APEX/envs/harbor-hostpy-pylib'); import fastapi, pydantic; assert fastapi.__file__.startswith('$APEX/envs/harbor-hostpy-pylib'); print('pylib ok', pydantic.VERSION)"
  ls ~/.asibench -ld
}

for s in "${STEPS[@]}"; do echo "=== $s"; "step_$s"; done
