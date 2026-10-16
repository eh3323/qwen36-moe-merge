#!/bin/bash
# ASI-Bench B3 agent phase on a rented GPU node, running the cluster scripts UNCHANGED by recreating the cluster
# layout (/scratch/eh3323/apex -> $WS): same vLLM serve flags, same opencode 1.17.15 agent. Container hosts that forbid
# user namespaces (RunPod: unshare -> EPERM) cannot run the net-locked linux_ns sandbox, so ASI_SANDBOX=task is the
# default here: asibench's per-task uv venv (same prewarmed task envs as linux_ns) without namespaces, i.e. no net lock,
# no filesystem hiding; recorded in run_meta.json and disclosed. (asibench's --sandbox os is Docker: not available.)
# The node never holds the reference answers: only the agent-visible instances (work/asi_instances_noref) are copied;
# scoring is on the cluster.
#
#   on your Mac:  rsync -az --exclude .git <repo>/ node:$WS/repo/
#                 rsync -az torch:/scratch/eh3323/apex/work/asi_instances_noref/ /tmp/asi_noref/ && \
#                 rsync -az /tmp/asi_noref/ node:$WS/work/asi_instances_noref/
#   on the node:  bash $WS/repo/scripts/asi_cloud.sh check          # sandbox (user namespaces), GPUs, disk
#                 export HF_TOKEN=...; bash $WS/repo/scripts/asi_cloud.sh setup; unset HF_TOKEN   (~40-60 min)
#                 bash $WS/repo/scripts/asi_cloud.sh run            # one allinone ASI run per slot, in parallel
#                 bash $WS/repo/scripts/asi_cloud.sh status
#                 bash $WS/repo/scripts/asi_cloud.sh pack           # -> $WS/asi_results.tgz (copy back to the cluster)
set -euo pipefail
WS=${WS:-/workspace/apex}; APEX=/scratch/eh3323/apex; REPO=$APEX/repo
SLOTS=${SLOTS:-"0:base:Qwen3.6-35B-A3B 1:merged_v4:merged/ties_l1_d0.5_v4"}   # e.g. SLOTS="0,1:base:Qwen3.6-35B-A3B" ASI_PAR=24
export USER=${USER:-root} PATH=$HOME/.local/bin:$PATH

check() {
  unshare --user --map-root-user --mount --net -- bash -c 'ip link set lo up && mount -t tmpfs t /mnt && echo SANDBOX_OK'
  nvidia-smi -L; df -h "$WS" | tail -1; echo "cpus=$(nproc)"; free -g | head -2
}

setup() {
  mkdir -p /scratch/eh3323/.huggingface "$WS"/{logs,work,tools/bin,models/merged,data/benchmarks,cache/tmp,envs,results}
  [ -e $APEX ] || ln -s "$WS" $APEX
  cp $REPO/env/env.sh $APEX/env.sh
  # the outer ASI namespace mounts an empty tmpfs over the reference copy: the path must exist (it stays empty here)
  mkdir -p $APEX/data/benchmarks/asibench_seed31415
  [ -d $APEX/work/asi_instances_noref ] || { echo "copy work/asi_instances_noref from the cluster first"; exit 2; }
  if command -v apt-get >/dev/null; then apt-get update -qq && apt-get install -y -qq iproute2 rsync git curl openssl >/dev/null; fi
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  source $APEX/env.sh
  [ -d $APEX/data/benchmarks/asibench_repo/.git ] || git clone -q https://github.com/apexin-ai/ASI-Bench.git $APEX/data/benchmarks/asibench_repo
  # exactly the cluster's envs/serve (configs/serve.lock.txt: torch 2.13.0+cu129 from the PyTorch cu129 index); a bare
  # `uv pip install vllm==0.30.0` now pulls PyPI's CUDA-13 torch build instead. UV_DEFAULT_INDEX may name a PyPI mirror
  # (PyPI's CDN can be very slow from rented nodes; uv also reads it for the asi/eval/task envs).
  [ -x $APEX/envs/serve/bin/vllm ] || { uv venv -q --python 3.12 $APEX/envs/serve; VIRTUAL_ENV=$APEX/envs/serve uv pip sync -q \
    $REPO/configs/serve.lock.txt --default-index "${UV_DEFAULT_INDEX:-https://pypi.org/simple}" \
    --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match; }
  # vllm 0.30.0 (PyPI wheel, CUDA 13 build) links libcudart.so.13 / libnvrtc.so.13, which torch cu129 does not preload
  # and a CUDA-12 container image does not have: register the copies shipped in the env (nvidia/cu13) with ld.so
  if ! ldconfig -p | grep -q 'libcudart.so.13 '; then
    local cu13=$APEX/envs/serve/lib/python3.12/site-packages/nvidia/cu13/lib
    mkdir -p /usr/local/lib/apex-cu13 && ln -sf $cu13/libcudart.so.13 $cu13/libnvrtc.so.13 $cu13/libnvrtc-builtins.so.13.* /usr/local/lib/apex-cu13/
    echo /usr/local/lib/apex-cu13 > /etc/ld.so.conf.d/zz-apex-cu13.conf && ldconfig
  fi
  [ -x $APEX/envs/eval/bin/python ] || { uv venv -q --python 3.12 $APEX/envs/eval; VIRTUAL_ENV=$APEX/envs/eval uv pip install -q aiohttp "huggingface_hub[hf_xet]"; }
  bash $REPO/scripts/eval_setup.sh asi opencode asi_tasks asi_prewarm
  local hf=$APEX/envs/eval/bin/hf
  $hf download Qwen/Qwen3.6-35B-A3B --revision 995ad96e --local-dir $APEX/models/Qwen3.6-35B-A3B >/dev/null
  if echo "$SLOTS" | grep -q merged_v4; then
    $hf download eh3323/Qwen3.6-35B-A3B-apex-merged-v4 --local-dir $APEX/models/merged/ties_l1_d0.5_v4 >/dev/null
  fi
  echo "setup done: $(ls $APEX/work/asi_instances_noref | wc -l) instances, task envs: $(ls $APEX/cache/asibench/task-envs | wc -l)"
}

run() {
  local n; n=$(echo $SLOTS | wc -w)
  for s in $SLOTS; do
    IFS=: read -r gpu run mdir <<<"$s"          # gpu may be a list ("0,1"): one data-parallel vLLM over those GPUs
    CUDA_VISIBLE_DEVICES=$gpu NGPU_OVERRIDE=$(echo "$gpu" | tr , '\n' | wc -l) SLURM_JOB_ID=$((900000 + ${gpu%%,*})) \
      SLURM_CPUS_PER_TASK=$(( $(nproc) / n )) SLURM_MEM_PER_NODE=$(( $(free -m | awk '/Mem:/{print $2}') / n )) \
      RUN=$run MODEL_DIR=$APEX/models/$mdir BENCHES=asi ASI_LEVELS=b3 ASI_PAR=${ASI_PAR:-12} ASI_SANDBOX=${ASI_SANDBOX:-task} \
      nohup bash $REPO/slurm/eval_allinone.sbatch > $APEX/logs/aio-$run.log 2>&1 &
    echo "started $run on GPU $gpu (log $APEX/logs/aio-$run.log)"
  done
}

status() {
  for d in $APEX/results/*/asi/*__asi__b3; do
    [ -d "$d" ] || continue
    echo "$(basename "$d"): results=$(ls "$d"/run/*/*__b3.json 2>/dev/null | wc -l)/60 requests=$(wc -l < "$d/llm_requests.jsonl" 2>/dev/null || echo 0)"
  done
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; uptime
}

pack() {
  # tmp/ holds the agent workspaces (run/instances/_workspaces/*/workspace_b3 are absolute symlinks into it):
  # scoring reads the agents' output files there, so it is packed too and must be unpacked at the same path
  cd $APEX/results && tar czf "$WS/asi_results.tgz" */asi/*__asi__b3
  ls -la "$WS/asi_results.tgz"
}

"${1:?check|setup|run|status|pack}"
