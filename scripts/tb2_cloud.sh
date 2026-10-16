#!/bin/bash
# Terminal-Bench 2.0 on a rented single VM (8x H100/H200, Docker), mirroring the cluster harness
# (configs/eval/harness.env: same vLLM 0.30.0 env and flags, Terminus-2 via Harbor @4778ceba, TB2 @2fd12b88,
# 3 h / 32 CPU / 48 GB per trial, T=1.0 top_p 0.95 top_k 20, max_tokens 81920, 256K ctx).
# Differences from the cluster, disclosed: Harbor's native Docker environment instead of the patched singularity one,
# and on 80 GB cards each replica is tensor-parallel over 2 GPUs (67 GiB of weights do not leave room for a KV cache
# on one H100). All models evaluated here (Base, final merge, Mixed, E-TB) share this node and setup, so their
# comparison is like-for-like.
#
#   rsync the repo to the node (e.g. ~/apex/repo), then on the node:
#   bash scripts/tb2_cloud.sh setup           # Docker, uv, envs, Harbor, TB2 dataset, models, images  (~60-90 min)
#   bash scripts/tb2_cloud.sh serve           # one vLLM replica per slot (see SLOTS)
#   bash scripts/tb2_cloud.sh run             # every attempt of every slot, all in parallel (~8 h)
#   bash scripts/tb2_cloud.sh status          # progress per attempt
#   bash scripts/tb2_cloud.sh normalize       # results/<group>/tb2/<run>__tb2__a<k>.jsonl (copy back to the cluster)
#
# SLOTS: "<gpus>:<run>:<attempts>" list, one vLLM replica per slot (TP = number of GPUs), all attempts of a slot
# share its replica. RUN names *_cloud keep these attempts apart from the cluster runs in summarize.py.
set -euo pipefail
W=${W:-$HOME/apex}; REPO=${REPO:-$W/repo}
SUDO=; [ "$(id -u)" = 0 ] || SUDO=sudo
# -i 0, not `| head -1`: under pipefail, nvidia-smi killed by SIGPIPE (busy GPUs) made every subcommand exit 141
GPU_MIB=$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || true)
if [ "${GPU_MIB:-0}" -lt 120000 ]; then   # H100 80GB: TP=2
  DEFAULT_SLOTS="0,1:base_cloud:1,2 2,3:merged_v4_cloud:1,2 4,5:mixed_cloud:1 6,7:e_tb_v2_cloud:1"
else                                       # H200/B200: TP=1, one attempt per GPU
  DEFAULT_SLOTS="0:base_cloud:1 1:base_cloud:2 2:merged_v4_cloud:1 3:merged_v4_cloud:2 4:mixed_cloud:1 5:e_tb_v2_cloud:1 6:base_cloud:3 7:merged_v4_cloud:3"
fi
SLOTS=${SLOTS:-$DEFAULT_SLOTS}
N_CONCURRENT=${N_CONCURRENT:-12}            # trials per attempt (cluster runs after 10-15: 12)
set -a; source "$REPO/configs/eval/harness.env"; set +a
BASE_MODEL_DIR=$W/models/Qwen3.6-35B-A3B; TB2_DIR=$W/data/terminal-bench-2
export PATH=$HOME/.local/bin:$W/envs/eval/bin:$PATH TB2_DIR HARBOR_ENV=docker
HF_USER=eh3323
model_of() {
  case "$1" in
    base*) echo "$BASE_MODEL_DIR" ;; merged_v4*) echo "$W/models/merged_v4" ;;
    mixed*) echo "$W/models/mixed" ;; e_tb_v2*) echo "$W/models/e_tb_v2" ;;
    *) echo "unknown run $1" >&2; exit 2 ;;
  esac
}
group_of() { case "$1" in base*) echo base ;; merged*) echo merged ;; mixed*) echo mixed ;; e_*) echo experts ;; esac; }

docker_setup() {
  command -v docker >/dev/null || curl -fsSL https://get.docker.com | $SUDO sh
  # ~72 concurrent trials, one compose network each: Docker's default address pools hold only ~30 networks
  # ("could not find an available, non-overlapping IPv4 address pool")
  local tmp; tmp=$(mktemp)
  $SUDO cat /etc/docker/daemon.json 2>/dev/null | python3 -c 'import json, sys
s = sys.stdin.read().strip(); d = json.loads(s) if s else {}
d["default-address-pools"] = [{"base": "10.210.0.0/16", "size": 24}]
print(json.dumps(d, indent=2))' > "$tmp"
  if ! $SUDO cmp -s "$tmp" /etc/docker/daemon.json; then
    $SUDO mkdir -p /etc/docker && $SUDO cp "$tmp" /etc/docker/daemon.json && $SUDO systemctl restart docker
  fi
  docker info >/dev/null 2>&1 || { $SUDO usermod -aG docker "$USER"; echo "added $USER to group docker: log in again, re-run setup"; exit 3; }
}

setup() {
  mkdir -p "$W"/{envs,models,data,logs,results,endpoints,src}
  docker_setup
  nvidia-smi -L
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  # exactly the cluster's envs/serve (see scripts/asi_cloud.sh): torch 2.13.0+cu129 from the PyTorch cu129 index
  [ -x "$W/envs/serve/bin/vllm" ] || { uv venv -q --python 3.12 "$W/envs/serve"; VIRTUAL_ENV=$W/envs/serve uv pip sync -q \
    "$REPO/configs/serve.lock.txt" --default-index "${UV_DEFAULT_INDEX:-https://pypi.org/simple}" \
    --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match; }
  # Triton JIT-compiles a C helper against Python.h when vLLM starts (missing on stock Ubuntu images)
  [ -f "$("$W/envs/serve/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')/Python.h" ] || {
    $SUDO apt-get update -qq && $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-dev >/dev/null; }
  if ! ldconfig -p | grep -q 'libcudart.so.13 '; then   # vllm 0.30.0 wheel links the CUDA 13 runtime (nvidia/cu13 in the env)
    local cu13=$W/envs/serve/lib/python3.12/site-packages/nvidia/cu13/lib
    $SUDO mkdir -p /usr/local/lib/apex-cu13 && $SUDO ln -sf $cu13/libcudart.so.13 $cu13/libnvrtc.so.13 $cu13/libnvrtc-builtins.so.13.* /usr/local/lib/apex-cu13/
    echo /usr/local/lib/apex-cu13 | $SUDO tee /etc/ld.so.conf.d/zz-apex-cu13.conf >/dev/null && $SUDO ldconfig
  fi
  if [ ! -x "$W/envs/eval/bin/harbor" ]; then   # the cluster's envs/eval versions (env/eval.freeze.txt), unpatched Harbor
    [ -d "$W/src/harbor/.git" ] || git clone -q https://github.com/laude-institute/harbor.git "$W/src/harbor"
    git -C "$W/src/harbor" checkout -q 4778ceba98314a3b0fe119249f7ac84fc0af08b7
    uv venv -q --python 3.12 "$W/envs/eval"
    if VIRTUAL_ENV=$W/envs/eval uv pip install -q -r <(grep -v '^harbor @' "$REPO/env/eval.freeze.txt"); then
      VIRTUAL_ENV=$W/envs/eval uv pip install -q --no-deps "$W/src/harbor"
    else
      VIRTUAL_ENV=$W/envs/eval uv pip install -q "$W/src/harbor" "litellm==1.93.0" pyyaml aiohttp "huggingface_hub[hf_xet]"
    fi
  fi
  [ -d "$TB2_DIR/.git" ] || { git clone -q https://github.com/laude-institute/terminal-bench-2.git "$TB2_DIR"; git -C "$TB2_DIR" checkout -q 2fd12b88aafdd04a52c298e3940bcb189f9766d6; }
  "$W/envs/eval/bin/hf" version >/dev/null 2>&1 || VIRTUAL_ENV=$W/envs/eval uv pip install -q "huggingface_hub[hf_xet]"
  local hf=$W/envs/eval/bin/hf
  $hf download Qwen/Qwen3.6-35B-A3B --revision 995ad96e --local-dir "$BASE_MODEL_DIR" >/dev/null &
  $hf download $HF_USER/Qwen3.6-35B-A3B-apex-merged-v4 --local-dir "$W/models/merged_v4" >/dev/null &
  $hf download $HF_USER/Qwen3.6-35B-A3B-apex-mixed-folded --local-dir "$W/models/mixed" >/dev/null &
  $hf download $HF_USER/Qwen3.6-35B-A3B-apex-e_tb_v2-folded --local-dir "$W/models/e_tb_v2" >/dev/null &
  # pre-pull the 89 task images (8 at a time) so attempts do not race on first pulls
  grep -hE '^\s*docker_image' "$TB2_DIR"/*/task.toml | cut -d'"' -f2 | sort -u | xargs -P 8 -n 1 docker pull -q >/dev/null
  wait
  echo "setup done: $(ls -d "$TB2_DIR"/*/ | wc -l) TB2 tasks, $(docker images -q | wc -l) images, models:"
  du -sh "$W"/models/*
}

start_vllm() {  # start_vllm <gpus> <run> <port> <log>
  local gpus=$1 run=$2 port=$3 log=$4 tp; tp=$(echo "$gpus" | tr , '\n' | wc -l)
  SEQ_ARGS=(); [ "${GPU_MIB:-0}" -lt 120000 ] && SEQ_ARGS=(--max-num-seqs 512)   # as slurm/serve_model.sbatch
  # env bin on PATH as after the cluster's `source envs/serve/bin/activate` (FlashInfer's JIT runs ninja from there)
  PATH=$W/envs/serve/bin:$PATH CUDA_VISIBLE_DEVICES=$gpus nohup "$W/envs/serve/bin/vllm" serve "$(model_of "$run")" \
    --served-model-name "$SERVED_MODEL_NAME" --host 127.0.0.1 --port $port \
    --tokenizer "$BASE_MODEL_DIR" --chat-template "$BASE_MODEL_DIR/chat_template.jinja" \
    --generation-config "$BASE_MODEL_DIR" --override-generation-config "$OVERRIDE_GENERATION_CONFIG" \
    --default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS" --tensor-parallel-size "$tp" \
    --max-model-len "$MAX_MODEL_LEN" --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --enable-prefix-caching --gpu-memory-utilization 0.88 --seed 0 "${SEQ_ARGS[@]}" > "$log" 2>&1 &
}

serve() {
  for s in $SLOTS; do
    IFS=: read -r gpus run atts <<<"$s"; g0=${gpus%%,*}; port=$((8100 + g0)); tp=$(echo "$gpus" | tr , '\n' | wc -l)
    start_vllm "$gpus" "$run" $port "$W/logs/vllm-$run.log"
    printf 'ENDPOINT_HOST=127.0.0.1\nENDPOINT_PORT=%s\nMODEL_DIR=%s\nTP=%s\nGPU_NAME=%s\n' $port "$(model_of "$run")" "$tp" \
      "'$(nvidia-smi -i $g0 --query-gpu=name --format=csv,noheader)'" > "$W/endpoints/$run.env"
  done
  for s in $SLOTS; do
    IFS=: read -r gpus run atts <<<"$s"; port=$((8100 + ${gpus%%,*}))
    until curl -sf -m 5 http://127.0.0.1:$port/health >/dev/null; do sleep 15; done; echo "vLLM up: $run on GPU $gpus (:$port)"
  done
}

# expand <gpus> <port> <run_id>...: once those attempts have finished, stop the replica on <port> and serve the final
# merge on its GPUs as an extra replica behind src/eval/sticky_lb.py (2026-10-15: two v4 attempts sharing one TP=2
# replica ran at half the cluster's per-trial speed). Same model, flags and sampling; only throughput changes.
expand() {
  local gpus=$1 port=$2; shift 2; local runs; runs=$(IFS='|'; echo "$*")
  while pgrep -f "harbor run -c $W/results/raw/($runs)/" >/dev/null; do sleep 60; done
  echo "$(date -Is) attempts finished: $*; replacing the replica on :$port"
  local pid; pid=$(ss -ltnpH "sport = :$port" | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  [ -n "$pid" ] && kill "$pid"
  for g in ${gpus//,/ }; do
    until [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 10; done
  done
  start_vllm "$gpus" merged_v4_cloud $port "$W/logs/vllm-merged_v4_cloud-$port.log"
  until curl -sf -m 5 http://127.0.0.1:$port/health >/dev/null; do sleep 15; done
  echo "127.0.0.1:$port" >> "$W/endpoints/merged_v4_cloud.backends"
  echo "$(date -Is) v4 replica up on GPU $gpus (:$port); backends: $(tr '\n' ' ' < "$W/endpoints/merged_v4_cloud.backends")"
}

run() {
  for s in $SLOTS; do
    IFS=: read -r gpus run atts <<<"$s"; g0=${gpus%%,*}
    for att in ${atts//,/ }; do
      rid=${run}__tb2__a$att; raw=$W/results/raw/$rid; mkdir -p "$raw"; pport=$((8200 + 10 * g0 + att))
      nohup "$W/envs/eval/bin/python" -I "$REPO/src/eval/llm_proxy.py" --endpoint-file "$W/endpoints/$run.env" \
        --listen 127.0.0.1:$pport --log "$raw/llm_requests.jsonl" > "$W/logs/proxy-$rid.log" 2>&1 &
      sleep 2
      PROXY_BASE=http://127.0.0.1:$pport/v1 HOSTED_VLLM_API_KEY=EMPTY TB2_DIR=$TB2_DIR \
        "$W/envs/eval/bin/python" "$REPO/src/eval/harbor_job.py" tb2 --job-name "$rid" --jobs-dir "$raw/jobs" -k 1 \
        -n "$N_CONCURRENT" --out "$raw/job_config.json" >/dev/null
      printf '{"run_id": "%s", "host": "%s", "gpu": "%s", "tensor_parallel": %s, "model_dir": "%s", "n_concurrent": %s, "harness_env_sha256": "%s", "backend": "docker", "started": "%s"}\n' \
        "$rid" "$(hostname)" "$(nvidia-smi -i $g0 --query-gpu=name --format=csv,noheader)" "$(echo "$gpus" | tr , '\n' | wc -l)" \
        "$(model_of "$run")" "$N_CONCURRENT" "$(sha256sum "$REPO/configs/eval/harness.env" | cut -d' ' -f1)" "$(date -Is)" > "$raw/run_meta.json"
      (cd "$raw" && HOSTED_VLLM_API_KEY=EMPTY nohup "$W/envs/eval/bin/harbor" run -c "$raw/job_config.json" -y \
        > "$W/logs/harbor-$rid.log" 2>&1 &)
      echo "started $rid on GPU $gpus (proxy :$pport)"
    done
  done
}

# extra <run> <rid> <proxy_port> <tasks_csv>: one more harbor job for a subset of tasks, as `run` does for a slot
# (same harness, endpoint and sampling; concurrency = number of tasks). 2026-10-16: the tasks the v4 attempt a1 had not
# started yet, so they need not wait for a1's 12 slots; for those tasks this job's trials count, a1's later duplicates
# of them are discarded (rule fixed before any of them ran).
extra() {
  local run=$1 rid=$2 pport=$3 tasks=$4 n; n=$(echo "$tasks" | tr , '\n' | wc -l)
  local raw=$W/results/raw/$rid; mkdir -p "$raw"
  nohup "$W/envs/eval/bin/python" -I "$REPO/src/eval/llm_proxy.py" --endpoint-file "$W/endpoints/$run.env" \
    --listen 127.0.0.1:$pport --log "$raw/llm_requests.jsonl" > "$W/logs/proxy-$rid.log" 2>&1 &
  sleep 2
  PROXY_BASE=http://127.0.0.1:$pport/v1 HOSTED_VLLM_API_KEY=EMPTY TB2_DIR=$TB2_DIR \
    "$W/envs/eval/bin/python" "$REPO/src/eval/harbor_job.py" tb2 --job-name "$rid" --jobs-dir "$raw/jobs" -k 1 \
    -n "$n" --tasks "$tasks" --out "$raw/job_config.json" >/dev/null
  printf '{"run_id": "%s", "host": "%s", "model_dir": "%s", "n_concurrent": %s, "tasks": "%s", "harness_env_sha256": "%s", "backend": "docker", "started": "%s"}\n' \
    "$rid" "$(hostname)" "$(model_of "$run")" "$n" "$tasks" "$(sha256sum "$REPO/configs/eval/harness.env" | cut -d' ' -f1)" \
    "$(date -Is)" > "$raw/run_meta.json"
  (cd "$raw" && HOSTED_VLLM_API_KEY=EMPTY nohup "$W/envs/eval/bin/harbor" run -c "$raw/job_config.json" -y \
    > "$W/logs/harbor-$rid.log" 2>&1 &)
  echo "started $rid: $n tasks (proxy :$pport)"
}

status() {
  for d in "$W"/results/raw/*__tb2__a*; do
    j=$d/jobs/$(basename "$d"); echo "$(basename "$d"): trials=$(ls -d "$j"/*__*/ 2>/dev/null | wc -l) done=$(ls "$j"/*__*/result.json 2>/dev/null | wc -l)"
  done
  free -g | sed -n 1,2p; uptime; docker ps -q | wc -l | sed 's/^/containers=/'
}

normalize() {
  for d in "$W"/results/raw/*__tb2__a*; do
    rid=$(basename "$d"); run=${rid%%__*}; att=${rid##*__a}; g=$(group_of "$run")
    mkdir -p "$REPO/results/$g/tb2"
    "$W/envs/eval/bin/python" "$REPO/src/eval/normalize.py" harbor --bench tb2 --job-dir "$d/jobs/$rid" --run-id "$rid" \
      --attempt "$att" --out "$REPO/results/$g/tb2/$rid.jsonl"
  done
}

"${1:?setup|serve|run|status|normalize|expand|extra}" "${@:2}"
