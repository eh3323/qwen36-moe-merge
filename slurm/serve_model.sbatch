#!/bin/bash
#SBATCH --job-name=apex-serve
#SBATCH --account=torch_pr_147_courant
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200|b200
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --output=/scratch/eh3323/apex/logs/serve/%x-%j.out
# vLLM serve job for one model. Eval jobs (CPU) find it via $APEX/work/endpoints/<RUN>.env.
#
#   sbatch --job-name=serve-base --export=ALL,RUN=base,MODEL_DIR=/scratch/eh3323/apex/models/Qwen3.6-35B-A3B slurm/serve_model.sbatch
#   more GPUs (data parallel, one replica per GPU):  sbatch --gres=gpu:4 --cpus-per-task=64 --mem=600G ...
#
# Identical for every model: tokenizer, chat template and generation_config come from the
# BASE model dir; only the weights (MODEL_DIR) differ. Settings: configs/eval/harness.env.
set -uo pipefail
source /scratch/eh3323/apex/env.sh
source $APEX/repo/configs/eval/harness.env
: "${RUN:?RUN=base|e_swe|e_tb|e_asi|e_hle|mixed|merged_*}" "${MODEL_DIR:?MODEL_DIR=<HF weights dir>}"
NGPU=$(nvidia-smi -L | wc -l)
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
HOST=$(hostname -f 2>/dev/null || hostname)
LOGDIR=$APEX/logs/serve-${RUN}-${SLURM_JOB_ID}
EP_DIR=$APEX/work/endpoints; EP=$EP_DIR/${ENDPOINT_NAME:-$RUN}.env
mkdir -p "$LOGDIR" "$EP_DIR" $APEX/logs/serve
source $APEX/envs/serve/bin/activate
# per-job node-local Triton cache: concurrent jobs compiling the same kernels into the shared /scratch cache
# hit "OSError: [Errno 116] Stale file handle" (NFS) and the engine dies (3 RTX PRO 6000 jobs, 2026-10-14)
export TRITON_CACHE_DIR=/tmp/$USER-triton-${SLURM_JOB_ID:-x}

if [ -f "$EP" ]; then  # another live serve job for this RUN? then do not clobber it
  source "$EP"
  if [ "${SERVE_JOB_ID:-}" != "$SLURM_JOB_ID" ] && curl -sf -m 10 "http://$ENDPOINT_HOST:$ENDPOINT_PORT/health" >/dev/null; then
    echo "endpoint for RUN=$RUN already served by job $SERVE_JOB_ID; exiting"; exit 0
  fi
fi

cleanup() {
  # remove the endpoint file only if it is ours
  grep -q "SERVE_JOB_ID=$SLURM_JOB_ID" "$EP" 2>/dev/null && rm -f "$EP"
  kill $VLLM_PID $KEEP_PID 2>/dev/null; wait 2>/dev/null
}
trap cleanup EXIT
trap 'echo "SIGTERM (preemption?)"; exit 143' TERM

echo "host=$HOST port=$PORT ngpu=$NGPU RUN=$RUN MODEL_DIR=$MODEL_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
DP_ARGS=(); [ "$NGPU" -gt 1 ] && DP_ARGS=(--data-parallel-size "$NGPU")
# <120GB cards (RTX PRO 6000 96GB): only ~700 Mamba cache blocks fit, below vLLM's default max_num_seqs 1024, and
# vLLM refuses to start. 512 still exceeds every client's concurrency (HLE_N 256). H200/B200 keep the default.
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
SEQ_ARGS=(); [ "${GPU_MIB:-0}" -lt 120000 ] && SEQ_ARGS=(--max-num-seqs "${MAX_NUM_SEQS:-512}")
VLLM_ARGS=(
  "$MODEL_DIR" --served-model-name "$SERVED_MODEL_NAME" --host 0.0.0.0 --port "$PORT"
  --tokenizer "$BASE_MODEL_DIR" --chat-template "$BASE_MODEL_DIR/chat_template.jinja"
  --generation-config "$BASE_MODEL_DIR"
  --override-generation-config "$OVERRIDE_GENERATION_CONFIG"
  --default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS"
  --tensor-parallel-size 1 "${DP_ARGS[@]}" --max-model-len "$MAX_MODEL_LEN"
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --enable-prefix-caching --gpu-memory-utilization 0.88 --seed 0 "${SEQ_ARGS[@]}"
)
printf '%q ' vllm serve "${VLLM_ARGS[@]}" > "$LOGDIR/vllm_cmd.txt"; echo >> "$LOGDIR/vllm_cmd.txt"
pip freeze 2>/dev/null > "$LOGDIR/serve_env_freeze.txt"
vllm serve "${VLLM_ARGS[@]}" > "$LOGDIR/vllm.log" 2>&1 &
VLLM_PID=$!

# Utilization filler (cluster cancels H200 jobs <60% util over 2h); backs off when vLLM is busy.
RLLM_GPU_SHARE_DEVICES=all RLLM_GPU_SHARE_TARGET_UTIL=92 RLLM_GPU_SHARE_LOG_CSV=$LOGDIR/keepalive.csv \
  python $APEX/work/gpu_keepalive_shared.py > "$LOGDIR/keepalive.log" 2>&1 &
KEEP_PID=$!

for i in $(seq 1 180); do  # up to 30 min for load + compile
  curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died during startup"; tail -50 "$LOGDIR/vllm.log"; exit 1; }
  sleep 10
done
curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "vLLM not healthy after 30 min"; exit 1; }
if [ "${SMOKE:-0}" = 1 ]; then
  python $APEX/work/smoke_serve.py --base-url "http://127.0.0.1:$PORT/v1" > "$LOGDIR/smoke_serve.txt" 2>&1
  tail -3 "$LOGDIR/smoke_serve.txt"
fi

cat > "$EP.tmp" <<EOF
RUN=$RUN
MODEL_DIR=$MODEL_DIR
ENDPOINT_HOST=$HOST
ENDPOINT_PORT=$PORT
API_BASE=http://$HOST:$PORT/v1
SERVED_MODEL=$SERVED_MODEL_NAME
SERVE_JOB_ID=$SLURM_JOB_ID
NGPU=$NGPU
STARTED=$(date -Is)
SERVE_LOGDIR=$LOGDIR
EOF
mv "$EP.tmp" "$EP"
echo "endpoint written: $EP"; cat "$EP"

# health loop: 3 consecutive failures -> exit non-zero (eval-side proxy waits for a requeue)
fails=0
while kill -0 $VLLM_PID 2>/dev/null; do
  if curl -sf -m 30 "http://127.0.0.1:$PORT/health" >/dev/null; then fails=0; else fails=$((fails+1)); fi
  [ $fails -ge 3 ] && { echo "$(date -Is) health check failed 3x"; exit 1; }
  sleep 60
done
echo "vLLM exited"; tail -50 "$LOGDIR/vllm.log"; exit 1
