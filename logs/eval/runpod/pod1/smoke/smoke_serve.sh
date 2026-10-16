#!/bin/bash
# vLLM smoke test: exactly the VLLM_ARGS of slurm/serve_model.sbatch (H200: no --max-num-seqs), port 8199, GPU 0
set -uo pipefail
export USER=root
source /scratch/eh3323/apex/env.sh
source $APEX/repo/configs/eval/harness.env
source $APEX/envs/serve/bin/activate
export TRITON_CACHE_DIR=/tmp/$USER-triton-smoke CUDA_VISIBLE_DEVICES=0
MODEL_DIR=$APEX/models/merged/ties_l1_d0.5_v4
PORT=8199
VLLM_ARGS=(
  "$MODEL_DIR" --served-model-name "$SERVED_MODEL_NAME" --host 0.0.0.0 --port "$PORT"
  --tokenizer "$BASE_MODEL_DIR" --chat-template "$BASE_MODEL_DIR/chat_template.jinja"
  --generation-config "$BASE_MODEL_DIR"
  --override-generation-config "$OVERRIDE_GENERATION_CONFIG"
  --default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS"
  --tensor-parallel-size 1 --max-model-len "$MAX_MODEL_LEN"
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --enable-prefix-caching --gpu-memory-utilization 0.88 --seed 0
)
printf "%q " vllm serve "${VLLM_ARGS[@]}"; echo
exec vllm serve "${VLLM_ARGS[@]}"
