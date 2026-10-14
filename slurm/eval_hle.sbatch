#!/bin/bash
#SBATCH --job-name=apex-hle
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --output=/scratch/eh3323/apex/logs/eval/%x-%j.out
# HLE (cais/hle @5a81a4c7, all 2500 incl. 342 image questions), official hle_eval prediction script
# (+ patches/hle_eval_apex.patch: sampling, timeout). Generation only; judging with o3-mini needs
# the OpenAI key -> scripts/eval_hle_judge.sh on the login node (key never enters a job/file).
#
#   sbatch --job-name=hle-base --export=ALL,RUN=base slurm/eval_hle.sbatch
#   smoke (5 questions incl. 1 image, net probe):  sbatch --time=1:00:00 --export=ALL,RUN=base,SMOKE=1 slurm/eval_hle.sbatch
# Re-running the same RUN_ID only fills in missing/failed questions (official resume behaviour).
RUN=${RUN:?} BENCH=hle
source /scratch/eh3323/apex/repo/scripts/eval_common.sh
SMOKE=${SMOKE:-0}; SEED_TAG=${SEED_TAG:-s1}
TAG=$SEED_TAG; [ "$SMOKE" = 1 ] && TAG=smoke
init_run "$TAG"
[ "$SMOKE" = 1 ] && net_probe
wait_endpoint || exit 3
start_proxy
trap 'kill $PROXY_PID 2>/dev/null' EXIT
export OPENAI_BASE_URL=$PROXY_BASE OPENAI_API_KEY=EMPTY HLE_REQUEST_TIMEOUT=$HLE_REQUEST_TIMEOUT_SEC
export HLE_SAMPLING=$(printf '{"temperature": %s, "top_p": %s, "presence_penalty": %s, "extra_body": {"top_k": %s}}' \
  "$TEMPERATURE" "$TOP_P" "$HLE_PRESENCE_PENALTY" "$TOP_K")
IDS_ARG=()
if [ "$SMOKE" = 1 ]; then
  IDS=$($HLE_PY -I -c "
import pandas as pd, glob
d = pd.read_parquet(glob.glob('$HLE_DATASET/data/test-*.parquet')[0], columns=['id','image'])
t = d[d.image.fillna('') == ''].id.head(4).tolist(); i = d[d.image.fillna('') != ''].id.head(1).tolist()
print(','.join(t + i))")
  IDS_ARG=(--ids "$IDS")
fi
write_run_meta "{\"hle_sampling\": $HLE_SAMPLING, \"max_completion_tokens\": $MAX_TOKENS, \"smoke\": $SMOKE}"
cd "$RAW_DIR"
for pass in 1 2 3; do   # official script skips failed calls; re-running fills them in
  $HLE_PY $APEX/work/hle_eval/run_model_predictions.py --dataset "$HLE_DATASET" --model "$SERVED_MODEL_NAME" \
    --max_completion_tokens "$MAX_TOKENS" --num_workers "${HLE_N:-$HLE_WORKERS}" "${IDS_ARG[@]}" 2>&1 | tail -20 | tee -a "$LOGDIR/hle_pred.log"
done
$HLE_PY $REPO/src/eval/normalize.py hle --pred "$RAW_DIR/hle_$SERVED_MODEL_NAME.json" --dataset "$HLE_DATASET" \
  ${IDS:+--ids "$IDS"} --run-id "$RUN_ID" --out "$NORM_DIR/$RUN_ID.jsonl"
echo "[eval] generation done; judge with: RUN=$RUN RUN_ID=$RUN_ID scripts/eval_hle_judge.sh"
