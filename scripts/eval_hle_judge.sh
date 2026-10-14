#!/bin/bash
# Judge HLE predictions with the OFFICIAL unmodified run_judge_results.py (o3-mini-2025-01-31).
# Run on the LOGIN node (light: API calls only). The key is read from your shell environment only:
#   read -rs OPENAI_API_KEY && export OPENAI_API_KEY      # never written to any file
#   RUN=base RUN_ID=base__hle__s1 scripts/eval_hle_judge.sh
set -euo pipefail
: "${RUN:?}" "${RUN_ID:?}" "${OPENAI_API_KEY:?export OPENAI_API_KEY in this shell first}"
BENCH=hle; source /scratch/eh3323/apex/repo/scripts/eval_common.sh
GROUP=$(group_of "$RUN"); RAW_DIR=$APEX/results/$GROUP/hle/$RUN_ID; NORM_DIR=$REPO/results/$GROUP/hle; mkdir -p "$NORM_DIR"
PRED=$RAW_DIR/hle_$SERVED_MODEL_NAME.json
JUDGED=$RAW_DIR/judged_$(basename "$PRED").json   # upstream naming: judged_<predictions basename>.json
[ -f "$PRED" ] || { echo "no predictions at $PRED"; exit 2; }
cd "$RAW_DIR"
# judge talks to api.openai.com: make sure no vLLM base URL leaks in from the environment
set +e
env -u OPENAI_BASE_URL -u OPENAI_API_BASE $HLE_PY $APEX/work/hle_eval/run_judge_results.py \
  --dataset "$HLE_DATASET" --predictions "$PRED" --judge "$HLE_JUDGE" --num_workers ${JUDGE_WORKERS:-50} \
  2>&1 | tee "$RAW_DIR/judge_stdout.txt" | tail -5
rc=${PIPESTATUS[0]}
set -e
if [ "$rc" -ne 0 ]; then
  # upstream calib_err (beta=100 bins) raises IndexError for < 100 judged rows; judgments are already saved
  if [[ $RUN_ID == *smoke* ]] && grep -q "in calib_err" "$RAW_DIR/judge_stdout.txt" && [ -f "$JUDGED" ]; then
    echo "[judge] official calib_err needs >= 100 rows (beta=100): metrics skipped for smoke run"
  else
    echo "[judge] run_judge_results.py failed (rc=$rc); see $RAW_DIR/judge_stdout.txt"; exit $rc
  fi
fi
[ -f "$JUDGED" ] || { echo "no judged file at $JUDGED"; exit 3; }
$HLE_PY $REPO/src/eval/normalize.py hle --pred "$PRED" --judged "$JUDGED" \
  --dataset "$HLE_DATASET" --run-id "$RUN_ID" $( [[ $RUN_ID == *smoke* ]] && echo --pred-only-ids ) --out "$NORM_DIR/$RUN_ID.jsonl"
echo "normalized -> $NORM_DIR/$RUN_ID.jsonl"
