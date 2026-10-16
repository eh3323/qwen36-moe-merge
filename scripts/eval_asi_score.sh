#!/bin/bash
# Score an ASI-Bench agent run locally (seed31415 public references; non-official score).
# 13/60 tasks need the official GPT-5.5 judge -> OPENAI_API_KEY from your shell only:
#   read -rs OPENAI_API_KEY && export OPENAI_API_KEY
#   RUN=base RUN_ID=base__asi__b3 scripts/eval_asi_score.sh
# Without a key the judge tasks come back evaluation_invalid (reported separately, not dropped silently).
# Login node is fine for a few instances; for full runs prefer an interactive CPU session (srun).
set -euo pipefail
: "${RUN:?}" "${RUN_ID:?}"
BENCH=asi; source /scratch/eh3323/apex/repo/scripts/eval_common.sh
GROUP=$(group_of "$RUN"); RAW_DIR=$APEX/results/$GROUP/asi/$RUN_ID; NORM_DIR=$REPO/results/$GROUP/asi; mkdir -p "$NORM_DIR"
[ -d "$RAW_DIR/run" ] || { echo "no run dir $RAW_DIR/run"; exit 2; }
[ -n "${OPENAI_API_KEY:-}" ] || echo "WARNING: OPENAI_API_KEY not set -> judge tasks will be evaluation_invalid"
# Judge temperature: the judges ask for temperature=0, which gpt-5.5 rejects ("Only the default (1) value is
# supported"; 2026-10-15: 8/60 Base instances evaluation_invalid, first in LiteLLM's own gpt-5 check, then at the API
# once the endpoint was named explicitly). src/eval/asi_judge_site/sitecustomize.py (on PYTHONPATH for this command
# only) removes the temperature from gpt-5* calls, so the judge runs at the API default; nothing else changes.
# $ASI_TASKS_ROOT on PYTHONPATH: some scorers import their helpers as `tasks.<domain>.<task>` (cmos_opamp_design),
# which the official editable install provides via the repo root; without it that scorer crashed (2026-10-16).
[ -f "$RAW_DIR/local_score_seed31415.json" ] && mv "$RAW_DIR/local_score_seed31415.json" "$RAW_DIR/local_score_seed31415.prev.$(date +%m%d%H%M).json"
env -u OPENAI_BASE_URL -u OPENAI_API_BASE PYTHONPATH=$REPO/src/eval/asi_judge_site:$ASI_TASKS_ROOT $APEX/envs/asi/bin/python $APEX/envs/asi/bin/asibench score --repo seed31415 \
  --results-dir "$RAW_DIR/run" --instances-dir "$ASI_FULL/tasks" --tasks-dir "$ASI_TASKS_ROOT/tasks" \
  --judge-api-base "${ASI_JUDGE_API_BASE:-https://api.openai.com/v1}" --judge-api-key-env OPENAI_API_KEY --judge-api-protocol openai \
  --parallel ${PARALLEL:-4} --output "$RAW_DIR/local_score_seed31415.json" 2>&1 | tail -20 \
  || echo "[asi] asibench score exited non-zero: instances it could not score stay evaluation_invalid in the report"
$EVAL_PY $REPO/src/eval/normalize.py asi --run-dir "$RAW_DIR/run" --score "$RAW_DIR/local_score_seed31415.json" \
  --run-id "$RUN_ID" --out "$NORM_DIR/$RUN_ID.jsonl"
echo "normalized -> $NORM_DIR/$RUN_ID.jsonl"
