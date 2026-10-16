#!/bin/bash
# one opencode 1.17.15 agent turn against the smoke vLLM (same provider config the asibench opencode_cli adapter builds);
# also populates the shared $XDG_CACHE_HOME/opencode provider-package cache before 24 agents start at once
set -uo pipefail
export USER=root; source /scratch/eh3323/apex/env.sh
export OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_MODELS_FETCH=1 OPENCODE_DISABLE_LSP_DOWNLOAD=1 OPENCODE_DISABLE_SHARE=1
W=$(mktemp -d $APEX/cache/tmp/oc-smoke.XXXXXX); cd $W
export XDG_DATA_HOME=$W/.state/data XDG_STATE_HOME=$W/.state/state
export OPENCODE_CONFIG_CONTENT='{"provider":{"bench":{"npm":"@ai-sdk/openai-compatible","name":"ASI-Bench Target","options":{"baseURL":"http://127.0.0.1:8199/v1","apiKey":"EMPTY"},"models":{"qwen3.6-35b-a3b":{"name":"qwen3.6-35b-a3b"}}}},"agent":{"build":{"tools":{"webfetch":false,"websearch":false}}}}'
echo "Use the bash tool to run: python3 -c \"print(6*7)\" > answer.txt . Then reply with the single word DONE." | \
  timeout 600 $APEX/tools/bin/opencode run --format json --auto -m bench/qwen3.6-35b-a3b > $APEX/logs/smoke/opencode_out.jsonl 2> $APEX/logs/smoke/opencode_err.log
echo "rc=$? workspace=$W"; ls -la $W; cat $W/answer.txt 2>/dev/null
