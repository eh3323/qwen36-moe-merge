# Training data: four LoRA experts + mixed control

Base model: `Qwen/Qwen3.6-35B-A3B`. Everything below is produced by `src/data/` with the settings in
`configs/data/sources.yaml`. Raw and processed data live on the cluster under
`/scratch/eh3323/apex/data/{raw,interim,processed}` and are not in git. Git holds only the manifests in
`data/manifests/`: ids, sha256, counts and drop statistics. After a build, `SUMMARY.md` there has the
final numbers.

## 1. Pipeline and reproduction

```
scripts/setup_data_env.sh          # uv venv /scratch/eh3323/apex/envs/data (CPython 3.12)
scripts/prepare_data.sh            # Slurm chain: prep -> 4 builds (parallel) -> finalize
  slurm/data_prep.sbatch           #   src.data.download (pinned revisions) + src.data.dev_sets
  slurm/data_build.sbatch EXPERT=  #   src.data.build --expert {swe,tb,asi,hle}
  slurm/data_finalize.sbatch       #   src.data.mix + src.data.verify + src.data.summarize
scripts/fetch_data_manifests.sh    # copy $APEX/data/manifests -> data/manifests (git)
```

For each expert, `build.py` runs these steps in order:
1. source filter
2. cheap decontamination (repo/name, canary, exact match, 13-gram)
3. conversion to the eval harness format
4. exact token count
5. embedding decontamination (ASI, HLE)
6. dedup
7. stratified, token-budgeted selection (deterministic, by sha256 hash)
8. 2 % validation split, grouped by `instance_id`

Every drop is counted by `source:reason` in `manifests/<expert>_stats.json`.

`src/data/selftest.py` runs every converter on a few hundred real rows. The whole chain was also run end
to end on a tiny copy of the data, from dev sets through verify.

**Benchmark firewall.** Every benchmark or raw read goes through `common.guard_path()`, which raises if
the path contains `seed31415`. That ASI-Bench split ships reference answers and is never read.
`verify.py` also scans all outputs for the HLE canary and for the string `seed31415`.

## 2. Unified sample schema (`processed/<expert>/{train,val}.jsonl`)

```json
{"id": "...", "source": "...", "instance_id": "...", "n_tokens": 12345, "meta": {...},
 "tools": [ {OpenAI function schema} ] | null,
 "messages": [{"role": "system|user|assistant|tool", "content": "...", "reasoning_content": "...",
               "tool_calls": [{"id": "call_0", "type": "function",
                               "function": {"name": "bash", "arguments": {"command": "..."}}}],
               "tool_call_id": "call_0"}]}
```

- **Tool-call arguments are JSON objects, not strings.** The Qwen3.6 chat template iterates
  `tool_call.arguments|items`.
- **Thinking.** `reasoning_content` holds the teacher's thinking.
- **`n_tokens`** is the exact length of `tokenizer.apply_chat_template(messages, tools,
  chat_template=<repo chat_template.jinja>, preserve_thinking=True)` with the Qwen3.6 tokenizer.
- **Why `preserve_thinking=True`.** By default the template drops `<think>` for assistant turns *before
  the last real user message*. We keep thinking in every assistant turn so that every trained turn
  teaches reasoning (`configs/train/base.yaml: preserve_thinking: true`). How this matches inference
  depends on the harness:
  - **mini-swe-agent (E-SWE)** and **opencode (E-ASI)**: all agent turns come after the single user
    query, so the stock template also keeps their thinking at inference. Training and inference match.
  - **Terminus-2 (E-TB)**: every observation is a *user* message, and Harbor does not resend reasoning
    (`interleaved_thinking=False`). At inference, earlier turns therefore have no thinking, while in
    training they do. This is a known and accepted train/inference mismatch.
  - **E-HLE**: single turn, so the choice has no effect.

## 3. Sources

Licences and teachers were checked on the HF cards on 2026-10-13. Only open-weight teachers with
permissive licences are used.

| key | HF id @ revision | licence | teacher | used for |
|---|---|---|---|---|
| swe_rebench_oh | nebius/SWE-rebench-openhands-trajectories @35455389 | CC-BY-4.0 | Qwen3-Coder-480B-A35B (Apache-2.0) | E-SWE primary |
| swe_hero_oh | nvidia/SWE-Hero-openhands-trajectories @150bc119 | CC-BY-4.0 (+ repo licences) | Qwen3-Coder-480B-A35B | E-SWE secondary |
| nemotron_terminal | nvidia/Nemotron-Terminal-Corpus @a1667c4f | CC-BY-4.0 | DeepSeek-V3.2 (MIT) | E-TB, E-ASI |
| ot_agent_100k | open-thoughts/OpenThoughts-Agent-SFT-100K @45fb28fc | Apache-2.0 | GLM-4.7-AWQ (MIT) | E-TB |
| jupyter_agent | jupyter-agent/jupyter-agent-dataset @10a8dd46 (6/103 `thinking` shards) | Apache-2.0 (Kaggle ToS caveat) | Qwen3-Coder-480B / Qwen3-32B | E-ASI |
| nemotron_science | nvidia/Nemotron-Science-v1 @82e1af46 (MCQ, RQA) | CC-BY-4.0 | gpt-oss-120b (Apache-2.0) | E-HLE |
| nemotron_math_v2 | nvidia/Nemotron-Math-v2 @8e793210, **streamed** `data/high.part_00.jsonl`, AoPS rows only | CC-BY-4.0 (AoPS; CC-BY-SA StackExchange rows skipped) | gpt-oss-120b | E-HLE |
| osr2 | nvidia/OpenScienceReasoning-2 @174b02c9, **streamed** two-pass (`input` column of all 9 row groups, full rows from the ≤2 richest), non-STEM slice | CC-BY-4.0 | DeepSeek-R1-0528 (MIT) | E-HLE |
| swe_rebench_v2 | nebius/SWE-rebench-V2 @10483de0 | CC-BY-4.0 | – | SWE dev |
| tblite | open-thoughts/OpenThoughts-TBLite @7b701113 | Apache-2.0 | – | TB dev |
| scienceagentbench | osunlp/ScienceAgentBench @9c6e96c9 | CC-BY-4.0 | – | ASI dev |
| gpqa | Idavidrein/gpqa @83022cef (gated) | CC-BY-4.0 | – | HLE dev |
| embed_model | BAAI/bge-small-en-v1.5 | MIT | – | embedding decontamination (CPU) |

**Excluded by decision:**
- Closed-model teachers or ToS risk: SWE-smith-trajectories, R2E-Gym-SFT, SWE-Gym SFT, tmax-sft (contains
  Gemini outputs), Kwai-Klear 66k (teacher undisclosed), data-agent-sft (teacher undisclosed).
- Non-commercial licences: MegaScience, natural_reasoning.
- Contamination risk: every HLE derivative, and every TB1/`tbench_oracle`-derived set. The TB1 snapshot
  contains 88 of the 89 TB2 tasks.

The Nemotron-Math-v2 repo is 204 GB and is never downloaded. One 38 GB JSONL shard is read as an HTTP
stream and stopped early, after at most 400k lines or once the candidate pool is full.

## 4. Per-expert recipes

### E-SWE: mini-swe-agent `mini_toolcall.yaml` (SWE-bench Pro locked protocol)

**Format.** Copied verbatim at build time from
`swebench_pro_repo/v2/tooling/configs/mini_toolcall.yaml`:
- `system_template`
- `instance_template`
- `observation_template`, rendered with jinja2 exactly as mini-swe-agent does. This gives the same
  `tojson` escaping and the head 5000 + tail 5000 truncation above 10 000 characters.

Other format details:
- **Tool.** The single `bash` tool schema of mini-swe-agent 2.4.6.
- **`{{task}}`.** The issue text wrapped in the exact SWE-Pro V2 `instruction.md` frame ("A code
  repository is available in the `/app` directory … `<pr_description>` … Constraints").
- **System info.** `system_info` is set in the config.
- **Paths.** The OpenHands repository root `/workspace/<repo>` is rewritten to `/app` everywhere.
- **Submit.** The last turn is the submit command `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.

**OpenHands → bash conversion** (`convert_swe.py`):

| OpenHands call | Converted to |
|---|---|
| `execute_bash` | Same command. Prefixed with `cd <cwd> &&` when OpenHands' persistent cwd ≠ `/app`, because mini-swe-agent runs each command in a fresh subshell. Status trailers are stripped and the exit code becomes `returncode`. Echoed command lines are removed. |
| `view` (file) | `nl -ba file [| sed -n 'a,bp']`. Same `cat -n` format. |
| `view` (directory) | `find … -maxdepth 2 -not -path '*/.*' \| sort` |
| `create` | `cat <<'EOF' > file` |
| `str_replace` / `insert` | A `python3` heredoc that does an exact, unique replacement and prints the same numbered snippet OpenHands showed. OpenHands error messages are mapped to that script's errors. |
| `think`, `task_tracker` | Folded into `reasoning_content` |
| `finish` | Submit command |

Other conversion rules:
- **Dropped turns.** A failed editor call that has no faithful shell equivalent is removed, and its
  reasoning is carried forward. Examples: `view_range` past EOF, `create` on an existing file. At most 3
  such turns per trajectory.
- **Dropped trajectories.**
  - interactive input (`is_input`), soft timeouts, or negative exit codes
  - stateful `export` / `source`
  - `undo_edit`
  - no `finish`
- **Thinking.** Qwen3-Coder is a non-thinking model. Its visible pre-call text, plus `think` thoughts,
  becomes `reasoning_content`, and `content` is left empty. About 28 % of turns have no pre-call text
  and so have an empty think block.

**Filters**, in this order:
1. Primary source: `resolved == 1` and `exit_status == submit`.
2. SWE-Hero: only R2E-Gym-sourced rows, which the card describes as execution-based. There is no
   resolved column.
3. The final patch changes at least one non-test, non-scratch source file.
4. At most 64 Ki tokens.

**Decontamination:**
- the 11 SWE-Pro repos (normalised `owner/repo`)
- the whole `ansible/*` org
- a fork heuristic: the repo basename equals a SWE-Pro basename, or the patch touches distinctive
  SWE-Pro paths such as `lib/ansible/`, `qutebrowser/`, `openlibrary/` (full list in the config)
- a 13-gram overlap (at least 1 gram, boilerplate grams shared by 5 or more references removed) between
  the first user message and the 642 V2 `problem_statement + requirements + interface`
- the dev-set repos

**Selection:**
- Stratified with "hard" = at least 2 changed source files or at least 40 assistant turns, targeted at
  50 %.
- SWE-Hero gets 25 % of the set.
- At most 2 trajectories per instance and at most 3 % per repo.

### E-TB: native Terminus-2 JSON (Harbor `terminus-2`, `templates/terminus-json-plain.txt` @4778ceba)

**Template check:**
- OT-Agent first prompts are byte-identical to the current Harbor template.
- Nemotron-Terminal uses an **older** Terminus-2 revision. One sentence about trailing newlines differs.
  We therefore always re-render the first user message as
  `template.format(instruction=…, terminal_state=…)`, which is what Terminus-2 does.
- Later observation strings already match Harbor: "New Terminal Output:", "Current Terminal Screen:",
  the completion double-confirmation, and the parse-error feedback.

**Normalisation:**
- The `<think>` block becomes `reasoning_content`.
- GLM's ```` ```json ```` fences are stripped.
- Teacher-artefact warnings ("Extra text detected before/after JSON object") are removed from the
  following observation.

**Rules.** Every kept assistant turn must be a bare, valid JSON object with
`analysis`/`plan`/`commands`. Trajectories are dropped if they have:
- any parse-error turn
- summarisation or context-warning hand-offs
- a missing think block

**Sources and selection:**
- Nemotron skill_based, medium + mixed, all categories except scientific_computing, data_science and
  model_training. Those go to E-ASI.
- OT-Agent-SFT-100K without the `swesmith` slice (keeps it disjoint from E-SWE). Rows with
  `result ∈ {None, ""}` only; error rows are dropped.
- Targets: 60 / 40 Nemotron / OT; at most 15 % trajectories without a final `task_complete`, kept for
  error-recovery behaviour; at most 2 per task; at most 32 Ki tokens.

**Decontamination:**
- 13-gram (at least 3 grams) of the task instruction against all 89 TB2 `instruction.md` +
  `solution/*` + `tests/*`
- task name in the 90 unsafe TB1 names or the 89 TB2 names
- TB2 task names that appear in the instruction
- TBLite dev task names

### E-ASI: opencode 1.17.15 native tool calling, bash tool only (`convert_asi.py`)

This is the eval agent of ASI-Bench's `opencode_cli` adapter. The vendored pieces are in
`configs/data/opencode/`, from github.com/anomalyco/opencode tag v1.17.15 = `5fb0d1cd`.

**System prompt:**
- `session/prompt/default.txt`. This is the provider prompt chosen for a model id that is not
  GPT/Claude/Gemini/Kimi.
- Plus the `<env>` block from `session/system.ts`:
  - model `bench/Qwen/Qwen3.6-35B-A3B`
  - working directory `/app`
  - not a git repo
  - linux
  - a fixed date

**bash tool:**
- **Description.** `tool/shell/prompt.ts` rendered by executing the original template code
  (`render_bash.js`) for bash, linux, 2000 lines / 51 200 bytes, 120 000 ms.
- **Parameters.** `{command, timeout?, workdir?}`.
- **Tool name.** The tool id stays `bash` (`shell/id.ts`).

**Approximations** (verify by logging one real opencode request through a proxy):
- The skills block and AGENTS.md instructions are omitted.
- The JSON-schema rendering of the effect Schema is hand-written.
- Only the bash tool is exposed, by decision. opencode's eval request also lists
  read/write/edit/glob/grep/… tools.

**Tool results:**
- raw output
- `(no output)` when empty
- tail-truncation with opencode's "...output truncated..." header
- no exit code is shown

The episode ends with an assistant message that has no tool calls.

**Nemotron Terminus-2 → opencode:**
- per turn: `reasoning_content` = think; `content` = plan
- one bash call whose command is the turn's keystrokes joined by newlines
- `workdir` is set from the tmux prompt when it differs from `/app`
- the observation is the tmux output with prompts, echoed commands and heredoc `> ` continuation lines
  removed
- the final message is the analysis of the completing turn

Dropped trajectories:
- incomplete
- parse errors
- wait turns
- tmux special keys or interactive programs
- command timeouts

**jupyter-agent → opencode:**
- Each code cell is appended to `analysis.py` and the script is re-run.
- The observation is the concatenated output of cells 1..k, i.e. deterministic re-run semantics.
- The last bare expression is wrapped in `print()`, and `display()` becomes `print()`.
- `!pip` / `%magic`-only cells are skipped.
- The E2B "Output is truncated" marker is removed (approximation).

Dropped:
- cells with errors
- final answer not matching the dataset's `answer`
- a missing think block

**Sources and selection:**
- Nemotron medium + mixed `scientific_computing` / `data_science`, plus medium `model_training`.
- jupyter-agent thinking: 6 shards, about 15 % of the set.
- "long" = at least 12 turns, about 60 % of the Nemotron share.
- At most 1 per task; at most 32 Ki tokens.

**Decontamination:**
- 13-gram (at least 3) of the task prompt against all seed42 B1–B4 prompts and the 62 `task_meta.yaml`
  name/tag/paper strings from `asibench_repo`
- named-method blacklist: ASI task methods such as lid-driven cavity, Ghia, Kosterlitz/XY, Neal's
  funnel, Gillespie, De Young–Keizer, SAPT, metadynamics, Ball–Berry/Farquhar, TOPMODEL/Richards, RNEA,
  pose graph, contact process, Glauber, VPFM, Marcus, VTST, LKH, Max-3-SAT, homotopy continuation,
  conformal map, UKF/bearing-only, particle filter, occupancy grid, ICP, … (full list in the config)
- embedding cosine of at least **0.85** against the B3 prompts. Embeddings use **bge-small-en-v1.5 on
  CPU**, chosen instead of bge-m3/e5-large to stay CPU-cheap. The max cosine of every kept sample is
  stored in `meta.asi_max_cos`.

### E-HLE: minimal text-only QA in the official HLE answer format (`convert_hle.py`)

**Prompt:**
- System = `SYSTEM_PROMPT` from `hle_eval/run_model_predictions.py`, verbatim.
- User = the question. MCQs are rewritten to HLE's "Answer Choices:\nA. …" style, and source
  instructions such as "put the answer inside \boxed{}" are stripped.

**Assistant:**
- `reasoning_content` = the teacher trace.
- `content` = `Explanation: <teacher's visible solution, \boxed removed>\nAnswer: <answer>\nConfidence: <c>%`.

**Confidence heuristic.** The result is clipped to [10, 95] and rounded to a multiple of 5.
- **Nemotron-Math-v2.** These rows carry teacher pass statistics: 8 gpt-oss-120b samples per mode. We
  use the Laplace-smoothed pass rate of the high-reasoning, no-tool mode, `(pass+1)/(count+2)`. This is
  an empirical estimate of how often this reasoning reaches the reference answer.
- **Other sources.** Source prior (MCQ 75, RQA 65, OSR-2 non-STEM 70), minus 5 per hedge per 1k
  reasoning words (at most −20), minus 10 for traces over 3k words, and a further −10 over 6k words.
  Hedge words: maybe, perhaps, not sure, wait, alternatively, …
- **Rationale.** HLE scores calibration error. Teacher traces carry no explicit confidence, so we use
  cheap, monotone proxies rather than a constant 100 %.

**Sources:**
- **Nemotron-Science MCQ / RQA.** Hardness proxy: reasoning at least 3k characters. The teacher's answer
  is used; there is no gold label.
- **Nemotron-Math-v2 AoPS.** Kept when the high-mode pass rate is between 0.125 and 0.75, the low mode
  is at most 0.5, the boxed answer matches `expected_answer`, there are no tool calls, and the row is not
  CC-BY-SA.
- **OSR-2 non-STEM.** Selected by a law/econ/philosophy/history/… keyword classifier with no STEM cue
  words. The answer must match `expected_answer`.
- Mix: 40 % math, 30 % MCQ, 15 % RQA, 15 % OSR-2. Text-only; no image self-distillation.

**Decontamination chain:**
1. HLE canary GUID / "BENCHMARK DATA SHOULD NEVER APPEAR"
2. exact match on the normalised question (NFKC, lower case, LaTeX spacing and `$` stripped) against
   the 2 500 HLE questions
3. 13-gram, at least 1 hit, ignoring grams made only of digits and single-letter LaTeX symbols
4. bge-small cosine of at least **0.90** against all HLE questions (`meta.hle_max_cos`)

Exact duplicate questions are removed.

**Held-out dev items.** 300 items, chosen per source in mix proportion as those with the longest
reasoning, are removed **before** selection and written to `processed/dev/hle_heldout.jsonl`.

## 5. Scale, token matching and mixed control

**Budget.** Every expert gets the same training token budget, 15 M tokens
(`token_budget_per_expert`), with exact Qwen3.6 template tokens including prompt and observations.
Selection stops at the budget or at the per-expert count cap, whichever comes first. Because trajectory
lengths differ by an order of magnitude, the **sample counts differ by design**:
- **SWE.** About 39 k tokens per trajectory (median), so about 400 trajectories for 15 M. Reaching the
  planned ≈800 needs about 30 M: raise `token_budget_per_expert` or the cap.
- **TB.** About 10–23 k per trajectory, so about 900.
- **ASI.** About 15 k per trajectory, so about 1 000.
- **HLE.** Planned at about 2–4 k per sample (2 000-sample cap at 5–7 M tokens). The long-trace hardness
  filter (reasoning ≥ 3 000 chars) made samples about 8 k tokens, so the built set is **1 838 samples /
  15.1 M tokens**: the token budget is hit before the cap, and HLE is token-matched like the others.

**Rationale.** Matching tokens rather than samples equalises optimizer exposure across experts. Agentic
experts get fewer but longer, multi-turn, action-dense samples. HLE stays minimal because QA SFT has low
yield on HLE and could dilute general reasoning.

**Mixed control.** `processed/mixed/{train,val}.jsonl` is the concatenation of the four expert train
(val) sets with **exactly the same ids**, deterministically shuffled. `mix.py` asserts that the id set
equals the union of the expert manifests.

## 6. Dev sets (independent; for checkpoint selection only)

| expert | dev set | manifest |
|---|---|---|
| SWE | 50 SWE-rebench-V2 tasks: 20 Go / 20 Python / 10 TS-JS; medium/hard; LLM-QA confidence ≥ 0.9, no detected issues; one task per repo, preferring repos absent from the trajectory corpora. **These repos are excluded from E-SWE training.** | `dev_swe.csv` |
| TB | OpenThoughts-TBLite minus `grpc-plant-position-server` (near-duplicate of TB2 `kv-store-grpc`) and any TB1-unsafe or TB2 name. Its names are excluded from E-TB training. | `dev_tb.csv` |
| ASI | 35 ScienceAgentBench tasks, round-robin over domains | `dev_asi.csv` |
| HLE | GPQA-Diamond (gated: request access, then re-run `download --only gpqa` and `dev_sets`) + 300 held-out items | `dev_hle_gpqa_diamond.csv`, `dev_hle_heldout.csv` |

## 7. Manifests (`data/manifests/`)

| file | contents |
|---|---|
| `<expert>_{train,val}.csv` | id, source, instance_id, n_tokens, sha256 of canonical JSON `{messages, tools}` |
| `<expert>_stats.json` | counts, tokens by source and stratum, every drop count by `source:reason`, candidate pool sizes, HLE confidence histogram |
| `mixed_{train,val}.csv`, `mixed_stats.json` | mixed control |
| `dev_*.csv`, `dev_sets_info.json` | dev sets |
| `download_record.json` | resolved HF sha and file sizes per source |
| `SUMMARY.md` | human-readable tables (`python -m src.data.summarize`) |

## 8. Known limitations

- **opencode format.** Reproduced from source, not from a captured request. Skills/AGENTS blocks and the
  exact JSON-schema serialisation are approximations, and training exposes only `bash`.
- **SWE format and language.**
  - SWE trajectories come from a non-thinking teacher. About 28 % of turns have an empty think block.
  - Converted `str_replace` commands are synthetic Python heredocs; the teacher did not write them.
  - Training data is Python-only, while HARD-51 is about 60 % Go/TS.
- **E-TB.** Nemotron-Terminal and OT-Agent are not success-filtered, following their papers. Thinking
  history does not match Terminus-2 inference (section 2). Both are fixed for E-TB-v2 only (section 10).
- **jupyter → script conversion.** Replaces stateful notebook execution with re-run semantics, and E2B
  output truncation is approximated.
- **OSR-2 non-STEM selection.** A keyword heuristic; its yield was not measured before the build (the login-node test exceeded the 3 GB memory cap).
- **HLE answers.** Nemotron-Science MCQ/RQA answers are the teacher's own, with no gold label. The
  confidence values are heuristic.
- **HLE answer-line repair (2026-10-13).** The first build wrote Nemotron-Science RQA answers from the raw
  `\boxed{}` content, so multi-line boxed answers (aligned/array blocks) broke the one-line `Answer:` format
  (138/1 838 train, 3/28 val, 21/300 held-out). `convert_hle.py` now applies `clean_answer()` to RQA too, and
  `python -m src.data.repair_hle` applied the same transformation to the built files (selection unchanged,
  `n_tokens`/manifests/stats recomputed, originals in `interim/hle/pre_repair/`).

## 9. Side experiment: E-SWE-v2 (`src/data/build_swe_v2.py`, `configs/train/e_swe_v2.yaml`)

E-SWE did not beat Base on HARD-51. Its training data was 100% Python, while HARD-51 is about 60% Go/TS/JS. It also picked up OpenHands habits: repro scripts and test edits left in the repo. E-SWE-v2 keeps the E-SWE recipe and changes only the data: same schema, same locked `mini_toolcall.yaml` rendering, the same hyperparameters, and the same token budget (14.97M vs 15.05M train tokens, 64K max). It is an ablation and is **not** part of the merge or the mixed control.

- **Source.** `nvidia/Open-SWE-Traces` @ `1db67417…` (CC-BY-4.0), config v1.2, split `minisweagent`, folder `data/minisweagent/qwen38_27b/swe-rebench-v2` (22 shards, 5.48 GB, 53.7k rows). The teacher is Qwen3.8-27B in thinking mode (Apache-2.0, open weights). The trajectories are native mini-swe-agent: one `bash` tool, and the same observation template and escaping as our eval.
- **Conversion.**
  - System and instance text are re-rendered from the PR description, wrapped like a SWE-bench Pro V2 instruction.
  - `/testbed` becomes `/app` everywhere.
  - The teacher's submission protocol does not match ours. It runs `git diff -- <files> > patch.txt && cat patch.txt` and then `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`, whereas our harness captures `git add -A && git diff --cached` after a bare submit. So the first command becomes `git diff -- <files>` (identical output) and the submit becomes the bare command. `?? patch.txt` status lines are removed. Every other mention of patch.txt drops the trajectory, including in assistant text, where it would create a thought/action mismatch.
  - Parallel tool calls are kept (356/383 samples).
- **Filters (drop counts).**

  | Drop reason | Count |
  |---|---|
  | not resolved | 31,007 |
  | other patch.txt use: commands | 9,088 |
  | other patch.txt use: assistant text | 3,546 |
  | other patch.txt use: observations | 87 |
  | language not Go/TS/JS/Python | 2,944 |
  | SWE-rebench-V2 dev-set repo held out | 1,754 |
  | fewer than 30 executed actions | 1,244 |
  | leftover repo files our `git add -A` would submit (approximation from writes, rm/checkout/clean and the last `git status`) | 990 |
  | git-history lookups (`--all`, reflog, fetch, shas) | 578 |
  | residual `/testbed` | 501 |
  | over 64K tokens | 319 |
  | SWE-Pro org / 13-gram | 12 |
  | test or scratch files in the teacher's own patch | 1 |
- **Selection.** From 1,599 candidates: language quota Go 40 / TS+JS 30 / Python 30, with a 75% preference for hard tasks (reference patch with ≥2 files or ≥30 lines), 1 trajectory per instance, at most 36 per repo.
  - **Counts:** Go 157, TS/JS 118, Python 118.
  - **Tokens:** 5.93M / 4.98M / 4.48M, mean 39k tokens per sample.
  - **Shape:** median 39 executed actions and 31 assistant turns, from 263 repos.
  - **Split:** 383 train / 10 val.
- **Limitations.**
  - The leftover-file check is a heuristic.
  - Teacher text still reflects its own workflow, such as creating a patch file with an explicit file list.
  - The training text keeps every turn's reasoning, while inference drops historical thinking (same as E-SWE).

## 10. Side experiment: E-TB-v2 (`src/data/build_tb_v2.py`, `configs/train/e_tb_v2.yaml`, not trained yet)

Why: E-TB scored below Base on TB2. Its data (Nemotron-Terminal-Corpus + OT-Agent-100K) is not
success-filtered: "complete" only means the teacher declared `task_complete`, and 112 trajectories never did.
The expert's TB2 failures claim verifications that never happened. E-TB-v2 keeps the E-TB recipe and changes
only the data.

- **Source:** `ai-and/termgrade-trajectories` @7d584d63 + `ai-and/termgrade-environments` @3a34ba29
  (Apache-2.0). These are Terminus-2 (Harbor) episodes on 1,004 executable tasks. Each trial is graded by the
  task's pytest verifier offline in a fresh container. The tasks were generated by DeepSeek-V4-Pro and kept only
  if the reference solution passes its own verifier.
- **Teachers:** DeepSeek-V4-Pro (MIT), GLM-5.2 (MIT), Qwen3.6-27B (Apache-2.0), Gemma-4-31B-it thinking
  (Apache-2.0). Kimi-K3 is excluded (license "other").
- **Filters:**
  - Only `reward == 1` trials that count toward the reported k, are scored and are not capped.
  - Hard tasks: some config solves it and the pooled pass rate over the 6 configs is <= 0.5 (tier 1). Tier 2
    (<= 0.67) is used only to fill the budget.
  - At most 2 trajectories per task, with distinct teachers and longer ones preferred.
  - At least 3 assistant turns, at most 64K tokens.
  - Trajectories with parse-error feedback, summarisation, or non-JSON replies are dropped.
- **Decontamination:**
  - TB1-unsafe/TB2 task names.
  - TB2 13-gram on instruction + reference solution (vs TB2 instruction/solution/tests, k >= 3): 2 tasks dropped.
  - 13-gram vs the TBLite dev set and the E-TB val instructions: 0 dropped.
- **Rendering:** identical to E-TB. TermGrade stores the Terminus-2 prompt as system (template prefix, an older
  revision) + user (`Task Description: …`). Instruction and initial screen are re-rendered with the locked
  Harbor template, which also gives the exact completion-confirmation strings. Reasoning stays in
  `reasoning_content`.
- **Result (2026-10-14):**
  - train: 486 trajectories / 282 tasks / 15.06M tokens (E-TB: 15.02M). val: 20 held-out tasks / 0.61M tokens.
  - Teachers (trajectories): Qwen3.6-27B 278, Gemma-4 thinking 117, GLM-5.2 64, DeepSeek-V4-Pro 27. Most
    DeepSeek/GLM traces are longer than 64K or wrap the JSON in extra text.
  - Pooled pass rate of the train tasks: median 0.28. Assistant turns: median 7, p90 13.
  - `verify --experts tb_v2`: 0 failures.
- **Known caveats:**
  - Horizons are still shorter than TB2's.
  - Python-heavy tasks (by a keyword heuristic): python 218, shell/text 115.
  - The thinking train/eval mismatch of all experts (training keeps reasoning in every assistant turn, Terminus-2
    at eval does not replay earlier turns' reasoning) is **fixed for E-TB-v2**; see the per-turn set below. The
    full-trajectory `tb_v2` set above still has it and is not used for training.
- **Per-turn set `tb_v2_turns` (what E-TB-v2 trains on), `build --per-turn`:**
  - Every trajectory becomes samples k = messages up to assistant turn k. Earlier assistant turns keep only
    their visible JSON (no `reasoning_content`); turn k keeps its reasoning.
  - Records carry `"render": {"preserve_thinking": false}` and `"loss": "last_assistant"`, honoured by
    `src/train/data.py` (records without these fields are unchanged). The loss covers exactly the tokens
    generated at eval: reasoning + `\n</think>\n\n` + JSON + `<|im_end|>`. The generation prompt
    `<|im_start|>assistant\n<think>\n` is context.
  - `check-turns` verifies on sampled train and val records that the training text before the target equals
    `apply_chat_template(prefix, add_generation_prompt=True, enable_thinking=True, preserve_thinking=False)`
    (the eval server's kwargs). The check is byte- and token-exact, and the mask is exactly the generated span.
  - Selection: <= 2 trajectories per task (all 471 non-val tasks with a verified pass, tier 1 first). One stably
    shuffled turn per trajectory per round, until the context + target token budget (~15.02M, same compute as
    E-TB) is reached. Val: <= 3 turns from each of the 20 held-out tasks, as a 5th val set. The 4 standard val
    sets are unchanged.
  - Counts, label tokens and step counts: `manifests/tb_v2_turns_stats.json` and the token-cache log.
