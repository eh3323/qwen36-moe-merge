#!/usr/bin/env python3
"""Upload the LoRA experts and the final merged model to the Hugging Face Hub.

    source /scratch/eh3323/apex/env.sh   # HF_HOME with the login token
    $APEX/envs/train/bin/python scripts/hf_upload.py experts   # ~1 GB, minutes
    $APEX/envs/train/bin/python scripts/hf_upload.py merged    # ~67 GB, one commit per 4 shards; re-run to resume
    $APEX/envs/train/bin/python scripts/hf_upload.py public    # flip both repos to public (at submission)
    $APEX/envs/train/bin/python scripts/hf_upload.py folded e_asi_r32 mixed_r32   # folded (dense) experts, public
    $APEX/envs/train/bin/python scripts/hf_upload.py folded_cards e_asi_r32 mixed_r32 e_tb_v2_r32   # their model cards
    $APEX/envs/train/bin/python scripts/hf_upload.py cards     # refresh the LoRA / merged model cards only

Repos are created PRIVATE; `public` is a separate, explicit step. Files are staged under
$APEX/work/hf_stage (symlinks to the weights, so nothing large is copied): adapter_config.json gets
base_model_name_or_path = the Hub id of the base instead of the local /scratch path, and each repo gets
its own model card instead of the auto-generated PEFT / base-model README.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

APEX = Path(os.environ.get("APEX", "/scratch/eh3323/apex"))
USER = "eh3323"
BASE_ID = "Qwen/Qwen3.6-35B-A3B"
BASE_REV = "995ad96e"
GITHUB = "https://github.com/eh3323/qwen36-moe-merge"
EXPERTS_REPO = f"{USER}/Qwen3.6-35B-A3B-apex-lora-experts"
MERGED_REPO = f"{USER}/Qwen3.6-35B-A3B-apex-merged-v4"
MERGED_DIR = APEX / "models/merged/ties_l1_d0.5_v4"
STAGE = APEX / "work/hf_stage"

# subfolder -> (run dir, description)
EXPERTS = {
    "e_swe_v2": ("e_swe_v2_r32", "SWE expert v2 (in the final merge)"),
    "e_tb_v2": ("e_tb_v2_r32", "Terminal-Bench expert v2, thinking-format fix (in the final merge)"),
    "e_asi": ("e_asi_r32", "ASI-Bench expert (in the final merge)"),
    "e_hle": ("e_hle_r32", "HLE expert (in the final merge)"),
    "mixed": ("mixed_r32", "single LoRA on the union of the expert data (control)"),
    "e_swe_v1": ("e_swe_r32", "SWE expert v1 (historical, not in the final merge)"),
    "e_tb_v1": ("e_tb_r32", "Terminal-Bench expert v1 (historical, excluded from the merge after a TB2 regression)"),
}

TRAINING = f"""## Training

- Method: LoRA (no quantization, bf16 base), rank 32, alpha 64, dropout 0.05. 250 target modules (38.3M parameters):
  q/k/v/o of the 10 full-attention layers, in_proj_qkv/in_proj_z/out_proj of the 30 Gated DeltaNet layers and the
  shared-expert gate/up/down projections of all 40 layers. The 256 routed experts, the router, the vision tower and
  the MTP head are not trained.
- About 15M training tokens per expert (token-matched, not sample-matched), 2 epochs, 262,144 tokens per optimizer step,
  peak learning rate 1e-4 (3% warmup, cosine to 1e-5), AdamW, seed 1234, 32K or 64K packing, one H200 or B200 per run.
  The final adapter is the last step (`adapter_final`); no checkpoint was selected on any benchmark.
- Data (all self-built from permissively licensed data with open-weight teachers; benchmark tasks, answers and tests
  were excluded; decontamination by 13-gram overlap, plus embedding similarity for ASI / HLE):

| expert | data | licence |
|---|---|---|
| SWE v2 | nvidia/Open-SWE-Traces (mini-swe-agent trajectories, Go / TS / JS / Python) | CC-BY-4.0 |
| Terminal-Bench v2 | ai-and/termgrade-trajectories, verifier-passed Terminus-2 trajectories, one sample per turn | Apache-2.0 |
| ASI-Bench | nvidia/Nemotron-Terminal-Corpus scientific-computing subset + jupyter-agent | CC-BY-4.0 / Apache-2.0 |
| HLE | nvidia/Nemotron-Math-v2 (AoPS), Nemotron-Science-v1, OpenScienceReasoning-2 | CC-BY-4.0 |
| mixed (control) | union of the v1 SWE / TB data and the ASI / HLE data | as above |

Full data description, revisions and per-run training logs: {GITHUB} (`data/README.md`, `logs/train/`).
"""

RESULTS = """## Evaluation (this experiment's settings)

Same vLLM 0.30.0 server settings for every model (thinking on, bf16, 256K context); environment failures count as 0.
Percent scores; Δ is the paired-bootstrap 95% CI against the measured base.

| benchmark | Base | expert on its own benchmark | Mixed | merged v4 | v4 − Base |
|---|---|---|---|---|---|
| SWE-bench Pro HARD-51 (resolved, 3 runs) | 21.57 | 27.45 (SWE v2) | 13.73 | 16.34 | −5.23 [−12.42, +1.31] |
| Terminal-Bench 2.0 (Harbor + Terminus-2) | 44.94 | 38.20 (TB v2) | 30.34 | 41.57 | −3.37 [−8.99, +2.25] |
| HLE (2500, multimodal, no tools) | 20.80 | 16.92 (HLE) | 16.60 | 19.21 | −1.60 [−3.16, −0.08] |
| ASI-Bench seed31415 B3 (local scoring) | 10.03 | 7.75 (ASI) | 7.39 | 7.53 | −2.50 [−6.79, +2.52] |
| macro average | 24.33 | — | 17.02 | 21.16 | −3.18 [−5.75, −0.67] |

## Limitations

- The merged model does not beat the base (macro −3.2 points); it beats the mixed-data control by +4.2 points.
- Merge-induced regressions seen in the failure analysis: committing before submitting (empty SWE patches), longer
  trajectories and more timeouts on Terminal-Bench, more truncated thinking on HLE, and most of the HLE expert's
  calibration gain lost.
- Few repeats: the Terminal-Bench numbers above are same-node runs (base 2, the others 1; 4 and 2 runs for base and v4
  including cluster runs) instead of the 5 the protocol asks for; experts and the control were run once.
- ASI-Bench scores are local (seed31415) scores, not official ones. Dropping the first SWE / TB experts and retraining
  them happened after looking at test results, so the SWE and TB deltas may be optimistic.
- Research artefact for a take-home study, not a production model.
"""

EXPERTS_CARD = f"""---
base_model: {BASE_ID}
library_name: peft
license: apache-2.0
tags: [lora, qwen3.6, moe, model-merging]
---

# Qwen3.6-35B-A3B LoRA experts (Apex take-home)

LoRA adapters trained on `{BASE_ID}` (revision `{BASE_REV}`), one per benchmark family, plus the mixed-data control.
The final merged model (TIES of `e_swe_v2`, `e_tb_v2`, `e_asi`, `e_hle`) is `{MERGED_REPO}`.
Code, configs, logs and per-task results: {GITHUB}

| subfolder | description | training run_id |
|---|---|---|
""" + "\n".join(f"| `{k}/` | {d} | `{r}` |" for k, (r, d) in EXPERTS.items()) + f"""

Each subfolder holds `adapter_config.json`, `adapter_model.safetensors` (about 153 MB) and `state.json` (final optimizer
step and training tokens seen).

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("{BASE_ID}", revision="{BASE_REV}", torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "{EXPERTS_REPO}", subfolder="e_swe_v2")
model = model.merge_and_unload()   # optional: dense weights, as served with vLLM in the evaluation
```

Tokenizer, chat template and generation config are the base model's.

{TRAINING}
{RESULTS}
## License

Apache-2.0, inherited from the base model. Training data licences: CC-BY-4.0 and Apache-2.0 (see the table above).
"""

MERGED_CARD = f"""---
base_model: {BASE_ID}
license: apache-2.0
tags: [qwen3.6, moe, model-merging, ties, lora]
---

# Qwen3.6-35B-A3B Apex merged model (v4, TIES)

`{BASE_ID}` (revision `{BASE_REV}`) with four LoRA experts merged into the weights: SWE v2, Terminal-Bench v2,
ASI-Bench and HLE experts from `{EXPERTS_REPO}`. Code, configs, logs and per-task results: {GITHUB}

## Merge recipe

- Each expert's update is expanded to a dense matrix, τ = (alpha / r) · B · A (fp32), for each of the 250 adapted
  modules; the routed experts and the router are identical to the base.
- TIES: keep the top 50% of each expert's entries by magnitude per module (density 0.5), elect the sign per entry by
  the weighted sum, average only the experts that agree with the elected sign (weights 1.0 each), scale by λ = 1.
- W' = W + τ in fp32, saved as bf16. `merge_info.json` in this repo holds the exact configuration
  (`configs/merge/ties_v4.yaml` in the GitHub repo).
- The hyper-parameters were chosen among 12 Linear / TIES / DARE configurations by validation loss on independent
  dev sets (no benchmark data).

## Usage

```bash
vllm serve {MERGED_REPO} --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --max-model-len 262144
```

or `AutoModelForCausalLM.from_pretrained("{MERGED_REPO}", torch_dtype="bfloat16")`. Tokenizer, chat template and
generation config are the base model's. No quantization.

{TRAINING}
{RESULTS}
## License

Apache-2.0, inherited from the base model (see `LICENSE`). Training data licences: CC-BY-4.0 and Apache-2.0.
"""

FOLDED_CARD = """---
base_model: {base}
license: apache-2.0
tags: [qwen3.6, moe, lora, folded]
---

# Qwen3.6-35B-A3B + `{sub}` LoRA, folded (Apex take-home)

Dense weights: `{base}` (revision `{rev}`) with the `{sub}/` adapter from `{experts}` folded into the
weights ({desc}; W + scale * B A, bf16, no quantization). This is exactly the checkpoint that was served with vLLM in
the evaluation; it is published so that rented evaluation nodes could download it, and is equivalent to loading the
base plus the adapter and calling `merge_and_unload()`. Tokenizer, chat template and generation config are the base
model's. Training run_id: `{run}`.

Result in this experiment: {result}. Training data, settings, limitations and the full results table are in the
adapter repo's card (`{experts}`) and in {github}.

License: Apache-2.0, inherited from the base model.
"""

FOLDED_RESULTS = {
    "e_asi": "ASI-Bench seed31415 B3 (local scoring) 7.75 vs base 10.03 (Δ −2.29, 95% CI [−7.33, +3.67])",
    "mixed": "SWE-bench Pro HARD-51 13.73, Terminal-Bench 2.0 30.34, HLE 16.60, ASI-Bench B3 7.39; macro 17.02 vs base 24.33",
    "e_tb_v2": "Terminal-Bench 2.0 38.20 vs base 44.94 (Δ −6.74, 95% CI [−14.61, +0.56])",
}


def stage_experts() -> Path:
    root = STAGE / "experts"
    shutil.rmtree(root, ignore_errors=True)
    for sub, (run, _) in EXPERTS.items():
        src, dst = APEX / "runs" / run / "adapter_final", root / sub
        dst.mkdir(parents=True)
        cfg = json.loads((src / "adapter_config.json").read_text())
        cfg["base_model_name_or_path"] = BASE_ID
        (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        os.symlink(src / "adapter_model.safetensors", dst / "adapter_model.safetensors")
        if (src / "state.json").exists():
            shutil.copy(src / "state.json", dst / "state.json")
    (root / "README.md").write_text(EXPERTS_CARD)
    return root


def stage_merged() -> Path:
    root = STAGE / "merged_v4"
    root.mkdir(parents=True, exist_ok=True)
    assert (MERGED_DIR / "merge_info.json").exists(), "merged model incomplete"
    for f in MERGED_DIR.iterdir():
        if f.is_file() and f.name != "README.md" and not (root / f.name).exists():
            os.symlink(f, root / f.name)
    (root / "README.md").write_text(MERGED_CARD)
    return root


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    api = HfApi()
    if what == "experts":
        root = stage_experts()
        api.create_repo(EXPERTS_REPO, private=True, exist_ok=True)
        info = api.upload_folder(repo_id=EXPERTS_REPO, folder_path=root, commit_message="LoRA experts + control")
        print("uploaded:", info)
    elif what == "merged":
        root = stage_merged()
        api.create_repo(MERGED_REPO, private=True, exist_ok=True)
        # one commit per 4 weight shards (each commit is durable; files already on the Hub are skipped on a re-run),
        # small files (config, index, tokenizer, card) last, so the repo only looks complete when it is
        done = set(api.list_repo_files(MERGED_REPO))
        shards = sorted(f.name for f in root.glob("model-*.safetensors"))
        for i in range(0, len(shards), 4):
            todo = [f for f in shards[i:i + 4] if f not in done]
            if todo:
                api.upload_folder(repo_id=MERGED_REPO, folder_path=root, allow_patterns=todo,
                                  commit_message=f"weights {todo[0]} .. {todo[-1]}")
                print("uploaded", todo, flush=True)
        api.upload_folder(repo_id=MERGED_REPO, folder_path=root, ignore_patterns=["model-*.safetensors", ".cache/*"],
                          commit_message="config, tokenizer, merge_info, model card")
        print("uploaded:", MERGED_REPO)
    elif what == "folded":  # dense experts (LoRA folded into the base), as evaluated; needed on rented eval nodes
        for name in sys.argv[2:]:
            src = APEX / "models/merged" / name
            assert (src / "config.json").exists(), src
            repo = f"{USER}/Qwen3.6-35B-A3B-apex-{name.replace('_r32', '')}-folded"
            api.create_repo(repo, private=False, exist_ok=True)
            done = set(api.list_repo_files(repo))
            shards = sorted(f.name for f in src.glob("model-*.safetensors"))
            for i in range(0, len(shards), 4):
                todo = [f for f in shards[i:i + 4] if f not in done]
                if todo:
                    api.upload_folder(repo_id=repo, folder_path=src, allow_patterns=todo,
                                      commit_message=f"weights {todo[0]} .. {todo[-1]}")
            api.upload_folder(repo_id=repo, folder_path=src, ignore_patterns=["model-*.safetensors", "README.md"],
                              commit_message="config, tokenizer")
            print("uploaded:", repo, flush=True)
        return
    elif what == "folded_cards":  # model card for each folded repo (the weights stay as uploaded)
        sub_of = {v[0]: k for k, v in EXPERTS.items()}
        for name in sys.argv[2:]:
            sub = sub_of[name]
            repo = f"{USER}/Qwen3.6-35B-A3B-apex-{name.replace('_r32', '')}-folded"
            card = FOLDED_CARD.format(base=BASE_ID, rev=BASE_REV, sub=sub, experts=EXPERTS_REPO, run=name,
                                      desc=EXPERTS[sub][1], github=GITHUB, result=FOLDED_RESULTS[sub])
            api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo,
                            commit_message="model card")
            print("card:", repo, "main @", next(b.target_commit for b in api.list_repo_refs(repo).branches
                                                if b.name == "main"), flush=True)
        return
    elif what == "cards":  # refresh only the model cards of the LoRA and merged repos (weights unchanged)
        for repo, card in ((EXPERTS_REPO, EXPERTS_CARD), (MERGED_REPO, MERGED_CARD)):
            api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo,
                            commit_message="model card: training, data, results, limitations")
            print("card:", repo, "main @", next(b.target_commit for b in api.list_repo_refs(repo).branches
                                                if b.name == "main"), flush=True)
        return
    elif what == "public":
        for r in (EXPERTS_REPO, MERGED_REPO):
            api.update_repo_settings(r, private=False)
            print("public:", r)
    else:
        sys.exit(__doc__)
    for r in (EXPERTS_REPO, MERGED_REPO):
        try:
            refs = api.list_repo_refs(r)
            print(r, "main @", next(b.target_commit for b in refs.branches if b.name == "main"))
        except Exception as e:
            print(r, "-", type(e).__name__)


if __name__ == "__main__":
    main()
