#!/usr/bin/env python3
"""Upload the LoRA experts and the final merged model to the Hugging Face Hub.

    source /scratch/eh3323/apex/env.sh   # HF_HOME with the login token
    $APEX/envs/train/bin/python scripts/hf_upload.py experts   # ~1 GB, minutes
    $APEX/envs/train/bin/python scripts/hf_upload.py merged    # ~67 GB, one commit per 4 shards; re-run to resume
    $APEX/envs/train/bin/python scripts/hf_upload.py public    # flip both repos to public (at submission)
    $APEX/envs/train/bin/python scripts/hf_upload.py folded e_asi_r32 mixed_r32   # folded (dense) experts, public
    $APEX/envs/train/bin/python scripts/hf_upload.py folded_cards e_asi_r32 mixed_r32 e_tb_v2_r32   # their model cards

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

EXPERTS_CARD = f"""---
base_model: {BASE_ID}
library_name: peft
license: apache-2.0
tags: [lora, qwen3.6, moe, model-merging]
---

# Qwen3.6-35B-A3B LoRA experts (Apex take-home)

LoRA adapters (rank 32, alpha 64, dropout 0.05; full-attention q/k/v/o, linear-attention in_proj_qkv/in_proj_z/out_proj and
shared-expert MLP projections in every layer; routed experts untouched) trained on `{BASE_ID}` (revision `{BASE_REV}`),
one per benchmark family, plus the mixed-data control. Training data, configs, evaluation and the merge are
documented in the GitHub repo: {GITHUB}

| subfolder | description |
|---|---|
""" + "\n".join(f"| `{k}/` | {d} |" for k, (_, d) in EXPERTS.items()) + f"""

The final merged model (TIES of `e_swe_v2`, `e_tb_v2`, `e_asi`, `e_hle`) is `{MERGED_REPO}`.
`state.json` records the final optimizer step and the number of training tokens seen.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("{BASE_ID}", torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "{EXPERTS_REPO}", subfolder="e_swe_v2")
```
"""

MERGED_CARD = f"""---
base_model: {BASE_ID}
license: apache-2.0
tags: [qwen3.6, moe, model-merging, ties, lora]
---

# Qwen3.6-35B-A3B Apex merged model (v4, TIES)

`{BASE_ID}` (revision `{BASE_REV}`) with four LoRA experts merged into the weights by TIES
(magnitude sparsification, density 0.5, sign election, lambda 1): SWE v2, Terminal-Bench v2, ASI-Bench and HLE
experts from `{EXPERTS_REPO}`. `merge_info.json` holds the exact merge configuration. Tokenizer, chat template
and generation config are the base model's. Evaluation (SWE-bench Pro HARD-51, HLE, ASI-Bench, Terminal-Bench 2.0)
against the base, each expert and a mixed-data control, with all settings and caveats: {GITHUB}

Serve exactly like the base model, e.g. `vllm serve {MERGED_REPO} --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`.

License: Apache-2.0, inherited from the base model (see `LICENSE`).
"""

FOLDED_CARD = """---
base_model: {base}
license: apache-2.0
tags: [qwen3.6, moe, lora, folded]
---

# Qwen3.6-35B-A3B + `{sub}` LoRA, folded (Apex take-home)

Dense weights: `{base}` (revision `{rev}`) with the `{sub}/` adapter from `{experts}` folded into the
weights ({desc}; W + scale * B A, bf16). This is exactly the checkpoint that was served with vLLM in the evaluation; it is
published so that rented evaluation nodes could download it, and is equivalent to loading the base plus the adapter
and calling `merge_and_unload()`. Tokenizer, chat template and generation config are the base model's.

Training, evaluation settings and results: {github}

License: Apache-2.0, inherited from the base model.
"""


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
            card = FOLDED_CARD.format(base=BASE_ID, rev=BASE_REV, sub=sub, experts=EXPERTS_REPO,
                                      desc=EXPERTS[sub][1], github=GITHUB)
            api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo,
                            commit_message="model card")
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
