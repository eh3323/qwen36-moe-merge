# 模型：链接、合版配方与加载方法

所有权重都在 Hugging Face 上（提交时设为公开，下载无需授权）。基座直接引用官方仓库。

| 模型 | 仓库 | revision | 内容 |
|---|---|---|---|
| Base | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | `995ad96e` | 官方基座，未改动 |
| **最终合版 v4** | [eh3323/Qwen3.6-35B-A3B-apex-merged-v4](https://huggingface.co/eh3323/Qwen3.6-35B-A3B-apex-merged-v4) | `223a1b15` | 完整权重（bf16，26 个分片）+ `merge_info.json` |
| 专家与对照（LoRA） | [eh3323/Qwen3.6-35B-A3B-apex-lora-experts](https://huggingface.co/eh3323/Qwen3.6-35B-A3B-apex-lora-experts) | `e08bd952` | 7 个 adapter，子目录见下 |
| E-ASI（折叠） | [eh3323/Qwen3.6-35B-A3B-apex-e_asi-folded](https://huggingface.co/eh3323/Qwen3.6-35B-A3B-apex-e_asi-folded) | `3476649b` | 基座 + E-ASI 的完整权重 |
| Mixed（折叠） | [eh3323/Qwen3.6-35B-A3B-apex-mixed-folded](https://huggingface.co/eh3323/Qwen3.6-35B-A3B-apex-mixed-folded) | `816e1ec6` | 基座 + Mixed 的完整权重 |
| E-TB v2（折叠） | [eh3323/Qwen3.6-35B-A3B-apex-e_tb_v2-folded](https://huggingface.co/eh3323/Qwen3.6-35B-A3B-apex-e_tb_v2-folded) | `9281bb8c` | 基座 + E-TB v2 的完整权重 |

- LoRA 仓库的子目录：`e_swe_v2/`、`e_tb_v2/`、`e_asi/`、`e_hle/`（进入 v4）、`mixed/`（对照）、`e_swe_v1/`、`e_tb_v1/`（历史版本）。
- “折叠”指把单个 adapter 用 `merge_and_unload()` 加回基座得到的稠密权重（bf16），租用节点上评测时直接加载。为了和多专家的“合版”区分，这里不叫 merged。

许可证：Apache-2.0，继承自基座。

## 版本、量化与 run_id

- **基座**：`Qwen/Qwen3.6-35B-A3B` @ `995ad96eacd98c81ed38be0c5b274b04031597b0`。tokenizer、chat template（`chat_template.jinja`）和 generation config 一律取基座目录，所有发布的模型都没有改动它们。
- **量化**：无。训练、合版、评测都用 bf16；LoRA 增量在 fp32 中计算后加回基座，再转为 bf16 保存。
- **推理设置**：vLLM 0.30.0，`enable_thinking=true`、`preserve_thinking=false`，详见主 README §6。

| 发布的模型 | 训练 run_id | 评测 run id 前缀 |
|---|---|---|
| merged-v4 | 合版 `ties_l1_d0.5_v4`（`configs/merge/ties_v4.yaml`） | `merged_v4`、`merged_v4_cloud` |
| `e_swe_v2/` | `e_swe_v2_r32` | `e_swe_v2` |
| `e_tb_v2/`、e_tb_v2-folded | `e_tb_v2_r32` | `e_tb_v2_cloud` |
| `e_asi/`、e_asi-folded | `e_asi_r32` | `e_asi` |
| `e_hle/` | `e_hle_r32` | `e_hle` |
| `mixed/`、mixed-folded | `mixed_r32` | `mixed`、`mixed_cloud` |
| `e_swe_v1/` | `e_swe_r32` | `e_swe` |
| `e_tb_v1/` | `e_tb_r32` | `e_tb` |

训练记录在 `logs/train/<训练 run_id>/`，逐任务结果在 `results/<group>/<bench>/<评测 run id>__<bench>__<tag>.jsonl`。

## LoRA 专家

- 秩 r=32，alpha=64，dropout 0.05。
- 每层都挂了 LoRA 的模块：全注意力的 q/k/v/o、线性注意力（Gated DeltaNet）的 in_proj_qkv/in_proj_z/out_proj，以及 shared expert MLP 的 gate/up/down。routed experts 和 router 不动。
- 训练配置在 `configs/train/<expert>.yaml`，每次运行的配置快照、日志和资源记录在 `logs/train/<run_id>/`。
- 每个子目录里的 `state.json` 记录最终步数和训练过的 token 数。

## 合版配方（v4，`configs/merge/ties_v4.yaml` → `ties_v3.yaml` → `ties.yaml`）

- **方法**：TIES。先把每个专家的 LoRA 增量 ΔW = (alpha/r)·B·A 展开到稠密空间；每个模块只保留幅值最大的 50%（density 0.5）；按元素做符号投票，只对与多数符号相同的专家取平均；最后乘 λ=1 加回基座。
- **专家**：E-SWE v2、E-TB v2、E-ASI、E-HLE，权重都是 1.0。
- **超参来源**：从 18 个 Linear/TIES/DARE 候选中，按独立开发集的 val loss 选出（`configs/merge/select.yaml`，结果见 `curves/merge_ablation.png`）。
- **完整配置**：实际使用的完整配置（专家路径、缩放、模块数）写在合版仓库的 `merge_info.json` 里。

本地复现合版：先下载 LoRA 仓库，把 `configs/merge/ties_v3.yaml` 和 `ties_v4.yaml` 里的专家路径改成本地下载的路径，然后执行：

```bash
python -m src.merge.merge_experts --config configs/merge/ties_v4.yaml   # 输出 models/merged/ties_l1_d0.5_v4
```

## 加载

合版模型（与基座用法相同；tokenizer、chat template、generation config 都来自基座）：

```bash
vllm serve eh3323/Qwen3.6-35B-A3B-apex-merged-v4 --revision 223a1b15 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --max-model-len 262144
```

单个专家（PEFT）：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-35B-A3B", revision="995ad96e", torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "eh3323/Qwen3.6-35B-A3B-apex-lora-experts", subfolder="e_swe_v2", revision="e08bd952")
model = model.merge_and_unload()   # 可选：折叠成稠密权重（评测时各专家就是这样折叠后用 vLLM 加载的）
```
