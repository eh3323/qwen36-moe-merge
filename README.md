# Qwen3.6-35B-A3B 多专家 LoRA 训练与模型合版

> 状态：🚧 实验进行中。凡标注“待填写”的部分会在实验完成后用真实结果替换；所有数字都可以由 `src/eval/summarize.py` 从原始日志重新生成。

## 0. 速览

| 项目 | 内容 |
|---|---|
| 基座 | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) @ `995ad96eacd98c81ed38be0c5b274b04031597b0`（MoE：总参数约 35B，每次激活约 3B；Apache-2.0）。未降级到 7B，理由见 §3 |
| 专家 | 4 个 LoRA 专家，每个对应一个 benchmark：E-SWE / E-TB / E-ASI / E-HLE |
| 合版 | 在稠密 ΔW 空间合并各专家的 LoRA：Linear（任务算术）/ TIES / DARE，并对合并系数做消融 |
| 主对照 | **多专家合版 vs 单模型混训**：用 4 个专家训练数据的并集（样本 ID 完全相同）、相同超参训练一个 LoRA |
| 评测 | SWE-bench Pro V2 HARD-51、HLE（2500 题，完整多模态）、ASI-Bench（seed31415）、Terminal-Bench 2.0（Harbor + Terminus-2，5 次运行） |
| 硬件 | NYU Torch 集群，NVIDIA H200 141GB（部分作业用 RTX Pro 6000 96GB），每个训练作业单卡 |
| 模型权重 | Hugging Face 链接见 §8（待填写） |

## 1. 研究问题

1. **专家划分**：按 benchmark 划分 4 个 LoRA 专家，每个专家只用约 1500 万 token 的目标域数据，能否在各自的 benchmark 上相对 Base 取得提升？
2. **合版**：把 4 个专家合成一个模型后，能保住多少专家增益？参数冲突（符号冲突、方向不一致）分布在哪些模块，不同合版方法（Linear / TIES / DARE）怎样处理这些冲突？
3. **合版的价值**：在训练 token 数和超参完全相同的前提下，“分别训专家再合版”和“直接用数据并集训一个模型”谁更好？

## 2. 实验矩阵

| 模型 | 训练数据 | 评测范围 |
|---|---|---|
| Base | — | 四项全量（TB2 跑 5 次） |
| E-SWE | SWE 数据，约 1500 万 token | SWE-bench Pro HARD-51 |
| E-TB | TB 数据，约 1500 万 token | Terminal-Bench 2.0（2 次；见 §6） |
| E-ASI | ASI 数据，约 1500 万 token | ASI-Bench |
| E-HLE | HLE 数据，最小规模（约 2000 条） | HLE |
| Mixed（对照） | 上面四份数据的并集，约 6000 万 token | 四项（TB2 跑 2 次） |
| Merged（最终） | 4 个专家的 LoRA 合版 | 四项全量（TB2 跑 5 次） |
| 合版消融 | Linear / TIES / DARE × 系数 λ | 在独立开发集上比较（不用 benchmark 选模型） |

LoRA 超参（所有专家和 Mixed 完全相同，`configs/train/base.yaml`）：r=32，α=64，dropout 0.05；学习率 1e-4，cosine 衰减到 10%，3% warmup；每个优化步 262,144 token；2 个 epoch；种子 1234；bf16。

LoRA 的目标模块共 250 个，约 3830 万参数：
- 10 个全注意力层的 `q/k/v/o_proj`；
- 30 个 Gated DeltaNet 层的 `in_proj_qkv/in_proj_z/out_proj`；
- 40 层的 `shared_expert.{gate,up,down}_proj`。

**不训练**的部分：256 个 routed experts、router、视觉塔、MTP 头。理由：adapter 更小，合版冲突只出现在共享模块上，便于解释；同时也避免让少量数据去扰动专家路由。

## 3. 基座选择与算力

- 选择 35B-A3B，没有降到 7B。原因：
  - 每个 token 只激活约 3B 参数，推理速度和 3B 左右的稠密模型相当；
  - bf16 权重 67GB，单张 H200 就放得下；
  - 实测单卡 LoRA 训练吞吐约 8.7k token/s（32K 序列，显存峰值 83GB；64K 序列 8.3k token/s，96GB），训一个专家 2 个 epoch 约 1 GPU 小时；
  - 评测的瓶颈在 Agent 沙箱的实际执行时间，换成 7B 并不会显著变快。
- 资源和时间的记录见 `logs/` 与 §9（待填写）。

## 4. Benchmark 版本与 split（训练前锁定，`configs/eval/benchmarks.yaml`）

| Benchmark | 来源与版本 | Split / 任务 | 主指标 | 说明 |
|---|---|---|---|---|
| SWE-bench Pro v2-hard | `ScaleAI/SWE-bench_Pro` @2d52cb3d（config `hard`）；repo `scaleapi/SWE-bench_Pro-os` @66f92766 | HARD-51（`v2/hard51_ids.txt`） | resolved rate（官方 patch_replay 重新判分） | mini-swe-agent 2.4.6 + `mini_toolcall.yaml`，Agent 运行时断网（apptainer `--network none` 补丁） |
| Terminal-Bench 2.0 | `laude-institute/terminal-bench-2` @2fd12b88（含 TB2.1 的任务修复）；Harbor @4778ceba | 89 个任务 | 成功率，5 次运行的均值 ± 标准差 | Terminus-2；3 小时超时、32 CPU / 48GB（名义值，见 §7）、temperature=1.0、top_p=0.95、top_k=20、max_tokens=80K、256K 上下文 |
| HLE | `cais/hle` @5a81a4c7 | test，2500 题（含 342 道图像题） | 准确率（无答案计错）+ 校准误差 | 官方 prompt，不用工具；判分模型 o3-mini-2025-01-31（官方脚本） |
| ASI-Bench | `apexin-ai/ASI-Bench` @8859f9b；`ASI-Bench-seed31415` @0fa1421 | 60 个任务 × B1–B4 | B3 得分（0–100），同时报告 B1–B4 和 Overall | opencode 1.17.15（官方 adapter）；seed42 只能在官网评分，所以用 seed31415 本地评分（**非官方分数**）；参考答案对 Agent 隐藏 |

所有被评测的模型都使用相同的 thinking 模式（开启；`preserve_thinking` 用 chat template 的默认值）、bf16、相同的工具、上下文和预算。

## 5. 训练数据（详见 `data/README.md`）

| 专家 | 来源（许可证 / 教师模型） | 条数 / token | 格式 |
|---|---|---|---|
| E-SWE | nebius/SWE-rebench-openhands-trajectories（只取 resolved）+ nvidia/SWE-Hero（CC-BY-4.0 / Qwen3-Coder-480B） | 432 / 15.05M | mini_toolcall（单个 bash 工具） |
| E-TB | nvidia/Nemotron-Terminal-Corpus（CC-BY-4.0 / DeepSeek-V3.2）+ OpenThoughts-Agent-SFT-100K（Apache-2.0 / GLM-4.7） | 973 / 15.02M | Terminus-2 JSON |
| E-ASI | Nemotron-Terminal 的科学计算、数据科学、模型训练子集 + jupyter-agent（Apache-2.0 / Qwen3-Coder） | 868 / 14.93M | opencode 的 bash 工具调用 |
| E-HLE | Nemotron-Math-v2 AoPS + Nemotron-Science-v1 + OpenScienceReasoning-2（CC-BY-4.0 / gpt-oss-120b、DeepSeek-R1） | 待填写 | HLE 官方答案格式 |

- **只用开源权重的教师模型**。用闭源 API 生成的轨迹、非商用许可证的数据、所有 HLE 衍生数据集都已排除。
- **去污染**：
  - SWE：剔除 11 个 SWE-Pro 仓库及整个 `ansible/*` 组织，fork 规则，13-gram 查重；
  - TB：剔除与 TB2 重合的 90 个 TB1 任务，13-gram 查重；
  - ASI：方法名黑名单、13-gram、embedding 相似度 ≥ 0.85；
  - HLE：canary、精确匹配、13-gram、embedding 相似度 ≥ 0.9。
- **benchmark 的题目、答案、参考补丁、隐藏测试一律不用于训练和选模型**。调参和选合版方法只用独立的开发集：SWE-rebench-V2 留出的仓库、OpenThoughts-TBLite、ScienceAgentBench、GPQA-Diamond 和留出题。

## 6. 结果（待填写）

| Benchmark | Base | E-SWE | E-TB | E-ASI | E-HLE | Mixed | Merged | Merged − Base（95% CI） |
|---|---|---|---|---|---|---|---|---|
| SWE-bench Pro HARD-51 | | | — | — | — | | | |
| Terminal-Bench 2.0 | | — | | — | — | | | |
| HLE | | — | — | — | | | | |
| ASI-Bench（B3） | | — | — | | — | | | |
| **宏平均** | | | | | | | | |

置信区间：对同一批任务做配对 bootstrap，10,000 次重采样。TB2 另外报告各次运行之间的标准差。运行失败率、成本和逐任务结果见 `results/` 和 `analysis/`。

## 7. 已知局限与偏离官方设置（持续补充）

1. **TB2 版本**：用的是锁定的本地快照 @2fd12b88，含 TB2.1 的任务修复。和 registry 中 `terminal-bench@2.0`（69671fb）相比，有 4 个 task.toml 及少量 tests/solution 不同。
2. **TB2 的资源配置**：32 CPU / 48GB 是名义设置。singularity 不能限制 CPU，内存只靠一个 95% 的监控进程来控制。多个任务共享一个节点，CPU 不是独占的。
3. **没有 Docker**：集群上只有 apptainer。我们给 Harbor 的 singularity 后端打了补丁（`patches/`），支持断网和端口隔离。SWE-Pro 的 Agent 阶段完全断网，判分阶段按协议联网。
4. **ASI-Bench**：用 seed31415 在本地评分，不是官方分数。我们修补了 linux_ns 沙箱，防止 Agent 读到参考答案和其他档位的工作目录（`patches/asibench_linux_ns_hide.patch`）。
5. **thinking 在训练和推理时不一致**：训练时每一轮助手回复都保留 thinking；Terminus-2 推理时默认不把历史轮次的 thinking 发回给模型。
   **仅 E-TB-v2（旁路实验）已修复**：数据按轮拆成独立样本。每个样本的上下文里，之前各轮只保留可见内容、去掉 thinking，按 `preserve_thinking=false` 渲染，loss 只算最后一轮。训练看到的上下文和评测时的 prompt 逐字节一致（`build_tb_v2.py check-turns`）。主实验的其他模型仍然保持原样。
6. **E-SWE 的训练数据只有 Python**，而 HARD-51 里约 30 道是 Go 或 JS/TS 题。

**TODO（低优先级，时间允许再做）**
- 把 E-SWE（v1）的任务向量缩放到 λ=0.5 后合成权重，只跑 SWE×3，检验减弱行为漂移能否让成绩回到 Base 水平。约 2.5 GPU 小时，不需要新数据。

## 8. 模型（Hugging Face，待填写）

| 模型 | 链接 | Revision | 内容 |
|---|---|---|---|
| E-SWE / E-TB / E-ASI / E-HLE | 待填写 | | LoRA adapter + 配置 |
| Mixed | 待填写 | | LoRA adapter |
| Merged | 待填写 | | 合并后的完整权重 + merge recipe |

加载方法、merge recipe 和许可证见 `models/README.md`。

## 9. 复现

```bash
# 0) 环境（集群登录节点；所有缓存放在 /scratch）
source /scratch/eh3323/apex/env.sh
bash scripts/train_setup.sh                 # envs/train（锁定文件：configs/train/train.lock.txt）
bash scripts/eval_setup.sh all              # Harbor 补丁、mini-swe-agent、asibench、opencode、hle_eval
bash scripts/setup_data_env.sh              # envs/data

# 1) 数据
bash scripts/prepare_data.sh                # 下载（锁定 revision）→ 构建 4 个专家 → 汇总校验

# 2) 训练（单卡；E-SWE 需要 H200/B200）
scripts/train_submit.sh configs/train/e_swe.yaml 1 04:00:00
CONSTRAINT="h200|b200|rtx6000" scripts/train_submit.sh configs/train/e_tb.yaml 1 06:00:00
# e_asi / e_hle 同上；mixed:
scripts/train_submit.sh configs/train/mixed.yaml 1 12:00:00

# 3) 合版
scripts/merge_submit.sh analysis                              # 冲突分析
scripts/merge_submit.sh experts configs/merge/ties.yaml       # 以及 linear / dare_* / 系数扫描

# 4) 评测（在同一个 GPU 作业里启动 vLLM 和沙箱）
sbatch --export=ALL,SETS=tb2+swe51 slurm/eval_prepull.sbatch
sbatch --export=ALL,RUN=base,MODEL_DIR=<dir>,BENCHES=hle+swepro slurm/eval_allinone.sbatch
# ...（每个模型 / benchmark 的完整命令见 docs，待补充）

# 5) 汇总与绘图
python src/eval/summarize.py table
python scripts/plot_curves.py
```

## 10. 目录结构

```
configs/   train/ merge/ eval/ data/      训练、合版、评测、数据配置（含锁定版本）
data/      README.md, manifests/          数据说明与清单（原始数据在 /scratch，不进 git）
src/       data/ train/ merge/ eval/      实现
scripts/   slurm/                         运行命令与 Slurm 作业
patches/                                  对 Harbor / ASI-Bench / hle_eval 的补丁
logs/<run_id>/                            训练、评测、错误与资源日志
curves/                                   loss 曲线（PNG）+ source/（CSV）
results/{base,experts,merged,mixed}/      逐任务结果与汇总
analysis/                                 失败案例分析、置信区间、合版冲突分析
models/README.md                          模型链接、merge recipe、加载方法
```
