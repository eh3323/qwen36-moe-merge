# E-ASI 训练数据调研与 E-ASI-v2 方案（2026-10-15）

**调研范围。** 只做了网络检索（HF API、数据卡、arXiv/GitHub），并阅读了本地 `data/README.md` §3/§4/§8/§10。没有在集群或租用机器上运行任何东西。

**核对方式。**
- 下表的 revision、许可证和规模都是 2026-10-15 通过 HF API 和数据卡逐条核对的。
- 标“待确认”的项目未能确认。
- "抽查"是指下载了几 MB 的分片，在本地只读检查。

**结论先说。** 目前**没有**一个现成数据集同时满足五个条件：原生 opencode 格式、带 thinking、经过验证、科学计算领域、许可证宽松。
- 最接近的是 SmolDataEnvs-multiharness-sft，但它没有许可证，也没有 thinking。
- 因此，E-ASI-v2 的主体数据需要用开源教师在**带 verifier 的现成任务环境**上自己生成。

## 1. 候选数据集（按对 E-ASI-v2 的价值排序）

| # | HF id @rev（日期） | 类型 / 规模 | 许可证 | 教师 | 格式 | 成功过滤 | ASI 相关度 | 污染风险 | 转换工作量 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | [nvidia/Nemotron-Terminal-Synthetic-Tasks](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Synthetic-Tasks) @7e53648e（02-23） | **环境，不含轨迹**。下载核对了 `mixed/scientific_computing`，共 1,000 个任务。另有 mixed/data_science；medium 分片共 0.9 GB，sc/ds 任务数未逐一统计 | CC-BY-4.0 | —（NVIDIA 合成任务，也就是现 E-ASI 数据所用的任务族） | Harbor 任务：`task.toml`、`instruction.md`、Dockerfile、`tests/test.sh`、pytest `test_outputs.py`。`solution/` 为空 | 有 verifier（pytest，带数值容差），可以只保留 reward=1 的轨迹 | **高**。抽查到的任务例如"MNA + 梯形法自适应步长的电路仿真"、"受迫阻尼摆稳定性"、"带噪声的弹簧-阻尼参数估计"：都要实现数值方法、把结果写成文件、按容差判分，形态与 ASI 最接近 | 中低。沿用现有去污染链（13-gram、方法黑名单、bge cos ≥ 0.85）即可，**在生成前对任务描述做过滤** | 中。`task.toml` 指向 NVIDIA 内网镜像，需改为用 Dockerfile 构建（基础镜像是公开的 t-bench ubuntu-24-04）。`test.sh` 运行时会 pip 安装 pytest，离线环境需要预装。轨迹要用教师生成 |
| 2 | [osunlp/D3-Gym](https://huggingface.co/datasets/osunlp/D3-Gym) @8499689a 和 [D3-Gym-Trajectories](https://huggingface.co/datasets/osunlp/D3-Gym-Trajectories) @b21b5006（05-05） | 565 个"数据驱动科学发现"任务，来自 239 个真实科研仓库。每个任务有 Docker Hub 镜像和自动生成的 `eval_script`，并带 6,373 条轨迹 | 数据集为 MIT。源仓库许可证混杂：43 个 GPL，39 个无许可证、卡片写"假定可用于研究" 待确认：| Qwen3-32B/14B/8B/4B（Apache-2.0） | 轨迹是**单轮** `instruction/input/output`，即 `<think>` 加一个完整的 `solution.py`，结果写到 `pred_results/`。这是 ScienceAgentBench 的直接提示格式，**没有工具调用** | 轨迹是否只保留通过的：待确认：卡片和摘要都没有说明。环境本身有 eval 脚本，可判 pass/fail | 领域**高**，格式**低**。真实科研代码，结果文件由脚本比对，正是 ASI 的输出形态 | **中**。需要按 `original_repo` 对 ASI `task_meta` 的论文名和仓库名查重。它与我们的 ASI 开发集 ScienceAgentBench 出自同一团队、同一格式，必须对 SAB 的仓库去重 | 现成轨迹转成 opencode 价值低，只能合成"write + bash 运行"两步。建议用其镜像重新做 agent 生成，工作量中 |
| 3 | [FineEnvs/SmolDataEnvs-harbor-train](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-harbor-train) @47196355（09-30）。同一批任务也发布为 [data-agent-harbor-train](https://huggingface.co/datasets/FineEnvs/data-agent-harbor-train) @3dbc6694 | 5,000 个数据分析任务（easy 1,433 / medium 2,845 / hard 722），另有 test 250 和 eval 144 | MIT | —。任务来自 jupyter-agent 的 Kaggle 问答。卡片说任务由"strong agent models"在沙箱中验证可解，但没说是哪个模型 待确认：。这只影响任务筛选，不进入训练文本 | Harbor。grader 是确定性的（精确匹配 → 数值容差 → 列表 → 符号等价）。卡片示例直接用 `--harness opencode` | 有 verifier | **中**。数据科学题，题目短，答案写入 `answer.txt`，不是数值方法实现 | 低 | 低。原生 Harbor + opencode，但要自己生成轨迹。可以替代现在 jupyter → 脚本的转换 |
| 4 | [FineEnvs/SmolDataEnvs-multiharness-sft](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-multiharness-sft) @c966b75d（10-01） | opencode 子集：4,825 条按轮样本，来自 801 条成功 rollout。四个 harness 合计 17,929 条 / 3,189 条 rollout / 888 个任务 | 待确认：**没有许可证**。卡片只写"源数据的许可证继续适用"。其来源 [qwen38-27b-harbor-rollouts](https://huggingface.co/datasets/FineEnvs/qwen38-27b-harbor-rollouts) 明确写了"no public redistribution license asserted"。**按我们的硬约束暂不可用** | Qwen3.8-27B @1d4bf0f2（Apache-2.0） | **原生 opencode 1.18.30 请求**：真实的 system prompt，8 个工具（bash/edit/glob/grep/read/skill/todowrite/write）。按轮组织，prompt → 下一条 completion，历史 assistant 消息只作上下文 | 是。reward=1，确定性 grader，每个任务×harness 取第一个成功的 rollout | **中**。格式最接近我们要的，但抽查 1 个分片（256 条）：reasoning **全部为空**（采集时没开 thinking）；214 次工具调用**全是 bash**。任务与第 3 行相同，都很短 | 低 | 如果许可证能澄清，工作量低。更现实的做法是用第 3 行的 MIT 任务和同一个教师自己生成 |
| 5 | [nvidia/Nemotron-Terminal-Corpus](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus) @a1667c4f（**现用**） | scientific_computing 19,482 条，data_science 17,565 条（按 laion 的分区统计） | CC-BY-4.0 | DeepSeek-V3.2（MIT） | Terminus-2 JSON / tmux 按键 | **否** | 中。任务就是第 1 行那些，但要转换格式，也没有成功过滤 | 已有去污染链 | 已实现。改进路径：用第 1 行的 verifier 重放后过滤（待确认：语料的 `task` 字段与任务目录名能否对应，未核实） |
| 6 | [nvidia/Nemotron-SFT-OpenCode-v1](https://huggingface.co/datasets/nvidia/Nemotron-SFT-OpenCode-v1) @556d5237（03-23） | 459K 条，约 30 GB。6 个子集：general、bash_only_tool、agent_skills、question_tool 等 | CC-BY-4.0 | 问题由 GPT-OSS-120B 合成，轨迹由 Qwen3-Coder-480B 生成（均为 Apache-2.0） | 原生 opencode：12 个工具的 schema，带 AGENTS.md / skills 变体 | 否 | **低**。通用编码 / CLI 问答。抽查 general 子集前 3 条：都是 system/user/assistant 单轮，没有工具调用，没有 reasoning | 低 | 低。最多作为 ≤5% 的"全工具集格式锚点"。因为没有 thinking，会教出空的 think，慎用 |
| 7 | [ai-and/termgrade-trajectories](https://huggingface.co/datasets/ai-and/termgrade-trajectories) @7d584d63（10-08） | 36,144 条带评分的 episode，其中 21,910 条通过；共 1,004 个任务 | Apache-2.0 | DeepSeek-V4-Pro、GLM-5.2、Qwen3.6-27B、Gemma-4（Kimi 已排除） | Terminus-2 | 是（pytest） | 低-中。通用终端任务，其中有一些 Python 数值题。它已用于 E-TB-v2，再用到 E-ASI 会让两个专家的数据重叠，干扰合版 | 低 | 中。Terminus 转 opencode 的问题与现状相同 |
| 8 | [nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1) @eaef2694（08-28） | 31,111 条按轮样本，来自 2,716 条成功轨迹、630 个任务 | CC-BY-4.0 | GLM-5.1（MIT） | Terminus-2 按轮（预测下一步动作） | 是（verifier） | 低。运维、数据管道、安全类场景 | 低 | 中 |

**核对后排除**（原因都已核实）：
- **SmolDataEnvs-sft 和 data-agent-sft**：两者都是 MIT，各有 4,677 条成功轨迹。但教师没有披露，而抽查到的 tool_call id 是 `toolu_01…`，这是 Anthropic API 的格式，很可能是 Claude 的输出。
- **qwen38-27b-harbor-rollouts**：没有授予再分发许可。
- **open-athena 的约 100 个 `*-qwen3.5-122b-131k-opencode-traces` 仓库**：是原生 opencode 轨迹，但没有许可证标注，领域是代码和指令遵循。
- **isaacrehg/Nemotron_Terminal_Synthetic_Tasks-GLM5.1-multiharness**：其中有 opencode 轨迹，但没有许可证，也没有成功标记。
- **harborframework/terminal-bench-science**：这是一个 benchmark，带有"不得进入训练语料"的 canary，而且是私有仓库。
- **camel-ai/seta-sft-kimi-k2.5-thinking**：Kimi-K2.5 的许可证是 other（modified-mit），与当初排除 Kimi-K3 的规则相同。
- **open-thoughts/AgentTrove**：教师中混有 GPT-5-nano/mini，而且没有科学计算类来源。
- **SciCodePile**：生成模型没有披露，单轮。
- **osunlp/AutoSDT-5K**：部分源仓库只允许学术或非商业使用。

**教师模型。**
- 不可用：GLM-5.3、Qwen3.8-Flash-Next、Qwen3.8-2.4T-A95B，许可证都是 other。
- 可用：
  - Qwen3.8-27B：Apache-2.0，27.8B，单张 H200 就能跑。
  - Qwen3.6-27B：Apache-2.0。
  - DeepSeek-V4.1-Flash：MIT，763B。
  - GLM-5.2：MIT，753B。
  - DeepSeek-V4-Pro：MIT。
- 后三个都需要 ≥8 卡的单节点作业。

## 2. E-ASI-v2 方案

**数据来源和配比**（按轮样本的 token 预算约 15M，与 E-TB-v2 相同；预计需要 600–900 条成功轨迹，切成 1.2k–2k 条按轮样本）：
- **约 60%：Nemotron-Terminal-Synthetic-Tasks 的 scientific_computing 任务**（mixed 加 medium）。少量 data_science 任务可以并入这一份。
- **约 20%：D3-Gym 的 565 个任务**，以 agent 方式重新生成。这是最接近"真实科研代码加结果文件"的来源。
- **约 15%：SmolDataEnvs 的 medium/hard 任务**，替代现在 jupyter → 脚本的转换。
- **可选，≤5%：Nemotron-SFT-OpenCode-v1** 中多轮、调用过 read/edit/write 的样本。因为没有 thinking，默认不用。

**生成方式**：
- 教师用 Qwen3.8-27B，开启 thinking。
- 用 Harbor 驱动 **opencode 1.17.15**，也就是 eval 用的版本，暴露**完整工具集**。
- 在 vLLM 前面放一个记录代理，**录下 opencode 真实发出的请求**。这样训练文本与 eval 字节一致，同时解决 §8 的格式近似问题（skills/AGENTS 块缺失、手写的 JSON schema、只暴露 bash）。
- opencode 自己发起的标题生成和摘要等旁路调用要过滤掉。
- user 消息可以选择用 asibench `opencode_cli` adapter 的通用外框来渲染：只用外框文本，不含任何任务内容，来自公开 repo，不碰 seed31415。

**过滤规则**：
- 只保留 verifier 判 reward=1 的轨迹。
- 优先难题：教师通过率 ≤0.67 的任务；每个任务最多 2 条轨迹。
- 丢弃以下轨迹：发生过上下文压缩或摘要的、超时的、泄露凭据的、少于 8 个 assistant 轮的。
- 偏好长轨迹（≥20 轮），并鼓励一条轨迹写多个结果文件、带自检步骤。

**解决 thinking 不一致**：
- 沿用 E-TB-v2 的 `build --per-turn` 和 `check-turns`。
- 每条样本是"截至第 k 个 assistant 轮的前缀"，前文中历史轮次不带 reasoning。
- 样本标记 `render.preserve_thinking=false` 和 `loss=last_assistant`。
- 校验时与 eval 端的 `apply_chat_template(..., enable_thinking=True, preserve_thinking=False)` 做字节级和 token 级比对。

**去污染**：
- 在**生成前**过滤任务描述，这样也省算力。沿用现有链条：13-gram 对 seed42 B1–B4 的 prompt 和 62 个 `task_meta` 字符串、方法黑名单、bge cos ≥ 0.85。
- 新增：D3-Gym 的 `original_repo` 对 ASI 的论文 / 仓库名，以及对 ScienceAgentBench 开发集的仓库做查重。
- `guard_path(seed31415)` 防火墙保持不变。

**工作量与算力估计**（诚实估计）：
- **方案 B，推荐，需要生成。人工约 17–23 h**：
  - Harbor + opencode 在 apptainer 上离线运行，包括 Nemotron 任务的 Dockerfile → SIF 改造和 pytest 预装：4–6 h。
  - 请求记录代理：3–4 h。
  - D3-Gym 适配：3–4 h。
  - 过滤器和按轮构建：3–4 h。
  - 去污染、verify 和 manifest：2 h。
  - 训练和开发集评测：2–3 h。
- **方案 B 的算力**：
  - 约 2,000 条 rollout（约 1,000–1,500 个任务，每个最多 2 次尝试；通过率按 40–50% 估）。
  - 每条约 20 轮、约 30k 生成 token。合计约 60M 生成 token，加约 0.8B prompt token（大部分命中前缀缓存）。
  - 折合 **约 15–30 H200 GPU 小时**，墙钟约 1 天（3–4 个单卡作业并行）。
  - **主要风险**：每张 GPU 只配 16 个 CPU，一个作业最多约 14 个并发沙箱，GPU 利用率可能低于 H200 两小时平均 60% 的阈值而被自动取消。需要提高并发，或改用 preemption 分区。
  - 训练和此前的专家一样，约 1 GPU 小时。
- **方案 A，兜底，不生成。人工约 6–8 h，只用 CPU**：
  - 把现有 E-ASI 数据改为按轮渲染，修正 thinking 不一致。
  - 用 Nemotron 的 verifier 重放 tmux 按键，做成功过滤。待确认：前提是 task 能对应上。
  - 方案 A 解决不了"只有 bash"和格式转换伪影。

**预期收益**：
- **目前没有证据**：E-ASI 的 B3 分数仍是 TBD；E-TB-v2 也只证明了 val loss 下降。
- 机制上，方案 B 能去掉四个已知问题：伪成功、格式转换伪影、只有 bash、thinking 不一致；任务形态也更接近 ASI（实现数值方法、写结果文件、按容差判分）。
- 但它**修不了时长差距**：Nemotron 任务的 agent 超时是 900 s，ASI 是 10,800 s。教师也只是一个 27B 模型。
- 预计对 B3 是"中性到小幅提升"，超过 5 分会让人意外。更确定的收益是合版时 E-ASI 带来的格式干扰更小。
- 建议先用两个门槛检查再决定是否进合版：
  - ScienceAgentBench 35 题开发集；
  - 留出的 Nemotron sc 任务的 opencode 通过率。

## 3. 现有 E-ASI 数据的主要问题（供 README「下一步」引用）

现有 E-ASI 数据（868 条，14.93M token）很可能与 ASI-Bench 不匹配，原因有五点。
1. **没有成功过滤。** 约 85% 的数据来自 Nemotron-Terminal-Corpus，所谓"完成"只代表 DeepSeek-V3.2 自己宣布任务完成，没有经过测试验证。E-TB v1 已经表明，这类数据会教出"声称验证过、其实没验证"的行为。jupyter 部分只核对了最终答案。
2. **格式是转换来的，而且只有 bash。**
   - Nemotron 部分把 Terminus-2 的 tmux 按键拼成每轮一次 bash 调用，观察内容是重建的。
   - jupyter 部分把有状态的 notebook 改成"重跑整个脚本"的语义，E2B 的输出截断也是近似处理。
   - opencode 的请求是照源码复现的，不是抓包得到的：缺 skills/AGENTS 块，JSON schema 是手写的。
   - 训练时只暴露 bash，而 eval 时 opencode 提供 read/write/edit/glob/grep 等完整工具集。
3. **thinking 上下文不一致。** 训练时每个历史 assistant 轮都保留了 reasoning，而 eval 使用 `preserve_thinking=false`，会丢掉历史 thinking。
4. **任务太短、太浅。** 每条轨迹约 15k token，≥12 轮就算"长"，上限 32K。任务是合成的技能题和 Kaggle 问答。ASI 则要求在最长 3 小时的运行中实现数值方法、产出结果文件，并与参考答案逐项比对。
5. **教师较旧**（DeepSeek-V3.2 / Qwen3-Coder）。

因此，E-ASI 在同分布验证集上 val loss 下降（0.585 → 0.340），并不能说明它在 ASI-Bench 上会有提升。
