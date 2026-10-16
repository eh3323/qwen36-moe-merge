# 失败案例目录：最终合版 merged_v4

> 状态：SWE、HLE、TB2（集群 a1）案例基于 2026-10-16 01:00 EDT 的只读快照；§6（Nebius TB2）和 §7.3（v4 的 ASI）在 05:30 EDT 所有运行和评分完成后补写。数字与 README §8 一致。
> 对象：最终合版 **merged_v4**，即 E-SWE v2、E-TB v2、E-ASI、E-HLE 四个 LoRA 专家的 TIES 合并（`ties_l1_d0.5_v4`，density 0.5，λ=1）。v3 = 去掉 E-TB v2 的同配方合版，本文只在 HLE 上作对照。
> 证据路径如无特别说明，相对 `/scratch/eh3323/apex/results`；ASI 的 RunPod 归档和 TB2 的 Nebius 路径另行标注。缩写见 §0.3。这些原始输出留在集群上，评审无法直接访问；仓库里可核对的是逐任务结果 `results/**/*.jsonl`、SWE 补丁 `results/*/swepro/patches/` 和 `logs/eval/` 下的日志。
> HLE 遵守防污染规则：只列题目 id、学科类别、答案类型、是否带图、判分结论、token 数、finish_reason、置信度，不含题干、标准答案或模型回答文本。
> 轨迹摘录每段不超过 5 行，只取自命令、命令输出、测试输出。格式沿用 `failure_cases_v4.md` 与 `failure_cases_draft.md`。

## 0. 范围、口径与缩写

### 0.1 数据范围

- **SWE-bench Pro HARD-51**
  - 纳入的运行：**v4**：`merged_v4__swepro__r1/r2/r3`；**Base**：`base__swepro__r1/r2/r3`；**E-SWE v2**：`e_swe_v2__swepro__r1`；**Mixed**：`mixed__swepro__r1`
  - 说明：mini_toolcall（单个 bash 工具），250 步上限，50 分钟预算；agent 与 regrade 分成两个 job。51 道题都能起沙箱（env_error 为 0）
- **HLE**
  - 纳入的运行：**v4 实测判分**：`merged/hle/merged_v4__hle__s1`；对照 Base `base__hle__s1`、E-HLE `e_hle__hle__s1`、Mixed `mixed__hle__s1`、v3 `merged__hle__s1`
  - 说明：**本次已用 v4 实际判分结果，替换了 `failure_cases_v4.md` 里的 v3 代理。** v4 判分完成 2459/2500，1 题 pending（`674818a0d2dbfbb2e99bd257`，Math/exactMatch），40 题三轮后仍无预测
- **TB2（Terminus-2 / Harbor）**
  - 纳入的运行：**v4 集群**：`merged/tb2/merged_v4__tb2__a1`（89/89）、`__a2`（截止时未跑完，不计入，仅作上下文）；**Base 集群**：`base__tb2__a1`、`__a2`（各 89/89）；**Nebius 同条件运行**（Base、E-TB v2、Mixed、v4）见 §0.2、§6
  - 说明：快照时 v4 a1 还有 2 道题（winning-avg-corewars、write-compressor）没判分，最终两题都是 timeout；下面的案例按快照时已判分的 87 题分析，成功率按 89 题分母报告（与最终结果相同）
- **ASI-Bench B3（seed31415，opencode agent）**
  - 纳入的运行：四个模型的 agent 运行都在 RunPod 上（README §11 第 19 条），归档与评分在集群 `results/*/asi/`
  - 说明：harness/环境案例（§7.1）、E-ASI 的一个模型行为案例（§7.2）、v4 的 4 个案例（§7.3，S1–S4）

### 0.2 云上（Nebius）运行的地位

TB2 的 Nebius VM 运行（8×H100，Docker 原生后端；Base 2 次、E-TB v2、Mixed、v4 各 1 次）是 README §8 主表用的同条件比较，最终结果见 §6。本文 §2.3 和 §5 的逐题案例写于 01:00 EDT 的快照，当时云上运行都没跑完（base_cloud a1 82/89、a2 71/89；merged_v4_cloud a1 58/89；e_tb_v2_cloud a1 71/89；mixed_cloud a1 84/89），所以 TB2 案例用的是集群运行（Docker vs 打补丁的 singularity，条件不同），云上运行只在个别案例中作上下文。

### 0.3 缩写与根因类别

- 运行：**V1/V2/V3** = merged_v4 的 swepro r1/r2/r3；**B1/B2/B3** = Base 的 r1/r2/r3；**E2** = E-SWE v2；**MX** = Mixed。TB2 用 `v4 a1`、`Base a1/a2`。
- SWE 路径：`V1:A` = `merged/swepro/merged_v4__swepro__r1/jobs/…__agent`，`V1:G` = 同一运行的 `…__regrade`，依此类推。TB2 trial 目录下有 `agent/trajectory.json`（Terminus-2 的 steps）、`result.json`（含 `verifier_result.rewards.reward`、`agent_result.metadata`）、`trial.log`。
- **根因类别**：
  - **MERGE**：合版引入。Base 和对应专家都没有（或明显更少），只在 v4 上出现；"专家能力被稀释"也归入这一类。
  - **MODEL**：模型行为，但 Base 或对应专家也有，v4 继承下来。
  - **HARNESS**：评测框架或环境问题，与模型无关。
  - **MIX**：harness 机制与模型行为共同造成。
- **回退（合版引入的回退）**：同一任务对照（Base，或对应专家）能做对、v4 做不对。本文据题目要求对这类案例显式标注。SWE 用"3 次里成功几次"判断。

---

## 1. 汇总表（全部案例）


**按题目要求的类别归因**：题目列出的类别是定位错误、补丁错误、测试理解错误、合版引入的退化、环境/工具失败、超时和其他。下表“题目类别”列给出每个案例的主类别，括号里是次要类别。“合版引入的退化”优先作为主类别；对 TB2、ASI，“补丁错误”指交付的解答本身错误，“测试理解错误”指误读任务要求或验证条件、没有验证就宣称完成。各类计数（按主类别）：

| 题目类别 | 案例数 | 案例 |
|---|---|---|
| 合版引入的退化 | 12 | A1、A2、A3、A4、A9、B3、B6、D3、D5、T1、T2、S3 |
| 定位错误 | 2 | A10、A11 |
| 补丁错误 | 9 | A5、A6、A12、B2、B4、T4、T5、T6、S2 |
| 测试理解错误 | 5 | B1、B5、T3、T7、S4 |
| 环境/工具失败 | 11 | C1、C2、C3、C4、D2、T8、T9、T10、H1、H2、H3 |
| 超时 | 3 | A7、A8、C5 |
| 其他 | 4 | D1、D4、H4、S1 |

> 列：id · 基准 · 模型/运行 · 任务 · 类别（合版引入的回退 / 能力回退 / harness-环境）· 题目类别 · 一句话。
> "合版引入的回退"= MERGE 且相对 Base 或专家有得分回退；"能力回退"= MODEL（v4 从 Base/专家继承的行为缺陷，或模型逻辑错误）；"harness-环境"= HARNESS/MIX。

| # | 基准 | 运行 / 任务 | 类别 | 题目类别 | 一句话 |
|---|---|---|---|---|---|
| A1 | SWE | V1 / flipt-e2bd | 合版引入的回退 | 合版引入的退化（补丁错误） | 提交前 `git add && git commit`，工作区变干净，导出补丁 0 字节（Base 2/3、E2 解出） |
| A2 | SWE | V2 / flipt-e2bd | 合版引入的回退 | 合版引入的退化（补丁错误） | "All tests pass. Now create a final commit"，补丁 0 字节 |
| A3 | SWE | V2 / openlibrary-0dc5 | 合版引入的回退 | 合版引入的退化（补丁错误） | 测试全过后 commit，补丁为空（Base 3/3） |
| A4 | SWE | V3 / navidrome-bf2b | 合版引入的回退 | 合版引入的退化（补丁错误） | 看到 `git diff --cached` 为空仍提交 |
| A5 | SWE | V1 / protonmail-715d | harness-环境? → MERGE（不回退） | 补丁错误 | commit 信息里的反引号被 shell 执行，又 `--amend`；补丁 0 字节（Base 0/3，不构成回退） |
| A6 | SWE | V3 / qutebrowser-e70f 等 | MERGE（多数不回退） | 补丁错误 | "submit the patch with a commit message"；12 个 commit trial，11 个空补丁 |
| A7 | SWE | V3 / NodeBB-8168 | 能力回退（继承 E2） | 超时 | 4 次请求各 600 s 被取消，第 5 次写满 81920 token；同一句话重复 963 次 |
| A8 | SWE | V3 / element-web-772d | 能力回退（继承 E2） | 超时 | 5 次取消后第 6 次写满 81920 token；同一句重复 399 次 |
| A9 | SWE | V1 / vuls-ca3f | 合版引入的回退 | 合版引入的退化（环境/工具失败） | 断网沙箱里把 `go mod download -x` 原样执行 132 次（Base 2/3、E2 解出） |
| A10 | SWE | V2 / flipt-2ca5 | MERGE（不回退） | 定位错误 | 同一条 grep 重复 210 次，一行没改 |
| A11 | SWE | V2 / vuls-ef2b | 能力回退 | 定位错误 | 只读不改，"Let's start implementing" 说 136 次直到上下文溢出 |
| A12 | SWE | v4 全部 / 补丁卫生 | MERGE（不影响得分） | 补丁错误 | 33/127 补丁带 `.bak`/`.backup`（Base 3/145） |
| B1 | SWE | V1 / vuls-5af1 | 能力回退 | 测试理解错误 | 删掉已有测试文件、改语义迁就自写测试（Base 3/3、E2 解出） |
| B2 | SWE | V3 / vuls-5af1 | 能力回退 | 补丁错误 | 多内核场景选错版本（427.18.1 而非 427.13.1） |
| B3 | SWE | V1–V3 / teleport-eda6 | 合版引入的回退（稀释） | 合版引入的退化（测试理解错误） | 三次都和隐藏测试 API 对不上，0/28；E2 是 28/28 |
| B4 | SWE | V1、V3 / teleport-f432 | 能力回退 | 补丁错误 | raw keystore 实现有误（Base 2/3） |
| B5 | SWE | V1、V3 / element-web-ecfd | 能力回退 | 测试理解错误（补丁错误） | 只修一部分（25/26），靠自写随后删掉的测试"验证"（E2 26/26） |
| B6 | SWE | V1–V3 / openlibrary-53d3 | 合版引入的回退（稀释） | 合版引入的退化（补丁错误） | 返回 dict 而非对象；E2 是 8/8 |
| C1 | SWE | V2 / element-web-ecfd | harness-环境（MIX） | 环境/工具失败 | jest 跑整个目录 OOM，之后重复提交 102 次，补丁丢失 |
| C2 | SWE | V2 / navidrome-bf2b | harness-环境 | 环境/工具失败 | 补丁末尾缺换行，gg.go 没应用（假回退） |
| C3 | SWE | V3 / flipt-e2bd | harness-环境 | 环境/工具失败 | 同一 bug 使整个 cache.go 没应用（假回退） |
| C4 | SWE | 全部运行 | harness-环境 | 环境/工具失败 | `stdout.strip()` 去掉补丁末尾换行，最后一个文件整体丢失（影响所有模型） |
| C5 | SWE | V3 / 多题 | harness-环境（MIX） | 超时（环境/工具失败） | 客户端 600 s 超时叠加长生成，r3 有 41 次取消重试 |
| D1 | HLE | v4 vs Base，208 题 | 能力回退（部分恢复） | 其他 | Base 做对、v4 做错的题里 v4 思考更短（中位 0.64×），中长推理题 Q3 −4.8 |
| D2 | HLE | v4 / 40 题无预测 | harness-环境（MIX） | 环境/工具失败（超时） | 生成首轮大面积 Connection error，第 2 轮补跑 2185 题；40 题三轮后无预测（Base 3） |
| D3 | HLE | v4 / 校准 | 合版引入的回退 | 合版引入的退化 | 校准误差 64.0（E-HLE 36.0），置信度 ≥95 却答错 736 题（E-HLE 47） |
| D4 | HLE | v4 / 带图 342 题 | 能力回退（部分恢复） | 其他 | 带图准确率 13.5（Base 15.5，−2.0）；图文推理长度较 v3 回升 |
| D5 | HLE | v4 / Chemistry 等 | 合版引入的回退 | 合版引入的退化 | Chemistry 9.7（Base 16.4，−6.7，p=0.035）；个别理科类别被拉低 |
| T1 | TB2 | v4 a1 / crack-7z-hash | 合版引入的回退 | 合版引入的退化（超时） | 对加密压缩包反复尝试递增长度口令的退化循环，把 `C-c` 当字面命令发；128 步超时（Base 12–13 步解出） |
| T2 | TB2 | v4 a1 / break-filter-js-from-html | 合版引入的回退 | 合版引入的退化（超时） | 写了 33 个 debugN.py 反复试，166 分钟超时（Base 2/2） |
| T3 | TB2 | v4 a1 / sparql-university | 能力回退 | 测试理解错误 | 查询语法反复报错，仍 4 次标 task_complete；596 条命令（Base 14 步解出） |
| T4 | TB2 | v4 a1 / build-pmars | 能力回退 | 补丁错误 | 走了更绕的构建路线（改成 curses 变体）仍判错；99 条命令（Base 30 条解出，v4 a2 解出） |
| T5 | TB2 | v4 a1 / extract-elf | 能力回退 | 补丁错误 | ELF 段内取值逻辑错，隐藏参考 97 处不一致（Base a1 解出） |
| T6 | TB2 | v4 a1 / financial-document-processor | 能力回退 | 补丁错误 | OCR 发票金额算错（期望 6558，得 4382）（Base 2/2） |
| T7 | TB2 | v4 a1 / overfull-hbox | 能力回退 | 测试理解错误 | 替换了不在允许同义词表里的词（horizon→edge），违反约束（Base a1 解出） |
| T8 | TB2 | v4 a1 / git-multibranch、mailman、build-pov-ray 等 | harness-环境 | 环境/工具失败（超时） | Terminus-2 摘要死循环（"Even fallback chat failed"）：v4 a1 有 18 个 trial，Base a1 11、a2 12 |
| T9 | TB2 | v4 a1 / rstan-to-pystan、sqlite-db-truncate、tune-mjcf | harness-环境 | 环境/工具失败 | 48G 内存看门狗杀容器（模型跑了重负载/多进程脚本），记 env_error |
| T10 | TB2 | v4 a1 / headless-terminal | harness-环境 | 环境/工具失败 | `RemoteProtocolError: Server disconnected`，trial 失败 |
| H1 | ASI | Base/E-ASI/Mixed/v4 | harness-环境 | 环境/工具失败（超时） | opencode 子代理的 `external_directory` 权限请求在 `--auto` 下不自动放行，会话卡到 3 h 超时 |
| H2 | ASI | Mixed、E-ASI | harness-环境 | 环境/工具失败 | agent 在无 PID namespace 的 RunPod 上 `pkill -f python*`，杀掉 vLLM/asibench |
| H3 | ASI | 评分阶段 | harness-环境 | 环境/工具失败 | 裁判 temperature 被 gpt-5.5 拒、`tasks` 导入路径崩溃、TSP 需 Docker |
| H4 | ASI | E-ASI / math.manifold_pde_meshfree | 能力回退 | 其他 | prompt 230K + 32K 输出 > 262K，ContextOverflowError（模型行为） |
| S1 | ASI | v4 / 全部 60 题 | 能力回退 | 其他 | 逐题分数与 E-ASI 相关 0.67（与 Base 0.22），掉分题与 E-ASI 重合：ASI 回退来自专家 |
| S2 | ASI | v4 / stable_latent_state_fitting 等 4 题 | 能力回退 | 补丁错误 | 第一轮不调用工具就结束或没写交付文件，hard gate 不通过（v4 32/60 vs Base 39/60） |
| S3 | ASI | v4 / 全部 60 题 | 合版引入的回退 | 合版引入的退化 | 每题模型调用中位数 81，Base 46、E-ASI 19：轨迹最长、得分更低 |
| S4 | ASI | v4 / math.symplectic_integrators | 能力回退 | 测试理解错误 | 交付了但说明里没写积分器名，soft gate 判 0（Base 42.8） |

**计数**：共 **46 个案例**（SWE 23 = A1–A12 + B1–B6 + C1–C5；HLE 5 = D1–D5；TB2 10 = T1–T10；ASI 8 = H1–H4 + S1–S4）。其中 **SWE 23 + HLE 5 + TB2 10 + ASI 4（S1–S4）= 42 个直接在最终合版 v4 的运行上**（C4 为"全部运行"口径但含 v4；H1、H3 也涉及 v4，H2/H4 是其他模型 / 专家对照）——远超题目 ≥20 的要求。**明确标为"合版引入的回退"的 = 12 个**（A1–A4、A9、B3、B6、D3、D5、T1、T2、S3），另有一批"能力回退"（A7、A8、A11、B1、B2、B4、B5、D1、D4、T3–T7、H4）是 v4 从 Base/专家继承或自身的能力缺陷。按根因与计数见 §8。

---

## 2. v4 的配对统计（低成本口径）

### 2.1 SWE-Pro HARD-51

| 模型 | 分数（r1/r2/r3 或单次） | 成功 trial | 与 v4 配对（`summarize.py compare`，paired bootstrap，env=zero） |
|---|---|---|---|
| Base | 17.65（19.61 / 11.76 / 21.57） | 27/153 | v4 − Base = **−4.58**，95% CI [−11.76, +1.96]，p = 0.165 |
| E-SWE v2 | 21.57 | 11/51 | v4 − E2 = **−8.50**，95% CI [−18.95, +0.65]，p = 0.061 |
| Mixed | 13.73 | 7/51 | v4 − MX = −0.65，p = 0.79 |
| **merged_v4** | **13.07**（15.69 / 15.69 / **7.84**） | 20/153 | 两个比较都不显著（n=51，方差大），但方向一致 |

逐题回退、行为指标与 C4 harness bug 的影响，完整表格见 `failure_cases_v4.md` §1.2–1.3（本文沿用，未变）。要点：v4 输出量约为 Base 的 2 倍、E2 的一半；合版引入的三类新现象是 `git commit`、`.bak` 备份、环境正常时的重复命令死循环；r3 的 7.84 分主要由失控生成 + 600 s 超时叠加造成（C5）。

### 2.2 HLE（v4 实测判分，n=2500）

官方脚本输出（`judge_stdout.txt`）：Base 20.8%、E-HLE 16.92%、Mixed 16.6%、**v3 17.36%**、**v4 19.2%**；校准误差 Base 74.0、E-HLE 36.0、Mixed 37.0、v3 61.0、**v4 64.0**。

配对比较（`summarize.py compare --bench hle`，10000 次 bootstrap，n_tasks=2499，1 题 pending 自动剔除）：

| 比较 | Δ（v4 − 对照，pp） | 95% CI | bootstrap p | McNemar 不一致对（对照独赢 : v4 独赢） | McNemar 精确 p |
|---|---|---|---|---|---|
| v4 vs Base | **−1.60** | [−3.16, −0.08] | 0.042 | 214 : 174 | 0.048 |
| v4 vs v3 | **+1.84** | [+0.32, +3.36] | 0.020 | 168 : 214 | 0.021 |
| v4 vs E-HLE | **+2.28** | [+0.76, +3.84] | 0.0046 | 170 : 227 | 0.0049 |
| v4 vs Mixed | **+2.60** | [+1.08, +4.12] | 0.0014 | 155 : 220 | 0.00093 |

**与 v3 代理相比最重要的修正**：加入 E-TB v2（v4 = v3 + E-TB v2）把 HLE 从 17.36 提到 19.21，**v4 显著高于 v3、E-HLE、Mixed**，只比 Base 低 1.60 pp（刚好显著）。`failure_cases_v4.md` 里基于 v3 得出的"思考长度减半、截断率是 Base 3 倍"等结论，在 v4 上**明显缓和**（见 D1、D2、D4）。但校准、Chemistry 等类别的回退依然成立。

分组准确率（%，v4 实测）：

| 分组 | n | Base | E-HLE | Mixed | v3 | **v4** | v4−Base | v4−E-HLE | v4−v3 | 不一致对 Base : v4 |
|---|---|---|---|---|---|---|---|---|---|---|
| Math | 1021 | 29.3 | 23.6 | 25.1 | 25.2 | **28.4** | −0.9 | +4.8 | +3.2 | 94 : 85 |
| Biology/Medicine | 280 | 17.5 | 14.3 | 12.5 | 12.9 | **13.9** | −3.6 | −0.4 | +1.1 | 29 : 19 |
| Computer Science/AI | 241 | 15.8 | 14.1 | 13.3 | 14.5 | **15.4** | −0.4 | +1.2 | +0.8 | 17 : 16 |
| Other | 233 | 9.0 | 8.6 | 6.9 | 6.4 | **7.7** | −1.3 | −0.9 | +1.3 | 14 : 11 |
| Physics | 230 | 16.1 | 16.5 | 13.5 | 13.5 | **14.8** | −1.3 | −1.7 | +1.3 | 21 : 18 |
| Humanities/Social | 219 | 13.7 | 7.8 | 7.8 | 12.3 | **12.3** | −1.4 | +4.6 | 0.0 | 16 : 13 |
| Chemistry | 165 | 16.4 | 11.5 | 9.1 | 11.5 | **9.7** | **−6.7** | −1.8 | −1.8 | 17 : 6（p=0.035） |
| Engineering | 111 | 17.1 | 12.6 | 11.7 | 12.6 | **17.1** | 0.0 | +4.5 | +4.5 | 6 : 6 |
| exactMatch | 1909 | 19.7 | 16.1 | 15.5 | 16.2 | **18.5** | −1.2 | +2.5 | +2.4 | 144 : 121 |
| multipleChoice | 591 | 24.2 | 19.6 | 20.1 | 21.2 | **21.3** | −2.9 | +1.7 | +0.2 | 70 : 53 |
| 带图 | 342 | 15.5 | 11.4 | 10.2 | 11.1 | **13.5** | −2.0 | +2.0 | +2.3 | 26 : 19 |
| 不带图 | 2158 | 21.6 | 17.8 | 17.6 | 18.4 | **20.1** | −1.5 | +2.3 | +1.8 | 188 : 155 |
| Base 长度 Q1（0–11.4K） | 625 | 14.9 | 13.4 | 12.0 | 13.4 | **14.6** | −0.3 | +1.1 | +1.1 | 44 : 42 |
| Q2（11.4–24.6K） | 625 | 24.5 | 21.1 | 19.0 | 20.8 | **23.2** | −1.3 | +2.1 | +2.4 | 51 : 43 |
| Q3（24.6–39.3K） | 625 | 26.6 | 19.4 | 20.8 | 20.2 | **21.8** | **−4.8** | +2.4 | +1.6 | 63 : 33（p=0.003） |
| Q4（39.3–81.9K） | 625 | 17.3 | 13.8 | 14.6 | 15.0 | **17.3** | 0.0 | +3.5 | +2.2 | 56 : 56 |

| 其他指标 | Base | E-HLE | Mixed | v3 | **v4** |
|---|---|---|---|---|---|
| 输出 token 中位数 / 均值 | 24.6K / 27.0K | 13.2K / 19.4K | 13.2K / 18.5K | 12.1K / 18.4K | **16.5K / 22.6K** |
| 带图题输出 token 中位数 | 19.1K | 6.9K | 10.0K | 7.0K | **12.9K** |
| 代理日志 finish=length 请求 | 62 | 107 | 87 | 233 | **370** |
| 三轮后仍无预测的题数 | 3 | 4 | 6 | 24 | **40**（见 D2，部分为生成/harness） |
| 自报置信度均值 | 97.3 | 54.6 | 55.2 | 80.9 | **85.5** |
| judge 校准误差 | 74.0 | 36.0 | 37.0 | 61.0 | **64.0** |
| 置信度 ≥95 却答错 | 1804 | 47 | 37 | 503 | **736** |

### 2.3 TB2（v4 集群 a1 对 Base 集群 a1/a2）

成功率（reward=1 的 trial / 89）：**Base a1 42.7%（38/89）、a2 40.4%（36/89）；v4 a1 38.2%（34/89；快照后补完的 2 题均为 timeout）**。v4 a2 截止时未跑完，不计入，只作上下文（快照时已判分 46/58，22 成功）。

逐题配对（v4 a1 vs Base，McNemar 不一致对）：vs Base a1 = 12 : 8（Base 独赢 : v4 独赢），vs Base a2 = 12 : 10，都不显著。方向与 SWE/HLE 一致（v4 略低于 Base）。

行为对比（每 trial，剔除 Terminus-2 的 "Technical difficulties" 占位步后的有效步数）：

| 指标 | **v4 a1** | Base a1 | Base a2 |
|---|---|---|---|
| 有效步数中位数 / 均值 | **45 / 62.3** | 23 / 53.6 | 26 / 55.8 |
| 每步命令数 | **3.55** | 2.02 | 1.90 |
| 命令总数（全 run） | 5418 步内 cmds 更多 | — | — |
| 代理请求（Terminus 侧）写满 81920 token | **10** | 77 | 72 |
| 单次 ≥20K 完成 token 的请求 | 77 | 126 | 102 |
| 客户端 600 s 取消的请求 | **127** | 0* | 0* |
| 摘要死循环（"Even fallback chat failed" ≥3）trial | **18** | 11 | 12 |
| 标过 task_complete 但 reward 0（无异常）的 trial | 24 | 28 | 28 |
| trial 成功时的有效步数中位数 | 14 | 13.5 | 13.0 |

\* Base 的 TB2 跑在 10-14，早于 10-15 00:52 给 `llm_proxy.py` 加上取消日志（`server.py:125`），所以 Base 的 0 次取消是**没记录**，不是没发生；这一项不能直接跨模型比较。

要点：
- v4 在 TB2 上**步数约为 Base 的 2 倍、每步命令更多**，这是合版把"长输出 / 反复探索"风格带到终端任务的直接表现，与 SWE 上观察到的一致。
- **摘要死循环（T8）v4 触发更多**（18 vs 11/12），因为它更容易把上下文顶到 180224 token 的 `max_input_tokens` 阈值——这是 harness 缺陷（README §11 第 3 条），但 v4 的长轨迹让它更频繁命中。
- 成功解出时，v4 的步数并不比 Base 多（都约 13–14 步）；多出来的步数几乎都花在失败 / 超时的任务上。
- `failure_cases_draft.md` 关心的 E-TB v1 症状（批量按键落进 pager、把注释当命令、"卡在 heredoc"约 300 轮循环）在 v4 a1 上**基本消失**：把 `C-x` 当字面命令发只有 2 个 trial（gpt2-codegolf、crack-7z-hash），格式解析错误（"parsing errors"）58 trial，与 Base 的 36/31 同量级，不是 v4 特有。

---

## 3. SWE：合版引入的新行为与继承自专家的失控生成

A1–A4、A6、A9–A12 以及 B1–B6、C1–C5 的详述见 `failure_cases_v4.md` §3（证据、路径、根因均经本轮抽查无误，未改动）。本节只列**本轮重新核对并修正**的 A5、A7，以及与 A7 同机制的 A8。

### A5. protonmail-715d（V1）：commit 信息被 shell 执行，再 amend（已复核）

- **运行/任务**：`V1:A/instance_protonmail__webclients__cvVxYHG`，任务 `protonmail__webclients-715dbd4e…`，补丁 **0 字节**，`exit_status=Submitted`，`n_calls=148`（轨迹里共 182 条 bash 命令，其中 34 个助手回合发了多条命令）。
- **复核结论**：原案例基本准确，补充两处精度：
  1. 第 179 条 `git add -A && git commit -m "…X-Pm-Encrypt-Untrusted…"`，commit 信息里的反引号被当成命令替换，产生 **19 行 `command not found`**（`X-Pm-Encrypt-Untrusted`、`PinnedKeysConfig`、`getContactPublicKeyModel`…）。但 **commit 本身成功了**（`[main c6a28baf5] Improve encryption handling…`），因为 `git add -A` 已经把工作区改动暂存并提交，只是 message 乱了。这正是补丁为空的原因——和 A1 同一机制。
  2. 第 180 条 `git commit --amend -m "…"`（去掉反引号）再次成功（`[main 35ee66dbb]`）。模型花两步维护 commit 历史，却不知道 harness 用 `git diff`（对 HEAD）取补丁，提交后 diff 为空。
- **证据**：
  ```
  #179 'cd /app && git add -A && git commit -m "Improve encryption handling for WKD contacts with X-Pm-Encrypt-Untrusted …'
       -> "bash: line 1: X-Pm-Encrypt-Untrusted: command not found" ×19 行；"[main c6a28baf5] Improve encryption handling…"
  #180 'cd /app && git commit --amend -m "…"' -> "[main 35ee66dbb] Improve encryption handling …"
  #181 'cd /app && git diff --stat HEAD~1' -> （显示 7 个文件的改动）
  #182 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'    |  model.patch: 0 字节
  ```
- **根因**：MERGE（同 A1/A6 的 commit 机制）。**回退**：Base 在 protonmail-715d 上也是 0/3（B1/B2/B3 reward 全 0），所以**不构成得分回退**，只属于同一类合版缺陷。原案例在这一点上也是对的。

### A7. NodeBB-8168（V3）：一次思考写满 81920 token，同一句话重复 963 次（已修正计数）

- **运行/任务**：`V3:A/instance_NodeBB__NodeBB-8168c6c4__HDuwNsG`。Base 也是 0/3。`exit_status=BudgetExhausted(in-flight call abandoned)`，`n_calls=49`，`agent_wall_sec=2939.9`。
- **修正**（原案例把次数写成"5 次取消 + 第 6 次写满、短语重复 481 次"，本轮逐条核对代理日志和轨迹后更正）：
  - 最后一轮对应轨迹 `messages[122]`（prompt = 122 条消息，承接上一条工具输出）。该请求在代理日志里**先后 4 次各跑满 600 s 被客户端取消**（起始相对上一条工具输出为 +1 s / +602 s / +1206 s / +1806 s），**第 5 次**在 598 s 内写满 81920 token（`finish=length`）。即 **4 次取消 + 1 次 length**，不是 5+1。
  - 这条响应 `content=None`、`reasoning_content` 长 **343,615 字符**。其中一个 64 字符的句子 `"Let's try to set up the winston logger before requiring winston."` 出现 **963 次**（从第 541 个字符起；旧稿的"481 次"只数了另一个落在换行处的 60 字符窗口）。按句子切分，出现 ≥10 次的句子占了 reasoning 的约 **98%**。
  - agent 在 2940 s 耗尽 50 分钟预算（in-flight call abandoned）；这条响应是截止后才返回并写进轨迹的（轨迹文件 mtime 09:38，result 09:09，最终响应 created 09:28），harness 判为 FormatError（无 tool call），exit 记为 TimeExceeded。
- **证据**：
  ```
  llm_requests.jsonl（n_messages=122，相对上条工具输出 10-15 08:38:17）:
     +1s/+602s/+1206s/+1806s  "client disconnected (cancelled)"  latency≈600.0
     +2409s  finish=length, completion 81920, prompt 85390, latency 598.02
  trajectory messages[122]: finish=length, reasoning_len=343615, content=None, tool_calls=None
     重复句 "Let's try to set up the winston logger before requiring winston." ×963
  agent/result.json: BudgetExhausted(in-flight call abandoned), n_calls=49, agent_wall_sec=2939.9
  ```
- **根因**：MODEL，继承自 E-SWE v2（Base 三次运行没有任何 ≥20K token 生成；E2 有 5 次 length、23 次取消）。r3 的 7.84 分与这类失控生成直接相关（见 C5）。**回退**：相对 Base 不是（都 0/3）。

### A8. element-web-772d（V3）：一轮生成约 1 小时，同一句重复 399 次（计数已修正）

- **运行/任务**：`V3:A/instance_element-hq__element-web__rfJh473`（Base 做对 1/3）。`exit_status=BudgetExhausted(in-flight call abandoned)`，`n_calls=71`，`agent_wall_sec=2938.8`。
- **现象**：机制同 A7。轨迹 `messages[142]`：**先后 5 次各在 600 s 被取消**（+1 / +602 / +1206 / +1806 / +2407 s），**第 6 次**在 542 s 写满 81920 token。`reasoning_content` 长 353,962 字符；出现最多的句子 `"Let's check the `DeviceListItem` component."` 等若干短语各重复约 **399 次**（旧稿写 198 次，偏低），出现 ≥10 次的句子占约 98%。
- **证据**：
  ```
  llm_requests.jsonl（n_messages=142）: +1/+602/+1206/+1806/+2407 s 取消；+3069s finish=length, completion 81920, prompt 102357, latency 542.6
  trajectory messages[142]: reasoning_len=353962, content=None；最高频句 ×399
  agent/result.json: BudgetExhausted(in-flight call abandoned), n_calls=71
  ```
- **根因**：MODEL（继承）。**回退**：相对 Base 是部分回退（1/3 → 0）。

> 其余 SWE 案例（A1–A4、A6、A9–A12、B1–B6、C1–C5）详见 `failure_cases_v4.md` §3；本轮抽查了 A1、A6 的 commit 全表和 C4 的 `server.py:152 stdout.strip()`，与原稿一致。

---

## 4. HLE：用 v4 实测判分重写（替换 v3 代理）

> 下列结论基于 `merged/hle/merged_v4__hle__s1` 的 normalized 行（`repo/results/merged/hle/merged_v4__hle__s1.jsonl`：`tokens_out`、`status`、`extra.finish_reason/confidence/category/answer_type/has_image`）与 `judge_stdout.txt`。只用数值和元数据，不读题干 / 答案 / 回答文本。

### D1. Base 做对、v4 做错的 208 题：思考更短，中长推理题损失最大（较 v3 明显缓和）

- **数据**：
  - Base 做对、v4 做错共 214 题，其中 208 题 v4 正常结束（`finish=stop`，status=fail）；这 208 题 v4 用的 token 与 Base 的比值中位数 **0.64**（v3 代理是 0.46，说明加入 E-TB v2 后思考长度回升）。其中 33 题 v4 token < 0.3×Base，这 33 题里有 24 题 v4 自报置信度 ≥90。
  - 按 Base 输出长度四分位，Q3（24.6–39.3K token）掉得最多：**−4.8 分**，不一致对 63 : 33，p=0.003；最短的 Q1 只掉 0.3 分，最长的 Q4 持平。
  - 这 214 题里，E-HLE 做对 64、v3 做对 66——专家 / v3 在其中约三成上也做对，说明是"思考被压短"而非"完全不会"。
- **代表性题目 id（只列元数据）**：

  | id | 类别 | 答案类型 | Base token | v4 token | v4 置信度 | E-HLE |
  |---|---|---|---|---|---|---|
  | `670285bc39fbddbbfdaffdfe` | Humanities/Social | multipleChoice | 22,565 | 551 | 95 | 对（11,178） |
  | `67256db1afb4a043e613c929` | Biology/Medicine | multipleChoice | 18,243 | 1,512 | 95 | 对（5,153） |
  | `66e932f00bd16d40d4bc4d81` | CS/AI | multipleChoice，带图 | 27,391 | 3,486 | 95 | 错 |
  | `67255d1e100c04e9bad57694` | Other | exactMatch，带图 | 38,085 | 3,224 | 100 | 错 |
- **根因**：MODEL，继承自 E-HLE（E-HLE、Mixed 的短思考分布）。**回退**：相对 Base 是，相对 E-HLE/v3 不是（v4 反而更好）。v3→v4 的净变化是 +214 题做对、−168 题做错（两者都做对 266），净 +46，对应 +1.84 pp。

### D2. 40 题三轮后仍无预测：首轮生成不稳定 + 思考截断（MIX）

- **数据**：
  - v4 有 40 题 status=env_error（三轮补跑后仍无预测）：全是 exactMatch，Math 21、CS/AI 7、Other 5、Physics 5、Chemistry 2，只有 1 题带图。Base 3、E-HLE 4、Mixed 6、v3 24——**v4 的 40 甚至高于 v3 的 24**。
  - 其中 5 题 Base 做对（都是 Math/exactMatch，Base token 27K–65K，中位 46.8K），这 5 题是 Base 也要长推理的难题；整体这 40 题 Base 只做对 5。
  - **这 40 题里有相当比例来自生成阶段的不稳定，不纯是模型截断**：`logs/merged_v4__hle__s1/hle_pred.log` 显示第 1 轮刚开始就连续 ~20 条 `Error: Connection error.`，随后第 2 轮补跑 **2185 题**（Base 第 2 轮只补 42 题）。说明第 1 轮大部分结果没落盘（endpoint 中断），第 2/3 轮在截止前没能把全部补完。代理日志里 finish=length 370 个、取消 81 个（集中在 10-14 23:00–10-15 00:19）。
- **样例 id**：`66e8a3ed3aa94517d4573b14`、`66ea355dd1767539f1308fc3`、`670dbfc042e55f85b247ba49`（都是 Math/exactMatch，Base 做对但 v4 无预测）。
- **根因**：MIX。模型侧的思考失控截断（finish=length 比例 v4 最高）是一部分，但 40 这个数被**生成作业的首轮 Connection error / 未跑完**放大了。对得分影响有限（env=zero 已计入 −0.x pp）。**未核实**：40 题里截断、超时、首轮未落盘各占多少没有逐条对齐。

### D3. 校准：E-HLE 的校准改进在合版后仍大部分丢失（回退）

- **数据**：

  | 指标 | Base | E-HLE | Mixed | v3 | **v4** |
  |---|---|---|---|---|---|
  | 校准误差（judge RMS） | 74.0 | 36.0 | 37.0 | 61.0 | **64.0** |
  | 置信度均值 | 97.3 | 54.6 | 55.2 | 80.9 | **85.5** |
  | 置信度 ≥95 却答错 | 1804 | 47 | 37 | 503 | **736** |
  | 自报置信度恰好 100 的题数 | 1440 | 56 | 41 | 326 | **499** |
  - v4 置信度直方图：70–79 档 348 题、80–89 档 744、90–99 档 675、**恰好 100 的 499 题**。按置信度分档的实际准确率：[0,40) 6.7%、[40,60) 8.9%、[60,80) 12.6%、[80,95) 17.5%、[95,100] 25.6%——排序上仍有信息量，但整体高报。
  - 对 v4 置信度 ≥95 却答错的 736 题，E-HLE 在同一批题上的置信度均值仅 54.9。
- **根因**：MERGE。Mixed 和 E-HLE 都学到了校准（≈36），v4 只保留约 1/4：(74.0−64.0)/(74.0−36.0) ≈ 26%（v3 为 ≈34%，加入 E-TB v2 后校准反而略差）。**回退**：相对 E-HLE/Mixed 是明显回退；相对 Base 仍是改善（64 < 74）。

### D4. 带图 342 题：推理长度较 v3 回升，准确率仍低于 Base（部分恢复）

- **数据**：带图准确率 Base 15.5、E-HLE 11.4、Mixed 10.2、v3 11.1、**v4 13.5**（v4−Base=−2.0，不一致对 26 : 19；v4−v3=+2.3）。带图题输出 token 中位数：Base 19.1K、E-HLE 7.0K、v3 7.0K、**v4 12.9K**。
- **根因**：MODEL（继承自只用纯文本训练的 E-HLE），但加入 E-TB v2 后图文推理长度和准确率都比 v3 回升。**回退**：相对 Base 是（−2.0，比不带图的 −1.5 略大）；相对 E-HLE/v3 不是。

### D5. 部分理科类别 v4 显著低于 Base（回退）

- **数据**：Chemistry **9.7**（Base 16.4，**−6.7**，不一致对 17 : 6，p=0.035，是 v4 唯一单类别显著回退）；Biology/Medicine 13.9（−3.6）、Physics 14.8（−1.3）、Other 7.7（−1.3）。相对 E-HLE，v4 在 Chemistry −1.8、Physics −1.7 仍更低。Humanities/Social 持平、Engineering 持平、Math −0.9。
- **根因**：MERGE。各专家参数在不同类别上的干扰方向不一，Chemistry 受损最重。各类别样本量小（111–280 题），除 Chemistry 外单类别差异都不显著，只能算方向性证据。

---

## 5. TB2：v4 实测案例（Terminus-2 / Harbor，集群 a1）

> 证据取自 `merged/tb2/merged_v4__tb2__a1/jobs/merged_v4__tb2__a1/<trial>/`：`agent/trajectory.json` 的 steps（每步有 `message` 的 analysis/plan/commands、`reasoning_content`、`metrics`）、`result.json`、`verifier/test-stdout.txt`、`trial.log`。对照 Base 的 a1/a2 同名任务。摘录只取命令、分析句和测试断言。

### 口径：哪些算"合版引入的回退"

v4 a1 相对 Base（Base 两次里至少一次解出、v4 a1 失败）共 16 道题。其中按根因拆分：
- **模型行为类回退（T1–T7，共 9 道题）**：crack-7z-hash、break-filter-js-from-html、build-pmars、extract-elf、financial-document-processor、overfull-hbox、sparql-university、cancel-async-tasks、adaptive-rejection-sampler。
- **harness / 环境类（T8–T10，共 7 道题）**：git-multibranch、mailman、build-pov-ray（摘要死循环）、rstan-to-pystan、sqlite-db-truncate、tune-mjcf（OOM）、headless-terminal（连接断开）——这些**不是模型失败**。

反方向（Base 两次都没对、v4 a1 解出）6 道：bn-fit-modify、feal-linear-cryptanalysis、large-scale-text-editing、llm-inference-batching-scheduler、mcmc-sampling-stan、sqlite-with-gcov。所以 TB2 上也有正迁移。

### T1. crack-7z-hash（v4 a1）：退化的穷举循环，把 `C-c` 当字面命令发（合版引入的回退）

- **运行/任务**：`crack-7z-hash__zpSegDz`，reward 0，`AgentTimeoutError`（3 h），128 个有效步、752 条命令。对照：Base a1/a2 都在 12–13 步解出，v4 a2 也解出。
- **现象**：任务要从一个加密压缩包里取出文件。v4 从第 11 步起转入对口令的**暴力尝试**，并在之后反复做同一件事：对同一压缩包反复发**长度递增、前缀相同**的解压尝试命令（单个助手回合一次发 10+ 条，口令串越拼越长），穿插调用 john 工具和下载词表；全程 752 条命令里仅"对同一压缩包的解压尝试"就有 **115 条几乎一样**。它还把终端中断当字面命令发：**`C-c` 作为命令被 bash 当成 `C-c: command not found` 共 20 次**（另有 7 条形如 `C-c\n` 的按键）。中途脚本在后台长跑时，模型连发 60 s `sleep` 轮询，一直耗到 3 h 超时。
- **证据**：
  ```
  step 100 analysis: "The for loop is still running (at 'robot')…we'll send a few more C-c signals…"
  terminal: "root@…# C-c\nbash: C-c: command not found" ×（反复出现）
  result.json: exception=AgentTimeoutError, agent_execution≈10800s；verifier: FileNotFoundError: '/app/solution.txt'（2 failed）
  behavior: 128 有效步 / 752 命令（Base a1 12 步 / 20 命令）；同一解压命令重复 115 次
  ```
- **根因**：MERGE（退化循环 + 命令误用，Base 不出现；这类"动作不随 reasoning 变化、把控制键当命令"与 SWE 的 A9/A10 同族）。**回退**：相对 Base 是（Base 2/2）；但 v4 a2 解出了，说明是高方差而非彻底不会。

### T2. break-filter-js-from-html（v4 a1）：写 33 个 debug 脚本反复试，166 分钟超时（合版引入的回退）

- **运行/任务**：`break-filter-js-from-html__izANao8`，reward 0，`AgentTimeoutError`，117 个有效步。对照：Base a1/a2 都解出（a1 用 44 步 / 140 命令，a2 34 步）。
- **现象**：任务是构造能绕过某过滤器的 HTML。v4 用 `cat > /app/debugN.py` 的方式**写了 33 个互不相同的调试脚本**（debug2…debug34），每个跑一个 selenium 变体然后 grep `Alert detected: True`，始终没命中；末尾大量步是空命令等待后台脚本（52 个"等待"步），单步最长间隔 1905 s（脚本长跑）。它一直没输出 `/app/out.html`，也从不停下来换思路。
- **证据**：
  ```
  files written via `cat >`: /app/debug.py … /app/debug34.py（共 36 处，33 个不同 debugN.py）
  step 118 analysis: "We have run debug34.py…grep 'Alert detected: True'" -> （空，无命中）
  result.json: AgentTimeoutError；verifier: "No alert detected after filtering. The XSS bypass failed."（1 failed）
  ```
- **根因**：MERGE（长轨迹 + 反复试错不收敛，Base 两次都在几十步内解出）。**回退**：相对 Base 是。

### T3. sparql-university（v4 a1）：查询反复报语法错，仍多次标 task_complete（能力回退）

- **运行/任务**：`sparql-university__wxn7No6`，reward 0，100 个有效步、**596 条命令**，中途标过 4 次 `task_complete`。对照：Base a1 用 14 步 / 15 命令解出。
- **现象**：要写一个 SPARQL 查询。v4 的查询在 rdflib / Jena / oxigraph 上反复报 `ParseException: Expected SelectQuery, found '{'` 和 `expected ENCODE_FOR_URI`；它在第 87 步就 `task_complete=true`（此时查询仍解析失败），被 harness 的"确认提交"挡回后，继续改查询、再标完成，如此 4 次（87、98、99、100）。最终提交的查询仍是语法/语义不对的版本（子查询里 `GROUP BY ?professor ?deptAny` 违反功能依赖）。
- **证据**：
  ```
  step 87 tc=True -> harness: "Are you sure you want to mark the task as complete?"
  step 88 analysis: "We need to fix the syntax error…FILTER EXISTS with a subquery containing GROUP BY/HAVING…"
  verifier: "pyparsing…ParseException: Expected SelectQuery, found '{' (at char 908)"（2 failed, 1 passed）
  behavior: 100 有效步 / 596 命令；标 task_complete 4 次
  ```
- **根因**：MODEL（能力 + "宣称完成但没验证"，与 draft 的 C1/B4 同族；v4 的长轨迹让它反复试）。**回退**：相对 Base 是。

### T4. build-pmars（v4 a1）：走了更绕的构建路线仍判错（能力回退）

- **运行/任务**：`build-pmars__QRgyKXB`，reward 0，24 步 / 99 命令。对照：Base a1/a2 都解出（各 ~13–14 步 / 30–35 命令），v4 a2 也解出。
- **现象**：任务要从 Debian 源码去掉 X11 构建 pmars。Base 直接把 Makefile 里的 `-DXWINGRAPHX` 和 `-lX11` 删掉就过了；v4 则**改成启用 curses 图形**（`-DXWINGRAPHX` → `-DCURSESGRAPHX`、`LIB=-lncurses`），构建能过、自测能跑，但隐藏测试 `test_pmars_works` 判 `Expected stepping, got 0 addresses`（调试器行为不对）。v4 还中途用 heredoc 把 Makefile 写坏又恢复，命令数是 Base 的 3 倍。
- **证据**：
  ```
  #19 sed -i 's/-DXWINGRAPHX/-DCURSESGRAPHX/' Makefile；sed 's|…-lX11|LIB = -lncurses|'
  verifier: "rave.red: Expected stepping, got 0 addresses / assert 0 >= 2"（1 failed, 3 passed）
  对照 Base: "remove -DXWINGRAPHX from CFLAGS and -lX11 from LIB"（直接去掉，通过）
  ```
- **根因**：MODEL（方案选择偏复杂、过度改动）。**回退**：相对 Base a1/a2 是；v4 a2 解出（高方差）。

### T5. extract-elf（v4 a1）：ELF 段内取值逻辑错（能力回退）

- **运行/任务**：`extract-elf__Na2jk5y`，reward 0，13 步 / 70 命令，标 task_complete。对照：Base a1 解出，a2 没解出。
- **现象**：要写 JS 从 ELF 的 LOAD 段按 4 字节对齐抽取小端 32 位整数。v4 自己写了 Python 参考实现来对拍，前几次因为 ELF 魔数判断写成 `b'ELF'`（漏了 `\x7f`）一直崩，修好后 JS 和自己的 Python "MATCH"，于是标完成——但两者**用了同样错误的取值假设**，隐藏参考里有 97 处地址值不一致（`4192: (0, 4196274163)` …期望 0 实得非 0）。典型的"自己跟自己对拍通过 ≠ 对"。
- **证据**：
  ```
  verifier: "Found memory addresses with inconsistent values: {'4192': (0, 4196274163), …} / assert 97 == 0"（1 failed, 1 passed）
  step 12 analysis: "…magic check was incorrect (data[:4] != b'ELF' instead of b'\x7fELF')…" 修好后 diff2.py -> MATCH -> task_complete
  ```
- **根因**：MODEL（逻辑错 + 自写对拍替代官方验证）。**回退**：相对 Base a1 是。

### T6. financial-document-processor（v4 a1）：OCR 发票金额算错（能力回退）

- **运行/任务**：`financial-document-processor__CDvcNZf`，reward 0，48 步 / 215 命令，标 task_complete。对照：Base a1/a2 都解出，v4 a2 也解出。
- **现象**：要从 OCR 文本提取发票并汇总。v4 生成了 summary.csv、目录结构、"原始文档目录清空"等检查都过，但某张发票的 `total_amount` 算成 4382，期望 6558（差 2176），`test_summary_csv_content` 失败。
- **证据**：
  ```
  verifier: "Wrong total_amount for ced798b2…: expected 6558.0, got 4382 / assert 2176.0 < 0.01"（1 failed, 6 passed）
  ```
- **根因**：MODEL（抽取/求和逻辑错）。**回退**：相对 Base 是；v4 a2 解出（高方差）。

### T7. overfull-hbox（v4 a1）：替换了不在允许同义词表里的词（能力回退）

- **运行/任务**：`overfull-hbox__mvE4zmU`，reward 0，24 步 / 67 命令，标 3 次 task_complete。对照：Base a1 解出，a2 没解出。
- **现象**：任务要改 LaTeX 消除 overfull hbox，且**只能把词替换成 synonyms.txt 里给的同义词**。v4 的替换把 `horizon` 换成了 `edge`，但 synonyms.txt 里 `edge` 的同义词组是 `border, boundary, margin, rim, periphery`——`horizon` 不在其中，违反约束。编译成功、无 overfull，但 `test_input_file_matches` 判 `modified input.tex must only modify words in synonyms.txt`。
- **证据**：
  ```
  #14 sed -i 's/\bhorizon\b/edge/g' input.tex（say: "Replace 'horizon' with 'edge'…"）
  synonyms.txt: "edge, border, boundary, margin, rim, periphery"
  verifier: "modified input.tex must only modify words in synonyms.txt / 'horizon' == 'edge'"（1 failed, 3 passed）
  ```
- **根因**：MODEL（没严格遵守"只用给定同义词"的约束）。**回退**：相对 Base a1 是。

### T8. 摘要死循环（harness，不是模型失败，但 v4 命中更多）

- **运行/任务**：v4 a1 有 **18 个 trial** 陷入 Terminus-2 的 "Even fallback chat failed" 死循环（Base a1 11、a2 12）。其中 Base 做对、v4 a1 死循环的有 git-multibranch、mailman、build-pov-ray（这三题 v4 a2 又解出了）。
- **机制**：和 `failure_cases_draft.md` A1、README §11 第 3 条一致——prompt 顶到 `max_input_tokens=180224` 后，Terminus-2 的 unwind 按 litellm 的 token_counter 估算（偏少），但 vLLM 按 prompt + 81920 > 262144 拒绝（HTTP 400），fallback chat 反复失败、无退出条件，直到 3 h 超时。v4 a1 代理日志里 400 请求 8700 个（Base a1 7229）。死循环开始后几乎零有效动作：18 个 trial 里 15 个进入死循环前的 max prompt 都 ≈179K–180K token，进入后有效步 0–1。
- **证据**：
  ```
  git-multibranch__HaSWqwg/trial.log: "Context length exceeded"×261, "Even fallback chat failed"×260；进入死循环在 +46 min，之后 260 步全是占位
  llm_requests.jsonl（v4 a1）: status 400 ×8700
  ```
- **根因**：HARNESS。**不是模型失败**，但 v4 的长轨迹让它更容易把上下文顶到阈值，所以命中数比 Base 高（18 vs 11/12）。这些 trial 在主指标里按超时 0 分计入，口径与 Base 相同。

### T9. 内存看门狗杀容器（harness/环境，记 env_error）

- **运行/任务**：v4 a1 的 `rstan-to-pystan__HJuUUSu`、`sqlite-db-truncate__ApN9XfJ`、`tune-mjcf__L6TV3mC` 都因 `MemoryLimitExceededError`（单 trial 48G 上限，看门狗 95% 杀整容器）失败。这三题 Base 都解出过。
- **现象**：模型跑了重负载 / 多进程脚本撑爆内存——rstan-to-pystan 开了 129 个 python 进程做 4 链诊断、内存 1.4 GB/s 上涨；sqlite-db-truncate 卡在 Python heredoc 里反复发 `C-d`；tune-mjcf 跑 MuJoCo 参数搜索。按 README §9 的兜底规则（有异常且无 verifier 判定）记 env_error，主指标按 0 分。
- **证据**：
  ```
  rstan-to-pystan trial.log: "Memory explosion detected: 45153MB…~1.1s until limit / Memory limit exceeded: 47379MB > 46694MB (96%). Killing container."
  sqlite-db-truncate: 同机制 + 1 次 litellm.Timeout；step 47-48 反复 "C-d\n" 想退出 heredoc
  ```
- **根因**：HARNESS（singularity 用看门狗而非 cgroup 硬限，超限直接杀容器、无 verifier 判定），由模型的高内存脚本触发。与 Docker 官方环境不同（README §11 第 10 条）。

### T10. headless-terminal（v4 a1）：连接断开

- **运行/任务**：`headless-terminal__CSjT75v`，`RemoteProtocolError: Server disconnected without sending a response`，trial 失败。Base a1 解出、a2 没解出。
- **根因**：HARNESS（endpoint 在长请求时断开，httpx 抛 RemoteProtocolError）。不是模型失败。

---

## 6. 云上（Nebius）TB2：最终的同条件比较

Nebius 上的运行全部完成后（README §11 第 21 条），同一台 8×H100 Docker 节点上的 TB2 成功率：Base 47.19 / 42.70（2 次），E-TB v2 38.20，v4 41.57（a1 + a1b 合成），Mixed 30.34。配对 bootstrap（README §8.2）：v4 − Base −3.37 [−8.99, +2.25]（不显著）；E-TB v2 − Base −6.74 [−14.61, +0.56]；v4 − Mixed +11.24 [+3.37, +20.22]（McNemar 13:3，p=0.021）。

- 方向与集群 a1 一致：v4 略低于 Base 但不显著，§5 的 T1–T7 是任务级的个例，不是系统性崩溃。
- **v4 比 E-TB v2 单专家还高 3.4**：TB2 上合版没有稀释掉终端能力，反而略好；这和 SWE 上"专家能力被稀释"（B3、B6）方向相反。
- Mixed 在 TB2 上明显最差（−14.6），它用的是 v1 的 TB 数据（README §11 第 9 条），和 E-TB v1 的退步（24.7）一致。
- 一个 v4 trial（gpt2-codegolf）把终端录像写到 99 GB，属于失控输出，但 verifier 照常判分；超大文件没有拷回（`logs/eval/nebius/cloud_tb2_raw_excluded.txt`）。

---

## 7. ASI-Bench

### 7.1 harness / 环境案例（不是模型失败）

数字与 README §11 第 19–20 条一致，已核对 RunPod pod2 归档（本机 `xfer/base_asi.tgz`、`e_asi__asi__b3.tgz`、`mixed__asi__b3.tgz` 的 `tmp/opencode-state.*/…/opencode/log/opencode.log` 与各 `run/<task>/*__b3.json`）。

#### H1. opencode 子代理的 `external_directory` 权限请求在 `--auto` 下不自动放行（会话挂起）

- **现象**：opencode 主代理的权限在 `opencode run --auto` 下走 `action.action=allow`（日志里 `evaluated … action.action=allow` 两万多次，bash/read/edit/external_directory 都放行）。但当主代理派生**子代理**（`message=created … parentID=ses_…`）、子代理去访问工作区之外的目录时，会触发一次新的 `external_directory` 权限 **`ask`**，这个 ask 没有被自动解析——日志在 `message=asking id=per_… permission=external_directory` 这一行之后**直接结束**，会话再无后续事件，一直挂到 3 h 超时。
- **规模**：归档里"日志以一条未解析的 external_directory ask 结尾"的子代理会话：**Mixed 2 个、E-ASI 1 个**（Base 的子代理 ask 都正常解析掉了，0 个挂起）。合版 v4 有 1 个（pod1，未取归档，按 README §11 记）。绝大多数 ask 正常放行（external_directory ask 合计 Base 72、E-ASI 167、Mixed 168，几乎都 `evaluated`）。对应的 ASI 实例超时（`status=timeout`、`timeout_phase=process_execution`、`termination_signal=15`）：E-ASI 的 `biology.contextual_dna_regime_mapping`、Mixed 的 `physics.lbm_square_wake` 的工作区 id（`ai4sci_ws_72r2cjkw`）正是挂起会话所在。
- **证据**：
  ```
  mixed…/opencode-state.iU19G5/…/opencode.log 末行:
    message=asking id=per_14252217e001… permission=external_directory patterns="[\"/tmp/*\"]"   （之后无任何事件）
  experts…/opencode-state.tFJ1jp 末行: asking … external_directory patterns="[…ai4sci_ws_1t5l0vg9/*]"
  run/physics.lbm_square_wake/..b3.json: status=timeout, execution_time_seconds=10800.06, termination_signal=15
  ```
- **根因**：HARNESS（opencode 子代理的 external_directory 请求不在 `--auto` 的自动放行范围内）。**不是模型失败**。

#### H2. agent 在无 PID namespace 的 RunPod 上 `pkill` 杀掉评测进程

- **现象**：ASI agent 以 root 跑、与 vLLM/asibench 同一 PID namespace。pod2 上三次事件（opencode 日志里 `evaluated permission=bash pattern="pkill …python…" action.action=allow`）：
  - 10-15 18:38 EDT，一个 **Mixed** agent（run=cc176e6e）`pkill -9 -f python3`，杀掉 vLLM 启动进程；
  - 19:55 EDT，一个 **E-ASI** agent（run=99a413b3）`pkill -f python3`，pod2 两个 vLLM 同时退出；
  - 20:42 EDT（加 vLLM 防护后），一个 **Mixed** agent（run=a8ef7de0/4cbf7954）`pkill -f python`，杀掉两个 asibench 进程，两次运行提前结束。
- **证据**：
  ```
  mixed opencode.log: evaluated permission=bash pattern="pkill -9 -f python3" action.action=allow（run=cc176e6e, 10-15 22:38 UTC=18:38 EDT）
  e_asi opencode.log: pattern="pkill -f python3"（run=99a413b3, 23:55 UTC=19:55 EDT）
  mixed opencode.log: pattern="pkill -f python" ×3（run=a8ef7de0, 01:19–01:26 UTC）
  ```
- **处理 / 防护**：每次停整组、作废在跑的题、`--resume` 重跑；vLLM/llm_proxy/asibench 改用不含 "python" 的解释器链接（`vllm-srv`/`llm-proxy`/`asibench-rt`）启动。备份 `*.died_2238utc`、`*.killed_2355utc`、`*.killed_0042utc`。
- **根因**：HARNESS（RunPod 容器不允许 user namespace，无法隔离 PID）。**不是模型失败**。残留风险：同节点 agent 间仍可互杀，无法完全量化。

#### H3. 评分阶段的 harness 问题（对所有模型相同）

- **裁判 temperature**：asibench 让裁判用 temperature=0，官方 gpt-5.5 只接受默认值 1。前两次给 Base 评分时有 **8 题** `evaluation_invalid`。最终用 `sitecustomize.py` 去掉 gpt-5 系列调用的 temperature，统一重评。
- **`tasks` 导入路径**：`electrical_engineering.cmos_opamp_design` 的评分器以 `tasks.<领域>.<任务>` 导入辅助模块，需把 `$ASI_TASKS_ROOT` 加进 PYTHONPATH，否则崩溃（之前被裁判报错挡住）。
- **TSP 需 Docker**：`computer_science.euclidean_tsp_tour_optimization` 的评分要在 Docker 里跑提交解，集群无 Docker，所有模型这一题都 `evaluation_invalid`，归一化记 env_error（主指标按 0 分，每个模型同样扣）。
- **根因**：HARNESS。数字与 README §11 第 20 条一致。

### 7.2 E-ASI 的一个模型行为案例

#### H4. E-ASI / math.manifold_pde_meshfree：ContextOverflowError（模型行为）

- **现象**：E-ASI 在该实例上 `status=failed`、exec 5169 s。opencode 日志里 `AI_APICallError: This model's maximum context length is 262144 tokens. However, you requested 32000 output tokens and your prompt contains at least 230145 input tokens`（prompt 230K + 输出 32K > 262K）。模型把工作区文件 / 长脚本反复读进上下文直到溢出。同类 ContextOverflow 在 Base 上也出现过 4 题（cmos_opamp_design、causal_attention、gillespie_ssssa、lid_driven_cavity_flow，其中 causal_attention exec 达 10756 s），说明是模型侧的上下文管理问题，不是某个专家独有。
- **证据**：
  ```
  experts/asi/e_asi__asi__b3/run/math.manifold_pde_meshfree/*.json: status=failed, execution_time_seconds=5168.9
  base opencode.log: "ContextOverflowError: maximum context length is 262144 … prompt contains at least 230145 input tokens" ×（多次）
  ```
- **根因**：MODEL（上下文超限是模型行为，题目要求把这一类计为模型失败）。这里只列 E-ASI 一例；Base 的 4 例作对照。

### 7.3 v4 的 ASI 案例

v4 的 ASI B3 为 7.53（60 题，无效题计 0），Base 10.03，v4 − Base −2.50 [−6.79, +2.52]，不显著；E-ASI 7.75、Mixed 7.39（README §8.2）。

#### S1. v4 的 ASI 表现基本继承自 E-ASI（能力回退，专家导致）

- 逐题分数的 Pearson 相关：v4 与 E-ASI **0.67**，v4 与 Base 0.22，v4 与 Mixed 0.27，E-ASI 与 Base 0.13。
- v4 相对 Base 掉分最多的题，E-ASI 也几乎同样掉分：`math.symplectic_integrators` Base 42.8 → v4 0.0（E-ASI 0.1），`computer_science.stable_latent_state_fitting` 39.0 → 0.0（0.0），`robotics.particle_filter` 28.4 → 0.0（0.0），`robotics.lidar_imu_time_offset_yaw_calibration` 77.1 → 35.6（34.3）。
- 涨分也同步：`earth_science.farquhar_ballberry_photosynthesis` 0 → 100（E-ASI 100），`earth_science.snow_3layer_compaction` 3.0 → 36.0（36.0）。
- 结论：合版后 ASI 上的行为由 E-ASI 主导，而 E-ASI 本身就比 Base 差（−2.29），所以 v4 的 ASI 回退来自专家，不是合版冲突。

#### S2. 交付文件缺失，硬性检查不通过（能力回退）

- 通过 hard gate 的题：v4 32/60，Base 39/60。
- `stable_latent_state_fitting`、`biology.gene_network_timeseries`：agent 只调用了 1 次模型，第一轮回复里没有任何工具调用，opencode 随即结束；工作目录里只有题目给的 `data/` 和说明，`analysis.py`、`results/*.csv` 都没有（已在集群上核对工作目录，不是搬运时的排除造成的）。
- `physics.ks_2d_anisotropic`（12 次调用）、`robotics.particle_filter`（35 次调用，只写了 `analysis.py`，没有 `results/`）：同样因为缺交付文件判 0，而 Base 在这些题上都通过了 hard gate。
- "第一轮不调用工具就结束"并非 v4 独有：调用次数 ≤ 2 的题，Base 6 题、E-ASI 2 题、Mixed 1 题、v4 6 题；差别在于 v4 恰好落在了 Base 能拿分的题上。

#### S3. 轨迹最长、得分却更低（合版引入）

- 每题模型调用次数的中位数：v4 **81**，Base 46，E-ASI 19，Mixed 19。v4 既不像 Base，也不像在 agent 轨迹上训练过的 E-ASI / Mixed，是四个模型里最长的。
- 这与 TB2 上 v4 步数中位数约为 Base 2 倍（§2.3）、SWE/HLE 上的失控生成（A7、A8、D2）是同一个现象：合版后的模型更不容易收尾。

#### S4. `math.symplectic_integrators`：硬性检查通过，但说明里缺关键词，判 0（能力回退）

- v4 交付了文件、通过了 hard gate，但 soft gate 报 "Required pattern not found: symplectic|leapfrog|verlet|yoshida"：方法说明里没有写出所用积分器的名字，整题 0 分；Base 42.8。
- 这类"做了但没按交付契约写清楚"的失分，在 E-ASI 上同样出现（0.1），属于从专家继承的输出习惯。

---

## 8. 规律（附计数）

1. **"提交协议"被破坏（MERGE）**：v4 有 12/153 个 SWE trial 在提交前 `git commit`，其中 11 个补丁为空；Base、E2、Mixed 都是 0。7 道回退题上的 16 次失败里 4 次属此类。估算修好能挽回约 +1.3 到 +2.6 分，但空补丁本身对不对无法复评。
2. **失控 / 退化生成（MODEL 继承 E2，循环部分为 MERGE）**：
   - SWE 单次写满 81920 token 7 次（Base 0），单段 reasoning 内同一句重复 **399–963 次**（A7 修正为 963、A8 修正为 399）。
   - 环境正常时同一命令 / 同一按键重复 43–210 次（SWE A9/A10），TB2 把加密包解压尝试重复 115 次、把 `C-c` 当命令 20 次（T1）、写 33 个 debug 脚本不收敛（T2）。
   - HLE 上表现为 v4 finish=length 请求最多（370），但加入 E-TB v2 后思考长度较 v3 回升（中位 16.5K vs 12.1K），比 v3 情况缓和。
3. **长轨迹把风格带到终端（MERGE）**：TB2 v4 有效步数中位数 45，是 Base（23/26）的约 2 倍，每步命令 3.55 对 2.0；多出来的步几乎都在失败 / 超时任务上，成功时步数与 Base 相当。也因此更易命中摘要死循环（18 vs 11/12）。
4. **专家能力被稀释（MERGE）**：E2 做对的 11 道 SWE 题 v4 只做对 36%；teleport-eda6 三次对不上隐藏 API（E2 28/28）；openlibrary-53d3 返回 dict（E2 8/8）；HLE 校准只保留约 1/4。但正迁移也在（SWE flipt-524f 0/3→2/3，TB2 有 6 道 Base 全错而 v4 解出）。
5. **"自写测试 / 自我对拍通过 ≠ 官方测试通过"（MODEL，Base 也有）**：SWE B1、B2、B5、B6；TB2 T3、T5。B1 甚至删了已有测试文件，T5 跟自己的 Python 实现对拍"MATCH"就提交。
6. **补丁卫生（MERGE）**：带 `.bak`/`.backup` 的补丁占 26%（Base 2%）。不影响判分。
7. **harness 层问题**：
   - SWE 补丁末尾换行被 `stdout.strip()` 吃掉，每次运行 11–32 个 trial 补丁 malformed、最后一个文件整体丢失（C4，影响所有模型，已确认 navidrome、flipt-e2bd 结果因此翻转）。
   - 客户端 600 s 超时导致无上限重试（C5）。
   - TB2 摘要死循环（T8）、内存看门狗杀容器（T9）、连接断开（T10）。
   - ASI 子代理权限挂起（H1）、同 PID namespace 被 pkill（H2）、裁判 temperature / 导入路径 / Docker（H3）。
   - HLE v4 生成首轮 Connection error、第 2 轮补 2185 题，把"无预测"从 24 推到 40（D2）。
8. **不是问题的部分**：SWE 格式错误 v4 比 Base 少；时间预算耗尽只 5 次（E2 14 次）。TB2 的 E-TB v1 症状（把 `C-x` 当命令、约 300 轮 heredoc 循环）在 v4 a1 上基本消失（只 2 个 trial 有少量 `C-x` 误发）。

**按根因计数**：

| 根因 | 数量 | 案例 |
|---|---|---|
| MERGE | 15 | A1–A6、A9、A10、A12、B3、B6、D3、D5、T1、T2 |
| MODEL | 15 | A7、A8、A11、B1、B2、B4、B5、D1、D4、T3、T4、T5、T6、T7、H4 |
| HARNESS | 9 | C2、C3、C4、T8、T9、T10、H1、H2、H3 |
| MIX | 3 | C1、C5、D2 |

合计 15 + 15 + 9 + 3 = 42，与 §1 一致。

**明确的"合版引入的回退"（题目要求项）**：A1、A2、A3、A4、A9（commit / 退化循环，Base 或 E2 能做对）、B3、B6（专家能力稀释）、D3、D5（HLE 校准 / 类别）、T1、T2（TB2 退化循环 / 不收敛，Base 能做对）、S3（ASI 轨迹最长、得分更低）——共 **12 个**，分布在全部四个基准上，全部发生在最终合版 v4。

**按基准计数**：SWE 23、HLE 5、TB2 10、ASI 8 = 46。**在 v4 上的案例 ≥ 20** 这一硬性要求满足（SWE 23 + HLE 5 + TB2 10 + ASI 4 = 42 个在 v4 上）。

---

## 9. 范围与口径（数据来源与代理说明）

- **已用实测、非代理**：
  - SWE：v4 r1/r2/r3 全部 trial（agent + regrade），Base 3 次，E2/Mixed 各 1 次。
  - HLE：**v4 实际判分**（`merged_v4__hle__s1`，2459/2500 判分，1 pending、40 无预测），替换了 `failure_cases_v4.md` 的 v3 代理。所有配对统计用 `summarize.py compare --bench hle`（env=zero，10000 bootstrap，seed 0）。
  - TB2：v4 集群 a1（快照时 87/89 判分，最终 89/89），对照 Base 集群 a1/a2（各 89/89）；Nebius 同条件运行全部完成（§6）。
  - ASI：四个模型在 RunPod（pod1、pod2）上的运行归档，以及集群上的本地评分（§7）。
- **仍是代理或部分数据**：
  - HLE 的 v3 列只作对照（去掉 E-TB v2 的消融）。
  - TB2 v4 a1 的 2 道题（winning-avg-corewars、write-compressor）快照时未判分，没有做逐题分析（最终都是 timeout）；v4 a2 为部分运行，只作上下文。Base 的 TB2 取消计数因日志时间差不可比（§2.3 脚注）。
- **每个数字都来自本轮只读抽查**。未能核实项：
  1. SWE 的空补丁（A1–A6）内容是否正确无法复评（只读，不能重 regrade）。
  2. C4 的"整个最后一个文件都没应用"是据 C3 推断，未在容器内复现。
  3. HLE 40 题无预测里截断 / 超时 / 首轮未落盘各占多少没逐条对齐（D2）；v4 HLE 生成首轮为何大面积 Connection error 未查（可能 endpoint 中断 / 抢占）。
  4. commit、`.bak` 习惯的来源仍只有训练数据 grep 的旁证（E-ASI 系统提示）；Mixed 用同份数据却不 commit，与假设矛盾。
  5. TB2 单次重复（v4 a1 一次、Base 两次），任务级"回退"受采样噪声影响大（build-pmars、crack-7z-hash 等 v4 a1 失败、a2 又成功），单题结论只能当定性证据。
  6. 只读、不改集群 / VM，不提交作业、不杀进程。

---

## 附录：历史 Base / v1 专家案例

Base、E-TB v1、E-SWE v1 的 32 个历史案例（TB2 摘要死循环、E-TB v1 行为漂移、E-SWE v1 的 OpenHands 模板化与临时文件残留、Base 的 HLE 过度自信等）保留在 **`analysis/failure_cases_draft.md`**，不并入本文主列表。本文的 v4 案例在编号和机制上与之对照（如本文 A7/A8 ↔ draft B2/B7 的思考失控，T8 ↔ draft A1 的摘要死循环，B1 ↔ draft D1 的删测试，T2/T3 ↔ draft C1/C2 的"没验证就宣称完成"），两份草稿 `failure_cases_v4.md`、`failure_cases_draft.md` 均保留，未删除。
