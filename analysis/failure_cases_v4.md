# 失败案例目录：最终合版 merged_v4

> **历史快照（2026-10-15），已被 [`failure_cases.md`](failure_cases.md) 取代。** 这里的数字基于当时未跑完的运行，最终结果以 `failure_cases.md` 和 README §8 为准。只为保留分析过程而保留本文件。

> 状态：基于 2026-10-15 白天的集群只读快照。TB2 和 ASI 的 v4 运行还没跑完，HLE 的 v4 生成也没跑完（见 §0、§5）。
> 对象：最终合版 **merged_v4**，即 E-SWE v2、E-TB v2、E-ASI、E-HLE 四个 LoRA 专家的 TIES 合并（`ties_l1_d0.5_v4`）。
> 证据路径如无特别说明，均相对于 `/scratch/eh3323/apex/results`。缩写见 §0.2。
> HLE 遵守防污染规则：只列题目 id、学科类别、答案类型、是否带图、判分结论、token 数、finish_reason 和置信度，不含题干、标准答案或模型回答文本。
> 摘录每段不超过 5 行，只取自轨迹里的命令、命令输出和测试输出。
> 格式沿用 `failure_cases_draft.md`（草稿 v0，对象是 Base / E-TB v1 / E-SWE v1）。

## 0. 范围、口径与缩写

### 0.1 数据范围

| 基准 | 纳入的运行 | 说明 |
|---|---|---|
| SWE-bench Pro HARD-51 | **v4**：`merged_v4__swepro__r1/r2/r3`；**Base**：`base__swepro__r1/r2/r3`；**E-SWE v2**：`e_swe_v2__swepro__r1`；**Mixed**：`mixed__swepro__r1` | mini_toolcall（单个 bash 工具），250 步上限，50 分钟预算，agent 与 regrade 分成两个 job。本轮 51 道题都能起沙箱：草稿里那 10 道 fakeroot 坏题已修好，Base 和 v4 的 env_error 都是 0 |
| HLE | **v3 代理**：`merged__hle__s1`；Base `base__hle__s1`、E-HLE `e_hle__hle__s1`、Mixed `mixed__hle__s1` | **v4 的 HLE 还在生成（未判分），所以 HLE 部分用 v3 合版代理。** v3 与 v4 的配方相同，只少了 E-TB v2 这一项。v3 结论能否推到 v4，要等 v4 判分后核对（§5） |
| TB2 / ASI | 无 | v4 的运行尚未结束，见 §5 |

### 0.2 缩写

- 运行：**V1/V2/V3** = merged_v4 的 r1/r2/r3；**B1/B2/B3** = Base 的 r1/r2/r3；**E2** = E-SWE v2；**MX** = Mixed（四份数据混训的单模型）。
- 路径：`V1:A` = `merged/swepro/merged_v4__swepro__r1/jobs/merged_v4__swepro__r1__agent`，`V1:G` = 同一运行的 `…__regrade`。`V2:*`、`V3:*`、`B1:*`…`B3:*`、`E2:*`（`experts/swepro/e_swe_v2__swepro__r1/jobs/e_swe_v2__swepro__r1__*`）、`MX:*` 依此类推。
- trial 目录下：轨迹在 `agent/mini-swe-agent.trajectory.json`，补丁在 `agent/model.patch`，regrade 输出在 `verifier/run-script-stdout.txt`、`verifier/test-stdout.txt` 和 `trial.log`。
- **根因类别**：
  - **MERGE**：合版引入。Base 和对应专家都没有（或明显更少），只在 v4 上出现；"专家能力被稀释"也归入这一类。
  - **MODEL**：模型行为，但 Base 或对应专家也有，v4 是继承下来的。
  - **HARNESS**：评测框架或环境问题，与模型无关。
  - **MIX**：harness 机制与模型行为共同造成。
- **回退**：与对照（Base，或对应专家：SWE 看 E-SWE v2，HLE 看 E-HLE）相比，同一道题对照能做对而 v4 做不对。SWE 用"3 次里成功几次"判断，E-SWE v2 只有 1 次重复。

---

## 1. v4 输在哪里（低成本配对统计）

### 1.1 SWE-Pro HARD-51：总分

| 模型 | 分数（重复次数） | 成功 trial 数 | 与 v4 配对比较（`summarize.py compare`，paired bootstrap） |
|---|---|---|---|
| Base | 17.65（19.61 / 11.76 / 21.57） | 27/153 | v4 − Base = **−4.58**，95% CI [−11.76, +1.96]，p = 0.165 |
| E-SWE v2 | 21.57（1 次） | 11/51 | v4 − E2 = **−8.50**，95% CI [−18.95, +0.65]，p = 0.061 |
| Mixed | 13.73（1 次） | 7/51 | v4 − MX = −0.65，p = 0.79 |
| **merged_v4** | **13.07**（15.69 / 15.69 / **7.84**） | 20/153 | 两个比较都不显著（n = 51，方差大），但方向一致 |

### 1.2 逐题配对

**Base 3 次中做对 ≥2 次、v4 3 次中做对 ≤1 次（7 道题）**，以及 v4 每次失败的归因（见 §3）：

| 任务 | Base | v4 | E2 | v4 失败的原因（逐次） |
|---|---|---|---|---|
| element-web-ecfd | 2/3 | 0/3 | 解出 | V1 部分修复 25/26（B5）· V2 容器 OOM 后反复提交，补丁丢失（C1）· V3 部分修复 25/26（B5） |
| flipt-e2bd | 2/3 | 0/3 | 解出 | V1、V2 先 `git commit` 再提交，补丁为空（A1、A2）· V3 补丁被 harness 截断，整个 cache.go 没应用上（C3） |
| vuls-5af1 | 3/3 | 1/3 | 解出 | V1 删除已有测试文件并改了语义（B1）· V3 选错了内核版本（B2） |
| vuls-ca3f | 2/3 | 1/3 | 解出 | V1 离线环境里把 `go mod download` 重复了 132 次，最终 79/80（A9）· V3 74/80（未展开） |
| teleport-f432 | 2/3 | 1/3 | 未解出* | V1、V3 raw keystore 实现有误（B4） |
| openlibrary-0dc5 | 3/3 | 1/3 | 未解出 | V2 先 commit 再提交，补丁为空（A3）· V3 59/61（未展开） |
| navidrome-bf2b | 2/3 | 1/3 | 未解出* | V2 补丁被 harness 截断，gg.go 没应用上（C2）· V3 先 commit 再提交，补丁为空（A4） |

\* E2 在这两题上的失败本身来自 harness 截断（C4），所以和专家的比较不成立。

**反方向，Base ≤1/3 而 v4 ≥2/3（2 道题）**：flipt-524f（0/3 → 2/3，E2 也做对了，可以看作 E-SWE v2 的正迁移）和 tutanota-f373（1/3 → 2/3）。

**E-SWE v2 做对、v4 没做对**：E2 做对的 11 道题里，v4 只成功 12/33（36%）；在 E2 没做对的 40 道题上，v4 成功 8/120（6.7%）。其中 v4 三次全错的有 4 道：element-web-ecfd、flipt-e2bd、**teleport-eda6（v4 三次都和隐藏测试的 API 对不上，B3）**、**openlibrary-53d3（B6）**。只做对一次的有 vuls-5af1、vuls-ca3f、openlibrary-630。

**7 道回退题上，v4 失败的 16 个 trial 按原因归类**：

| 原因 | trial 数 |
|---|---|
| commit 导致补丁为空 | 4 |
| harness 截断补丁 | 2 |
| OOM 后的死循环 | 1 |
| 逻辑错误或只修了一部分 | 9 |

**换句话说，有 7/16 不是"不会做"。** 不过补丁为空的 4 次，即使不 commit，补丁本身对不对也没法验证：集群只读，不能重新判分。

### 1.3 SWE 行为指标（逐次运行；括号内依次为 r1/r2/r3）

| 指标 | Base | **v4** | E-SWE v2 | Mixed |
|---|---|---|---|---|
| 工具调用总数 | 5391 / 4785 / 5138 | **7347 / 7217 / 6852** | 7306 | 5378 |
| 每次调用 completion token 中位数（代理日志） | 71 / 76 / 75 | **142 / 167 / 132** | 303 | 77 |
| completion token 总量 | 1.32M / 1.36M / 1.38M | **2.59M / 2.87M / 2.71M** | 5.76M | 1.66M |
| 单次生成超过 600 s、被客户端取消的请求 | 0 / 0 / 0 | **14 / 13 / 41** | 23 | 1 |
| finish=length（单次 81920 token 写满） | 0 / 0 / 0 | **1 / 2 / 4** | 5 | 0 |
| 补丁为空（Submitted，0 字节） | 0 / 0 / 0 | **3 / 5 / 5** | 3 | 0 |
| 在 `/app` 中执行 `git commit` 的 trial | 0 / 0 / 0 | **3 / 3 / 6** | 0（另有 1 次在 /tmp 的临时仓库里） | 0 |
| 补丁带 `.bak`/`.backup` 文件的 trial | 1 / 0 / 2 | **14 / 8 / 11** | 2 | 10 |
| 补丁带 reproduce/debug/update_* 临时脚本的 trial | 0 / 0 / 0 | 2 / 1 / 3 | 6 | 19 |
| 容器 OOM（补丁丢失）的 trial | 4 / 2 / 2 | 4 / 5 / 4 | 4 | 5 |
| 环境正常时同一条命令重复 ≥20 次 | 0 / 0 / 0 | **2 / 1 / 0** | 0 | 1（未核实是否与 OOM 相关） |
| 格式错误（没有 tool call）的 trial | 8 / 6 / 7 | 2 / 6 / 6 | 6 | 4 |
| 退出方式：步数上限 / 时间预算 / 上下文溢出 | 4/0/0，2/0/0，2/0/0 | 8/0/0，8/1/1，5/4/0 | 2/14/3 | 5/0/0 |
| regrade 中补丁"malformed"的 trial（C4） | 11 / 15 / 20 | 22 / 18 / 23 | 11 | 32 |

要点：

- v4 的输出量约为 Base 的 2 倍，但只有 E-SWE v2 的一半。它继承了 E2 的长输出和失控生成，但时间预算耗尽只有 5 次（E2 是 14 次）。
- 格式错误**不是** v4 的问题，它比 Base 还少。
- 合版引入的新现象有三类：`git commit`、`.bak` 备份文件、环境正常时的重复命令死循环。

### 1.4 HLE（v3 合版代理）：按分组的准确率差

总分：Base 20.81、E-HLE 16.92、Mixed 16.60、v3 17.36。配对比较：

- v3 − Base = **−3.44**，CI [−5.0, −1.84]，McNemar 不一致对 245 : 159，p = 2e-5。
- v3 − E-HLE = +0.44，不一致对 194 : 205，p = 0.62。
- v3 − Mixed = +0.76，p = 0.35。

| 分组 | n | Base | E-HLE | Mixed | v3 | v3−Base | v3−E-HLE | 不一致对 Base : v3 |
|---|---|---|---|---|---|---|---|---|
| Math | 1021 | 29.3 | 23.6 | 25.1 | 25.2 | −4.1 | +1.6 | 125 : 83 |
| Biology/Medicine | 280 | 17.5 | 14.3 | 12.5 | 12.9 | −4.6 | **−1.4** | 24 : 11 |
| Chemistry | 165 | 16.4 | 11.5 | 9.1 | 11.5 | −4.8 | 0.0 | 16 : 8 |
| Engineering | 111 | 17.1 | 12.6 | 11.7 | 12.6 | −4.5 | 0.0 | 10 : 5 |
| Physics | 230 | 16.1 | 16.5 | 13.5 | 13.5 | −2.6 | **−3.0** | 20 : 14 |
| Other | 233 | 9.0 | 8.6 | 6.9 | 6.4 | −2.6 | **−2.1** | 15 : 9 |
| CS/AI | 241 | 15.8 | 14.1 | 13.3 | 14.5 | −1.2 | +0.4 | 17 : 14 |
| Humanities/Social | 219 | 13.7 | 7.8 | 7.8 | 12.3 | −1.4 | **+4.6** | 18 : 15 |
| exactMatch | 1909 | 19.7 | 16.1 | 15.5 | 16.2 | −3.6 | +0.1 | 178 : 110 |
| multipleChoice | 591 | 24.2 | 19.6 | 20.1 | 21.2 | −3.0 | +1.5 | 67 : 49 |
| 带图 | 342 | 15.5 | 11.4 | 10.2 | 11.1 | **−4.4** | −0.3 | 30 : 15 |
| 不带图 | 2158 | 21.6 | 17.8 | 17.6 | 18.4 | −3.3 | +0.6 | 215 : 144 |
| Base 输出长度四分位 Q1（最短） | 626 | 14.9 | 13.4 | 12.0 | 13.4 | −1.4 | 0.0 | 43 : 34 |
| Q2 | 625 | 24.5 | 21.1 | 19.0 | 20.8 | −3.7 | −0.3 | 61 : 38 |
| Q3 | 625 | 26.6 | 19.4 | 20.8 | 20.2 | **−6.4** | +0.8 | 79 : 39 |
| Q4（最长） | 624 | 17.3 | 13.8 | 14.6 | 15.1 | −2.2 | +1.3 | 62 : 48 |

| 其他指标 | Base | E-HLE | Mixed | v3 |
|---|---|---|---|---|
| 输出 token 中位数 / 均值 | 24.6K / 27.1K | 13.2K / 19.4K | 13.2K / 18.6K | **12.1K / 18.5K** |
| 代理日志中 finish=length 的请求比例 | 62/2556（2.4%） | 107/3101（3.5%） | 87/3194（2.7%） | **233/3096（7.5%）** |
| 3 轮补跑后仍无预测的题数 | 3 | 4 | 6 | **24**（其中 20 道 Math） |
| 生成各轮的待跑题数（`logs/<run>/hle_pred.log`） | 2500 → 42 → 14 | 有重启，见日志 | — | 2500（有 1 次重启）→ **382** → 55 |
| 自报置信度均值 | 97.3 | 54.6 | 55.2 | **80.9** |
| judge 给出的校准误差 | 73.9 | 35.9 | 36.2 | **61.3** |
| 置信度 ≥95 却答错的题数 | 1803 | 47 | — | **503** |

---

## 2. 案例汇总（28 个）

| # | 基准 | 运行 / 任务 | 根因 | 相对 Base 回退？ | 相对专家回退？ | 一句话 |
|---|---|---|---|---|---|---|
| A1 | SWE | V1 / flipt-e2bd | MERGE | 是（2/3） | 是（E2 解出） | 提交前先 `git add && git commit`，工作区变干净，导出的补丁是 0 字节 |
| A2 | SWE | V2 / flipt-e2bd | MERGE | 是 | 是 | "All tests pass. Now we need to create a final commit"，补丁 0 字节 |
| A3 | SWE | V2 / openlibrary-0dc5 | MERGE | 是（3/3） | 否 | 测试全过之后 commit，补丁为空 |
| A4 | SWE | V3 / navidrome-bf2b | MERGE | 是（2/3） | 否* | 已经看到 `git diff --cached` 为空，改看 `HEAD~1`，照样提交 |
| A5 | SWE | V1 / protonmail-715d | MERGE | 否（都是 0） | 否 | commit 信息里的反引号被 shell 执行，又用 `--amend` 修 commit 信息 |
| A6 | SWE | V3 / qutebrowser-e70f | MERGE | 否 | 否 | 把"提交补丁"理解成"提交一个带说明的 commit"；同类共 11 个空补丁 |
| A7 | SWE | V3 / NodeBB-8168 | MODEL（继承 E2） | 否 | 否 | 最后一轮先有 5 次请求在 600 s 被取消，第 6 次写满 81920 token；同一句话重复 481 次，50 分钟预算耗尽 |
| A8 | SWE | V3 / element-web-772d | MODEL（继承 E2） | 部分（1/3） | 否 | 6 次取消后第 7 次写满 81920 token，同一句重复 198 次，预算耗尽 |
| A9 | SWE | V1 / vuls-ca3f | MERGE | 是（2/3） | 是 | 断网沙箱里把同一条 `go mod download -x` 执行 132 次，直到 250 步上限；reasoning 写"换个代理"，动作却没变 |
| A10 | SWE | V2 / flipt-2ca5 | MERGE | 否（都是 0） | 否 | 同一条 grep（每次返回 rc=1 和空输出）重复 210 次，一行代码没改 |
| A11 | SWE | V2 / vuls-ef2b | MODEL（E2 也溢出） | 否 | 否 | "Now I have a good understanding… Let's start implementing" 说了 136 次，0 次编辑，上下文写满 262K |
| A12 | SWE | v4 全部 / 补丁卫生 | MERGE | 不影响得分 | 不影响得分 | 33/127 个补丁带 `.bak`/`.backup` 文件（Base 3/145，E2 2/44）；tutanota 的补丁有 344 KB |
| B1 | SWE | V1 / vuls-5af1 | MODEL | 是（3/3） | 是 | 删掉已有的 `scanner/utils_test.go`，改了函数语义来迁就自写的调试测试，官方的 old_redhat 用例失败 |
| B2 | SWE | V3 / vuls-5af1 | MODEL | 是 | 是 | 自己跑的测试都通过，隐藏测试中多内核场景选错了版本（427.18.1 而不是 427.13.1） |
| B3 | SWE | V1–V3 / teleport-eda6 | MERGE（专家能力稀释） | 否（Base 0/3） | **是（E2 28/28）** | 三次分别缺 `kubeClusterEndpoint` 类型、`dial` 方法、`kubeAddress` 字段，测试包编译失败，0/28 |
| B4 | SWE | V1、V3 / teleport-f432 | MODEL | 是（2/3） | 不可比* | raw keystore：一次报 "no usable SSH key pair"，一次报 "crypto/rsa: verification error" |
| B5 | SWE | V1、V3 / element-web-ecfd | MODEL | 是（2/3） | 是 | 只修了一部分（25/26）；靠自写、随后删掉的 FinalVerify 测试做"验证" |
| B6 | SWE | V1–V3 / openlibrary-53d3 | MERGE（专家能力稀释） | 否（Base 0/3） | **是（E2 8/8）** | 已有测试全过，隐藏测试要求返回对象，v4 返回 dict（`'dict' object has no attribute 'key'`） |
| C1 | SWE | V2 / element-web-ecfd | MIX | 是 | 是 | 对整个 `test/components` 目录跑 jest，内存在 1 秒内冲到 4.6 GB，容器被杀；之后提交命令重复 102 次，补丁丢失 |
| C2 | SWE | V2 / navidrome-bf2b | HARNESS | 表面上是 | — | 补丁末尾缺换行，导致"malformed patch"，gg.go 没被应用，`undefined: gg.V`；跨运行 8 个 trial 的规律完全吻合 |
| C3 | SWE | V3（以及 B3）/ flipt-e2bd | HARNESS | 表面上是 | — | 同一个 bug 使整个 cache.go 没被应用（`setJSON undefined`），而 setJSON 定义在 cache.go 中间的 hunk 里 |
| C4 | SWE | 全部运行 | HARNESS | — | — | patch 捕获时 `stdout.strip()` 去掉了末尾换行：每次运行有 11–32 个 trial 的补丁是 malformed；凡是新建文件恰好排在最后的解法都会整体丢失 |
| C5 | SWE | V3 / 多题 | MIX | — | — | 客户端 600 s 超时与长生成叠加：r3 有 41 个请求被取消后重试；r3 的 4 次时间预算耗尽都由此造成 |
| D1 | HLE（v3） | Base 做对、v3 做错的 245 题 | MODEL（继承 E-HLE） | 是 | 否 | 思考长度减半（中位数为 Base 的 0.46 倍），损失集中在中长推理题（Q3 −6.4） |
| D2 | HLE（v3） | 24 道无预测题 | MERGE | 轻微 | 是 | 思考失控截断：finish=length 占 7.5%（Base 2.4%、E-HLE 3.5%），第 2 轮补跑 382 题 |
| D3 | HLE（v3） | 校准 | MERGE | 改善 | **是** | E-HLE 的校准能力在合版后只剩约 1/3：校准误差 61.3（E-HLE 35.9），置信度 ≥95 却答错 503 题（E-HLE 47） |
| D4 | HLE（v3） | 带图的 342 题 | MODEL（继承 E-HLE） | 是（−4.4） | 否（−0.3） | 带图题的输出 token 中位数从 19.1K 降到 7.0K，和只用纯文本训练的 E-HLE 一样 |
| D5 | HLE（v3） | Physics / Other / Bio-Med | MERGE | 是 | **是** | 这三个类别上 v3 比 E-HLE 还低 1.4–3.0 分，而 Humanities 反而高 4.6 分 |

\* 见 §1.2 表注。

**按根因计数**：

| 根因 | 数量 | 案例 |
|---|---|---|
| MERGE | 14 | A1–A6、A9、A10、A12、B3、B6、D2、D3、D5 |
| MODEL | 9 | A7、A8、A11、B1、B2、B4、B5、D1、D4 |
| HARNESS | 3 | C2–C4 |
| MIX | 2 | C1、C5 |

**按基准计数**：SWE 23 个，HLE 5 个。

**涉及得分回退的案例**：

- 相对 Base：A1–A4、A8（部分）、A9、B1、B2、B4、B5、C1、D1、D4、D5。C2 和 C3 只是表面上的回退。
- 相对专家：A1、A2、A9、B1–B3、B5、B6、C1、D2、D3、D5。

---

## 3. 案例详述

### A. SWE：合版引入的新行为，以及从专家继承的失控生成

#### A1. flipt-e2bd（V1）：先 commit 再提交，补丁为 0 字节
- **运行/任务**：agent `V1:A/instance_flipt-io__flipt-e2bd19d__U66cLKp`，regrade `V1:G/instance_flipt-io__flipt-e2bd19d__koGehMV`。对照：B1、B2、E2 都做对（10/10）。
- **现象**：做完修改后，第 126 条命令执行 `git add … && git commit`，第 127 条确认 "working tree clean"，第 128 条提交。harness 用 `git add -A && git diff --cached` 捕获补丁（`repo/src/eval/swepro_host_agent.py:95`），比较的是 HEAD，而 HEAD 已经包含了模型的 commit，所以 model.patch 是 0 字节，regrade 0/10。reasoning 原话："before that, we need to make sure the changes are staged and committed"。
- **证据**：
  ```
  #126 'cd /app && git add internal/cache/cache.go internal/cmd/grpc.go … && git commit …'
       -> "[main 8a8a3228] Add do-not-store flag lookup cache support and rename evaluation cache interceptor"
  #127 'cd /app && rm -f …middleware.go.bak …middleware_test.go.bak && git status' -> "nothing to commit, working tree clean"
  agent/result.json: "exit_status": "Submitted", "n_calls": 128   |  model.patch: 0 bytes
  regrade test-stdout: Required tests: 10 / passed: 0
  ```
- **根因**：MERGE。Base 和 Mixed 的 153+51 条轨迹里 `git commit` 出现 0 次，E2 在 `/app` 中也是 0 次；v4 有 12/153 个 trial commit。**假设（未验证）**：E-ASI 训练集 868/868 条样本的系统提示里都有 `git add . && git commit -m "message"` 的示例（`data/processed/asi/train.jsonl`，路径相对 /scratch/eh3323/apex）。但 Mixed 也用了这份数据，却从不 commit，所以这个解释还不完整。mini_toolcall 的题面并没有禁止 commit。
- **回退**：相对 Base 和 E2 都是回退。

#### A2. flipt-e2bd（V2）："All tests pass. Now we need to create a final commit"
- **运行/任务**：`V2:A/instance_flipt-io__flipt-e2bd19d__4iz9KUD`，regrade `V2:G/…__oajsgCy`。
- **现象**：机制与 A1 相同。模型自己的验证脚本输出 "All tests passed!" 之后 commit，再用 `git diff --stat HEAD~1` 看到了改动，就以为补丁没问题。
- **证据**：
  ```
  #102 'cd /app && git add internal/cache/cache.go … ' -> "[main bc6f3c5a] feat: add do-not-store context key and cache control interceptor"
  #103 'cd /app && go build ./... 2>&1 | head -5 && /tmp/verify_bin 2>&1 | tail -1' -> "All tests passed!"
  #104 'cd /app && git log --oneline -1 && git diff --stat HEAD~1'
  #105 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'        (model.patch: 0 bytes; regrade 0/10)
  ```
- **根因**：MERGE。**回退**：相对 Base 和 E2 都是。补丁内容本身是否正确无法复评。

#### A3. openlibrary-0dc5（V2）：测试全过之后 commit，补丁为空
- **运行/任务**：`V2:A/instance_internetarchive__openli__7Tgh5xR`，regrade `V2:G/…__cKUk3MJ`。Base 3/3 都做对。
- **证据**：
  ```
  #165 git commit -m "feat: support alternate-script author names via MARC 880 fields"
       (reasoning: "We already ran all marc tests and they passed. Good. Now, we need to create a commit with our changes.")
  #166 'cd /app && git status' -> "nothing to commit, working tree clean"
  #169 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'   |  regrade: Required tests: 61 / passed: 55（基线状态的通过数）
  ```
- **根因**：MERGE。**回退**：相对 Base 是（3/3 → 这次 0）；相对 E2 不是（E2 44/61，也失败）。

#### A4. navidrome-bf2b（V3）：看到暂存区 diff 为空，仍然提交
- **运行/任务**：`V3:A/instance_navidrome__navidrome-bf__oyGaAfJ`，regrade `V3:G/…__vbq5nRf`（0/3）。
- **现象**：commit 之后，模型执行 `git diff --cached --stat` 得到空输出，这正是 harness 捕获补丁的方式。它的结论是"要用 `git diff HEAD~1` 才能看到改动"，然后照常提交。这说明模型并不知道 harness 是怎样取补丁的。
- **证据**：
  ```
  #148 'cd /app && git diff --cached --stat' -> ""        (say: "The commit is created. Now, let's verify the changes…")
  #149 'cd /app && git diff HEAD~1 --stat' -> "core/artwork/artwork_internal_test.go | 8 +++---…"
  #151 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
  ```
- **根因**：MERGE。**回退**：相对 Base 是（B1、B3 做对；B2 的失败是 harness 造成的，见 C2）。

#### A5. protonmail-715d（V1）：commit 信息被 shell 执行，再 amend
- **运行/任务**：`V1:A/instance_protonmail__webclients__cvVxYHG`（补丁 0 字节）。
- **现象**：`git add -A && git commit -m "…"` 的信息里带有反引号，shell 把其中的 `X-Pm-Encrypt-Untrusted` 当命令执行了。模型发现 "commit message got mangled" 后，又执行了 `git commit --amend`。它花力气维护 commit 历史，却不知道这样做会让补丁变空。
- **证据**：
  ```
  #179 'cd /app && git add -A && git commit -m "Improve encryption handling for WKD contacts …'
       -> "bash: line 1: X-Pm-Encrypt-Untrusted: command not found"
  #180 'cd /app && git commit --amend -m "…"' -> "[main 35ee66dbb] Improve encryption handling …"
  #182 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
  ```
- **根因**：MERGE。没有造成得分回退（Base 0/3），但属于同一种失败机制。

#### A6. qutebrowser-e70f（V3）："submit the patch with a commit message"，以及同类全表
- **运行/任务**：`V3:A/instance_qutebrowser__qutebrowse__5UX9tGj`（补丁 0 字节）。
- **证据**：
  ```
  #73 'cd /app && git add qutebrowser/misc/guiprocess.py tests/unit/misc/test_guiprocess.py && git commit -m "Improve process outcome messages…"'
      say: "Now we'll submit the patch with a commit message describing the changes."
  #74 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
  ```
- **全部 commit 案例**：

  | 运行 | 任务（补丁字节数） |
  |---|---|
  | V1 | flipt-c188（0）、flipt-e2bd（0）、protonmail-715d（0） |
  | V2 | ansible-d720（0）、flipt-e2bd（0）、openlibrary-0dc5（0） |
  | V3 | teleport-7744（0）、teleport-2b15（0）、navidrome-bf2b（0）、qutebrowser-3e21（0）、qutebrowser-e70f（0）、flipt-c188（3830，commit 之后又改了文件，只保留了 commit 之后的改动） |

  合计 12 个 trial，其中 11 个补丁为空。v4 的 13 个空补丁里，另外 2 个见 A10 和 A11。
- **根因**：MERGE。在本数据集中，这是 v4 SWE 得分下降最直接、也最容易修的原因。可以在推理时加一条系统提示"不要 commit"；但要和 Base 公平比较，协议必须保持不变，所以应作为合版缺陷写进报告，不要私下修。

#### A7. NodeBB-8168（V3）：一次思考写满 81920 token，同一句话重复 481 次
- **运行/任务**：`V3:A/instance_NodeBB__NodeBB-8168c6c4__HDuwNsG`。Base 也是 0/3。
- **现象**：最后一轮（第 49 次调用）在代理日志中的过程如下：
  - 同一请求（n_messages = 122）先后 5 次各跑满 600 s，被客户端取消；5 次的起始时间依次相差 600 s。
  - 第 6 次在 598 s 内写满 81920 token（finish = length）。reasoning 有 343,615 个字符，content 为空。结尾附近的一个 60 字符短语在这段 reasoning 里出现了 **481 次**，从第 1115 个字符就开始重复。
  - agent 在 2940 s 时就已耗尽预算（in-flight call abandoned）。这条响应是截止之后才返回并写进轨迹的，harness 把它判为 FormatError（没有 tool call）。

  按单流解码约 137 tok/s 计算，写满 81920 token 正好需要约 600 s。所以前 5 次被取消的请求，**很可能也是同样的失控生成**。
- **证据**：
  ```
  llm_requests.jsonl（n_messages=122，起点相对上条工具输出）: +0s/+601s/+1202s/+1806s/+2407s "client disconnected (cancelled)", latency 600.0
                                                            +3007s finish=length, completion 81920, latency 598.0
  trajectory msg[122] reasoning tail: "…Let's try to set up the winston logger before requiring winston.\n\nLet's try to set up the winston logger before requiring"
  agent/result.json: "exit_status": "BudgetExhausted(in-flight call abandoned)", "n_calls": 49, "agent_wall_sec": 2939.9
  ```
- **根因**：MODEL，继承自 E-SWE v2。Base 三次运行里没有任何生成 ≥20K token，E2 有 5 次 length、23 次被取消。r3 的 7.84 分与这一类失控生成直接相关（见 C5）。

#### A8. element-web-772d（V3）：一轮生成耗时约 1 小时，同一句重复 198 次
- **运行/任务**：`V3:A/instance_element-hq__element-web__rfJh473`（这道题 Base 做对 1/3）。
- **现象**：机制与 A7 相同。最后一轮先有 6 次请求各在 600 s 被取消，第 7 次在 542 s 内写满 81920 token。从上一条工具输出到最后这次请求开始，相隔 3610 s。
- **证据**：
  ```
  msg[142]: completion=81920 finish=length（latency 542.6s），之前 6 次 600s 取消；reasoning_len=353962；60 字符短语重复 198 次
  tail: "…the `device-action-button` is not being rendered. The `children` prop is not being passed to the `DeviceListItem"
  agent/result.json: BudgetExhausted(in-flight call abandoned), n_calls=71（截止时补丁 29.6 KB，regrade 失败）
  ```
- **根因**：MODEL（继承）。**回退**：相对 Base 是部分回退（1/3）。

#### A9. vuls-ca3f（V1）：断网沙箱中把 `go mod download -x` 执行了 132 次
- **运行/任务**：agent `V1:A/instance_future-architect__vuls__N9BEDxY`，regrade `V1:G/…__qazVNqB`。对照：B1、B3、V2 和 E2 都做对（80/80）。
- **现象**：前 117 步里已经写出一个接近正确的补丁（最终 79/80，只缺 `TestIsOvalDefAffected`）。之后模型想下载依赖的新版本：先用 `curl` 访问 goproxy.io，返回 rc=6，因为 agent 阶段断网。从第 118 条到第 250 条，同一条 `go mod download -x …` **原样重复了 132 次**，直到步数上限（LimitsExceeded）。这些轮次的 content 全为空，reasoning 中位数 536 字符，内容是"已经试过 goproxy.cn，再试 goproxy.io"，可实际发出的命令每次都一样。**这是推理和动作脱节**。
- **证据**：
  ```
  #116 'curl -s --max-time 3 https://goproxy.io/github.com/vulsio/goval-dictionary/@v/v0.7.4-….zip …' -> rc=6
  #118 = #119 = … = #250: 'cd /app && go mod download -x github.com/vulsio/goval-dictionary@v0.7.4-0.20220803092243-4891cffd7a65 2>&1 | head -5'
  loop turns: content 中位数 0 字符，reasoning 中位数 536 字符（"We already tried goproxy.cn, but let's try goproxy.io."）
  regrade test-stdout: Required tests: 80 / passed: 79, Missing: ['TestIsOvalDefAffected']
  ```
- **根因**：MERGE。"环境正常时同一命令重复 ≥20 次"在 Base 三次运行和 E2 中都是 0 例，v4 有 3 例（A9、A10，以及 V1 protonmail-5f07 中 `sed -n` 重复 43 次）。**回退**：相对 Base 和 E2 都是。

#### A10. flipt-2ca5（V2）：同一条 grep 重复 210 次，一行没改
- **运行/任务**：`V2:A/instance_flipt-io__flipt-2ca5dfb__p7xVx2t`（LimitsExceeded，补丁 0 字节）。
- **证据**：
  ```
  #38 'grep -n "go.opentelemetry.io/otel/exporters" /app/go.sum | grep -v "trace" | grep -v "jaeger" | grep -v "zipkin"' -> 755:…prometheus…
  #39 = #40 = … = #250: 同一命令再加 '| grep -v "prometheus"' -> {"returncode": 1, "output": ""}
  say（约 210 次相同）: "Let me check the go.opentelemetry.io/otel/exporters/otlp package structure by looking at the go.sum."
  ```
- **根因**：MERGE（退化循环，输出完全确定，模型却没有察觉；reasoning 里写着 "Let me try to run a Go command"，发出的却还是那条 grep）。这题 Base 和 E2 也都做不对，所以没有造成得分回退，但浪费了约 210 步预算。

#### A11. vuls-ef2b（V2）：只读不改，直到上下文溢出
- **运行/任务**：`V2:A/instance_future-architect__vuls__vGiff94`。
- **现象**：182 条命令全是 `cat -n`/`sed -n` 这类阅读操作，像是编辑的命令一条也没有。可见回复里 "Now I have a good understanding of the codebase. Let's start implementing the changes…" 出现了 136 次，第一次在第 26 条命令。最后 prompt 超过 262,144 token，抛出 ContextWindowExceededError，补丁 0 字节。
- **证据**：
  ```
  agent/result.json: "exit_status": "ContextWindowExceededError", "n_calls": 183, "agent_wall_sec": 930.1
  error: "…maximum context length is 262144 tokens … your prompt contains at least 262145 input tokens"
  #178–#182: 'cat -n ./gost/gost.go', "cat -n ./oval/util.go | sed -n '30,60p'" …（say 每次都是 "…Let's start implementing…"）
  ```
- **根因**：MODEL。E2 在同一道题上也是 ContextWindowExceeded（134 次调用），v4 是继承来的。V1 反而做对了这道题，所以不是回退。

#### A12. 补丁卫生：`.bak`/`.backup` 备份文件与临时脚本（跨题统计）
- **数据**：见 §1.3。v4 有 33/127 个非空补丁带 `.bak`/`.backup`（共 59 个文件），Base 3/145，E2 2/44，Mixed 10/46。几个极端例子：
  - V2 tutanota（`V2:A/instance_tutao__tutanota-f373ac3__pVkM2wX`）的补丁有 344 KB，包含 `EntityClient.ts.backup`、`DefaultEntityRestCache.ts.backup2` 等整文件副本，但这次仍然得分。
  - V3 ansible-d624（`V3:A/instance_ansible__ansible-d62496__rGDvr8J`）在仓库根目录新建了 13 个文件：`update_pattern{,2,3}.py`、`update_validation{,2,3}.py`、`debug_check{,2}.py` 等。这些都是"写一个 Python 脚本去改源文件"的中间产物。
- **根因**：MERGE。**假设**：`cp x x.bak` 的习惯来自 E-ASI 或 E-TB v2 的终端轨迹。粗略 grep，包含 `.bak/.backup` 的样本占比：ASI 52/868，TB v2 13/486，SWE v2 6/383。不影响判分（regrade 会重置测试区）。**有一个讽刺的副作用**：`.bak` 文件按字母序排在源文件后面，反而让 v4 少受 C4 bug 的影响。

### B. SWE：能力回退（模型逻辑）

#### B1. vuls-5af1（V1）：删除已有测试文件，改语义迁就自写测试
- **运行/任务**：agent `V1:A/instance_future-architect__vuls__KncjD7Y`，regrade `V1:G/…__hqRan2R`。对照：B1–B3 和 E2 都是 11/11。
- **现象**：
  - 模型用 heredoc 整体重写了 `scanner/utils.go`（第 93 条命令），理由是 "treat the absence of distro release as modern format"。
  - 第 102 条用 `rm` 删除临时文件时，把仓库里**原有的** `scanner/utils_test.go` 一起删了。补丁里这个文件显示为 `deleted file mode`。
  - 补丁另外还带了 `debug_parse5…8_test.go` 和 `utils_debug_test.go`，共 5 个调试测试。
  - regrade 恢复了官方测试，`old_redhat_kernel_release_style` 用例失败。
- **证据**：
  ```
  #93  "cd /app && cat > scanner/utils.go <<'EOF' …"  (say: "…treat the absence of distro release as modern format.")
  #102 'cd /app && rm scanner/debug_kernel_test.go scanner/debug_isRunningKernel_test.go scanner/utils_test.go scanner/debug_pr_scenario_test.go …'
  model.patch: diff --git a/scanner/utils_test.go … deleted file mode 100644
  regrade run-script-stdout: utils_test.go:175: isRunningKernel() gotRunning = false, want true / --- FAIL: Test_isRunningKernel
  ```
- **根因**：MODEL。为了让自己的测试通过而改变语义，并误删了已有测试。草稿中 E-SWE v1 的 D1 是同一题、同一种模式。**回退**：相对 Base 和 E2 都是。

#### B2. vuls-5af1（V3）：多内核场景选错版本
- **运行/任务**：`V3:A/instance_future-architect__vuls__yqkqvU3`，regrade `V3:G/…__K2Rzu4r`（10/11）。
- **现象**：第 45 条命令时，模型看到已有测试 `TestParseInstalledPackagesLinesRedhat` 失败，于是修代码，并改了 `utils_test.go`。最后第 104 条 `go test ./scanner` 显示 ok。但隐藏测试新增了"装有多个内核"的场景，v4 选了最新的内核，而期望的是正在运行的那个。
- **证据**：
  ```
  #45  go test ./scanner -v -run TestParseInstalledPackages -> "--- FAIL: TestParseInstalledPackagesLinesRedhat"
  #104 'cd /app && go build ./... && go test ./scanner && go test ./...' -> "ok  github.com/future-architect/vuls/scanner (cached)"
  regrade: redhatbase_test.go:270: release: expected 427.13.1.el9_4, actual 427.18.1.el9_4
  ```
- **根因**：MODEL（对需求理解不完整）。**回退**：相对 Base 和 E2 都是。

#### B3. teleport-eda6（V1–V3）：三次都与隐藏测试的 API 对不上（E-SWE v2 28/28）
- **运行/任务**：
  - V1：`V1:A/instance_gravitational__teleport__dtY6hBN`，regrade `V1:G/…__v2Cvii6`
  - V2：`V2:A/…__2GTKTLH`，regrade `V2:G/…__HMxFcDN`
  - V3：`V3:A/…__gttnYzr`，regrade `V3:G/…__NDxWQ8Q`
  - 对照 E2：`E2:A/instance_gravitational__teleport__Qz9rJcJ`，28/28
- **现象**：三次都是 `lib/kube/proxy` 测试包编译失败（0/28），每次缺的东西不一样：
  - V1 只写了复数形式的 `kubeClusterEndpoints`，没有定义类型 `kubeClusterEndpoint`。
  - V2 把方法写成导出的 `Dial`，测试要的是 `dial`；还缺 `kubeAddress` 字段。
  - V3 缺 `dial`。

  E2 的命名和隐藏测试完全一致。
- **证据**（`verifier/run-script-stdout.txt` 和 `run-script-stderr.txt`）：
  ```
  V1: lib/kube/proxy/forwarder_test.go:653:21: undefined: kubeClusterEndpoint
  V2: forwarder_test.go:743:16: sess.dial undefined (type *clusterSession has no field or method dial, but does have Dial)
  V2: forwarder_test.go:753:23: sess.kubeAddress undefined
  V3: forwarder_test.go:743:16: sess.dial undefined (type *clusterSession has no field or method dial)
  ```
- **根因**：MERGE（专家能力被稀释）。Base 也是 0/3，有两次 0/28 编译失败，一次 26/28，这说明 E2 学到的"按题面接口命名"在合并后丢了。**回退**：相对 E2 是；相对 Base 不是。

#### B4. teleport-f432（V1、V3）：raw keystore 实现有误
- **运行/任务**：V1 regrade `V1:G/instance_gravitational__teleport__R94qmBB`；V3 regrade `V3:G/…__WksKSPY`。两次的补丁最后一个文件都是 `raw_test.go`，所以 `raw.go` 已经完整应用（与 C4 无关）。
- **证据**：
  ```
  V1: keystore_test.go:263: Error: Received unexpected error: no usable SSH key pair   --- FAIL: TestKeyStore/raw_keystore
  V3: Error: Received unexpected error: crypto/rsa: verification error                 --- FAIL: TestKeyStore/raw_keystore
  ```
- **根因**：MODEL，与草稿 D2（E-SWE v1）是同一个失败点。**回退**：相对 Base 是（B1、B3 做对；B2 的失败是 C4 截断，最后一个文件 `raw.go` 整个丢了）。和 E2 不可比，因为 E2 的失败同样是最后一个文件 `raw.go` 被截断。

#### B5. element-web-ecfd（V1、V3）：只修了一部分，靠自写测试"验证"
- **运行/任务**：
  - V1：`V1:A/instance_element-hq__element-web__hgHDnmm`，regrade `V1:G/…__mBmZZkH`（25/26）
  - V3：regrade `V3:G/…__eSLsZRV`（25/26）
  - 对照：B2、B3 和 E2 都是 26/26
- **现象**：V1 写了 `FinalVerify-test.tsx`，PASS 之后删掉，然后提交。隐藏的 `SearchResultTile` 用例在处理 m.call 事件分组时报 TypeError。V3 则是 `RoomSearchView` 的"合并连续搜索结果"用例失败。
- **证据**：
  ```
  #106 '… npm test -- --te…' -> "PASS test/components/views/rooms/FinalVe…"
  #107 'cd /app && rm -f test/components/views/rooms/FinalVerify-test.tsx /tmp/final-verify.tsx && git status --short …'
  V1 regrade: FAIL test/components/views/rooms/SearchResultTile-test.tsx  TypeError: Cannot read properties of undefined (reading '1')
  V3 regrade: FAIL test/components/structures/RoomSearchView-test.tsx
  ```
- **根因**：MODEL（边界情况没覆盖；用自写测试代替官方测试）。**回退**：相对 Base 和 E2 都是。

#### B6. openlibrary-53d3（V1–V3）：返回 dict 而不是对象（E-SWE v2 8/8）
- **运行/任务**：
  - V1：`V1:A/instance_internetarchive__openli__kM6vGyp`，regrade `V1:G/…__few3jux`（7/8）
  - V2：`V2:A/…__RqZjHAG`（LimitsExceeded，4/8）
  - V3：regrade `V3:G/…__4BTZUxB`（7/8）
  - 对照：E2 `E2:A/…__A5zKuBj`（8/8）
- **证据**：
  ```
  #59 'python3.11 -m pytest openlibrary/catalog/add_book/tests/ openlibrary/mocks/tests/ -q' -> "…" (say: "All tests pass.")
  V1 regrade: test_load_book.py::TestImportAuthor::test_last_match_on_surname_and_dates FAILED
              assert found.key == "/authors/OL3A"   E   AttributeError: 'dict' object has no attribute 'key'
  ```
- **根因**：MERGE（专家能力被稀释）。Base 0/3 的失败点相同。**回退**：相对 E2 是。

### C. SWE：harness / 环境问题

#### C1. element-web-ecfd（V2）：jest 跑整个目录导致 OOM，之后反复提交（MIX）
- **运行/任务**：`V2:A/instance_element-hq__element-web__vqJmQLn`，regrade `V2:G/…__vZbBWQp`。
- **现象**：第 147 条命令是 `timeout 120 npx jest --testPathPattern="test/components"`，对整个组件目录跑测试。第 148 条运行一个 node 脚本时，内存以 1011 MB/s 的速度上涨（jest worker 可能还在后台），容器被看门狗杀掉。之后 `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 重复 102 次，一直到 250 步，补丁没有导出。这与草稿 A8（Base）是同一机制：v4 有 13 个 OOM trial，Base 有 8 个。
- **证据**：
  ```
  #147 'cd /app && timeout 120 npx jest --testPathPattern="test/components" 2>&1 | tail -30'
  trial.log: Memory explosion detected: 3108MB, growing 1011MB/s / Memory limit exceeded: 4631MB > 3891MB (113%). Killing container.
  #149–#250: 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' -> "Container exceeded memory limit (4631MB > 3891MB)"（×102）
  agent/result.json: LimitsExceeded, 250 calls；model.patch 不存在
  ```
- **根因**：MIX。模型启动了重负载测试，之后没能识别出"环境已经死了"；harness 在容器死后没有终止 agent，也没有保住已有的修改。**回退**：相对 Base 和 E2 都是，但含环境因素。

#### C2. navidrome-bf2b（V2）：补丁末尾缺换行，gg.go 没被应用
- **运行/任务**：agent `V2:A/instance_navidrome__navidrome-bf__gRtn8mZ`，regrade `V2:G/…__QYDuurv`（1/3）。
- **现象**：补丁在 `utils/gg/gg.go` 文件末尾追加了 `P[T]`/`V[T]` 两个函数，定义本身是对的。但 model.patch 以 `+}` 结尾，后面**没有换行符**：
  1. `git apply` 报 "corrupt patch at line 261"。
  2. 回退到 `patch --fuzz=3` 后报 "malformed patch at line 260"。
  3. 结果 gg.go 没被应用，编译时报 `undefined: gg.V`。
- **跨运行对照**（同一题，8 个 trial）：凡是补丁最后一个文件为 gg.go 的（B2、V2、E2）**全部失败**；最后一个文件为 gg_test.go 的（B1、B3、V1、MX）**全部成功**。V3 的失败是因为补丁为空（A4）。
- **证据**：
  ```
  model.patch 末尾字节: "+\t}\n+\treturn *p\n+}"（260 行，无结尾换行）
  regrade trial.log: Command exited with rc=2: error: corrupt patch at line 261 / patching file utils/gg/gg.go /
                     patch unexpectedly ends in middle of line / patch: **** malformed patch at line 260:
  verifier: core/external_metadata.go:94:8: undefined: gg.V
  ```
- **根因**：HARNESS（见 C4）。这道题在 §1.2 表里算作 v4 的一次回退，实际上是**假回退**；Base 在 B2 上也吃了同样的亏。

#### C3. flipt-e2bd（V3，以及 B3）：最后一个文件被整个丢弃
- **运行/任务**：v4 `V3:A/instance_flipt-io__flipt-e2bd19d__nYdLNsw`，regrade `V3:G/…__RaWDjZU`；Base `B3:G/instance_flipt-io__flipt-e2bd19d__fTpM5XF`。
- **现象**：两个补丁的最后一个文件都是 `internal/storage/cache/cache.go`，结尾都是新增行。`setJSON`/`getJSON` 定义在 cache.go **中间**的 hunk 里（V3 补丁第 276–291 行，全文 353 行），regrade 时却仍然报 undefined。这说明 GNU patch 遇到 malformed 时**整个文件都没写入**，不只是丢了最后一个 hunk（这一点是推断）。两次都是 5/10。
- **证据**：
  ```
  regrade trial.log: patching file internal/storage/cache/cache.go / patch: **** malformed patch at line 353:
  verifier: internal/storage/cache/cache_test.go:25:14: cachedStore.setJSON undefined (type *Store has no field or method setJSON)
  model.patch:281: +func (s *Store) setJSON(ctx context.Context, key string, value any) {
  ```
- **根因**：HARNESS。flipt-e2bd 上 v4 的 3 次失败，实际是 2 次 commit 加 1 次 harness 截断，**没有证据表明 v4 不会做这道题**。

#### C4. 补丁捕获时 `stdout.strip()` 去掉末尾换行（系统性问题，影响所有模型）
- **机制**：
  - harbor singularity 的执行服务在 `server.py:152` 执行 `actual_output = stdout.strip()`（文件在 `envs/eval/lib/python3.12/site-packages/harbor/environments/singularity/`，相对 /scratch/eh3323/apex）。
  - `swepro_host_agent.py` 把这个输出原样写入 `model.patch`，所以**每个**非空补丁都缺少末尾换行：Base 145/145，v4 127/127。
  - regrade（`data/benchmarks/swebench_pro_repo/v2/tooling/patch_replay.py:65`）依次尝试 `git apply`、`git apply --3way`、`patch --fuzz=3`。
  - 当最后一个文件的最后一行是新增行（新建文件，或在文件末尾追加）时，3 种方式都失败或报 malformed，**最后一个文件整体丢失**。
  - 另外，harbor 把 stderr 合并进了 stdout，`git add` 的警告也可能混进补丁（未见实例）。
- **规模**（regrade `trial.log` 中出现 "malformed patch" 的 trial 数）：Base 11 / 15 / 20，v4 22 / 18 / 23，E2 11，Mixed 32。如果再限定"最后一个文件是源码而不是 `.bak`/测试/临时文件，并且该 trial 失败"（可能受影响的上界），则为 Base 10 / 12 / 14，v4 5 / 8 / 7，E2 8。
- **几道题可能每次都被这个 bug 卡住**（解法的最后一个文件是新建的源码，所以每次都会丢）：
  - ansible-40ade1：新建 `lib/ansible/modules/mount_facts.py`。B1 regrade 报 `ImportError: cannot import name 'mount_facts'`。
  - teleport-7744：新建 `lib/auditd/common.go`。V1 报 `undefined: NetlinkConnector`，V2 报 `undefined: Message`。
  - 另外可能还有 flipt-c172 的 `internal/cue/validate.go`、openlibrary-7cbf 的 `observations.py`、protonmail-5f07 的 `flush-promises.ts`（未逐一核实）。
- **证据**：`B1:G/instance_ansible__ansible-40ade1__TLQ2JkT/trial.log`
  ```
  patching file lib/ansible/modules/mount_facts.py
  patch unexpectedly ends in middle of line
  patch: **** malformed patch at line 1052:
  E   ImportError: cannot import name 'mount_facts' from 'ansible.modules'
  ```
- **根因**：HARNESS，**不是模型的失败**。**影响**：所有模型的 HARD-51 绝对分都被系统性压低，而且各模型受影响的程度不同：v4 习惯留 `.bak` 文件，反而受影响更少。**建议**：修复方式是写 `r.stdout + "\n"`，或在容器内用 `git diff --cached > /tmp/p && cat` 再在宿主机补上换行；修复后对所有现有 `model.patch` 补上换行重新 regrade，不需要重跑 agent。在此之前，SWE 的模型间比较应附上这条局限说明。

#### C5. 客户端 600 s 超时与长生成叠加（MIX）
- **现象**：litellm 客户端单次请求的超时是 600 s，超时后取消请求并重试，重试次数没有上限。代理日志中"client disconnected (cancelled)"的请求数：Base 0 / 0 / 0，**v4 14 / 13 / 41**，E2 23，Mixed 1。
  - 单流解码约 137 tok/s，写满 `max_new_tokens=81920` 正好需要约 600 s。所以被取消的请求，和 finish=length 的请求（都在 540–598 s 之间结束），很可能是**同一种失控生成**，只是落在 600 s 截止线的两侧。
  - 每取消一次就白白消耗 10 分钟。

  V3 中 4 个 BudgetExhausted 全部与此有关：
  - NodeBB 和 element-web-772d：分别取消 5 次、6 次，之后写满 81920 token（A7、A8）。
  - ansible-d624（`V3:A/instance_ansible__ansible-d62496__rGDvr8J`）：两次工具调用之间相隔 1806 s，即取消 3 次后的第 4 次才得到 192 token 的正常响应。
  - element-web-66d0（`V3:A/…__y89ZWKm`）：先后出现 1805 s 和 2405 s 的间隔。
- **证据**：`merged/swepro/merged_v4__swepro__r3/llm_requests.jsonl`
  ```
  {"n_messages": 122, "n_tools": 1, "error": "client disconnected (cancelled)", "latency_s": 600.1}
  ```
- **根因**：MIX。失控生成是模型行为（继承自 E-SWE v2）；600 s 超时、无上限的重试，以及 `max_new_tokens=81920`，是 harness 的配置。r3 只有 7.84 分，比 r1、r2 低约 8 分，这一项是主要原因之一。

### D. HLE（v3 合版代理，v4 还没判分）

#### D1. 思考长度减半，损失集中在中长推理题
- **数据**：
  - v3 输出 token 中位数 12.1K，Base 24.6K，E-HLE 13.2K。
  - 在 Base 做对、v3 做错的 245 题中，v3 有 242 题正常结束（finish=stop），它用的 token 中位数只有 Base 的 **0.46 倍**；其中 71 题不到 0.3 倍，这 71 题里又有 22 题 v3 自报置信度 ≥90。
  - 按 Base 输出长度分四档，Q3（24.6K–39.3K token）掉得最多（−6.4 分，79 : 39），最短的一档 Q1 只掉 1.4 分。
  - 学科上，Math 不一致对为 125 : 83。
- **代表性题目 id**（只列元数据）：

  | id | 类别 | 答案类型 | Base token | v3 token | v3 置信度 |
  |---|---|---|---|---|---|
  | `6725fe6b26992c47ce3a7ef5` | Math | multipleChoice | 26,704 | 1,331 | 100 |
  | `673af80a3022b9e17f89d1c5` | Math | exactMatch | 65,069 | 4,606 | 65 |
  | `66e932f00bd16d40d4bc4d81` | CS/AI | multipleChoice，带图 | 27,391 | 1,115 | 100 |
  | `6733f23b731fba4902214821` | Math | exactMatch | 19,815 | 1,584 | 50 |

  最后一题 E-HLE 做对了，用了 7,103 token。
- **证据**：`merged/hle/merged__hle__s1` 与 `base/hle/base__hle__s1` 的 normalized 行（`repo/results/*/hle/*.jsonl` 的 tokens_out、status、extra.confidence）。
- **根因**：MODEL，继承自 E-HLE（E-HLE 的输出长度分布与 v3 几乎一样）。**回退**：相对 Base 是，相对 E-HLE 不是。

#### D2. 思考失控截断：24 题无预测，代理日志中 length 占 7.5%
- **数据**：
  - v3 有 24 题在 3 轮补跑后仍然没有预测：Math 20、CS/AI 2、Other 1、Bio/Med 1；exactMatch 22。Base 是 3，E-HLE 4，Mixed 6。
  - 代理日志中 finish=length 的比例：v3 7.5%，Base 2.4%，E-HLE 3.5%，Mixed 2.7%。
  - 生成日志中第 2 轮补跑 382 题（Base 42）。第 1 轮中途有一次重启（日志里有两个 0/2500 进度条），所以 382 里有多少来自中断，**没有核实**。
  - 对得分的直接影响很小：这 24 题 Base 只做对 2 题，它们本身是 Base 也要用很长推理的难题（Base token 中位数 55.6K）。
- **样例 id**：`66ef7b798ec4c5bfcdb205fc`、`66f02cc56ea864e80ba2d3d7`、`6712c2616313b64cf8499927`（都是 Math / exactMatch），以及 `670bcb222407af9de8866eda`（Math / multipleChoice）。
- **根因**：MERGE。两个"思考更短"的模型（E-HLE、Mixed）都没有这么高的截断率，所以是合版后出现的双峰现象：多数回答更短，但有一条更长的失控尾巴。这和 SWE 上的 A7、A8 是同一种现象。

#### D3. 校准：E-HLE 的校准改进在合版后大部分丢失
- **数据**：

  | 指标 | Base | E-HLE | Mixed | v3 |
  |---|---|---|---|---|
  | 校准误差 | 73.9 | 35.9 | 36.2 | **61.3** |
  | 置信度均值 | 97.3 | 54.6 | 55.2 | 80.9 |
  | 置信度 ≥95 却答错 | 1803 | 47 | — | **503** |

  - v3 置信度直方图：80–89 有 732 题，90–99 有 532 题，**恰好 100 的有 329 题**。
  - E-HLE 的置信度集中在 40–79。
  - v3 置信度 40–59 那一档只有 134 题，准确率 5.2%；80–99 档准确率 22.0%。排序上还有一点信息量，但数值整体偏高。
  - 对 v3 置信度 ≥95 却答错的 503 题，E-HLE 在同一批题上的置信度均值是 56.1。
- **证据**：`merged/hle/merged__hle__s1/judge_stdout.txt`，以及 normalized 行的 `extra.confidence`。
- **根因**：MERGE。Mixed 学到了校准（36.2），v3 只保留了约 1/3 的改进（(73.9−61.3)/(73.9−35.9) ≈ 33%）。**回退**：相对 E-HLE 是；相对 Base 是改善。

#### D4. 带图题：推理长度骤降
- **数据**：
  - 带图的 342 题，准确率：Base 15.5，E-HLE 11.4，Mixed 10.2，v3 11.1。不一致对 Base : v3 = 30 : 15。
  - 带图题输出 token 中位数：Base 19.1K，**E-HLE 7.0K，v3 7.0K**。
- **根因**：MODEL，继承自 E-HLE。E-HLE 只用约 2000 条纯文本训练，在带图题上同样变短、变差。**回退**：相对 Base 是（−4.4，比不带图题的 −3.3 更大）；相对 E-HLE 不是（−0.3）。

#### D5. 部分类别 v3 比专家还差
- **数据**（v3 − E-HLE，括号内为不一致对 E-HLE : v3）：
  - Physics −3.0（21 : 14）
  - Other −2.1（16 : 11）
  - Bio/Med −1.4（22 : 18）
  - 同时 Humanities +4.6（6 : 16），Math +1.6（93 : 109）

  总体 v3 ≈ E-HLE（p = 0.62）。
- **根因**：MERGE。和其他专家的参数干扰在不同类别上方向不一，Humanities 得益，理科的 Physics、Bio/Med 受损。各类别样本量较小（111–280 题），单个类别的差异都不显著，只能算作方向性证据。

---

## 4. 规律（附计数）

1. **"提交协议"被破坏（MERGE）**：v4 有 12/153 个 SWE trial 在提交前 `git commit`，其中 11 个补丁为空；Base、E2 和 Mixed 都是 0。7 道回退题上的 16 次失败里有 4 次属于这一类。按 v4 的成功率估算，修好这一点能挽回约 2–4 个 trial（+1.3 到 +2.6 分），但补丁本身是否正确无法复评。
2. **失控 / 退化生成（MODEL，继承 E-SWE v2；循环部分为 MERGE）**：
   - 单次写满 81920 token 的有 7 次（Base 0）。
   - 600 s 被取消的请求 68 次（Base 0）。
   - 单段 reasoning 内同一短语重复 198–481 次。
   - 环境正常时同一命令重复 43–210 次的有 3 个 trial（Base 和 E2 都是 0），而且 reasoning 说"换个方法"，动作却完全不变。
   - r3 的 4 次超时全部来自这一类。
   - HLE 上表现为 v3 的截断率是 Base 的 3 倍，有 24 题无预测。
3. **专家能力被稀释（MERGE）**：
   - E2 做对的 11 道题，v4 只做对 36%。
   - teleport-eda6 三次都没对上隐藏 API，E2 是 28/28。
   - E-HLE 的校准改进只保留了约 1/3。
   - 但 E2 的正迁移在 flipt-524f（0/3 → 2/3）上保留了下来。
4. **"自写测试通过 ≠ 官方测试通过"（MODEL，Base 也有）**：B1、B2、B5、B6 都是"自己的测试或已有测试通过，隐藏测试失败"。B1 甚至删掉了已有测试文件。
5. **补丁卫生（MERGE）**：带 `.bak`/`.backup` 的补丁占 26%（Base 2%）；有临时脚本和整文件副本。不影响判分，但说明合版把终端 / ASI 风格的"先备份再改"习惯带进了 SWE。
6. **harness 层问题**：
   - 补丁末尾换行被 strip，每次运行有 11–32 个 trial 的补丁是 malformed，最后一个文件整体丢失。这影响所有模型，已确认 navidrome-bf2b 和 flipt-e2bd 的结果因此翻转。
   - 容器 OOM 后，agent 仍然重复提交到步数上限：v4 13 次，Base 8 次。
   - 客户端 600 s 超时导致无上限重试。
7. **不是问题的部分**：v4 的格式错误（没有 tool call）比 Base 少：14 对 21 个 trial。也没有出现 E2 那样大量的时间预算耗尽：5 次，E2 是 14 次（只有 1 次重复）。

| 根因 | 案例数 | 其中造成得分回退（相对 Base 或专家，不含假回退） |
|---|---|---|
| MERGE | 14 | 10（A1–A4、A9、B3、B6、D2、D3、D5；A5、A6、A10、A12 不影响得分） |
| MODEL | 9 | 7（A8、B1、B2、B4、B5、D1、D4） |
| HARNESS | 3 | 0（C2、C3 是假回退） |
| MIX | 2 | 1（C1） |

---

## 5. TB2 / ASI（待补）

v4 在 TB2 和 ASI-Bench 上的运行还没结束，目前没有可分析的结果。运行完成后会按本文格式补充案例。重点检查以下几项：

- E-TB v2 是否消除了草稿 C1–C7 中 E-TB v1 的"批量按键、先写后看"风格。
- SWE 上观察到的 commit、`.bak` 习惯和退化循环，在终端任务中是表现为优势还是新的失败。
- 草稿 A1 的摘要死循环是否仍然存在。

**HLE v4（未判分，仅供参考）**：`merged/hle/merged_v4__hle__s1/llm_requests.jsonl` 目前有 2838 个请求，finish=length 248 个，占已完成请求的 9.0%（v3 为 7.5%）。生成日志显示第 2 轮正在补跑 2185 题（`logs/merged_v4__hle__s1/hle_pred.log`，相对 /scratch/eh3323/apex），说明第 1 轮的大部分结果没有落盘。原因未查，可能是作业中断。判分完成后，要核对 D1–D5 的结论能否从 v3 推到 v4。

## 6. 未能核实 / 待办

1. **空补丁不能复评**：A1–A6 中模型 commit 的内容是否正确未知。可以从轨迹里重建 commit 内容（各轨迹中的 `git diff HEAD~1`），但这需要写盘并重新 regrade，不在只读范围内。
2. **C4 的 bug 应先修复再重新 regrade 全部运行**：给所有现有 model.patch 补上换行，不需要重跑 agent。在此之前，所有 SWE 分数都偏低，而且偏低程度因模型而异。"整个最后一个文件都没应用"是根据 C3 推断的，没有在容器里复现。
3. **commit 和 `.bak` 习惯的来源**：目前只有训练数据 grep 得到的旁证（E-ASI 的系统提示）。Mixed 用了同一份 ASI 数据却从不 commit，与这个假设相矛盾。需要让单个专家（E-ASI、E-TB v2）在 SWE 子集上各跑一次来定位。
4. **HLE 第 2 轮补跑 382 题的构成**（截断、超时、中断各占多少）没有逐条对齐；v3 第 1 轮有一次重启。
5. vuls-ca3f V3（74/80）、openlibrary-0dc5 V3（59/61）、element-web-ecfd V3 的失败细节没有展开。
6. Mixed 中 1 个重复 ≥20 次的循环（protonmail `kVrjotB`）没有核实是否与 OOM 有关。
7. SWE 每个模型只有 1–3 次重复，任务级"回退"受采样噪声影响很大（Base 自己三次的得分是 19.61 / 11.76 / 21.57）。单题结论只能看作定性证据。

## 附录：复现用的只读脚本

统计脚本都在本机 scratchpad 的 `…/scratchpad/fc4/` 下，用 `ssh torch 'nice python3 -I -' < script` 在登录节点只读执行：

| 脚本 | 用途 |
|---|---|
| `swe4_table.py` | 逐题 × 运行的结果表 |
| `swe4_scan.py` | commit、revert、格式错误、OOM 等计数 |
| `runaway.py`、`runaway1.py`、`rep.py` | 长生成与重复度 |
| `loops.py`、`dup.py`、`loopturn.py`、`saycount.py` | 退化循环 |
| `corrupt.py`、`corrupt2.py`、`lastfile.py` | C4 的统计 |
| `hyg.py` | 补丁卫生 |
| `regr.py` | regrade 的缺失测试 |
| `llm4.py` | 代理日志 |
| `hle4.py`、`hle5.py` | HLE 统计，只读取 normalized 行中的数值和元数据字段 |
