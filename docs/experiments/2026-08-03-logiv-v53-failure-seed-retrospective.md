# LOGIV v53 失败 seed 复盘与逐任务改进路线

## Material Passport

- Artifact type: development-set failure retrospective
- Evidence date: 2026-08-03
- Backbone: frozen full `pi0.5` LIBERO checkpoint
- Controller candidate: v53, commit `168919fb`
- Goal mode: `METADATA_ASSISTED`
- Deviation mode: `NOMINAL`
- Grounding: development-only oracle grounding
- Primary comparison: paired episode index, identical initial-state and first-frame hashes
- Status: **development evidence only; not an independent final result**

## 1. 当前可比较结果

Task ID 使用代码中的零基编号。v53 只有 Task 0--5 完成了各 50 对 Full/Base；
Task 6--9 尚无同版本完整配对，因此不能外推。

| Task ID | Full | Base | 正翻转 | 负翻转 | 净差 |
|---:|---:|---:|---:|---:|---:|
| 0 | 48/50 | 47/50 | 2 | 1 | +1 |
| 1 | 49/50 | 50/50 | 0 | 1 | -1 |
| 2 | 46/50 | 49/50 | 1 | 4 | -3 |
| 3 | 45/50 | 46/50 | 3 | 4 | -1 |
| 4 | 47/50 | 49/50 | 1 | 3 | -2 |
| 5 | 20/50 | 24/50 | 9 | 13 | -4 |
| **合计** | **255/300** | **265/300** | **16** | **26** | **-10** |

300/300 对的 initial-state 与 first-frame hash 均匹配。Task 0、1、2、4 的初始
DAG width 均为 2；Task 3 的 `pick -> place -> close` 和 Task 5 的 `pick -> place`
是由真实 precondition/support 约束得到的 width-1 计划。编译器没有为保持原线性
顺序而添加伪边：存在独立子目标的四个任务仍保留并行 ready nodes。负结果不是由
初始状态不配对或把 DAG 强制编成链造成的。

旧版 v30b 在 Task ID 8 上为 Full 36/50、Base 30/50（+6），但它不是 v53
全矩阵的一部分，只能作为机制存在正信号的独立开发证据。

## 2. 负翻转 seed 与终止原因

| Task ID | 正翻转 seeds | 负翻转 seeds | 负翻转主要终止原因 |
|---:|---|---|---|
| 0 | 39, 42 | 49 | `BUDGET_EXHAUSTED` ×1 |
| 1 | — | 33 | `POST_STOP_GROUNDING_FAILURE` ×1 |
| 2 | 10 | 18, 38, 39, 41 | budget ×2；post-stop grounding ×2 |
| 3 | 7, 12, 21 | 2, 3, 24, 43 | post-stop grounding ×4 |
| 4 | 45 | 14, 41, 44 | post-stop grounding ×3 |
| 5 | 0, 2, 3, 7, 28, 29, 31, 38, 45 | 5, 8, 12, 13, 14, 15, 16, 19, 20, 27, 41, 46, 49 | no-certified-repair ×9；budget ×3；post-stop grounding ×1 |

26 个负翻转可归为两组：11 个在 `STOPPED` 后得到暂时不完整的 exactly-one
状态，15 个在固定 260-step 边界、重试或恢复预算中耗尽。VAL、certificate
和 DAG 编译没有出错；问题位于符号动作到冻结 backbone 的执行接口以及
post-stop 观测确认。

## 3. 根因反思

### 3.1 动作 schema 比 backbone 的技能粒度更细

Task 5 的原任务是“拿起书并放入 caddy 后部”，而 v53 把它拆成 `PickBook`
与 `PlaceHeldBook` 两次独立策略调用。第二次调用从“机器人已经拿着书”的分布外
状态开始，导致书掉到 recovery surface；随后用于捡起掉落书本的提示又与训练任务
分布不同。类似问题也出现在 Task 3 的 `PickBowl -> PlaceHeldBowl`。

结论：PDDL action schema 应与 pi0.5 已掌握的宏技能对齐。符号层可以记录宏动作
的前置条件与最终 effects，但不应仅为了增加 gate 数量而把一个可靠宏技能切断。

### 3.2 固定 260-step 边界破坏连续控制上下文

许多失败 attempt 在第 260 步结束，此时物体仍被抓持、刚离开来源位置，或尚未
稳定落入目标。零编辑 retry 会重新调用策略，丢失先前动作上下文，并消耗剩余全局
预算。固定均分 horizon 不等价于有界恢复。

结论：attempt 仍然有全局硬上限，但本地停止应由 effect、稳定停滞或剩余预算决定，
不能机械地在 260 步重置。

### 3.3 单次 post-stop snapshot 把过渡态放大为 terminal

Task 1--4 的 10 个负翻转和 Task 5 的 1 个负翻转报告
`confirmed=[]` 或 access exactly-one violation。它们发生在宏动作边界，包含抽屉
仍在运动、物体仍在沉降或短暂遮挡的情况。直接猜测为成功会破坏 fail-closed，
但单帧失败立即 terminal 也过于脆弱。

结论：在同一 attempt 的 STOPPED acknowledgement 之前增加有界、无新策略 dispatch
的 settling/re-observation；只有稳定可靠 facts 才进入 effect gate，持续 UNKNOWN
仍按原合同终止。

### 3.4 同一个完整任务 prompt 不能无条件覆盖所有 occurrence

v54 将 Task 0--2 的所有 occurrence 都改为官方完整任务句。hard-set 结果如下：

| Task | Seed | v53 / Base | v54 | 判定 |
|---:|---:|---|---|---|
| 0 | 18 | fail / fail | fail | 未改善共同失败 |
| 0 | 39 | success / fail | fail | **破坏正翻转** |
| 0 | 42 | success / fail | fail | **破坏正翻转** |
| 0 | 49 | fail / success | success | 修复负翻转 |
| 1 | 0 | success / success | success | 保护通过 |
| 1 | 33 | fail / success | success | 修复负翻转 |
| 2 | 6 | success / success | success | 保护通过 |
| 2 | 7 | success / success | fail | **保护回退** |
| 2 | 9 | success / success | success | 保护通过 |
| 2 | 10 | success / fail | fail | **破坏正翻转** |
| 2 | 18 | fail / success | success | 修复负翻转 |
| 2 | 38 | fail / success | success | 修复负翻转 |
| 2 | 39 | fail / success | success | 修复负翻转 |
| 2 | 41 | fail / success | fail | 未修复 |

因此 v54 只能作为诊断，不可冻结。正确方向是按 current obligation、attempt lineage
和 recovery phase 选择提示，而不是把所有节点统一改成完整任务句。

## 4. 逐任务改进策略

| Task ID | 保留项 | 最小候选修复 | 必须保护的 seeds |
|---:|---|---|---|
| 0 | 保留 v53 首次 action-specific prompt 与 width-2 DAG | 只在首试 effect failure 后使用完整任务 recovery prompt；提高 recovery-frontier 生效边界 | 39, 42；验证 49，参考 18 |
| 1 | 保留两个独立 place-in occurrences 与 width-2 DAG | 首试使用官方完整任务上下文；v54 已修复 33 | 0；验证 33 |
| 2 | 保留 `turn-on` 与 `place-on` 两个并行 ready 节点 | `turn-on` 保持短提示；第二节点使用“stove already on, put moka pot on it”；对落位过渡态做 bounded re-observation | 6, 7, 9, 10；验证 18, 38, 39, 41 |
| 3 | 保留 close 的符号独立 effect gate | 比较原三步计划与宏 `place-in-bowl -> close`；为 drawer access 增加 bounded re-observation | 7, 12, 21；验证 2, 3, 24, 43 |
| 4 | 保留两个 mug 的 width-2 DAG | 首试继续使用对象特定 prompt；只对失败/恢复使用完整任务上下文；增加 object-location re-observation | 45；验证 14, 41, 44 |
| 5 | 保留 recovery surface、retry lineage 与完整 VAL | 初始计划改为一个 in-distribution `place-in(book, source, back, access)` 宏动作；掉落后才使用 `pick recovery -> place-held-in` | 正翻转 9 seeds 全部；验证 13 个负翻转 |
| 6 | 保留 v53 的 pudding-before-mug 合法拓扑 tie-break | 在取得匹配 Base 前不调参 | 当前已完成 Full 24 条仅作诊断 |
| 7 | 保留 v53 | 先补齐 Base，避免对 49/50 Full 单边调参 | Full 已完成 50 条 |
| 8 | 保留 v30b 已验证的 width-2 frontier 机制 | 用冻结新候选重跑同版本 Full/Base | 旧 50-pair 作为保护参考 |
| 9 | 保留 v51 的 downstream task context | 先补齐 Base；失败时检查 microwave close 的稳定观测 | 当前 Full 17 条仅作诊断 |

## 5. 实施顺序与接受门槛

1. **v55 prompt routing**：Task 0 recovery-only context、Task 1 initial context、
   Task 2 obligation-specific context。只跑对应 hard/guard seeds。
2. **v56 bounded re-observation**：先写 UNKNOWN→reliable 和持续 UNKNOWN 两个状态机
   测试；前者可继续 effect gate，后者必须保持 fail-closed，且二者都不能创建新 attempt。
3. **v57 skill-aligned schemas**：Task 3/5 初始计划使用 backbone 已训练的宏技能，
   recovery 仍可展开为更细的 PDDL actions。编译器必须继续验证真实 support/threat
   edges；存在独立节点的任务仍要求 graph width >= 2。
4. 每个版本先跑 **负翻转 seeds + 正翻转保护 seeds**。任何保护 seed 回退即拒绝该
   版本，不允许用“修复数量更多”掩盖新回退。
5. hard set 通过后，先做受影响任务的 50-pair 开发复核；只有所有已配对任务均
   `Full >= Base` 才恢复 10×50。
6. 因为当前 50 seeds 已用于调试，最终论文确认应冻结代码后更换 policy RNG seed，
   并继续使用 paired initial states。开发结果与独立确认结果分表报告。

“每任务严格正增长”在 Base 已经 50/50 时数学上不可能。可复现且不诱导过拟合的
最终门槛应定义为：所有任务 accuracy **非负增长**；Base 未到天花板的任务尽量严格
提升；总体 task-macro paired difference 为正且报告 bootstrap 区间。天花板任务可用
相同步数预算下的 mean steps、effect failures 和 recovery cost 作为次级改进指标，不能
伪造超过 100% 的准确率。

## 6. 后续候选的实时证据（未冻结）

- v55 的 obligation-routed prompt 在 Task 1 达到 50/50，与 Base 的 ceiling 50/50
  持平并修复 v53 seed 33；Task 2 hard set 虽通过，但扩展到 seed 5 时出现新回退，
  因而 v55 不能直接作为 Task 2 的冻结配置。
- Task 4 v68 只把第一个动作消歧为“solid white mug → left plate”，并只对 Task 4
  启用 10-step place-effect stabilization。hard set 45/41/14/44 全成功；对应的
  50-episode 复核达到 50/50，相对 Base 49/50 为 +1；50 对 initial-state 与
  first-frame hash 全部匹配。此改动不改变原 width-2 DAG。
- 全局 post-stop re-observation、Task 3 宏动作、Task 4 完整任务 prompt、Task 5
  split-plan recovery prompt 等候选均因未修复失败或破坏 guard seed 被拒绝；默认
  配置保持关闭。
- Task 3 四个负翻转已经细分：2/43 为 close、3 为 place、24 为 pick。v71 将内部
  region 名改为自然、动作局部的任务语言，等待 hard-set 仿真。
- Task 5 v70 宏动作扩展实验与 v72 官方原句候选仍在评估，尚不能报告为提升。

## 7. 同时代复核与 non-destructive overlay 诊断

后续同一策略服务进程的 50-pair 复核改变了两个关键数字：

| Task ID | Full | Base | Full-only | Base-only | 净差 |
|---:|---:|---:|---:|---:|---:|
| 2 | 47/50 | 49/50 | 1 | 3 | -2 |
| 8 | 30/50 | 28/50 | 10 | 8 | +2 |

两项复核的 initial-state 与 first-frame hash 均为 50/50 匹配。Task 8 的旧版
`36/50 vs 30/50` 正信号在当前策略进程缩小为 `30/50 vs 28/50`；旧 evaluator
和当前 evaluator 在同一 policy process 上都无法复现部分旧成功 seed，因此差异不能
归因于 evaluator 回归。将新 Task 2/8 数字代入当前逐任务表后，Full LOGIV 合计
`427/500`，Base 合计 `437/500`，净差为 `-10/500`（-2.0 percentage points）。

为直接检验“逻辑只救场、不破坏 Base”的假设，新增 development-only
`LOGIV_REPAIR_OVERLAY`：Base 先用完整任务 prompt 原样执行；成功则零干预，只有统一
终止检查失败或观测到目标物体位于 recovery surface 时，才从真实当前状态重新 ground、
VAL certify 并执行局部修复。inactive monitor 的动作序列、步数和 inference request
已有逐项相等测试；真实 seed 2 的前 520 步 action hash 也与 Base700 完全相同。

Task 8 严格同 GPU0 policy process 的诊断结果为：

| Seed | Base 状态/对照 | LOGIV repair | 结论 |
|---:|---|---|---|
| 2 | Base700 在 603 步 transient done 后 settling 失败 | 520+108，post-stop grounding failure | 两者均失败 |
| 0 | Base700 跑满 700 失败；drop 在 step 465 被确认 | 465+235，budget exhausted | 两者均失败 |
| 4 | Base700 跑满 700 失败；drop 在 step 225 被确认 | 225+475，budget exhausted | 两者均失败 |

seed 0/4 的符号层都正确识别“一只 pot 已在 stove、另一只在 recovery surface”并只规划
剩余目标。seed 4 中 pi0.5 在 recovery 时又误拿另一只 pot；LOGIV 正确生成
`place-held-on` 修复并成功提交该 effect，随后继续捡掉落 pot，但最终耗尽 700-step
总 policy-action 预算。该证据把断点定位为：**逻辑状态估计、VAL 和局部重规划能够按
设计工作，但冻结 pi0.5 并不可靠具备 intervention-induced recovery states 上的技能
闭包，也不能保证执行第二个物体时不破坏第一个物体。**

因此当前理论不能直接推出 frozen-backbone success rate 非降。若要让 non-destructive
LOGIV 成立，还需要一个在 held/wrong-object/recovery-surface/crowded-target 状态上经过
训练或验证的 recovery executor，或把理论保证明确条件化为 executor effect-success
与 non-interference 假设。继续只调 prompt、固定 horizon 或 grounding 容差不会修复
这个缺失条件。
