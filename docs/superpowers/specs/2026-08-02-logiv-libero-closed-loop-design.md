# LOGIV × π₀.₅ × LIBERO-10 闭环执行设计

**日期：** 2026-08-02  
**状态：** 待用户书面复核  
**目标：** 在冻结的 π₀.₅ LIBERO policy 与 LIBERO-10 环境之间加入一个无需训练新模型的 LOGIV 闭环插件，完成多模态提案、完整 VAL 认证、因果 DAG 编译、事实门控执行和有界认证修复，并与现有 10×50 open-loop 基线进行可复现比较。

## 1. 已核验的仓库事实

- 仓库已经包含官方 `libero_10` 的 10 份 BDDL、固定初始状态、完整 π₀.₅ 与 2,000-step π₀.₅ checkpoint、现有 evaluator，以及两组各 500 episode 的基线记录。
- 官方 LIBERO BDDL 给出对象、区域、初始关系和 Goal，但不包含可直接交给 VAL 的高层 action schemas。LOGIV 必须另外公开固定执行 Domain。
- 本机已有 VAL 4，路径为 `/home/xingrui/.local/bin/Validate`。
- 完整 π₀.₅ 基线为 460/500（92.0%）。最弱任务是 task 8“双 moka pot 放到炉子”，为 27/50（54.0%）；失败录像表明常见情况是完成一个目标后在另一个目标上卡住。
- 当前 evaluator 是同步 `env.step` 循环，每五个低层动作重新请求一次 policy。它没有真实机器人的异步 `STOPPED` acknowledgement，因此需要单独定义仿真停止适配器，不能把视觉稳定冒充执行器栅栏。

## 2. 范围与实验声明

本阶段不调用外部 VLM API。主代理依据任务指令、初始图像、注册对象和固定词表，为 10 个任务制作版本化的 Scripted-VLM 提案；运行时 facts 由 LIBERO 原生 predicate 与机器人状态组成的 privileged oracle 提供。该结果必须标记为 `metadata-assisted/oracle-grounding`，只验证 LOGIV 控制闭环和 π₀.₅ subtask 执行，不宣称真实 VLM 感知性能。

未来接入 API 时，`ProposeSubtasks` 与 `GroundFacts` 的实现可以替换，Domain、VAL wrapper、DAG、Controller、Repair 和实验记录格式保持不变。Goal-prediction 与 metadata-assisted 结果必须分表。

以下内容不在首个实现范围内：训练或微调 π₀.₅、修改 OpenPI checkpoint、碰撞规划、连续动力学认证、真实机器人 emergency halt 实现，以及把 BDDL 冒充完整 PDDL action Domain。

## 3. 总体架构

系统分成六个有明确边界的组件：

1. `ProposalProvider`：产生 Initial State、Grounded Goal、证据和候选 grounded actions。
2. `FixedDomain` 与 `ValWrapper`：定义动作语义，生成 Domain/Problem/Plan，并调用真实 VAL。
3. `RepairOperator`：在固定 grounded action 集合中进行有界插入、删除和重排，所有最终候选必须重新通过 VAL。
4. `CausalDagCompiler`：从 certificate-bound 线性计划和固定动作语义编译部分序 DAG 与 Canonical Agenda。
5. `FactGrounder` 与 `AttemptExecutor`：获得 fresh facts，并把一个高层 action 映射为冻结 π₀.₅ 的 subtask prompt 和同步动作循环。
6. `LogivController`：实现 precondition/effect/goal gate、retry lineage、预算、修复和原子安装。

正常数据流为：

```text
initial observation
→ Scripted-VLM proposal
→ candidate Problem + candidate.plan
→ bounded Repair + complete VAL
→ certificate + causal DAG + canonical agenda
→ fresh-fact gate
→ π₀.₅ subtask attempt
→ synchronous STOPPED fence + settling + fresh facts
→ commit / retry / certified repair / terminal
```

## 4. 第一遍 Subtask 的 PDDL-like 持久化

第一遍提案同时写出三类不可混淆的产物：

- `proposal.json`：epoch、任务、registered objects、initial facts、goal facts、视觉证据、候选 actions、prompt version 和 provider version。
- `initial_problem.pddl`：由提案中的 Initial State 与 Grounded Goal 生成的标准 PDDL Problem。
- `candidate.plan`：标准 VAL plan 文本，每行一个 grounded action，例如：

```text
(place-in akita_black_bowl_1 kitchen_table_akita_black_bowl_init_region white_cabinet_1_bottom_region white_cabinet_1_bottom_region)
(close-access white_cabinet_1_bottom_region)
```

另有 `candidate.occurrences.json` 保存稳定 occurrence ID、原始候选索引、action lineage 和自然语言 VLA 指令。PDDL plan 文本本身不承担 occurrence ID、证据或 retry 元数据。

`candidate.plan` 只是 VLM 输出的可审计表示，不能覆盖固定 Domain，不能生成因果边，也不构成执行 authorization。Repair 后另写 `certified.plan`；只有其 VAL certificate 可被 Compiler 和 Controller 接受。

所有实验产物保存在 run-specific 目录，prompt 与 schema 存在版本控制目录。任何密钥都不得进入这些文件。

## 5. 固定 LOGIV PDDL Domain

### 5.1 抽象粒度

实际执行 Domain 使用与 π₀.₅ 训练分布更接近的高层 VLA 宏动作，而不是强制把所有任务拆成裸 `Pick`/`Place`：

- `place-on(object, from, surface)`
- `place-in(object, from, container-region, access)`
- `place-relative(object, from, target-region)`
- `open-access(access)`
- `close-access(access)`
- `turn-on(device)`
- `turn-off(device)`
- `put-down(object, surface)`
- `place-held-on(object, surface)`
- `place-held-in(object, container-region, access)`
- `place-held-relative(object, target-region)`

一个 placement 宏动作内部允许 π₀.₅ 完成接近、抓取、搬运和释放。这样既与现有 policy 的自然语言能力匹配，也避免单夹爪内部资源把所有多物体任务不必要地编译成全序链。

nominal placement 的 precondition 包含 `handempty`，成功后仍建立 `handempty`；其内部抓取过程不暴露为新的 occurrence。若 effect failure 后 oracle 确认物体仍被夹持，Current Problem 使用 `holding(object)` 且不包含该物体的 location。首版固定 Domain 已包含 `place-held-*` 与 `put-down` 恢复动作，Repair 可以完成当前放置或先安全放下后重试；VLM 不能临时发明这些 schema。

### 5.2 符号映射

BDDL 的 `On`、`In`、`Open`、`Close`、`Turnon` 和 `Turnoff` 被映射为 LOGIV 的受限 typed STRIPS facts。对象与区域使用原始 registered IDs；`On/In` 的关系类型保存在 location 类型和 sidecar 中，不能仅靠自然语言猜测。

互斥变量至少包括：

- 每个 movable object 在可信离散状态下恰好处于一个 location 或 `holding(object)`，二者恰一；
- `handempty` 与任一 `holding(object)` 互斥，单臂设置最多存在一个 holding object；
- 每个 access 恰好是 open/closed 之一；
- 每个 switchable device 恰好是 on/off 之一。

Domain 的 Python action model 是生成、修复和 DAG 编译的单一语义源，并确定性渲染出公开的 `.pddl` 文件。测试必须比较渲染结果并让 VAL 验证代表性合法/非法计划，防止 Python 与 PDDL 漂移。

## 6. VAL wrapper 与证书

Wrapper 只返回 `VALID`、`INVALID` 或 `VALIDATION_ERROR`。超时、进程错误、解析错误和无法识别输出全部属于 `VALIDATION_ERROR`，不能伪装成普通不可行计划。

每次调用把 Domain、Problem、Plan、occurrence sidecar、VAL binary/version、wrapper version、调用参数、retry ledger version 和 forbidden retry keys 的 SHA-256 写入 certificate。Compiler 与 Controller 在使用前重算哈希；不匹配即终止，不尝试宽松恢复。

Repair 可用本地 STRIPS 状态转移进行候选剪枝，但最终 authorization 必须来自外部 VAL 对完整计划的验证。本地模拟器不能代替 VAL。

## 7. Causal DAG 与“不能退化为链表”

每个 certified action occurrence 对应一个节点，另加不可执行的 `INIT` 和 `GOAL`。边只表达必要 precedence：

- causal support：source 提供 target 的 precondition 或 Goal fact，且支持在中间没有被删除；
- conflict protection：保持一个已认证 causal link 所必需的顺序。

编译器禁止为了复刻 VAL 线性计划而添加 `step[i] → step[i+1]` 邻接边。原线性索引只用于 Kahn topological sort 的稳定平局规则，形成唯一 Canonical Agenda。

不能用伪造分支满足“非链式”要求：若某任务的必要因果关系确实是链，图可以是链。结构性验收要求是：

- synthetic independent-goals fixture 的最大拓扑层宽度至少为 2；
- 双 moka pot 任务的两个 nominal `place-on` occurrence 之间没有边，二者分别支持 `GOAL`；
- Compiler 输出中不存在仅由相邻线性索引产生的边；
- 任一 repair slice 只包含受影响 branch 及其必要前驱，不把无关 sibling branch 强制拉入。

Canonical Agenda 仍然一次分派一个 occurrence；“图是部分序”不代表单臂机器人并行执行物理动作。

## 8. 仿真 attempt、STOPPED 与 fact grounding

### 8.1 π₀.₅ subtask attempt

每个 high-level occurrence 生成固定模板的 subtask prompt：只描述当前动作、对象、目标、可观察停止条件和禁止的额外目标。一个 attempt 内每次 policy inference 复用同一 prompt；Controller 不让 π₀.₅自行选择下一个 subtask。

当前 OpenPI client 返回本地 action chunk，`env.step` 是同步调用。因此仿真 `STOPPED` fence 定义为：

1. 当前 `env.step` 已返回；
2. 丢弃本地尚未执行的旧 action chunk；
3. 不存在异步命令消费者或未确认 token；
4. 使用独立的 simulator hold/settling protocol 后采集新 observation。

settling command 必须保持最后一个可信 gripper 状态，不能无条件使用会张开夹爪的现有 dummy action。真实机器人适配器不得复用这个仿真假设，必须等待最终命令消费者的 `STOPPED(attempt_id)` acknowledgement。

### 8.2 facts

本阶段 `LiberoOracleGrounder` 从以下确定性来源形成 snapshot：

- LIBERO object/site predicates：`On`、`In`、`Open/Close`、`Turnon/Turnoff`；
- robot proprioception 与 simulator contact/grasp 状态；
- registered object/location mapping 和 exactly-one lint。

同一 post-action snapshot 在有效期内依次用于当前 effects、下一动作 preconditions 和空 Agenda 后的 Goal gate。每个 physical dispatch 后必须产生新 epoch；没有新 dispatch 时不重复 grounding。

## 9. Controller、提交与恢复

Controller 持有 active graph version、certificate、agenda cursor、active attempt、receipts、retry ledger 和三个全局预算。动作只有在 fresh facts 满足 preconditions 时才能 dispatch。

Effect 成功时，在一个事务中写 completed receipt、提交 occurrence、推进 cursor 并清除 active attempt。Effect 失败时，失败动作没有提交，因此 remaining plan 从该 occurrence 开始。

恢复严格区分：

- precondition failure：直接构造缺失 precondition obligation；
- empty-agenda goal failure：直接构造缺失 Goal obligation；
- effect failure：只在此处对未修改 remaining plan 做一次完整 VAL recertification。

若 effect-failed suffix 仍 VAL-valid、action 可重复且预算允许，则创建新 attempt。否则 deterministic trace 产生最早 ACTION_FAILURE 或最终 GOAL_FAILURE；若只因 retry 禁止而不能重试，则产生 `EXCLUDED_RETRY_RECOVERY`。

Repair 从 Current Problem、原始 Goal、causal slice、旧 canonical ranks 和 forbidden retry keys 出发，在固定 grounded action 集合中进行有界搜索。候选排序使用确定性代价：retry-policy violation 首先淘汰，其后依次比较是否保留已满足 Goal、编辑距离、plan length、旧 rank 和 grounded action lexical key。每个候选去重后才消耗 VAL budget。

新计划必须同时满足 VAL 与 RetryPolicy。安装只在没有运行 attempt 时进行，并原子替换 graph、agenda、cursor、certificate 和 source epoch；旧 graph 不原地修改。

## 10. 有界性和终止

实现使用跨 graph version 不重置的：

- `max_physical_attempts`
- `max_repair_rounds`
- `max_total_val_calls`

Repair 另有限制 `max_edits`、`max_candidates` 和 `val_timeout_seconds`。所有预算均在操作前原子 check-and-increment。耗尽预算、grounding failure、VAL error、compiler/install error 或 fence failure后进入 `TERMINAL / no further dispatch`。

只有 simulator adapter 已同步返回或真实 executor 已确认停止时，报告才可以使用 `Safe Stop`。`NO_CERTIFIED_REPAIR_WITHIN_BUDGET` 不得表述为全局无解。

## 11. Prompt 版本化与优化纪律

需要版本化两类 prompt：

- Proposal prompt：未来 VLM API 使用；当前 Scripted-VLM fixture 按同一 schema 人工生成。
- π₀.₅ subtask prompt：真实参与 simulator 执行，是当前可以测量和优化的 prompt。

每个 prompt version 保存模板哈希、任务绑定、运行配置和结果。优化只使用预先划分的 development initial states，优先覆盖历史失败 episode；最终 holdout 在 prompt 锁定前不可查看其 LOGIV 结果。每次只改变一个 prompt 因素，并保存相同 initial-state 集上的前后对比。

建议将每个任务前 20 个 initial states 作为 development，后 30 个作为 locked holdout。最终论文主结果仍运行全部 50 states，并明确 prompt 开发使用了哪些 states。

## 12. 测试与真实仿真验收

### 12.1 自动化测试

单元和集成测试至少覆盖：

- Domain render/lint、registered IDs、type errors 和 exactly-one violations；
- VAL 的 valid、invalid、validation-error 和 hash mismatch；
- candidate plan 与 occurrence sidecar 持久化；
- support edge、conflict edge、edge-reason merge、cycle/self-loop rejection；
- 真实分支 DAG 和 canonical tie-breaking；
- precondition/effect/goal gates 与 snapshot reuse；
- effect failure remaining plan 包含失败 occurrence；
- zero-edit recertification 只发生在 effect failure；
- repeatable retry、forbidden stable retry key 和跨 graph lineage；
- STOPPED-before-replace、atomic commit/install、stale install conflict；
- 三个预算耗尽后无 further dispatch；
- 黑碗任务文档中的四类恢复，以及多 producer 恢复顺序。

测试使用同步 fake executor/fact world 验证精确状态机，再使用 LIBERO oracle adapter 做 simulator smoke。不能用纯 mock 通过来声称真实 π₀.₅ 提升。

### 12.2 分阶段 simulator 评估

1. 先在 task 8 双 moka pot 和 task 9 微波杯任务的 development states 上运行 baseline-compatible smoke。
2. 针对失败轨迹调整 π₀.₅ subtask prompt；每次调整保留完整轨迹、事实、VAL 和 attempt records。
3. prompt 锁定后运行 task 8 的 50 个固定 states，目标从 27/50 提升到至少 32/50。
4. 再运行完整 LIBERO-10 10×50。最低目标是超过当前 460/500，同时报告逐任务成功率、Wilson 95% interval、attempts、repair success、VAL calls、terminal causes 和 DAG width。

由于 LOGIV 会改变 inference request 数量和 policy RNG 消耗，不能把它宣传成严格同 RNG 的 paired action-level 比较。可比较项是相同 checkpoint、相同 50 个 initial states、相同环境和 horizon 下的 episode outcome；报告同时给出按 initial-state 配对的描述性变化与不依赖错误 RNG 假设的置信区间。

若首轮没有提升，先按系统化调试顺序区分：错误事实、错误 subtask boundary、prompt 分布偏移、attempt 步数不足、宏动作 effect 判定、repair 重复错误动作，以及低层 policy 本身无法完成该子任务。每轮只验证一个根因假设；连续两轮同类修改失败后必须换本质不同的方案。

## 13. 与论文初稿的明确差异

- 代码中的 nominal placement 默认使用高层 VLA 宏动作；论文黑碗示例若继续使用 `PickBowl/PlaceBowl`，必须说明它是更细粒度的说明性 Domain，或改成与公开执行 Domain 一致的 `PlaceIn/Close`。
- 仿真 STOPPED 是同步客户端 fence，不是现实机器人 acknowledgement；论文正文只保留抽象合同，附录说明适配器差异。
- 无 API 阶段的结果属于 Scripted-VLM + oracle grounding，不属于 GPT-4o goal-prediction setting。
- “图不能退化成链”实现为“禁止线性邻接边，并要求真实独立任务产生分支”，而不是为每个本来串行的任务伪造分支。

## 14. 成功判据

实现完成必须同时满足：

1. 所有新增和既有自动化测试通过，且关键行为有已观察的 red-green 证据。
2. 真实 VAL 验证固定 Domain 与代表性 10-task plans。
3. task 8 的 nominal graph 有真实分支，Compiler 未把所有计划复制成链。
4. 故障注入测试证明任何 deviation 后都不会在未重新授权时继续 dispatch。
5. 至少完成 task 8 的固定 50-state LOGIV 评估并与 27/50 基线比较；只有实测超过基线才报告“提升”。
6. 完成全 10×50 后，才可以对总体 π₀.₅ success rate 作提升或回退结论。
