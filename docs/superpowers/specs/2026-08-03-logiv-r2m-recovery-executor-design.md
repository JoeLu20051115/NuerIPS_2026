# LOGIV-R2M：Base 保底的执行器感知恢复设计

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-03
- Verification Status: DESIGN_REVIEWED_IMPLEMENTATION_UNVERIFIED
- Version Label: code_plan_v2
- Status: 第二轮逻辑修订冻结候选；设计审查结论与实现/经验验证分开记录
- Upstream Evidence:
  - `docs/experiments/2026-08-03-logiv-v53-failure-seed-retrospective.md`
  - `docs/superpowers/specs/2026-08-02-logiv-libero-closed-loop-design.md`
  - `docs/superpowers/specs/2026-08-02-logiv-frontier-execution-optimization-design.md`

## 1. 决策摘要

后续主线停止继续优化“从第 0 步接管并把完整任务拆成多次冻结 π₀.₅ 调用”的
`FULL_LOGIV`。新主线采用 **LOGIV-R2M（Repair-to-Manifold）**：

1. 原始 `π_base` 使用官方完整任务文本连续执行，checkpoint 和 nominal 行为保持不变。
2. episode 开始时保留一次 `InitialProposal`：VLM/ScriptedProposal 只产生候选初始状态、
   子目标与计划；输出必须通过 coverage、schema、signed-state 和 VAL gate，且不能改变
   `π_base` 的 prompt、动作或 policy RNG。
3. LOGIV 在影子模式维护 Ground Facts、Goal、版本化 Causal DAG、已完成目标与剩余义务；
   Initial DAG 只是可失效的执行先验，不是跨 episode 永久有效的 authorization。
4. 稳定异常只有同时具备历史失败证据、排除正常中间态且绑定 fresh certificate 时，才是
   confirmed deviation；恢复能力与预算属于后续 handoff permit，不属于失败判定本身。
5. 一个独立训练的 `π_recover` 连续执行与 VLA 训练粒度一致的恢复宏，不在
   `holding` 等中间状态切断策略上下文。
6. 外部状态监督器在 fresh observation 上验证目标 effect、protected invariants 和剩余
   预算；之后必须由 native Goal success 或经 held-out continuation 验证的 Base re-entry
   二者之一闭合提交合同。Phase 1 的 Task 5/8 主臂只声明 terminal local recovery，不用
   terminal success 冒充已经证明的 Repair-to-Reentry。

该决策保留 LOGIV 的规划、认证、局部修复、事实门控与审计逻辑，只撤销已经被实验
否证的假设：任意符号合法动作都能由冻结 π₀.₅ 通过一次零样本 prompt 可靠执行。

## 2. 证据与问题定义

当前同协议选定结果为 `FULL_LOGIV=427/500`、`Base=437/500`，净差为
`-10/500`。Task 8 的 non-destructive overlay 诊断进一步定位了断点：LOGIV 能正确识别
“一只 moka pot 已在 stove、另一只在 recovery surface”，也能生成并验证合理的局部
repair；但冻结 π₀.₅ 在 recovery-induced state 上会超时、误拿已完成物体，或者无法在
剩余预算内完成恢复。

这意味着 VAL 只能证明符号计划合法，不能证明物理执行器具备对应技能。旧设计的实际
失败链是：

```text
合法 PDDL action
→ 任意自然语言 subtask prompt
→ 冻结 π₀.₅ 的分布外执行
→ effect 失败或 collateral damage
→ 再次分段/重试
→ 520-step 预算耗尽
```

同时，Base 的 63 个失败中 Task 5 有 26 个、Task 8 有 22 个；两者合计占 48/63。
因此第一阶段应优先建立 `place-in` 与 crowded-target `place-on` 的恢复闭环，而不是在
十个任务上平均投入数据和算力。

## 3. 目标、非目标与研究问题

### 3.1 主要目标

- 新系统未触发时逐动作复现 `π_base`，使 nominal 性能下限回到当前 Base，而不是从
  旧 Full 的 427/500 起步。
- 在相同官方 520 policy-action budget 下，利用 LOGIV 认证的恢复技能净救回 Base
  失败，且尽量不产生 Base-success → LOGIV-failure 的负翻转。
- 保留论文中 LOGIV 的核心贡献：grounded planning、完整 VAL、真实非链 DAG、localized
  repair、effect/invariant gate、bounded termination 与可审计 provenance。
- 将理论保证明确条件化为经过测量的 executor effect-success、non-interference 与
  step-cost 合同，不再作无条件 frozen-backbone 非降声明。

### 3.2 研究问题

> 在相同 π₀.₅-LIBERO Base checkpoint、相同初始状态、相同 policy/simulator seed 和
> 520-step 总预算下，经过能力校准的 LOGIV 局部恢复是否能提高 LIBERO-10 等权任务
> macro success rate，并保持非触发 episode 与 Base 行为一致？

### 3.3 非目标

- 不继续通过大量 task-specific prompt/horizon 宽容度调参挽救旧 Full controller。
- 不训练一个从任意失败状态直接完成所有 LIBERO-10 任务的通用新 VLA。
- 不让 learned critic 替代 VAL，也不让 VAL validity 冒充物理技能成功率。
- 不把 privileged simulator data collection 或 oracle grounding 表述成真实机器人感知。
- 第一阶段不处理 Task 5/8 之外尚未通过能力验证的恢复 schema。

## 4. 备选路线与取舍

### 4.1 采用：Base-preserving recovery overlay

`π_base` 保持原样，LOGIV 只在可靠偏差后调用独立 `π_recover`。优点是 nominal 路径
风险最小、旧 LOGIV 逻辑保留充分、收益和损害可通过 paired flips 直接解释。缺点是
只能救回能较早检测且剩余预算足够的失败。

### 4.2 否决为主线：trained Full LOGIV option policy

从 episode 开始就由 LOGIV 分派训练后的 option。它最完整地保留控制权，但仍会产生
频繁切换、动作上下文丢失和多任务数据需求；在 Base 已有 437/500 时，下行风险过高。
它只作为未来消融或第二阶段研究，不作为当前补救主线。

### 4.3 仅作辅助：failure critic / value gate

critic 可以估计某次修复的成功率、干扰风险和耗时，却不能凭空补齐缺失的低层恢复
技能。因此它可用于能力门控和计划排序，但不能代替 `π_recover`。

## 5. 总体架构与组件边界

系统由九个可独立测试的组件组成：

1. `NominalPolicyClient`：只向原始 checkpoint 发送官方完整任务文本；接口行为与 Base
   evaluator 相同。
2. `InitialProposalProvider`：在 episode 初始 observation 上调用一次 VLM 或版本化
   `ScriptedProposalProvider`，产生候选 Problem/actions/DAG seed；输出无效时只禁用 LOGIV
   intervention，不影响 `NominalPolicyClient`。
3. `ShadowLogivMonitor`：读取 observation 和外部动作事件，维护 fresh facts、已完成目标、
   异常候选、历史失败证据与当前剩余义务；没有控制权限。
4. `ShadowCertificateReconciler`：只在任务相关 signed-fact 投影上比较 observed transition
   与当前 DAG；覆盖外变化会令证书 `STALE`，重建前禁止产生 handoff permit。
5. `RecoveryPlanner`：从 handoff state 重建 Current Problem，调用原有 Repair、VAL 和
   DAG compiler，只产生局部、certificate-bound recovery plan。
6. `RecoveryCapabilityRegistry`：声明哪些 grounded recovery option 有训练支持，保存
   effect success、protected-invariant violation、step-cost 与数据/checkpoint provenance。
7. `RecoveryPolicyClient`：只连接独立 `π_recover` checkpoint，每次 permit 只执行一个
   有限模板定义的原子 recovery macro。
8. `ExternalInvariantSupervisor`：独立于 recovery prompt 和 policy 输出，在执行前、执行中
   和停止后验证 declared effect、已完成 Goal invariants、native Goal/certified Base re-entry
   与 budget reserve；它是 recovery commit 的唯一事实来源。
9. `OverlayController`：原子完成 Base → Recovery → Base 的切换、action-chunk flush、
   budget accounting、effect/invariant gate 和 terminal handling。

```text
official task → π_base → action stream ───────────────────────────────────┐
initial obs → InitialProposal → advisory shadow DAG                       │
                                  ↓                                      │
                 facts/events → reconcile certificate                    │
                                  ↓                                      │
           confirmed deviation with historical evidence? ── no ──────────┘
                                  │ yes
                                  ↓
             fresh Current Problem + protected facts + fresh certificate
                                  ↓
                      Repair + complete VAL + local DAG
                                  ↓
                 capability / interference / budget / retry gate
                         │ reject                 │ permit
                         ↓                        ↓
                   continue π_base        one π_recover macro
                                                  ↓
                                ExternalInvariantSupervisor
                                                  ↓
                effect ∧ invariants ∧ budget ∧ (goal ∨ certified re-entry)
                                  │ fail                │ pass
                                  ↓                     ↓
                         bounded terminal        success or π_base
```

组件之间只交换版本化 records，不共享可变内部状态。`RecoveryPlanner` 不知道模型服务
实现，`RecoveryPolicyClient` 不决定任务 Goal，能力注册表不修改 PDDL 语义，prompt 中的
`Preserve` 文本也不能替代 `ExternalInvariantSupervisor` 的物理事实验证。

### 5.1 InitialProposal 合同

系统层面保留 LOGIV 原始的开局 VLM 预测，但它是 proposal，不是执行 authorization：

- 每个 episode 最多产生一个 accepted initial proposal；重复/迟到响应按 context generation
  记录为 stale no-op。
- Phase 0 只允许确定性的本地 `ScriptedProposalProvider`；普通 step-0 路径恰好发出一次
  proposal request。若 callback 输入深拷贝本身失败，则记录 `NOT_ATTEMPTED`、零请求并继续
  Base。未来 remote provider 必须异步且由 client 强制 deadline；同步 Python wrapper 不宣称
  能抢占任意阻塞外部调用。
- `METADATA_ASSISTED` 主臂冻结官方 BDDL Goal；当前 `ScriptedProposalProvider` 只提出
  candidate facts/actions，不允许改写 Goal。
- `GOAL_PREDICTION` 扩展臂才允许真实 VLM 提出 Goal，并单独报告 Goal accuracy、请求和
  失败率，不与 metadata-assisted 主表混合。
- candidate 必须通过 coverage manifest、registered object/schema、signed-state、VAL 和
  DAG lint，才能供 shadow planner 使用。
- signed-state/VAL certification 必须使用共享 Base rollout 在 reset/set-init/wait 后复制的
  step-0 live observation 所产生的 fresh Ground Facts；不能用 proposal 自己携带的 initial
  snapshot 作为独立验证证据。该只读 callback 位于第一个 Base policy request 之前，异常
  fail open 且恢复 Python/NumPy RNG 状态。
- Phase 0 的 `initial_epoch_id` 冻结为 0；provider request、proposal package、live step-0
  `FactSnapshot.epoch_id` 和 initial certificate source epoch 必须完全相等，任一错配 fail open
  拒绝 proposal，不能隐式取默认 epoch。
- proposal 拒绝、超时或异常只令 `logiv_intervention_enabled=False`；`π_base` 继续使用官方
  task prompt 原样执行，不能因为 LOGIV 初始化失败而终止 episode。
- accepted proposal 也不能改变 Base prompt、action chunk、policy request generation 或
  RNG。只有后续 confirmed deviation + fresh certificate + capability permit 才能发生物理
  接管。

### 5.2 Shadow certificate 与动态一致性合同

Initial DAG 表示一条对初始 observation 合法的偏序计划，不表示 `π_base` 必须采用该
顺序或 producer。它只能提供预期 effects、正常中间态和因果依赖的影子先验。每个可用于
监控或修复的 `ShadowCertificate` 至少绑定：

```text
ShadowCertificate := (
    relevant_fact_hash,
    goal_hash,
    graph_version,
    source_epoch,
    observation_generation,
    grounding_rule_hash
)
```

`relevant_fact_hash` 只覆盖 Goal、注册 action precondition/effect、holding/access、protected
facts 和 monitor 事件所需的 signed predicates；不使用每帧都会变化的原始 simulator-state
hash 判断逻辑证书新鲜度。原始 physics-state hash 仍用于 root artifact 精确复现。

每次 monitor observation 先在该相关谓词投影上执行 reconciliation：

1. transition 被当前 DAG 的 effects/偏序覆盖：更新 achieved/remaining obligations，并签发
   同 graph version、递增 observation generation、绑定新 relevant-fact hash/epoch 的 shadow
   certificate；
2. transition 是保持 Goal/invariants 的替代合法 producer：当前证书标记 `STALE`；完整系统
   只能从 fresh facts 显式本地重编译，Phase 0 则停在 STALE 并仅记录 provenance；
3. transition 既未覆盖又可能破坏 Goal、正常阶段或计划假设：标记 `STALE`，只记录异常
   candidate，重建前不得授权接管；
4. 仅相机噪声、连续位姿或无关 predicate 变化不使证书失效。

`STALE` 是粘滞状态：`CURRENT → STALE → STALE`。后续无变化或又出现一个被旧 DAG 覆盖的
transition 都不能清除失效标记；只有从 fresh facts 成功重建、通过 VAL/DAG lint 并获得新
graph version 的显式 reconciliation 才产生 `RECONCILED/CURRENT`。重建失败保持 STALE。

Initial certificate 永远不能直接授权物理恢复。handoff 必须从当前 fresh snapshot 重建
Current Problem，生成新的 recovery certificate 和 local DAG；dispatch 前再次检查当前
`relevant_fact_hash`、epoch 和 observation generation 与 permit 完全一致。任一不一致均按
TOCTOU/stale no-op 拒绝接管，而不是沿用旧 DAG。

## 6. Nominal 非破坏合同

新系统必须首先满足下列硬合同：

- monitor inactive 或处于纯 shadow 方法臂时，Base 的 prompt、policy seed 消耗、action
  arrays、replan boundary、总 policy-action steps、`π_base` policy inference requests 和
  native task outcome 逐项相同。
- LOGIV 的 shadow grounding 和 VAL 计算不得清空 Base action chunk。
- 只有 `OverlayController` 持有单一切换权限；切换时等待当前同步 `env.step` 返回，再
  丢弃尚未执行的本地 chunk，记录 flush 数量和 handoff step。
- recovery 完成后创建新的 policy request generation，禁止 Base/Recovery 的迟到结果
  跨 generation 生效。
- external native LIBERO evaluator 是 episode success 的唯一裁决者。

Base request provenance 由 rollout protocol 在发出请求前原子写入不可变 envelope log，记录
episode seed、zero-based request index 和冻结 policy-client/checkpoint config hash；server 回显
匹配后标记 acknowledged。callback 同时获得已产生当前 logical chunk 的 active index/envelope
和下一请求 index/replay envelope，不能从递增后的 client counter 或
`ceil(policy_step/replan_steps)` 反推。协议侧 clock、RNG、hash、deep-copy、envelope-read 与
observer-escape 异常写入结构化 `ShadowFailureRecord(policy_step, stage, reason)`，使 step-0
observer 未进入时仍能审计 `NOT_ATTEMPTED`，同时不终止 Base。

主 episode record 自身保存 `initial_proposal_status/reason_code/request_count`，不依赖 sidecar
才能校验：BASE 只能是 `NOT_APPLICABLE/0/no-reason`；Shadow 的 ACCEPTED/REJECTED 必须恰好
一次 request，NOT_ATTEMPTED 必须零次；后两种失败状态必须有稳定 reason code，ACCEPTED
不得有。完整异常消息只留在受控诊断 artifact。

第一阶段主结果中的 `ShadowLogivMonitor` 限定为本地、只读模块：使用已声明的 oracle/
registered facts，在既有 replan boundary 检查结果，不等待外部 VLM/LLM，也不改变动作
调度。它可以增加 wall-clock latency 和本地计算量，因此必须单独报告 monitor calls、
grounding/VAL latency、CPU/GPU time 与峰值内存。

未来如果接入在线 VLM/LLM，必须使用独立 client 和 RNG、异步非阻塞请求，并把
`base_policy_requests`、`initial_proposal_requests`、`shadow_vlm_requests`、
`recovery_policy_requests` 分桶记录。此时
不能再声称“总推理请求与 Base 相同”，只能声称动作流、policy-action steps 和未接管
episode outcome 与配对 Base 相同。外部请求结果只能在既有 observation generation
边界被采纳，Controller 不得停住 Base 等待响应。

这使新方法的工程起点是 Base 437/500；旧 Full 427/500 只作为消融，不是新系统需要
先追回的性能债务。

## 7. 偏差触发与预算门控

### 7.1 第一阶段可触发偏差

第一阶段把“异常候选”“确认偏差”和“接管许可”分为三种不同 records。稳定位置本身只
能产生 anomaly candidate，不能证明发生失败，也不能授权 recovery：

```text
anomaly_candidate := stable_bad_fact

confirmed_deviation :=
    anomaly_candidate
    ∧ historical_failure_evidence
    ∧ phase_eligible
    ∧ ¬normal_or_transient_intermediate

handoff_permit :=
    confirmed_deviation
    ∧ fresh_recovery_certificate_valid
    ∧ capability_available
    ∧ intervention_advantage_calibrated
    ∧ interference_gate_passed
    ∧ budget_sufficient
    ∧ retry_ledger_allows
```

能力或预算不可用不会抹掉 failure record：系统仍保存 confirmed deviation/root 供后续数据
闭环，但不得物理接管。这样 trigger recall、permit rate 和 recovery coverage 可以分别报告，
也避免只记录已有技能能够处理的失败而产生选择偏差。

同一 `deviation_event_id` 的记录状态只能单调变化：`UNSEEN → ANOMALY_CANDIDATE →
CONFIRMED_DEVIATION`。历史证据晚于第三个稳定 sample 到达时允许把 candidate 升级一次，
不能因 candidate 已经写盘而压制后续确认；graph/certificate hash 改变不能制造新的事件或
把 confirmed 状态降级。首次稳定时强证据已经存在则只写 confirmed record，不在同一
snapshot 重复写 candidate 和 confirmed 两条 root。

第一阶段的 `historical_failure_evidence` 只接受预注册、可审计的事件：

1. **Goal regression**：某 signed Goal literal 曾被 fresh observation 确认为已实现，随后
   稳定变为相反值；
2. **Attempted-effect timeout**：外部 action-event detector 已确认 task-relevant
   grasp/place/release attempt，但其预注册 effect 在对应 deadline 内未成立；
3. **Abnormal transfer**：物体在 task-relevant contact/manipulation 后由 nominal source
   转移到 registered abnormal surface；
4. **Progress timeout**：仅可作为弱证据，必须与最近的 task-relevant attempt 或异常转移
   联合，第一阶段不得单独授权接管。

action-event detector 从 gripper 开合、末端/物体接触、holding 过渡、物体与末端的相关
运动和 registered region proximity 产生版本化事件；不能把 VLA 的连续 7D action 直接
解释为已经完成某个符号 `grasp/place`。事件定义、deadline 和 grounder 版本必须写入 root
artifact。

每个 callback 的 feature reader 必须按 applicable rule 产生记录，并显式包含
`rule_id/object_id/source_region/destination_region`；tracker 对照冻结的 `ActionEventRule`
逐字段验证，拒绝同一 rule/step 的重复记录和任何身份错配。raw transition hash 只覆盖这些
登记 feature、policy step 与 rule/tracker version，不覆盖 pixels 或未登记观察键。reader
提供 raw gripper qpos、contact count、holding truth、region truth/distance 与 motion correlation；
open/close transition 和 region-distance gate 由 tracker 按 contract threshold 计算，不能由
reader 传入预标注事件绕过冻结阈值。reader 与每条 feature 都携带并校验
monitor-contract hash/tracker version；每个注册 rule 每 callback 恰好一条 record。缺键用
canonical null/UNKNOWN，不用 0/NaN sentinel；null、非有限值、缺/重复 rule 或版本错配会断开
该 rule 的 transition continuity，只能产生 diagnostic，不能桥接出强证据。

具体 tracker 状态机必须区分 attempt 与 failure evidence：组合物理特征成立后才创建
`ATTEMPT_ACTIVE`；effect 在 due 前/当步为 TRUE 则取消 timeout；due 时为 FALSE 才发出一次
`ATTEMPTED_EFFECT_TIMEOUT`；UNKNOWN 只记诊断，不产生强证据。evidence 只在独立 TTL
窗口内可 join，过期不能因 graph rebuild 复活。abnormal-transfer 也必须关联同对象、同规则
的已登记 manipulation attempt，不能由静态位置直接生成。

effect TRUE 只终止 timeout 分支，不能删除 recent-manipulation provenance。每条 rule 另有
从 attempt start 起算的绝对 `manipulation_attribution_ttl_policy_steps`；在该窗口内，“成功
holding 后同对象落入异常区域”仍产生 abnormal-transfer，异对象或过期后不得产生。一个
attempt 可分别产生 timeout 与 abnormal-transfer，但每种最多一次；`evidence_id` 绑定
attempt/kind/emission/support hashes，去重按 evidence ID 和 `(attempt_id, kind)`，不是禁止同一
attempt 出现不同类型证据。

历史证据不是无对象的字符串，而是每个 Base action/observation 都更新的结构化 record：

```text
ActionEventEvidence := (
    evidence_id,
    evidence_kind,
    rule_id,
    object_id,
    attempted_effect,
    source_region,
    destination_region,
    attempt_id,
    start_policy_step,
    effect_due_policy_step,
    emitted_policy_step,
    evidence_expires_policy_step,
    supporting_transition_hashes,
    detector_hash
)
```

稳定 fact monitor 仍可每 5 步采样，但 event tracker 每个 Base policy action 都读取复制的
observation/action，不能漏掉采样间的 gripper/contact/holding 过渡。确认时必须按
`object_id + attempted_effect/source/destination` 精确 join，并要求当前 step 已到
`effect_due_policy_step` 且未超过 `evidence_expires_policy_step`；对象 A 的 timeout、未到期/
已过期 attempt 或无关 effect 不得确认对象 B 的异常。证据 ledger
按 attempt/evidence ID 有界保存并确定性过期。

timeout 的 due 为 attempt start 加预注册窗口；due 时 UNKNOWN 会把该 timeout attempt 关闭为
inconclusive，后续 FALSE 不得追溯补发。可即时观测的 `GOAL_REGRESSION` 与已登记 manipulation
后的 `ABNORMAL_TRANSFER_AFTER_MANIPULATION` 把 due 设为 emission step，expiry 仍为 emission
加独立 TTL，因而共用同一个 active-window 检查而不会被 timeout 窗口错误延迟。

`GOAL_REGRESSION` 由 fact observer 从“首次可靠满足”的 Goal evidence hash/step 与“首次可靠
相反”的 hash/step 构造一次 typed record：start 是 achieved step，due=emitted=regression step，
attempt ID 绑定 Goal literal 和 achieved provenance，expiry 使用 contract 的独立 Goal-regression
TTL。UNKNOWN 不发证也不延长 TTL。它仍受 detector/contract hash、累计 evidence cap 和事件
状态单调性约束，不能退化成无来源字符串。fact observer 必须通过 tracker 的
`record_fact_event` 写入 Goal regression/progress records，使它们与 manipulation evidence
共享同一个累计 cap、去重表和 overflow metric，不能各自获得一套 128 条配额。
两类 fact event 的 rule ID 由公共 `ReservedFactEventRuleId` 枚举唯一声明；普通 action rule
禁止使用 `__...__` 保留命名空间，loader/observer/test 不得各自散写字符串常量。

每个 Phase 1 state class 必须冻结 `MonitorEvidenceContract`：task/object types、nominal
sources、abnormal surfaces、task-relevant effects、gripper/contact/holding 事件规则、
concrete versioned `ActionEventRule`（gripper 开闭阈值、contact 数、holding/region truth、
EEF/object motion correlation、以 Base policy step 表示的 effect due 和 evidence TTL）、
`monitor_interval_steps`、`confirmation_count`、
`settling_grace_observations`、`progress_window_observations` 和
`progress_evidence_ttl_policy_steps`、`goal_regression_evidence_ttl_policy_steps`、
`max_active_attempts_per_object`、`max_attempt_records_per_episode`、
`max_evidence_records_per_episode` 与 contract hash。初始开发默认 monitor interval=5、
每条 effect rule due=15 Base steps、manipulation attribution TTL=30 Base steps、evidence
TTL=20 Base steps、
confirmation count=3、settling/progress 分别为 2/4 个 monitor observations（即 10/20 Base
steps）、progress/Goal-regression TTL=20 Base steps；progress
window 只产生弱证据。阈值可在 Phase 0/2 development 上修改，但进入
locked confirmation 前必须冻结，不能用主结果回选。

Phase 0/1 的 event ledger hard cap 为每对象 8 个 active attempts、每 episode 累计 128 条
attempt records 和累计 128 条 evidence records；完成/过期不退还累计配额。到达容量时拒绝
新增并报告 overflow，不能为腾空间静默删除未过期强证据，也不能
让 overflow 本身成为 failure evidence。

`normal_or_transient_intermediate` 至少包括：对象仍在声明的 initial/nominal source 且无
操作证据；当前 holding/释放/settling 过渡；reconciled DAG 覆盖的合法替代顺序；以及
task/state-class 预注册 grace window 内的接近和调整。该谓词必须由有限白名单实现，不能
依赖自由文本判断。

第一阶段 anomaly patterns 限定为：

- Task 5 类：目标 book 稳定处于 registered abnormal recovery surface，且不在目标 caddy
  region；
- Task 8 类：一个目标 pot 已稳定在 stove，另一个 pot 稳定处于 registered abnormal
  recovery surface；或 task-relevant attempt 后当前 holding object 与 remaining target
  不一致。

“稳定”要求同一 candidate signature 在三个连续 monitor observation 中为 TRUE；UNKNOWN、
单帧 location divergence 或抓取/释放过渡态重置 streak。monitor observation 默认与
`replan_steps=5` 对齐，因此三次确认最多占用 15 个 Base policy steps，但不增加动作预算。
Phase 0 可以保存 anomaly candidate 用于 detector 开发；Phase 1 只有
`confirmed_deviation` 才能进入许可门控。

每次 monitor snapshot 都必须携带第 10.1 节完整 fact-universe audit 字段，并与 initial
certificate 的 universe version/hash 一致；Goal、relevant facts 与 contract facts 必须全部已
注册。缺覆盖、universe 漂移或 payload 不可重算都只记 snapshot error、清空稳定 streak 并
继续 Base，不能把 absent predicate 当作 UNKNOWN 后继续授权。

即使 confirmed deviation 为真，Base 也可能自行重试成功。因此 permit 不能只问 recovery
会不会做，还必须在同类 saved roots 上配对比较“继续 Base”和“立即 recovery”的最终任务
成功率。`intervention_advantage_calibrated` 使用冻结 branch seeds 的 paired lower bound：

```text
LCB95[
    P(final success | certified recovery, C)
    - P(final success | continue π_base, C)
] > 0
```

未通过时仍记录偏差但继续 Base。该 gate 是 state-class 级经验合同，不宣称单 episode 的
反事实可被直接观测；Phase 2 的真实 positive/negative flips 仍是最终裁决。

该 gate 的 sampling frame 是在看到任一分支 terminal outcome 前前瞻登记的全部 eligible
confirmed deviations，而不是事后筛选出的 Base failures。每个 `independence_unit_id` 只指定
一个 primary root/outcome；其 continue-Base 分支先执行保存的 pending action suffix，并从保存
的 Base request count、active/next zero-based indices 与 policy replay envelope 延续，recovery
分支才 flush 同一 suffix。如果
隐藏 policy state 无法精确恢复，两臂必须共同 flush/reseed 并进入单独 diagnostic estimand，
不得把不同 counterfactual 协议的结果合并到 advantage gate。

### 7.2 许可条件

一个修复计划只有同时满足以下条件才能替换 Base：

1. 当前 deviation 满足第 7.1 节历史证据和阶段排除条件；若判断依赖 DAG 的 expected-phase
   信息，则该 shadow certificate 必须 current，纯 Goal regression/外部 action-event 证据可在
   shadow DAG stale 时记录失败，但仍不能直接接管。
2. 当前 snapshot 通过 exactly-one、holding/handempty 和 access state lint。
3. 从该 snapshot fresh 重建的完整 repair plan 通过 signed trace、VAL 与 RetryPolicy。
4. permit 的 relevant-fact hash、epoch、observation generation 与 dispatch 前 snapshot 一致。
5. 被选中的 physical option 存在匹配当前 state class 的已启用能力合同。
6. 被选中的单一 physical option 的 protected invariants 包含当前已完成 Goal facts。
7. `remaining_steps >= plan_step_q90 + post_recovery_reserve`；reserve 按第 8.3 节定义。
8. 候选的校准成功率和干扰风险通过第 12 节门槛，且第 13 节 retry ledger 允许。
9. 同一 state class 的 paired branch validation 通过上述 intervention-advantage gate。

permit 同时冻结本次 `option_action_cap = min(contract.hard_action_cap,
remaining_steps - post_recovery_reserve)`。Phase 1 的 `hard_action_cap` 等于预注册 step q90
且不超过 180；达到上限后必须停止、采集 fresh snapshot 并执行提交合同，不能侵占为 Base
continuation 预留的 actions。原始 maximum 仅作尾部审计，不能作为运行时透支理由。

不满足任一条件时，若 Base 仍在预算内则继续 Base；若已没有动作预算则正常记录 Base
失败。主结果不允许在 520 步后额外赠送 recovery budget。

## 8. Recovery option 与 Repair-to-Manifold 语义

### 8.1 符号层与执行层的粒度

PDDL 仍可表达 `holding`、source、destination、access 和 protected facts，但一次物理
dispatch 必须与训练数据中的连续宏对齐。第一阶段 option 为：

- `recover-place-in(object, recovery-surface, container-region, access)`
- `recover-place-on(object, recovery-surface, target-surface, protected-objects)`
- `restore-held-object(object, safe-or-goal-surface)`

`recover-place-*` 内部允许接近、抓取、搬运、放置与释放；Controller 不因中间
`holding(object)` effect 成立而 STOP/flush。中间 facts 只用于外部 invariant/safety
监控，不创建新的 policy occurrence。一次 permit 只授权一个 option；`π_recover` 不得
在完成该 option 后自行解释剩余 Goal、选择下一个 DAG node 或继续完成整个任务。
Phase 1 的 accepted repair plan 因而恰好包含一个可 dispatch physical option；规划器可以
枚举至多两个候选 option 作排序，但这不是授权顺序执行两个宏。

### 8.2 Re-entry state

为避免用 terminal recovery 冒充 Base continuation，本设计将认证目标集合显式定义为：

```text
M_certified = M_goal ∪ M_base_reentry
```

- `M_goal`：native evaluator 已确认完整任务 Goal，后续不再调用 Base；
- `M_base_reentry`：未完成 Goal 的状态类，同时通过符号 re-entry signature 和独立 Base
  continuation 能力合同。

恢复目标按下列优先级选择：

1. 直接满足缺失 Goal 且保持所有已完成 Goal；
2. 到达 demonstration-derived re-entry signature：handempty/access 状态一致、已完成
   Goal 保持、剩余对象位于 nominal policy 已覆盖的位置类别，并通过下述 held-out
   continuation 合同；
3. 没有经过验证的 re-entry target 时拒绝接管。

非终止 re-entry 不能按单个精确状态估计，必须按冻结的 state class `C` 和 remaining goal
class `g` 登记：

```text
certified_base_reentry(C, g) :=
    symbolic_reentry_signature(C)
    ∧ LCB95[P(π_base completes g | s ∼ heldout(C))] ≥ τ_resume
    ∧ resume_steps_q90(C, g) ≤ remaining_steps
    ∧ continuation_checkpoint_hash = nominal_checkpoint_hash
```

continuation rollout 使用互斥 `recovery_group_id`、冻结 Base prompt/checkpoint 和显式 branch
policy seeds；报告 success/trials、置信区间、resume-step 分布和 terminal cause。不能把同一
root 的相邻帧或多个 branch seeds 当作独立 roots。

非终止扩展的 development 门槛冻结为 continuation success point estimate ≥ 80%、Wilson
95% lower bound ≥ 65%，即上式 `τ_resume=0.65`；论文同时报告点估计和区间，不把 65%
解释为性能目标。若未来改变阈值，必须在新的 development protocol 上预注册，不能看过
locked confirmation 结果后修改。

Task 5/8 Phase 1 主要使用第 1 类，即恢复宏直接完成掉落分支，因此主张限定为
`terminal local recovery`。第 2 类只建立接口、记录格式和独立诊断 benchmark；在其能力
合同通过前，不声明已经证明 Base re-entry 或 Repair-to-Reentry。内部名称可继续使用
LOGIV-R2M，但论文必须明确 Phase 1 只验证 `M_goal` 子集；若不做非终止 benchmark，公开
方法名使用更保守的 `LOGIV-Recovery`。

### 8.3 原子恢复提交合同

`π_recover` 停止后必须采集 fresh、稳定 snapshot。只有以下联合谓词为 TRUE 才能提交
本次恢复：

```text
recovery_commit :=
    effect_satisfied
    ∧ completed_goal_invariants_preserved
    ∧ remaining_budget_sufficient
    ∧ (native_goal_success ∨ certified_base_reentry)
```

- `effect_satisfied`：option 声明的正/负 effects 均由 fresh facts 可靠确认；UNKNOWN 不
  满足该项。
- `completed_goal_invariants_preserved`：handoff 时已满足的所有 Goal facts 在执行中未被
  确认破坏，停止后仍可靠为 TRUE。prompt 的 `Preserve` 句只提供模型条件，不是证据。
- `native_goal_success`：完整冻结 Goal 由 fresh facts 满足，且 external native evaluator
  确认成功；此分支直接 terminal success，不再要求或调用 Base。
- `certified_base_reentry`：snapshot 匹配第 8.2 节已启用 continuation contract，且
  handempty/holding、object location、access 和 mutually-exclusive facts 无冲突；只满足
  option effect 但仍拿错物体、处于半完成 articulation 状态或只有符号签名而没有 held-out
  Base continuation 证据时，不得切回 Base。
- `remaining_budget_sufficient`：如果 `native_goal_success`，后续所需 model-action steps
  为 0；否则必须保留对应 continuation contract 的 `resume_steps_q90`。该 q90 由独立
  held-out Base continuation rollouts 冻结，不能只由 nominal demonstration 或当前 episode
  动态猜测。permit 阶段的 `post_recovery_reserve` 使用同一定义。

effect confirmation 和 post-stop observation 使用所有方法臂相同、预先冻结的只读/保持
状态协议，不得借机完成新的物理目标，也不得作为 LOGIV 独享的额外 model-action budget。
520-step 主预算统一统计 `π_base` 与 `π_recover` 实际产生并执行的 model actions。

联合谓词任一项为 FALSE 或 UNKNOWN 时，本次 option 不提交，`π_recover` 必须停止；后续
除下述完整 `safe_abort_to_base` receipt 外，只能由 LOGIV 在 fresh Current Problem 上产生
新的 certificate 和 permit，不能让同一次 policy 调用顺势执行下一个宏。该“新 permit”
只适用于未来预注册的 multi-attempt 扩展；Phase 1 已消耗唯一 physical attempt 后只能满足
`safe_abort_to_base` 或进入对应 terminal cause。

“恢复成功提交”和“失败后的安全退出”是不同 receipts：

```text
safe_abort_to_base :=
    ¬effect_satisfied
    ∧ completed_goal_invariants_preserved
    ∧ certified_base_reentry
    ∧ remaining_budget_sufficient
```

`safe_abort_to_base` 允许 Phase 1 在唯一 recovery attempt 失败但状态确实回到已认证 Base
入口时停止损失并继续 Base；它记录为 recovery failure/safe abort，不能计作 effect success、
positive flip 或 `recovery_commit`。没有 certified re-entry、出现 UNKNOWN 或 invariant damage
时不得安全退出。

handoff 本身使用显式事务状态机，不能把“检查 permit、flush Base、切 generation、扣预算、
发 recovery request”分散成可重入 callbacks：

```text
BASE_ACTIVE
  -> PREPARED                # read-only permit/option/receipt construction
  -> RECOVERY_COMMITTED      # sole CAS mutation point
  -> DISPATCHING
  -> VERIFYING
  -> NATIVE_SUCCESS | BASE_RESUMED | TERMINAL_*
```

`PREPARED` 只在一个已返回的 `env.step` 与下一 Base request 之间创建；单线程 rollout 持有
`OverlayController` mutex，不允许同时发 Base request。它冻结 CAS expected tuple：controller
state/generation、policy step、relevant-fact hash、epoch、observation generation、pending-chunk
hash/offset、remaining action budget，以及 handoff/attempt/no-repeat ledger heads。准备或 durable
write-ahead receipt 失败不改变 deque/generation/ledger；已写 PREPARE 只追加 aborted marker，不
删除审计记录，然后按原 pending
Base suffix 继续。

唯一 commit 点是 `compare_and_set(expected, HandoffCommitReceipt)`。一次临界区内同时：(1)
consume handoff/physical-attempt/no-repeat token；(2) 把选中的唯一 option 和 action cap 固化；
(3) 将 pending Base suffix 连同 hash/count 标记 `DISCARDED_BY_HANDOFF` 并清空；(4) generation
加一使迟到 Base response 无效；(5) 状态置为 `RECOVERY_COMMITTED`。任一 expected 字段漂移则
CAS 零副作用失败、receipt 标记 aborted 并继续 Base。action budget 仍只在每次成功进入
`env.step` 前逐动作扣减，不能在 commit 时把整个 cap 当成已执行。

为使上述 CAS 真正可执行，pending chunk、ledger heads、generation 与 controller status 都是
一个不可变 `OverlayControllerState` 的字段。先在锁外完成所有可能抛错的 validation、序列化
和 next-state 构造；锁内只比较 expected tuple 并做一次 `self._state = next_state` 指针替换，
不在临界区调用 deque clear、文件 I/O、模型 client 或用户 callback。旧 suffix 保留在 immutable
receipt 作审计，但 rollout 只从新 state 读取，因此不存在 generation 已变而 deque 未清的
半提交状态。

commit 后禁止回滚旧 Base chunk 或退还 attempt token。Recovery request 使用由 receipt hash
导出的 idempotency key；重复 callback 只能读取同一 receipt/response，不能重复 enqueue 或
再次扣账。commit 后的 server、enqueue、artifact 或 callback 异常进入第 13 节 fresh verify /
safe-abort/terminal 路径，不能假装从未接管。对 PREPARE journal、CAS 前后、deque clear、
generation bump、request send/response、每个 action enqueue 和 VERIFYING 边界分别做 fault
injection；断言要么完全保持 Base 前缀，要么只有一份 committed receipt 和至多一个物理宏。

### 8.4 计划排序

VAL-valid 候选按以下确定性顺序排序：

1. 淘汰 capability 或 budget gate 不通过者；
2. 最大化各 option 校准成功率下界的乘积；
3. 最小化 protected-invariant violation 上界；
4. 最小化 step q90、plan length、编辑距离；
5. 使用旧 canonical rank 和 grounded lexical key 打破平局。

## 9. Recovery prompt 合同

`π_recover` 不接受自由改写的临时 prompt，只接受版本化有限模板。模板保留原始任务
语言，同时显式给出当前义务、当前可信状态、必须保护的 facts 和停止条件。例如：

```text
Recovery mode.
Original task: put both moka pots on the stove
Current obligation: pick up moka_pot_2 from the kitchen table and place it on the stove.
Preserve: moka_pot_1 must remain on the stove.
Stop when: moka_pot_2 is released and stable on the stove.
```

训练和推理必须使用完全相同的 template version。内部 PDDL region ID 不直接暴露给
VLA；renderer 使用 coverage manifest 中的自然语言对象/位置名称。prompt record 保存
模板哈希、grounded option、protected facts 和原始任务文本。

`Preserve:` 与 `Stop when:` 都是 policy conditioning，不是 authorization。执行期间
`ExternalInvariantSupervisor` 至少在每个既有 replan boundary 读取 fresh facts；一旦某个
protected invariant 被可靠判定为 FALSE，立即撤销剩余 option actions，并进入第 13 节的
损害恢复路径。停止后仍需完整联合提交合同，不能因 prompt 写过保护要求而默认保护成功。

## 10. Failure-state 数据闭环

### 10.1 Root snapshot artifact

evaluator 在稳定偏差首次成立、Base terminal 以及 recovery effect failure 时保存：

- MuJoCo flattened simulator state；
- task/episode/master seed、policy seed、simulator seed、collection label、
  parent-trajectory-lineage hash、event-origin parent hash 和 handoff step；
- BDDL、initial-state hash、first-frame hash、Base prompt/checkpoint/policy-client-config hashes、
  当前 observation/proprioception；
- Base action-prefix hash、未执行 Base action chunk 的完整 dtype/shape/bytes/hash、当前 chunk
  原始 response size、实际入队 logical size/offset、Base request count、active/next zero-based
  request indices、对应 canonical envelope JSON/hash、held-out replay-contract hash、policy
  request generation 与 episode-seeded policy envelope；
- fact epoch、versioned monitored fact universe/hash、可重算的 canonical grounding-evidence payload、fresh
  true/false/unknown partition、Goal、protected facts、candidate repair 和 certificate；
- trigger class、完整 canonical `ActionEventEvidence` records、detector hash、内嵌 canonical
  monitor-contract JSON/self-hash 和 native
  evaluator status；裸 evidence-kind 字符串不是有效 provenance。

verified loader 必须仅凭 final root directory 重算 contract self-hash，以 evidence 的 `rule_id`
解析 action rule 或 reserved Goal/progress rule，并验证 due/expiry/attribution 公式、detector hash、
`evidence_id` 与 `(attempt_id, kind)` 唯一性。只保存一个无法解析的外部 contract hash 不足以
形成独立可审计 artifact。

这里的 UNKNOWN 不是“未出现在 true/false 列表就猜成 UNKNOWN”。每个可用于 root 或 live
certificate 的 `FactSnapshot` 必须携带完整、版本化的 registered fact universe、其
domain-separated hash，以及 grounder 当时用于生成判定的 exact canonical evidence JSON。
`true_facts`、`false_facts`、`unknown_facts` 必须无交集且并集恰好等于该 universe；unknown
只能由完整 universe 减去 true/false 得到。loader 必须重算 universe hash 和 evidence hash，
并核对 payload 内的 observation hash、epoch 与全部 `(fact, TruthValue)` 判定。未注册 predicate
保持 absent，不能被悄悄纳入 UNKNOWN；缺 universe、部分 audit 字段或 payload/partition 不一致
均拒绝作为 recovery/certificate 证据。旧的四字段 snapshot 只保留普通单测兼容性。

LIBERO `OffScreenRenderEnv` 已提供 `get_sim_state()` 与 `set_state()`。恢复 root 时使用
fresh env、设置 simulator state、`sim.forward()` 并强制更新 observables。physics state
round-trip 与 policy RNG continuation 分开验证；branch rollout 使用新的显式 branch seed，
不声称复现原 Base 的隐藏 policy state。

root 发布必须以 sibling 临时目录写完并 fsync state/JSON 后，通过同文件系统单次目录 rename
发布；loader 只读同时含两文件的 final directory。崩溃遗留 temp directory 只由 validator
报告/显式清理，不当作 root；重试遇到内容相同 final root 为幂等成功，内容不同则 fail
closed。不能用两次独立 file rename 却把中间 state-only 目录称为 atomic artifact。

### 10.2 状态来源

仅从 nominal clean demonstration 或干净 scripted trajectory 训练不能建立恢复能力合同。
数据必须由下列两部分共同组成：

- 真实失败：当前 Task 5/8 Base、旧 Full 与 overlay 暴露的 drop、wrong-held、crowded-target
  状态。
- 系统扰动：从 nominal demonstration state 创建同类可控偏差，包括 object drop、
  wrong-object grasp、partial placement/未完全释放、一个 Goal 已满足、crowded target 被
  已完成物体占用，以及适用任务中的 drawer/microwave articulation half-open。

每个保存状态有三个不同用途的稳定 ID：`root_id` 精确绑定 task、initial-state、
perturbation/branch seeds、capture policy step、deviation status/event ID 与 root physics-state
hash，用于复现并区分某一次物理状态采集；
`recovery_group_id` 则绑定 task、scene XML hash、object instance/type IDs、initial-state
hash、`event_origin_parent_sha256` 和 perturbation family，但有意排除
perturbation/branch seeds、
相机噪声与微小 object pose。后者是数据划分的最小原子单位。所有满足以下任一关系的
samples 必须进入同一 split：

- 来自同一 initial state 或同一 synthetic source-parent snapshot；
- 只改变 perturbation seed、branch seed、相机噪声或微小 object pose；
- 使用相同 root snapshot 但由不同 teacher/policy 产生；
- 是同一 partial-placement、drop、wrong-grasp 或 half-open 事件的时间邻近帧。

这里的 parent 有两个不能混用的层级。`parent_trajectory_lineage_sha256` 在 Base/demonstration
parent trajectory 启动前，由 task、scene、initial state、冻结 Base checkpoint/prompt 配置和
master-seed envelope 生成，整条轨迹不变；`event_origin_parent_sha256` 在该
`deviation_event_id` 第一次 anomaly observation 时锁存。其 domain-separated payload 只含
task/trigger/object/effect/registered semantic region、protected-goal signature 和排序后的相关
signed facts；明确排除 epoch/generation、grounding evidence/observation/pixel hash、相机噪声和
bin 内微小 pose，不是完整 `FactSnapshot` 或稍后写盘帧的 physics hash。后续
candidate→confirmed upgrade
即使在更晚 physics frame 写盘，也必须沿用同一个 event-origin hash。精确 capture-frame
physics hash 进入 `root_id`。`source_parent_snapshot_sha256` 只表示生成 synthetic
perturbation 前的 clean simulator snapshot；自然 observed failure 必须为 null，不能伪造 clean
parent。该字段只作复现/泄漏审计，不能重定义 group 或独立单位。

统计独立性使用更粗的 `independence_unit_id = hash(task_id, scene_hash,
initial_state_hash, parent_trajectory_lineage_sha256)`；它有意忽略 capture/source-parent snapshot、
perturbation family/seed、candidate→confirmed 状态和 branch seed。同一 Base/demonstration
parent trajectory 衍生的多个 snapshots/families 只能贡献一个独立单位。第 12.2 节的精确
零事件声明每个 independence unit 预注册一个 primary intervention outcome；额外 roots 只作
敏感性/覆盖诊断，不增加二项分母。

train/dev/held-out 使用互斥的 `recovery_group_id`、initial-state IDs、root IDs 和
perturbation seeds；split validator 在训练前检查交集并计算 pose/state fingerprint
近重复。若 LIBERO 固定任务没有
新的 object instance 或 scene 可用于真正的 object/scene holdout，能力合同必须明确写成
“仅在已见 object/scene class 内验证”，不得宣称跨物体或跨场景泛化。当前已查看的 50
episodes 只用于 development；冻结代码后的确认运行使用新的 policy master seed。

派生数据角色在任何用于选择/校准的 branch outcome 产生前写入不可变 dataset manifest：
`TRAIN`、`DEV_SELECTION`、`PERMIT_CALIBRATION`、`PAPER_CONFIRMATION`。同一
`independence_unit_id` 及其所有 `recovery_group_id` 只能属于一个角色。Phase 0 root
artifact 的 immutable collection label 统一为 `DEV_COLLECTION`；它们只能在后续按
initial-state lineage 整体分配到 TRAIN/DEV_SELECTION dataset manifest，不得进入
calibration 或 paper confirmation。

角色完整性不能依赖调用者声称“已提供全部旧 manifests”。每个 immutable source-dataset
manifest 的全部 `independence_unit_id` 必须在 outcome 前由一个完整
`RoleAllocationManifest` 恰好分配一次；allocation hash 写入 run config。项目使用锁定路径和
registry ID 的 append-only `RoleRegistryHead`，通过 expected-parent-head CAS 原子追加并拒绝
任何历史 unit 再分配。每个 registry entry 同时保存 source-dataset hash 作 provenance，但
还保存由 unit/group/initial-state/source-parent/parent-lineage/state-fingerprint/task-seed
关系重算的 domain-prefixed `role_isolation_keys`；registry 对 unit ID 和全部 isolation keys
全局判重，不能靠重新封装 source manifest 或换一个 unit ID 绕过。training/calibration/confirmation builder 必须同时验证 source hash、
完整 allocation 和 canonical registry head；缺 unit、多 unit、旧 head 或替代 registry 均
fail closed。

角色隔离还必须复用 raw split 的全部相关性键：同一 recovery group、initial state、非空
synthetic source parent、parent trajectory lineage、近重复 state fingerprint 或 task 内
perturbation seed 均不得跨 TRAIN/DEV_SELECTION/PERMIT_CALIBRATION/PAPER_CONFIRMATION，不能只
检查 unit ID。`IMPORTED_FROZEN` 的 TRAIN/DEV/HELDOUT 分别只映射 TRAIN、DEV_SELECTION、
后两种确认角色；Phase 0 的 `DEV_COLLECTION` 例外只允许转入 TRAIN/DEV_SELECTION，永远不能
进入 permit/paper。PERMIT_CALIBRATION 与 PAPER_CONFIRMATION 即使同为 HELDOUT 也互相隔离。

raw root directory 不是训练输入。版本化 training-manifest builder 必须先解析结构化
`ActionEventEvidence`、执行 split/近重复检查，只接纳 confirmed 且证书为
`CURRENT/RECONCILED` 的 roots。`STALE` root 只有在 fresh Current Problem 上重新标注，并有
匹配 root-state hash、fresh observation/fact epoch、fact-universe、可重算 fresh evidence、
event ID、相关 facts、历史证据、grounding/detector/monitor-contract hashes 和新
graph/certificate hash 的签名
`FreshRecoveryLabel` 时才能进入 manifest；candidate 永远不能进入。trainer 只能读取包含
root/group/independence-unit IDs、数据角色、builder version 和内容 hash 的不可变 manifest，
不得绕过 builder 直接扫描 root 文件夹。

### 10.3 Teacher 与自动筛选

按成本从低到高生成 bridge trajectory：

1. 从 root state 对现有 pi0.5 和固定 recovery template 做有限 branch-and-filter。
2. 从扰动 root state 使用 privileged operational-space scripted teacher 执行“上方接近—
   下降—抓取—抬升—移动—释放”恢复宏，并记录 LIBERO 7D actions；teacher 动作可以干净，
   但起始状态必须覆盖真实或合成失败分布。
3. scripted teacher 无法稳定处理的 articulation 类才进入单独 teleoperation 数据收集；
   不阻塞 Task 5/8 第一阶段。

一个 trajectory 只有在 fresh facts 连续确认目标 effect、所有 protected invariants 保持、
native simulator 无异常且 step cost 合法时才成为正训练样本。失败 rollout 和 collateral
damage 保存为能力校准/负例，不混入 action imitation 正样本。

每个用于 intervention-advantage 校准的 held-out deviation root 还必须在相同 branch master
seeds 下运行两条不用于训练的闭环分支：继续冻结 `π_base`，以及执行完整 recovery pipeline
后到 native terminal。两臂均记录最终任务 success、negative/collateral outcome 和总步数；
不能只在已知 Base 最终失败的 roots 上比较，否则会按结果选择并夸大 recovery 优势。

continue-Base 分支首先按记录的 pending chunk/offset 执行尚未执行的原 Base actions，后续
请求通过可复现的 episode-seeded wrapper 从记录的 next request index 继续；recovery 分支按真实
Overlay 行为 flush 同一 pending chunk、执行恢复，并在需要时以新的明确 generation 回到
Base。如果策略服务无法恢复/验证该 RNG/request envelope，该 root 不进入“live Base
continuation” estimand，只能进入一个预注册的“两臂都 flush 并重新 seed”诊断 estimand，
二者不得混合报告。

### 10.4 LeRobot 数据格式

bridge trajectory 转换为 OpenPI 已支持的 LIBERO LeRobot schema：双视角图像、机器人
state、7D actions、固定 recovery prompt、episode/root/branch IDs 和 split。动作仍使用
LIBERO delta convention、`action_horizon=10`、`extra_delta_transform=False`。优先复用
`pi05_libero` 的 state/action normalization assets，并把新数据统计作为漂移审计，而不是
静默替换推理 normalization。

## 11. `π_recover` 训练与模型选择

`π_recover` 从当前 `pi05_libero` checkpoint 初始化，作为单独 checkpoint/服务存在；
nominal `π_base` 永不加载 recovery 权重。JAX OpenPI 原生具备 freeze filter、action
expert LoRA 和自定义 LeRobot data config，PyTorch 路线因当前不支持 LoRA，不作为第一
阶段主路径。

固定比较两个候选：

1. **REC-A（推荐首选）**：冻结视觉语言主干，完整微调 action expert 与必要 action
   interface；容量高，且独立服务消除了 nominal catastrophic regression 风险。
2. **REC-B（低内存对照）**：视觉语言主干与 action expert 使用 LoRA，只训练 LoRA
   parameters；成本较低，但不能预先假设性能等同 REC-A。

训练 loss 只用于优化诊断，checkpoint 选择只依据固定 `DEV_SELECTION` root states 上的
闭环 effect success、protected-invariant violation 和 steps。选定 checkpoint、template、
state classifier 与 trigger thresholds 后全部冻结，再进入互斥 `PERMIT_CALIBRATION`；禁止
根据 permit calibration、500-episode 最终主结果回选任何组件。

如果两者都未通过能力门槛，先增加失败类别对应的 teacher bridges；只有在数据覆盖充足
而 action-expert 路线仍失败时，才单独设计 full-model recovery fine-tuning，不直接进入
全矩阵。

## 12. 能力合同与启用门槛

每个 `RecoverySkillContract` 至少包含：

- `skill_id`、schema、state class、template/checkpoint hashes；
- signed symbolic `preconditions` 与 re-entry signature；
- `supported_object_types`、object instances、locations、access/articulation classes 与
  unsupported cases；
- `expected_duration` 的 median/q90/maximum；
- train/dev/test root IDs 和数据版本；
- `heldout_success_rate`、successes/trials 与 Wilson 95% interval；
- `interference_risk`、protected-invariant violations/trials 与置信区间；
- 非终止合同还必须包含 `base_continuation_success`、`τ_resume`、continuation trials、
  one-sided lower bound 与 resume-step median/q90；
- `base_continue_success_from_deviation`、`recovery_pipeline_final_success`、配对差值及其
  state-class stratified bootstrap interval；
- status：`EXPERIMENTAL`、`ENABLED` 或 `REJECTED`。

`state class` 不能是人工口头标签。版本化 `RecoveryStateClassRegistry` 的 classifier hash、
特征和支持范围至少包括：task ID、failed schema/effect、object type/instance support、
nominal/abnormal source region、destination region、holding/handempty、access/articulation bin、
protected-goal signature、target occupancy/crowding，以及能力合同需要的连续 pose/distance
区间。任一必需 feature 为 UNKNOWN、超出登记区间或 classifier hash 漂移均判 OOD 并
fail closed；不能把低支持子类合并进粗 class 来稀释负翻转。

development `interference_gate_passed` 的数值定义为：每个 state-class stratum 至少 30 个
distinct `independence_unit_id`（不是 recovery groups），protected-invariant violation 为 0，
且 95% 单侧精确上界 ≤10%。
这只是允许进入 paired development 的风险上限；论文低于 2% 的声明仍必须满足第 12.2
节更大独立 confirmation set，不能把 10% 工程门槛表述成低风险保证。

### 12.1 开发启用门槛

一个 state class 至少有 30 个未用于训练、checkpoint/threshold/classifier 选择的独立
`PERMIT_CALIBRATION` units，且同时满足以下条件才可在 paired development 中 `ENABLED`：

- effect success point estimate ≥ 80%；
- Wilson 95% lower bound ≥ 65%；
- validation 中 protected-invariant violation 为 0；
- policy-action step q90 ≤ 180；
- capability validation 使用与 runtime 完全相同的 `hard_action_cap`；被 cap 截断且 effect 未
  成立的 rollout 计为失败，不能先用无界 rollout 估计高成功率、上线后再截断；
- 在相同 saved roots/branch seeds 上，recovery pipeline 相对继续 Base 的 paired success-rate
  difference 下界 > 0；样本不足或区间跨 0 时合同保持 `EXPERIMENTAL`；
- root-state round-trip、prompt rendering 和 checkpoint provenance 全部通过。

所有 eligible confirmed deviations 必须在看到 continue-Base 或 recovery terminal outcome
之前前瞻性写入 calibration manifest；Base 最终成功、失败和自恢复 episodes 都保留。禁止
先跑完 Base 后只选择失败 roots，也禁止因 recovery branch 失败而删除单位。

这 30 个 calibration units 和零次 observed violation 仅是工程启用门槛。按 95% 单侧精确
Clopper-Pearson 区间，`0/30` 的 violation-rate 上界仍约为 9.50%，因此不能据此表述
“破坏率低于 2%”或真实世界安全概率保证。运行时只有 `ENABLED` 合同可以
产生 runtime handoff permit；runtime snapshot 必须与登记 preconditions、object/location/articulation
支持范围逐项匹配。合同不匹配、统计缺失或 hash 漂移均 fail closed 到继续 Base。

### 12.2 论文确认门槛

若论文要声明注册分布内 protected-invariant violation rate 的 95% 单侧上界低于 2%，
必须使用 checkpoint、template、threshold 全部冻结后的独立 confirmation set，并预注册
精确 Clopper-Pearson 口径。零次 violation 时：

```text
upper95_one_sided(n, 0) = 1 - 0.05^(1/n)
```

最少需要 `n=149` 个独立 `independence_unit_id` 才使该上界严格低于 2%；统一采用 150 作为
论文最小目标。若采用双侧 95% 精确区间，则相同零违规目标至少需要 183 个独立单位。
同一 root 的相邻帧、微小 pose 变体和多 branch trials 不增加独立样本数。

split/manifest validator 必须机械报告 unique independence-unit 数；共享 LIBERO initial
state/parent trajectory 的 roots 不能通过更换 perturbation family 或 source-parent snapshot 增加
分母。仍同时按 initial-state/parent cluster 报告 cluster bootstrap 敏感性分析；若有效独立
unit 不足，论文只能报告 observed trial-distribution 上界，不能把它外推成跨场景或跨物体
的 2% 上界。

该结论只能按实际验证层级声明：每个 contract 各有 150 个 distinct
`independence_unit_id` 时可以逐 contract
声明；只在多个技能上合并 150 个 independence units 时只能声明 pooled intervention rate。多个
contract 的同时声明需预注册 multiplicity 处理，或者逐项报告未调整区间而不作 family-wise
保证。所有表述限定于 LIBERO、oracle/registered grounder 和登记 state classes，不使用
“安全保证”措辞。

## 13. 错误处理与有界终止

520-action 总预算只约束物理动作，不足以阻止零动作 certificate rebuild、VAL 循环或跨
graph-version 重复接管。所有运行臂必须同时冻结并统一记账以下上界：

```text
Phase 1 Task 5/8:
    max_handoffs_per_episode = 1
    max_attempts_per_deviation = 1
    max_attempts_per_skill = 1
    max_skills_considered_per_deviation = 2
    max_option_candidates_per_deviation = 2
    max_certificate_builds_per_deviation = 2
    max_repair_rounds_per_certificate = 2
    max_edits_per_repair = 3
    max_candidates_per_repair = 256
    max_val_calls_per_repair = 8
    max_total_val_calls_per_episode = 24
    val_timeout_seconds = 5
    max_planner_wall_seconds_per_deviation = 15
```

这些值保存在版本化 `Phase1RuntimeLimits` schema 并进入 run/certificate hash。启动时任何
缺失、非正数、运行配置高于上述 hard maximum 或不同组件值不一致都拒绝启用 recovery。
每个 skill enumeration、option expansion、certificate build、Repair 和 VAL call 必须在调用
前原子 consume；不能在返回后才记账。每次 VAL/solver timeout 使用
`min(5 seconds, remaining planner wall time)`，wall deadline 到达后不再启动新调用。

Phase 1 使用 terminal local recovery，失败后没有证据支持第二次物理接管，因此上限为 1；
未来非终止 re-entry 扩展只有在独立 paired development 证明负翻转受控后才能提高上限。

retry ledger 的禁止重复键必须跨 certificate/graph version 稳定：

```text
no_repeat_key := (
    episode_id,
    deviation_event_id,
    semantic_state_class,
    skill_id,
    protected_goal_signature
)
```

不能把 `certificate_hash` 作为唯一去重依据，因为 rebuild 会生成新 hash 并绕过限制。
certificate/permit 仍记录 hash 作为 provenance，但重复判定使用上述语义事件键。

- monitor grounding UNKNOWN：记录并继续 Base，不猜测偏差。
- root snapshot 保存失败：禁止 recovery handoff，继续 Base。
- VAL error、certificate/hash mismatch：拒绝接管。
- recovery server timeout/exception：停止当前同步 dispatch；只有 fresh snapshot 仍通过
  第 8.3 节完整联合提交合同时才可切回 Base 或进入 native success 检查。Phase 1 已消耗
  唯一 physical attempt 时其余情况 terminal；未来多尝试协议才可在新 certificate/permit
  和剩余显式上界内修复。
- effect 未成立但 invariants 保持：Phase 1 消耗唯一 physical attempt；只有第 8.3 节
  `safe_abort_to_base` 完整成立时才能回 Base，并记录 recovery failure，否则 terminal。未来
  允许多候选时也必须同时满足显式 handoff/attempt/no-repeat 上界。
- 任一 protected invariant 由 TRUE 变 FALSE：立即停止 recovery；仅允许 LOGIV 生成以
  该 invariant 为首要 obligation 的已启用 damage-repair skill。Phase 1 未登记
  damage-repair skill，因此直接 `TERMINAL_COLLATERAL_DAMAGE`，不能回 Base、换 certificate
  或继续普通恢复掩盖损害。
- 520 actions 耗尽：停止所有 policy dispatch；settling/只读评价不计 policy action，
  但不得执行改变物理状态的额外动作。
- 当前 option 达到 permit-bound `option_action_cap`：立即停止并验证 effect；失败按唯一
  attempt/continuation/terminal 规则处理，不能继续消耗 post-recovery reserve。

所有预算跨 Base/Recovery/graph version 统一记账，不因切换服务或重新安装 repair graph
重置。

协议的终止证明使用以下字典序单调量：

```text
(
    remaining_handoffs,
    remaining_physical_attempts,
    remaining_skills_considered,
    remaining_options_considered,
    remaining_certificate_builds,
    remaining_repair_rounds,
    remaining_repair_candidates,
    remaining_val_calls,
    remaining_planner_wall_milliseconds,
    remaining_action_budget
)
```

每次 dispatch、rebuild、Repair 或 VAL 至少严格消耗对应分量；planner timeout 直接消耗本次
build 并 fail closed。不存在不消耗该向量任何分量的证书再生成路径。

## 14. 实验设计与消融

### 14.1 方法臂

1. `BASE`：原始 `π_base`、官方任务文本和官方 evaluator 行为。
2. `SHADOW_LOGIV`：运行本地 monitor/grounding，但永不接管；用于证明动作、policy-action
   steps、Base policy requests 和 native outcome 等价，同时单独测量 shadow compute。
3. `BASE_RECOVERY_NO_LOGIV`：相同 `π_recover`，使用非 VAL 的简单 recovery-surface
   heuristic；隔离“训练恢复策略 + 简单启发式”的总体贡献，但不用于归因 DAG。
4. `LOGIV_FROZEN_RECOVERY`：LOGIV overlay 调用原 frozen π₀.₅；保留当前负/零改善机制
   证据。
5. `R2M_NO_TOPOLOGY`：与 Full 使用完全相同的 confirmed trigger、Current Problem、
   protected Goal facts、Repair、完整 VAL、`π_recover`、Capability Registry、Supervisor、
   budget 和 seeds；输入相同 trigger-bound failed obligation，但把 causal DAG/local causal
   slice 换成该 obligation 的 flat predicates 和预注册 canonical option ordering，不使用
   prerequisite/descendant topology。
6. `R2M_GLOBAL_REPLAN`：与 Full 完全相同，唯一变化是从 fresh Current Problem 对全部
   remaining goals 全局重规划，而不是只修复 failed causal slice。
7. `R2M_CHAIN`（补充）：将同一 causal plan 强制全序化，用于判断收益是否来自偏序而非
   仅有 plan validation；只有 benchmark 中存在宽度大于 1 的 DAG 时才运行。
8. `LOGIV_R2M`：完整能力门控 LOGIV + 训练后的 `π_recover`，唯一主候选。

`R2M_NO_TOPOLOGY` 与 `R2M_GLOBAL_REPLAN` 必须复用 Full 已冻结的 trigger/event detector，
因此它们只隔离 recovery planning topology，不把检测差异混入比较。no-topology 仍使用同一
protected Goal 集合和完整 VAL；否则它同时移除了 invariant protection 或 symbolic validity，
不能解释为 DAG 消融。

三臂共享的 failed obligation 必须由 topology-free signed-fact/Goal comparator 从 fresh Current
Problem 和结构化 trigger evidence 产生，只包含 failed literal/object/effect，不包含 occurrence
ID、causal predecessor、slice 或 canonical DAG rank。Full 收到同一 flat obligation 后才允许
用 DAG 映射并扩展 causal slice；否则 no-topology 已经泄漏 Full 的拓扑选择。

三臂都可以构建包含多个符号步骤/候选宏的 planner artifact，但统一经过同一个
`DispatchOptionSelection` 边界：按第 8.4 节排序后只选择一个已验证 physical option，并要求
该单一 option 的停止状态属于 `M_goal ∪ M_base_reentry`，再进入同一 handoff CAS。global
replan 的“全局”只描述其搜索/依赖范围，不授权顺序执行全局计划。若其全局解必须依赖两个
或更多 physical options 才能闭合提交合同，该 root 对 runtime handoff 为 INELIGIBLE，只保留
planner/node/VAL 诊断；不得给 global arm 额外宏、handoff 或 action budget。

### 14.2 公平性

- 所有主比较使用相同 checkpoint lineage、initial state、fresh-env-per-episode、first
  frame、policy/simulator seed、`replan_steps=5` 和 520 policy actions。
- Base prefix 到 handoff step 的 action hash 必须与配对 Base 相同。
- planner/VAL 的 wall time 不折算为 policy actions，但单独报告总 wall time、
  `base_policy_requests`、`initial_proposal_requests`、`shadow_vlm_requests`、
  `recovery_policy_requests`、monitor calls 和 planner calls。
- 每个方法独立启动或使用显式 episode-seeded policy wrapper，不能让方法顺序共享隐藏
  RNG 状态。

### 14.3 主要变量与 estimand

- 自变量：方法臂。
- 主要因变量：native LIBERO episode success。
- 关键次级指标：Base-only/LOGIV-only flips、anomaly/confirmed/permit rate、confirmed
  deviations on paired Base-success episodes、Base self-recovery after confirmed deviation、bridge success、
  collateral violation、handoff step、repair steps、repair node count、physical option count、
  planner/VAL calls、total steps 和 terminal cause。
- 主要 estimand：十个任务等权的 paired success-rate difference；每个任务 50 episodes，
  因样本数相同，其点估计与 500-episode 总成功率差一致。

### 14.4 统计分析

- 每任务报告 `x/50`、Wilson 95% interval、正翻转与负翻转。
- 总体在 task 内对共享 pairs 重采样，做 10,000 次 paired stratified bootstrap。
- 报告总体 paired difference 的 95% interval；McNemar exact test 作为 discordant-pair
  敏感性分析，不替代效应量和区间。
- checkpoint/trigger 开发只看 development 结果；locked 500 只运行冻结配置一次。

### 14.5 拓扑机制 benchmark

如果 recovery root 只有一个显而易见的失败叶节点，Full、no-topology 与 global-replan
可能生成相同宏，此时实验不能识别 DAG 贡献。拓扑 benchmark 必须在互斥 held-out
`recovery_group_id` 上预注册至少以下状态：一个 Goal 已完成而 sibling branch 失败、两个
独立 remaining branches、crowded target、以及候选恢复可能威胁 protected sibling 的状态。

所有方法在同一 root、branch policy seed、trigger record 和 action budget 上配对运行，报告：

- native recovery/continuation success 与 terminal cause；
- protected-invariant violation；
- repair 涉及节点数、重新打开的已完成节点数和 physical option 数；
- planner/VAL calls、wall time、recovery steps；
- 在真实 paired Task 5/8 episodes 上的净 flips。

每个 root 在看到 outcome 前还要计算 topology-activity gate：DAG width/branching、local slice
与 global remaining set 是否不同，以及各臂 pre-dispatch plan/option 是否发生差异。所有臂
生成相同 option 的 root 标记 `NON_IDENTIFYING`，可计入总体性能但不能计入“DAG 导致机制
改善”的分母。冻结 benchmark 最少包含 40 个 distinct `independence_unit_id`，上述四类状态
各至少 10 个；其中至少 30 个必须在 outcome 前由 activity gate 判为 `IDENTIFYING`。不足时
自动删除 DAG 经验贡献声明，不能事后降低分母。始终报告 total/identifying/non-identifying
denominator 及每类数量。

DAG 机制的预注册主要判据是：在 identifying pairs 上，Full 相对 no-topology 和
global-replan 的 native final-success paired difference 两个 95% 区间下界都大于 0，同时
protected-invariant violation 不更高；repair nodes、reopened completed nodes、VAL calls、steps
和 wall time只作有方向的次级机制指标并完整报告。40/30 是最低可识别性门槛，不保证统计
功效；区间跨 0 时结论仍为“不足以证明 DAG 带来经验收益”，不能改用次级指标补救主张。

Phase 2 在 Task 5/8 paired development 和 topology benchmark 上运行上述严格消融；Phase 3
完整 10×50 主确认只要求 `BASE` 与冻结的 `LOGIV_R2M`，避免为机制消融消耗不必要的完整
矩阵算力。若 Full 未在预注册拓扑指标或净 flips 上优于 matched controls，论文不得声称
因果 DAG 带来经验收益，只能将其表述为结构化实现组件。

## 15. 分阶段实施与 Go/No-Go

### Phase 0：状态与数据基础设施

- 实现 root snapshot 保存/恢复、branch seed、trajectory recorder 和 dataset manifest。
- 实现 anomaly candidate、历史 action event、confirmed deviation 与 shadow certificate
  reconciliation；Phase 0 只保存数据，所有返回值仍无控制权限。
- 通过 simulator-state round-trip、fresh fact consistency、root-level split、stale certificate
  fail-closed 和无触发 Base 等价测试。
- 未通过时不实现训练或运行大规模 rollout。

### Phase 1：Task 5/8 recovery skill

- 收集/生成两类 macro bridge，训练 REC-A/REC-B。
- 每个 skill-state class 按第 12.1 节进行 development capability validation；Phase 1 的
  Task 5/8 只启用 terminal local recovery，`max_handoffs_per_episode=1`。
- 不达标的合同标记 `REJECTED`，不能通过放宽 trigger 或增加运行预算上线。

### Phase 2：Task 5/8 paired development

- 在已暴露 development protocol 上运行配对 Task 5/8，并在 topology benchmark 上运行
  `R2M_NO_TOPOLOGY`、`R2M_GLOBAL_REPLAN` 和适用时的 `R2M_CHAIN`。
- Go 条件：两任务合计净提升至少 `+10/100`，每个任务负翻转不超过 2，且所有干预均有
  匹配能力合同。
- 未通过时只允许修改一个已定位因素：数据覆盖、teacher、训练候选、trigger 或能力
  校准；禁止返回无穷 prompt sweep。

### Phase 3：冻结与全矩阵确认

- 冻结代码、dataset version、checkpoint、template、capability registry 和 thresholds。
- 在独立 post-selection confirmation groups 上按第 12.2 节确认 violation 上界；样本不足时
  报告实际上界，不宣称低于 2%。
- 使用新的 policy master seed 完成 Base 与 LOGIV-R2M 的 10×50 配对确认。
- 目标门槛：LOGIV-R2M ≥ 447/500、相对 Base 净提升 ≥ 10/500、paired bootstrap 95%
  interval 下界 > 0，且任一任务回退不超过 1/50。
- 如果只获得正点估计但区间跨 0，结果如实报告为趋势，不宣称显著提升。

### Phase 4：扩展

只有 Phase 3 通过后，才根据 Base 剩余失败频次向 Task 3/6/9 扩展 articulation、switch
或 relative-placement recovery。每个新 state class 重复独立 capability gate，不能因
已有 Task 5/8 checkpoint 而默认启用。

## 16. 测试要求

### 16.1 单元测试

- anomaly candidate 的三次 TRUE、TRUE→UNKNOWN、抓取/释放过渡态和 false-positive 去抖；
- 对象从开局位于 nominal surface、尚未被操作、Base 合法替代顺序和正常 settling 均不得
  形成 confirmed deviation；
- Goal regression、attempted-effect timeout 和 manipulation 后 abnormal transfer 分别产生
  historical evidence；progress timeout 单独出现时不得产生 handoff permit；
- versioned fact universe 的 TRUE/FALSE/UNKNOWN 恰好分区且 evidence payload 可重算；遗漏、
  conflict、universe/hash 漂移拒绝 root；
- concrete action tracker 在 due 前不产生 timeout、及时 effect TRUE 取消、due 后 FALSE
  单次发证、UNKNOWN 不发强证据、TTL 后过期；cross-object/region 和累计 overflow 不误确认；
- feature reader 每 rule/step 完整、null/nonfinite 断 continuity、contract/version 错配拒绝，
  且只读 adapter fake 从不调用 step/reset/set-state；
- 成功 holding 后同对象在 attribution TTL 内掉入异常区仍发证，异对象与过期窗口不发证；
- event-origin hash 对 epoch/camera noise/bin 内微 pose 不变，对 object/effect/semantic region/
  protected-goal 改变敏感；
- capability/budget 缺失只拒绝 permit，不删除 confirmed deviation/root record；
- 相关谓词覆盖的 transition 更新 DAG；合法替代 producer 令证书 STALE 并 reconcile；无关
  连续状态变化不令证书失效；STALE 或 dispatch 前 hash/generation 漂移必须 no-op；
- shadow 本地计算增加 latency 但不改变 Base action/outcome；模拟 external VLM 时额外请求
  被分桶且 Controller 不等待响应；
- InitialProposal 通过时只安装 shadow plan；拒绝、超时、异常或 stale response 时禁用
  intervention 并保持 Base action/outcome；
- Base/Recovery request generation 与 stale callback no-op；
- capability hash/state-class/remaining-budget 匹配；
- protected facts 自动进入 repair constraints；
- 联合提交中 effect/invariant/budget 以及 `goal ∨ certified-reentry` 分别 FALSE/UNKNOWN 时
  均不 commit，也不执行第二个 option；terminal Goal success 不要求 Base continuation，
  非终止符号 signature 没有 held-out continuation contract 时不得回 Base；
- candidate 排序、q90 budget 和没有能力合同时继续 Base；
- initial state、scene、object、perturbation/root seed 与近重复 state 的 split 防泄漏；
- source dataset 的完整 role allocation、canonical registry expected-head CAS 和跨 role unit
  再分配均机械检查；
- 30 个零违规的单侧精确上界约 9.50%，149 个约 1.99%，统计 helper 必须复现该边界；
- total 520-step budget 跨策略切换不重置；semantic no-repeat key 跨 certificate hash 生效，
  handoff/attempt/build/Repair/VAL 任一上界耗尽后均 terminal 或 fail closed。
- handoff PREPARE/CAS/flush/generation/request/enqueue/verify 每个 mutation 边界 fault injection；
  CAS 前失败保持原 Base suffix，CAS 后只有一个 receipt/宏且不能回滚或重复扣账。

### 16.2 集成测试

- inactive monitor 与 Base action arrays、steps、`π_base` policy requests 完全相等；
- InitialProposal 的 accepted/rejected 两条路径都不改变 Base prompt、action-prefix 或 RNG；
- 若启用在线 shadow VLM，只要求 Base policy requests 相等，并验证额外请求与延迟独立
  记录；
- simulator snapshot 恢复后 true/false facts 与保存时一致；
- Base 采用 Initial DAG 未覆盖但合法的 producer 时，旧证书失效且 fresh reconciliation 不
  改变 Base action stream；handoff 只能使用当前 snapshot 产生的新 recovery certificate；
- handoff 前 Base action-prefix hash 相等；
- 一个成功 recovery macro 只产生一个 physical option receipt，并通过 effect/invariant/
  budget 与 `native-goal-success ∨ certified-base-reentry` 联合 gate；
- prompt 含 `Preserve` 但外部监督器观察到 invariant FALSE 时必须拒绝 commit；
- wrong-object/collateral case 不错误提交原 occurrence；
- protected invariant 破坏且无已启用 damage-repair skill 时立即
  `TERMINAL_COLLATERAL_DAMAGE`；
- recovery server/VAL/grounder 故障及零动作 rebuild loop 均按第 13 节有界终止；
- topology matched arms 复用同一 trigger、root、seed、Capability、Supervisor 和 VAL，只有
  flat goals、global plan 与 local causal slice 不同。

### 16.3 真实仿真验收

单元测试不能替代真实 rollout。每个能力合同的成功率、干扰率和 step distribution 必须
来自 fresh-env LIBERO 仿真，并保存视频、动作、facts、certificate 和 checkpoint hashes。
非终止 re-entry 还必须保存 Base continuation rollout；terminal Task 5/8 结果不得计入
`base_continuation_success`。论文若作低于 2% 的 violation-rate 声明，确认样本必须满足
第 12.2 节独立性、post-selection 和区间口径要求。

## 17. 理论声明边界

新理论不声称“VAL-valid ⇒ physical success”。令触发状态为 `s`，已认证 option 为 `o`，
能力合同只提供在定义 state class 内的经验条件：

- `P(effect(o) ∧ preserve(I) | s)` 的校准下界；
- option step-cost 的 q90；
- 不支持状态的显式拒绝集合。

LOGIV 的作用是确保只从当前 facts 生成 goal-consistent、invariant-preserving、VAL-valid
且预算可行的候选；`π_recover` 的作用是提供这些候选的物理技能闭包。系统总体性能是否
非降最终由 paired positive flips 减去 negative flips 决定，必须实验验证，不能由符号
正确性单独推出。

同样不作以下无条件声明：三帧稳定事实不等于失败；Initial DAG 不等于 Base 实际轨迹；
符号 re-entry signature 不等于 Base continuation；开发集零 observed violation 不等于低
破坏率；存在 520-action budget 不等于 overlay 不会震荡；Full 优于一个多因素 baseline
不等于 DAG 单独有效。这六项分别由第 5.2、7、8.2、12.2、13 和 14.5 节的可验证合同约束。

相关设计原则与既有工作一致：SayCan 使用技能 value/affordance 约束高层计划；RT-H
对语言干预层级进行训练；FLARE 通过失败扰动和 bridging segments 学习恢复。本项目的
差异点是把训练后的恢复执行器放入 certificate-bound、局部、能力校准的 LOGIV overlay，
并保持 nominal Base policy 不变。

## 18. 产物与审计

每次运行至少产生：

- root snapshot manifest、physics state、pending Base chunk/request envelope、event-origin parent、
  parent-trajectory lineage 和统计 independence-unit；
- bridge trajectory/dataset manifest、root/group/independence-unit split、fresh relabel provenance
  与训练 manifest builder hash；
- training config、normalization provenance、checkpoint hashes 和 validation report；
- `RecoverySkillContract` registry；
- Base prefix、candidate/confirmed evidence、monitor-contract hash、shadow/recovery certificate
  generations、repair plan、VAL certificate、permit/retry ledger、option receipt、effect/invariant
  facts；
- paired continue-Base/recovery branch outcomes、continuation contracts、intervention-advantage
  intervals 与 violation upper-bound configuration；
- paired episode records、逐任务报告与 bootstrap configuration。

未经第 12 节验证的 checkpoint、手工挑选的成功视频、700-step diagnostic 或已暴露 seed
上的调参结果只能标记为 development evidence，不能进入论文主表。

## 19. 参考资料

- OpenPI fine-tuning 与 LIBERO 配置：`external_repos/openpi/README.md`、
  `external_repos/openpi/src/openpi/training/config.py`
- LIBERO simulator state API：
  `external_repos/openpi/third_party/libero/libero/libero/envs/env_wrapper.py`
- SayCan: <https://arxiv.org/abs/2204.01691>
- Inner Monologue: <https://proceedings.mlr.press/v205/huang23c.html>
- Partial-order plan execution monitoring: <https://www.ijcai.org/Abstract/11/330>
- RT-H: <https://arxiv.org/abs/2403.01823>
- FLARE: <https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_FLARE_A_Failure-Aware_Framework_for_Autonomous_Correction_and_Recovery_in_CVPR_2026_paper.html>
