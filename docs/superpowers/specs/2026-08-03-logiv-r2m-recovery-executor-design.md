# LOGIV-R2M：Base 保底的执行器感知恢复设计

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-03
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1
- Status: 已按第一轮书面反馈修订，待用户二次复核
- Upstream Evidence:
  - `docs/experiments/2026-08-03-logiv-v53-failure-seed-retrospective.md`
  - `docs/superpowers/specs/2026-08-02-logiv-libero-closed-loop-design.md`
  - `docs/superpowers/specs/2026-08-02-logiv-frontier-execution-optimization-design.md`

## 1. 决策摘要

后续主线停止继续优化“从第 0 步接管并把完整任务拆成多次冻结 π₀.₅ 调用”的
`FULL_LOGIV`。新主线采用 **LOGIV-R2M（Repair-to-Manifold）**：

1. 原始 `π_base` 使用官方完整任务文本连续执行，checkpoint 和 nominal 行为保持不变。
2. LOGIV 在影子模式维护 Ground Facts、Goal、VAL certificate、Causal DAG、已完成目标与
   剩余义务，但在没有可靠偏差时不改变 Base 动作。
3. 只有稳定、可验证且存在已校准恢复技能的偏差才触发局部接管。
4. 一个独立训练的 `π_recover` 连续执行与 VLA 训练粒度一致的恢复宏，不在
   `holding` 等中间状态切断策略上下文。
5. 外部状态监督器在 fresh observation 上验证目标 effect、protected invariants、Base
   兼容性和剩余预算；四项同时成立后才能提交恢复并切回 `π_base` 或进入 native success
   检查。

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

系统由七个可独立测试的组件组成：

1. `NominalPolicyClient`：只向原始 checkpoint 发送官方完整任务文本；接口行为与 Base
   evaluator 相同。
2. `ShadowLogivMonitor`：读取 observation，维护 fresh facts、已完成目标、稳定偏差与
   当前剩余义务；没有控制权限。
3. `RecoveryPlanner`：从 handoff state 重建 Current Problem，调用原有 Repair、VAL 和
   DAG compiler，只产生局部、certificate-bound recovery plan。
4. `RecoveryCapabilityRegistry`：声明哪些 grounded recovery option 有训练支持，保存
   effect success、protected-invariant violation、step-cost 与数据/checkpoint provenance。
5. `RecoveryPolicyClient`：只连接独立 `π_recover` checkpoint，每次 permit 只执行一个
   有限模板定义的原子 recovery macro。
6. `ExternalInvariantSupervisor`：独立于 recovery prompt 和 policy 输出，在执行前、执行中
   和停止后验证 declared effect、已完成 Goal invariants、Base-compatible state 与 budget
   reserve；它是 recovery commit 的唯一事实来源。
7. `OverlayController`：原子完成 Base → Recovery → Base 的切换、action-chunk flush、
   budget accounting、effect/invariant gate 和 terminal handling。

```text
official task → π_base → action stream ───────────────────────────────┐
                    │                                                │
                    └→ Shadow LOGIV → stable deviation? ── no ──────┘
                                             │ yes
                                             ↓
                            re-ground Current Problem + protected facts
                                             ↓
                              Repair + complete VAL + local DAG
                                             ↓
                          capability / interference / budget gate
                                  │ reject              │ permit
                                  ↓                     ↓
                            continue π_base       one π_recover macro
                                                        ↓
                                      ExternalInvariantSupervisor
                                                        ↓
                           effect ∧ invariants ∧ base-compatible ∧ budget
                                          │ fail             │ pass
                                          ↓                  ↓
                                  bounded replan       π_base or success
```

组件之间只交换版本化 records，不共享可变内部状态。`RecoveryPlanner` 不知道模型服务
实现，`RecoveryPolicyClient` 不决定任务 Goal，能力注册表不修改 PDDL 语义，prompt 中的
`Preserve` 文本也不能替代 `ExternalInvariantSupervisor` 的物理事实验证。

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

第一阶段主结果中的 `ShadowLogivMonitor` 限定为本地、只读模块：使用已声明的 oracle/
registered facts，在既有 replan boundary 检查结果，不等待外部 VLM/LLM，也不改变动作
调度。它可以增加 wall-clock latency 和本地计算量，因此必须单独报告 monitor calls、
grounding/VAL latency、CPU/GPU time 与峰值内存。

未来如果接入在线 VLM/LLM，必须使用独立 client 和 RNG、异步非阻塞请求，并把
`base_policy_requests`、`shadow_vlm_requests`、`recovery_policy_requests` 分桶记录。此时
不能再声称“总推理请求与 Base 相同”，只能声称动作流、policy-action steps 和未接管
episode outcome 与配对 Base 相同。外部请求结果只能在既有 observation generation
边界被采纳，Controller 不得停住 Base 等待响应。

这使新方法的工程起点是 Base 437/500；旧 Full 427/500 只作为消融，不是新系统需要
先追回的性能债务。

## 7. 偏差触发与预算门控

### 7.1 第一阶段可触发偏差

第一阶段只允许经过能力注册的事实模式触发：

- Task 5 类：目标 book 稳定处于 registered recovery surface，且不在目标 caddy region。
- Task 8 类：一个目标 pot 已稳定在 stove，另一个 pot 稳定处于 recovery surface；或
  当前 holding object 与 remaining target 不一致。

“稳定”要求同一事实在三个连续 monitor observation 中为 TRUE；UNKNOWN、单帧 location
divergence 或抓取过渡态均不触发。monitor observation 默认与 `replan_steps=5` 对齐，
因此三次确认最多占用 15 个 Base policy steps，但不增加动作预算。

### 7.2 许可条件

一个修复计划只有同时满足以下条件才能替换 Base：

1. 当前 snapshot 通过 exactly-one、holding/handempty 和 access state lint。
2. 完整 repair plan 通过 signed trace、VAL 与 RetryPolicy。
3. 每个 physical option 都存在匹配当前 state class 的已启用能力合同。
4. 计划所有 option 的 protected invariants 包含当前已完成 Goal facts。
5. `remaining_steps >= plan_step_q90 + post_recovery_reserve`；reserve 按第 8.3 节定义。
6. 候选的校准成功率和干扰风险通过第 12 节门槛。

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

### 8.2 Re-entry state

恢复目标按下列优先级选择：

1. 直接满足缺失 Goal 且保持所有已完成 Goal；
2. 到达 demonstration-derived re-entry signature：handempty/access 状态一致、已完成
   Goal 保持、剩余对象位于 nominal policy 已覆盖的位置类别；
3. 没有经过验证的 re-entry target 时拒绝接管。

Task 5/8 第一阶段主要使用第 1 类，即恢复宏直接完成掉落分支。第 2 类只建立接口和
记录格式，不在没有数据支持时扩展为任意“看起来正常”的状态。

### 8.3 原子恢复提交合同

`π_recover` 停止后必须采集 fresh、稳定 snapshot。只有以下联合谓词为 TRUE 才能提交
本次恢复：

```text
recovery_commit :=
    effect_satisfied
    ∧ completed_goal_invariants_preserved
    ∧ base_compatible_state
    ∧ remaining_budget_sufficient
```

- `effect_satisfied`：option 声明的正/负 effects 均由 fresh facts 可靠确认；UNKNOWN 不
  满足该项。
- `completed_goal_invariants_preserved`：handoff 时已满足的所有 Goal facts 在执行中未被
  确认破坏，停止后仍可靠为 TRUE。prompt 的 `Preserve` 句只提供模型条件，不是证据。
- `base_compatible_state`：snapshot 匹配能力合同登记的 re-entry signature，且
  handempty/holding、object location、access 和 mutually-exclusive facts 无冲突；只满足
  option effect 但仍拿错物体或处于半完成 articulation 状态时不得切回 Base。
- `remaining_budget_sufficient`：如果 native Goal 已满足且外部 evaluator 确认成功，后续
  所需 model-action steps 为 0；否则必须保留该 re-entry signature 的
  `nominal_resume_steps_q90`。该 q90 由 nominal demonstrations/paired Base 轨迹冻结，
  不能在当前 episode 动态猜测。permit 阶段的 `post_recovery_reserve` 使用同一定义。

effect confirmation 和 post-stop observation 使用所有方法臂相同、预先冻结的只读/保持
状态协议，不得借机完成新的物理目标，也不得作为 LOGIV 独享的额外 model-action budget。
520-step 主预算统一统计 `π_base` 与 `π_recover` 实际产生并执行的 model actions。

联合谓词任一项为 FALSE 或 UNKNOWN 时，本次 option 不提交，`π_recover` 必须停止；后续
只能由 LOGIV 在 fresh Current Problem 上产生新的 certificate 和 permit，不能让同一次
policy 调用顺势执行下一个宏。

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
- task/episode/master seed、policy seed、simulator seed 和 handoff step；
- BDDL、initial-state hash、first-frame hash、当前 observation/proprioception；
- Base action-prefix hash、未执行 chunk 长度与 policy request generation；
- fresh true/false/unknown facts、Goal、protected facts、candidate repair 和 certificate；
- trigger class、monitor evidence 和 native evaluator status。

LIBERO `OffScreenRenderEnv` 已提供 `get_sim_state()` 与 `set_state()`。恢复 root 时使用
fresh env、设置 simulator state、`sim.forward()` 并强制更新 observables。physics state
round-trip 与 policy RNG continuation 分开验证；branch rollout 使用新的显式 branch seed，
不声称复现原 Base 的隐藏 policy state。

### 10.2 状态来源

仅从 nominal clean demonstration 或干净 scripted trajectory 训练不能建立恢复能力合同。
数据必须由下列两部分共同组成：

- 真实失败：当前 Task 5/8 Base、旧 Full 与 overlay 暴露的 drop、wrong-held、crowded-target
  状态。
- 系统扰动：从 nominal demonstration state 创建同类可控偏差，包括 object drop、
  wrong-object grasp、partial placement/未完全释放、一个 Goal 已满足、crowded target 被
  已完成物体占用，以及适用任务中的 drawer/microwave articulation half-open。

数据划分使用可机械检查的 `recovery_group_id`。该 ID 至少绑定 task、scene XML hash、
object instance/type IDs、initial-state hash、parent nominal snapshot、perturbation family
和 root physics-state hash。所有满足以下任一关系的 samples 必须进入同一 split：

- 来自同一 initial state 或同一 parent nominal snapshot；
- 只改变 perturbation seed、branch seed、相机噪声或微小 object pose；
- 使用相同 root snapshot 但由不同 teacher/policy 产生；
- 是同一 partial-placement、drop、wrong-grasp 或 half-open 事件的时间邻近帧。

train/dev/held-out 使用互斥的 initial-state IDs、root IDs 和 perturbation seeds；split
validator 在训练前检查交集并计算 pose/state fingerprint 近重复。若 LIBERO 固定任务没有
新的 object instance 或 scene 可用于真正的 object/scene holdout，能力合同必须明确写成
“仅在已见 object/scene class 内验证”，不得宣称跨物体或跨场景泛化。当前已查看的 50
episodes 只用于 development；冻结代码后的确认运行使用新的 policy master seed。

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

训练 loss 只用于优化诊断，checkpoint 选择只依据固定 held-out root states 上的闭环
effect success、protected-invariant violation 和 steps。禁止根据 500-episode 最终主结果
回选 checkpoint、模板或 trigger threshold。

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
- status：`EXPERIMENTAL`、`ENABLED` 或 `REJECTED`。

一个 state class 至少有 30 个未用于训练的 root states，且同时满足以下条件才可
`ENABLED`：

- effect success point estimate ≥ 80%；
- Wilson 95% lower bound ≥ 65%；
- validation 中 protected-invariant violation 为 0；
- policy-action step q90 ≤ 180；
- root-state round-trip、prompt rendering 和 checkpoint provenance 全部通过。

这些是工程启用门槛，不表述为真实世界安全概率保证。运行时只有 `ENABLED` 合同可以
产生 safety permit；runtime snapshot 必须与登记 preconditions、object/location/articulation
支持范围逐项匹配。合同不匹配、统计缺失或 hash 漂移均 fail closed 到继续 Base。

## 13. 错误处理与有界终止

- monitor grounding UNKNOWN：记录并继续 Base，不猜测偏差。
- root snapshot 保存失败：禁止 recovery handoff，继续 Base。
- VAL error、certificate/hash mismatch：拒绝接管。
- recovery server timeout/exception：停止当前同步 dispatch；只有 fresh snapshot 仍通过
  第 8.3 节完整联合提交合同时才可切回 Base 或进入 native success 检查，否则必须在新
  certificate/permit 下进行有界修复或 terminal。
- effect 未成立但 invariants 保持：在总预算和 RetryPolicy 内尝试下一个已认证候选。
- 任一 protected invariant 由 TRUE 变 FALSE：立即停止 recovery；仅允许 LOGIV 生成以
  该 invariant 为首要 obligation 的认证恢复，不能直接回 Base 掩盖损害。
- 520 actions 耗尽：停止所有 policy dispatch；settling/只读评价不计 policy action，
  但不得执行改变物理状态的额外动作。

所有预算跨 Base/Recovery/graph version 统一记账，不因切换服务或重新安装 repair graph
重置。

## 14. 实验设计与消融

### 14.1 方法臂

1. `BASE`：原始 `π_base`、官方任务文本和官方 evaluator 行为。
2. `SHADOW_LOGIV`：运行本地 monitor/grounding，但永不接管；用于证明动作、policy-action
   steps、Base policy requests 和 native outcome 等价，同时单独测量 shadow compute。
3. `BASE_RECOVERY_NO_LOGIV`：相同 `π_recover`，使用非 VAL 的简单 recovery-surface
   heuristic；隔离训练本身的贡献。
4. `LOGIV_FROZEN_RECOVERY`：LOGIV overlay 调用原 frozen π₀.₅；保留当前负/零改善机制
   证据。
5. `LOGIV_R2M`：完整能力门控 LOGIV + 训练后的 `π_recover`，唯一主候选。

### 14.2 公平性

- 所有主比较使用相同 checkpoint lineage、initial state、fresh-env-per-episode、first
  frame、policy/simulator seed、`replan_steps=5` 和 520 policy actions。
- Base prefix 到 handoff step 的 action hash 必须与配对 Base 相同。
- planner/VAL 的 wall time 不折算为 policy actions，但单独报告总 wall time、
  `base_policy_requests`、`shadow_vlm_requests`、`recovery_policy_requests`、monitor calls
  和 planner calls。
- 每个方法独立启动或使用显式 episode-seeded policy wrapper，不能让方法顺序共享隐藏
  RNG 状态。

### 14.3 主要变量与 estimand

- 自变量：方法臂。
- 主要因变量：native LIBERO episode success。
- 关键次级指标：Base-only/LOGIV-only flips、recovery trigger rate、permit rate、bridge
  success、collateral violation、handoff step、repair steps、total steps 和 terminal cause。
- 主要 estimand：十个任务等权的 paired success-rate difference；每个任务 50 episodes，
  因样本数相同，其点估计与 500-episode 总成功率差一致。

### 14.4 统计分析

- 每任务报告 `x/50`、Wilson 95% interval、正翻转与负翻转。
- 总体在 task 内对共享 pairs 重采样，做 10,000 次 paired stratified bootstrap。
- 报告总体 paired difference 的 95% interval；McNemar exact test 作为 discordant-pair
  敏感性分析，不替代效应量和区间。
- checkpoint/trigger 开发只看 development 结果；locked 500 只运行冻结配置一次。

## 15. 分阶段实施与 Go/No-Go

### Phase 0：状态与数据基础设施

- 实现 root snapshot 保存/恢复、branch seed、trajectory recorder 和 dataset manifest。
- 通过 simulator-state round-trip、fresh fact consistency、root-level split 和无触发 Base
  等价测试。
- 未通过时不实现训练或运行大规模 rollout。

### Phase 1：Task 5/8 recovery skill

- 收集/生成两类 macro bridge，训练 REC-A/REC-B。
- 每个 skill-state class 按第 12 节进行 held-out capability validation。
- 不达标的合同标记 `REJECTED`，不能通过放宽 trigger 或增加运行预算上线。

### Phase 2：Task 5/8 paired development

- 在已暴露 development protocol 上运行配对 Task 5/8。
- Go 条件：两任务合计净提升至少 `+10/100`，每个任务负翻转不超过 2，且所有干预均有
  匹配能力合同。
- 未通过时只允许修改一个已定位因素：数据覆盖、teacher、训练候选、trigger 或能力
  校准；禁止返回无穷 prompt sweep。

### Phase 3：冻结与全矩阵确认

- 冻结代码、dataset version、checkpoint、template、capability registry 和 thresholds。
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

- stable trigger 的三次 TRUE、TRUE→UNKNOWN、抓取过渡态和 false-positive 去抖；
- shadow 本地计算增加 latency 但不改变 Base action/outcome；模拟 external VLM 时额外请求
  被分桶且 Controller 不等待响应；
- Base/Recovery request generation 与 stale callback no-op；
- capability hash/state-class/remaining-budget 匹配；
- protected facts 自动进入 repair constraints；
- 联合提交谓词四个分量分别 FALSE/UNKNOWN 时均不 commit，也不执行第二个 option；
- candidate 排序、q90 budget 和没有能力合同时继续 Base；
- initial state、scene、object、perturbation/root seed 与近重复 state 的 split 防泄漏；
- total 520-step budget 跨策略切换不重置。

### 16.2 集成测试

- inactive monitor 与 Base action arrays、steps、`π_base` policy requests 完全相等；
- 若启用在线 shadow VLM，只要求 Base policy requests 相等，并验证额外请求与延迟独立
  记录；
- simulator snapshot 恢复后 true/false facts 与保存时一致；
- handoff 前 Base action-prefix hash 相等；
- 一个成功 recovery macro 只产生一个 physical option receipt，并通过 effect/invariant
  /base-compatibility/budget 四项 gate 返回 Base 或 native success；
- prompt 含 `Preserve` 但外部监督器观察到 invariant FALSE 时必须拒绝 commit；
- wrong-object/collateral case 不错误提交原 occurrence；
- recovery server/VAL/grounder 故障均按第 13 节有界终止。

### 16.3 真实仿真验收

单元测试不能替代真实 rollout。每个能力合同的成功率、干扰率和 step distribution 必须
来自 fresh-env LIBERO 仿真，并保存视频、动作、facts、certificate 和 checkpoint hashes。

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

相关设计原则与既有工作一致：SayCan 使用技能 value/affordance 约束高层计划；RT-H
对语言干预层级进行训练；FLARE 通过失败扰动和 bridging segments 学习恢复。本项目的
差异点是把训练后的恢复执行器放入 certificate-bound、局部、能力校准的 LOGIV overlay，
并保持 nominal Base policy 不变。

## 18. 产物与审计

每次运行至少产生：

- root snapshot manifest 与 physics state；
- bridge trajectory/dataset manifest 与 root-level split；
- training config、normalization provenance、checkpoint hashes 和 validation report；
- `RecoverySkillContract` registry；
- Base prefix、trigger、repair plan、VAL certificate、option receipt、effect/invariant facts；
- paired episode records、逐任务报告与 bootstrap configuration。

未经第 12 节验证的 checkpoint、手工挑选的成功视频、700-step diagnostic 或已暴露 seed
上的调参结果只能标记为 development evidence，不能进入论文主表。

## 19. 参考资料

- OpenPI fine-tuning 与 LIBERO 配置：`external_repos/openpi/README.md`、
  `external_repos/openpi/src/openpi/training/config.py`
- LIBERO simulator state API：
  `external_repos/openpi/third_party/libero/libero/libero/envs/env_wrapper.py`
- SayCan: <https://arxiv.org/abs/2204.01691>
- RT-H: <https://arxiv.org/abs/2403.01823>
- FLARE: <https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_FLARE_A_Failure-Aware_Framework_for_Autonomous_Correction_and_Recovery_in_CVPR_2026_paper.html>
