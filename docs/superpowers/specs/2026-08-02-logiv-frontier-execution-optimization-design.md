# LOGIV DAG-Frontier 执行优化设计

**日期：** 2026-08-02  
**状态：** 已按用户的持续执行授权冻结  
**目标：** 修复 task 8 上 Full LOGIV 因不自然 occurrence 边界而低于 Base 的回退，同时保持固定 PDDL、完整 VAL、事实门控、认证修复和真实非链 DAG。

## 1. 已验证的根因

- 官方 OpenPI LIBERO evaluator 对原始任务文本连续执行，`replan_steps=5`，LIBERO-10 horizon 为 520。
- task 8 的 50 条训练示范长度中位数为 408.5，范围为 341--517。`moka_pot_2` 到炉子的中位步骤为 181，`moka_pot_1` 为 396；48/50 条可完整核验示范均先完成 pot 2。
- 当前 PDDL 顺序 `pot_2 -> pot_1` 与示范顺序一致，且两个节点之间没有必要边，初始 DAG action-layer width 为 2。
- 当前 executor 却在 pot 2 effect 成立时立即停止并清空 action chunk，再从新 observation 重新启动第二个 occurrence。20-seed paired 诊断中，Full 为 10/20，Base 为 12/20；9/10 Full failures 以 520 步预算耗尽结束。
- 同一个 policy server 上，seed 5 使用官方任务原文 `put both moka pots on the stove` 的 Base 在 397 步成功；使用 `Put both moka pots on the stove.` 的连续 Stage-only 跑满 520 步失败。nominal prompt 的大小写和标点漂移足以改变 π0.5 轨迹。

因此主要问题不是 VAL 找不到计划，而是固定 260-step occurrence 边界和 nominal prompt 分布漂移打断了训练中的长程行为。

## 2. 备选方案

### A. 继续细化 `pick -> place-held`

优点是 PDDL 状态更细。缺点是前序 v3/v4 已证明它增加 policy restart 和 prompt distribution shift；训练示范的自然单位是完整双物体任务，不是每个裸技能。否决。

### B. 只动态调换 ready DAG 节点

优点是符号语义最保守。缺点是 task 8 的顺序已与 48/50 示范一致，重排不能消除每个节点处的 STOP/flush。保留为后续独立优化，不作为本轮首选。

### C. DAG-frontier 连续执行（采用）

从当前 certified graph 计算 ready action antichain。只有 frontier 至少含两个节点，且 executor 对这些动作渲染出的 prompt 完全一致时，才允许当前 attempt 使用 frontier completion hint。π0.5 继续执行同一个训练原文；在线 verifier 检查 frontier 所有动作 `Add/Del` 的并集，而不是在当前节点 effect 首次出现时立即停止。

这与低频 macro execution、频繁事实验证、事件触发重规划的思路一致；不会向 PDDL 添加虚假 `place-both` schema，也不会添加 action-adjacency edge。

## 3. 精确语义

1. Graph 仍由 certificate-bound 线性计划编译；task 8 的两个 `place-on` 节点之间不得有边，初始 width 必须为 2。
2. `ready_action_ids(committed)` 只返回所有 action predecessor 已提交的未提交节点，按 canonical rank 稳定排序。
3. Controller 的 dispatch gate 仍只为 canonical current occurrence 创建 attempt。frontier hint 完全由当前 graph/certificate 派生，不能由 VLM 或 prompt 文件声明因果关系。
4. Executor 只有在 frontier 中所有动作渲染为完全相同的 prompt 时才 coalesce；否则退回单 occurrence effect stop。
5. Coalesced attempt 的停止条件为 frontier union effects 连续满足配置的 confirmation steps，或当前 primary action 出现已定义的 factual target divergence，或既有安全/预算/执行器终止条件。
6. STOPPED、settling 和 fresh post snapshot 后，Controller 仍只检查并提交 primary occurrence 的声明 effects。任何 sibling occurrence 都不能共享 receipt 或被隐式提交。
7. 若连续执行顺带完成 sibling effect，下一次 precondition gate 会从 fresh facts 发现旧 source 已不成立，进入普通认证 Repair。Repair 可以在 Current Problem 已满足原始 Goal 时返回空计划，但必须重新通过 signed trace、VAL、RetryPolicy 和 fresh graph installation。
8. 首个候选保持 episode-global 520 policy-action budget，不给 Full 额外动作。`max_action_steps` 提高到 520 只是取消每 occurrence 的 260 人工截断，global budget 不变。
9. nominal task-8 prompt 精确复用官方任务字符串 `put both moka pots on the stove`。恢复 prompt 仍由 grounded action 和当前 facts 决定，不复用官方 Goal evaluator。

## 4. 审计字段

每个 attempt artifact 增加：

- `completion_actions`：本次 verifier 检查的 certified grounded actions；
- `completion_mode`：`OCCURRENCE` 或 `DAG_FRONTIER`；
- `completion_positive` / `completion_negative`：实际在线检查的 union literals；
- stop reason `observed frontier effects`，与普通 `observed declared effects` 区分。

这些字段只描述执行 hint，不改变 certificate、retry key、action lineage 或 receipt ownership。

## 5. 测试与实验门槛

- 单元测试证明 task 8 ready frontier 含两个无边节点，串行 drawer 任务的 frontier 不会越过 precedence edge。
- executor 测试证明 primary effect 先成立时不会早停，只有 sibling effect 后才以 frontier reason 停止；prompt 不同或 frontier 为单节点时保持旧行为。
- controller 测试证明只提交 primary receipt，sibling 已完成后仍需 fresh certified empty repair 才能结束。
- 先在已暴露 development seeds 上做单变量对比；与 Base 必须共用同一 server process、initial-state hash、first-frame hash、policy/simulator seed、520 global action budget 和 `replan_steps=5`。
- 候选冻结后才运行 episode 20--49 的 paired holdout。若未超过同进程 Base，不进入全量，并回到失败事实审计。
- task 8 的 50-state Full 结果至少达到 32/50 且高于同协议 Base，才满足既有 acceptance gate；随后才运行 10-task 全量。

## 6. 参考依据

- OpenPI 官方 LIBERO evaluator：<https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py>
- Open-Loop Planning, Closed-Loop Verification: Speculative Verification for VLA：<https://arxiv.org/abs/2604.02965>
- Towards Long-horizon Embodied Agents with Tool-Aligned Vision-Language-Action Models：<https://arxiv.org/abs/2605.13119>

后者还表明细粒度 VLA tool invocation 通常需要 tool-aligned post-training；因此本轮优先恢复 checkpoint 已训练的原始任务 prompt 分布。如果无训练的 frontier 方案仍不能越过 Base，再单独设计 task-segment relabeling/fine-tuning，而不把训练收益伪装成无训练 LOGIV 收益。
