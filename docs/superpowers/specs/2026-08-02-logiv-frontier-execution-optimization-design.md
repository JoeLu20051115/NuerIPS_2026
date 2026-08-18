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

## 7. Development iteration 2：瞬时偏离去抖与 follow-up 预算

首个 frontier 候选在同一 policy server 的开发 seeds `2,5,7,8,14` 上得到 `3/5`，与 Base 的 `3/5` 持平，因此没有进入 holdout。artifact 给出两个新的直接证据：

- seed 14 在第 127 步因单帧 target-location divergence 停止，但 settling 后 pot 2 已明确处于 `holding`。随后针对 held-pot 的恢复 attempt 消耗余下 393 步仍未完成。target divergence 必须与 effect gate 一样连续确认，不能由抓取转换期的单帧位置关系触发。
- seed 2 在 520 步结束时 pot 2 已在 stove、pot 1 仍在初始位置，但所有预算已经消耗在同一个 frontier attempt，无法进入第二分支的事实驱动执行。训练示范中 pot 2 到 pot 1 完成的观测间隔为 107--199 步，因此设置 180 个连续步骤的 primary-effect follow-up 窗口；超时只停止当前 attempt，不提交 sibling，并把保留预算交回普通 fresh-fact gate。

iteration 2 仍不添加 PDDL macro 或 action edge。frontier 模式只使用两个 ready actions 共同的 `render_frontier(action)`；若 holding fact 已被确认，可切换为 primary action 的 finish phase，但 receipt ownership 与 effect gate 不变。若后续只剩单一 action，则恢复该 grounded action 的 targeted prompt。全局 520-step budget 不变。候选只有再次在同一开发 seeds 上严格高于 Base 才允许扩大。

后续配对审计还发现，LIBERO 的同一 `init_state` 在独立进程中作为首个 episode 与作为第五个 episode 时可能产生不同的 post-wait first frame；`env.seed()`、NumPy/Python seed 与 `set_init_state()` 的组合不足以清空全部 env-local history。为使 episode seed 真正独立并保证 resume 不改变后续 episode，evaluator 从此为每个 episode 新建并关闭一个 `OffScreenRenderEnv`，并在 run manifest 写入 `simulator_env_lifecycle=fresh-env-per-episode-v1`。旧的 shared-env 结果只保留为开发诊断，不能与新协议结果混合。
