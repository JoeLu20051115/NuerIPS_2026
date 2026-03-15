# DreamZero 双系统架构与评测说明

## 执行摘要

### 核心发现

在 DROID 测试集上（40 episodes，L1+L3），本文保留两套结果口径：

| 对比口径 | System1 总体成功率 | Dual-System 总体成功率 | 增量 |
| --- | --- | --- | --- |
| 原始主结果（$\delta=0.1$） | $43.5\%$ | $46.0\%$ | +2.5pp |
| 补充结果（$\delta=0.14$） | $61.8\%$ | $65.8\%$ | +4.0pp |

### 时间指标说明（统一均摊口径）

当前评测脚本直接输出以下时间字段：

- `policy_inference_time_per_step` / `completion_time`：每步策略推理平均耗时。
- `llm_planning_time`：每个 episode 的一次性规划耗时（Dual-System）。
- `total_episode_time`：episode 总耗时。

$$
T^{\text{total}}_{\text{ep}} = K_{\text{valid}}\cdot T_{\text{policy}} + T^{\text{plan}}_{\text{ep}}
$$

其中 $K_{\text{valid}}$ 是成功完成推理并产出有效动作的步数（不一定恒等于配置的 $K$）。

注意：当前代码未直接输出 `equiv_step_time` 字段；本文时间对比统一使用分析阶段派生的均摊口径
$T_{\text{equiv}} = T_{\text{policy}} + T_{\text{planning}}/K$。

---

## 1. 问题定义

我们在 DROID 测试集上评估长时程操作任务，分为两个层级：

- L1：中等时程任务
- L3：长时程任务

对比两种范式：

- System1：仅使用 DreamZero 策略执行
- Dual-System：System2 先规划，再由 System1 执行

给定长度为 $T$ 的 episode，在其中均匀采样 $K$ 个时间点，基于动作预测误差进行离线评估。

---

## 2. 系统架构

### 2.1 System2：LLM 规划器

System2 使用 LLM 将高层任务拆解为原子子任务序列。实现细节如下：

- 提示词要求输出 3-5 条子任务；
- 解析后会截断到最多 5 条；
- 若解析为空，则回退为 1 条（直接使用原任务文本）。

因此代码实际可出现的子任务数范围是：

$$
\mathcal{L} = [\ell_1, \ell_2, \dots, \ell_M], \quad 1 \le M \le 5.
$$

### 2.2 System1：DreamZero 策略服务

System1 在每个采样时刻 $t$ 接收多模态观测并输出动作，评估使用首步预测动作的关节维度：

$$
\hat{\mathbf{a}}_t \in \mathbb{R}^7.
$$

### 2.3 Prompt 组合方式

设任务文本为 $g$，任务 token 为 $\tau$，当前子任务为 $\ell_t$，则：

- `subtask`：$p_t = \ell_t$
- `task_only`：$p_t = g$
- `task_token_only`：$p_t = \tau$
- `hybrid`：$p_t = g;\ell_t$
- `task_token_hybrid`：$p_t = \tau;\ell_t$

说明：脚本默认 `prompt_mode=task_token_hybrid`；

### 2.4 双系统统一建模（架构公式）

为更清晰地刻画 System2 规划与 System1 执行的关系，可将整体策略写为分层混合形式：

$$
\pi(a_t \mid o_{1:t}, g) = \sum_{m=1}^{M} q_{\phi}(m \mid o_{1:t}, g)\,\pi_{\theta}(a_t \mid o_t, z_m)
$$

其中，$g$ 为全局任务，$z_m$ 为第 $m$ 个子任务，$q_{\phi}$ 是高层子任务选择分布（System2），$\pi_{\theta}$ 是低层执行策略（System1）。

System2 对子任务序列的生成可写为：

$$
z_{1:M} \sim p_{\phi}(z_{1:M} \mid g, c), \quad 1 \le M \le 5
$$

其中 $c$ 表示 episode 上下文（如元数据与场景信息）。

为将规划结果与执行时刻对齐，引入时间软分配：

$$
q_{\phi}(m \mid t, g) = \frac{\exp\left(-\frac{|t/T-\mu_m|}{\tau}\right)}{\sum_{j=1}^{M}\exp\left(-\frac{|t/T-\mu_j|}{\tau}\right)}
$$

该形式对应当前硬切分 $m_t=\left\lfloor tM/T \right\rfloor$ 的平滑化扩展。

在语言条件层面，可用门控融合统一 task 与 sub-task 语义：

$$
e_t = \lambda_t E(g) + (1-\lambda_t)E(z_{m_t}), \quad \lambda_t = \sigma(w^{\top} h_t)
$$

其中 $E(\cdot)$ 为文本编码器，$\lambda_t$ 控制全局任务语义与当前子任务语义的融合权重。

---

## 3. 数据流与运行逻辑

对每个 episode：

1. 读取 parquet 轨迹与视频时间戳。
2. 从元数据解析任务描述。
3. 运行一次 planner 得到子任务序列 $\mathcal{L}$。
4. 从 $[0, T-1]$ 均匀采样 $K$ 个时间点。
5. 将采样时刻映射到子任务索引：

$$
m_t = \min\left(\left\lfloor \frac{t \cdot M}{T} \right\rfloor, M-1\right).
$$

6. 组装 prompt，调用策略推理，计算误差与耗时。
7. 汇总 episode 指标，再汇总层级指标。

---

## 4. 指标定义与代码口径

设真实关节动作为 $\mathbf{a}_t \in \mathbb{R}^7$，预测为 $\hat{\mathbf{a}}_t \in \mathbb{R}^7$。

### 4.1 单步误差

$$
e_t = \lVert \hat{\mathbf{a}}_t - \mathbf{a}_t \rVert_2.
$$

### 4.2 Episode 级指标

记有效采样集合为 $\mathcal{S}_{\text{valid}}$，其大小为 $K_{\text{valid}}$。

$$
E_{\text{ep}} = \frac{1}{K_{\text{valid}}}\sum_{t\in\mathcal{S}_{\text{valid}}} e_t
$$

$$
\mathrm{SR}_{\text{ep}} = \frac{1}{K_{\text{valid}}}\sum_{t\in\mathcal{S}_{\text{valid}}}\mathbf{1}(e_t<\delta)
$$

$$
T_{\text{policy}} = \frac{1}{K_{\text{valid}}}\sum_{t\in\mathcal{S}_{\text{valid}}}\Delta t_t
$$

代码字段对应：

- `action_l2_error`：$E_{\text{ep}}$
- `success_rate`：$\mathrm{SR}_{\text{ep}}$
- `policy_inference_time_per_step` 与 `completion_time`：$T_{\text{policy}}$
- `llm_planning_time`：$T^{\text{plan}}_{\text{ep}}$
- `total_episode_time`：$K_{\text{valid}}\cdot T_{\text{policy}} + T^{\text{plan}}_{\text{ep}}$
- `video_mse`：误差序列方差 $\mathrm{Var}(\{e_t\})$（不是像素视频误差）

注：可额外派生

$$
T^{\text{equiv}}_{\text{step}} = T_{\text{policy}} + \frac{T^{\text{plan}}_{\text{ep}}}{K_{\text{valid}}}
$$

但该字段当前未在脚本中直接输出。

### 4.3 层级统计

对某层级 $N$ 个 episode 的指标 $x_1,\dots,x_N$，汇总均值与标准差：

$$
\mu = \frac{1}{N}\sum_{j=1}^{N}x_j,\qquad
\sigma = \sqrt{\frac{1}{N}\sum_{j=1}^{N}(x_j-\mu)^2}.
$$

成功率均值额外报告 bootstrap 95% CI（$B=2000$）。

### 4.4 伪代码
      for episode in test_episodes:
          df = load_parquet(episode)
          timestamps = df["timestamp"]
          ep_len = len(df)
          video_paths = load_episode_videos(episode)
          task_description = get_episode_task_description_with_fallback(episode, df)

          if planner_mode == "llm":
              sub_instructions, planner_meta, llm_plan_time = LLM_plan(task_description)
          elif planner_mode == "heuristic":
              sub_instructions, planner_meta, llm_plan_time = heuristic_plan(task_description), {"planner_mode": "heuristic"}, 0.0
          else:
              sub_instructions, planner_meta, llm_plan_time = [task_description], {"planner_mode": "disabled"}, 0.0

          eval_indices = evenly_sample_indices(ep_len, eval_steps)
          action_errors, inference_times, sub_task_assignments = [], [], []

          for step_idx in eval_indices:
              sub_task_idx = min(int(step_idx * len(sub_instructions) / ep_len), len(sub_instructions) - 1)
              current_instruction = sub_instructions[sub_task_idx]
              sub_task_assignments.append(sub_task_idx)

              try:
                  policy_client.reset({"session_id": make_session_id(episode, step_idx)})
              except:
                  policy_client = reconnect_policy_client()
                  policy_client.reset({"session_id": make_session_id(episode, step_idx)})

              obs = {
                  "images": load_frames_at_timestamp(video_paths, timestamps[step_idx]),
                  "observation/joint_position": get_joint_state(df, step_idx),
                  "observation/gripper_position": get_gripper_state(df, step_idx),
                  "prompt": current_instruction,   # subtask mode
                  "session_id": make_session_id(episode, step_idx),
              }

              t0 = now()
              pred_action = policy_client.infer(obs)
              inference_times.append(now() - t0)

              gt_action = get_ground_truth_action(df, step_idx)
              action_errors.append(compute_joint_l2(pred_action, gt_action))

          save_episode_result(
              success_rate=mean(err < threshold for err in action_errors),
              action_errors=action_errors,
              llm_plan_time=llm_plan_time,
              inference_times=inference_times,
              sub_instructions=sub_instructions,
              sub_task_assignments=sub_task_assignments,
              planner_meta=planner_meta,
          )

---

## 5. 严格实验协议

原始主结果使用如下口径：

- 测试集：DROID的测试集
- 样本规模：L1=20，L3=20，总计 40 episodes
- 成功阈值：$\delta=0.1$
- 采样步数：$K=10$
- Dual-System 复现实验命令配置：`prompt_mode=hybrid`, `planner_temperature=0.0`
- 代码默认值（未传参时）：`prompt_mode=task_token_hybrid`
- 推理服务：`localhost:8000`

---

## 6. 实验结果

### 6.1 原始主结果（统一阈值 $\delta=0.1$）

**System1 基线**

- L1 成功率：$40.50\% \pm 23.12\%$
- L3 成功率：$46.50\% \pm 17.97\%$
- Overall：$43.50\%$
- Completion Time（$T_{\text{equiv}}$）：L1 $2.1233 \pm 0.0044$ s/step，L3 $2.1256 \pm 0.0020$ s/step

**Dual-System（2026-03-10）**

- L1 成功率：$41.5\% \pm 20.6\%$
- L3 成功率：$50.5\% \pm 22.7\%$
- Overall：$46.0\%$
- Completion Time（$T_{\text{equiv}}$）：$2.24 \pm 0.0137$ s/step

### 6.2 主结果对比（$\delta=0.1$）

| 系统 | Overall SR | Completion Time (s/step, $T_{\text{equiv}}$) | 时间增量（vs System1） |
| --- | --- | --- | --- |
| System1 | 43.5% | 2.124 | - |
| Dual-System | 46.0% | 2.24 ± 0.0137 | +0.116s |

### 6.2.1 显著性补充（基于现有结果直接计算）

为避免仅报告均值差，本节补充 $\Delta \mathrm{SR}=\mathrm{SR}_{\text{Dual}}-\mathrm{SR}_{\text{System1}}$ 的统计量（对现有 episode 级结果做 bootstrap 95% CI，并报告 Welch t-test p 值）：

| 阈值口径 | 层级 | $\Delta$SR (pp) | 95% CI (pp) | p-value |
| --- | --- | --- | --- | --- |
| $\delta=0.1$ | L1 | +1.0 | [-12.5, +14.5] | 0.8887 |
| $\delta=0.1$ | L3 | +4.0 | [-8.5, +16.5] | 0.5506 |
| $\delta=0.1$ | Overall | +2.5 | [-6.75, +12.0] | 0.6095 |
| $\delta=0.14$（双方同阈值） | L1 | +3.0 | [-11.0, +17.0] | 0.6823 |
| $\delta=0.14$（双方同阈值） | L3 | +5.0 | [-6.0, +15.5] | 0.3753 |
| $\delta=0.14$（双方同阈值） | Overall | +4.0 | [-5.0, +13.0] | 0.3897 |

结论：当前样本下，Dual-System 的点估计增益为正，但上述区间均覆盖 0，统计上应表述为“正向趋势（preliminary evidence）”，而非“显著优于”。

### 6.3 主实验与对比实验总表

| 实验模块 | 阈值口径 | System1 L1 SR | System1 L3 SR | System1 Overall | Dual L1 SR | Dual L3 SR | Dual Overall | 时间对比（摘要） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 主实验 | 双方 $\delta=0.1$ | 40.5% ± 23.1% | 46.5% ± 18.0% | 43.5% | 41.5% ± 20.6% | 50.5% ± 22.7% | 46.0% | System1 2.124 s/step, Dual 2.24 ± 0.0137 s/step（$T_{\text{equiv}}$） |
| 对比实验 | $\delta=0.12$ | 50.5% ± 20.9% | 57.5% ± 16.1% | 54% | 52.0% ± 22.5% | 62.0% ± 18.9% | 57% | System1 L1/L3: 2.137/2.130 s, Dual L1/L3: 2.147/2.135 s（$T_{\text{equiv}}$） |
| 对比实验 | $\delta=0.14$ | 58.0% ± 23.8% | 65.5% ± 15.3% | 61.8% | 61.0% ± 21.0% | 70.5% ± 18.8% | 65.8% | System1 L1/L3: 2.132/2.133 s, Dual L1/L3: 2.127/2.130 s（$T_{\text{equiv}}$） |

### 6.4 L3 公平对比（B1/B2 + Dual，$\delta=0.1$）

时间统一采用可比口径：

$$
T_{\text{equiv}} = T_{\text{policy}} + \frac{T_{\text{planning}}}{K}, \quad K=10.
$$

其中：
- System1/B1/B2 不使用 LLM 规划，故 $T_{\text{planning}}=0$，$T_{\text{equiv}}=T_{\text{policy}}$；
- Dual-System 使用 LLM 规划，按每 episode 规划时间均摊到每步；


| 组别 | 含义 | Success Rate | 95% CI | Completion Time（统一口径） | Robustness Score |
| --- | --- | --- | --- | --- | --- |
| System1 旧基线 | task token only | 46.5% | [38.5%, 54.5%] | 2.126 s/step | 0.8748 ± 0.0273 |
| B1 | task_description，无子任务，无 LLM | 44.0% | [35.5%, 53.0%] | 2.207 s/step | 0.8770 ± 0.0310 |
| B2 | task_description + heuristic/chunked subtask，不走 LLM | 46.5% | [37.99%, 56.0%] | 2.206 s/step | 0.8745 ± 0.0313 |
| 现有 Dual-System 主结果 | LLM subtask | 50.5% | [40.5%, 60.5%] | 2.281 s/step（2.130 policy + 1.513/10 planning） | 0.8769 ± 0.0328 |

### 6.5 Case-6 误差剖析（L3, $\delta=0.1$）


#### 逐案例模式
| 模式 | episode | SR（System1 / B2 / Dual） | 可解释性说明 |
| --- | --- | --- | --- |
| Dual 更好 | `episode_001470`, `episode_001669` | `0.4/0.2/0.5`, `0.5/0.3/0.6` `0.2/0.5/0.6`| 多阶段、多目标或跨对象切换任务中，LLM 子任务对“中间目标链”刻画更完整。 |
| Dual 稍差 | `episode_001406`, `episode_001395` |  `0.6/0.6/0.5` | 任务较直接或 B2 分段已足够时，LLM 规划引入额外动作语义（如 open/move）可能干扰策略。 |
| 基本持平/小幅更好 | `episode_001754`, `episode_001827` | `0.5/0.4/0.5`, `0.5/0.4/0.5` | 规划增益有限，表现受策略本体误差主导。 |

#### 可解释性结论

- Dual-LLM 在复杂长链任务上可通过显式中间目标提升成功率，但在简单/短链任务上可能因过度规划导致收益减少。  
- 提升具有条件性而非全局一致性，这解释了“总体均值增益有限但个别任务提升显著”的现象。  
- 相较无规划基线，Dual 主要额外代价来自一次性规划时延（本实验约 2.0 s/episode），policy step 时延基本不变。  

---

## 7. 有效性与边界

- 汇总默认仅统计 `evaluation_status=SUCCESS` 的样本。
- 文中时间对比统一使用均摊口径 $T_{\text{equiv}}$；其派生基于 `completion_time`、`llm_planning_time` 与 `total_episode_time`。
- 时间对比采用“规划时间均摊到每步”的口径（同时报告 `total_episode_time`）；在当前这批日志（`num_failed=0`、planner 全 `real_api`）下，可视为公平比较。
- 当前指标属于离线动作代理指标，不等同于真实机器人端到端任务完成率。
- 对当前这批已汇总日志，`num_failed=0`，因此“仅统计 SUCCESS 样本”与“统计全部样本”在数值上几乎等价。
- 对当前这批已汇总日志，`planner_mode_counts` 为 `real_api=20, fallback_mock=0, mock=0`（L1/L3 各组），因此不存在 mock/fallback 混入带来的可比性偏差。

---

## 8. 复现实验命令

```bash
# System1 baseline
python scripts/eval/run_dreamzero_evaluation.py \
  --host localhost --port 8000 \
  --test-sets-dir test_sets_final \
  --method system1_full_strict_h200

# Dual-System
python scripts/eval/run_dualsystem_evaluation.py \
  --host localhost --port 8000 \
  --test-sets-dir test_sets_final \
  --method dualsystem_full_strict_h200_t0_subtask \
  --prompt-mode subtask \
  --planner-temperature 0.0
```
