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

## 2.4 双系统统一建模（DreamZero-Compatible Formulation）

我们将双系统策略表示为“高层语言规划 + 低层条件执行”的统一形式。给定时刻 $t$ 的观测：

$$
x_t = (I_t, s_t, g),
$$

其中 $I_t$ 表示多视角图像观测，$s_t$ 表示机器人本体状态（如关节、夹爪状态），$g$ 表示全局任务描述。System2 首先根据任务与上下文生成一个子任务序列：

$$
z_{1:M} \sim p_\phi(z_{1:M} \mid g, c), \quad 1 \le M \le M_{\max},
$$

其中 $c$ 为 episode 上下文，$M$ 为规划长度。与固定长度规划不同，我们令 $M$ 随任务复杂度自适应变化：

$$
M = \mathrm{clip}(\lfloor \alpha \, \ell(g,c) + \beta \rceil, 1, M_{\max}),
$$

其中 $\ell(g,c)$ 是任务复杂度或时长估计函数。短程任务对应较小的 $M$，长程任务对应较大的 $M$。在本文实验中，$M_{\max} = 5$。

为了兼容 task_token_only、subtask 与 hybrid 三种语言驱动方式，我们定义执行时刻的条件提示为：

$$
u_t^{(m)} = \Psi(g, z_m; \lambda),
$$

其中 $\Psi$ 是提示组合函数，$\lambda$ 控制提示模式：

$$
\Psi(g, z_m; \lambda) =
\begin{cases}
g, & \lambda = \text{task-only} \\
z_m, & \lambda = \text{subtask} \\
[g; z_m], & \lambda = \text{hybrid}
\end{cases}
$$

其中 $[g; z_m]$ 表示将全局任务与当前子任务拼接为联合语言条件。这样，DreamZero 的低层策略建模为：

$$
\pi_\theta(a_t \mid I_t, s_t, u_t^{(m)}),
$$

即在当前视觉观测、本体状态和语言条件下输出动作。

高层 System2 不直接输出动作，而是为每个时刻分配一个子任务权重。统一写为：

$$
\pi(a_t \mid I_t, s_t, g) = \sum_{m=1}^{M} q_\phi(m \mid t, g, c) \, \pi_\theta(a_t \mid I_t, s_t, \Psi(g, z_m; \lambda)).
$$

其中，$q_\phi(m \mid t, g, c)$ 为高层子任务分配分布。为兼容当前实现中的硬切分与未来可微扩展，我们采用时间软分配形式：

$$
q_\phi(m \mid t, g, c) =
\frac{\exp\left(-\frac{|t/T - \mu_m|}{\tau(g,c)}\right)}
{\sum_{j=1}^{M} \exp\left(-\frac{|t/T - \mu_j|}{\tau(g,c)}\right)}.
$$

其中 $T$ 为 episode 长度，$\mu_m$ 为第 $m$ 个子任务对应的归一化时间中心，$\tau(g,c)$ 为与任务复杂度相关的温度参数。

- 对短程任务，$\tau$ 较小，分配更尖锐，更接近单阶段执行；
- 对长程任务，$\tau$ 较大，切换更平滑，更适合多阶段规划。

当前实现中的硬切分：

$$
m_t = \left\lfloor \frac{tM}{T} \right\rfloor
$$

可视为上述软分配在 $\tau \to 0$ 时的特例。

---

最终，这一统一公式表明：

- System2 通过生成与调度子任务序列来调制语言条件；
- System1（DreamZero）在视觉—状态—语言联合输入下完成具体动作执行；
- subtask 与 hybrid 分别对应局部执行优先与全局-局部联合约束；
- 规划长度 $M$ 与分配温度 $\tau$ 共同实现对短程与长程任务的自适应建模。

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

# 7. Dreamjodo的适配性以及大体量数据集的结果
在 DreamZero 里
Mean L2 指的是：模型预测动作 和 数据集真实动作 之间的平均 L2 距离。模型这一步动作和示范动作差多远

在 DreamDojo 里
Mean L2 指的是：预测视频帧和 GT 视频帧之间的视觉 L2 误差代理，是视觉终态/轨迹对齐误差

Mean Task Progress 指的是：judge 根据最终视觉状态给出的任务完成度，再对所有 episode 取平均

success rate的含义：动作是是否完成，是否合规

## 7.1 Dreamzero
## Dreamzero的DROID 数据集（150）


| Dataset    | Mode             | Mean L2 | Mean Task Progress | Success Rate | Rate of L2 < 0.1 |
|------------|------------------|---------|--------------------|--------------|------------------|
| DRO_L1_150 | description_only | 0.1284  | 0.4913             | 29.3%        | 44.3%            |
| DRO_L1_150 | dual_llm         | 0.1258  | 0.5060             | 32.0%        | 45.1%            |
| DRO_L3_150 | description_only | 0.1262  | 0.4580             | 23.3%        | 45.9%            |
| DRO_L3_150 | dual_llm         | 0.1236  | 0.4793             | 27.3%        | 47.6%            |

## Dreamzero的AgiBot 数据集（150）

| Dataset    | Mode             | Mean L2 | Mean Task Progress | Success Rate | Rate of L2 < 0.1 |
|------------|------------------|---------|--------------------|--------------|------------------|
| Agi_L1_150 | description_only | 0.0862  | 0.5160             | 38.7%        | 68.7%            |
| Agi_L1_150 | dual_llm         | 0.0773  | 0.5053             | 36.7%        | 74.2%            |
| Agi_L3_150 | description_only | 0.0739  | 0.4460             | 30.0%        | 79.8%            |
| Agi_L3_150 | dual_llm         | 0.0602  | 0.5187             | 41.3%        | 92.9%            |

## dreamzero的显著性检验
| Dataset    | t value | p value  | Mean L2 diff (baseline - dual) | 95% CI            |
|------------|---------|----------|---------------------------------|-------------------|
| DRO_L1_150 | 2.1213  | 0.0356   | 0.00255                         | [0.00017, 0.00492] |
| DRO_L3_150 | 2.0120  | 0.0460   | 0.00259                         | [0.00005, 0.00514] |
| Agi_L1_150 | 6.1530  | 6.69e-09 | 0.00884                         | [0.00600, 0.01168] |
| Agi_L3_150 | 9.9497  | 3.36e-18 | 0.01368                         | [0.01096, 0.01640] |

## 7.2 Dreamjodo
## Dreamjodo 上面的AgiBot 数据集（130）

| Dataset    | Mode            | Mean L2 | Mean Task Progress | Success Rate | Rate of L2 < 0.1 |
|------------|-----------------|---------|--------------------|--------------|------------------|
| Agi_L1_130 | description_only | 0.1301  | 0.5992             | 56.9%        | 31.3%            |
| Agi_L1_130 | dual_llm        | 0.1278  | 0.6085             | 61.5%        | 32.1%            |
| Agi_L3_130 | description_only | 0.1485  | 0.2892             | 19.2%        | 40.4%            |
| Agi_L3_130 | dual_llm        | 0.1380  | 0.3254             | 24.6%        | 43.6%            |


## DreamDojo 上面的EgoDex数据集（130）


| Dataset    | Mode            | Mean L2 | Mean Task Progress | Success Rate | Rate of L2 < 0.1 |
|------------|-----------------|---------|--------------------|--------------|------------------|
| Ego_L1_130 | description_only | 0.1077  | 0.4171             | 54.1%        |   42.3%         |
| Ego_L1_130 | dual_llm        | 0.1003  | 0.4388             | 55.5%        |   45.8%          |
| Ego_L3_130 | description_only | 0.1214  | 0.3060             | 21.7%        |   37.9%         |
| Ego_L3_130 | dual_llm        | 0.1180  | 0.3798             | 25.4%        |  41.2%

## DreamDoJo的显著性检验
| Dataset    | Mean L2 diff (description_only - dual) | t value | p value (paired t-test) | 95% CI             |
|------------|----------------------------------------|---------|--------------------------|--------------------|
| Agi_L1_130 | 0.00223                                | 3.4672  | 7.15e-04                 | [0.00097, 0.00355] |
| Agi_L3_130 | 0.01055                                | 4.5296  | 1.33e-05                 | [0.00594, 0.01516] | 
| Ego_L1_130 | 0.00741                                | 3.1387  | 3.14e-03                 | [0.00272, 0.01211] |
| Ego_L3_130 | 0.00342                                | 3.8634  | 7.34e-05                 | [0.00171, 0.00520] | 

## 8. 有效性与边界

- 汇总默认仅统计 `evaluation_status=SUCCESS` 的样本。
- 文中时间对比统一使用均摊口径 $T_{\text{equiv}}$；其派生基于 `completion_time`、`llm_planning_time` 与 `total_episode_time`。
- 时间对比采用“规划时间均摊到每步”的口径（同时报告 `total_episode_time`）；在当前这批日志（`num_failed=0`、planner 全 `real_api`）下，可视为公平比较。
- 当前指标属于离线动作代理指标，不等同于真实机器人端到端任务完成率。
- 对当前这批已汇总日志，`num_failed=0`，因此“仅统计 SUCCESS 样本”与“统计全部样本”在数值上几乎等价。
- 对当前这批已汇总日志，`planner_mode_counts` 为 `real_api=20, fallback_mock=0, mock=0`（L1/L3 各组），因此不存在 mock/fallback 混入带来的可比性偏差。

---

## 9. 复现实验命令

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
