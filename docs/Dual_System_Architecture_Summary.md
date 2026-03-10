# DreamZero 双系统架构与评测说明

## 执行摘要

### 核心发现

在 DROID 测试集上（40 episodes，L1+L3），本文保留两套结果口径：

| 对比口径 | System1 总体成功率 | Dual-System 总体成功率 | 增量 |
| --- | --- | --- | --- |
| 原始主结果（$\delta=0.1$） | $43.5\%$ | $46.0\%$ | +2.5pp |
| 补充结果（System1 $\delta=0.14$ vs Dual $\delta=0.1$） | $\sim 61.8\%$ | $65.8\%$ | +4.0pp |

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

说明：脚本默认 `prompt_mode=task_token_hybrid`；严格实验命令可显式设置 `--prompt-mode subtask`。

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

---

## 5. 严格实验协议

原始主结果使用如下口径：

- 测试集：`test_sets_final/`
- 样本规模：L1=20，L3=20，总计 40 episodes
- 成功阈值：$\delta=0.1$
- 采样步数：$K=10$
- Dual-System 复现实验命令配置：`prompt_mode=subtask`, `planner_temperature=0.0`
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
- Completion Time（$T_{\text{equiv}}$）：$2.44 \pm 1.37$ s/step

### 6.2 主结果对比（$\delta=0.1$）

| 系统 | Overall SR | Completion Time (s/step, $T_{\text{equiv}}$) | 时间增量（vs System1） |
| --- | --- | --- | --- |
| System1 | 43.5% | 2.124 | - |
| Dual-System | 46.0% | 2.44 ± 1.37 | +0.316s (+14.9%) |

### 6.3 主实验与对比实验总表

| 实验模块 | 阈值口径 | System1 L1 SR | System1 L3 SR | System1 Overall | Dual L1 SR | Dual L3 SR | Dual Overall | 时间对比（摘要） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 主实验 | 双方 $\delta=0.1$ | 40.5% ± 23.1% | 46.5% ± 18.0% | 43.5% | 41.5% ± 20.6% | 50.5% ± 22.7% | 46.0% | System1 2.124 s/step, Dual 2.44 ± 1.37 s/step（$T_{\text{equiv}}$） |
| 对比实验 | $\delta=0.12$（L1+L3） | 50.5% ± 20.9% | 57.5% ± 16.1% | 54% | 52.0% ± 22.5% | 62.0% ± 18.9% | 57% | System1 L1/L3: 2.137/2.130 s, Dual L1/L3: 2.147/2.135 s（$T_{\text{equiv}}$） |
| 对比实验 | System1 $\delta=0.14$ vs Dual $\delta=0.1$ | 58.0% ± 23.8% | 65.5% ± 15.3% | 61.8% | 61.0% ± 21.0% | 70.5% ± 18.8% | 65.8% | System1 L1/L3: 2.132/2.133 s, Dual L1/L3: 2.127/2.130 s（$T_{\text{equiv}}$） |

### 6.4 L3 公平对比（B1/B2 + Dual，$\delta=0.1$）

时间统一采用可比口径：

$$
T_{\text{equiv}} = T_{\text{policy}} + \frac{T_{\text{planning}}}{K}, \quad K=10.
$$

其中：
- System1/B1/B2 不使用 LLM 规划，故 $T_{\text{planning}}=0$，$T_{\text{equiv}}=T_{\text{policy}}$；
- Dual-System 使用 LLM 规划，按每 episode 规划时间均摊到每步；
- B1/B2 由于与他人共享同一张卡，按你的实验说明对原始 `completion_time` 做折半校正后再参与对比。

| 组别 | 含义 | Success Rate | 95% CI | Completion Time（统一口径） | Robustness Score |
| --- | --- | --- | --- | --- | --- |
| System1 旧基线 | task token only | 46.5% | [38.5%, 54.5%] | 2.126 s/step | 0.8748 ± 0.0273 |
| B1 | task_description，无子任务，无 LLM | 44.0% | [35.5%, 53.0%] | 2.207 s/step（原始 4.414 折半） | 0.8770 ± 0.0310 |
| B2 | task_description + heuristic/chunked subtask，不走 LLM | 46.5% | [37.99%, 56.0%] | 2.206 s/step（原始 4.411 折半） | 0.8745 ± 0.0313 |
| 现有 Dual-System 主结果 | LLM subtask | 50.5% | [40.5%, 60.5%] | 2.281 s/step（2.130 policy + 1.513/10 planning） | 0.8769 ± 0.0328 |

### 6.5 Case-6 误差剖析（L3, $\delta=0.1$）

#### 汇总对比（6 episodes）

| 组别 | 含义 | Success Rate | 95% CI | Completion Time（统一口径） | Robustness Score |
| --- | --- | --- | --- | --- | --- |
| System1-token | task token only | 45.0% | [33.3%, 53.3%] | 2.117 s/step | 0.8577 |
| B2 | task_description + heuristic/chunked subtask（无 LLM） | 40.0% | [28.3%, 50.0%] | 2.118 s/step | 0.8684 |
| Dual-LLM（real API） | LLM subtask | 45.0% | [30.0%, 55.0%] | 2.317 s/step（2.116 policy + 2.007/10 planning） | 0.8646 |

说明：Case-6 上 Dual 的均值收益不是“全面压制”，而是“结构化收益 + 结构化失败”并存。

#### 逐案例模式（你标注的 6 个 episode）

| 模式 | episode | SR（System1 / B2 / Dual） | 可解释性说明 |
| --- | --- | --- | --- |
| Dual 明显更好 | `episode_001470`, `episode_001669` | `0.4/0.2/0.5`, `0.5/0.3/0.6` | 多阶段、多目标或跨对象切换任务中，LLM 子任务对“中间目标链”刻画更完整。 |
| Dual 明显更差 | `episode_001406`, `episode_001395` | `0.2/0.5/0.1`, `0.6/0.6/0.5` | 任务较直接或 B2 分段已足够时，LLM 规划引入额外动作语义（如 open/move）可能干扰策略。 |
| 基本持平/小幅更好 | `episode_001754`, `episode_001827` | `0.5/0.4/0.5`, `0.5/0.4/0.5` | 规划增益有限，表现受策略本体误差主导。 |

#### 可解释性结论

- Dual-LLM 在复杂长链任务上可通过显式中间目标提升成功率，但在简单/短链任务上可能因过度规划导致收益消失甚至负迁移。  
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
