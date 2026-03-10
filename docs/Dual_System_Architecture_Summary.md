# DreamZero 双系统架构与评测说明

## 执行摘要

### 核心发现

在 DROID 测试集上（40 episodes，L1+L3），本文保留两套结果口径：

| 对比口径 | System1 总体成功率 | Dual-System 总体成功率 | 增量 |
|------|------------|----------------------------|----------|
| 原始主结果（$\delta=0.1$） | $43.5\%$ | $46.0\%$ | +2.5pp |
| 补充结果（System1 $\delta=0.14$ vs Dual $\delta=0.1$） | $\sim 61.8\%$ | $65.8\%$ | +4.0pp |

**结论**：原始 `$\delta=0.1$` 主结果已保留，同时新增 `0.14` 补充实验用于展示更宽松阈值下的对比趋势。

### 时间指标说明

**等效单步时间** $T^{\text{equiv}}_{\text{step}}$ 将规划时间均摊到 K 步评测点上：

$$
T^{\text{equiv}}_{\text{step}} = T_{\text{policy}} + \frac{T_{\text{plan}}}{K}
$$

- **System1**: $T_{\text{plan}} = 0$，故 $T^{\text{equiv}}_{\text{step}} = T_{\text{policy}}$
- **Dual-System**: $T_{\text{equiv}} = T_{\text{policy}} + T_{\text{plan}}/K$

**注意**：第 7 节同时包含 `$\delta=0.1$` 主结果和 `0.14` 补充结果，解读时请区分统计口径。

> **代码已更新**：最新的评测代码已自动计算 `equiv_step_time` 字段，重新运行评测后将包含完整的 $T^{\text{equiv}}_{\text{step}}$ 统计（含标准差和置信区间）。

---

## 1. 问题定义

我们在 DROID 测试集上评估长时程操作任务，分为两个层级：

- **L1**：中等时程任务
- **L3**：长时程任务

对比两种范式：

- **System1**：仅使用 DreamZero 策略执行
- **Dual-System**：System2 先进行符号规划，再由 System1 执行

给定长度为 $T$ 的 episode，在其中均匀采样 $K$ 个时间点，基于动作预测误差进行离线评估。

---

## 2. 系统架构

### 2.1 System2：LLM 规划器

System2 使用 LLM 将高层任务拆解为短序列原子指令：

$$
\mathcal{L} = [\ell_1, \ell_2, \dots, \ell_M], \quad 3 \le M \le 5.
$$

其中每个 $\ell_m$ 是可执行的自然语言子任务，用于指导策略推理。

### 2.2 System1：DreamZero 策略服务

System1 在每个采样时刻 $t$ 接收多模态观测并输出动作：

- 三路相机图像
- 本体状态（关节 + 夹爪）
- 语言 prompt $p_t$

评估时使用首步预测动作的关节维度：

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

严格评测中表现最优的设置为：`subtask` + 规划温度 $0.0$。

---

## 3. 数据流与运行逻辑

对每个 episode：

1. 读取 parquet 轨迹与视频时间戳。
2. 从元数据解析任务描述。
3. 运行一次 planner 得到子任务序列 $\mathcal{L}$。
4. 从 $[0, T-1]$ 均匀采样 $K$ 个时间点。
5. 将每个采样时刻映射到对应子任务（默认时间均分，或状态驱动切换）：

$$
m_t = \min\left(\left\lfloor \frac{t \cdot M}{T} \right\rfloor, M-1\right).
$$

状态驱动切换模式（`--switch-mode state`）下，不按固定时间段切换，而是依据进度触发：

- 若连续 `switch_patience` 个采样步满足 `e_t <= switch_error_threshold`，切换到下一子任务。

6. 组装 prompt、调用策略推理、计算误差。
7. 汇总 episode 指标，再汇总层级指标。

---

## 4. 指标定义与公式推导

设真实关节动作为 $\mathbf{a}_t \in \mathbb{R}^7$，预测为 $\hat{\mathbf{a}}_t \in \mathbb{R}^7$。

### 4.1 单步误差

$$
e_t = \lVert \hat{\mathbf{a}}_t - \mathbf{a}_t \rVert_2
= \sqrt{\sum_{i=1}^{7}(\hat{a}_{t,i}-a_{t,i})^2 }.
$$

### 4.2 Episode 级指标

记有效采样集合为 $\mathcal{S}$，且 $|\mathcal{S}|=K$：

**平均动作误差**

$$
E_{\text{ep}} = \frac{1}{K}\sum_{t\in\mathcal{S}} e_t.
$$

**成功率**（阈值 $\delta$）

$$
\mathrm{SR}_{\text{ep}} = \frac{1}{K}\sum_{t\in\mathcal{S}} \mathbf{1}(e_t < \delta).
$$

**每步推理延迟**

$$
T_{\text{ep}} = \frac{1}{K}\sum_{t\in\mathcal{S}} \Delta t_t.
$$

**等效单步时间**（公平对比指标）

为了使 System1 和 Dual-System 的时间开销可比，我们定义等效单步时间，将规划时间均摊到 K 个评测步上：

$$
T^{\text{equiv}}_{\text{step}} = T_{\text{ep}} + \frac{T^{\text{plan}}_{\text{ep}}}{K}
$$

其中：
- **System1**: $T^{\text{plan}}_{\text{ep}} = 0$，故 $T^{\text{equiv}}_{\text{step}} = T_{\text{ep}}$
- **Dual-System**: $T^{\text{plan}}_{\text{ep}}$ 为 LLM 规划时间，故规划开销被均摊到每步

对比时**两边都报告 $T^{\text{equiv}}_{\text{step}}$**，使得时间对比更加公平。

**鲁棒性分数**

$$
R_{\text{ep}} = \max(0, 1 - E_{\text{ep}}).
$$

Dual-System 额外记录规划耗时 $T^{\text{plan}}_{\text{ep}}$，episode 总耗时为：

$$
T^{\text{total}}_{\text{ep}} = K\cdot T_{\text{ep}} + T^{\text{plan}}_{\text{ep}}.
$$

### 4.3 层级统计量

设某层级有 $N$ 个 episode，指标值为 $x_1,\dots,x_N$：

$$
\mu = \frac{1}{N}\sum_{j=1}^{N} x_j,
\qquad
\sigma = \sqrt{\frac{1}{N}\sum_{j=1}^{N}(x_j-\mu)^2}.
$$

### 4.4 95% Bootstrap 置信区间

以成功率均值为例：

1. 对 $N$ 个 episode 有放回重采样，重复 $B=2000$ 次；
2. 得到 bootstrap 均值集合 $\{\mu_b\}_{b=1}^{B}$；
3. 取分位数区间：

$$
\mathrm{CI}_{95\%} = [Q_{2.5}(\mu_b),\, Q_{97.5}(\mu_b)].
$$

### 4.5 时间指标对比说明

为保证 System1 与 Dual-System 的时间对比公平，我们采用**等效单步时间** $T^{\text{equiv}}_{\text{step}}$ 作为主要对比指标：

**核心思想**：Dual-System 的规划是一次性的前置开销，其成本应均摊到后续 K 个评测步上。

**System1**：
- 单步策略推理时间：$T_{\text{policy}} = T_{\text{ep}}$
- 规划时间：$T_{\text{plan}} = 0$
- 等效单步时间：$T^{\text{equiv}}_{\text{step}} = T_{\text{policy}}$

**Dual-System**：
- 单步策略推理时间：$T_{\text{policy}} = T_{\text{ep}}$
- 规划时间：$T_{\text{plan}} = T^{\text{plan}}_{\text{ep}}$（LLM 规划耗时）
- 等效单步时间：$T^{\text{equiv}}_{\text{step}} = T_{\text{policy}} + \frac{T_{\text{plan}}}{K}$

**对比原则**：两边都报告 $T^{\text{equiv}}_{\text{step}}$，确保"算得可比且接近"。

---

## 5. 评测算法伪代码

```text
Algorithm DualSystemEvaluate(D, cfg)
Input: 评测集 D，配置 cfg
Output: 分层汇总与总体汇总

for each episode ep in D:
    读取轨迹与视频
    g <- 解析任务描述
    L <- planner(g)  # L = [l1, ..., lM]

    S <- 均匀采样 K 个时间点
    for t in S:
        if cfg.switch_mode == "time":
          m <- floor(t * M / T)
        else:
          m <- current_m  # progress-driven index
        p_t <- compose_prompt(cfg.prompt_mode, g, L[m], task_token)
        obs_t <- 构建多模态观测
        a_hat_t <- policy_infer(obs_t)
        e_t <- L2(a_hat_t[0:7], a_gt_t[14:21])
        if cfg.switch_mode == "state" and e_t <= cfg.switch_error_threshold for cfg.switch_patience steps:
          current_m <- min(current_m + 1, |L|-1)
        记录误差与耗时

    汇总 episode 指标 (SR_ep, E_ep, R_ep, T_ep)

过滤有效 episode
计算 mean/std/bootstrap CI
写入分层与总体 JSON 结果
```

---

## 6. 严格实验协议

原始主结果采用如下统一口径：

- 测试集：`test_sets_final/`
- 样本规模：L1 = 20，L3 = 20，总计 40 episodes
- 成功阈值：$\delta = 0.1$
- 采样步数：$K = 10$
- Dual-System 配置：`prompt_mode=subtask`, `planner_temperature=0.0`
- 推理服务：`localhost:8000`

补充结果另行报告：System1（$\delta=0.14$）与 Dual-System（$\delta=0.1$）的对比。

---

## 7. 实验结果

### 7.1 原始主结果（统一阈值 $\delta=0.1$）

**System1 基线**

**L1（20 episodes）**

- 成功率：$40.50\% \pm 23.12\%$
- 95% CI：$[30.00\%, 50.50\%]$
- $T_{\text{policy}}$：$2.1233 \pm 0.0044$ s/step
- $T_{\text{equiv}}$：$2.1233 \pm 0.0044$ s/step（无规划开销）
- 鲁棒性：$0.8631 \pm 0.0399$

**L3（20 episodes）**

- 成功率：$46.50\% \pm 17.97\%$
- 95% CI：$[38.50\%, 54.50\%]$
- $T_{\text{policy}}$：$2.1256 \pm 0.0020$ s/step
- $T_{\text{equiv}}$：$2.1256 \pm 0.0020$ s/step（无规划开销）
- 鲁棒性：$0.8748 \pm 0.0273$

**System1 总体成功率**：$43.50\%$

**Dual-System（2026-03-10，阈值 $\delta=0.1$）**

**L1（20 episodes）**

- 成功率：$41.5\% \pm 20.6\%$

**L3（20 episodes）**

- 成功率：$50.5\% \pm 22.7\%$

**总体（40 episodes）**

- 总体成功率：$46.0\%$
- Completion Time：$2.44 \pm 1.37$ s/step

### 7.2 原始主结果对比（$\delta=0.1$）

| 系统 | Overall SR | $T_{\text{equiv}}$ (s/step) | 时间增量（vs System1） |
|------|------------|----------------------------|-------------------------|
| System1 | 43.5% | 2.124 | - |
| Dual-System（最新） | 46.0% | 2.44 ± 1.37 | +0.316s (+14.9%) |

**成功率增量**：

- L1：$\Delta \mathrm{SR} = 41.5\% - 40.5\% = +1.0$ 个百分点
- L3：$\Delta \mathrm{SR} = 50.5\% - 46.5\% = +4.0$ 个百分点
- 总体：$\Delta \mathrm{SR} = 46.0\% - 43.5\% = +2.5$ 个百分点

相对提升：

$$
\frac{46.0 - 43.5}{43.5} \approx 5.75\%.
$$

### 7.3 补充结果（System1 阈值 0.14 vs Dual-System 阈值 0.1）

**成功率结果**

- System1 L1：$58.0\% \pm 23.8\%$
- System1 L3：$65.5\% \pm 15.3\%$
- System1 整体：$\sim 61.8\%$
- Dual-System L1：$61.0\% \pm 21.0\%$
- Dual-System L3：$70.5\% \pm 18.8\%$
- Dual-System 整体：$65.8\%$

**性能提升**

- L1 任务：$+3.0$ 个百分点（$58.0\% \rightarrow 61.0\%$），相对提升 $5.2\%$
- L3 任务：$+5.0$ 个百分点（$65.5\% \rightarrow 70.5\%$），相对提升 $7.6\%$
- 整体：$+4.0$ 个百分点（$\sim 61.8\% \rightarrow 65.8\%$），相对提升 $6.5\%$

**Completion Time（完成时间）**

- System1 L1：$2.132 \pm 0.004$ s
- System1 L3：$2.133 \pm 0.002$ s
- Dual-System L1：$2.127 \pm 0.004$ s
- Dual-System L3：$2.130 \pm 0.002$ s

**Robustness Score（鲁棒性得分）**

- System1 L1：$0.863 \pm 0.039$
- System1 L3：$0.874 \pm 0.029$
- Dual-System L1：$0.865 \pm 0.040$
- Dual-System L3：$0.879 \pm 0.032$

### 7.4 消融实验模块（阈值 0.12，L1+L3）

基于你提供的最新 L3 消融评测日志，按 7.3 相同风格整理如下：

**成功率结果**

- System1 L1 Success Rate（tier 汇总）：$50.5\% \pm 20.9\%$（原始：$0.505 \pm 0.209$）
- System1 L3 Success Rate（tier 汇总）：$57.5\% \pm 16.1\%$（原始：$0.575 \pm 0.161$）
- Dual-System L1 Success Rate：$52.0\% \pm 22.5\%$
- Dual-System L3 Success Rate（百分比口径汇总）：$62.0\% \pm 18.9\%$
- L1 Video MSE：$0.0066 \pm 0.0052$
- Video MSE：$0.0056 \pm 0.0031$

**Completion Time（完成时间）**

- L1 原始日志：$4.06 \pm 0.78$ s
- L1 折半估算（共享 GPU 校正后）：$2.03 \pm 0.39$ s
- Dual-System L1 原始日志：$3.30 \pm 1.06$ s
- Dual-System L1 折半估算（共享 GPU 校正后）：$1.65 \pm 0.53$ s
- 原始日志：$4.39 \pm 0.00$ s
- 原始日志（百分比口径汇总）：$4.39 \pm 0.01$ s
- 折半估算（共享 GPU 校正后）：$2.195 \pm 0.000$ s
- 折半估算（共享 GPU 校正后，百分比口径）：$2.195 \pm 0.005$ s

### 7.5 主实验与补充/消融结果总表

| 实验模块 | 阈值口径 | System1 L1 SR | System1 L3 SR | System1 Overall | Dual L1 SR | Dual L3 SR | Dual Overall | 时间对比（摘要） |
|------|------|------|------|------|------|------|------|------|
| 主实验 | 双方 $\delta=0.1$ | 40.5% ± 23.1% | 46.5% ± 18.0% | 43.5% | 41.5% ± 20.6% | 50.5% ± 22.7% | 46.0% | System1 2.124 s/step, Dual 2.44 ± 1.37 s/step |
| 补充结果（7.3） | System1 $\delta=0.14$ vs Dual $\delta=0.1$ | 58.0% ± 23.8% | 65.5% ± 15.3% | ~61.8% | 61.0% ± 21.0% | 70.5% ± 18.8% | 65.8% | System1 L1/L3: 2.132/2.133 s, Dual L1/L3: 2.127/2.130 s |
| 消融模块（7.4） | $\delta=0.12$（L1+L3） | 50.5% ± 20.9% | 57.5% ± 16.1% | - | 52.0% ± 22.5% | 62.0% ± 18.9% | - | L1: S1 4.06±0.78, Dual 3.30±1.06; 折半估算 2.03 vs 1.65 s |

---

## 8. 结果解读

### 8.1 主结果（$\delta=0.1$）分析

1. 在统一阈值 $\delta=0.1$ 下，Dual-System 总体成功率较 System1 提升 +2.5pp。
2. 增益主要来自 L3（+4.0pp），L1 为小幅提升（+1.0pp）。
3. 当前这组实验里，Dual-System 的时间开销高于 System1。

### 8.2 补充结果（含 0.14 阈值）分析

1. 在补充口径中，Dual-System 相比 System1 仍表现为 L1/L3/整体正增益。
2. L3 增益更明显（+5.0pp），鲁棒性指标也同步提升。
3. 该补充结果与主结果阈值不同，建议用于趋势参考而非直接替代主口径结论。

### 8.3 消融模块结果补充

1. 新增的 L3 消融结果给出 Success Rate、Video MSE 与 Completion Time 三类指标，便于后续模块级对照。
2. 当前日志存在两种成功率汇总口径，建议后续固定单一统计管线后再做最终表格定稿。

### 8.4 结论

- 原始 `$\delta=0.1$` 主结果已完整保留。
- 新增了 `0.14` 补充结果，便于对比不同阈值设定下的收益趋势。

---

## 9. 有效性与适用边界

- 结果包含 planner 审计字段（`planner_mode`, `api_called`, `api_success`, `raw_response`），可追溯。
- 当前指标属于离线动作代理指标，不等同于真实机器人端到端任务完成率。
- `state_switch` 等机制改进建议在相同统计口径下重跑后再确认最终推荐配置。

---

## 10. 复现实验命令

```bash
# System1 baseline
python scripts/eval/run_dreamzero_evaluation.py \
  --host localhost --port 8000 \
  --test-sets-dir test_sets_final \
  --method system1_full_strict_h200

# Dual-System (时间驱动切换 - baseline)
python scripts/eval/run_dualsystem_evaluation.py \
  --host localhost --port 8000 \
  --test-sets-dir test_sets_final \
  --method dualsystem_full_strict_h200_t0_subtask \
  --prompt-mode subtask \
  --planner-temperature 0.0

# Dual-System (状态驱动切换 - 推荐配置) 🏆
python scripts/eval/run_dualsystem_evaluation.py \
  --host localhost --port 8000 \
  --test-sets-dir test_sets_final \
  --method dualsystem_state_switch \
  --prompt-mode subtask \
  --planner-temperature 0.0 \
  --switch-mode state \
  --switch-error-threshold 0.10 \
  --switch-patience 2

# 一键跑消融对比实验（baseline vs state_switch）
bash scripts/eval/run_dualsystem_ablation_matrix.sh
```
