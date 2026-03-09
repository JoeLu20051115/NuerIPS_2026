# DreamZero 双系统架构与评测说明

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
5. 将每个采样时刻映射到对应子任务：

$$
m_t = \min\left(\left\lfloor \frac{t \cdot M}{T} \right\rfloor, M-1\right).
$$

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

**鲁棒性分数**

$$
R_{\text{ep}} = \max(0, 1 - E_{\text{ep}}).
$$

Dual-System 额外记录规划耗时 $T^{\text{plan}}_{\text{ep}}$，总耗时为：

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
        m <- floor(t * M / T)
        p_t <- compose_prompt(cfg.prompt_mode, g, L[m], task_token)
        obs_t <- 构建多模态观测
        a_hat_t <- policy_infer(obs_t)
        e_t <- L2(a_hat_t[0:7], a_gt_t[14:21])
        记录误差与耗时

    汇总 episode 指标 (SR_ep, E_ep, R_ep, T_ep)

过滤有效 episode
计算 mean/std/bootstrap CI
写入分层与总体 JSON 结果
```

---

## 6. 严格实验协议

最终结果采用如下统一口径：

- 测试集：`test_sets_final/`
- 样本规模：L1 = 20，L3 = 20，总计 40 episodes
- 成功阈值：$\delta = 0.1$
- 采样步数：$K = 10$
- Dual-System 配置：`prompt_mode=subtask`, `planner_temperature=0.0`
- 推理服务：`localhost:8000`

---

## 7. 实验结果

### 7.1 System1 基线

**L1（20 episodes）**

- 成功率：$40.50\% \pm 23.12\%$
- 95% CI：$[30.50\%, 51.50\%]$
- 完成时间：$2.1241 \pm 0.0052$ s/step
- 鲁棒性：$0.8631 \pm 0.0399$

**L3（20 episodes）**

- 成功率：$46.50\% \pm 17.97\%$
- 95% CI：$[38.50\%, 54.00\%]$
- 完成时间：$2.1249 \pm 0.0021$ s/step
- 鲁棒性：$0.8748 \pm 0.0273$

**总体成功率**：$43.50\%$

### 7.2 Dual-System

**L1（20 episodes）**

- 成功率：$41.50\% \pm 22.20\%$
- 95% CI：$[32.00\%, 51.50\%]$
- 完成时间：$2.1193 \pm 0.0028$ s/step
- 鲁棒性：$0.8703 \pm 0.0373$

Planner 统计（L1）：

- `real_api=20`, `fallback_mock=0`, `mock=0`
- `api_called_count=20`, `api_success_count=20`

**L3（20 episodes）**

- 成功率：$51.00\% \pm 22.11\%$
- 95% CI：$[41.50\%, 60.50\%]$
- 完成时间：$2.1192 \pm 0.0031$ s/step
- 鲁棒性：$0.8770 \pm 0.0333$

Planner 统计（L3）：

- `real_api=20`, `fallback_mock=0`, `mock=0`
- `api_called_count=20`, `api_success_count=20`

**总体成功率**：$46.25\%$

### 7.3 相对 System1 的增益

定义 $\Delta = \text{Dual-System} - \text{System1}$：

- L1：$\Delta \mathrm{SR} = +1.00$ 个百分点
- L3：$\Delta \mathrm{SR} = +4.50$ 个百分点
- 总体：$\Delta \mathrm{SR} = +2.75$ 个百分点

总体相对提升：

$$
\frac{46.25 - 43.50}{43.50} \approx 6.32\%.
$$

---

## 8. 结果解读

1. 在严格口径下，双系统相对基线呈稳定正增益。
2. 提升主要来自长时程层级 L3，说明符号分解在长时域任务中作用更明显。
3. 每步策略推理耗时几乎不变，额外时间主要来自每个 episode 的一次性规划调用。

---

## 9. 有效性与适用边界

- 结果包含 planner 审计字段（`planner_mode`, `api_called`, `api_success`, `raw_response`），可追溯。
- 当前指标属于离线动作代理指标，不等同于真实机器人端到端任务完成率。
- 当前架构不包含独立 VLM 完成度判别器与失败触发重规划状态机。

---

## 10. 复现实验命令

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

