# DreamZero 双系统长程操控架构：System 2 赋能下的 Train-Free 层次化策略

> **核心思想：** 通过引入 LLM 作为 System 2 符号规划器，将不可解的长程操控问题分解为可解的短程子任务序列，由 DreamZero (System 1) 逐一执行，并借助 VLM 实现闭环状态评估与失败恢复。整个过程对 DreamZero 零侵入、无需额外训练。

---

## 一、动机与问题定义

### 1.1 DreamZero 的"短视"瓶颈

DreamZero 作为世界动作模型（World Action Model），其核心能力是在 **单个 Action Chunk**（约 1.6 秒）的时间尺度上，联合生成未来视频与动作序列。在代码实现中，这对应于 `lazy_joint_video_action` 方法中的单次去噪推理循环：

```python
# wan_flow_matching_action_tf.py, line 929
def lazy_joint_video_action(self, backbone_output, action_input, latent_video=None):
    # 单次推理：生成一个 chunk 的 video + action
    ...
```

然而，当面对长程任务（如"整理桌面上所有物品并放入抽屉"），模型需要跨越数十个 chunk 的时间跨度。此时：

- **误差级联（Error Cascading）：** 每个 chunk 的微小偏差会在后续 chunk 中指数放大。
- **语义断裂（Semantic Disconnection）：** 模型无法在底层连续动作空间中进行高层逻辑推理（如"先拿 A 再拿 B"的顺序决策）。
- **单一指令瓶颈：** 代码中 `text` 字段在整个推理过程中保持不变，无法动态调整子目标。

```python
# wan_flow_matching_action_tf.py, line 968-975
# 当前实现：语言指令变化时重置 KV 缓存
if self.language is None:
    self.language = data["text"]
    self.current_start_frame = 0
elif not torch.equal(self.language, data["text"]):
    self.current_start_frame = 0
    self.language = data["text"]
```

**关键发现：** DreamZero 已经内置了语言指令变化检测机制——当 `data["text"]` 发生变化时，模型自动重置 `current_start_frame` 和 KV 缓存。这意味着 **外部系统可以通过动态替换文本指令来切换子任务，无需修改模型内部任何代码**。

### 1.2 认知科学启示：双过程理论

本架构借鉴 Kahneman 的双系统理论：

| | System 1（快思考） | System 2（慢思考） |
|---|---|---|
| **角色** | DreamZero (WAM) | LLM (GPT-4 / Claude) |
| **特征** | 高频、直觉、连续、物理交互 | 低频、符号、离散、逻辑推理 |
| **时间尺度** | ~1.6s (单个 chunk) | ~10-60s (语义阶段) |
| **输入** | 视觉观测 $o_t$ + 原子指令 $l_k$ | 语义状态 $s_t$ + 长程目标 $L_{long}$ |
| **输出** | 动作序列 $a_{t:t+H}$ + 视频 $z_{t:t+H}$ | 子指令序列 $[l_1, l_2, \dots, l_n]$ |

---

## 二、系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Long-horizon Instruction L_long                  │
│                  "Clean the table and put items away"               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ╔══════════════════════════════════════════════════════════════╗    │
│  ║          System 2: LLM Symbolic Planner (Π_sys2)           ║    │
│  ║                                                             ║    │
│  ║  Input:  L_long + s_t + History H_t                        ║    │
│  ║  Output: Atomic sub-instruction l_k                        ║    │
│  ║  Example: l_1="Pick up the red cup"                        ║    │
│  ║           l_2="Place it in the drawer"                     ║    │
│  ║           l_3="Close the drawer"                           ║    │
│  ╚══════════════════════════════╦═══════════════════════════════╝    │
│                                 ║ l_k (text instruction)            │
│                                 ▼                                    │
│  ╔══════════════════════════════════════════════════════════════╗    │
│  ║       System 1: DreamZero WAM (π_sys1) — UNCHANGED         ║    │
│  ║                                                             ║    │
│  ║  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  ║    │
│  ║  │  Text   │  │  Image   │  │   VAE    │  │  Action/   │  ║    │
│  ║  │ Encoder │  │ Encoder  │  │ Encoder  │  │  State     │  ║    │
│  ║  │ (UMT5)  │  │ (CLIP)   │  │ (WAN)    │  │  Encoder   │  ║    │
│  ║  └────┬────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘  ║    │
│  ║       │            │             │               │          ║    │
│  ║       ▼            ▼             ▼               ▼          ║    │
│  ║  ┌──────────────────────────────────────────────────────┐   ║    │
│  ║  │         CausalWanModel (32-layer DiT)                │   ║    │
│  ║  │  ┌──────────────┐ ┌──────────────┐ ┌─────────────┐  │   ║    │
│  ║  │  │ Causal Self- │ │    Cross-    │ │     FFN     │  │   ║    │
│  ║  │  │  Attention   │ │  Attention   │ │             │  │   ║    │
│  ║  │  │ (video+act+  │ │ (text+image  │ │ (nonlinear  │  │   ║    │
│  ║  │  │  state)      │ │  condition)  │ │  transform) │  │   ║    │
│  ║  │  └──────────────┘ └──────────────┘ └─────────────┘  │   ║    │
│  ║  └──────────────────────────────────────────────────────┘   ║    │
│  ║                          │                                  ║    │
│  ║            ┌─────────────┴──────────────┐                   ║    │
│  ║            ▼                            ▼                   ║    │
│  ║     Action a_{t:t+H}          Video z_{t:t+H}              ║    │
│  ╚══════════════╦═══════════════════════════════════════════════╝    │
│                 ║ Execute action in real world                       │
│                 ▼                                                    │
│  ╔══════════════════════════════════════════════════════════════╗    │
│  ║      Evaluator: VLM State Evaluator (E)                    ║    │
│  ║                                                             ║    │
│  ║  Input:  Real observation o_{t+H} + sub-instruction l_k    ║    │
│  ║  Output: Semantic state s_{t+H} + Success flag b_k         ║    │
│  ║                                                             ║    │
│  ║  b_k = 1 → Next sub-task l_{k+1}                          ║    │
│  ║  b_k = 0 → Re-plan with error message                     ║    │
│  ╚══════════════════════════════════════════════════════════════╝    │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 三大核心模块

#### 模块 A：System 2 — LLM 符号化长程规划器 ($\Pi_{sys2}$)

- **实现：** GPT-4o / Claude / Qwen 等大语言模型，通过 API 调用。
- **职责：** 接收长程指令 $L_{long}$ 和当前语义状态 $s_t$，输出原子化子指令 $l_k$。
- **关键特性：** Train-Free，利用 LLM 的先验知识进行零样本规划。

#### 模块 B：System 1 — DreamZero 世界动作模型 ($\pi_{sys1}$)

- **实现：** 原始 DreamZero 模型，**完全不修改**。
- **职责：** 在原子指令 $l_k$ 的条件下，生成 chunk 级别的动作和视频。
- **关键接口：** `lazy_joint_forward_causal(batch)` — 只需替换 `batch` 中的 `text` 字段即可切换子任务。

#### 模块 C：Evaluator — VLM 状态转移反馈器 ($\mathcal{E}$)

- **实现：** 多模态大模型（如 GPT-4o-Vision / Qwen-VL）。
- **职责：** 将真实观测 $o_{t+H}$ 映射为语义状态 $s_{t+H}$，判断子任务是否完成。
- **关键特性：** 充当物理世界与符号世界之间的"翻译官"。

---

## 三、数学建模

### 3.1 长程策略分解

原始 DreamZero 试图直接拟合长程联合分布：

$$P(a_{0:T}, o_{0:T} \mid o_0, L_{long})$$

当 $T$ 极大时，这个分布的建模是不可行的。我们利用贝叶斯定理和马尔可夫假设，将时间轴切分为 $K$ 个语义阶段，得到层次化分解：

$$P(\tau_{0:T} \mid o_0, L_{long}) \approx \prod_{k=1}^{K} \underbrace{P(l_k \mid s_{t_{k-1}}, L_{long}, \mathcal{H}_{t_{k-1}})}_{\text{System 2: 符号规划}} \cdot \underbrace{P(\tau_{t_{k-1}:t_k} \mid o_{t_{k-1}}, l_k)}_{\text{System 1: 连续控制}}$$

其中 $\tau_{t_{k-1}:t_k} = (a_{t_{k-1}:t_k}, o_{t_{k-1}:t_k})$ 为阶段内的连续轨迹，$\mathcal{H}_t$ 为历史状态转移记录。

### 3.2 System 2 的自回归规划

LLM 的规划过程是一个离散的条件生成：

$$l_k = \Pi_{sys2}(L_{long}, s_t, \mathcal{H}_t) = \arg\max_{l} P_{LLM}(l \mid s_t, L_{long}, \mathcal{H}_t)$$

通过精巧的 Prompt Engineering，将 LLM 的先验概率直接转化为后验输出，实现 **Train-Free** 的零样本规划。

### 3.3 System 1 的连续去噪生成

DreamZero 的训练目标（Flow Matching）保持不变，但推理时条件文本被替换为 System 2 下发的 $l_k$：

$$\mathcal{L}_{WAM}(\theta) = \mathbb{E}_{x, t} \left[ \left\| u_\theta\left([z_{t_{noise}}, a_{t_{noise}}]; \mathcal{C}_{clean}, \mathbf{l_k}, q\right) - v_{target} \right\|^2 \right]$$

其中：
- $u_\theta$：CausalWanModel（32 层 DiT）
- $\mathcal{C}_{clean}$：干净的历史视频上下文帧
- $l_k$：System 2 下发的原子指令（通过 UMT5 编码后经交叉注意力注入）
- $q$：机器人本体感知状态（关节位置、夹爪状态等，通过 `CategorySpecificMLP` 编码）
- $v_{target}$：Flow Matching 的速度目标

### 3.4 VLM 状态评估与重规划

定义状态映射 $\mathcal{E}_{VLM} : \mathcal{O} \rightarrow \mathcal{S}$，将高维像素映射为低维语义描述。定义指示函数判定子任务后置条件：

$$b_k = \mathbb{I}\left(\mathcal{E}_{VLM}(o_{t+H}) \models \text{PostCond}(l_k)\right)$$

状态机转移规则：
- **$b_k = 1$（成功）：** $\mathcal{H}_{t+H} = \mathcal{H}_t \cup \{(s_{t+H}, l_k)\}$，弹出下一个指令 $l_{k+1}$
- **$b_k = 0$（失败）：** 触发重规划 $l_k' = \Pi_{sys2}(L_{long}, s_{t+H}, \text{ErrorMsg})$

---

## 四、与 DreamZero 代码的对接分析

### 4.1 零侵入可行性论证

**核心论点：** System 2 的接入不需要修改 DreamZero 的任何模型代码，只需在外部推理循环中动态替换文本指令。

**代码证据 1：语言变化自动重置机制**

DreamZero 已经内置了对语言指令变化的感知：

```python
# wan_flow_matching_action_tf.py, line 968-975
if self.language is None:
    self.language = data["text"]
    self.current_start_frame = 0
elif not torch.equal(self.language, data["text"]):
    # 当文本指令发生变化时，自动重置 KV 缓存
    self.current_start_frame = 0
    self.language = data["text"]
```

这段代码说明：当外部传入的 `data["text"]` 发生变化时，模型会：
1. 重置 `current_start_frame = 0`，清空 KV 缓存
2. 重新编码新的文本指令
3. 重新编码当前观测的 CLIP 特征

这正是 System 2 切换子任务时所需要的行为。

**代码证据 2：推理接口的简洁性**

```python
# socket_test_optimized_AR.py, line 278-284
batch = Batch(obs=converted_obs)
with torch.no_grad():
    result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
```

外部只需构造 `batch.obs`（包含图像和文本），即可调用推理。System 2 只需在每次调用前替换 `batch.obs` 中的语言字段。

**代码证据 3：交叉注意力的条件注入**

```python
# wan_video_dit_action_casual_chunk.py, line 1744-1749
context = self.text_embedding(context)  # UMT5 编码文本指令
if clip_feature is not None:
    clip_embedding = self.img_emb(clip_feature)  # CLIP 编码图像
    context = torch.cat([clip_embedding, context], dim=1)
```

文本指令通过 `text_embedding` 编码后，与图像条件拼接，通过每一层的交叉注意力注入。替换文本指令等价于改变所有层的交叉注意力条件，从而改变模型的生成目标。

### 4.2 System 1 内部的多模态处理路径

在 DreamZero 的 DiT 内部，不同模态的信息通过不同路径处理：

| 模态 | 编码器 | 注入方式 | 在 DiT 中的角色 |
|------|--------|---------|----------------|
| **视频帧** | WAN VAE → Patch Embedding | 因果自注意力（主序列） | 被去噪的主要目标之一 |
| **动作** | `MultiEmbodimentActionEncoder` | 因果自注意力（与视频拼接） | 被去噪的主要目标之一 |
| **机器人状态** | `CategorySpecificMLP` | 因果自注意力（与动作拼接） | 提供本体感知上下文 |
| **文本指令** | UMT5 Text Encoder | 交叉注意力（每层共享） | 语义条件引导 |
| **首帧图像** | CLIP Image Encoder | 交叉注意力（与文本拼接） | 视觉条件引导 |
| **时间步** | Sinusoidal → MLP | Adaptive LayerNorm | 扩散过程调制 |

**关键洞察：** 文本指令和首帧图像通过交叉注意力注入，是 DiT 的 **外部条件**。替换它们不会影响模型的内部权重和因果自注意力结构，这是零侵入的理论基础。

---

## 五、算法伪代码

```
Algorithm: Dual-System Long-Horizon Manipulation with DreamZero

Input: Long-horizon instruction L_long, Initial observation o_0
Output: Complete trajectory τ_{0:T}

1:  Initialize: H_0 ← ∅, k ← 0
2:  s_0 ← E_VLM(o_0)                          // VLM extracts initial semantic state
3:  [l_1, ..., l_n] ← Π_sys2(L_long, s_0)     // LLM generates sub-instruction sequence
4:  
5:  while k < n do
6:      k ← k + 1
7:      t ← current_timestep
8:      
9:      // ===== System 1: DreamZero Closed-loop Execution =====
10:     Reset DreamZero KV cache (triggered by text change)
11:     Pack l_k into batch.obs["text"]
12:     
13:     repeat  // Inner loop: chunk-wise execution
14:         o_t ← capture_real_observation()
15:         Pack o_t into batch.obs["video"]
16:         
17:         // DreamZero inference (UNCHANGED model)
18:         a_{t:t+H}, z_{t:t+H} ← π_sys1.lazy_joint_forward_causal(batch)
19:         
20:         Execute a_{t:t+H} on robot (or partial execution with receding horizon)
21:         t ← t + H
22:     until VLM_check_interval reached
23:     
24:     // ===== Evaluator: VLM State Assessment =====
25:     o_{t} ← capture_real_observation()
26:     s_t ← E_VLM(o_t)
27:     b_k ← I(s_t ⊨ PostCond(l_k))
28:     
29:     if b_k = 1 then                         // Sub-task succeeded
30:         H_t ← H_t ∪ {(s_t, l_k)}
31:         continue to next sub-instruction
32:     else                                     // Sub-task failed
33:         error_msg ← describe_failure(s_t, l_k)
34:         [l_k', ..., l_m'] ← Π_sys2(L_long, s_t, H_t, error_msg)  // Re-plan
35:         Update remaining sub-instructions
36:     end if
37: end while
38:
39: return Complete trajectory τ_{0:T}
```

---

## 六、可行性分析与核心优势

### 6.1 技术可行性

| 维度 | 分析 | 可行性 |
|------|------|--------|
| **接口兼容性** | DreamZero 通过 `data["text"]` 接收文本指令，替换即可 | ✅ 完全兼容 |
| **KV 缓存重置** | 语言变化时自动重置，无需手动干预 | ✅ 已内置 |
| **图像条件更新** | `current_start_frame=0` 时自动重新编码 CLIP 特征 | ✅ 自动处理 |
| **闭环控制** | System 1 已支持 Policy Mode（真实观测反馈） | ✅ 原生支持 |
| **模型权重** | 无需任何微调或梯度更新 | ✅ Train-Free |

### 6.2 核心优势

1. **极度轻量 (Compute-Efficient)：** $\Pi_{sys2}$ 和 $\mathcal{E}$ 可直接调用成熟的闭源 API（GPT-4o）或轻量级开源 VLM（Qwen-VL）。对 DreamZero 是 **零侵入 (Zero-Intrusion)** 的，不需要任何梯度更新——这就是 Train-Free 的本质。

2. **可解释性极强 (Highly Interpretable)：** 纯端到端 VLA 失败时是黑盒。但在双系统架构下，如果任务失败，可以明确从日志中归因：
   - **LLM 规划错误 (System 2)：** 子指令序列不合理
   - **DreamZero 执行失败 (System 1)：** 动作生成不准确
   - **VLM 评估错误 (Evaluator)：** 状态判断有误
   
   这种 **白盒化的错误归因 (Error Attribution)** 在消融实验 (Ablation Study) 中非常容易出图表。

3. **失败恢复 (Failure Recovery)：** 当 VLM 检测到子任务失败时，LLM 可以根据当前真实状态动态重规划，而非盲目继续执行。

4. **泛化性强 (Generalizable)：** System 2 和 Evaluator 可以随时替换为更强的模型（如从 GPT-4 升级到 GPT-5），无需重新训练 System 1。

### 6.3 潜在挑战与应对

| 挑战 | 描述 | 应对策略 |
|------|------|---------|
| **VLM 延迟** | VLM 评估引入额外延迟 | 异步评估：执行与评估并行 |
| **子指令粒度** | 粒度过粗/过细都会影响性能 | 实验消融不同粒度 |
| **语义-物理鸿沟** | LLM 可能生成物理上不可行的指令 | 在 Prompt 中加入物理约束 |
| **VLM 幻觉** | VLM 可能误判任务完成状态 | 多帧验证 + 置信度阈值 |

---

## 七、实验设计建议

### 7.1 消融实验矩阵

| 实验 | System 2 | System 1 | Evaluator | 目标 |
|------|----------|----------|-----------|------|
| Baseline | ✗ (单一长指令) | DreamZero | ✗ | 原始性能上界 |
| +Planning | ✓ (LLM) | DreamZero | ✗ (固定执行) | 规划的增益 |
| +Evaluation | ✗ | DreamZero | ✓ (VLM) | 评估的增益 |
| **Full System** | **✓** | **DreamZero** | **✓** | **完整系统** |
| Oracle Planning | ✓ (人工标注) | DreamZero | ✓ | 规划上界 |
| Oracle Eval | ✓ | DreamZero | ✓ (Ground Truth) | 评估上界 |

### 7.2 评估指标

- **任务成功率 (Task Success Rate)：** 长程任务的整体完成率
- **子任务成功率 (Sub-task Success Rate)：** 各原子指令的完成率
- **重规划次数 (Re-planning Count)：** 平均每个任务的失败恢复次数
- **总执行时间 (Total Execution Time)：** 包含 LLM/VLM 调用开销
- **错误归因分布 (Error Attribution)：** System 2 / System 1 / Evaluator 各自的失败占比

---

## 八、总结

本方案的核心洞察在于：**DreamZero 的代码已经为动态指令切换提供了原生支持**（语言变化检测 → KV 缓存重置 → 条件重编码）。我们只需在外部搭建一个 LLM + VLM 的编排层，即可将 DreamZero 从"短视的直觉执行器"升级为"具备长程规划能力的智能体"——而这一切都是 **Train-Free** 的。

这种设计哲学类似于操作系统中的"微内核"思想：DreamZero 作为高效的"内核态"执行引擎保持不变，LLM 和 VLM 作为"用户态"的智能服务提供高层决策支持。
