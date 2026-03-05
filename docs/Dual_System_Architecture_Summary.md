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
│  ║  │         CausalWanModel (40-layer DiT)                │   ║    │
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
│  ║      Evaluator: VLM + Proprioception Verifier (E)         ║    │
│  ║                                                             ║    │
│  ║  Input:  o_{t+H} + l_k + proprioception q_{t+H}           ║    │
│  ║  Output: s_{t+H} + Success flag b_k (cross-verified)      ║    │
│  ║                                                             ║    │
│  ║  b_k = 1 → Confirmation Window (0.5s) → Next l_{k+1}     ║    │
│  ║  b_k = 0 → Re-plan with error message                     ║    │
│  ╚══════════════════════════════════════════════════════════════╝    │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 三大核心模块

#### 模块 A：System 2 — LLM 符号化长程规划器 ($\Pi_{sys2}$)

- **实现：** GPT-4o / Claude / Qwen 等大语言模型，通过 API 调用。
- **职责：** 接收长程指令 $L_{long}$ 和当前语义状态 $s_t$，输出原子化子指令 $l_k$。
- **关键特性：** Train-Free，利用 LLM 的先验知识进行零样本规划。
- **约束：** 必须从预定义的 **原子技能库 (Atomic Skill Library)** 中选择指令（详见第六节）。

#### 模块 B：System 1 — DreamZero 世界动作模型 ($\pi_{sys1}$)

- **实现：** 原始 DreamZero 模型，**完全不修改**。
- **职责：** 在原子指令 $l_k$ 的条件下，生成 chunk 级别的动作和视频。
- **关键接口：** `lazy_joint_forward_causal(batch)` — 只需替换 `batch` 中的 `text` 字段即可切换子任务。
- **关键参数：** `action_horizon=24`（每个 chunk 24 步动作），`num_frame_per_block=2`（每个 block 2 帧视频）。

#### 模块 C：Evaluator — 多模态交叉验证器 ($\mathcal{E}$)

- **实现：** VLM（如 GPT-4o-Vision / Qwen-VL）+ 机器人本体感知（Proprioception）。
- **职责：** 将真实观测 $o_{t+H}$ 和本体感知 $q_{t+H}$ 联合映射为语义状态 $s_{t+H}$，通过交叉验证判断子任务是否完成。
- **关键特性：** 不单独信任视觉或本体感知，而是两者同时确认才跳转（详见第六节）。

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

$$l_k = \Pi_{sys2}(L_{long}, s_t, \mathcal{H}_t) = \arg\max_{l \in \mathcal{A}} P_{LLM}(l \mid s_t, L_{long}, \mathcal{H}_t)$$

其中 $\mathcal{A}$ 为原子技能库（见 6.1 节），约束 LLM 只能从物理可行的指令中选择。

### 3.3 System 1 的连续去噪生成

DreamZero 的训练目标（Flow Matching）保持不变，但推理时条件文本被替换为 System 2 下发的 $l_k$：

$$\mathcal{L}_{WAM}(\theta) = \mathbb{E}_{x, t} \left[ \left\| u_\theta\left([z_{t_{noise}}, a_{t_{noise}}]; \mathcal{C}_{clean}, \mathbf{l_k}, q\right) - v_{target} \right\|^2 \right]$$

其中：
- $u_\theta$：CausalWanModel（40 层 DiT，dim=5120，40 heads）
- $\mathcal{C}_{clean}$：干净的历史视频上下文帧
- $l_k$：System 2 下发的原子指令（通过 UMT5 编码后经交叉注意力注入）
- $q$：机器人本体感知状态（关节位置、夹爪状态等，通过 `CategorySpecificMLP` 编码）
- $v_{target}$：Flow Matching 的速度目标

### 3.4 多模态交叉验证状态评估

定义状态映射 $\mathcal{E}_{VLM} : \mathcal{O} \rightarrow \mathcal{S}$，将高维像素映射为低维语义描述。定义多模态指示函数判定子任务后置条件：

$$b_k = \mathbb{I}\left(\mathcal{E}_{VLM}(o_{t+H}) \models \text{PostCond}(l_k)\right) \wedge \mathbb{I}\left(\mathcal{V}_{proprio}(q_{t+H}) \models \text{PostCond}(l_k)\right)$$

即：**VLM 视觉判断成功 AND 本体感知判断成功**，两者同时为真才认为子任务完成。

状态机转移规则：
- **$b_k = 1$（成功）：** 进入确认窗口 → 静止 0.5s → $\mathcal{H}_{t+H} = \mathcal{H}_t \cup \{(s_{t+H}, l_k)\}$ → 弹出 $l_{k+1}$
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

**代码证据 4：Prompt 字段的动态替换接口**

```python
# socket_test_optimized_AR.py, line 170-174
if "prompt" in obs:
    converted["annotation.language.action_text"] = obs["prompt"]
```

WebSocket 接口中 `prompt` 字段可以逐帧动态替换，这是 System 2 下发子指令的天然入口。

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
         (with Soft Reset, Cross-Verification, and Confirmation Window)

Input:  Long-horizon instruction L_long, Initial observation o_0
        Atomic Skill Library A, Confirmation duration T_confirm = 0.5s
Output: Complete trajectory τ_{0:T}

 1:  Initialize: H_0 ← ∅, k ← 0
 2:  s_0 ← E_VLM(o_0)                              // VLM extracts initial state
 3:  [l_1, ..., l_n] ← Π_sys2(L_long, s_0, A)      // LLM plans from skill library
 4:  
 5:  while k < n do
 6:      k ← k + 1
 7:      t ← current_timestep
 8:      chunk_count ← 0
 9:      
10:      // ===== System 1: DreamZero Execution =====
11:      // Text change triggers automatic KV cache reset (line 972-975)
12:      Pack l_k into batch.obs["prompt"]
13:      
14:      repeat
15:          o_t ← capture_real_observation()
16:          q_t ← read_proprioception()           // joint pos, gripper, force/torque
17:          Pack o_t into batch.obs["video"]
18:          
19:          // DreamZero inference (UNCHANGED model)
20:          a_{t:t+H}, z_{t:t+H} ← π_sys1.lazy_joint_forward_causal(batch)
21:          
22:          Execute a_{t:t+H} on robot
23:          t ← t + H
24:          chunk_count ← chunk_count + 1
25:          
26:          // === Async VLM check (runs in background every 3 chunks) ===
27:          if chunk_count mod 3 == 0 then
28:              Launch async: vlm_result ← E_VLM(o_t, l_k)
29:          end if
30:          
31:          // === Check for async VLM result ===
32:          if vlm_result is ready then
33:              break
34:          end if
35:          
36:      until chunk_count >= MAX_CHUNKS_PER_SUBTASK
37:      
38:      // ===== Evaluator: Multi-Modal Cross-Verification =====
39:      o_t ← capture_real_observation()
40:      q_t ← read_proprioception()
41:      
42:      // Cross-verify: VLM + Proprioception must BOTH agree
43:      b_vlm   ← E_VLM(o_t) ⊨ PostCond(l_k)
44:      b_prop  ← V_proprio(q_t) ⊨ PostCond(l_k)
45:      b_k     ← b_vlm AND b_prop
46:      
47:      if b_k = 1 then
48:          // ===== Confirmation Window: stabilize before next task =====
49:          Hold current position for T_confirm seconds
50:          o_confirm ← capture_real_observation()
51:          b_stable ← check_stability(o_confirm, o_t)  // object didn't fall/slip
52:          
53:          if b_stable then
54:              H_t ← H_t ∪ {(s_t, l_k, "success")}
55:              continue to next sub-instruction
56:          else
57:              // Object unstable after confirmation → retry
58:              error_msg ← "Object became unstable during confirmation"
59:              [l_k', ...] ← Π_sys2(L_long, s_t, H_t, error_msg)
60:              Update remaining sub-instructions
61:          end if
62:      else
63:          // ===== Failure: diagnose and re-plan =====
64:          error_msg ← diagnose_failure(b_vlm, b_prop, o_t, q_t, l_k)
65:          H_t ← H_t ∪ {(s_t, l_k, "failed", error_msg)}
66:          [l_k', ..., l_m'] ← Π_sys2(L_long, s_t, H_t, error_msg)
67:          Update remaining sub-instructions
68:      end if
69:  end while
70:
71:  return Complete trajectory τ_{0:T}
```

---

## 六、五大关键工程问题与解决方案

### 6.1 规避"物理断层"：软切换 (Soft Reset)

**问题：** 当 System 2 下发新指令 $l_{k+1}$ 时，代码中 `current_start_frame` 被重置为 0，KV 缓存被完全清空。这意味着模型在处理新指令的第一帧时，完全丧失了之前的运动上下文，可能导致动作突变（如瞬间甩手、松开夹爪）。

**代码根因：**

```python
# wan_flow_matching_action_tf.py, line 972-975
elif not torch.equal(self.language, data["text"]):
    self.current_start_frame = 0  # ← 硬重置，所有历史丢失
    self.language = data["text"]
```

当 `current_start_frame = 0` 时，第 1051-1063 行会重新初始化 KV 缓存：

```python
# line 1051-1058
if self.current_start_frame == 0:
    self.kv_cache1, self.kv_cache_neg = self._create_kv_caches(...)
    self.crossattn_cache, self.crossattn_cache_neg = self._create_crossattn_caches(...)
```

**解决方案：保留最后 N 帧的 KV 缓存作为"惯性残影"**

不要让 `current_start_frame` 直接归零，而是保留最后 `num_frame_per_block`（当前配置为 2）帧的 KV 缓存在新序列的头部。

**具体做法：**

```python
# 修改 wan_flow_matching_action_tf.py, line 972-975
elif not torch.equal(self.language, data["text"]):
    # Soft Reset: 保留最后 num_frame_per_block 帧的 KV 缓存
    # 而不是硬重置到 0
    retain_frames = self.num_frame_per_block  # = 2
    if self.current_start_frame > retain_frames:
        self.current_start_frame = retain_frames
        # KV 缓存中 [0:retain_frames] 的位置保留旧数据
        # 新指令从 retain_frames 位置开始写入
    else:
        self.current_start_frame = 0  # 如果历史太短，仍然硬重置
    self.language = data["text"]
```

**原理：** 模型在处理新指令的第一帧时，因果自注意力仍能看到前 2 帧的"残影"（旧 KV 缓存），动作输出会顺着之前的惯性平滑过渡。随着新帧的积累，旧残影会被自然滚出 `local_attn_size` 窗口（当前配置 `max_chunk_size=4`，`local_attn_size = 4 * 2 + 1 = 9`），不会永久污染。

**可行性：高。** 只需修改 2-3 行代码，不影响模型权重和推理逻辑。KV 缓存的滚动机制（`local_attn_size`）天然支持这种"部分保留"。

**风险与兜底：** 如果软切换导致新指令的前几帧动作"拖泥带水"（例如该放手时还在抓），可以通过环境变量 `SOFT_RESET_FRAMES` 控制保留帧数（0 = 硬重置，即退回原始行为）。

### 6.2 规避"VLM 幻觉"：多模态交叉验证 (Cross-Verification)

**问题：** VLM 可能产生幻觉，例如看到"手靠近物体"就判断"已经抓住了"，但实际上夹爪是空的。如果只依赖视觉判断就跳转到下一个子任务，会导致级联失败。

**解决方案：建立"状态真值表"，VLM + 本体感知双重确认**

```python
# 编排脚本中的交叉验证逻辑
def cross_verify(vlm_result: dict, proprioception: dict, subtask: str) -> tuple[bool, str]:
    """
    VLM 和本体感知必须同时确认成功，才认为子任务完成。
    
    Returns:
        (success: bool, diagnosis: str)
    """
    # 定义每种子任务的本体感知验证条件
    PROP_CONDITIONS = {
        "pick": lambda q: q["gripper_aperture"] < 0.3,      # 夹爪闭合 = 抓到东西
        "place": lambda q: q["gripper_aperture"] > 0.7,     # 夹爪张开 = 放下了
        "push": lambda q: abs(q["ee_velocity"]) < 0.01,     # 末端速度≈0 = 推完停下
        "open": lambda q: q["gripper_aperture"] > 0.5,      # 夹爪张开
        "close": lambda q: q["gripper_aperture"] < 0.2,     # 夹爪闭合
    }
    
    # 从子任务文本中提取动作类型
    action_type = extract_action_type(subtask)  # "Pick up X" → "pick"
    
    b_vlm = vlm_result["success"]
    b_prop = PROP_CONDITIONS.get(action_type, lambda q: True)(proprioception)
    
    if b_vlm and b_prop:
        return True, "Both VLM and proprioception confirm success"
    elif b_vlm and not b_prop:
        return False, f"VLM says success but proprioception disagrees (gripper={proprioception['gripper_aperture']:.2f})"
    elif not b_vlm and b_prop:
        return False, f"Proprioception suggests success but VLM disagrees: {vlm_result['reason']}"
    else:
        return False, f"Both VLM and proprioception indicate failure: {vlm_result['reason']}"
```

**原理：** 传感器数据（夹爪开合度、关节力矩、末端速度）是**绝对真实的**，不会产生幻觉。用它来修正 VLM 的视觉误差。

**可行性：中高。** 需要机器人提供本体感知数据，但 DROID 数据集本身就包含 `joint_position` 和 `gripper_position`，代码中已有对应接口：

```python
# socket_test_optimized_AR.py, line 152-168
if "observation/joint_position" in obs:
    joint_pos = obs["observation/joint_position"]
    converted["state.joint_position"] = joint_pos.astype(np.float64)
if "observation/gripper_position" in obs:
    gripper_pos = obs["observation/gripper_position"]
    converted["state.gripper_position"] = gripper_pos.astype(np.float64)
```

### 6.3 规避"执行边界模糊"：确认窗口 (Confirmation Window)

**问题：** 很多端到端模型在任务结束瞬间会有"甩尾"动作（例如放下杯子后手臂突然抖动）。如果一看到 VLM 说"成功"就立刻跳转到下一个指令，可能在甩尾期间就开始新任务，导致物体被碰倒。

**解决方案：成功信号后强制静止 0.5 秒**

```python
# 编排脚本中的确认窗口逻辑
async def confirmation_window(robot, duration=0.5, stability_threshold=0.005):
    """
    成功信号后，强制机器人保持当前位置不动。
    检查物体是否在物理上已经稳定（不会因惯性倒下）。
    """
    start_time = time.time()
    initial_obs = robot.capture_observation()
    
    while time.time() - start_time < duration:
        # 发送"保持当前位置"的动作（零速度）
        robot.hold_position()
        await asyncio.sleep(0.03)  # 30Hz 控制频率
    
    final_obs = robot.capture_observation()
    
    # 检查场景是否稳定（像素级变化很小）
    pixel_diff = np.abs(final_obs["image"] - initial_obs["image"]).mean()
    joint_diff = np.abs(final_obs["joint_position"] - initial_obs["joint_position"]).max()
    
    is_stable = (pixel_diff < stability_threshold) and (joint_diff < 0.01)
    return is_stable
```

**原理：** 0.5 秒的静止窗口确保：
1. 机器人的甩尾动作已经衰减
2. 物体在物理上已经稳定（杯子放稳了、抽屉关到位了）
3. 下一个子任务的首帧观测是"干净"的（没有运动模糊）

**可行性：高。** 纯编排层逻辑，不涉及模型修改。0.5 秒的延迟在长程任务中几乎可以忽略。

### 6.4 规避"语义-物理鸿沟"：原子技能库 (Atomic Skill Library)

**问题：** LLM 可能给出看似合理但 DreamZero 执行不了的指令。例如 LLM 说"把杯子倒扣"，但如果 DreamZero 的训练集里没有旋转手腕的动作，这个子任务必定失败。

**解决方案：在 Prompt 中明确约束 LLM 的输出空间**

```python
ATOMIC_SKILL_LIBRARY = """
You are a robot task planner. You can ONLY use the following atomic skills:
- "Pick up [object]": Grasp an object with the gripper
- "Place [object] on/in [location]": Put a held object down
- "Push [object] [direction]": Push an object forward/backward/left/right
- "Open [container]": Open a drawer, door, or lid
- "Close [container]": Close a drawer, door, or lid
- "Move arm to [location]": Move end-effector to a position (no grasping)

CONSTRAINTS:
1. Each sub-instruction must be completable within 5-8 seconds (3-5 action chunks)
2. Do NOT use skills not listed above (no rotating, flipping, pouring, etc.)
3. Each sub-instruction must have a clear, visually verifiable success condition
4. If the task requires a skill not in the library, respond with "UNSUPPORTED_SKILL: [reason]"

FORMAT: Return a JSON array of sub-instructions, each with:
{
    "instruction": "Pick up the red cup",
    "success_condition": "Red cup is held by the gripper, lifted off the table",
    "expected_duration_chunks": 4,
    "proprioception_check": {"gripper_aperture": "<0.3"}
}
"""
```

**原理：** 通过 Prompt Engineering 将 LLM 的输出空间限制在 DreamZero 训练分布内的可执行动作上。每个子指令都附带明确的成功条件和本体感知检查条件，为后续的交叉验证提供依据。

**可行性：高。** 纯 Prompt 设计，不涉及任何代码修改。技能库可以根据 DreamZero 的训练数据集（DROID）的动作分布来定义。

### 6.5 规避"闭环延迟"：异步流水线 (Async Pipeline)

**问题：** 如果每执行一个 chunk 都要等 VLM 推理 2 秒来判断是否成功，机器人会非常"卡顿"。

**解决方案：System 1 连续执行，VLM 在后台异步评估**

```
时间线：
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ C1   │ C2   │ C3   │ C4   │ C5   │ C6   │ ...  │  ← System 1 连续执行 chunks
└──────┴──────┴──────┴──────┴──────┴──────┴──────┘
       │             │             │
       ▼             ▼             ▼
    ┌──────┐      ┌──────┐      ┌──────┐
    │ VLM  │      │ VLM  │      │ VLM  │           ← VLM 异步评估（每 3 chunks）
    │ eval │      │ eval │      │ eval │
    └──┬───┘      └──┬───┘      └──┬───┘
       │             │             │
       ▼             ▼             ▼
    result_1      result_2      result_3
    (ignored      (success →    
     if ok)        interrupt!)
```

**具体做法：**

```python
import asyncio
import concurrent.futures

class AsyncVLMEvaluator:
    def __init__(self, vlm_client, check_interval=3):
        self.vlm_client = vlm_client
        self.check_interval = check_interval  # 每 N 个 chunk 检查一次
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._pending_future = None
        self._latest_result = None
    
    def maybe_submit(self, chunk_count, observation, subtask):
        """每 check_interval 个 chunk 提交一次异步 VLM 评估"""
        if chunk_count % self.check_interval != 0:
            return
        if self._pending_future is not None and not self._pending_future.done():
            return  # 上一次评估还没完成，跳过
        self._pending_future = self.executor.submit(
            self.vlm_client.evaluate, observation, subtask
        )
    
    def poll_result(self):
        """非阻塞地检查是否有评估结果"""
        if self._pending_future is not None and self._pending_future.done():
            self._latest_result = self._pending_future.result()
            self._pending_future = None
            return self._latest_result
        return None
```

**原理：** System 1（DreamZero）以 ~0.6-3s/chunk 的速度连续执行，不等 VLM。VLM 在后台线程中异步运行，每 3 个 chunk（约 5-10 秒）检查一次。如果 VLM 检测到子任务已完成或失败，通过中断信号通知编排层切换指令。

**延迟分析：**
- 最坏情况：VLM 在第 1 个 chunk 后就检测到成功，但要等到第 3 个 chunk 才被检查 → 多执行了 2 个 chunk（约 3-6 秒）
- 这在长程任务中是可接受的，远好于"每 chunk 都等 2 秒 VLM"的方案

**可行性：高。** 纯 Python 异步编程，不涉及模型修改。

---

## 七、指令粒度设计

### 7.1 最优粒度：3-5 个 Action Chunks（约 5-8 秒物理时间）

DreamZero 的关键参数：
- `action_horizon = 24`（每个 chunk 24 步动作）
- 控制频率 30Hz → 每个 chunk 约 **0.8 秒**
- `num_frame_per_block = 2`（每个 block 2 帧视频）
- `max_chunk_size = 4` → `local_attn_size = 4 * 2 + 1 = 9`（KV 缓存窗口 9 帧）

| 粒度 | Chunks | 物理时间 | 优点 | 缺点 |
|------|--------|---------|------|------|
| 太细 | 1-2 | 0.8-1.6s | 精确控制 | System 2 频繁调用 API，延迟大 |
| **最优** | **3-5** | **2.4-4.0s** | 平衡精度与效率 | — |
| 太粗 | 8+ | 6.4s+ | 减少 API 调用 | DreamZero 误差级联，"短视"问题重现 |

### 7.2 粒度自适应

不同类型的子任务可以使用不同粒度：

```python
SUBTASK_CHUNK_BUDGET = {
    "pick": 4,      # 抓取：需要精确对准，4 chunks
    "place": 3,     # 放置：相对简单，3 chunks
    "push": 5,      # 推动：需要持续力，5 chunks
    "open": 4,      # 开启：需要拉动，4 chunks
    "close": 3,     # 关闭：相对简单，3 chunks
    "move": 2,      # 移动：最简单，2 chunks
}
```

---

## 八、实验设计

### 8.1 消融实验矩阵

| 实验 | System 2 | System 1 | Evaluator | Soft Reset | Cross-Verify | Confirm | 目标 |
|------|----------|----------|-----------|------------|-------------|---------|------|
| Baseline | ✗ (单一长指令) | DreamZero | ✗ | ✗ | ✗ | ✗ | 原始性能上界 |
| +Planning | ✓ (LLM) | DreamZero | ✗ (固定执行) | ✗ | ✗ | ✗ | 规划的增益 |
| +Soft Reset | ✓ | DreamZero | ✗ | ✓ | ✗ | ✗ | 软切换的增益 |
| +Evaluation | ✓ | DreamZero | ✓ (VLM only) | ✓ | ✗ | ✗ | VLM 评估的增益 |
| +Cross-Verify | ✓ | DreamZero | ✓ | ✓ | ✓ | ✗ | 交叉验证的增益 |
| **Full System** | **✓** | **DreamZero** | **✓** | **✓** | **✓** | **✓** | **完整系统** |
| Oracle Planning | ✓ (人工标注) | DreamZero | ✓ | ✓ | ✓ | ✓ | 规划上界 |
| Oracle Eval | ✓ | DreamZero | ✓ (Ground Truth) | ✓ | ✓ | ✓ | 评估上界 |

### 8.2 评估范式：复刻与超越（对齐原论文 Figure/Table）

在保留原论文主干指标的基础上，评估分为两层：  
1) **复刻层**：对齐原论文经典表格/图；  
2) **超越层**：专门量化 System 2 对长程任务的增益。

| 编号 | 对应原论文 | 建议展示 | 核心结论 | 对应代码指标 | 可行度 |
|------|-----------|---------|---------|-------------|--------|
| 1 | Table 1 (Zero-shot Evaluation) | **Task Success Rate 表**（任务列 + Long-Horizon 复合任务列） | 单任务持平，长程复合任务显著拉开差距 | `success_rate`, `total_episodes` | **高** |
| 2 | Figure 5 (Future Prediction) 变体 | **Trajectory Drift Visualization**（2D 路径 + chunk 标注） | 原生在第 3 个 chunk 后漂移，System 2 在切换点纠偏 | `error_accumulation`, `chunk_count`, `instruction_switches` | **中**（需补 2D 轨迹日志） |
| 3 | Table 3 (Effect of Language Instructions) | **Instruction Granularity 表**（Single Long vs Decoupled） | 指令分解将复杂语义转化为可执行低复杂度子目标 | `instruction_complexity`, `subtask_count`, `completion_rate` | **高** |
| 4 | Figure 8 (Robustness Tasks) | **Recovery after Perturbation 柱状图** | 干扰后原生失败，System 2 通过动态重规划恢复 | `successful`, `failed`, `dynamic_adjustment` | **中**（需显式 perturbation/recovery 标记） |

### 8.3 深度新增图（针对三大痛点）

| 编号 | 图名 | 解决痛点 | 绘图逻辑 | 对应代码指标 | 可行度 |
|------|------|---------|---------|-------------|--------|
| 5 | **Sawtooth Error Plot**（误差重置锯齿图） | 误差级联 | X: chunk/time；Y: `error`；在 `instruction_switches`/`dynamic_adjustment` 处应出现误差回落 | `error_accumulation`, `instruction_switches`, `dynamic_adjustment` | **高** |
| 6 | **Logical Consistency Heatmap**（语义逻辑一致性矩阵） | 语义断裂 | 统计不同任务深度下 `logical_errors` 频次，原生 vs System 2 对比热力图 | `logical_errors`, `subtask_count`, `completion_rate` | **中-低**（当前 `logical_errors` 基本未实算） |
| 7 | **Instruction Efficiency & Switches**（指令效能效率图） | 单一指令瓶颈 | 堆叠柱/双轴：`instruction_duration` + `ideal_switches` vs `instruction_switches` | `instruction_duration`, `ideal_switches`, `instruction_switches` | **高** |
| 8 | **Computation-Performance Trade-off**（时间-收益平衡图） | 回应“推理慢”质疑 | X: 总执行时长（含 LLM/VLM 开销）；Y: 成功率；强调长程收益 | `success_rate`, `total_execution_time` | **中**（需统一时延日志） |

### 8.4 指标采集规范（最小补充）

为保证上述图表可稳定复现，建议在 `evaluation_log.json` 增加以下字段（不改 DreamZero 模型本体）：

- `chunk_id`, `chunk_start_ts`, `chunk_end_ts`, `chunk_error`
- `instruction_id`, `instruction_text`, `instruction_switch`（布尔）
- `dynamic_adjustment`（布尔）, `replan_reason`
- `ee_xy` 或 `object_xy`（用于 2D 轨迹）
- `perturbation_type`, `recovery_mode`, `recovery_success`
- `logical_error_type`（如 precondition_violation / ordering_error）
- `llm_latency_ms`, `vlm_latency_ms`, `total_execution_time_ms`

### 8.5 推荐实验场景

基于 DROID 数据集的动作分布，推荐以下长程任务：

| 场景 | 子任务数 | 难度 | 测试重点 |
|------|---------|------|---------|
| 收拾桌面 | 3-5 | 中 | 多物体顺序操作 |
| 厨房整理 | 5-8 | 高 | 开关抽屉 + 抓放 |
| 工具使用 | 4-6 | 高 | 工具交互 |
| 物品分类 | 6-10 | 中高 | 大量重复子任务 |

---

## 九、总结

本方案的核心洞察在于：**DreamZero 的代码已经为动态指令切换提供了原生支持**（语言变化检测 → KV 缓存重置 → 条件重编码）。我们只需在外部搭建一个 LLM + VLM 的编排层，即可将 DreamZero 从"短视的直觉执行器"升级为"具备长程规划能力的智能体"——而这一切都是 **Train-Free** 的。

相比原始方案，本版本新增了五大工程保障机制：

| 机制 | 解决的问题 | 改动范围 |
|------|-----------|---------|
| **软切换 (Soft Reset)** | 子任务切换时的动作突变 | 修改 2-3 行模型代码 |
| **多模态交叉验证** | VLM 幻觉导致的误判 | 编排层逻辑 |
| **确认窗口** | 甩尾动作导致的物体不稳定 | 编排层逻辑 |
| **原子技能库** | LLM 生成不可执行指令 | Prompt 设计 |
| **异步流水线** | VLM 评估导致的卡顿 | 编排层逻辑 |

这种设计哲学类似于操作系统中的"微内核"思想：DreamZero 作为高效的"内核态"执行引擎保持不变（或仅做最小修改），LLM 和 VLM 作为"用户态"的智能服务提供高层决策支持。
