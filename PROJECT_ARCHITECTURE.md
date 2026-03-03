# DreamZero 项目架构文档

> 本文档梳理了 DreamZero 项目的完整骨架，包括宏观架构、数据流向、渲染模块位置以及每个文件的核心类输入输出。

---

## 目录

- [1. 项目总览](#1-项目总览)
- [2. 目录结构](#2-目录结构)
- [3. 宏观架构（数据预处理 → 模型定义 → 训练逻辑 → 主函数入口）](#3-宏观架构)
- [4. 数据流向（Tensor 从 DataLoader 到输出的维度变化）](#4-数据流向)
- [5. 渲染模块位置](#5-渲染模块位置)
- [6. 单个文件核心类输入输出详解](#6-单个文件核心类输入输出详解)

---

## 1. 项目总览

DreamZero 是一个 **视觉-语言-动作（VLA）模型**，核心思想是：

- 基于 **Wan2.1 视频扩散模型** 做视频生成（世界模型）
- 在扩散过程中 **联合预测动作**（flow matching）
- 支持 **多 embodiment**（DROID、GR1、Unitree G1 等）
- 采用 **因果注意力 + KV cache** 实现自回归推理

**一句话概括**：输入当前观测图像 + 语言指令 → 扩散模型联合生成未来视频帧 + 机器人动作序列。

---

## 2. 目录结构

```
dreamzero/
├── groot/                          # 核心代码库
│   └── vla/
│       ├── common/utils/           # 通用工具（图像、视频、IO、数据结构）
│       ├── configs/                # Hydra YAML 配置
│       │   ├── conf.yaml           # 主配置入口
│       │   ├── data/dreamzero/     # 数据配置
│       │   └── model/dreamzero/    # 模型配置
│       ├── data/                   # 数据加载与预处理
│       │   ├── dataset/            # Dataset 类（LeRobot 格式）
│       │   ├── transform/          # 数据变换（视频/状态/动作/语言）
│       │   ├── schema/             # 数据元数据定义（Pydantic）
│       │   └── conversion/         # 数据格式转换（GR1、DROID）
│       ├── model/                  # 模型定义
│       │   ├── dreamzero/          # DreamZero 核心模型
│       │   │   ├── base_vla.py     # VLA 顶层模型
│       │   │   ├── action_head/    # 动作头（Flow Matching）
│       │   │   ├── backbone/       # Backbone（Identity）
│       │   │   ├── modules/        # 子模块（DiT、VAE、编码器、注意力等）
│       │   │   └── transform/      # 模型专用数据变换
│       │   └── n1_5/              # N1.5 相关模块（ActionHead 基类、策略封装）
│       ├── experiment/             # 训练逻辑
│       │   ├── base.py             # BaseTrainer + BaseExperiment
│       │   ├── experiment.py       # VLAExperiment + main 入口
│       │   └── utils.py            # 训练工具函数
│       └── utils/                  # 项目工具
├── eval_utils/                     # 评估工具
│   ├── policy_server.py            # WebSocket 策略服务抽象
│   ├── policy_client.py            # WebSocket 策略客户端
│   └── run_sim_eval.py             # 仿真评估入口
├── scripts/                        # 脚本
│   ├── compare_loss.py             # Loss 对比工具
│   └── data/convert_droid.py       # DROID 数据转换
├── socket_test_optimized_AR.py     # AR 策略服务（dreamzero-server 入口）
├── test_client_AR.py               # AR 策略测试客户端
├── pyproject.toml                  # 项目配置与依赖
└── checkpoints/                    # 预训练权重
```

---

## 3. 宏观架构

### 3.1 整体流水线

```
┌─────────────────────────────────────────────────────────────────────┐
│                        训练流水线                                    │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌───────────┐ │
│  │ LeRobot  │───>│  Transform   │───>│  Model   │───>│  Trainer  │ │
│  │ Dataset  │    │  Pipeline    │    │  (VLA)   │    │  (HF)     │ │
│  └──────────┘    └──────────────┘    └──────────┘    └───────────┘ │
│       ↑                ↑                  ↑               ↑        │
│   数据加载          数据预处理          模型定义         训练逻辑    │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 数据预处理阶段

**入口**：`groot/vla/data/dataset/` + `groot/vla/data/transform/`

```
原始数据（LeRobot parquet + mp4 视频）
    │
    ▼
LeRobotSingleDataset / ShardedLeRobotSingleDataset
    │  读取 parquet（state/action）+ 解码视频帧
    │  输出: dict { "video.image_side_0": ndarray, "state.xxx": ndarray, "action.xxx": ndarray, "language": str }
    ▼
Transform Pipeline（按顺序执行）:
    1. VideoTransform（裁剪、缩放、颜色增强）
    2. VideoToTensor（uint8 → float32, HWC → CHW）
    3. VideoNormalize（均值/方差归一化）
    4. StateActionToTensor（ndarray → Tensor）
    5. StateActionTransform（旋转表示转换 + 归一化）
    6. LanguageTransform（tokenize）
    7. ConcatTransform（多视角/多键合并）
    │
    ▼
DreamTransform（模型专用变换）:
    - 多视角视频拼接为 (B, T, V, H, W, 3)
    - 语言 tokenize 为 input_ids
    - State/Action padding 到 max_dim
    - 添加 embodiment_id
    │
    ▼
DefaultDataCollator（batch collate）
    │
    ▼
DataLoader 输出 → 模型输入
```

### 3.3 模型定义阶段

**入口**：`groot/vla/model/dreamzero/base_vla.py`

```
VLA（顶层模型）
├── Backbone（IdentityBackbone —— 当前不使用额外 backbone）
│   └── 输出: backbone_features (B, 1, 0)
│
└── ActionHead（WANPolicyHead —— 核心）
    ├── WanTextEncoder（T5）        → 文本 embedding (B, L, 4096)
    ├── WanImageEncoder（CLIP）     → 图像特征 (B, L, 1280)
    ├── WanVideoVAE                 → 视频 latent (B, 16, T/4, H/8, W/8)
    ├── CausalWanModel（DiT）       → 联合去噪：视频 noise pred + 动作 noise pred
    ├── FlowMatchScheduler          → 训练时加噪/计算目标
    └── FlowUniPCMultistepScheduler → 推理时多步采样
```

### 3.4 训练逻辑

**入口**：`groot/vla/experiment/experiment.py` → `main()`

```
main()（Hydra 装饰）
    │
    ▼
VLAExperiment.__init__(cfg)
    ├── create_model(cfg)          → 实例化 VLA
    ├── create_train_dataset(cfg)  → 实例化 ShardedLeRobotMixtureDataset
    └── create_trainer(cfg)        → 实例化 VLATrainer
    │
    ▼
VLAExperiment.train()
    └── VLATrainer.train()（继承 HuggingFace Trainer）
        │
        ├── get_train_dataloader()  → DataLoader（IterableDataset + 分片采样）
        │
        └── training_step(model, inputs)
            ├── model.forward(inputs)
            │   ├── prepare_input(inputs)     → backbone_input, action_input
            │   ├── backbone(backbone_input)   → backbone_output
            │   └── action_head(backbone_output, action_input)
            │       ├── encode_prompt()        → 文本条件
            │       ├── encode_image()         → 图像条件
            │       ├── encode_video()         → 视频 latent
            │       ├── 加噪（flow matching）
            │       ├── CausalWanModel()       → 去噪预测
            │       └── 计算 loss（dynamics_loss + action_loss）
            │
            └── loss.backward() → 梯度更新
```

### 3.5 主函数入口汇总

| 入口命令 | 文件 | 函数 | 用途 |
|----------|------|------|------|
| `python -m groot.vla.experiment.experiment` | `groot/vla/experiment/experiment.py` | `main()` | **训练** |
| `dreamzero-server` | `socket_test_optimized_AR.py` | `main()` | **推理服务** |
| `python eval_utils/run_sim_eval.py` | `eval_utils/run_sim_eval.py` | `main()` | **仿真评估** |
| `python test_client_AR.py` | `test_client_AR.py` | `main()` | **测试客户端** |
| `python scripts/data/convert_droid.py` | `scripts/data/convert_droid.py` | `__main__` | **数据转换** |
| `python scripts/compare_loss.py` | `scripts/compare_loss.py` | `main()` | **Loss 对比** |

---

## 4. 数据流向（Tensor 从 DataLoader 到输出的维度变化）

### 4.1 完整维度变化链

以 DROID 数据集、49 帧、2 视角、action_horizon=48 为例：

```
阶段                          键名                        维度 / 类型
─────────────────────────────────────────────────────────────────────────

【Dataset __getitem__】
  视频帧（原始）              video.image_side_0          (49, H_raw, W_raw, 3) uint8
  视频帧（原始）              video.image_side_1          (49, H_raw, W_raw, 3) uint8
  状态                        state.cartesian_position    (1, 6) float64
  状态                        state.gripper_position      (1, 1) float64
  动作                        action.cartesian_position   (48, 6) float64
  动作                        action.gripper_position     (48, 1) float64
  语言                        language                    str

【VideoTransform（裁剪+缩放）】
  视频帧                      video.image_side_0          (49, 480, 832, 3) uint8
  视频帧                      video.image_side_1          (49, 480, 832, 3) uint8

【VideoToTensor】
  视频帧                      video.image_side_0          (49, 3, 480, 832) float32 [0,1]

【VideoNormalize】
  视频帧                      video.image_side_0          (49, 3, 480, 832) float32 归一化

【StateActionToTensor + StateActionTransform】
  状态（归一化后）            state.cartesian_position    (1, 6) float32
  动作（归一化后）            action.cartesian_position   (48, 6) float32

【ConcatTransform】
  视频（多视角合并）          video                       (49, 2, 480, 832, 3)
  状态（多键 concat）         state                       (1, 7)    → [cart_pos(6) + gripper(1)]
  动作（多键 concat）         action                      (48, 7)   → [cart_pos(6) + gripper(1)]

【DreamTransform（模型专用）】
  视频                        video                       (49, 2, 480, 832, 3) uint8
  语言 token                  input_ids                   (max_length,) int64
  注意力掩码                  attention_mask              (max_length,) int64
  状态（padding 到 max_dim）  state                       (1, 64) float32
  动作（padding 到 max_dim）  action                      (48, 64) float32
  embodiment_id               embodiment_id               () int64

【DataLoader Collate（batch_size=B）】
  视频                        video                       (B, 49, 2, 480, 832, 3)
  语言 token                  input_ids                   (B, max_length)
  状态                        state                       (B, 1, 64)
  动作                        action                      (B, 48, 64)
  embodiment_id               embodiment_id               (B,)

─────────────────── 进入模型 ───────────────────

【VLA.prepare_input】
  视频重排                    videos                      (B, 3, 49, 480, 832) → 转为 CHW 格式
  → 多视角拼接为宽图                                      (B, 3, 49, 480, 1664) → 2 视角横向拼接

【WANPolicyHead.forward】

  ┌─ encode_prompt(input_ids, attention_mask)
  │   输入: (B, L), (B, L)
  │   输出: context (B, L, 4096)                          T5 文本编码
  │
  ├─ encode_image(videos[:,:,:1,:,:])
  │   输入: (B, 3, 1, 480, 1664)                          首帧
  │   输出: clip_context (B, 257, 1280)                    CLIP 视觉特征
  │          y (B, 512)                                    全局 embedding
  │
  ├─ encode_video(videos)
  │   输入: (B, 3, 49, 480, 1664)
  │   VAE 编码: (B, 16, 13, 60, 208)                      T/4=13, H/8=60, W/8=208
  │   输出: latents (B, 16, 13, 60, 208)
  │
  ├─ FlowMatchScheduler.add_noise(latents, noise, timestep)
  │   输入: latents (B, 16, 13, 60, 208), noise 同形状
  │   输出: noisy_latents (B, 16, 13, 60, 208)
  │
  ├─ CausalWanModel._forward_train(noisy_latents, t, t_action, context, action, state, embodiment_id)
  │   输入:
  │     x: (B, 16, 13, 60, 208)     noisy 视频 latent
  │     t: (B,)                       视频时间步
  │     t_action: (B,)                动作时间步
  │     context: (B, L, 4096)         文本条件
  │     action: (B, 48, 64)           noisy 动作
  │     state: (B, 1, 64)             状态条件
  │     embodiment_id: (B,)           embodiment 标识
  │
  │   内部处理:
  │     视频 patch 化: (B, 16, 13, 60, 208) → (B, N_video, dim)
  │     动作编码: (B, 48, 64) → (B, N_action, dim)
  │     拼接: [video_tokens, action_tokens] → 因果注意力
  │
  │   输出:
  │     video_noise_pred: (B, 16, 13, 60, 208)
  │     action_noise_pred: (B, 48, 64)
  │
  └─ 计算 Loss
      dynamics_loss = MSE(video_noise_pred, video_target)    视频重建损失
      action_loss = MSE(action_noise_pred, action_target)    动作预测损失
      loss = dynamics_loss + action_loss

─────────────────── 推理阶段 ───────────────────

【WANPolicyHead.lazy_joint_video_action】
  多步去噪循环（FlowUniPCMultistepScheduler）:
    for step in range(num_inference_steps):
        noise_pred = CausalWanModel._forward_inference(...)
        latents = scheduler.step(noise_pred, t, latents)

  最终输出:
    action_pred: (B, 48, 64)                               预测动作序列
    video_pred: (B, 49, 480, 1664, 3) uint8                预测视频（VAE 解码后）
```

### 4.2 维度变化总结图

```
输入图像 (B,49,2,480,832,3)
    │
    ▼ 拼接视角 + CHW
(B,3,49,480,1664)
    │
    ├──── VAE Encode ────► (B,16,13,60,208) latent
    │                           │
    │                           ▼ 加噪
    │                      (B,16,13,60,208) noisy latent
    │                           │
    ├──── T5 Encode ─────► (B,L,4096) text context
    │                           │
    ├──── CLIP Encode ───► (B,257,1280) image context
    │                           │
    │                           ▼
    │                    CausalWanModel (DiT)
    │                    ┌──────┴──────┐
    │                    ▼             ▼
    │            video_noise_pred  action_noise_pred
    │            (B,16,13,60,208)  (B,48,64)
    │                    │             │
    │                    ▼             ▼
    │              VAE Decode     反归一化
    │                    │             │
    │                    ▼             ▼
    │            video_pred       action_pred
    │         (B,49,480,1664,3)   (B,48,7)
    │              uint8          原始动作空间
    └──────────────────────────────────────
```

---

## 5. 渲染模块位置

DreamZero 的"渲染"即 **视频生成**，由以下模块协同完成：

| 功能 | 文件 | 核心类/函数 |
|------|------|-------------|
| **VAE 解码（latent → 像素）** | `groot/vla/model/dreamzero/modules/wan_video_vae.py` | `WanVideoVAE.decode()` |
| **扩散去噪（生成 latent）** | `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` | `CausalWanModel._forward_inference()` |
| **推理调度器** | `groot/vla/model/dreamzero/modules/flow_unipc_multistep_scheduler.py` | `FlowUniPCMultistepScheduler.step()` |
| **联合生成入口** | `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` | `WANPolicyHead.lazy_joint_video_action()` |
| **视频保存工具** | `groot/vla/common/utils/misc/video_utils.py` | 视频编解码工具函数 |
| **图像工具** | `groot/vla/common/utils/misc/image_utils.py` | 图像读写与预处理 |

**渲染流程**：
```
CausalWanModel (DiT 去噪)
    → 输出 video latent (B, 16, 13, 60, 208)
    → WanVideoVAE.decode(latent)
    → 输出 video pixels (B, 3, 49, 480, 1664)
    → 转为 uint8 (B, 49, 480, 1664, 3)
```

---

## 6. 单个文件核心类输入输出详解

### 6.1 数据模块 (`groot/vla/data/`)

#### `dataset/lerobot.py` — LeRobot 数据集加载

| 类 | 继承 | 说明 |
|----|------|------|
| `LeRobotSingleDataset` | `torch.utils.data.Dataset` | 单数据集，读取 parquet + 视频 |
| `CachedLeRobotSingleDataset` | `LeRobotSingleDataset` | 预缓存视频帧版本 |
| `LeRobotMixtureDataset` | `torch.utils.data.Dataset` | 多数据集混合 |

**`LeRobotSingleDataset.__init__`**:
- `dataset_path: str` — 数据集根目录
- `modality_configs: dict[str, ModalityConfig]` — 各模态的采样配置
- `embodiment_tag: EmbodimentTag` — embodiment 标识
- `transforms: list[ModalityTransform]` — 变换流水线
- `relative_action: bool` — 是否使用相对动作
- `fps: int`, `max_chunk_size: int` 等

**`__getitem__(index: int) → dict`**:
- 输出: `{"video.image_side_0": (T,H,W,3), "state.xxx": (T,D), "action.xxx": (T,D), "language": str}`

---

#### `dataset/lerobot_sharded.py` — 分片数据集

| 类 | 继承 | 说明 |
|----|------|------|
| `ShardedLeRobotSingleDataset` | `LeRobotSingleDataset` | 分片单数据集 |
| `ShardedLeRobotSubLangSingleActionChunkDatasetDROID` | `LeRobotSingleDataset` | DROID 语言对齐分片 |
| `ShardedLeRobotMixtureDataset` | `LeRobotMixtureDataset`, `IterableDataset` | 分片混合（训练用） |

**`ShardedLeRobotMixtureDataset.__iter__`**:
- 按分片迭代，每个分片预加载到内存
- yield 经 transform 处理的样本 dict

---

#### `dataset/registry.py` — 数据集注册表

- `EMBODIMENT_TAGS_TO_DATASET_PATHS: dict[EmbodimentTag, list[Path]]`
- `DATASET_PATHS_TO_EMBODIMENT_TAGS: dict[Path, EmbodimentTag]`

---

#### `transform/base.py` — 变换基类

| 类 | 说明 |
|----|------|
| `ModalityTransform` | 抽象基类，`apply(data: dict) → dict` |
| `InvertibleModalityTransform` | 可逆变换，额外提供 `unapply()` |
| `ComposedModalityTransform` | 组合多个变换，按顺序执行 |

---

#### `transform/video.py` — 视频变换

| 类 | 输入 | 输出 |
|----|------|------|
| `VideoCrop` | `(T, H, W, C)` | `(T, H', W', C)` |
| `VideoResize` | `(T, H, W, C)` | `(T, H_new, W_new, C)` |
| `VideoColorJitter` | `(T, C, H, W) float` | `(T, C, H, W) float` |
| `VideoToTensor` | `(T, H, W, C) uint8` | `(T, C, H, W) float32 [0,1]` |
| `VideoNormalize` | `(T, C, H, W) float` | `(T, C, H, W) float 归一化` |
| `VideoRandomErasing` | `(T, C, H, W)` | `(T, C, H, W)` 随机擦除 |

---

#### `transform/state_action.py` — 状态/动作变换

| 类 | 输入 | 输出 |
|----|------|------|
| `StateActionToTensor` | `ndarray (T, D)` | `Tensor (T, D)` |
| `RotationTransform` | `Tensor (T, D)` | `Tensor (T, D')` 旋转表示转换 |
| `Normalizer` | `Tensor (T, D)` | `Tensor (T, D)` 归一化 |
| `StateActionTransform` | `dict` | `dict` 旋转+归一化 |
| `StateActionPerturbation` | `Tensor (T, D)` | `Tensor (T, D)` 加噪 |
| `StateActionDropout` | `Tensor (T, D)` | `Tensor (T, D)` 随机置零 |

---

#### `transform/concat.py` — 多键合并

**`ConcatTransform.apply(data: dict) → dict`**:
- 视频: 多个 `video.xxx (T,H,W,C)` → `video (T, V, H, W, C)`，V=视角数
- 状态: 多个 `state.xxx (T,D_i)` → `state (T, sum(D_i))`
- 动作: 多个 `action.xxx (T,D_i)` → `action (T, sum(D_i))`

**`ConcatTransform.unapply(data: dict) → dict`**: 逆操作，拆回多个键

---

#### `transform/language.py` — 语言变换

| 类 | 输入 | 输出 |
|----|------|------|
| `LanguageTransform` | `str` | `input_ids (seq_len,) int64` |
| `LanguageRemovePrefix` | `str` | `str`（去掉 `"xxx: "` 前缀） |

---

#### `schema/lerobot.py` — 数据元数据定义

| 类 | 说明 |
|----|------|
| `DatasetMetadata` | 顶层元数据：statistics + modalities + embodiment_tag |
| `DatasetStatistics` | 统计量：mean/std/min/max/q01/q99 |
| `DatasetModalities` | 各模态元数据（video/state/action） |
| `LeRobotModalityMetadata` | LeRobot 格式的模态字段描述 |

---

#### `schema/embodiment_tags.py` — Embodiment 枚举

`EmbodimentTag` 枚举，包含：`OXE_DROID`, `REAL_GR1_ARMS_ONLY`, `UNITREE_G1_*`, `DREAM` 等。

---

### 6.2 模型模块 (`groot/vla/model/`)

#### `dreamzero/base_vla.py` — VLA 顶层模型

| 类 | 继承 | 说明 |
|----|------|------|
| `VLAConfig` | `PretrainedConfig` | 模型配置 |
| `VLA` | `PreTrainedModel` | 顶层 VLA 模型 |
| `CotrainVLA` | `VLA` | 联合训练版本 |

**`VLA.forward(inputs: dict) → BatchFeature`**:
- 输入: `{"video": (B,T,V,H,W,3), "state": (B,1,D), "action": (B,T_a,D), "input_ids": (B,L), ...}`
- 输出: `BatchFeature` 含 `loss`, `dynamics_loss`, `action_loss`

**`VLA.get_action(inputs: dict) → BatchFeature`**:
- 输出: `BatchFeature` 含 `action_pred (B, action_horizon, action_dim)`

**`VLA.lazy_joint_video_action(inputs: dict) → BatchFeature`**:
- 输出: `BatchFeature` 含 `action_pred (B, T_a, D)`, `video_pred (B, T, H, W, 3)`

---

#### `dreamzero/action_head/wan_flow_matching_action_tf.py` — 动作头

| 类 | 继承 | 说明 |
|----|------|------|
| `WANPolicyHeadConfig` | dataclass | 动作头配置 |
| `WANPolicyHead` | `ActionHead` | Flow Matching 动作头 |

**`WANPolicyHead.forward(backbone_output, action_input) → BatchFeature`**:
- 输入:
  - `backbone_output.backbone_features`: `(B, 1, 0)` （Identity backbone）
  - `action_input.videos`: `(B, 3, T, H, W)` float [-1,1]
  - `action_input.action`: `(B, T_a, D)`
  - `action_input.state`: `(B, 1, D)`
  - `action_input.input_ids`: `(B, L)`
- 输出: `loss`, `dynamics_loss`, `action_loss`

**`WANPolicyHead.encode_prompt(input_ids, attention_mask)`**:
- 输入: `(B, L)`, `(B, L)`
- 输出: `(B, L, 4096)` 文本 embedding

**`WANPolicyHead.encode_video(input_video)`**:
- 输入: `(B, 3, T, H, W)` → VAE → `(B, 16, T/4, H/8, W/8)`

**`WANPolicyHead.encode_image(image)`**:
- 输入: `(B, 3, 1, H, W)` → CLIP → `(B, 257, 1280)` + `(B, 512)`

---

#### `dreamzero/backbone/identity.py` — Identity Backbone

**`IdentityBackbone.forward(backbone_input) → BatchFeature`**:
- 输出: `backbone_features (B, 1, 0)` — 空特征，直接透传数据给 action head

---

#### `dreamzero/modules/wan_video_dit_action_casual_chunk.py` — 因果 DiT（核心）

| 类 | 说明 |
|----|------|
| `CausalWanModel` | 因果视频+动作联合扩散模型 |
| `MultiEmbodimentActionEncoder` | 多 embodiment 动作编码器 |
| `CausalWanSelfAttention` | 因果自注意力 |
| `CausalWanAttentionBlock` | 因果注意力块 |

**`CausalWanModel._forward_train(...)`**:
- 输入:
  - `x`: `(B, 16, T_latent, H_latent, W_latent)` — noisy 视频 latent
  - `timestep`: `(B,)` — 视频扩散时间步
  - `timestep_action`: `(B,)` — 动作扩散时间步
  - `context`: `(B, L, 4096)` — 文本条件
  - `action`: `(B, T_a, D)` — noisy 动作
  - `state`: `(B, 1, D)` — 状态条件
  - `embodiment_id`: `(B,)` — embodiment 标识
- 输出:
  - `video_noise_pred`: `(B, 16, T_latent, H_latent, W_latent)`
  - `action_noise_pred`: `(B, T_a, D)`

**`CausalWanModel._forward_inference(...)`**:
- 支持 KV cache，逐步生成
- 输出同上

---

#### `dreamzero/modules/wan_video_vae.py` — 视频 VAE

**`WanVideoVAE.encode(videos)`**:
- 输入: `(B, 3, T, H, W)` float [-1,1]
- 输出: `(B, 16, T/4, H/8, W/8)` latent

**`WanVideoVAE.decode(hidden_states)`**:
- 输入: `(B, 16, T/4, H/8, W/8)` latent
- 输出: `(B, 3, T, H, W)` float [-1,1]

支持 `tiled_encode` / `tiled_decode` 以节省显存。

---

#### `dreamzero/modules/wan_video_text_encoder.py` — 文本编码器

**`WanTextEncoder.forward(ids, mask)`**:
- 输入: `ids (B, L)`, `mask (B, L)`
- 输出: `(B, L, 4096)` T5 文本 embedding

---

#### `dreamzero/modules/wan_video_image_encoder.py` — 图像编码器

**`WanImageEncoder.encode_image(videos)`**:
- 输入: `(B, 3, T, H, W)`
- 输出: `(B, L, 1280)` CLIP 视觉特征

---

#### `dreamzero/modules/flow_match_scheduler.py` — Flow Match 调度器

**`FlowMatchScheduler`**:
- `add_noise(original, noise, timestep)`: 加噪 → 同形状
- `training_target(sample, noise, timestep)`: 训练目标 = `noise - sample`
- `step(model_output, timestep, sample)`: 单步去噪
- `training_weight(timestep)`: 时间步权重

---

#### `dreamzero/modules/flow_unipc_multistep_scheduler.py` — UniPC 多步调度器

**`FlowUniPCMultistepScheduler`**:
- 推理时使用，支持 UniP predictor + UniC corrector
- `step(model_output, timestep, sample, step_index)`: 多步去噪

---

#### `dreamzero/modules/attention.py` — 注意力实现

**`flash_attention(q, k, v, ...)`**:
- 输入: `q (B, Lq, Nq, C)`, `k (B, Lk, Nk, C)`, `v (B, Lk, Nk, C)`
- 输出: `(B, Lq, Nq, C)`
- 支持 Flash Attention 2/3 和 SDPA

---

#### `dreamzero/modules/wan2_1_attention.py` — Wan2.1 注意力模块

| 类 | 说明 |
|----|------|
| `WanSelfAttention` | 自注意力 |
| `WanT2VCrossAttention` | 文本→视频交叉注意力 |
| `WanI2VCrossAttention` | 图像→视频交叉注意力 |
| `WanAttentionBlock` | 完整注意力块（self + cross） |
| `WanModel` | Wan 扩散 backbone |

---

#### `dreamzero/modules/wan2_1_submodule.py` — 子模块

- `WanRMSNorm`, `WanLayerNorm`: 归一化
- `rope_params`, `rope_apply`: 旋转位置编码
- `sinusoidal_embedding_1d`: 时间步嵌入
- `MLPProj`: MLP 投影
- `Head`: 输出头

---

#### `dreamzero/modules/wan_video_camera_controller.py` — 相机控制

**`SimpleAdapter.forward(x)`**:
- 输入: `(B, C, F, H, W)` 相机控制 latent
- 输出: `(B, out_dim, F, H/8, W/8)` 适配后特征

---

#### `dreamzero/modules/vram_management.py` — 显存管理

- `enable_vram_management(model, ...)`: 递归包装模块，支持 CPU↔GPU offload
- `AutoWrappedModule.offload()` / `.onload()`: 按需迁移

---

#### `dreamzero/transform/dreamzero_cotrain.py` — 模型专用数据变换

**`DreamTransform`**:

| 方法 | 输入 | 输出 |
|------|------|------|
| `apply_single(data)` | 单样本 dict | 模型输入 dict |
| `apply_batch(data, batch_size)` | batch dict | 模型输入 batch dict |
| `_prepare_video(data)` | `video (T,V,H,W,C)` | `video (T,V,H,W,C)` 重排 |
| `_prepare_language(data)` | `language: str` | `input_ids (L,)`, `attention_mask (L,)` |
| `_prepare_state(data)` | `state (1,D)` | `state (1, max_state_dim)` padding |
| `_prepare_action(data)` | `action (T,D)` | `action (T, max_action_dim)` padding |

**`DefaultDataCollator.__call__(features)`**:
- 输入: `list[dict]`
- 输出: batch dict（stack 所有 tensor）

---

### 6.3 训练模块 (`groot/vla/experiment/`)

#### `experiment/base.py` — 训练基类

| 类 | 继承 | 说明 |
|----|------|------|
| `BaseTrainer` | `transformers.Trainer` | 训练器基类 |
| `BaseExperiment` | ABC | 实验基类 |
| `LossLoggerCallback` | `TrainerCallback` | Loss 日志回调 |
| `CheckpointFormatCallback` | `TrainerCallback` | Checkpoint 格式回调 |

**`BaseTrainer.training_step(model, inputs)`**:
- 输入: `model (VLA)`, `inputs (dict)`
- 输出: `loss_dict`

**`BaseExperiment`**:
- `create_model(cfg)` → VLA 实例
- `create_train_dataset(cfg, model)` → Dataset 实例
- `create_trainer(cfg, ...)` → Trainer 实例
- `train()` → 启动训练

---

#### `experiment/experiment.py` — VLA 训练入口

**`VLAExperiment(BaseExperiment)`**:
- `__init__(cfg: DictConfig)`: 从 Hydra 配置初始化
- `create_model()`: 实例化 VLA + 加载预训练权重 + LoRA
- `create_train_dataset()`: 实例化 ShardedLeRobotMixtureDataset
- `create_trainer()`: 实例化 VLATrainer

**`main(cfg: DictConfig)`**: Hydra 入口，创建 VLAExperiment 并调用 `.train()`

---

### 6.4 评估与推理模块

#### `socket_test_optimized_AR.py` — AR 策略服务（`dreamzero-server`）

| 类 | 说明 |
|----|------|
| `ARDroidRoboarenaPolicy` | 将 roboarena 观测转为 AR_droid 格式 |

**`ARDroidRoboarenaPolicy.infer(obs: dict) → np.ndarray`**:
- 输入: `obs` 含 `images (dict[str, ndarray])`, `joint_positions`, `instruction`
- 输出: `(N, 8)` 动作序列（6 关节 + 1 夹爪 + 1 终止标志）

---

#### `eval_utils/run_sim_eval.py` — 仿真评估

**`DreamZeroJointPosClient.infer(obs, instruction) → dict`**:
- 输入: `obs` 含图像、关节位置等
- 输出: `{"action": ndarray, "viz": dict}` 动作 + 可视化数据

---

#### `n1_5/sim_policy.py` — 仿真策略封装

| 类 | 说明 |
|----|------|
| `GrootSimPolicy` | 完整 VLA 推理策略 |
| `GrootSimRLPolicy` | 离线 RL 策略 |

**`GrootSimPolicy.forward(batch) → dict`**:
- 输入: batch dict（经 `apply()` 归一化）
- 输出: `{"action_pred": (B, T_a, D)}` 反归一化后的动作

**`GrootSimPolicy.lazy_joint_forward(batch) → dict`**:
- 输出: `{"action_pred": ..., "video_pred": ...}` 动作 + 视频

---

### 6.5 配置文件

| 配置文件 | 作用 |
|----------|------|
| `configs/conf.yaml` | Hydra 主配置：trainer、training_args、wandb |
| `configs/data/dreamzero/droid_relative.yaml` | DROID 数据配置：路径、混合、相对动作 |
| `configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml` | 基础数据配置：帧数、分辨率、增强 |
| `configs/model/dreamzero/vla.yaml` | VLA 模型配置：backbone + action_head + transform |
| `configs/model/dreamzero/action_head/wan_flow_matching_action_tf.yaml` | 动作头配置：DiT、VAE、编码器、LoRA |
| `configs/model/dreamzero/backbone/identity.yaml` | Backbone 配置：Identity |
| `configs/model/dreamzero/transform/dreamzero_cotrain.yaml` | 数据变换配置：tokenizer、collator |

---

### 6.6 工具模块

| 文件 | 核心功能 |
|------|----------|
| `groot/vla/utils/timer.py` | `ContextTimer` 上下文计时器 |
| `groot/vla/utils/action_args_override_utils.py` | 动态覆盖 action_horizon / action_dim 配置 |
| `groot/vla/common/utils/misc/video_utils.py` | 视频读取（decord/ffmpeg/opencv/torchcodec） |
| `groot/vla/common/utils/misc/image_utils.py` | 图像读写、预处理、显示 |
| `groot/vla/common/utils/misc/torch_utils.py` | PyTorch 工具 |
| `groot/vla/common/utils/io/config_utils.py` | 配置加载工具 |
| `groot/vla/common/utils/io/hdf5_utils.py` | HDF5 读写 |
| `groot/vla/common/utils/data_structure/shape_utils.py` | Shape 工具 |
| `groot/vla/common/utils/data_structure/tree_utils.py` | 嵌套字典树操作 |
| `scripts/data/convert_droid.py` | DROID RLDS → LeRobot 格式转换 |
| `scripts/compare_loss.py` | LoRA vs 全量微调 Loss 对比 |
