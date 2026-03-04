# DreamZero: World Action Models Are Zero-Shot Policies
[[Project Page](https://dreamzero0.github.io/)] [[Paper](https://arxiv.org/abs/2602.15922)]

<!-- 
[中文概述]
DreamZero 是一个 World Action Model（世界动作模型），能够同时预测动作和视频，
在未见过的任务上实现强大的零样本（zero-shot）性能。
本仓库包含：预训练模型加载、分布式 WebSocket 推理服务、LoRA/全量微调训练脚本。

整体流程：安装环境 → 下载权重 → 启动推理服务 → 用客户端测试 → （可选）训练微调
-->

DreamZero is a World Action Model that jointly predicts actions and videos, achieving strong zero-shot performance on unseen tasks. This release package contains everything needed to load a pretrained DreamZero model and run distributed inference via a WebSocket server.

## Features

<!-- 
[中文] 当前已发布的功能列表：
- 预训练 DROID 模型权重
- 基于 WebSocket 的分布式推理服务（支持 GB200 / H100 GPU）
- DiT 缓存加速推理（GB200 约 0.6s，H100 约 3s）
- DROID 仿真评测
- RoboArena 真实机器人集成
- 视频生成并保存为 MP4
- LoRA 训练脚本 + 全量微调
-->

**Available Now**
- Pretrained DreamZero-DROID model checkpoint
- Distributed WebSocket inference server (GB200, H100)
- DiT caching for optimized inference (~0.6s on GB200, ~3s on H100)
- DROID simulation evaluation support
- [RoboArena](https://robo-arena.github.io/) integration (DROID real)
- Video generation and saving (MP4)
- **New 02/16:** LoRA training script for DreamZero, with preprocessed DROID dataset
- **New 02/20:** Full fine-tuning support
  
**Coming Soon**
- DreamZero training script on new embodiment (e.g. YAM)
- [PolaRiS](https://polaris-evals.github.io/) simulation environment support
- [Genie 3.0](https://arxiv.org/abs/2601.02078) sim environment support

## Testing Out DreamZero in Simulation with API

<!-- 
[中文] 最快体验路线（无需本地 GPU）：
1. 先通过表单申请 API 访问权限
2. 克隆 sim-evals 仓库并安装依赖
3. 下载仿真环境资源
4. 运行 eval 脚本，连接远程 API 进行评测
结果保存在 runs/ 目录下
-->

We provide an inference script that directly evaluates a hosted DreamZero-DROID policy on [`sim_evals`](https://github.com/arhanjain/sim-evals). To test out the policy, first request access to the API via this form [link](https://forms.gle/zCj5zjDvHsoeuMXU7). Then, follow these instructions to install [`sim_evals`](https://github.com/arhanjain/sim-evals) and launch evaluation.

```bash
# 克隆仓库（含子模块）
git clone --recurse-submodules https://github.com/arhanjain/sim-evals.git
cd sim-evals

# 安装 uv 包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh

# 用 uv 同步依赖并激活虚拟环境
uv sync
source .venv/bin/activate

# [可选] 更新 PyTorch 版本到 CUDA 12.9
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129

# 下载仿真环境资源（可能需要先 export HF_TOKEN=<你的HuggingFace Token>）
uvx hf download owhan/DROID-sim-environments --repo-type dataset --local-dir assets

# 运行评测脚本，<API_HOST> 和 <API_PORT> 替换为申请到的 API 地址
cd ..
python eval_utils/run_sim_eval.py --host <API_HOST> --port <API_PORT> 
```

The outputs are saved in `runs` directory.

## Running Evaluation and Analysis

### Step 1: Start Inference Server

```bash
# 启动推理服务（在一个终端）
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --standalone --nproc_per_node=1 \
  socket_test_optimized_AR.py --port 5000 --enable-dit-cache \
  --model-path /home/xingrui/lueq/NuerIPS_2026/checkpoints/DreamZero-DROID
```

### Step 2: Run Evaluation

```bash
# 在另一个终端运行评估（连接到推理服务）
python eval_utils/run_sim_eval.py --host localhost --port 5000 --episodes 10 --scene 1
```

Evaluation results (video files) will be saved in `runs/YYYY-MM-DD/HH-MM-SS/` directory.

### Step 3: Analyze Results

#### Basic Analysis (Success Rate & Time Horizon)

```bash
# 分析评估结果，生成准确率和时间范围图表
python eval_utils/analyze_results.py --runs-dir runs/2024-01-01/12-00-00
```

Generates:
- `success_rate_metrics.png`: Episode success rates
- `time_horizon_analysis.png`: Short-term vs long-term time horizons
- `evaluation_report.txt`: Text summary

#### "Short-Sightedness" Bottleneck Analysis

```bash
# 分析 DreamZero 的"短视"瓶颈（误差级联、语义断裂、单一指令瓶颈）
python eval_utils/analyze_short_sightedness.py --runs-dir runs/2024-01-01/12-00-00
```

Generates comprehensive visualizations for three core bottlenecks:

1. **Error Cascading (误差级联)**: Cumulative error growth, chunk distribution
2. **Semantic Disconnection (语义断裂)**: Instruction complexity vs completion rate
3. **Single Instruction Bottleneck (单一指令瓶颈)**: Ideal vs actual instruction switches

Output files:
- `error_cascading_analysis.png`
- `semantic_disconnection_analysis.png`
- `single_instruction_bottleneck_analysis.png`
- `short_sightedness_analysis_report.txt`

## Quick Start

<!-- 
[中文] 本地部署的完整流程（需要多卡 GPU）：
  前置条件 → 安装依赖 → 下载权重 → 启动服务 → 测试
-->

### Prerequisites

- **Python**: 3.11
- **Hardware**: GPU setup (tested on GB200, H100)
  - Single GPU is sufficient for inference
- **CUDA**: Compatible GPU with CUDA 12.9+

### Installation

<!-- 
[中文] 安装步骤：
  第 1 步：创建 conda 虚拟环境
  第 2 步：以可编辑模式安装本项目及其依赖（自动从 PyTorch CUDA 12.9 源拉取）
  第 3 步：编译安装 flash-attn（加速注意力计算）
  第 4 步：仅 GB200 需要安装 Transformer Engine，H100 跳过
-->

1. **Create conda environment:**
```bash
conda create -n dreamzero python=3.11
conda activate dreamzero
```

2. **Install dependencies (PyTorch 2.8+ with CUDA 12.9+):**
```bash
# 以可编辑模式安装，依赖会从 setup.py/pyproject.toml 自动解析
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129
```

3. **Install flash attention:**
```bash
# MAX_JOBS=8 控制编译并行度，可根据 CPU 核数调整
MAX_JOBS=8 pip install --no-build-isolation flash-attn
```

4. **[GB200 ONLY, SKIP FOR H100] Install Transformer Engine:**
```bash
# 仅在 GB200 GPU 上需要，H100 用户请跳过此步
pip install --no-build-isolation transformer_engine[pytorch]
```

## Downloading the Pretrained Checkpoint

<!-- 
[中文] 下载预训练权重：
- 模型大小：14B 参数
- 来源：HuggingFace 上的 GEAR-Dreams/DreamZero-DROID
- 下载后将 <path/to/checkpoint> 替换为你的实际存储路径
- 后续启动推理服务时需要用 --model-path 指向这个路径
-->

We release a 14B pretrained DROID checkpoint on [Huggingface](https://huggingface.co/GEAR-Dreams/DreamZero-DROID). To download the checkpoint, run

```bash
# 将 <path/to/checkpoint> 替换为你想保存权重的本地路径，例如 ./checkpoints/DreamZero-DROID
hf download GEAR-Dreams/DreamZero-DROID --repo-type model --local-dir /home/xingrui/lueq/NuerIPS_2026/checkpoints/DreamZero-DROID
```

## Running the Inference Server

<!-- 
[中文] 推理服务说明：
- 使用 PyTorch 的 torch.distributed.run 在多卡上并行加载模型
- 通过 WebSocket 对外提供推理接口
- 建议开启 --enable-dit-cache 以加速推理
- 首次推理需要几分钟预热，之后速度稳定（GB200 ~0.6s, H100 ~3s）
-->

### Command Overview

```bash
# 启动推理服务
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --standalone --nproc_per_node=1 \
  socket_test_optimized_AR.py --port 5000 --enable-dit-cache \
  --model-path /home/xingrui/lueq/NuerIPS_2026/checkpoints/DreamZero-DROID
```

The first few inferences will take a few minutes to warm up. After warming up, inference takes ~0.6s on GB200 and ~3s on H100.

### Command-line Arguments

- `--port`: WebSocket server port (default: 8000)
- `--model-path`: Path to pretrained model checkpoint (required)
- `--enable-dit-cache`: Enable DiT layer caching for faster inference (recommended)
- `--max-chunk-size`: Override max_chunk_size for inference (optional)
- `--timeout-seconds`: Server timeout in seconds (default: 50000)




## Training

<!-- 
[中文] 训练部分概述：
如果你想在 DROID 数据集上微调 DreamZero，需要：
1. 下载基础模型权重（Wan2.1-I2V-14B-480P）和分词器（umt5-xxl）
2. 下载预处理好的 DROID 数据集（约 131GB）
3. 配置环境变量并运行训练脚本

训练使用 Hydra 管理配置，DeepSpeed ZeRO Stage 2 进行分布式训练。
默认 max_steps=10 仅用于快速验证，正式训练需要调大。
-->

### Downloading Pretrained Base Model Weights

<!-- 
[中文] DreamZero 基于 Wan2.1-I2V-14B-480P 视频生成模型构建，
使用 umt5-xxl 作为文本分词器。训练前需要先下载这两个：
- Wan2.1 模型权重约 28GB
- 如果不手动下载，训练脚本会自动下载，但建议提前下好避免启动延迟
-->

DreamZero is built on top of [Wan2.1-I2V-14B-480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) and uses the [umt5-xxl](https://huggingface.co/google/umt5-xxl) tokenizer. Download both before training:

```bash
pip install "huggingface_hub[cli]"

# 如果需要认证，先设置 HuggingFace Token：
# export HF_TOKEN=<你的HuggingFace Token>

# 下载 Wan2.1 基础模型权重（约 28GB）
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P

# 下载 umt5-xxl 分词器
hf download google/umt5-xxl --local-dir ./checkpoints/umt5-xxl
```

> **Note:** The training script will auto-download these if they are not found at the configured paths, but pre-downloading is recommended to avoid delays at launch.

### DROID Dataset

<!-- 
[中文] DROID 数据集说明：
- 基于 DROID 1.0.1 数据集预处理而来
- 格式：从 RLDS/TFDS 转换为 LeRobot v2.0 格式
- 过滤规则：去除空闲帧、无语言标注的片段、失败片段
- 包含 3 个相机视角：外部左1、外部左2、腕部左
- 总大小约 131GB
- 如需自行从原始数据转换，参考 docs/DROID_CONVERSION.md
-->

We release the preprocessed DROID dataset used to train DreamZero on HuggingFace: [GEAR-Dreams/DreamZero-DROID-Data](https://huggingface.co/datasets/GEAR-Dreams/DreamZero-DROID-Data).

This dataset is derived from the [DROID 1.0.1](https://droid-dataset.github.io/) dataset with the following modifications:
- Converted from RLDS/TFDS format to [LeRobot](https://github.com/huggingface/lerobot) v2.0 format
- Idle frames removed using [Physical Intelligence's idle frame detector](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md#data-filtering) (`droid_sample_ranges_v1_0_1.json`)
- Episodes without language annotations are filtered out
- Successful episodes only (episodes with non-zero reward)
- 3 camera views: `exterior_image_1_left`, `exterior_image_2_left`, `wrist_image_left`

**To download the preprocessed dataset (~131GB):**

```bash
# 下载预处理好的 DROID 数据集到本地 ./data/droid_lerobot 目录
huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir ./data/droid_lerobot
```

If you want to reproduce the dataset conversion from raw DROID 1.0.1 yourself (or modify the filtering), see [docs/DROID_CONVERSION.md](docs/DROID_CONVERSION.md).

### Running Training

<!-- 
[中文] 启动训练：
1. 设置环境变量指定数据路径、输出路径、GPU 数量、模型权重路径
2. 运行 scripts/train/droid_training.sh 脚本
-->

```bash
# 配置路径（根据实际情况修改）
export DROID_DATA_ROOT="./data/droid_lerobot"      # DROID 数据集路径
export OUTPUT_DIR="./checkpoints/dreamzero_droid"   # 训练输出保存路径
export NUM_GPUS=4                                   # 使用的 GPU 数量

# 指定基础模型权重路径（如果没放在默认位置）
export WAN_CKPT_DIR="./checkpoints/Wan2.1-I2V-14B-480P"  # Wan2.1 模型权重
export TOKENIZER_DIR="./checkpoints/umt5-xxl"             # umt5-xxl 分词器

# 启动训练
bash scripts/train/droid_training.sh
```

### Training Configuration

<!-- 
[中文] 训练配置说明（使用 Hydra + DeepSpeed ZeRO Stage 2）：
- NUM_GPUS=4              GPU 数量
- batch_size=1            每张卡的 batch size
- learning_rate=1e-5      学习率
- max_steps=10            最大训练步数（默认 10 步仅用于快速验证，正式训练需调大）
- image: 320×176          输入图像分辨率
- num_frames=33           视频帧数
- action_horizon=24       动作预测步长
- save_lora_only=true     默认只保存 LoRA 权重（节省空间）
- bf16=true               使用 bfloat16 混合精度
-->

The training script uses Hydra for configuration and DeepSpeed ZeRO Stage 2 for distributed training. Key defaults:

| Parameter | Default | Description |
|---|---|---|
| `NUM_GPUS` | 4 | Number of GPUs |
| `per_device_train_batch_size` | 1 | Batch size per GPU |
| `learning_rate` | 1e-5 | Learning rate |
| `max_steps` | 10 | Max training steps (increase for full training) |
| `warmup_ratio` | 0.05 | Warmup ratio |
| `weight_decay` | 1e-5 | Weight decay |
| `image_resolution_width` | 320 | Image width |
| `image_resolution_height` | 176 | Image height |
| `num_frames` | 33 | Number of video frames |
| `action_horizon` | 24 | Action prediction horizon |
| `save_lora_only` | true | Only save LoRA weights |
| `bf16` | true | Use bfloat16 precision |

> **Note:** `max_steps=10` is set for a quick sanity check. For full training, increase this to your desired number of steps and configure `save_steps` / `save_strategy` accordingly.


## Citation

If you use DreamZero in your research, please cite:

```bibtex
@misc{ye2026worldactionmodelszeroshot,
      title={World Action Models are Zero-shot Policies}, 
      author={Seonghyeon Ye and Yunhao Ge and Kaiyuan Zheng and Shenyuan Gao and Sihyun Yu and George Kurian and Suneel Indupuru and You Liang Tan and Chuning Zhu and Jiannan Xiang and Ayaan Malik and Kyungmin Lee and William Liang and Nadun Ranawaka and Jiasheng Gu and Yinzhen Xu and Guanzhi Wang and Fengyuan Hu and Avnish Narayan and Johan Bjorck and Jing Wang and Gwanghyun Kim and Dantong Niu and Ruijie Zheng and Yuqi Xie and Jimmy Wu and Qi Wang and Ryan Julian and Danfei Xu and Yilun Du and Yevgen Chebotar and Scott Reed and Jan Kautz and Yuke Zhu and Linxi "Jim" Fan and Joel Jang},
      year={2026},
      eprint={2602.15922},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.15922}, 
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Support

For issues and questions:
- Check the troubleshooting section above
- Review server logs for detailed error messages
- Verify your checkpoint is compatible with this release

[![Star History Chart](https://api.star-history.com/svg?repos=dreamzero0/dreamzero&type=Date)](https://star-history.com/#dreamzero0/dreamzero&Date)
