# π₀.₅ LIBERO-Long 复现任务与结果

## 1. 当前任务信息

本项目的目标是在 LIBERO-Long（对应官方 `libero_10`）上复现两个
π₀.₅ checkpoint：

1. 完整 π₀.₅ LIBERO checkpoint，公开总体参考成功率为 92.4%。
2. 早期 π₀.₅（COAST，2,000 training steps），公开总体参考成功率为 43%。

每个 checkpoint 在 10 个任务上各运行 50 次：

```text
10 tasks × 50 trials = 500 episodes/checkpoint
2 checkpoints × 500 episodes = 1,000 episodes total
```

当前状态：**已完成，总体复现验收 PASS**。

## 2. 模型、数据与源码

| 项目 | 来源或版本 |
| --- | --- |
| 完整 π₀.₅ 权重 | `gs://openpi-assets/checkpoints/pi05_libero/` |
| 早期 π₀.₅ 权重 | `brandonyang/openpi-libero-2000@aaeeabc72f8a50a8fa2d04544332c8ec1cd0142e` |
| OpenPI 源码 | commit `650c5b0283a49c42784fb5055a0507da2c6d347d` |
| LIBERO 源码 | commit `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |
| LIBERO-10 数据 | `yifengzhu-hf/LIBERO-datasets`, `libero_10` |
| 仿真环境 | MuJoCo 3.2.3 + robosuite 1.4.1 + BDDL 1.0.1 |
| 容器镜像 | `pi05-libero-eval:650c5b0` |

两个 checkpoint 都使用锁定的 `pi05_libero` 架构：Gemma 2B PaliGemma
backbone、Gemma 300M action expert、bfloat16、10-step action horizon、连续机器人
状态输入和 32 维内部 action padding，LIBERO 输出取前 7 维。

## 3. 评测协议

- 随机种子为 7。
- 每个任务使用 LIBERO 前 50 个固定初始状态。
- 每个 checkpoint 使用一个不中断的 OpenPI policy 进程。
- 输入包含 agent-view RGB、wrist-view RGB 和官方 8D 机器人状态。
- LIBERO 原生 256 像素渲染，官方 180° 旋转后 resize/pad 到 224×224。
- reset 后执行 10 个 dummy stabilization steps。
- policy horizon 为 10，每执行 5 个 action 重新规划一次。
- 每个 episode 最多执行 520 个 policy-controlled steps。
- 只使用 LIBERO 环境原生 `done` 和 `check_success` 判定成功。

## 4. 10 个任务的具体结果

| # | LIBERO-Long 任务 | 完整 π₀.₅ | 早期 π₀.₅ | 早期公开参考 |
| ---: | --- | ---: | ---: | ---: |
| 1 | Cream cheese + Butter → Basket | **50/50 (100.0%)** | **36/50 (72.0%)** | 60% |
| 2 | Black bowl → Bottom drawer + Close | **48/50 (96.0%)** | **20/50 (40.0%)** | 40% |
| 3 | Turn on stove + Moka pot | **46/50 (92.0%)** | **22/50 (44.0%)** | 53% |
| 4 | Two mugs → Left/Right plates | **49/50 (98.0%)** | **8/50 (16.0%)** | 7% |
| 5 | Both moka pots → Stove | **27/50 (54.0%)** | **5/50 (10.0%)** | 20% |
| 6 | Alphabet soup + Cream cheese → Basket | **50/50 (100.0%)** | **28/50 (56.0%)** | 80% |
| 7 | Mug → Microwave + Close | **47/50 (94.0%)** | **2/50 (4.0%)** | 13% |
| 8 | Alphabet soup + Tomato sauce → Basket | **47/50 (94.0%)** | **27/50 (54.0%)** | 40% |
| 9 | Book → Back compartment of caddy | **49/50 (98.0%)** | **38/50 (76.0%)** | 67% |
| 10 | Mug → Plate + Chocolate pudding right of plate | **47/50 (94.0%)** | **32/50 (64.0%)** | 53% |
| — | **总体** | **460/500 (92.0%)** | **218/500 (43.6%)** | **43%** |

## 5. 总体结论

| Checkpoint | 复现结果 | 参考值 | 绝对差距 | 验收区间 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 完整 π₀.₅ | 92.0% | 92.4% | -0.4 个百分点 | 89.4%–95.4% | PASS |
| 早期 π₀.₅ | 43.6% | 43.0% | +0.6 个百分点 | 38.0%–48.0% | PASS |

早期 checkpoint 公开的单任务数字是粗粒度小样本参考，因此不要求
每个 50-trial 单项精确相等。本次请求的 500-trial 总体结果为 43.6%，
与公开 43% 参考仅相差 0.6 个百分点。

## 6. 完整性审计

- 两个 JSONL 日志各有 500 条唯一且有效的 episode 记录。
- 1,000 个 episode 全部满足 `success == done == check_success`。
- 不存在 `invalid.json`。
- 1,000 个视频均为 H.264、224×224，解码帧数与对应控制步数一致。
- 两个 checkpoint 的 500 对任务/初始状态/首帧哈希完全一致。
- 仓库自动化测试结果为 31 passed。
- checkpoint、数据集和两组主评测 SHA-256 manifest 全部校验通过。

## 7. 结果和产物位置

| 产物 | 项目内路径 |
| --- | --- |
| 机器可读结果 | `results/pi05-libero-long-summary.json` |
| 精简逐任务报告 | `results/pi05-libero-long-summary.md` |
| 最终协议与完整性审计 | `results/pi05-libero-long-audit.md` |
| 完整模型原始记录 | `runs/primary-full/episodes.jsonl` |
| 早期模型原始记录 | `runs/primary-early/episodes.jsonl` |
| 完整模型运行清单 | `artifacts/manifests/primary-full.json` |
| 早期模型运行清单 | `artifacts/manifests/primary-early.json` |

大型权重、HDF5 数据、视频和原始日志保留在当前工作区，不直接提交到
Git；它们由已提交的 SHA-256 manifests 锁定。
