# AgiBot WorldModel Local Eval

Official local evaluation flow used here:

1. Run baseline inference to generate `*_dataset`.
2. Resize all predicted frames to `(480, 640)` before evaluation.
3. Run EWMBench preprocessing to produce gripper detections and trajectories.
4. Evaluate with:
   - `psnr`
   - `scene_consistency`
   - `trajectory_consistency` (contains `ndtw`)

Key local paths:

- Baseline repo:
  `/home/xingrui/lueq/NuerIPS_2026/external_repos/AgiBotWorldChallengeICRA2026-WorldModelBaseline`
- EWMBench repo:
  `/home/xingrui/lueq/NuerIPS_2026/external_repos/EWMBench`
- Combined eval requirements:
  `/home/xingrui/lueq/NuerIPS_2026/requirements_agibot_eval.txt`
- Dedicated eval env:
  `/home/xingrui/lueq/NuerIPS_2026/.venv_agibot_eval`

Local config files prepared:

- Smoke config:
  `/home/xingrui/lueq/NuerIPS_2026/external_repos/EWMBench/config_local_smoke.yaml`
- Full validation config:
  `/home/xingrui/lueq/NuerIPS_2026/external_repos/EWMBench/config_local_val.yaml`

Typical commands:

```bash
/home/xingrui/lueq/NuerIPS_2026/.venv_agibot_eval/bin/python \
  /home/xingrui/lueq/NuerIPS_2026/external_repos/EWMBench/processing/video_resize.py \
  --config_path /home/xingrui/lueq/NuerIPS_2026/external_repos/EWMBench/config_local_val.yaml

/home/xingrui/lueq/NuerIPS_2026/.venv_agibot_eval/bin/python \
  /home/xingrui/lueq/NuerIPS_2026/external_repos/EWMBench/processing/detection_tracking.py \
  --config_path /home/xingrui/lueq/NuerIPS_2026/external_repos/EWMBench/config_local_val.yaml

/home/xingrui/lueq/NuerIPS_2026/.venv_agibot_eval/bin/python \
  /home/xingrui/lueq/NuerIPS_2026/external_repos/EWMBench/evaluate.py \
  --dimension scene_consistency trajectory_consistency psnr \
  --config_path /home/xingrui/lueq/NuerIPS_2026/external_repos/EWMBench/config_local_val.yaml
```
