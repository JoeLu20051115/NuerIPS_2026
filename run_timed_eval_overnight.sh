#!/usr/bin/env bash
set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
REPO="/home/xingrui/lueq/NuerIPS_2026"
VENV_PYTHON="${REPO}/.venv/bin/python"
MODEL_PATH="${REPO}/checkpoints/DreamZero-DROID"
PORT=8000
GPU=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATE_SUFFIX=$(date +%Y%m%d)
GEN_VIDEO_DIR="${REPO}/checkpoints/real_world_eval_gen_${DATE_SUFFIX}_0/DreamZero-DROID"

LOG_DIR="${REPO}/logs/timed_eval_${DATE_SUFFIX}"
mkdir -p "${LOG_DIR}" "${GEN_VIDEO_DIR}"

EVAL_SCRIPT="${REPO}/scripts/eval/run_compare_tasktoken_dual_judged.py"
RESULT_DIR="${REPO}/evaluation_results_dualsystem"

echo "[$(date)] Starting overnight timed evaluation"
echo "  GPU          : ${GPU}"
echo "  model        : ${MODEL_PATH}"
echo "  gen_video_dir: ${GEN_VIDEO_DIR}"
echo "  logs         : ${LOG_DIR}"

# ── 1. start policy server ────────────────────────────────────────────────────
echo ""
echo "[$(date)] Starting policy server on GPU ${GPU}, port ${PORT}..."

CUDA_VISIBLE_DEVICES=${GPU} nohup "${VENV_PYTHON}" "${REPO}/socket_test_optimized_AR.py" \
    --model-path "${MODEL_PATH}" \
    --port ${PORT} \
    --embodiment-tag oxe_droid \
    > "${LOG_DIR}/server.log" 2>&1 &
SERVER_PID=$!
echo "[$(date)] Policy server PID: ${SERVER_PID}"

# ── 2. wait for server ready (port open) ─────────────────────────────────────
echo "[$(date)] Waiting for port ${PORT} to be ready..."
for i in $(seq 1 180); do
    if python3 -c "
import socket, sys
s = socket.socket()
s.settimeout(2)
try:
    s.connect(('localhost', ${PORT}))
    s.close()
    sys.exit(0)
except:
    sys.exit(1)
" 2>/dev/null; then
        echo "[$(date)] Server ready after ${i}s"
        break
    fi
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "[$(date)] ERROR: server process died. Check ${LOG_DIR}/server.log"
        exit 1
    fi
    sleep 5
done

# Extra buffer for model weights to fully load
sleep 10

# ── 3. run L1 evaluation (150 episodes, 3 modes) ─────────────────────────────
echo ""
echo "[$(date)] === Starting L1 evaluation (150 episodes x 3 modes) ==="

cd "${REPO}/scripts/eval"
"${VENV_PYTHON}" "${EVAL_SCRIPT}" \
    --dataset-root "${REPO}/data/final_data/DRO_L1_150" \
    --generated-video-dir "${GEN_VIDEO_DIR}" \
    --num-episodes 150 \
    --eval-steps 10 \
    --judge-model gpt-4o-mini \
    --success-threshold 0.7 \
    --log-json "${RESULT_DIR}/DRO_L1_150_timed_compare.json" \
    2>&1 | tee "${LOG_DIR}/L1_eval.log"

echo "[$(date)] === L1 done ==="

# ── 4. run L3 evaluation (150 episodes, 3 modes) ─────────────────────────────
echo ""
echo "[$(date)] === Starting L3 evaluation (150 episodes x 3 modes) ==="

"${VENV_PYTHON}" "${EVAL_SCRIPT}" \
    --dataset-root "${REPO}/data/final_data/DRO_L3_150" \
    --generated-video-dir "${GEN_VIDEO_DIR}" \
    --num-episodes 150 \
    --eval-steps 10 \
    --judge-model gpt-4o-mini \
    --success-threshold 0.7 \
    --log-json "${RESULT_DIR}/DRO_L3_150_timed_compare.json" \
    2>&1 | tee "${LOG_DIR}/L3_eval.log"

echo ""
echo "[$(date)] === All done. Results saved to ${RESULT_DIR}/ ==="
echo "  DRO_L1_150_timed_compare.json"
echo "  DRO_L3_150_timed_compare.json"

# Print quick summary
python3 - <<'EOF'
import json
from pathlib import Path

result_dir = Path("/home/xingrui/lueq/NuerIPS_2026/evaluation_results_dualsystem")
for name in ["DRO_L1_150_timed_compare.json", "DRO_L3_150_timed_compare.json"]:
    p = result_dir / name
    if not p.exists():
        continue
    data = json.loads(p.read_text())
    summary = data.get("summary", {})
    print(f"\n=== {name} ===")
    for mode, stats in summary.items():
        print(f"  {mode}: success_rate={stats.get('success_rate',0):.3f} | "
              f"mean_l2={stats.get('mean_l2') or 0:.4f} | "
              f"mean_progress={stats.get('mean_task_progress') or 0:.3f}")
EOF
