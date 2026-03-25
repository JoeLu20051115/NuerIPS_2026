#!/usr/bin/env bash
# 每5分钟检查一次 GPU 0 的空余内存
# 超过 40000 MiB 就自动启动评估，否则继续等

THRESHOLD_MIB=40000
REPO="/home/xingrui/lueq/NuerIPS_2026"
LOG="${REPO}/logs/wait_and_run.log"
mkdir -p "${REPO}/logs"

echo "[$(date)] Waiting for GPU 0 to have >= ${THRESHOLD_MIB} MiB free..." | tee -a "${LOG}"

while true; do
    FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
    echo "[$(date)] GPU 0 free: ${FREE_MIB} MiB (need ${THRESHOLD_MIB})" | tee -a "${LOG}"

    if [ "${FREE_MIB}" -ge "${THRESHOLD_MIB}" ] 2>/dev/null; then
        echo "[$(date)] Enough memory! Starting evaluation..." | tee -a "${LOG}"
        bash "${REPO}/run_timed_eval_overnight.sh" 2>&1 | tee -a "${LOG}"
        echo "[$(date)] Evaluation script finished." | tee -a "${LOG}"
        break
    fi

    sleep 300  # check every 5 minutes
done
