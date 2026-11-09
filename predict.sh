#!/bin/bash
# ==============================================================================
# DATAMATE - predict.sh: doc /data, ghi /result/submission.jsonl, log tung cau.
# Chay moi GPU 1 instance vLLM (2 x RTX 5090 => 2 instance, cong 8000/8001).
# ==============================================================================
set -uo pipefail

INPUT_DIR="${INPUT_DIR:-/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/result}"
OUTPUT_FILE="${OUTPUT_DIR}/submission.jsonl"
LOG_FILE="${OUTPUT_DIR}/execution.log"
MODEL_DIR="${MODEL_DIR:-/workspace/models/Qwen3.5-9B}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"   # giay cho vLLM khoi dong

mkdir -p "${OUTPUT_DIR}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}"; }

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "${NGPU}" -ge 1 ] || NGPU=1
[ "${NGPU}" -le 2 ] || NGPU=2
log "Starting DataMate offline evaluation with ${NGPU} GPU(s)"

PIDS=()
ENDPOINTS=()
cleanup() {
    for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT

for ((i = 0; i < NGPU; i++)); do
    port=$((8000 + i))
    CUDA_VISIBLE_DEVICES=$i python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_DIR}" \
        --host 127.0.0.1 --port "${port}" \
        --dtype bfloat16 \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90 \
        --seed 0 \
        --served-model-name qwen3.5-9b \
        >> "${OUTPUT_DIR}/vllm_gpu${i}.log" 2>&1 &
    PIDS+=($!)
    ENDPOINTS+=("http://127.0.0.1:${port}/v1")
done

# Cho tung server san sang; dung ngay neu process chet hoac qua timeout.
for idx in "${!ENDPOINTS[@]}"; do
    ep="${ENDPOINTS[$idx]}"
    waited=0
    until curl -sf "${ep}/models" > /dev/null; do
        if ! kill -0 "${PIDS[$idx]}" 2>/dev/null; then
            log "ERROR: vLLM (GPU ${idx}) exited early, see vllm_gpu${idx}.log"; exit 1
        fi
        if [ "${waited}" -ge "${READY_TIMEOUT}" ]; then
            log "ERROR: vLLM (GPU ${idx}) not ready after ${READY_TIMEOUT}s"; exit 1
        fi
        sleep 3; waited=$((waited + 3))
    done
    log "vLLM ready: ${ep}"
done

python3 /workspace/src/run_eval.py \
    --input_dir "${INPUT_DIR}" \
    --output_file "${OUTPUT_FILE}" \
    --log_file "${LOG_FILE}" \
    --endpoints "${ENDPOINTS[@]}"
rc=$?

log "run_eval exit code ${rc}"
exit ${rc}
