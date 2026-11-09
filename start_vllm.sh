#!/bin/bash
# ==============================================================================
# DATAMATE INFRA - LUONG B (MINH)
# Start vLLM server (>= 0.17) cho Qwen3.5-9B tren 1 GPU.
# Dung:  GPU=0 PORT=8000 ./start_vllm.sh      (chay lan 2 voi GPU=1 PORT=8001)
# ==============================================================================
set -euo pipefail

GPU="${GPU:-0}"
PORT="${PORT:-8000}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"   # trong Docker: /workspace/models/Qwen3.5-9B
MAX_LEN="${MAX_LEN:-32768}"

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "Starting vLLM for ${MODEL_PATH} on GPU ${GPU}, port ${PORT}..."

exec vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --dtype bfloat16 \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 1 \
    --seed 0 \
    --served-model-name qwen3.5-9b
