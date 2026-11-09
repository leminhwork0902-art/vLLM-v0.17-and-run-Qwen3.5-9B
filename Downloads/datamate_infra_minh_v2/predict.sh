#!/bin/bash
# /data -> /result/submission.jsonl, moi GPU 1 instance vLLM (cong 8000, 8001)
set -uo pipefail

IN="${INPUT_DIR:-/data}"; OUT="${OUTPUT_DIR:-/result}"
MODEL="${MODEL_DIR:-/workspace/models/Qwen3.5-9B}"
WAIT="${READY_TIMEOUT:-1200}"
LOG="$OUT/execution.log"
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1

n=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$n" -ge 1 ] || n=1
[ "$n" -le 2 ] || n=2
log "start, $n GPU"

pids=(); eps=()
trap 'kill "${pids[@]}" 2>/dev/null; wait 2>/dev/null' EXIT

for ((i = 0; i < n; i++)); do
    port=$((8000 + i))
    CUDA_VISIBLE_DEVICES=$i python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --host 127.0.0.1 --port "$port" --dtype bfloat16 \
        --max-model-len 32768 --gpu-memory-utilization 0.90 --seed 0 \
        --served-model-name qwen3.5-9b >> "$OUT/vllm_gpu$i.log" 2>&1 &
    pids+=($!); eps+=("http://127.0.0.1:$port/v1")
done

for i in "${!eps[@]}"; do
    t=0
    until curl -sf "${eps[$i]}/models" > /dev/null; do
        kill -0 "${pids[$i]}" 2>/dev/null || { log "vLLM GPU $i chet, xem vllm_gpu$i.log"; exit 1; }
        [ "$t" -lt "$WAIT" ] || { log "vLLM GPU $i qua $WAIT giay chua san sang"; exit 1; }
        sleep 3; t=$((t + 3))
    done
    log "ready ${eps[$i]}"
done

python3 /workspace/src/run_eval.py --input_dir "$IN" --output_file "$OUT/submission.jsonl" \
    --log_file "$LOG" --endpoints "${eps[@]}"
rc=$?
log "run_eval exit $rc"
exit $rc
