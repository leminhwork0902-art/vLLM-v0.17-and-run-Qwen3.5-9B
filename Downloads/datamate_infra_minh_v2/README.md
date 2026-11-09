# DataMate - Sub-team B (Finetune / Infra - Minh)

## Overview
This package contains setup scripts, Docker configurations, and training pipelines for **Qwen3.5-9B** under the DataMate competition requirements.

### Files Included:
1. `start_vllm.sh`: Starts vLLM (>= v0.17) server on port 8000 with bfloat16 & 32k context window.
2. `test_client.py`: Python client for Phú, Thảo, and Thịnh to interact with vLLM OpenAI-compatible endpoint.
3. `docker-compose.yml`: Containerized setup for local GPU environment.
4. `train_lora_unsloth.py`: Unsloth LoRA fine-tuning script (~22GB VRAM, BF16/FP16, no 4-bit).
5. `Dockerfile`: Docker build specification (CUDA 12.8+, offline mode).
6. `predict.sh`: Execution entrypoint for BTC offline evaluation (/data -> /result/submission.jsonl).

### Rules & Checklist:
- Offline execution (no external API calls during evaluation).
- vLLM serving Qwen3.5-9B with temperature 0.0 (greedy decoding).
- Output format strict adherence (number, percent, date, boolean, string, list, table, unanswerable).
