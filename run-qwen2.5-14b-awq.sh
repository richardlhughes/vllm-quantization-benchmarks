#!/usr/bin/env bash
# Qwen2.5-14B-Instruct on a 16GB RTX 4080 SUPER via pre-quantized AWQ (4-bit).
# The plain dense repo doesn't fit here: bf16 is ~29.5GB, and even FP8
# (~14.7GB) leaves no headroom for KV cache once the ~1.7GB WSL/desktop
# overhead is subtracted from 16GB. This vLLM build also has no on-the-fly
# bitsandbytes 4-bit support. AWQ weights are ~7.35GB, leaving plenty of
# room for KV cache.
exec docker run --gpus all -p 8000:8000 --rm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-14B-Instruct-AWQ \
  --quantization awq \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
