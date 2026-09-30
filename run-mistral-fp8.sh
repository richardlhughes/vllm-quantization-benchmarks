#!/usr/bin/env bash
# Mistral-7B-Instruct-v0.3 on a 16GB RTX 4080 SUPER via on-the-fly FP8 quantization.
# bf16 weights (~14.5GB) do not fit; FP8 brings them to ~7.3GB.
exec docker run --gpus all -p 8000:8000 --rm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
  vllm/vllm-openai:latest \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --quantization fp8 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
