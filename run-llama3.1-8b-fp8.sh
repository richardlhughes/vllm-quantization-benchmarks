#!/usr/bin/env bash
# Llama-3.1-8B-Instruct on a 16GB RTX 4080 SUPER via on-the-fly FP8 quantization.
# bf16 weights (~16GB) do not fit; FP8 brings them to ~8GB.
# Gated repo: requires HF_TOKEN in your shell env, and you must accept the
# license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct first.
exec docker run --gpus all -p 8000:8000 --rm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
  -e HF_TOKEN \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --quantization fp8 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
