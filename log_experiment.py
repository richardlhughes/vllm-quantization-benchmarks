import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import mlflow
import requests

VLLM_URL = "http://host.docker.internal:8000"
PROMPT = "Explain the difference between TCP and UDP in two sentences."
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.90
GPU = "RTX 4080 SUPER 16GB"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id served by vLLM")
    parser.add_argument("--quantization", required=True, help="e.g. fp8, awq")
    parser.add_argument("--run-name", required=True, help="MLflow run name")
    parser.add_argument("--num-requests", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1, help="parallel in-flight requests")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="record-only: whether the vLLM server this run hit was started with --enforce-eager",
    )
    parser.add_argument(
        "--completions",
        action="store_true",
        help="use /v1/completions (raw prompt) instead of /v1/chat/completions, for base models with no chat template",
    )
    return parser.parse_args()


def do_request(model, max_tokens, completions=False):
    t0 = time.perf_counter()
    if completions:
        endpoint, body = "/v1/completions", {"model": model, "prompt": PROMPT, "max_tokens": max_tokens}
    else:
        endpoint, body = "/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
        }
    resp = requests.post(f"{VLLM_URL}{endpoint}", json=body, timeout=120)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    usage = resp.json()["usage"]
    return elapsed, usage["completion_tokens"], usage["prompt_tokens"]


def main():
    args = parse_args()

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("vllm-model-comparison")

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("model", args.model)
        mlflow.log_param("quantization", args.quantization)
        mlflow.log_param("gpu_memory_utilization", GPU_MEMORY_UTILIZATION)
        mlflow.log_param("max_model_len", MAX_MODEL_LEN)
        mlflow.log_param("gpu", GPU)
        mlflow.log_param("num_requests", args.num_requests)
        mlflow.log_param("concurrency", args.concurrency)
        mlflow.log_param("enforce_eager", args.enforce_eager)
        mlflow.log_param("prompt", PROMPT)

        wall_t0 = time.perf_counter()
        results = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [
                ex.submit(do_request, args.model, args.max_tokens, args.completions)
                for _ in range(args.num_requests)
            ]
            for i, fut in enumerate(futures):
                elapsed, completion_tokens, prompt_tokens = fut.result()
                results.append((elapsed, completion_tokens, prompt_tokens))
                mlflow.log_metric("e2e_latency_seconds", elapsed, step=i)
                mlflow.log_metric("completion_tokens", completion_tokens, step=i)
                mlflow.log_metric("prompt_tokens", prompt_tokens, step=i)
                mlflow.log_metric("token_throughput_tokens_per_sec", completion_tokens / elapsed, step=i)
        wall_elapsed = time.perf_counter() - wall_t0

        latencies = [r[0] for r in results]
        per_request_throughput = [r[1] / r[0] for r in results]
        total_completion_tokens = sum(r[1] for r in results)

        mlflow.log_metric("mean_latency_seconds", statistics.mean(latencies))
        mlflow.log_metric("p50_latency_seconds", statistics.median(latencies))
        mlflow.log_metric("mean_token_throughput_tokens_per_sec", statistics.mean(per_request_throughput))
        mlflow.log_metric("wall_clock_seconds", wall_elapsed)
        mlflow.log_metric("aggregate_throughput_tokens_per_sec", total_completion_tokens / wall_elapsed)

        print(f"concurrency={args.concurrency} num_requests={args.num_requests}")
        print(f"Wall clock: {wall_elapsed:.3f}s")
        print(f"Aggregate throughput: {total_completion_tokens / wall_elapsed:.2f} tok/s")
        print(f"Mean per-request latency: {statistics.mean(latencies):.3f}s")
        print(f"Mean per-request throughput: {statistics.mean(per_request_throughput):.2f} tok/s")


if __name__ == "__main__":
    main()
