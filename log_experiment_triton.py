import argparse
import statistics
import threading
import time

import mlflow
import numpy as np
import tritonclient.grpc as grpcclient

TRITON_URL = "host.docker.internal:8001"
PROMPT = "Explain the difference between TCP and UDP in two sentences."
GPU = "RTX 4080 SUPER 16GB"


def build_inputs(max_tokens):
    inputs = [
        grpcclient.InferInput("text_input", [1], "BYTES"),
        grpcclient.InferInput("stream", [1], "BOOL"),
        grpcclient.InferInput("sampling_parameters", [1], "BYTES"),
        grpcclient.InferInput("return_num_output_tokens", [1], "BOOL"),
        grpcclient.InferInput("return_num_input_tokens", [1], "BOOL"),
    ]
    inputs[0].set_data_from_numpy(np.array([PROMPT.encode()], dtype=object))
    inputs[1].set_data_from_numpy(np.array([False]))
    inputs[2].set_data_from_numpy(
        np.array([f'{{"max_tokens":"{max_tokens}"}}'.encode()], dtype=object)
    )
    inputs[3].set_data_from_numpy(np.array([True]))
    inputs[4].set_data_from_numpy(np.array([True]))
    return inputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Triton model name (e.g. vllm_model)")
    parser.add_argument("--underlying-model", required=True, help="actual HF model id being served")
    parser.add_argument("--quantization", default="none")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--num-requests", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("vllm-model-comparison")

    outputs = [
        grpcclient.InferRequestedOutput("text_output"),
        grpcclient.InferRequestedOutput("num_output_tokens"),
        grpcclient.InferRequestedOutput("num_input_tokens"),
    ]

    pending = {}
    results = {}
    lock = threading.Lock()
    done_event = threading.Event()
    completed_count = 0
    sem = threading.Semaphore(args.concurrency)
    errors = []

    def callback(result, error):
        nonlocal completed_count
        if error:
            errors.append(error)
            with lock:
                completed_count += 1
                if completed_count == args.num_requests:
                    done_event.set()
            sem.release()
            return
        req_id = result.get_response().id
        elapsed = time.perf_counter() - pending[req_id]
        completion_tokens = int(np.asarray(result.as_numpy("num_output_tokens")).reshape(-1)[0])
        prompt_tokens = int(np.asarray(result.as_numpy("num_input_tokens")).reshape(-1)[0])
        with lock:
            results[req_id] = (elapsed, completion_tokens, prompt_tokens)
            completed_count += 1
            if completed_count == args.num_requests:
                done_event.set()
        sem.release()

    client = grpcclient.InferenceServerClient(url=TRITON_URL)
    client.start_stream(callback=callback)

    wall_t0 = time.perf_counter()
    for i in range(args.num_requests):
        sem.acquire()
        req_id = str(i)
        pending[req_id] = time.perf_counter()
        client.async_stream_infer(
            model_name=args.model,
            inputs=build_inputs(args.max_tokens),
            outputs=outputs,
            request_id=req_id,
        )

    if not done_event.wait(timeout=300):
        raise TimeoutError(f"only {completed_count}/{args.num_requests} requests completed")
    wall_elapsed = time.perf_counter() - wall_t0
    client.stop_stream()

    if errors:
        raise errors[0]

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("serving_layer", "triton")
        mlflow.log_param("triton_model_name", args.model)
        mlflow.log_param("model", args.underlying_model)
        mlflow.log_param("quantization", args.quantization)
        mlflow.log_param("gpu", GPU)
        mlflow.log_param("num_requests", args.num_requests)
        mlflow.log_param("concurrency", args.concurrency)
        mlflow.log_param("prompt", PROMPT)

        ordered = [results[str(i)] for i in range(args.num_requests)]
        for i, (elapsed, completion_tokens, prompt_tokens) in enumerate(ordered):
            mlflow.log_metric("e2e_latency_seconds", elapsed, step=i)
            mlflow.log_metric("completion_tokens", completion_tokens, step=i)
            mlflow.log_metric("prompt_tokens", prompt_tokens, step=i)
            mlflow.log_metric("token_throughput_tokens_per_sec", completion_tokens / elapsed, step=i)

        latencies = [r[0] for r in ordered]
        per_request_throughput = [r[1] / r[0] for r in ordered]
        total_completion_tokens = sum(r[1] for r in ordered)

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
