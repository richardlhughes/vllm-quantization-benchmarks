import argparse
import queue

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="host.docker.internal:8001")
    parser.add_argument("--prompt", default="The difference between TCP and UDP is")
    args = parser.parse_args()

    client = grpcclient.InferenceServerClient(url=args.url)

    inputs = [
        grpcclient.InferInput("text_input", [1], "BYTES"),
        grpcclient.InferInput("stream", [1], "BOOL"),
        grpcclient.InferInput("sampling_parameters", [1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(np.array([args.prompt.encode()], dtype=object))
    inputs[1].set_data_from_numpy(np.array([False]))
    inputs[2].set_data_from_numpy(
        np.array([b'{"temperature":"0.1","max_tokens":"64"}'], dtype=object)
    )

    outputs = [grpcclient.InferRequestedOutput("text_output")]

    result_queue = queue.Queue()

    def callback(result, error):
        result_queue.put((result, error))

    client.start_stream(callback=callback)
    client.async_stream_infer(
        model_name="vllm_model",
        inputs=inputs,
        outputs=outputs,
        request_id="1",
    )

    result, error = result_queue.get(timeout=60)
    client.stop_stream()

    if error:
        raise error

    text_output = result.as_numpy("text_output")
    print("PROMPT:", args.prompt)
    print("OUTPUT:", text_output[0].decode())


if __name__ == "__main__":
    main()
