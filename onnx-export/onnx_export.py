# onnx_export.py

import torch
import time
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import onnxruntime as ort


# ------------------------------------------------------------------ #
# Wrapper — disables KV cache for clean ONNX export                   #
# ------------------------------------------------------------------ #

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        outputs = self.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=False,
        )
        return outputs[0]  # logits only


# ------------------------------------------------------------------ #
# Benchmark helper                                                     #
# ------------------------------------------------------------------ #

def benchmark(fn, *args, n_warmup=5, n_runs=20):
    for _ in range(n_warmup):
        fn(*args)
    start = time.perf_counter()
    for _ in range(n_runs):
        fn(*args)
    end = time.perf_counter()
    return (end - start) / n_runs * 1000


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    model.eval()

    prompt = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nWhat is 2+2?</s>\n<|assistant|>\n"
    inputs  = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Input shape: {input_ids.shape}")

    # wrap model
    wrapped = ModelWrapper(model)

    # ------------------------------------------------------------------ #
    # PyTorch eager baseline                                               #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def eager_forward():
        return wrapped(input_ids)

    print("\nBenchmarking PyTorch eager...")
    eager_ms = benchmark(eager_forward)
    print(f"Eager: {eager_ms:.2f} ms")

    # ------------------------------------------------------------------ #
    # Export to ONNX                                                       #
    # ------------------------------------------------------------------ #

    print("\nExporting to ONNX...")
    onnx_path = "tinyllama.onnx"

    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (input_ids, ),
            onnx_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits":    {0: "batch_size", 1: "sequence_length"}
            },
            opset_version = 17,
            do_constant_folding=True
        )

    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"Exported to {onnx_path} ({size_mb:.1f} MB)")

    # ------------------------------------------------------------------ #
    # ONNX Runtime                                                         #
    # ------------------------------------------------------------------ #

    print("\nLoading ONNX Runtime session...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers = ["CPUExecutionProvider"]
    )

    def onnx_forward():
        return session.run(
            None,
            {"input_ids": input_ids.numpy()}
        )

    # correctness check
    with torch.no_grad():
        torch_out = wrapped(input_ids).detach().numpy()
    onnx_out = onnx_forward()[0]

    max_diff = np.max(np.abs(torch_out - onnx_out))
    print(f"Max difference PyTorch vs ONNX: {max_diff:.6f}")
    print(f"Match: {max_diff < 1e-3}")

    # ------------------------------------------------------------------ #
    # Benchmark ONNX Runtime                                               #
    # ------------------------------------------------------------------ #

    print("\nBenchmarking ONNX Runtime...")
    onnx_ms = benchmark(onnx_forward)
    print(f"ONNX Runtime: {onnx_ms:.2f} ms")

    # ------------------------------------------------------------------ #
    # Results                                                              #
    # ------------------------------------------------------------------ #

    print(f"\n{'='*40}")
    print(f"PyTorch eager: {eager_ms:.2f} ms")
    print(f"ONNX Runtime:  {onnx_ms:.2f} ms")
    print(f"Speedup:       {eager_ms/onnx_ms:.2f}x")
    print(f"Model size:    {size_mb:.0f} MB")
    print(f"\nONNX advantages:")
    print(f"  - No PyTorch dependency at runtime")
    print(f"  - Portable: same file on any ONNX-compatible runtime")
    print(f"  - Next step: TensorRT can optimize this further on GPU")


if __name__ == "__main__":
    main()