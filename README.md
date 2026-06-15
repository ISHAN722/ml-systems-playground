# ml-systems-playground

A hands-on tour of the ML systems stack — from model optimization and quantization to vector search, experiment tracking, data pipelines, and production serving. Each component is self-contained and runnable.

## What's inside

- **onnx-export** — PyTorch → ONNX export + benchmarking (1.83x speedup on CPU, 46x GPU vs CPU)
- **rag-demo** — Retrieval-Augmented Generation with FAISS + TinyLlama
- **grpc-inference** — gRPC inference server (~741ms end-to-end latency)
- **data-pipeline** — Arrow/Parquet processing (469x faster reads with column pruning)
- **wandb-demo** — Experiment tracking with Weights & Biases

## Environment

Most components run locally on Apple Silicon (M4 Pro). GPU components (AWQ quantization, CUDA kernels) were run on Google Colab T4.
