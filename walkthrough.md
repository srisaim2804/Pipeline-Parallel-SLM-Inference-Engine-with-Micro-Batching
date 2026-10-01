# Walkthrough: Distributed 2-GPU Pipeline-Parallel SLM Inference Prototype

We have built an end-to-end **Pipeline-Parallel Small Language Model (SLM) Inference System** designed to partition a 6-layer Transformer SLM across 2 pipeline stages and process concurrent requests using micro-batching.

---

## 🛠️ Key Components Built

1. [`src/config.py`](file:///home/doraemon/Documents/Projects/SLM_Optim/src/config.py): Dataclass configurations for SLM architecture (`SLMConfig`) and pipeline scheduling (`PipelineConfig`).
2. [`src/model.py`](file:///home/doraemon/Documents/Projects/SLM_Optim/src/model.py): PyTorch Transformer decoder components (`EmbeddingLayer`, `TransformerBlock`, `LMHead`), partitioned `PipelineStage` (Stage 0: Layers 0–2, Stage 1: Layers 3–5 + LM Head), and baseline `SequentialSLM`.
3. [`src/device.py`](file:///home/doraemon/Documents/Projects/SLM_Optim/src/device.py): Smart `DeviceManager` with three execution modes:
   - **Physical Multi-GPU**: `cuda:0` and `cuda:1` for multi-GPU setups.
   - **Virtual 2-GPU Mode (Laptop Single GPU)**: Uses separate, asynchronous `torch.cuda.Stream` instances on your **NVIDIA GeForce RTX 3060 Laptop GPU** to simulate 2 distinct pipeline stages and asynchronous inter-stage activation transfers.
   - **CPU Mode**: Fallback for CPU environments.
4. [`src/pipeline.py`](file:///home/doraemon/Documents/Projects/SLM_Optim/src/pipeline.py): Baseline `SequentialInferenceEngine` vs micro-batched `PipelineParallelEngine`.
5. [`src/benchmark.py`](file:///home/doraemon/Documents/Projects/SLM_Optim/src/benchmark.py): Automated benchmark runner measuring execution latency, throughput (requests/sec, tokens/sec), stage active/idle durations, communication overhead, and logit output correctness.
6. [`src/visualizer.py`](file:///home/doraemon/Documents/Projects/SLM_Optim/src/visualizer.py): Terminal ASCII Gantt chart timeline visualizer, tabulate comparison table, and `matplotlib` chart exporter (`benchmark_results.png`).
7. [`main.py`](file:///home/doraemon/Documents/Projects/SLM_Optim/main.py): CLI interface (`run`, `benchmark`, `test`).
8. [`tests/test_pipeline.py`](file:///home/doraemon/Documents/Projects/SLM_Optim/tests/test_pipeline.py): Automated unit test suite verifying logit equality, tensor shapes, device allocation, and metrics calculation.

---

## 🧪 Verification & Test Results

### 1. Automated Unit Tests
Executed via `.venv/bin/python main.py test`:
- `test_device_manager_cpu`: Passed
- `test_stage_forward_shapes`: Passed
- `test_sequential_vs_pipeline_equivalence`: Passed (Sequential vs Pipeline outputs match with `atol < 1e-4`)
- `test_pipeline_metrics`: Passed

### 2. End-to-End GPU Execution
Executed on **NVIDIA GeForce RTX 3060 Laptop GPU**:

```text
==================================================
     STARTING PIPELINE SLM BENCHMARK RUN
==================================================
Execution Mode: Virtual Multi-GPU (1 GPU with CUDA Streams)
Detected GPU: NVIDIA GeForce RTX 3060 Laptop GPU
Stage 0 Device: cuda:0 | Stage 1 Device: cuda:0
CUDA Available: True (Device count: 1)
Total Requests: 16
Micro-Batch Size: 2 (Num Micro-Batches: 8)
Sequence Length: 128 | Hidden Dim: 512
--------------------------------------------------

[Correctness Verification]: SUCCESS! Sequential and Pipeline-Parallel logits match perfectly.

==================================================
  Pipeline-Parallel Timeline (Pipeline-Parallel)
==================================================
Stage 0 [GPU 0]: |00111....22....333...444....555...666...777.......| Active: 11.53ms | Idle: 15.41ms
Stage 1 [GPU 1]: |.....000...111....222...333....44....555...666.777| Active: 11.81ms | Idle: 15.13ms
--------------------------------------------------
Legend: Numbers 0..N indicate Micro-Batch ID running on that stage.
Total Wall-Clock Time: 26.94 ms
```

### 3. Generated Benchmark Artifact
The benchmark runner automatically exported visual charts to [`benchmark_results.png`](file:///home/doraemon/Documents/Projects/SLM_Optim/benchmark_results.png).

---

## 🚀 How to Run the Project

```bash
# Activate virtual environment
source .venv/bin/activate

# 1. Run default 4-request demo
python main.py run

# 2. Run scaled 16-request micro-batch benchmark
python main.py run --total-requests 16 --micro-batch-size 2 --hidden-dim 512 --seq-len 128

# 3. Run with simulated 2ms network communication latency
python main.py run --total-requests 16 --micro-batch-size 2 --comm-delay-ms 2.0

# 4. Run unit tests
python main.py test
```
