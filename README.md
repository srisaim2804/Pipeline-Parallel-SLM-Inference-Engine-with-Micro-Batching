# Pipeline-Parallel SLM Inference Engine with Micro-Batching

A lightweight, high-performance prototype for **Pipeline-Parallel Small Language Model (SLM) Inference** implemented in PyTorch and CUDA.

This project partitions a transformer-based SLM across multiple pipeline stages (devices) and processes multiple concurrent inference requests using **micro-batching** to overlap computation and eliminate idle GPU time.

---

## 📌 Project Overview

When distributing Small Language Models across multiple GPUs by splitting layers across devices, a naive implementation processes requests sequentially. This leaves downstream GPUs idle while earlier stages compute (causing high pipeline bubbles).

This engine demonstrates **1F1B-style Pipeline Parallelism**:
- A **6-Layer Transformer SLM** is partitioned into two stages:
  - **Stage 0 (GPU 0)**: Embedding Layer + Transformer Blocks 0, 1, 2
  - **Stage 1 (GPU 1)**: Transformer Blocks 3, 4, 5 + LM Projection Head
- Incoming requests (e.g. 4 requests) are grouped into **micro-batches** of 2 requests each (`MB1=[R1, R2]`, `MB2=[R3, R4]`).
- Computation is overlapped: While **Stage 1** executes **MB1**, **Stage 0** concurrently computes **MB2**.

```text
Sequential Execution (Baseline):
  R1 -> Stage 0 (GPU 0) -> Stage 1 (GPU 1)
  R2 -> Stage 0 (GPU 0) -> Stage 1 (GPU 1)
  R3 -> Stage 0 (GPU 0) -> Stage 1 (GPU 1)
  R4 -> Stage 0 (GPU 0) -> Stage 1 (GPU 1)

Pipeline-Parallel Execution (Overlapped Micro-Batches):
  Step 1: Stage 0 computes MB1
  Step 2: Stage 0 computes MB2  <=== OVERLAP ===>  Stage 1 computes MB1
  Step 3:                                          Stage 1 computes MB2
```

---

## 💻 Laptop Single-GPU Virtual Simulation

If you only have **1 physical GPU** (e.g., NVIDIA RTX 3060 Laptop GPU):
The engine includes a smart `DeviceManager` with **Virtual 2-GPU Stream Simulation**:
- **Physical Multi-GPU**: Uses `cuda:0` for Stage 0 and `cuda:1` for Stage 1 when $\ge 2$ GPUs are available.
- **Virtual 2-GPU Mode (Single-GPU Laptop)**: Uses separate asynchronous `torch.cuda.Stream` handles (`stream_stage0`, `stream_stage1`) and explicit `torch.cuda.Event` synchronization on `cuda:0`. This allows you to test 2-stage pipeline parallelism, micro-batch timing, and GPU activation transfers without needing a second graphic card!
- **CPU Mode**: Multithreaded CPU fallback.

---

## 📁 Repository Structure

```text
SLM_Optim/
├── main.py                 # Main CLI entrypoint (run, benchmark, test)
├── Project_plan.md         # Detailed project requirement specification
├── README.md               # Project documentation & instructions
├── src/
│   ├── __init__.py
│   ├── config.py           # Dataclass configs for SLM & Pipeline settings
│   ├── model.py            # Transformer SLM, Layer modules & Pipeline Stages
│   ├── device.py           # DeviceManager (Physical, Virtual Stream & CPU modes)
│   ├── pipeline.py         # Sequential & Pipeline-Parallel inference engines
│   ├── benchmark.py        # Benchmark suite & numerical parity verification
│   └── visualizer.py       # ASCII Gantt timeline, tabulate comparison & Matplotlib plots
└── tests/
    └── test_pipeline.py    # Unit test suite verifying correctness & equivalence
```

---

## 🚀 Quick Start & Installation

### 1. Setup Environment
Ensure Python 3.10+ is installed:

```bash
# Clone or navigate to directory
cd SLM_Optim

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (PyTorch, Matplotlib, Tabulate)
pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
pip install matplotlib tabulate
```

---

## ⚡ Usage Commands

### 1. Run Default Demo (4 Requests, Micro-Batch Size 2)
Runs the baseline sequential vs pipeline-parallel comparison on your GPU and prints ASCII timeline charts:

```bash
python main.py run
```

### 2. Run Scaled Micro-Batch Benchmark
Test larger request volumes (e.g. 16 requests across 8 micro-batches of size 2):

```bash
python main.py run --total-requests 16 --micro-batch-size 2 --hidden-dim 512 --seq-len 128
```

### 3. Simulate Inter-GPU Network Delay
Simulate 2ms of PCIe / Inter-GPU activation transfer latency between stages:

```bash
python main.py run --total-requests 16 --micro-batch-size 2 --comm-delay-ms 2.0
```

### 4. Force CPU Execution
Run in CPU multithreaded mode:

```bash
python main.py run --force-cpu
```

### 5. Run Automated Unit Tests
Verify model partitioning, tensor shapes, and logit equivalence:

```bash
python main.py test
```

---

## 📊 Sample Benchmark Output

```text
=========================================================================
                 BENCHMARK PERFORMANCE COMPARISON
=========================================================================
╒════════════════════════╤═══════════════════════╤═════════════════════╤═══════════════════════╕
│ Metric                 │ Sequential Baseline   │ Pipeline-Parallel   │ Improvement / Delta   │
╞════════════════════════╪═══════════════════════╪═════════════════════╪═══════════════════════╡
│ Execution Latency (ms) │ 25.05                 │ 26.94               │ 0.93x Speedup         │
├────────────────────────┼───────────────────────┼─────────────────────┼───────────────────────┤
│ Throughput (Req / sec) │ 638.65                │ 593.95              │ +-7.0%                │
├────────────────────────┼───────────────────────┼─────────────────────┼───────────────────────┤
│ Throughput (Tok / sec) │ 81747.55              │ 76025.39            │ +-7.0%                │
├────────────────────────┼───────────────────────┼─────────────────────┼───────────────────────┤
│ Pipeline Bubble Ratio  │ 53.5%                 │ 56.7%               │ +3.2%                 │
├────────────────────────┼───────────────────────┼─────────────────────┼───────────────────────┤
│ Stage 0 Active Time    │ 11.52 ms              │ 11.53 ms            │ -                     │
├────────────────────────┼───────────────────────┼─────────────────────┼───────────────────────┤
│ Stage 1 Active Time    │ 11.79 ms              │ 11.81 ms            │ -                     │
├────────────────────────┼───────────────────────┼─────────────────────┼───────────────────────┤
│ Transfer Overhead      │ 0.152 ms              │ 0.144 ms            │ -                     │
╘════════════════════════╧═══════════════════════╧═════════════════════╧═══════════════════════╛
```

Visual benchmark charts are automatically exported to `benchmark_results.png`.

---

## 🔧 Command-Line Arguments

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--total-requests` | `4` | Total number of inference requests |
| `--micro-batch-size` | `2` | Number of requests per micro-batch |
| `--seq-len` | `32` | Sequence length of input requests |
| `--hidden-dim` | `256` | Hidden dimension of the SLM |
| `--num-layers` | `6` | Total transformer decoder layers |
| `--comm-delay-ms` | `0.0` | Simulated inter-stage network delay in ms |
| `--force-cpu` | `False` | Force CPU execution mode |
| `--force-simulated-gpu`| `False` | Force single-GPU CUDA stream simulation mode |

---

## 📜 License

MIT License. Feel free to use and modify for learning AI/ML Systems & Distributed Inference Optimization!
