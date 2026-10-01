## Pipeline-Parallel SLM Inference with Micro-Batching

Small Language Models (SLMs) can be distributed across multiple GPUs by partitioning their neural network layers across devices. However, a naive implementation processes requests sequentially through these model partitions, leaving GPUs idle while another stage is computing.

This project aims to build a lightweight prototype for **pipeline-parallel SLM inference** that distributes the layers of a transformer-based SLM across multiple GPUs and processes multiple independent requests using **micro-batching and pipeline execution**.

For example, a six-layer SLM running on two GPUs will be partitioned as:

* GPU 0 → Layers 0–2
* GPU 1 → Layers 3–5

Four independent requests will be divided into micro-batches of two requests each. While GPU 1 processes the first micro-batch, GPU 0 will begin processing the next micro-batch. Intermediate activation tensors will be transferred between GPUs as requests move through the pipeline.

The project will implement the model partitioning, micro-batch formation, pipeline scheduling, inter-GPU activation transfer, and inference execution using PyTorch and CUDA. A sequential inference implementation will be maintained as a baseline.

The system will then benchmark sequential execution against pipeline-parallel execution across different micro-batch sizes and numbers of pipeline stages, measuring inference latency, throughput, GPU utilization, communication overhead, and pipeline efficiency.

### Core Objective

The primary objective is to demonstrate that **overlapping computation across multiple model stages and multiple micro-batches can improve GPU utilization and inference throughput compared with sequential execution**, while also exposing the trade-offs introduced by inter-GPU communication, pipeline bubbles, micro-batch size, and uneven layer workloads.

### Core Concepts

* Model parallelism
* Pipeline parallelism
* Layer-wise model partitioning
* Micro-batching
* GPU-to-GPU activation transfer
* CUDA asynchronous execution
* Pipeline scheduling
* GPU utilization
* Inference throughput
* Pipeline bubbles
* Communication overhead
* Load balancing across model stages

### Initial Prototype

The first prototype will use:

```text
Model:          Small Transformer-based SLM
GPUs:           2
Model layers:   6
Partition:      3 layers per GPU
Requests:       4
Micro-batch:    2 requests
```

The prototype will compare:

```text
Sequential:

R1 → GPU0 → GPU1
R2 → GPU0 → GPU1
R3 → GPU0 → GPU1
R4 → GPU0 → GPU1


Pipeline:

MB1 → GPU0 → GPU1
MB2 → GPU0 → GPU1

GPU0 processes MB2 while GPU1 processes MB1.
```

The project will initially remain focused on **AI/ML systems and inference optimization**, without introducing production-oriented backend infrastructure such as Kubernetes, message queues, databases, authentication, or cloud deployment.
