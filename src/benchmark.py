import torch
from typing import List, Tuple, Dict, Any
from src.config import SLMConfig, PipelineConfig
from src.device import DeviceManager
from src.model import SequentialSLM
from src.pipeline import SequentialInferenceEngine, PipelineParallelEngine, PipelineMetrics
from src.visualizer import Visualizer

def generate_dummy_requests(
    total_requests: int,
    micro_batch_size: int,
    seq_len: int,
    vocab_size: int
) -> List[torch.Tensor]:
    """Generates dummy request token batches partitioned into micro-batches."""
    assert total_requests % micro_batch_size == 0, (
        f"total_requests ({total_requests}) must be divisible by micro_batch_size ({micro_batch_size})"
    )
    num_micro_batches = total_requests // micro_batch_size
    micro_batches = []
    
    # Use deterministic seed for reproducibility
    torch.manual_seed(42)
    for _ in range(num_micro_batches):
        mb_tensor = torch.randint(0, vocab_size, (micro_batch_size, seq_len), dtype=torch.long)
        micro_batches.append(mb_tensor)

    return micro_batches

class BenchmarkRunner:
    """Runs pipeline benchmark comparisons and micro-batch sweeps."""

    def __init__(self, pipe_config: PipelineConfig, slm_config: SLMConfig):
        self.pipe_config = pipe_config
        self.slm_config = slm_config
        self.device_mgr = DeviceManager(pipe_config)

    def run(self) -> Tuple[PipelineMetrics, PipelineMetrics, bool]:
        print("\n==================================================")
        print("     STARTING PIPELINE SLM BENCHMARK RUN")
        print("==================================================")
        print(self.device_mgr.get_summary())
        print(f"Total Requests: {self.pipe_config.total_requests}")
        print(f"Micro-Batch Size: {self.pipe_config.micro_batch_size} (Num Micro-Batches: {self.pipe_config.total_requests // self.pipe_config.micro_batch_size})")
        print(f"Sequence Length: {self.pipe_config.seq_len} | Hidden Dim: {self.slm_config.hidden_dim}")
        print("--------------------------------------------------")

        # Create identical model weights
        full_model = SequentialSLM(self.slm_config)

        seq_engine = SequentialInferenceEngine(self.pipe_config, self.slm_config, self.device_mgr)
        pipe_engine = PipelineParallelEngine(self.pipe_config, self.slm_config, self.device_mgr)

        seq_engine.set_weights(full_model)
        pipe_engine.set_weights(full_model)

        input_batches = generate_dummy_requests(
            total_requests=self.pipe_config.total_requests,
            micro_batch_size=self.pipe_config.micro_batch_size,
            seq_len=self.pipe_config.seq_len,
            vocab_size=self.slm_config.vocab_size
        )

        # Warmup run
        print("\n[Warmup] Running initial inference iterations...")
        seq_engine.run(input_batches)
        pipe_engine.run(input_batches)

        # Timed Sequential Run
        print("\n[1/2] Executing Baseline Sequential Inference...")
        seq_outputs, seq_metrics = seq_engine.run(input_batches)

        # Timed Pipeline Parallel Run
        print("[2/2] Executing Pipeline-Parallel Inference...")
        pipe_outputs, pipe_metrics = pipe_engine.run(input_batches)

        # Numerical correctness check
        all_close = True
        for idx, (seq_out, pipe_out) in enumerate(zip(seq_outputs, pipe_outputs)):
            if not torch.allclose(seq_out, pipe_out, atol=1e-4, rtol=1e-3):
                all_close = False
                print(f"WARNING: Output mismatch detected in Micro-Batch {idx}!")

        if all_close:
            print("\n[Correctness Verification]: SUCCESS! Sequential and Pipeline-Parallel logits match perfectly.")
        else:
            print("\n[Correctness Verification]: FAILED! Logits differ between sequential and pipeline runs.")

        # Print Visualizations
        Visualizer.print_ascii_gantt_chart(seq_metrics, title="Sequential Timeline")
        Visualizer.print_ascii_gantt_chart(pipe_metrics, title="Pipeline-Parallel Timeline")
        Visualizer.print_comparison_table(seq_metrics, pipe_metrics)
        Visualizer.plot_benchmark_results(seq_metrics, pipe_metrics)

        return seq_metrics, pipe_metrics, all_close
