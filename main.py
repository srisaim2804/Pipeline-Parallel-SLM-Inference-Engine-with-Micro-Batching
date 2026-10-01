import sys
import argparse
import unittest
from src.config import SLMConfig, PipelineConfig
from src.benchmark import BenchmarkRunner

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline-Parallel SLM Inference Prototype & Benchmark"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: run / benchmark
    run_parser = subparsers.add_parser("run", help="Run pipeline parallel SLM demo")
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark suite")
    test_parser = subparsers.add_parser("test", help="Run automated unit tests")

    for p in [run_parser, bench_parser]:
        p.add_argument("--total-requests", type=int, default=4, help="Total number of inference requests")
        p.add_argument("--micro-batch-size", type=int, default=2, help="Micro-batch size (e.g., 2 requests per MB)")
        p.add_argument("--seq-len", type=int, default=32, help="Sequence length of input requests")
        p.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension of SLM")
        p.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers (default 6)")
        p.add_argument("--force-cpu", action="store_true", help="Force CPU execution")
        p.add_argument("--force-simulated-gpu", action="store_true", help="Force single-GPU virtual CUDA stream simulation")
        p.add_argument("--comm-delay-ms", type=float, default=0.0, help="Simulated communication delay in ms")

    return parser.parse_args()

def main():
    args = parse_args()

    if args.command == "test":
        print("\nRunning unit tests...")
        suite = unittest.TestLoader().discover("tests")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

    if args.command in ["run", "benchmark", None]:
        total_reqs = getattr(args, "total_requests", 4)
        mb_size = getattr(args, "micro_batch_size", 2)
        seq_len = getattr(args, "seq_len", 32)
        hidden_dim = getattr(args, "hidden_dim", 256)
        num_layers = getattr(args, "num_layers", 6)
        force_cpu = getattr(args, "force_cpu", False)
        force_sim_gpu = getattr(args, "force_simulated_gpu", False)
        comm_delay = getattr(args, "comm_delay_ms", 0.0)

        slm_config = SLMConfig(
            vocab_size=1000,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=8,
            intermediate_dim=hidden_dim * 2,
            max_seq_len=max(2048, seq_len)
        )


        pipe_config = PipelineConfig(
            num_stages=2,
            micro_batch_size=mb_size,
            total_requests=total_reqs,
            seq_len=seq_len,
            force_cpu=force_cpu,
            force_simulated_gpu=force_sim_gpu,
            simulate_comm_delay_ms=comm_delay
        )

        runner = BenchmarkRunner(pipe_config, slm_config)
        runner.run()

if __name__ == "__main__":
    main()
