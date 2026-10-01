import os
from typing import List, Dict, Any
from tabulate import tabulate
import matplotlib.pyplot as plt
from src.pipeline import PipelineMetrics

class Visualizer:
    """
    Renders terminal ASCII Gantt charts, summary tables, and matplotlib comparison plots.
    """

    @staticmethod
    def print_ascii_gantt_chart(metrics: PipelineMetrics, title: str = "Pipeline Execution Timeline"):
        print(f"\n==================================================")
        print(f"  {title} ({metrics.mode})")
        print(f"==================================================")
        
        events = sorted(metrics.timeline_events, key=lambda x: x["start_ms"])
        if not events:
            print("No events recorded.")
            return

        total_duration = max(e["end_ms"] for e in events)
        num_slots = 50  # Width of ASCII timeline bar

        scale = num_slots / total_duration if total_duration > 0 else 1.0

        for stage_id in [0, 1]:
            stage_events = [e for e in events if e["stage_id"] == stage_id]
            timeline = ["."] * num_slots
            for e in stage_events:
                start_idx = min(int(e["start_ms"] * scale), num_slots - 1)
                end_idx = min(max(start_idx + 1, int(e["end_ms"] * scale)), num_slots)
                mb_label = str(e["mb_id"] % 10)
                for i in range(start_idx, end_idx):
                    timeline[i] = mb_label

            timeline_str = "".join(timeline)
            active_ms = metrics.stage_active_time_ms.get(stage_id, 0.0)
            idle_ms = metrics.stage_idle_time_ms.get(stage_id, 0.0)
            print(f"Stage {stage_id} [GPU {stage_id}]: |{timeline_str}| Active: {active_ms:.2f}ms | Idle: {idle_ms:.2f}ms")

        print("-" * 50)
        print(f"Legend: Numbers 0..N indicate Micro-Batch ID running on that stage.")
        print(f"Total Wall-Clock Time: {metrics.total_time_ms:.2f} ms")
        print(f"Pipeline Bubble Ratio: {metrics.bubble_ratio * 100:.1f}%\n")

    @staticmethod
    def print_comparison_table(seq_metrics: PipelineMetrics, pipe_metrics: PipelineMetrics):
        speedup = (
            seq_metrics.total_time_ms / pipe_metrics.total_time_ms
            if pipe_metrics.total_time_ms > 0
            else 0.0
        )
        throughput_increase = (
            ((pipe_metrics.throughput_req_per_sec - seq_metrics.throughput_req_per_sec) / seq_metrics.throughput_req_per_sec) * 100.0
            if seq_metrics.throughput_req_per_sec > 0
            else 0.0
        )

        headers = ["Metric", "Sequential Baseline", "Pipeline-Parallel", "Improvement / Delta"]
        table = [
            ["Execution Latency (ms)", f"{seq_metrics.total_time_ms:.2f}", f"{pipe_metrics.total_time_ms:.2f}", f"{speedup:.2f}x Speedup"],
            ["Throughput (Req / sec)", f"{seq_metrics.throughput_req_per_sec:.2f}", f"{pipe_metrics.throughput_req_per_sec:.2f}", f"+{throughput_increase:.1f}%"],
            ["Throughput (Tok / sec)", f"{seq_metrics.throughput_tok_per_sec:.2f}", f"{pipe_metrics.throughput_tok_per_sec:.2f}", f"+{throughput_increase:.1f}%"],
            ["Pipeline Bubble Ratio", f"{seq_metrics.bubble_ratio * 100:.1f}%", f"{pipe_metrics.bubble_ratio * 100:.1f}%", f"{(pipe_metrics.bubble_ratio - seq_metrics.bubble_ratio)*100:+.1f}%"],
            ["Stage 0 Active Time", f"{seq_metrics.stage_active_time_ms[0]:.2f} ms", f"{pipe_metrics.stage_active_time_ms[0]:.2f} ms", "-"],
            ["Stage 1 Active Time", f"{seq_metrics.stage_active_time_ms[1]:.2f} ms", f"{pipe_metrics.stage_active_time_ms[1]:.2f} ms", "-"],
            ["Transfer Overhead", f"{seq_metrics.transfer_time_ms:.3f} ms", f"{pipe_metrics.transfer_time_ms:.3f} ms", "-"]
        ]

        print("\n=========================================================================")
        print("                 BENCHMARK PERFORMANCE COMPARISON")
        print("=========================================================================")
        print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

    @staticmethod
    def plot_benchmark_results(
        seq_metrics: PipelineMetrics,
        pipe_metrics: PipelineMetrics,
        output_path: str = "benchmark_results.png"
    ):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        categories = ["Sequential", "Pipeline-Parallel"]

        # Latency Plot
        latencies = [seq_metrics.total_time_ms, pipe_metrics.total_time_ms]
        bars1 = axes[0].bar(categories, latencies, color=["#e74c3c", "#2ecc71"])
        axes[0].set_title("Total Latency (ms) [Lower is better]")
        axes[0].set_ylabel("Milliseconds (ms)")
        for bar in bars1:
            yval = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.02), f"{yval:.1f} ms", ha='center', va='bottom', fontweight='bold')

        # Throughput Plot
        throughputs = [seq_metrics.throughput_req_per_sec, pipe_metrics.throughput_req_per_sec]
        bars2 = axes[1].bar(categories, throughputs, color=["#e74c3c", "#2ecc71"])
        axes[1].set_title("Throughput (Req/sec) [Higher is better]")
        axes[1].set_ylabel("Requests per Second")
        for bar in bars2:
            yval = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.02), f"{yval:.1f}", ha='center', va='bottom', fontweight='bold')

        # Bubble Ratio Plot
        bubbles = [seq_metrics.bubble_ratio * 100, pipe_metrics.bubble_ratio * 100]
        bars3 = axes[2].bar(categories, bubbles, color=["#e74c3c", "#3498db"])
        axes[2].set_title("Pipeline Bubble Ratio (%) [Lower is better]")
        axes[2].set_ylabel("Percentage (%)")
        for bar in bars3:
            yval = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.02), f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"\nSaved benchmark plot to: {os.path.abspath(output_path)}")
