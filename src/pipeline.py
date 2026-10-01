import time
import torch
import dataclasses
from typing import List, Dict, Tuple, Optional, Any
from src.config import SLMConfig, PipelineConfig
from src.device import DeviceManager
from src.model import PipelineStage, SequentialSLM

@dataclasses.dataclass
class MicroBatchResult:
    micro_batch_id: int
    input_ids: torch.Tensor
    logits: torch.Tensor
    stage_timings: List[Dict[str, Any]]

@dataclasses.dataclass
class PipelineMetrics:
    mode: str
    total_time_ms: float
    throughput_req_per_sec: float
    throughput_tok_per_sec: float
    stage_active_time_ms: Dict[int, float]
    stage_idle_time_ms: Dict[int, float]
    transfer_time_ms: float
    bubble_ratio: float
    timeline_events: List[Dict[str, Any]]

class BaseInferenceEngine:
    def __init__(self, config: PipelineConfig, slm_config: SLMConfig, device_mgr: DeviceManager):
        self.config = config
        self.slm_config = slm_config
        self.device_mgr = device_mgr

        # Initialize 2 pipeline stages
        self.stage0 = PipelineStage(stage_id=0, config=slm_config, start_layer=0, end_layer=3)
        self.stage1 = PipelineStage(stage_id=1, config=slm_config, start_layer=3, end_layer=6)

        # Place stages on designated devices
        self.stage0.to(self.device_mgr.get_stage_device(0))
        self.stage1.to(self.device_mgr.get_stage_device(1))
        self.stage0.eval()
        self.stage1.eval()

    def set_weights(self, full_model: SequentialSLM):
        full_model.copy_weights_to_pipeline_stages(self.stage0, self.stage1)

class SequentialInferenceEngine(BaseInferenceEngine):
    """
    Baseline Sequential Inference Engine.
    Processes micro-batches one by one sequentially across Stage 0 and Stage 1.
    No stage compute overlap.
    """

    def run(self, input_batches: List[torch.Tensor]) -> Tuple[List[torch.Tensor], PipelineMetrics]:
        self.device_mgr.synchronize_all()
        start_pipeline_time = time.perf_counter()

        outputs = []
        timeline_events = []
        stage_active = {0: 0.0, 1: 0.0}
        total_transfer_time = 0.0

        for mb_idx, mb_input in enumerate(input_batches):
            # --- Stage 0 Computation ---
            s0_device = self.device_mgr.get_stage_device(0)
            mb_input_dev = mb_input.to(s0_device)

            self.device_mgr.synchronize_stage(0)
            t_s0_start = time.perf_counter()
            with torch.no_grad():
                with torch.cuda.stream(self.device_mgr.get_stage_stream(0)) if self.device_mgr.get_stage_stream(0) else torch.no_grad():
                    activation = self.stage0(mb_input_dev)
            self.device_mgr.synchronize_stage(0)
            t_s0_end = time.perf_counter()
            s0_duration = (t_s0_end - t_s0_start) * 1000.0
            stage_active[0] += s0_duration

            timeline_events.append({
                "mb_id": mb_idx,
                "stage_id": 0,
                "start_ms": (t_s0_start - start_pipeline_time) * 1000.0,
                "end_ms": (t_s0_end - start_pipeline_time) * 1000.0,
                "duration_ms": s0_duration
            })

            # --- Activation Transfer Stage 0 -> Stage 1 ---
            activation_dst, xfer_ms = self.device_mgr.transfer_activation(
                activation, src_stage=0, dst_stage=1, async_op=False
            )
            total_transfer_time += xfer_ms

            # --- Stage 1 Computation ---
            self.device_mgr.synchronize_stage(1)
            t_s1_start = time.perf_counter()
            with torch.no_grad():
                with torch.cuda.stream(self.device_mgr.get_stage_stream(1)) if self.device_mgr.get_stage_stream(1) else torch.no_grad():
                    logits = self.stage1(activation_dst)
            self.device_mgr.synchronize_stage(1)
            t_s1_end = time.perf_counter()
            s1_duration = (t_s1_end - t_s1_start) * 1000.0
            stage_active[1] += s1_duration

            timeline_events.append({
                "mb_id": mb_idx,
                "stage_id": 1,
                "start_ms": (t_s1_start - start_pipeline_time) * 1000.0,
                "end_ms": (t_s1_end - start_pipeline_time) * 1000.0,
                "duration_ms": s1_duration
            })

            outputs.append(logits.cpu())

        self.device_mgr.synchronize_all()
        end_pipeline_time = time.perf_counter()
        total_time_ms = (end_pipeline_time - start_pipeline_time) * 1000.0

        total_requests = len(input_batches) * self.config.micro_batch_size
        total_tokens = total_requests * self.config.seq_len

        throughput_req = total_requests / (total_time_ms / 1000.0) if total_time_ms > 0 else 0
        throughput_tok = total_tokens / (total_time_ms / 1000.0) if total_time_ms > 0 else 0

        total_stage_capacity = total_time_ms * 2.0
        active_capacity = stage_active[0] + stage_active[1]
        bubble_ratio = max(0.0, (total_stage_capacity - active_capacity) / total_stage_capacity)

        stage_idle = {
            0: total_time_ms - stage_active[0],
            1: total_time_ms - stage_active[1]
        }

        metrics = PipelineMetrics(
            mode="Sequential",
            total_time_ms=total_time_ms,
            throughput_req_per_sec=throughput_req,
            throughput_tok_per_sec=throughput_tok,
            stage_active_time_ms=stage_active,
            stage_idle_time_ms=stage_idle,
            transfer_time_ms=total_transfer_time,
            bubble_ratio=bubble_ratio,
            timeline_events=timeline_events
        )

        return outputs, metrics

class PipelineParallelEngine(BaseInferenceEngine):
    """
    Pipeline-Parallel Inference Engine.
    Overlaps Stage 0 computation of Micro-Batch N+1 with Stage 1 computation of Micro-Batch N.
    Uses asynchronous CUDA streams and events.
    """

    def run(self, input_batches: List[torch.Tensor]) -> Tuple[List[torch.Tensor], PipelineMetrics]:
        self.device_mgr.synchronize_all()
        start_pipeline_time = time.perf_counter()

        num_micro_batches = len(input_batches)
        outputs: Dict[int, torch.Tensor] = {}
        timeline_events = []
        stage_active = {0: 0.0, 1: 0.0}
        total_transfer_time = 0.0

        # Number of pipeline steps = num_micro_batches + (num_stages - 1)
        total_steps = num_micro_batches + 1

        # Track intermediate activations per micro-batch
        activations: Dict[int, torch.Tensor] = {}

        for step in range(total_steps):
            # Determine which micro-batch is at Stage 0 and Stage 1 during this step
            mb_s0 = step if step < num_micro_batches else None
            mb_s1 = (step - 1) if (step - 1 >= 0 and step - 1 < num_micro_batches) else None

            # --- Launch Stage 0 for mb_s0 ---
            t_s0_start, t_s0_end = 0.0, 0.0
            if mb_s0 is not None:
                s0_device = self.device_mgr.get_stage_device(0)
                mb_input_dev = input_batches[mb_s0].to(s0_device)

                self.device_mgr.synchronize_stage(0)
                t_s0_start = time.perf_counter()
                with torch.no_grad():
                    stream0 = self.device_mgr.get_stage_stream(0)
                    if stream0:
                        with torch.cuda.stream(stream0):
                            act = self.stage0(mb_input_dev)
                    else:
                        act = self.stage0(mb_input_dev)

                self.device_mgr.synchronize_stage(0)
                t_s0_end = time.perf_counter()
                s0_dur = (t_s0_end - t_s0_start) * 1000.0
                stage_active[0] += s0_dur

                timeline_events.append({
                    "mb_id": mb_s0,
                    "stage_id": 0,
                    "start_ms": (t_s0_start - start_pipeline_time) * 1000.0,
                    "end_ms": (t_s0_end - start_pipeline_time) * 1000.0,
                    "duration_ms": s0_dur
                })

                # Transfer activation to Stage 1 asynchronously
                act_dst, xfer_ms = self.device_mgr.transfer_activation(
                    act, src_stage=0, dst_stage=1, async_op=True
                )
                total_transfer_time += xfer_ms
                activations[mb_s0] = act_dst

            # --- Launch Stage 1 for mb_s1 ---
            if mb_s1 is not None:
                act_input = activations[mb_s1]
                self.device_mgr.synchronize_stage(1)
                t_s1_start = time.perf_counter()
                with torch.no_grad():
                    stream1 = self.device_mgr.get_stage_stream(1)
                    if stream1:
                        with torch.cuda.stream(stream1):
                            logits = self.stage1(act_input)
                    else:
                        logits = self.stage1(act_input)

                self.device_mgr.synchronize_stage(1)
                t_s1_end = time.perf_counter()
                s1_dur = (t_s1_end - t_s1_start) * 1000.0
                stage_active[1] += s1_dur

                timeline_events.append({
                    "mb_id": mb_s1,
                    "stage_id": 1,
                    "start_ms": (t_s1_start - start_pipeline_time) * 1000.0,
                    "end_ms": (t_s1_end - start_pipeline_time) * 1000.0,
                    "duration_ms": s1_dur
                })

                outputs[mb_s1] = logits.cpu()

        self.device_mgr.synchronize_all()
        end_pipeline_time = time.perf_counter()
        total_time_ms = (end_pipeline_time - start_pipeline_time) * 1000.0

        ordered_outputs = [outputs[i] for i in range(num_micro_batches)]

        total_requests = len(input_batches) * self.config.micro_batch_size
        total_tokens = total_requests * self.config.seq_len

        throughput_req = total_requests / (total_time_ms / 1000.0) if total_time_ms > 0 else 0
        throughput_tok = total_tokens / (total_time_ms / 1000.0) if total_time_ms > 0 else 0

        total_stage_capacity = total_time_ms * 2.0
        active_capacity = stage_active[0] + stage_active[1]
        bubble_ratio = max(0.0, (total_stage_capacity - active_capacity) / total_stage_capacity)

        stage_idle = {
            0: total_time_ms - stage_active[0],
            1: total_time_ms - stage_active[1]
        }

        metrics = PipelineMetrics(
            mode="Pipeline-Parallel",
            total_time_ms=total_time_ms,
            throughput_req_per_sec=throughput_req,
            throughput_tok_per_sec=throughput_tok,
            stage_active_time_ms=stage_active,
            stage_idle_time_ms=stage_idle,
            transfer_time_ms=total_transfer_time,
            bubble_ratio=bubble_ratio,
            timeline_events=timeline_events
        )

        return ordered_outputs, metrics
