import time
import torch
from enum import Enum
from typing import Dict, Tuple, Optional
from src.config import PipelineConfig

class ExecutionMode(Enum):
    PHYSICAL_MULTI_GPU = "Physical Multi-GPU (2+ GPUs)"
    SIMULATED_SINGLE_GPU = "Virtual Multi-GPU (1 GPU with CUDA Streams)"
    CPU = "CPU Multithreaded"

class DeviceManager:
    """
    Manages GPU hardware, CUDA streams, and virtual multi-GPU simulation for Pipeline Parallelism.
    
    Supports:
    1. Physical Multi-GPU: Stage 0 -> cuda:0, Stage 1 -> cuda:1
    2. Virtual Single-GPU: Stage 0 -> cuda:0 (Stream 0), Stage 1 -> cuda:0 (Stream 1)
    3. CPU Fallback: Stage 0 -> cpu, Stage 1 -> cpu
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.mode = self._determine_mode()
        self.stage_devices: Dict[int, torch.device] = {}
        self.stage_streams: Dict[int, Optional[torch.cuda.Stream]] = {}
        self.transfer_streams: Dict[Tuple[int, int], Optional[torch.cuda.Stream]] = {}
        self._setup_devices_and_streams()

    def _determine_mode(self) -> ExecutionMode:
        if self.config.force_cpu or not torch.cuda.is_available():
            return ExecutionMode.CPU
        
        gpu_count = torch.cuda.device_count()
        if gpu_count >= 2 and not self.config.force_simulated_gpu:
            return ExecutionMode.PHYSICAL_MULTI_GPU
        else:
            return ExecutionMode.SIMULATED_SINGLE_GPU

    def _setup_devices_and_streams(self):
        if self.mode == ExecutionMode.PHYSICAL_MULTI_GPU:
            self.stage_devices[0] = torch.device("cuda:0")
            self.stage_devices[1] = torch.device("cuda:1")
            self.stage_streams[0] = torch.cuda.Stream(device=self.stage_devices[0])
            self.stage_streams[1] = torch.cuda.Stream(device=self.stage_devices[1])
            self.transfer_streams[(0, 1)] = torch.cuda.Stream(device=self.stage_devices[1])

        elif self.mode == ExecutionMode.SIMULATED_SINGLE_GPU:
            dev = torch.device("cuda:0")
            self.stage_devices[0] = dev
            self.stage_devices[1] = dev
            self.stage_streams[0] = torch.cuda.Stream(device=dev)
            self.stage_streams[1] = torch.cuda.Stream(device=dev)
            self.transfer_streams[(0, 1)] = torch.cuda.Stream(device=dev)

        else:  # CPU mode
            dev = torch.device("cpu")
            self.stage_devices[0] = dev
            self.stage_devices[1] = dev
            self.stage_streams[0] = None
            self.stage_streams[1] = None
            self.transfer_streams[(0, 1)] = None

    def get_stage_device(self, stage_id: int) -> torch.device:
        return self.stage_devices[stage_id]

    def get_stage_stream(self, stage_id: int) -> Optional[torch.cuda.Stream]:
        return self.stage_streams[stage_id]

    def transfer_activation(
        self,
        tensor: torch.Tensor,
        src_stage: int,
        dst_stage: int,
        async_op: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """
        Transfers activation tensor from src_stage to dst_stage asynchronously or synchronously.
        Returns the transferred tensor and the elapsed transfer time (in ms).
        """
        start_time = time.perf_counter()
        dst_device = self.stage_devices[dst_stage]

        # Optional artificial communication delay simulation
        if self.config.simulate_comm_delay_ms > 0:
            time.sleep(self.config.simulate_comm_delay_ms / 1000.0)

        if self.mode == ExecutionMode.PHYSICAL_MULTI_GPU:
            # Physical GPU-to-GPU transfer (p2p/PCIe)
            with torch.cuda.stream(self.transfer_streams[(src_stage, dst_stage)]):
                transferred = tensor.to(dst_device, non_blocking=async_op)
                event = torch.cuda.Event()
                event.record(self.transfer_streams[(src_stage, dst_stage)])
                
                # Make dst_stage stream wait for transfer completion event
                dst_stream = self.stage_streams[dst_stage]
                if dst_stream:
                    dst_stream.wait_event(event)

        elif self.mode == ExecutionMode.SIMULATED_SINGLE_GPU:
            # Single GPU Virtual Stream transfer
            src_stream = self.stage_streams[src_stage]
            dst_stream = self.stage_streams[dst_stage]
            
            if src_stream and dst_stream:
                event = torch.cuda.Event()
                event.record(src_stream)
                dst_stream.wait_event(event)
            
            transferred = tensor.clone()

        else:
            # CPU copy
            transferred = tensor.detach().clone()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return transferred, elapsed_ms

    def synchronize_stage(self, stage_id: int):
        stream = self.stage_streams[stage_id]
        if stream is not None:
            stream.synchronize()

    def synchronize_all(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def get_summary(self) -> str:
        gpu_name = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "No GPU (CPU)"
        )
        return (
            f"Execution Mode: {self.mode.value}\n"
            f"Detected GPU: {gpu_name}\n"
            f"Stage 0 Device: {self.stage_devices[0]} | Stage 1 Device: {self.stage_devices[1]}\n"
            f"CUDA Available: {torch.cuda.is_available()} (Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0})"
        )
