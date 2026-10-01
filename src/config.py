import dataclasses
from typing import List, Optional

@dataclasses.dataclass
class SLMConfig:
    """Configuration for Small Language Model architecture."""
    vocab_size: int = 1000
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    intermediate_dim: int = 512
    max_seq_len: int = 2048
    dropout: float = 0.0

@dataclasses.dataclass
class PipelineConfig:
    """Configuration for Pipeline-Parallel execution."""
    num_stages: int = 2
    micro_batch_size: int = 2
    total_requests: int = 4
    seq_len: int = 32
    force_cpu: bool = False
    force_simulated_gpu: bool = False
    simulate_comm_delay_ms: float = 0.0  # Optional delay to simulate inter-stage network transfer

    @property
    def layers_per_stage(self) -> int:
        return 3

