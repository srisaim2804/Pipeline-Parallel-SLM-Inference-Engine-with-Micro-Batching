import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from src.config import SLMConfig

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for SLM sequence representation."""

    def __init__(self, hidden_dim: int, max_seq_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class EmbeddingLayer(nn.Module):
    """Token + Positional Embedding layer."""

    def __init__(self, config: SLMConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_encoding = PositionalEncoding(config.hidden_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        return self.dropout(x)

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention module."""

    def __init__(self, config: SLMConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        assert (
            self.head_dim * config.num_heads == config.hidden_dim
        ), "hidden_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Causal mask
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
        scores = scores + causal_mask

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(context)

class MLP(nn.Module):
    """Feed-Forward Network (MLP)."""

    def __init__(self, config: SLMConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_dim, config.intermediate_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))

class TransformerBlock(nn.Module):
    """Single Transformer Decoder Block."""

    def __init__(self, config: SLMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class LMHead(nn.Module):
    """Final LayerNorm and Language Model projection head."""

    def __init__(self, config: SLMConfig):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_f(x)
        return self.head(x)

class PipelineStage(nn.Module):
    """
    Represents one stage in the model pipeline.
    
    Stage 0: Includes EmbeddingLayer + first slice of TransformerBlocks (layers 0..2).
    Stage 1: Includes second slice of TransformerBlocks (layers 3..5) + LMHead.
    """

    def __init__(self, stage_id: int, config: SLMConfig, start_layer: int, end_layer: int):
        super().__init__()
        self.stage_id = stage_id
        self.config = config
        self.start_layer = start_layer
        self.end_layer = end_layer

        # Stage 0 holds embedding
        self.embedding = EmbeddingLayer(config) if stage_id == 0 else None

        # Transformer blocks assigned to this stage
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(start_layer, end_layer)
        ])

        # Stage 1 holds LM head
        self.lm_head = LMHead(config) if stage_id == 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stage_id == 0:
            # x is input_ids (batch_size, seq_len)
            hidden_states = self.embedding(x)
        else:
            # x is activation tensor from previous stage (batch_size, seq_len, hidden_dim)
            hidden_states = x

        for block in self.blocks:
            hidden_states = block(hidden_states)

        if self.stage_id == 1:
            # Final logits output (batch_size, seq_len, vocab_size)
            logits = self.lm_head(hidden_states)
            return logits

        return hidden_states

class SequentialSLM(nn.Module):
    """Full un-partitioned SLM model for baseline sequential comparison."""

    def __init__(self, config: SLMConfig):
        super().__init__()
        self.config = config
        self.embedding = EmbeddingLayer(config)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.lm_head = LMHead(config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)

    def copy_weights_to_pipeline_stages(self, stage0: PipelineStage, stage1: PipelineStage):
        """Helper to copy weights from full model to pipeline stages for exact parity testing."""
        # Stage 0
        stage0.embedding.load_state_dict(self.embedding.state_dict())
        for idx in range(3):
            stage0.blocks[idx].load_state_dict(self.blocks[idx].state_dict())
        
        # Stage 1
        for idx in range(3):
            stage1.blocks[idx].load_state_dict(self.blocks[idx + 3].state_dict())
        stage1.lm_head.load_state_dict(self.lm_head.state_dict())
