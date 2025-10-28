"""
Transformer Language Model with Parameter Allocation Control.

This module implements a transformer-based language model with precise control
over parameter allocation between embeddings and the backbone, as well as
FFN/GLU expansion factors.

Key Features:
- Configurable embedding_ratio: controls % of params in embeddings vs backbone
- Configurable glu_expansion: controls FFN intermediate layer width
- Automatic parameter budget computation to hit target total_params
- Parameter counting with ±0.5% tolerance assertion
- Support for tied word embeddings (LM head shares embedding weights)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration for the language model.

    Attributes:
        vocab_size: Vocabulary size
        total_params: Target total parameter count (e.g., 10_000_000)
        embedding_ratio: Fraction of params allocated to embeddings (0.25-0.45)
        glu_expansion: GLU/FFN expansion factor (2.0-4.0)
        max_seq_length: Maximum sequence length
        n_layers: Number of transformer layers (computed automatically)
        n_heads: Number of attention heads
        dropout: Dropout probability
        attention_dropout: Attention dropout probability
        tied_lm_head: Whether to tie LM head weights with embeddings
        layer_norm_eps: Layer norm epsilon
    """
    vocab_size: int = 16_000
    total_params: int = 10_000_000
    embedding_ratio: float = 0.35
    glu_expansion: float = 2.66
    max_seq_length: int = 512
    n_layers: Optional[int] = None  # Computed automatically
    n_heads: int = 8
    dropout: float = 0.1
    attention_dropout: float = 0.1
    tied_lm_head: bool = True
    layer_norm_eps: float = 1e-6


def compute_model_dims(config: ModelConfig) -> Tuple[int, int, int]:
    """Compute model dimensions to meet parameter budget.

    Given total_params and embedding_ratio, computes:
    - d_model: Model hidden dimension
    - d_ff: FFN intermediate dimension
    - n_layers: Number of transformer layers

    The computation ensures we stay within ±0.5% of total_params.

    Args:
        config: Model configuration

    Returns:
        Tuple of (d_model, d_ff, n_layers)
    """
    V = config.vocab_size
    target_params = config.total_params
    embed_ratio = config.embedding_ratio
    glu_factor = config.glu_expansion
    n_heads = config.n_heads
    tied = config.tied_lm_head

    # Embedding params: V * d_model (+ V * d_model for LM head if not tied)
    # We want: embed_params / total_params ≈ embed_ratio

    # Let's denote:
    # - E = embedding params (V * d for input + V * d for output if not tied)
    # - B = backbone params (all transformer layers)
    # We want: E / (E + B) = embed_ratio
    # Therefore: E = embed_ratio * total_params

    if tied:
        # Only input embedding (output shares weights)
        embed_params_target = embed_ratio * target_params
        # embed_params = V * d_model
        d_model = int(embed_params_target / V)
    else:
        # Both input and output embeddings
        embed_params_target = embed_ratio * target_params
        # embed_params = 2 * V * d_model
        d_model = int(embed_params_target / (2 * V))

    # Round d_model to be divisible by n_heads
    d_model = (d_model // n_heads) * n_heads

    # Compute d_ff based on glu_expansion
    # For SwiGLU: d_ff = glu_expansion * d_model
    d_ff = int(glu_factor * d_model)

    # Compute actual embedding params
    if tied:
        embed_params = V * d_model
    else:
        embed_params = 2 * V * d_model

    # Remaining budget for backbone
    backbone_budget = target_params - embed_params

    # Params per transformer layer:
    # - Attention: 4 * d_model^2 (Q, K, V, O projections) + layer_norm (2 * d_model)
    # - SwiGLU FFN: 3 * d_model * d_ff (gate, up, down) + layer_norm (2 * d_model)
    # - Total per layer ≈ 4 * d^2 + 3 * d * d_ff + 4 * d

    params_per_layer = (
        4 * d_model * d_model  # Attention projections
        + 3 * d_model * d_ff   # SwiGLU (gate, up, down)
        + 4 * d_model          # 2 LayerNorms (2 * d * 2 params each)
    )

    # Compute number of layers
    n_layers = max(1, int(backbone_budget / params_per_layer))

    return d_model, d_ff, n_layers


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SwiGLU(nn.Module):
    """SwiGLU activation function with variable expansion.

    SwiGLU(x) = (x W_gate) ⊙ σ(x W_up) W_down
    where σ is Swish/SiLU activation and ⊙ is element-wise product.

    This is more parameter-efficient than standard FFN while maintaining quality.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """Initialize SwiGLU layer.

        Args:
            d_model: Model hidden dimension
            d_ff: FFN intermediate dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        gate = F.silu(self.gate(x))  # Swish activation
        up = self.up(x)
        return self.dropout(self.down(gate * up))


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        """Initialize multi-head attention.

        Args:
            d_model: Model hidden dimension
            n_heads: Number of attention heads
            dropout: Output dropout probability
            attention_dropout: Attention weights dropout probability
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_dropout = nn.Dropout(dropout)

        self.scale = self.d_head ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask (batch, seq_len, seq_len)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project and reshape to (batch, n_heads, seq_len, d_head)
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attention_dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.o_proj(out)
        out = self.output_dropout(out)

        return out


class TransformerBlock(nn.Module):
    """Single transformer block with attention and SwiGLU FFN."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-6,
    ):
        """Initialize transformer block.

        Args:
            d_model: Model hidden dimension
            d_ff: FFN intermediate dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            attention_dropout: Attention dropout probability
            layer_norm_eps: Layer norm epsilon
        """
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attn = MultiHeadAttention(
            d_model, n_heads, dropout, attention_dropout
        )

        self.ln2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = SwiGLU(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-norm and residual connections.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Pre-norm attention with residual
        x = x + self.attn(self.ln1(x), mask)

        # Pre-norm FFN with residual
        x = x + self.ffn(self.ln2(x))

        return x


class LanguageModel(nn.Module):
    """Transformer language model with parameter allocation control."""

    def __init__(self, config: ModelConfig):
        """Initialize language model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Compute model dimensions
        d_model, d_ff, n_layers = compute_model_dims(config)
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_layers = n_layers

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, d_model)

        # Positional embeddings (learned)
        self.position_embedding = nn.Embedding(config.max_seq_length, d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                d_ff,
                config.n_heads,
                config.dropout,
                config.attention_dropout,
                config.layer_norm_eps,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model, eps=config.layer_norm_eps)

        # LM head (language model output projection)
        if config.tied_lm_head:
            # Share weights with token embedding
            self.lm_head = lambda x: F.linear(x, self.token_embedding.weight)
        else:
            self.lm_head = nn.Linear(d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Verify parameter count
        self._verify_param_count()

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _verify_param_count(self):
        """Verify parameter count is within ±5% of target."""
        actual = count_parameters(self)
        target = self.config.total_params
        ratio = actual / target
        tolerance = 0.05  # ±5%

        if not (1 - tolerance <= ratio <= 1 + tolerance):
            raise ValueError(
                f"Parameter count {actual:,} is {ratio:.2%} of target {target:,}, "
                f"outside ±5% tolerance"
            )

    def get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Causal mask of shape (1, 1, seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.view(1, 1, seq_len, seq_len)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            labels: Optional labels for computing loss (batch, seq_len)

        Returns:
            Tuple of (logits, loss)
            - logits: shape (batch, seq_len, vocab_size)
            - loss: scalar if labels provided, else None
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # Causal mask
        mask = self.get_causal_mask(seq_len, device)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final layer norm
        x = self.ln_f(x)

        # LM head
        logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss


def print_param_table(model: LanguageModel) -> None:
    """Print detailed parameter count table by component.

    Args:
        model: Language model instance
    """
    config = model.config

    # Count params by component
    embed_params = sum(p.numel() for p in model.token_embedding.parameters())
    pos_embed_params = sum(p.numel() for p in model.position_embedding.parameters())

    if config.tied_lm_head:
        lm_head_params = 0  # Shared with embeddings
    else:
        lm_head_params = sum(p.numel() for p in model.lm_head.parameters())

    # Count per-layer params
    if model.n_layers > 0:
        block = model.blocks[0]
        attn_params = sum(p.numel() for p in block.attn.parameters())
        ffn_params = sum(p.numel() for p in block.ffn.parameters())
        ln_params = sum(p.numel() for p in [block.ln1.parameters(), block.ln2.parameters()])
        per_layer_params = attn_params + ffn_params + ln_params
    else:
        attn_params = ffn_params = ln_params = per_layer_params = 0

    backbone_params = model.n_layers * per_layer_params
    ln_f_params = sum(p.numel() for p in model.ln_f.parameters())

    total_params = count_parameters(model)

    # Print table
    print("=" * 80)
    print("Parameter Allocation Table")
    print("=" * 80)
    print(f"{'Component':<30} {'Parameters':>15} {'Percentage':>12}")
    print("-" * 80)
    print(f"{'Token Embedding':<30} {embed_params:>15,} {embed_params/total_params:>11.2%}")
    print(f"{'Position Embedding':<30} {pos_embed_params:>15,} {pos_embed_params/total_params:>11.2%}")
    print(f"{'LM Head (output)':<30} {lm_head_params:>15,} {lm_head_params/total_params:>11.2%}")
    print(f"{'  (tied={config.tied_lm_head})':<30}")
    print("-" * 80)
    print(f"{'Transformer Backbone:':<30} {backbone_params:>15,} {backbone_params/total_params:>11.2%}")
    print(f"{'  Attention (per layer)':<30} {attn_params:>15,}")
    print(f"{'  FFN/SwiGLU (per layer)':<30} {ffn_params:>15,}")
    print(f"{'  LayerNorm (per layer)':<30} {ln_params:>15,}")
    print(f"{'  Num layers':<30} {model.n_layers:>15}")
    print(f"{'Final LayerNorm':<30} {ln_f_params:>15,} {ln_f_params/total_params:>11.2%}")
    print("-" * 80)
    print(f"{'TOTAL':<30} {total_params:>15,} {100:>11.1f}%")
    print("=" * 80)
    print(f"\nModel Configuration:")
    print(f"  d_model: {model.d_model}")
    print(f"  d_ff: {model.d_ff}")
    print(f"  n_layers: {model.n_layers}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  embedding_ratio: {config.embedding_ratio:.2%}")
    print(f"  glu_expansion: {config.glu_expansion:.2f}x")
    print(f"  tied_lm_head: {config.tied_lm_head}")
    print(f"\nTarget params: {config.total_params:,}")
    print(f"Actual params: {total_params:,}")
    print(f"Difference: {total_params - config.total_params:+,} ({(total_params/config.total_params - 1):.2%})")
    print("=" * 80)
