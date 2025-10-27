"""Unit tests for language model implementation."""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))

from models.lm import (
    LanguageModel,
    ModelConfig,
    SwiGLU,
    MultiHeadAttention,
    TransformerBlock,
    compute_model_dims,
    count_parameters,
    print_param_table,
)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()

        assert config.vocab_size == 16_000
        assert config.total_params == 10_000_000
        assert config.embedding_ratio == 0.35
        assert config.glu_expansion == 2.66
        assert config.n_heads == 8
        assert config.tied_lm_head is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelConfig(
            vocab_size=8000,
            total_params=5_000_000,
            embedding_ratio=0.25,
            glu_expansion=2.0,
        )

        assert config.vocab_size == 8000
        assert config.total_params == 5_000_000
        assert config.embedding_ratio == 0.25
        assert config.glu_expansion == 2.0


class TestComputeModelDims:
    """Tests for compute_model_dims function."""

    def test_dims_10m_params(self):
        """Test dimension computation for 10M params."""
        config = ModelConfig(
            total_params=10_000_000,
            vocab_size=16_000,
            embedding_ratio=0.35,
            glu_expansion=2.66,
        )

        d_model, d_ff, n_layers = compute_model_dims(config)

        # Check dimensions are reasonable
        assert d_model > 0
        assert d_ff > 0
        assert n_layers > 0

        # Check d_model is divisible by n_heads
        assert d_model % config.n_heads == 0

        # Check d_ff matches expansion factor roughly
        assert abs(d_ff / d_model - config.glu_expansion) < 0.5

    def test_tied_vs_untied(self):
        """Test that tied embeddings result in different dimensions."""
        config_tied = ModelConfig(tied_lm_head=True)
        config_untied = ModelConfig(tied_lm_head=False)

        d_model_tied, _, _ = compute_model_dims(config_tied)
        d_model_untied, _, _ = compute_model_dims(config_untied)

        # Untied should have smaller d_model (more params in output head)
        assert d_model_untied < d_model_tied


class TestSwiGLU:
    """Tests for SwiGLU activation."""

    def test_swiglu_forward(self):
        """Test SwiGLU forward pass."""
        batch_size, seq_len, d_model = 2, 10, 64
        d_ff = 128

        swiglu = SwiGLU(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = swiglu(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)

    def test_swiglu_param_count(self):
        """Test SwiGLU parameter count."""
        d_model, d_ff = 64, 128

        swiglu = SwiGLU(d_model, d_ff)
        params = count_parameters(swiglu)

        # gate: d_model * d_ff
        # up: d_model * d_ff
        # down: d_ff * d_model
        # Total: 3 * d_model * d_ff
        expected = 3 * d_model * d_ff

        assert params == expected


class TestMultiHeadAttention:
    """Tests for multi-head attention."""

    def test_attention_forward(self):
        """Test attention forward pass."""
        batch_size, seq_len, d_model = 2, 10, 64
        n_heads = 8

        attn = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output = attn(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)

    def test_attention_with_mask(self):
        """Test attention with causal mask."""
        batch_size, seq_len, d_model = 2, 10, 64
        n_heads = 8

        attn = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.view(1, 1, seq_len, seq_len)

        output = attn(x, mask=mask)

        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)

    def test_attention_param_count(self):
        """Test attention parameter count."""
        d_model, n_heads = 64, 8

        attn = MultiHeadAttention(d_model, n_heads)
        params = count_parameters(attn)

        # Q, K, V, O projections: 4 * d_model * d_model
        expected = 4 * d_model * d_model

        assert params == expected


class TestTransformerBlock:
    """Tests for transformer block."""

    def test_block_forward(self):
        """Test transformer block forward pass."""
        batch_size, seq_len, d_model = 2, 10, 64
        d_ff, n_heads = 128, 8

        block = TransformerBlock(d_model, d_ff, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output = block(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)

    def test_block_with_mask(self):
        """Test transformer block with mask."""
        batch_size, seq_len, d_model = 2, 10, 64
        d_ff, n_heads = 128, 8

        block = TransformerBlock(d_model, d_ff, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)

        output = block(x, mask=mask)

        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)


class TestLanguageModel:
    """Tests for complete language model."""

    def test_model_initialization(self):
        """Test model can be initialized."""
        config = ModelConfig(
            total_params=1_000_000,  # Smaller for faster testing
            vocab_size=1000,
            embedding_ratio=0.35,
        )

        model = LanguageModel(config)

        assert model is not None
        assert model.d_model > 0
        assert model.n_layers > 0

    def test_model_forward(self):
        """Test model forward pass."""
        config = ModelConfig(
            total_params=1_000_000,
            vocab_size=1000,
        )

        model = LanguageModel(config)
        batch_size, seq_len = 2, 10

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, loss = model(input_ids)

        # Check logits shape
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        # No labels, so loss should be None
        assert loss is None

    def test_model_forward_with_labels(self):
        """Test model forward pass with labels."""
        config = ModelConfig(
            total_params=1_000_000,
            vocab_size=1000,
        )

        model = LanguageModel(config)
        batch_size, seq_len = 2, 10

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, loss = model(input_ids, labels=labels)

        # Check shapes
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss is not None
        assert loss.shape == ()  # Scalar

    def test_param_count_tolerance(self):
        """Test parameter count is within ±0.5% tolerance."""
        config = ModelConfig(
            total_params=1_000_000,
            vocab_size=1000,
        )

        model = LanguageModel(config)
        actual = count_parameters(model)
        target = config.total_params

        ratio = actual / target
        tolerance = 0.005  # ±0.5%

        assert 1 - tolerance <= ratio <= 1 + tolerance, \
            f"Param count {actual:,} is {ratio:.2%} of target {target:,}"

    def test_embedding_ratios(self):
        """Test different embedding ratios."""
        ratios = [0.25, 0.35, 0.45]

        for ratio in ratios:
            config = ModelConfig(
                total_params=1_000_000,
                vocab_size=1000,
                embedding_ratio=ratio,
            )

            model = LanguageModel(config)

            # Count embedding params
            embed_params = sum(p.numel() for p in model.token_embedding.parameters())
            total_params = count_parameters(model)

            # Check embedding ratio is approximately correct
            actual_ratio = embed_params / total_params
            # Allow some slack due to discretization
            assert abs(actual_ratio - ratio) < 0.1, \
                f"Embedding ratio {actual_ratio:.2%} far from target {ratio:.2%}"

    def test_glu_expansion_factors(self):
        """Test different GLU expansion factors."""
        factors = [2.0, 2.66, 3.0, 4.0]

        for factor in factors:
            config = ModelConfig(
                total_params=1_000_000,
                vocab_size=1000,
                glu_expansion=factor,
            )

            model = LanguageModel(config)

            # Check d_ff matches expansion
            assert abs(model.d_ff / model.d_model - factor) < 0.5

    def test_tied_vs_untied_head(self):
        """Test tied vs untied LM head."""
        # Tied head
        config_tied = ModelConfig(
            total_params=1_000_000,
            vocab_size=1000,
            tied_lm_head=True,
        )
        model_tied = LanguageModel(config_tied)

        # Untied head
        config_untied = ModelConfig(
            total_params=1_000_000,
            vocab_size=1000,
            tied_lm_head=False,
        )
        model_untied = LanguageModel(config_untied)

        # Both should have similar total params
        params_tied = count_parameters(model_tied)
        params_untied = count_parameters(model_untied)

        # Both within tolerance of target
        assert abs(params_tied - config_tied.total_params) / config_tied.total_params < 0.005
        assert abs(params_untied - config_untied.total_params) / config_untied.total_params < 0.005

    def test_causal_mask(self):
        """Test causal mask creation."""
        config = ModelConfig(total_params=1_000_000, vocab_size=1000)
        model = LanguageModel(config)

        seq_len = 5
        mask = model.get_causal_mask(seq_len, torch.device('cpu'))

        # Check shape
        assert mask.shape == (1, 1, seq_len, seq_len)

        # Check it's lower triangular
        mask_2d = mask[0, 0]
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    assert mask_2d[i, j] == 0
                else:
                    assert mask_2d[i, j] == 1


def test_print_param_table(capsys):
    """Test parameter table printing."""
    config = ModelConfig(
        total_params=1_000_000,
        vocab_size=1000,
    )

    model = LanguageModel(config)
    print_param_table(model)

    captured = capsys.readouterr()
    output = captured.out

    # Check key components are in output
    assert "Parameter Allocation Table" in output
    assert "Token Embedding" in output
    assert "Transformer Backbone" in output
    assert "TOTAL" in output
    assert "d_model" in output
    assert "glu_expansion" in output


def test_count_parameters():
    """Test parameter counting utility."""
    # Create simple model
    model = torch.nn.Linear(10, 5)
    params = count_parameters(model)

    # Linear layer: 10 * 5 weights + 5 biases = 55
    assert params == 55
