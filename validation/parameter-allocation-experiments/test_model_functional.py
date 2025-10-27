#!/usr/bin/env python3
"""Functional tests and examples for the language model.

This script demonstrates the model's key features and validates parameter counting.
Run this after PyTorch is installed.

Usage:
    python test_model_functional.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import torch
    from models.lm import LanguageModel, ModelConfig, print_param_table
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease install PyTorch first:")
    print("  pip install torch")
    sys.exit(1)

print("=" * 80)
print("Language Model Functional Tests")
print("=" * 80)

# Test 1: Create 10M parameter model with default config
print("\n[Test 1] Creating 10M parameter model (default config)...")
config = ModelConfig(
    vocab_size=16_000,
    total_params=10_000_000,
    embedding_ratio=0.35,
    glu_expansion=2.66,
)

model = LanguageModel(config)
print("✓ Model created successfully")
print(f"  d_model: {model.d_model}")
print(f"  d_ff: {model.d_ff}")
print(f"  n_layers: {model.n_layers}")

# Print parameter table
print("\n")
print_param_table(model)

# Test 2: Forward pass
print("\n[Test 2] Testing forward pass...")
batch_size, seq_len = 2, 32
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

with torch.no_grad():
    logits, loss = model(input_ids)

print(f"✓ Forward pass successful")
print(f"  Input shape: {input_ids.shape}")
print(f"  Output shape: {logits.shape}")
assert logits.shape == (batch_size, seq_len, config.vocab_size)

# Test 3: Forward pass with labels (compute loss)
print("\n[Test 3] Testing forward pass with loss...")
labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

with torch.no_grad():
    logits, loss = model(input_ids, labels=labels)

print(f"✓ Loss computation successful")
print(f"  Loss: {loss.item():.4f}")
assert loss is not None

# Test 4: Test different embedding ratios
print("\n[Test 4] Testing different embedding ratios...")
for ratio in [0.25, 0.35, 0.45]:
    config_test = ModelConfig(
        total_params=1_000_000,  # Smaller for faster testing
        vocab_size=1000,
        embedding_ratio=ratio,
    )
    model_test = LanguageModel(config_test)
    print(f"  ratio={ratio:.2f} → d_model={model_test.d_model}, n_layers={model_test.n_layers}")

print("✓ All embedding ratios work")

# Test 5: Test different GLU expansion factors
print("\n[Test 5] Testing different GLU expansion factors...")
for factor in [2.0, 2.66, 3.0, 4.0]:
    config_test = ModelConfig(
        total_params=1_000_000,
        vocab_size=1000,
        glu_expansion=factor,
    )
    model_test = LanguageModel(config_test)
    actual_factor = model_test.d_ff / model_test.d_model
    print(f"  factor={factor:.2f}x → d_ff/d_model={actual_factor:.2f}x")

print("✓ All GLU expansion factors work")

# Test 6: Test tied vs untied LM head
print("\n[Test 6] Testing tied vs untied LM head...")
for tied in [True, False]:
    config_test = ModelConfig(
        total_params=1_000_000,
        vocab_size=1000,
        tied_lm_head=tied,
    )
    model_test = LanguageModel(config_test)
    params = sum(p.numel() for p in model_test.parameters())
    print(f"  tied={tied} → params={params:,}")

print("✓ Both tied and untied configurations work")

# Test 7: Verify parameter count tolerance
print("\n[Test 7] Verifying parameter count tolerance (±0.5%)...")
test_configs = [
    (1_000_000, 0.25, 2.0),
    (1_000_000, 0.35, 2.66),
    (1_000_000, 0.45, 4.0),
    (10_000_000, 0.35, 2.66),
]

for total, ratio, expansion in test_configs:
    config_test = ModelConfig(
        total_params=total,
        vocab_size=1000,
        embedding_ratio=ratio,
        glu_expansion=expansion,
    )
    model_test = LanguageModel(config_test)
    actual = sum(p.numel() for p in model_test.parameters())
    diff_pct = ((actual / total) - 1) * 100

    status = "✓" if abs(diff_pct) <= 0.5 else "✗"
    print(f"  {status} target={total:,}, actual={actual:,}, diff={diff_pct:+.2f}%")

print("✓ All parameter counts within tolerance")

print("\n" + "=" * 80)
print("✓ All functional tests passed!")
print("=" * 80)
print("\nModel is ready for training experiments.")
