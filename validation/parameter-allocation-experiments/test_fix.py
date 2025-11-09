#!/usr/bin/env python3
"""
Quick test to verify the bug fixes are working.

This creates a random untrained model and evaluates it.
Expected result: PPL should be close to vocab_size (16,000).

Usage:
    python test_fix.py
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.lm import LanguageModel, ModelConfig


def test_random_model():
    """Test that a random untrained model gives PPL ≈ vocab_size."""
    print("="*70)
    print("TESTING BUG FIXES")
    print("="*70)
    print()

    # Create config
    config = ModelConfig(
        vocab_size=16000,
        total_params=10_000_000,
        embedding_ratio=0.35,
        glu_expansion=2.66,
        max_seq_length=512,
    )

    print(f"Creating random model with config:")
    print(f"  vocab_size: {config.vocab_size:,}")
    print(f"  total_params: {config.total_params:,}")
    print()

    # Create random model (don't load any checkpoint!)
    model = LanguageModel(config)
    model.eval()

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print()

    # Create random batch
    batch_size = 4
    seq_length = 128

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    labels = input_ids.clone()

    print(f"Testing on random batch:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_length: {seq_length}")
    print(f"  total positions: {batch_size * seq_length:,}")
    print()

    # Forward pass
    with torch.no_grad():
        logits, loss = model(input_ids, labels=labels)

    print(f"Forward pass results:")
    print(f"  logits shape: {logits.shape}")
    print(f"  loss (sum): {loss.item():.2f}")
    print()

    # Calculate perplexity manually
    # Model now returns sum of losses (reduction='sum')
    # We need to divide by number of tokens

    # Count tokens (shifted, no padding in this test)
    shift_labels = labels[:, 1:]
    num_tokens = shift_labels.numel()

    avg_loss = loss.item() / num_tokens
    perplexity = math.exp(avg_loss)

    print(f"Metrics:")
    print(f"  num_tokens (shifted): {num_tokens:,}")
    print(f"  avg_loss: {avg_loss:.4f}")
    print(f"  perplexity: {perplexity:.2f}")
    print()

    # Expected values
    expected_loss = math.log(config.vocab_size)
    expected_ppl = config.vocab_size

    print(f"Expected (for random uniform predictions):")
    print(f"  avg_loss: {expected_loss:.4f} (ln({config.vocab_size}))")
    print(f"  perplexity: ~{expected_ppl:,}")
    print()

    # Check if close
    print("="*70)
    print("VERIFICATION")
    print("="*70)
    print()

    # Allow 30% tolerance since it's random
    tolerance = 0.3

    if abs(avg_loss - expected_loss) / expected_loss < tolerance:
        print("✅ PASS: Loss is close to expected!")
        print(f"   Got {avg_loss:.4f}, expected ~{expected_loss:.4f}")
    else:
        print("❌ FAIL: Loss is not close to expected!")
        print(f"   Got {avg_loss:.4f}, expected ~{expected_loss:.4f}")
        print(f"   Difference: {abs(avg_loss - expected_loss):.4f}")

    print()

    if abs(perplexity - expected_ppl) / expected_ppl < tolerance:
        print("✅ PASS: Perplexity is close to expected!")
        print(f"   Got {perplexity:.2f}, expected ~{expected_ppl:,}")
    else:
        print("❌ FAIL: Perplexity is not close to expected!")
        print(f"   Got {perplexity:.2f}, expected ~{expected_ppl:,}")
        print(f"   Ratio: {perplexity / expected_ppl:.2f}x")

    print()

    # Check that it's not the old buggy value
    if abs(perplexity - 1.054) < 0.1:
        print("❌ CRITICAL: Got PPL ~1.054 - the bug is NOT fixed!")
        print("   The model is still giving unrealistic results.")
    elif perplexity > 1000:
        print("✅ GOOD: Perplexity is high (> 1000), bug is likely fixed!")
    else:
        print("⚠️  WARNING: Perplexity is suspicious")
        print(f"   Got {perplexity:.2f}, expected ~{expected_ppl:,}")

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()

    if abs(perplexity - expected_ppl) / expected_ppl < tolerance:
        print("✅ BUG FIX VERIFIED!")
        print()
        print("The model is correctly:")
        print("  1. Shifting tokens for next-token prediction")
        print("  2. Using reduction='sum' and averaging correctly")
        print("  3. Counting tokens properly")
        print()
        print("Next steps:")
        print("  1. Re-evaluate your Phase 1 checkpoints")
        print("  2. Expect PPL of 80-250 for trained models")
        print("  3. Re-run Phase 1 if rankings are important")
    else:
        print("❌ SOMETHING IS STILL WRONG")
        print()
        print("The perplexity is not matching expected values.")
        print("Please check:")
        print("  1. Did all file edits save correctly?")
        print("  2. Are you using the updated code?")
        print("  3. Try restarting Python/reloading modules")

    print()


if __name__ == '__main__':
    try:
        test_random_model()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you're in the correct directory and have dependencies installed:")
        print("  cd validation/parameter-allocation-experiments")
        print("  pip install torch")
