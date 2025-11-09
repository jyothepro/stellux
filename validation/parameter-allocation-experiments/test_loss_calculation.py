#!/usr/bin/env python3
"""
Test script to verify loss calculation is correct.

Run this on your Lambda GPU box where you have the training code.

Usage:
    python test_loss_calculation.py --checkpoint path/to/checkpoint.pt
"""

import argparse
import math
import torch
import torch.nn.functional as F


def test_random_model_loss(vocab_size=16000, batch_size=4, seq_length=128):
    """
    Test loss calculation with random predictions.
    Should give loss ≈ ln(vocab_size) and PPL ≈ vocab_size.
    """
    print("="*70)
    print("TEST 1: RANDOM MODEL BASELINE")
    print("="*70)

    # Create random logits (uniform distribution)
    logits = torch.randn(batch_size, seq_length, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_length))

    print(f"\nLogits shape: {logits.shape}")
    print(f"Labels shape: {labels.shape}")

    # Method 1: Cross-entropy with reduction='mean'
    loss_mean = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        reduction='mean'
    )
    ppl_mean = math.exp(loss_mean.item())

    print(f"\nMethod 1 (reduction='mean'):")
    print(f"  Loss: {loss_mean.item():.4f}")
    print(f"  PPL: {ppl_mean:.2f}")

    # Method 2: Cross-entropy with reduction='sum' then manual averaging
    loss_sum = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        reduction='sum'
    )
    total_tokens = batch_size * seq_length
    avg_loss = loss_sum.item() / total_tokens
    ppl_sum = math.exp(avg_loss)

    print(f"\nMethod 2 (reduction='sum' + manual average):")
    print(f"  Total loss: {loss_sum.item():.4f}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Avg loss: {avg_loss:.4f}")
    print(f"  PPL: {ppl_sum:.2f}")

    # Expected values
    expected_loss = math.log(vocab_size)
    expected_ppl = vocab_size

    print(f"\nExpected (for uniform random):")
    print(f"  Loss: ~{expected_loss:.4f} (ln({vocab_size}))")
    print(f"  PPL: ~{expected_ppl}")

    # Check if close
    print(f"\nVerification:")
    if abs(loss_mean.item() - expected_loss) < 1.0:
        print(f"  ✓ Method 1 matches expected (~{expected_loss:.2f})")
    else:
        print(f"  ❌ Method 1 doesn't match! Got {loss_mean.item():.4f}, expected ~{expected_loss:.4f}")

    if abs(avg_loss - expected_loss) < 1.0:
        print(f"  ✓ Method 2 matches expected (~{expected_loss:.2f})")
    else:
        print(f"  ❌ Method 2 doesn't match! Got {avg_loss:.4f}, expected ~{expected_loss:.4f}")

    print(f"\n{'='*70}\n")

    return loss_mean.item(), ppl_mean


def test_shifted_loss(vocab_size=16000, batch_size=4, seq_length=128):
    """
    Test loss calculation with shifted labels for next-token prediction.
    This is the correct way to compute language model loss.
    """
    print("="*70)
    print("TEST 2: NEXT-TOKEN PREDICTION (SHIFTED)")
    print("="*70)

    # Create random logits and labels
    logits = torch.randn(batch_size, seq_length, vocab_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

    print(f"\nOriginal shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  Input IDs: {input_ids.shape}")

    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    print(f"\nAfter shifting:")
    print(f"  Shift logits: {shift_logits.shape}")
    print(f"  Shift labels: {shift_labels.shape}")

    # Calculate loss
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction='mean'
    )
    ppl = math.exp(loss.item())

    print(f"\nResults:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  PPL: {ppl:.2f}")

    expected_loss = math.log(vocab_size)
    expected_ppl = vocab_size

    print(f"\nExpected (for uniform random):")
    print(f"  Loss: ~{expected_loss:.4f}")
    print(f"  PPL: ~{expected_ppl}")

    print(f"\n{'='*70}\n")

    return loss.item(), ppl


def test_loss_averaging_bug():
    """
    Demonstrate the common bug of averaging over batches instead of tokens.
    """
    print("="*70)
    print("TEST 3: COMMON BUG - AVERAGING OVER BATCHES")
    print("="*70)

    vocab_size = 16000
    batch_size = 4
    seq_length = 128
    num_batches = 10

    # Simulate multiple batches
    batch_losses = []
    batch_losses_sum = []

    for _ in range(num_batches):
        logits = torch.randn(batch_size, seq_length, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_length))

        # Calculate loss for this batch
        loss_mean = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            reduction='mean'
        )
        loss_sum = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            reduction='sum'
        )

        batch_losses.append(loss_mean.item())
        batch_losses_sum.append(loss_sum.item())

    # ❌ WRONG: Average of batch averages
    wrong_avg = sum(batch_losses) / len(batch_losses)
    wrong_ppl = math.exp(wrong_avg)

    # ✓ CORRECT: Sum all losses, divide by total tokens
    total_loss = sum(batch_losses_sum)
    total_tokens = num_batches * batch_size * seq_length
    correct_avg = total_loss / total_tokens
    correct_ppl = math.exp(correct_avg)

    print(f"\n❌ WRONG METHOD (averaging batch losses):")
    print(f"  Average loss: {wrong_avg:.4f}")
    print(f"  PPL: {wrong_ppl:.2f}")

    print(f"\n✓ CORRECT METHOD (sum then average):")
    print(f"  Total loss: {total_loss:.4f}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Average loss: {correct_avg:.4f}")
    print(f"  PPL: {correct_ppl:.2f}")

    print(f"\nDifference:")
    print(f"  Loss difference: {abs(wrong_avg - correct_avg):.6f}")
    print(f"  PPL difference: {abs(wrong_ppl - correct_ppl):.2f}")

    print(f"\n💡 NOTE: Both methods should give similar results for random predictions,")
    print(f"   but they can diverge significantly during training.")

    print(f"\n{'='*70}\n")


def test_ignore_index_bug():
    """
    Demonstrate the bug of using ignore_index incorrectly.
    """
    print("="*70)
    print("TEST 4: IGNORE_INDEX BUG")
    print("="*70)

    vocab_size = 16000
    batch_size = 4
    seq_length = 128
    ignore_index = -100

    # Create logits and labels
    logits = torch.randn(batch_size, seq_length, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_length))

    # Simulate padding: set some labels to ignore_index
    # Let's say 50% are padding
    mask = torch.rand(batch_size, seq_length) > 0.5
    labels[mask] = ignore_index

    num_real_tokens = (~mask).sum().item()
    num_ignored = mask.sum().item()

    print(f"\nSetup:")
    print(f"  Total positions: {batch_size * seq_length}")
    print(f"  Real tokens: {num_real_tokens}")
    print(f"  Padding (ignored): {num_ignored}")

    # Calculate loss with ignore_index
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=ignore_index,
        reduction='mean'
    )
    ppl = math.exp(loss.item())

    print(f"\nResults:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  PPL: {ppl:.2f}")

    print(f"\n💡 NOTE: This is correct! ignore_index properly handles padding.")
    print(f"   Loss is averaged over {num_real_tokens} real tokens, not {batch_size * seq_length} total.")

    # Show what happens if you forget ignore_index
    loss_no_ignore = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1).clamp(0, vocab_size-1),  # Clip to valid range
        reduction='mean'
    )

    print(f"\n⚠️  If you forgot ignore_index:")
    print(f"  Loss: {loss_no_ignore.item():.4f} (includes padding)")
    print(f"  PPL: {math.exp(loss_no_ignore.item()):.2f}")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Test loss calculation')
    parser.add_argument('--vocab-size', type=int, default=16000,
                       help='Vocabulary size')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for tests')
    parser.add_argument('--seq-length', type=int, default=128,
                       help='Sequence length for tests')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("LOSS CALCULATION VERIFICATION TESTS")
    print("="*70)
    print(f"\nVocab size: {args.vocab_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seq length: {args.seq_length}")
    print()

    # Run all tests
    test_random_model_loss(args.vocab_size, args.batch_size, args.seq_length)
    test_shifted_loss(args.vocab_size, args.batch_size, args.seq_length)
    test_loss_averaging_bug()
    test_ignore_index_bug()

    print("="*70)
    print("SUMMARY")
    print("="*70)
    print("\nKEY TAKEAWAYS:")
    print("\n1. Random model should give:")
    print(f"   - Loss ≈ {math.log(args.vocab_size):.2f} (ln({args.vocab_size}))")
    print(f"   - PPL ≈ {args.vocab_size}")
    print("\n2. Your Phase 1 results:")
    print("   - Loss ≈ 0.053 ❌ WAY TOO LOW!")
    print("   - PPL ≈ 1.054 ❌ IMPOSSIBLE!")
    print("\n3. Most likely bug:")
    print("   - Wrong loss averaging")
    print("   - Wrong loss function")
    print("   - Shape mismatch")
    print("\n4. To fix:")
    print("   - Use reduction='sum' then divide by total tokens")
    print("   - Shift logits/labels for next-token prediction")
    print("   - Test with random model first")
    print()


if __name__ == '__main__':
    main()
