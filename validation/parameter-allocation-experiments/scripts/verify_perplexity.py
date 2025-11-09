#!/usr/bin/env python3
"""
Verify perplexity calculation correctness.

This script checks:
1. Correct perplexity formula: PPL = exp(average cross-entropy loss)
2. Base of logarithm (should be natural log, not log10 or log2)
3. Loss calculation correctness
4. Comparison with known baselines

Usage:
    python scripts/verify_perplexity.py --checkpoint <path> --data <path>
"""

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerplexityValidator:
    """Validates perplexity calculation correctness."""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def calculate_perplexity_manual(
        self,
        dataloader: DataLoader,
        max_batches: int = None
    ) -> Dict[str, float]:
        """
        Calculate perplexity using multiple methods to cross-verify.

        Returns:
            Dictionary with perplexity calculations using different methods
        """
        total_loss = 0.0
        total_tokens = 0
        batch_losses = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)

                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                # Shift for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()

                # Calculate loss manually
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='sum'
                )

                # Count tokens (excluding padding if mask exists)
                if attention_mask is not None:
                    # Shift mask to match shifted labels
                    shift_mask = attention_mask[..., 1:].contiguous()
                    num_tokens = shift_mask.sum().item()
                else:
                    num_tokens = shift_labels.numel()

                total_loss += loss.item()
                total_tokens += num_tokens
                batch_losses.append(loss.item() / num_tokens)

        # Method 1: Standard PPL = exp(total_loss / total_tokens)
        avg_loss = total_loss / total_tokens
        ppl_method1 = math.exp(avg_loss)

        # Method 2: Average of per-batch perplexities (can differ slightly)
        avg_batch_loss = sum(batch_losses) / len(batch_losses)
        ppl_method2 = math.exp(avg_batch_loss)

        # Method 3: Using different log bases for comparison
        ppl_log2 = 2 ** avg_loss  # Wrong if using natural log
        ppl_log10 = 10 ** avg_loss  # Wrong if using natural log

        return {
            'perplexity': ppl_method1,
            'perplexity_batch_avg': ppl_method2,
            'avg_loss': avg_loss,
            'total_tokens': total_tokens,
            'total_batches': len(batch_losses),
            'ppl_if_log2': ppl_log2,
            'ppl_if_log10': ppl_log10,
            'min_batch_loss': min(batch_losses),
            'max_batch_loss': max(batch_losses),
        }

    def sanity_check_random_model(self, vocab_size: int, seq_length: int = 128):
        """
        Test perplexity on random uniform predictions.
        Should give PPL ≈ vocab_size
        """
        logger.info("=== Random Model Sanity Check ===")

        # Create random logits (uniform distribution)
        batch_size = 4
        random_logits = torch.randn(batch_size, seq_length, vocab_size)
        random_targets = torch.randint(0, vocab_size, (batch_size, seq_length))

        # Calculate loss
        loss = F.cross_entropy(
            random_logits[:, :-1].reshape(-1, vocab_size),
            random_targets[:, 1:].reshape(-1),
            reduction='mean'
        )

        ppl = math.exp(loss.item())
        expected_ppl = vocab_size  # For uniform random predictions

        logger.info(f"Random model perplexity: {ppl:.2f}")
        logger.info(f"Expected (≈vocab_size): {expected_ppl}")
        logger.info(f"Ratio (should be ~1.0): {ppl / expected_ppl:.2f}")

        if abs(ppl / expected_ppl - 1.0) > 0.5:
            logger.warning("⚠️  Random model PPL is far from expected!")

        return ppl

    def check_loss_reduction(self, dataloader: DataLoader):
        """
        Verify that loss reduction is correct (mean vs sum).
        Common bug: using 'sum' when expecting 'mean' or vice versa.
        """
        logger.info("=== Loss Reduction Check ===")

        batch = next(iter(dataloader))
        input_ids = batch['input_ids'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss_mean = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean'
            )

            loss_sum = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'
            )

            num_tokens = shift_labels.numel()

            logger.info(f"Loss (mean): {loss_mean.item():.4f}")
            logger.info(f"Loss (sum): {loss_sum.item():.4f}")
            logger.info(f"Num tokens: {num_tokens}")
            logger.info(f"Ratio (sum/mean): {loss_sum.item() / loss_mean.item():.2f}")
            logger.info(f"Expected ratio: {num_tokens}")

            if abs(loss_sum.item() / loss_mean.item() - num_tokens) > 1:
                logger.warning("⚠️  Loss reduction might be incorrect!")


def check_perplexity_sanity(ppl: float, vocab_size: int) -> None:
    """Check if perplexity value is within reasonable bounds."""
    logger.info("\n=== Perplexity Sanity Checks ===")

    issues = []

    # Check 1: PPL should be >= 1.0
    if ppl < 1.0:
        issues.append(f"❌ PPL < 1.0 is impossible! Got {ppl:.4f}")

    # Check 2: PPL of ~1.0 suggests severe overfitting or bug
    if 1.0 <= ppl < 2.0:
        issues.append(f"⚠️  PPL = {ppl:.4f} is suspiciously low (near-perfect prediction)")
        issues.append("   → Check for data leakage (val data in training set)")
        issues.append("   → Check if model is just memorizing a tiny val set")
        issues.append("   → Verify loss calculation uses natural log (not log10/log2)")

    # Check 3: For small models, PPL should be reasonably high
    if ppl < 10 and vocab_size > 1000:
        issues.append(f"⚠️  PPL = {ppl:.4f} is unusually low for a {vocab_size:,} token vocab")
        issues.append("   → A 10M model on WikiText-2 typically gets PPL 80-200+")

    # Check 4: PPL shouldn't exceed vocab_size by much (random model baseline)
    if ppl > vocab_size * 2:
        issues.append(f"⚠️  PPL = {ppl:.2f} is very high (worse than random)")

    if issues:
        logger.warning("Issues detected:")
        for issue in issues:
            logger.warning(issue)
    else:
        logger.info("✓ Perplexity appears reasonable")

    # Reference values
    logger.info("\n=== Reference Perplexity Values ===")
    logger.info("WikiText-2 (word-level):")
    logger.info("  - LSTM baseline: ~100-120")
    logger.info("  - Transformer (large): ~20-40")
    logger.info("  - 10M model (expected): ~80-200+")
    logger.info("WikiText-2 (BPE tokens):")
    logger.info("  - Values typically higher than word-level")
    logger.info(f"  - Random model baseline: ~{vocab_size} (vocab size)")


def main():
    parser = argparse.ArgumentParser(description='Verify perplexity calculation')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to validation data')
    parser.add_argument('--vocab-size', type=int, default=16000,
                      help='Vocabulary size')
    parser.add_argument('--max-batches', type=int, default=None,
                      help='Limit number of batches to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use')

    args = parser.parse_args()

    # TODO: Load model and data - this will depend on your implementation
    # model = load_model(args.checkpoint)
    # dataloader = load_data(args.data)

    logger.info("=== Perplexity Verification Tool ===")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Vocab size: {args.vocab_size}")

    # Print instructions for now since actual implementation depends on user's code
    logger.info("\n" + "="*60)
    logger.info("TO USE THIS SCRIPT:")
    logger.info("="*60)
    logger.info("1. Load your trained model from the checkpoint")
    logger.info("2. Load your validation dataloader")
    logger.info("3. Instantiate: validator = PerplexityValidator(model)")
    logger.info("4. Run: results = validator.calculate_perplexity_manual(dataloader)")
    logger.info("5. Run: validator.sanity_check_random_model(vocab_size)")
    logger.info("6. Run: validator.check_loss_reduction(dataloader)")
    logger.info("7. Run: check_perplexity_sanity(results['perplexity'], vocab_size)")
    logger.info("="*60)

    logger.info("\nKEY CHECKS:")
    logger.info("✓ Perplexity formula: PPL = exp(cross_entropy_loss)")
    logger.info("✓ Use natural log (not log2 or log10)")
    logger.info("✓ Average loss over tokens (not batches)")
    logger.info("✓ Correctly handle padding tokens")
    logger.info("✓ Use 'mean' or 'sum' reduction consistently")


if __name__ == '__main__':
    main()
