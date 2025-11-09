#!/usr/bin/env python3
"""
Analyze validation set size and quality.

This script analyzes:
1. Validation set size (tokens, sequences)
2. Diversity metrics (unique n-grams, vocabulary coverage)
3. Difficulty distribution (to ensure it's not too easy)
4. Comparison with training set characteristics

Usage:
    python scripts/analyze_validation_set.py --train-data <path> --val-data <path>
"""

import argparse
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationSetAnalyzer:
    """Comprehensive validation set analysis."""

    def __init__(self, train_data: List, val_data: List):
        """
        Args:
            train_data: Training sequences
            val_data: Validation sequences
        """
        self.train_data = train_data
        self.val_data = val_data

    def analyze_size(self) -> Dict[str, any]:
        """Analyze validation set size."""
        logger.info("=== Validation Set Size Analysis ===")

        # Count sequences
        num_train_seqs = len(self.train_data)
        num_val_seqs = len(self.val_data)

        # Count tokens
        def count_tokens(data):
            total = 0
            for seq in data:
                if isinstance(seq, str):
                    total += len(seq.split())
                elif isinstance(seq, (list, tuple)):
                    total += len(seq)
                else:
                    total += 1
            return total

        train_tokens = count_tokens(self.train_data)
        val_tokens = count_tokens(self.val_data)

        total_seqs = num_train_seqs + num_val_seqs
        total_tokens = train_tokens + val_tokens

        val_seq_ratio = num_val_seqs / total_seqs if total_seqs > 0 else 0
        val_token_ratio = val_tokens / total_tokens if total_tokens > 0 else 0

        results = {
            'train_sequences': num_train_seqs,
            'val_sequences': num_val_seqs,
            'train_tokens': train_tokens,
            'val_tokens': val_tokens,
            'val_seq_ratio': val_seq_ratio,
            'val_token_ratio': val_token_ratio,
        }

        logger.info(f"Training:   {num_train_seqs:,} sequences, {train_tokens:,} tokens")
        logger.info(f"Validation: {num_val_seqs:,} sequences, {val_tokens:,} tokens")
        logger.info(f"Val ratio:  {val_seq_ratio*100:.2f}% sequences, "
                   f"{val_token_ratio*100:.2f}% tokens")

        # Size assessment
        self._assess_size(val_tokens, val_token_ratio)

        return results

    def _assess_size(self, val_tokens: int, val_ratio: float):
        """Assess if validation set size is appropriate."""
        issues = []

        if val_tokens < 1000:
            issues.append("❌ CRITICAL: Val set < 1,000 tokens")
            issues.append("   → PPL on this is meaningless, model likely memorized it")
            issues.append("   → Explains PPL of 1.054!")
        elif val_tokens < 10000:
            issues.append("⚠️  WARNING: Val set < 10,000 tokens")
            issues.append("   → Too small for reliable evaluation")
            issues.append("   → Easy to overfit")
        elif val_tokens < 50000:
            issues.append("⚠️  Val set < 50,000 tokens (small but usable)")

        if val_ratio < 0.01:
            issues.append("⚠️  Val set < 1% of data (consider increasing)")
        elif val_ratio > 0.3:
            issues.append("⚠️  Val set > 30% of data (consider using more for training)")

        if issues:
            logger.warning("\nISSUES DETECTED:")
            for issue in issues:
                logger.warning(f"  {issue}")
        else:
            logger.info("\n✓ Validation set size is reasonable")

        logger.info("\nRECOMMENDATIONS:")
        logger.info("  • Minimum: 50,000 tokens")
        logger.info("  • Good: 100,000-500,000 tokens")
        logger.info("  • Ideal for WikiText-2: use the standard val split (~200k tokens)")
        logger.info("  • Ratio: 5-10% of total data")

    def analyze_diversity(self) -> Dict[str, any]:
        """
        Analyze diversity of validation set.
        Low diversity = easy to memorize = unreliable PPL.
        """
        logger.info("\n=== Validation Set Diversity Analysis ===")

        def get_diversity_metrics(data, name="Dataset"):
            """Calculate diversity metrics for a dataset."""
            # Collect all tokens
            all_tokens = []
            for seq in data:
                if isinstance(seq, str):
                    all_tokens.extend(seq.split())
                elif isinstance(seq, (list, tuple)):
                    all_tokens.extend(seq)

            if not all_tokens:
                return {}

            # Token-level metrics
            token_counter = Counter(all_tokens)
            vocab_size = len(token_counter)
            total_tokens = len(all_tokens)
            unique_ratio = vocab_size / total_tokens if total_tokens > 0 else 0

            # Calculate entropy (higher = more diverse)
            probs = np.array([count / total_tokens for count in token_counter.values()])
            entropy = -np.sum(probs * np.log2(probs))

            # Top-k concentration (what % of tokens are the top 10 most common?)
            top_10_count = sum(count for _, count in token_counter.most_common(10))
            top_10_ratio = top_10_count / total_tokens if total_tokens > 0 else 0

            metrics = {
                'total_tokens': total_tokens,
                'vocab_size': vocab_size,
                'unique_ratio': unique_ratio,
                'entropy': entropy,
                'top_10_concentration': top_10_ratio,
            }

            logger.info(f"\n{name}:")
            logger.info(f"  Total tokens: {total_tokens:,}")
            logger.info(f"  Vocabulary: {vocab_size:,}")
            logger.info(f"  Unique ratio: {unique_ratio:.4f}")
            logger.info(f"  Entropy: {entropy:.2f} bits")
            logger.info(f"  Top-10 concentration: {top_10_ratio*100:.2f}%")

            return metrics

        train_metrics = get_diversity_metrics(self.train_data, "Training Set")
        val_metrics = get_diversity_metrics(self.val_data, "Validation Set")

        # Compare diversity
        if val_metrics and train_metrics:
            self._compare_diversity(train_metrics, val_metrics)

        return {
            'train_diversity': train_metrics,
            'val_diversity': val_metrics,
        }

    def _compare_diversity(self, train_metrics: Dict, val_metrics: Dict):
        """Compare diversity between train and val."""
        logger.info("\n=== Diversity Comparison ===")

        # Compare entropy
        entropy_ratio = val_metrics['entropy'] / train_metrics['entropy']
        logger.info(f"Entropy ratio (val/train): {entropy_ratio:.3f}")

        if entropy_ratio < 0.7:
            logger.warning("⚠️  Val set is much less diverse than train")
            logger.warning("   → Might be easier to memorize")
        elif entropy_ratio > 1.3:
            logger.warning("⚠️  Val set is much more diverse than train")
            logger.warning("   → Might contain out-of-distribution examples")
        else:
            logger.info("✓ Val and train have similar diversity")

        # Compare vocabulary coverage
        val_vocab_coverage = val_metrics['vocab_size'] / train_metrics['vocab_size']
        logger.info(f"Val vocab coverage: {val_vocab_coverage*100:.2f}% of train vocab")

        if val_vocab_coverage < 0.3:
            logger.warning("⚠️  Val set covers < 30% of train vocabulary")
            logger.warning("   → Val set might be too small or too narrow")

    def check_for_repetition(self) -> Dict[str, any]:
        """
        Check if validation set has excessive repetition.
        High repetition = easy to memorize = explains low PPL.
        """
        logger.info("\n=== Repetition Analysis ===")

        def get_repetition_metrics(data, name="Dataset"):
            """Calculate repetition metrics."""
            # Get all sequences as strings for comparison
            seq_strs = [str(seq) for seq in data]
            seq_counter = Counter(seq_strs)

            total_seqs = len(seq_strs)
            unique_seqs = len(seq_counter)
            repetition_ratio = 1 - (unique_seqs / total_seqs) if total_seqs > 0 else 0

            # Find most repeated sequences
            most_repeated = seq_counter.most_common(5)

            metrics = {
                'total_sequences': total_seqs,
                'unique_sequences': unique_seqs,
                'repetition_ratio': repetition_ratio,
                'most_repeated': most_repeated,
            }

            logger.info(f"\n{name}:")
            logger.info(f"  Total sequences: {total_seqs:,}")
            logger.info(f"  Unique sequences: {unique_seqs:,}")
            logger.info(f"  Repetition: {repetition_ratio*100:.2f}%")

            if repetition_ratio > 0.1:
                logger.warning(f"⚠️  High repetition in {name}")
                logger.info("  Most repeated sequences:")
                for seq, count in most_repeated[:3]:
                    preview = str(seq)[:50] + "..." if len(str(seq)) > 50 else str(seq)
                    logger.info(f"    {count}x: {preview}")

            return metrics

        train_rep = get_repetition_metrics(self.train_data, "Training Set")
        val_rep = get_repetition_metrics(self.val_data, "Validation Set")

        # High repetition in val is very bad
        if val_rep['repetition_ratio'] > 0.3:
            logger.error("❌ Val set has >30% repeated sequences!")
            logger.error("   → This could explain the low PPL of 1.054")
            logger.error("   → Model has memorized these repeated sequences")

        return {
            'train_repetition': train_rep,
            'val_repetition': val_rep,
        }

    def estimate_difficulty(self) -> Dict[str, any]:
        """
        Estimate dataset difficulty using simple heuristics.

        Easy datasets (low difficulty) lead to unrealistically low PPL.
        """
        logger.info("\n=== Dataset Difficulty Estimation ===")

        def get_difficulty_metrics(data, name="Dataset"):
            """Estimate difficulty metrics."""
            lengths = []
            for seq in data:
                if isinstance(seq, str):
                    lengths.append(len(seq.split()))
                elif isinstance(seq, (list, tuple)):
                    lengths.append(len(seq))

            metrics = {
                'mean_length': np.mean(lengths) if lengths else 0,
                'std_length': np.std(lengths) if lengths else 0,
                'min_length': min(lengths) if lengths else 0,
                'max_length': max(lengths) if lengths else 0,
            }

            logger.info(f"\n{name}:")
            logger.info(f"  Sequence length: {metrics['mean_length']:.1f} ± "
                       f"{metrics['std_length']:.1f} tokens")
            logger.info(f"  Range: {metrics['min_length']} - {metrics['max_length']} tokens")

            # Very short sequences are easier to memorize
            if metrics['mean_length'] < 10:
                logger.warning("⚠️  Very short sequences (< 10 tokens)")
                logger.warning("   → Easier to memorize")

            return metrics

        train_diff = get_difficulty_metrics(self.train_data, "Training Set")
        val_diff = get_difficulty_metrics(self.val_data, "Validation Set")

        return {
            'train_difficulty': train_diff,
            'val_difficulty': val_diff,
        }

    def generate_report(self) -> str:
        """Generate a comprehensive validation set quality report."""
        logger.info("\n" + "="*70)
        logger.info("COMPREHENSIVE VALIDATION SET ANALYSIS REPORT")
        logger.info("="*70)

        size_results = self.analyze_size()
        diversity_results = self.analyze_diversity()
        repetition_results = self.check_for_repetition()
        difficulty_results = self.estimate_difficulty()

        # Generate summary
        logger.info("\n" + "="*70)
        logger.info("SUMMARY & RECOMMENDATIONS")
        logger.info("="*70)

        issues = []
        if size_results['val_tokens'] < 50000:
            issues.append("CRITICAL: Validation set too small")
        if repetition_results['val_repetition']['repetition_ratio'] > 0.3:
            issues.append("CRITICAL: High repetition in validation set")
        if diversity_results.get('val_diversity', {}).get('entropy', 0) < 5:
            issues.append("WARNING: Low diversity in validation set")

        if issues:
            logger.error("\nCRITICAL ISSUES FOUND:")
            for issue in issues:
                logger.error(f"  ❌ {issue}")
            logger.error("\nThese issues explain the unrealistic PPL of 1.054!")
            logger.error("RECOMMENDED ACTIONS:")
            logger.error("  1. Use a larger validation set (100k+ tokens)")
            logger.error("  2. Ensure proper train/val split (sequential, no overlap)")
            logger.error("  3. Use standard WikiText-2 validation split")
            logger.error("  4. Re-run evaluation with corrected data")
        else:
            logger.info("\n✓ Validation set appears properly configured")

        return {
            'size': size_results,
            'diversity': diversity_results,
            'repetition': repetition_results,
            'difficulty': difficulty_results,
            'issues': issues,
        }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze validation set size and quality'
    )
    parser.add_argument('--train-data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--val-data', type=str, required=True,
                       help='Path to validation data')
    parser.add_argument('--format', type=str, default='text',
                       choices=['text', 'jsonl', 'tokens'],
                       help='Data format')

    args = parser.parse_args()

    logger.info("=== Validation Set Analysis Tool ===")
    logger.info(f"Train data: {args.train_data}")
    logger.info(f"Val data: {args.val_data}")

    # TODO: Load data based on format - depends on user's implementation
    logger.info("\n" + "="*60)
    logger.info("TO USE THIS SCRIPT:")
    logger.info("="*60)
    logger.info("1. Load your train and validation data into lists")
    logger.info("2. analyzer = ValidationSetAnalyzer(train_data, val_data)")
    logger.info("3. report = analyzer.generate_report()")
    logger.info("="*60)

    logger.info("\nKEY METRICS TO CHECK:")
    logger.info("✓ Val set size: 50k-500k tokens (not 1k!)")
    logger.info("✓ Val ratio: 5-10% of total data")
    logger.info("✓ Low repetition: < 10% repeated sequences")
    logger.info("✓ Good diversity: entropy similar to train set")
    logger.info("✓ Proper difficulty: not too easy to memorize")


if __name__ == '__main__':
    main()
