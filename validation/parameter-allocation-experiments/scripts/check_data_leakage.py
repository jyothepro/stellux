#!/usr/bin/env python3
"""
Check for data leakage between train and validation sets.

This script checks:
1. No duplicate sequences between train and val
2. No substring overlaps (n-gram contamination)
3. Proper splitting (no accidental shuffling)
4. Token distribution differences
5. File-level contamination

Usage:
    python scripts/check_data_leakage.py --train-data <path> --val-data <path>
"""

import argparse
import hashlib
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLeakageDetector:
    """Detects various forms of data leakage between train and validation sets."""

    def __init__(self, train_data: List[str], val_data: List[str]):
        """
        Args:
            train_data: List of training sequences (strings or token IDs)
            val_data: List of validation sequences
        """
        self.train_data = train_data
        self.val_data = val_data

    def check_exact_duplicates(self) -> Dict[str, any]:
        """Check for exact duplicate sequences."""
        logger.info("Checking for exact duplicates...")

        train_hashes = {hashlib.md5(str(seq).encode()).hexdigest() for seq in self.train_data}
        val_hashes = {hashlib.md5(str(seq).encode()).hexdigest() for seq in self.val_data}

        duplicates = train_hashes & val_hashes
        duplicate_ratio = len(duplicates) / len(val_hashes) if val_hashes else 0

        results = {
            'num_duplicates': len(duplicates),
            'duplicate_ratio': duplicate_ratio,
            'train_unique': len(train_hashes),
            'val_unique': len(val_hashes),
        }

        if duplicates:
            logger.warning(f"❌ Found {len(duplicates)} exact duplicates!")
            logger.warning(f"   {duplicate_ratio*100:.2f}% of validation data is in training set")
        else:
            logger.info("✓ No exact duplicates found")

        return results

    def check_ngram_overlap(self, n: int = 8) -> Dict[str, float]:
        """
        Check for n-gram contamination between train and val.

        Args:
            n: N-gram size (default 8, which is a good indicator of contamination)
        """
        logger.info(f"Checking for {n}-gram overlap...")

        def get_ngrams(sequences: List, n: int) -> Set:
            """Extract all n-grams from sequences."""
            ngrams = set()
            for seq in sequences:
                # Handle both string and list inputs
                tokens = list(seq) if isinstance(seq, (list, tuple)) else list(str(seq))
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i+n])
                    ngrams.add(ngram)
            return ngrams

        train_ngrams = get_ngrams(self.train_data, n)
        val_ngrams = get_ngrams(self.val_data, n)

        overlap = train_ngrams & val_ngrams
        overlap_ratio = len(overlap) / len(val_ngrams) if val_ngrams else 0

        results = {
            f'{n}gram_overlap_count': len(overlap),
            f'{n}gram_overlap_ratio': overlap_ratio,
            f'train_{n}grams': len(train_ngrams),
            f'val_{n}grams': len(val_ngrams),
        }

        # High overlap is expected for language data, but 100% is suspicious
        if overlap_ratio > 0.9:
            logger.warning(f"⚠️  Very high {n}-gram overlap: {overlap_ratio*100:.2f}%")
            logger.warning("   This could indicate data leakage or very small val set")
        elif overlap_ratio > 0.5:
            logger.info(f"⚠️  Moderate {n}-gram overlap: {overlap_ratio*100:.2f}%")
            logger.info("   (Some overlap is normal for language data)")
        else:
            logger.info(f"✓ {n}-gram overlap: {overlap_ratio*100:.2f}% (reasonable)")

        return results

    def check_token_statistics(self) -> Dict[str, any]:
        """
        Compare token distributions between train and val.
        Large differences might indicate different data sources.
        """
        logger.info("Checking token statistics...")

        def get_token_stats(sequences: List) -> Dict:
            """Calculate token statistics."""
            all_tokens = []
            for seq in sequences:
                if isinstance(seq, str):
                    all_tokens.extend(seq.split())
                elif isinstance(seq, (list, tuple)):
                    all_tokens.extend(seq)

            counter = Counter(all_tokens)
            return {
                'total_tokens': len(all_tokens),
                'unique_tokens': len(counter),
                'vocab_size': len(counter),
                'most_common': counter.most_common(10),
            }

        train_stats = get_token_stats(self.train_data)
        val_stats = get_token_stats(self.val_data)

        logger.info(f"Train: {train_stats['total_tokens']:,} tokens, "
                   f"{train_stats['unique_tokens']:,} unique")
        logger.info(f"Val:   {val_stats['total_tokens']:,} tokens, "
                   f"{val_stats['unique_tokens']:,} unique")

        return {
            'train_stats': train_stats,
            'val_stats': val_stats,
        }

    def check_sequence_lengths(self) -> Dict[str, any]:
        """Check if sequence length distributions are similar."""
        logger.info("Checking sequence length distributions...")

        train_lengths = [len(seq) for seq in self.train_data]
        val_lengths = [len(seq) for seq in self.val_data]

        results = {
            'train_mean_length': np.mean(train_lengths),
            'train_std_length': np.std(train_lengths),
            'val_mean_length': np.mean(val_lengths),
            'val_std_length': np.std(val_lengths),
        }

        logger.info(f"Train: {results['train_mean_length']:.1f} ± "
                   f"{results['train_std_length']:.1f} tokens")
        logger.info(f"Val:   {results['val_mean_length']:.1f} ± "
                   f"{results['val_std_length']:.1f} tokens")

        # Check if distributions are suspiciously different
        mean_diff_pct = abs(results['train_mean_length'] - results['val_mean_length']) / results['train_mean_length']
        if mean_diff_pct > 0.5:
            logger.warning(f"⚠️  Large difference in mean lengths: {mean_diff_pct*100:.1f}%")
            logger.warning("   Train and val might come from different sources")

        return results


def check_file_level_leakage(train_files: List[Path], val_files: List[Path]) -> bool:
    """
    Check if train and val files are properly separated.

    Common issues:
    - Same file used for both train and val
    - Files from same source documents mixed improperly
    """
    logger.info("Checking file-level leakage...")

    train_names = {f.name for f in train_files}
    val_names = {f.name for f in val_files}

    overlap = train_names & val_names

    if overlap:
        logger.error(f"❌ Found {len(overlap)} files in both train and val!")
        logger.error(f"   Overlapping files: {list(overlap)[:5]}")
        return True
    else:
        logger.info("✓ No file overlap detected")
        return False


def check_split_randomness(indices: List[int], total_size: int) -> bool:
    """
    Check if data split appears random or sequential.

    Sequential splitting (e.g., first 80% train, last 20% val) is usually
    better than random for language models to avoid contamination.
    """
    logger.info("Checking split method...")

    if not indices:
        logger.warning("⚠️  No indices provided")
        return False

    # Check if indices are sequential
    is_sequential = all(indices[i] < indices[i+1] for i in range(len(indices)-1))

    # Check for large gaps (might indicate improper splitting)
    if len(indices) > 1:
        gaps = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
        max_gap = max(gaps)
        avg_gap = np.mean(gaps)

        logger.info(f"Sequential: {is_sequential}")
        logger.info(f"Max gap: {max_gap}, Avg gap: {avg_gap:.1f}")

        if max_gap > avg_gap * 10:
            logger.warning("⚠️  Large gaps in indices - check split implementation")
            return False

    if is_sequential:
        logger.info("✓ Data appears to be split sequentially (good for LM)")
    else:
        logger.warning("⚠️  Data appears randomly shuffled")
        logger.warning("   For language models, sequential splits are usually better")

    return is_sequential


def analyze_validation_set_size(
    num_train: int,
    num_val: int,
    val_tokens: int
) -> Dict[str, any]:
    """
    Analyze if validation set is appropriately sized.

    Args:
        num_train: Number of training examples
        num_val: Number of validation examples
        val_tokens: Total tokens in validation set
    """
    logger.info("\n=== Validation Set Size Analysis ===")

    val_ratio = num_val / (num_train + num_val)

    logger.info(f"Train examples: {num_train:,}")
    logger.info(f"Val examples:   {num_val:,}")
    logger.info(f"Val ratio:      {val_ratio*100:.2f}%")
    logger.info(f"Val tokens:     {val_tokens:,}")

    issues = []

    # Check 1: Validation set too small (< 1000 tokens)
    if val_tokens < 1000:
        issues.append("❌ Validation set is VERY small (< 1000 tokens)")
        issues.append("   → Perplexity on such a small set is unreliable")
        issues.append("   → Model could easily memorize this")

    # Check 2: Validation set small (< 10k tokens)
    elif val_tokens < 10000:
        issues.append("⚠️  Validation set is small (< 10k tokens)")
        issues.append("   → Consider using at least 50k-100k tokens for reliable metrics")

    # Check 3: Validation ratio too small
    if val_ratio < 0.01:
        issues.append("⚠️  Validation set is < 1% of total data")

    # Check 4: Validation ratio too large
    if val_ratio > 0.3:
        issues.append("⚠️  Validation set is > 30% of total data")
        issues.append("   → Consider using more data for training")

    if issues:
        logger.warning("Issues detected:")
        for issue in issues:
            logger.warning(f"  {issue}")
    else:
        logger.info("✓ Validation set size appears reasonable")

    # Recommendations
    logger.info("\n=== Recommendations ===")
    logger.info("For reliable validation:")
    logger.info("  • Val set should have 50k-100k+ tokens")
    logger.info("  • Typically 5-10% of total data")
    logger.info("  • For 50M total tokens: ~2.5M-5M val tokens")
    logger.info("  • Current val tokens: {val_tokens:,}")

    recommended_val_tokens = max(50000, int(num_train * 0.05))
    logger.info(f"  • Recommended: {recommended_val_tokens:,}+ tokens")

    return {
        'val_ratio': val_ratio,
        'val_tokens': val_tokens,
        'issues': issues,
        'recommended_val_tokens': recommended_val_tokens,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Check for data leakage between train and validation sets'
    )
    parser.add_argument('--train-data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--val-data', type=str, required=True,
                       help='Path to validation data')
    parser.add_argument('--format', type=str, default='text',
                       choices=['text', 'jsonl', 'tokens'],
                       help='Data format')
    parser.add_argument('--ngram-size', type=int, default=8,
                       help='N-gram size for overlap check')

    args = parser.parse_args()

    logger.info("=== Data Leakage Detection Tool ===")
    logger.info(f"Train data: {args.train_data}")
    logger.info(f"Val data: {args.val_data}")

    # TODO: Load data based on format - depends on user's implementation
    # For now, print instructions
    logger.info("\n" + "="*60)
    logger.info("TO USE THIS SCRIPT:")
    logger.info("="*60)
    logger.info("1. Load your train and validation data")
    logger.info("2. detector = DataLeakageDetector(train_data, val_data)")
    logger.info("3. Run all checks:")
    logger.info("   - detector.check_exact_duplicates()")
    logger.info("   - detector.check_ngram_overlap(n=8)")
    logger.info("   - detector.check_token_statistics()")
    logger.info("   - detector.check_sequence_lengths()")
    logger.info("4. analyze_validation_set_size(num_train, num_val, val_tokens)")
    logger.info("="*60)

    logger.info("\nKEY CHECKS:")
    logger.info("✓ No exact duplicates between train and val")
    logger.info("✓ N-gram overlap is reasonable (< 90%)")
    logger.info("✓ Val set has enough tokens (50k-100k+)")
    logger.info("✓ Sequential splitting (not random shuffling)")
    logger.info("✓ No file-level contamination")


if __name__ == '__main__':
    main()
