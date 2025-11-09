#!/usr/bin/env python3
"""
Quick validation checks without loading models.

This is a lightweight script that can quickly check:
1. Validation set size (file size, line count, token estimate)
2. Basic data leakage (exact line duplicates)
3. File-level issues

Usage:
    python scripts/quick_check.py \
        --train path/to/train.txt \
        --val path/to/val.txt
"""

import argparse
import logging
import hashlib
from pathlib import Path
from typing import Dict, Set

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation)."""
    # Simple whitespace split
    return len(text.split())


def check_file_size(file_path: Path) -> Dict:
    """Check file size and basic stats."""
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return {}

    file_size = file_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        text = ''.join(lines)

    num_lines = len(lines)
    num_chars = len(text)
    est_tokens = estimate_tokens(text)

    return {
        'file_size_bytes': file_size,
        'file_size_mb': file_size_mb,
        'num_lines': num_lines,
        'num_chars': num_chars,
        'est_tokens': est_tokens,
    }


def check_duplicates(train_path: Path, val_path: Path) -> Dict:
    """Check for exact line duplicates between train and val."""
    logger.info("Checking for exact duplicates...")

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_lines = set(line.strip() for line in f if line.strip())

    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_lines = set(line.strip() for line in f if line.strip())

    duplicates = train_lines & val_lines
    duplicate_ratio = len(duplicates) / len(val_lines) if val_lines else 0

    return {
        'num_duplicates': len(duplicates),
        'duplicate_ratio': duplicate_ratio,
        'train_unique_lines': len(train_lines),
        'val_unique_lines': len(val_lines),
    }


def main():
    parser = argparse.ArgumentParser(description='Quick validation checks')
    parser.add_argument('--train', type=str, required=True,
                       help='Path to training data file')
    parser.add_argument('--val', type=str, required=True,
                       help='Path to validation data file')

    args = parser.parse_args()

    train_path = Path(args.train)
    val_path = Path(args.val)

    logger.info("="*70)
    logger.info("QUICK VALIDATION CHECK")
    logger.info("="*70)
    logger.info(f"Train: {train_path}")
    logger.info(f"Val:   {val_path}")
    logger.info("")

    # Check train file
    logger.info("=== Training Data ===")
    train_stats = check_file_size(train_path)
    if train_stats:
        logger.info(f"File size:     {train_stats['file_size_mb']:.2f} MB")
        logger.info(f"Lines:         {train_stats['num_lines']:,}")
        logger.info(f"Characters:    {train_stats['num_chars']:,}")
        logger.info(f"Est. tokens:   {train_stats['est_tokens']:,}")
    logger.info("")

    # Check val file
    logger.info("=== Validation Data ===")
    val_stats = check_file_size(val_path)
    if val_stats:
        logger.info(f"File size:     {val_stats['file_size_mb']:.2f} MB")
        logger.info(f"Lines:         {val_stats['num_lines']:,}")
        logger.info(f"Characters:    {val_stats['num_chars']:,}")
        logger.info(f"Est. tokens:   {val_stats['est_tokens']:,}")
    logger.info("")

    # Check validation set size
    if val_stats:
        est_tokens = val_stats['est_tokens']
        logger.info("=== Validation Set Size Assessment ===")

        if est_tokens < 1000:
            logger.error("❌ CRITICAL: Val set < 1,000 tokens")
            logger.error("   This is EXTREMELY small!")
            logger.error("   → Model likely memorized it")
            logger.error("   → Explains PPL of 1.054")
            logger.error("   → MUST use larger val set (50k+ tokens)")
        elif est_tokens < 10000:
            logger.warning("⚠️  WARNING: Val set < 10,000 tokens")
            logger.warning("   This is very small")
            logger.warning("   → Unreliable for evaluation")
            logger.warning("   → Increase to 50k+ tokens")
        elif est_tokens < 50000:
            logger.warning("⚠️  Val set < 50,000 tokens")
            logger.warning("   This is small but usable")
            logger.warning("   → Recommended: 100k-500k tokens")
        else:
            logger.info(f"✓ Val set size is reasonable ({est_tokens:,} tokens)")

        # Calculate ratio
        if train_stats:
            total_tokens = train_stats['est_tokens'] + val_stats['est_tokens']
            val_ratio = val_stats['est_tokens'] / total_tokens
            logger.info(f"Val ratio: {val_ratio*100:.2f}%")

            if val_ratio < 0.01:
                logger.warning("⚠️  Val set < 1% of data (very small)")
            elif val_ratio > 0.3:
                logger.warning("⚠️  Val set > 30% of data (could use more for training)")
            else:
                logger.info(f"✓ Val ratio is reasonable")

        logger.info("")

    # Check for duplicates
    if train_path.exists() and val_path.exists():
        logger.info("=== Data Leakage Check ===")
        dup_stats = check_duplicates(train_path, val_path)

        logger.info(f"Train unique lines: {dup_stats['train_unique_lines']:,}")
        logger.info(f"Val unique lines:   {dup_stats['val_unique_lines']:,}")
        logger.info(f"Duplicates:         {dup_stats['num_duplicates']:,}")

        if dup_stats['num_duplicates'] > 0:
            logger.error(f"❌ Found {dup_stats['num_duplicates']:,} duplicate lines!")
            logger.error(f"   {dup_stats['duplicate_ratio']*100:.2f}% of val data is in training")
            logger.error("   → This is DATA LEAKAGE")
            logger.error("   → Could explain low PPL")
        else:
            logger.info("✓ No exact line duplicates found")

        logger.info("")

    # Summary
    logger.info("="*70)
    logger.info("SUMMARY")
    logger.info("="*70)

    issues = []

    if val_stats and val_stats['est_tokens'] < 50000:
        issues.append("Val set is too small (< 50k tokens)")

    if 'num_duplicates' in dup_stats and dup_stats['num_duplicates'] > 0:
        issues.append("Data leakage detected (duplicates found)")

    if issues:
        logger.error("\n❌ ISSUES FOUND:")
        for issue in issues:
            logger.error(f"   • {issue}")
        logger.error("\nThese issues explain the unrealistic PPL of 1.054!")
        logger.error("\nRECOMMENDED ACTIONS:")
        logger.error("1. Create a larger validation set (50k+ tokens)")
        logger.error("2. Ensure train and val are properly separated")
        logger.error("3. Use standard WikiText-2 validation split (~200k tokens)")
        logger.error("4. Re-run evaluation with corrected data")
    else:
        logger.info("\n✓ No obvious issues detected")
        logger.info("\nNext steps:")
        logger.info("• Run full validation: python scripts/run_all_validations.py")
        logger.info("• Verify perplexity calculation")
        logger.info("• Check for n-gram contamination")

    logger.info("")


if __name__ == '__main__':
    main()
