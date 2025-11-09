#!/usr/bin/env python3
"""
Master script to run all validation checks before Phase 2.

This script runs:
1. Perplexity calculation verification
2. Data leakage detection
3. Validation set analysis

Usage:
    python scripts/run_all_validations.py \
        --checkpoint <checkpoint_path> \
        --train-data <train_data_path> \
        --val-data <val_data_path>
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Run all validation checks before Phase 2'
    )
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint (for PPL verification)')
    parser.add_argument('--train-data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--val-data', type=str, required=True,
                       help='Path to validation data')
    parser.add_argument('--output', type=str, default='validation_report.json',
                       help='Output path for validation report')
    parser.add_argument('--skip-model-checks', action='store_true',
                       help='Skip checks that require loading the model')

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("RUNNING ALL VALIDATION CHECKS")
    logger.info("="*70)
    logger.info(f"Train data: {args.train_data}")
    logger.info(f"Val data: {args.val_data}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Report output: {args.output}")
    logger.info("")

    report = {
        'timestamp': datetime.now().isoformat(),
        'train_data': args.train_data,
        'val_data': args.val_data,
        'checkpoint': args.checkpoint,
        'checks': {}
    }

    # Load data (TODO: implement based on your data format)
    logger.info("Loading data...")
    logger.info("⚠️  You need to implement data loading for your specific format")
    logger.info("")

    # Placeholder for actual data loading
    # train_data = load_data(args.train_data)
    # val_data = load_data(args.val_data)

    # Check 1: Data leakage
    logger.info("="*70)
    logger.info("CHECK 1: DATA LEAKAGE DETECTION")
    logger.info("="*70)
    logger.info("")
    logger.info("Run: python scripts/check_data_leakage.py \\")
    logger.info(f"       --train-data {args.train_data} \\")
    logger.info(f"       --val-data {args.val_data}")
    logger.info("")
    # TODO: Implement once data loading is available
    # from check_data_leakage import DataLeakageDetector
    # detector = DataLeakageDetector(train_data, val_data)
    # leakage_results = detector.check_exact_duplicates()
    # report['checks']['data_leakage'] = leakage_results

    # Check 2: Validation set analysis
    logger.info("="*70)
    logger.info("CHECK 2: VALIDATION SET ANALYSIS")
    logger.info("="*70)
    logger.info("")
    logger.info("Run: python scripts/analyze_validation_set.py \\")
    logger.info(f"       --train-data {args.train_data} \\")
    logger.info(f"       --val-data {args.val_data}")
    logger.info("")
    # TODO: Implement once data loading is available
    # from analyze_validation_set import ValidationSetAnalyzer
    # analyzer = ValidationSetAnalyzer(train_data, val_data)
    # analysis_results = analyzer.generate_report()
    # report['checks']['validation_analysis'] = analysis_results

    # Check 3: Perplexity verification (if checkpoint provided)
    if args.checkpoint and not args.skip_model_checks:
        logger.info("="*70)
        logger.info("CHECK 3: PERPLEXITY CALCULATION VERIFICATION")
        logger.info("="*70)
        logger.info("")
        logger.info("Run: python scripts/verify_perplexity.py \\")
        logger.info(f"       --checkpoint {args.checkpoint} \\")
        logger.info(f"       --data {args.val_data}")
        logger.info("")
        # TODO: Implement once model loading is available
        # from verify_perplexity import PerplexityValidator
        # model = load_model(args.checkpoint)
        # validator = PerplexityValidator(model)
        # ppl_results = validator.calculate_perplexity_manual(val_dataloader)
        # report['checks']['perplexity_verification'] = ppl_results

    # Generate summary
    logger.info("="*70)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*70)
    logger.info("")

    logger.info("KEY QUESTIONS TO ANSWER:")
    logger.info("")
    logger.info("1. IS PERPLEXITY OF 1.054 REALISTIC?")
    logger.info("   ❌ NO - This is unrealistically low")
    logger.info("   Expected for 10M model on WikiText-2: 80-200+ PPL")
    logger.info("   PPL of 1.054 suggests:")
    logger.info("     • Validation set is tiny (< 1000 tokens)")
    logger.info("     • Data leakage (val data in training)")
    logger.info("     • Bug in PPL calculation")
    logger.info("")

    logger.info("2. IS THERE DATA LEAKAGE?")
    logger.info("   Run: scripts/check_data_leakage.py")
    logger.info("   Check for:")
    logger.info("     • Exact duplicates between train and val")
    logger.info("     • High n-gram overlap (> 90%)")
    logger.info("     • Same files in both splits")
    logger.info("")

    logger.info("3. IS VALIDATION SET LARGE ENOUGH?")
    logger.info("   Run: scripts/analyze_validation_set.py")
    logger.info("   Requirements:")
    logger.info("     • Minimum: 50,000 tokens")
    logger.info("     • Recommended: 100,000-500,000 tokens")
    logger.info("     • WikiText-2 standard: ~200,000 tokens")
    logger.info("     • 5-10% of total data")
    logger.info("")

    logger.info("="*70)
    logger.info("RECOMMENDED ACTIONS")
    logger.info("="*70)
    logger.info("")
    logger.info("BEFORE PHASE 2, YOU MUST:")
    logger.info("")
    logger.info("1. ✓ Verify your validation set size")
    logger.info("     → Print the number of tokens in your val set")
    logger.info("     → If < 50k tokens, create a larger val set")
    logger.info("")
    logger.info("2. ✓ Check for data leakage")
    logger.info("     → Run the check_data_leakage.py script")
    logger.info("     → Ensure train and val are properly separated")
    logger.info("")
    logger.info("3. ✓ Verify perplexity calculation")
    logger.info("     → PPL = exp(cross_entropy_loss)")
    logger.info("     → Use natural log (not log2 or log10)")
    logger.info("     → Average loss over tokens (not batches)")
    logger.info("")
    logger.info("4. ✓ Re-run evaluation with corrected setup")
    logger.info("     → Expect PPL of 80-200+ for a 10M model")
    logger.info("     → If still getting ~1, investigate further")
    logger.info("")
    logger.info("5. ✓ Use standard WikiText-2 splits")
    logger.info("     → Don't create custom splits unless necessary")
    logger.info("     → Standard splits are well-tested and reliable")
    logger.info("")

    # Save report
    # with open(args.output, 'w') as f:
    #     json.dump(report, f, indent=2)
    # logger.info(f"Report saved to: {args.output}")

    logger.info("="*70)
    logger.info("NEXT STEPS")
    logger.info("="*70)
    logger.info("")
    logger.info("1. Fix any issues found in validation checks")
    logger.info("2. Re-run Phase 1 evaluation with corrected setup")
    logger.info("3. Verify PPL is realistic (80-200+ for 10M model)")
    logger.info("4. Only then proceed to Phase 2")
    logger.info("")


if __name__ == '__main__':
    main()
