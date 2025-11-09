#!/usr/bin/env python3
"""
Quick script to check the actual size of your validation data.
This will help diagnose why PPL is unrealistically low (1.054).
"""

import sys
from pathlib import Path

try:
    from datasets import load_dataset
    import datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️  datasets library not available")

def check_wikitext2():
    """Check WikiText-2 dataset size."""
    print("="*70)
    print("CHECKING WIKITEXT-2 DATASET")
    print("="*70)

    if not DATASETS_AVAILABLE:
        print("❌ datasets library not installed")
        print("   Run: pip install datasets")
        return

    try:
        print("\nLoading WikiText-2...")
        wikitext = load_dataset("wikitext", "wikitext-2-raw-v1")

        print("\n=== Dataset Splits ===")
        for split_name in ['train', 'validation', 'test']:
            if split_name in wikitext:
                split = wikitext[split_name]
                num_examples = len(split)

                # Count tokens (rough estimate)
                total_text = ' '.join(split['text'])
                est_tokens = len(total_text.split())

                print(f"\n{split_name.upper()}:")
                print(f"  Examples: {num_examples:,}")
                print(f"  Est. tokens: {est_tokens:,}")

                # Check if validation set is too small
                if split_name == 'validation':
                    print(f"\n=== VALIDATION SET ANALYSIS ===")
                    if est_tokens < 1000:
                        print("❌ CRITICAL: Val set < 1,000 tokens")
                        print("   → This explains PPL of 1.054!")
                        print("   → Model memorized this tiny set")
                    elif est_tokens < 10000:
                        print("⚠️  WARNING: Val set < 10,000 tokens")
                        print("   → Very small, unreliable for evaluation")
                    elif est_tokens < 50000:
                        print("⚠️  Val set < 50,000 tokens (small but usable)")
                    else:
                        print("✓ Val set size is reasonable")

                    print(f"\nWith eval_batch_size=512 and seq_length=512:")
                    tokens_per_batch = 512 * 512
                    num_batches = est_tokens / tokens_per_batch
                    print(f"  Expected batches: ~{num_batches:.0f}")
                    print(f"  You saw: 118 batches")

                    if abs(num_batches - 118) < 20:
                        print("  ✓ Matches your results (118 batches)")

    except Exception as e:
        print(f"❌ Error loading WikiText-2: {e}")
        print("\nTry downloading manually:")
        print("  python scripts/download_wikitext.py --output data/wikitext-2")

def check_tokenized_data():
    """Check if tokenized data exists."""
    print("\n" + "="*70)
    print("CHECKING TOKENIZED DATA")
    print("="*70)

    data_dir = Path("data/lm_tokenized")
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("\nRun preprocessing:")
        print("  python scripts/preprocess_lm.py \\")
        print("      --input data/wikitext-2 \\")
        print("      --vocab_size 16000 \\")
        print("      --output data/lm_tokenized")
        return

    print(f"✓ Data directory exists: {data_dir}")

    # Check for data files
    for split in ['train', 'validation', 'test']:
        split_file = data_dir / f"{split}.pt"
        if split_file.exists():
            size_mb = split_file.stat().st_size / (1024 * 1024)
            print(f"  {split}: {size_mb:.2f} MB")
        else:
            print(f"  {split}: not found")

def estimate_from_results():
    """Estimate validation size from your results."""
    print("\n" + "="*70)
    print("ESTIMATING FROM YOUR RESULTS")
    print("="*70)

    # From your logs: "Evaluating: 118 batches"
    num_batches = 118
    batch_size = 512  # from base_config.yaml
    seq_length = 512  # typical

    total_tokens = num_batches * batch_size * seq_length

    print(f"\nBased on 118 evaluation batches:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total tokens (MB of text): ~{total_tokens * 4 / 1024 / 1024:.2f} MB")

    # WikiText-2 validation is ~200k tokens
    wikitext2_val_tokens = 200_000
    print(f"\nExpected WikiText-2 validation tokens: {wikitext2_val_tokens:,}")

    if total_tokens < 1_000:
        print("\n❌ CRITICAL: < 1,000 tokens")
        print("   → Model memorized the val set")
        print("   → Explains PPL of 1.054")
    elif total_tokens < 10_000:
        print("\n⚠️  WARNING: < 10,000 tokens")
        print("   → Too small for reliable evaluation")
    elif total_tokens < 50_000:
        print("\n⚠️  < 50,000 tokens (small but usable)")
    else:
        print("\n✓ Validation set size appears reasonable")

    print(f"\n🔍 DIAGNOSIS:")
    print(f"   Your eval data: ~{total_tokens:,} tokens")
    print(f"   PPL: 1.054 (unrealistically low)")
    print(f"   Expected PPL: 80-200+ for 10M model")

    if total_tokens < 50_000:
        print(f"\n💡 LIKELY CAUSE:")
        print(f"   → Validation set is too small ({total_tokens:,} tokens)")
        print(f"   → Model has essentially memorized it")
        print(f"   → This explains the near-perfect PPL of 1.054")

def main():
    print("\n" + "="*70)
    print("VALIDATION DATA SIZE DIAGNOSTIC")
    print("="*70)
    print("Purpose: Diagnose why PPL is unrealistically low (1.054)")
    print()

    # Method 1: Check raw WikiText-2
    check_wikitext2()

    # Method 2: Check tokenized data
    check_tokenized_data()

    # Method 3: Estimate from results
    estimate_from_results()

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Verify your validation set has enough tokens:")
    print("   → Minimum: 50,000 tokens")
    print("   → Recommended: 100,000-500,000 tokens")
    print("   → WikiText-2 standard: ~200,000 tokens")

    print("\n2. If val set is too small:")
    print("   → Use standard WikiText-2 validation split")
    print("   → Or create larger custom validation set")

    print("\n3. Re-run evaluation with proper validation set")
    print("   → Expect PPL of 80-200+ for a 10M model")

    print("\n4. Run full validation checks:")
    print("   python scripts/quick_check.py --train <train> --val <val>")
    print()

if __name__ == '__main__':
    main()
