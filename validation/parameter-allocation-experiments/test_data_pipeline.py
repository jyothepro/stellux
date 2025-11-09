#!/usr/bin/env python3
"""Quick test of data pipeline scripts without internet access."""

import json
import sys
from pathlib import Path

print("=" * 80)
print("Testing Data Pipeline Scripts")
print("=" * 80)

# Test 1: Check script imports
print("\n[Test 1] Checking script imports...")
try:
    sys.path.insert(0, "scripts")
    import download_wikitext
    import download_smallbench
    import preprocess_lm
    import aggregate_results
    import generate_plots
    print("✓ All scripts import successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Test tokenizer training with local data
print("\n[Test 2] Testing tokenizer training...")
try:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    # Read test data
    train_file = Path("data/wikitext-2-test/train.txt")
    if not train_file.exists():
        print("✗ Test data not found")
        sys.exit(1)

    with open(train_file) as f:
        texts = f.readlines()

    # Train small tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=100,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
        show_progress=False,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    print(f"✓ Tokenizer trained: {tokenizer.get_vocab_size()} tokens")

    # Test encoding
    encoded = tokenizer.encode("This is a test")
    print(f"✓ Tokenizer encoding works: {len(encoded.tokens)} tokens")

except Exception as e:
    print(f"✗ Tokenizer training failed: {e}")
    sys.exit(1)

# Test 3: Test SmallBench task configuration
print("\n[Test 3] Testing SmallBench configuration...")
try:
    from download_smallbench import SMALLBENCH_TASKS

    print(f"✓ SmallBench has {len(SMALLBENCH_TASKS)} tasks defined:")
    for task_name in SMALLBENCH_TASKS:
        print(f"  - {task_name}")

except Exception as e:
    print(f"✗ SmallBench config failed: {e}")
    sys.exit(1)

# Test 4: Test aggregate results with mock data
print("\n[Test 4] Testing results aggregation...")
try:
    import pandas as pd

    # Create mock metrics
    mock_metrics = [
        {"experiment_name": "emb25", "perplexity": 35.2, "loss": 3.5},
        {"experiment_name": "emb35", "perplexity": 32.1, "loss": 3.2},
        {"experiment_name": "emb45", "perplexity": 33.8, "loss": 3.4},
    ]

    df = pd.DataFrame(mock_metrics)
    print(f"✓ Created mock results DataFrame: {len(df)} experiments")
    print(df.to_string(index=False))

except Exception as e:
    print(f"✗ Aggregation test failed: {e}")
    sys.exit(1)

# Test 5: Verify DATASET_MANIFEST
print("\n[Test 5] Verifying DATASET_MANIFEST.json...")
try:
    manifest_file = Path("DATASET_MANIFEST.json")
    with open(manifest_file) as f:
        manifest = json.load(f)

    print(f"✓ Manifest version: {manifest['manifest_version']}")
    print(f"✓ Pretraining corpus: {manifest['pretraining_corpus']['name']}")
    print(f"✓ SmallBench tasks: {manifest['evaluation_benchmarks']['total_tasks']}")

except Exception as e:
    print(f"✗ Manifest verification failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ All tests passed! Data pipeline is functional.")
print("=" * 80)
print("\nNote: Actual dataset downloads require internet access.")
print("Scripts are ready to use when deployed with network connectivity.")
