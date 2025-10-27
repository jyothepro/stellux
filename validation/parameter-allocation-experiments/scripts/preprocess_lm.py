#!/usr/bin/env python3
"""Preprocess language modeling data and train tokenizer.

This script:
1. Trains a BPE tokenizer on the training data
2. Tokenizes the dataset with the trained tokenizer
3. Saves tokenized data and computes statistics

Usage:
    python scripts/preprocess_lm.py \\
        --input data/wikitext-2 \\
        --vocab_size 16000 \\
        --output data/lm_tokenized
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import load_from_disk
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_tokenizer(
    texts: List[str], vocab_size: int, special_tokens: List[str]
) -> Tokenizer:
    """Train a BPE tokenizer on the provided texts.

    Args:
        texts: List of text strings to train on
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens (e.g., [PAD], [UNK], etc.)

    Returns:
        Trained tokenizer
    """
    logger.info(f"Training BPE tokenizer with vocab_size={vocab_size}")

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Configure trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Train on texts
    tokenizer.train_from_iterator(texts, trainer=trainer)

    logger.info(f"Tokenizer trained. Actual vocab size: {tokenizer.get_vocab_size()}")

    return tokenizer


def compute_tokenizer_hash(tokenizer_path: Path) -> str:
    """Compute SHA256 hash of tokenizer for reproducibility tracking.

    Args:
        tokenizer_path: Path to tokenizer file

    Returns:
        SHA256 hash string
    """
    with open(tokenizer_path / "tokenizer.json", "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def tokenize_dataset(dataset, tokenizer, max_length: int = 512) -> Dict:
    """Tokenize the dataset and compute statistics.

    Args:
        dataset: HuggingFace dataset to tokenize
        tokenizer: Trained tokenizer
        max_length: Maximum sequence length

    Returns:
        Tokenized dataset and statistics
    """
    logger.info("Tokenizing dataset...")

    def tokenize_function(examples):
        """Tokenize text examples."""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
        )

    # Tokenize all splits
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    # Compute statistics
    stats = {"splits": {}, "vocab_size": len(tokenizer), "max_length": max_length}

    for split_name in tokenized_dataset.keys():
        split_data = tokenized_dataset[split_name]
        input_ids = split_data["input_ids"]

        lengths = [len(ids) for ids in input_ids]
        total_tokens = sum(lengths)

        stats["splits"][split_name] = {
            "num_examples": len(split_data),
            "total_tokens": total_tokens,
            "mean_length": float(np.mean(lengths)),
            "median_length": float(np.median(lengths)),
            "std_length": float(np.std(lengths)),
            "min_length": int(np.min(lengths)),
            "max_length": int(np.max(lengths)),
        }

        logger.info(
            f"Split '{split_name}': {total_tokens:,} tokens, "
            f"mean length: {stats['splits'][split_name]['mean_length']:.1f}"
        )

    return tokenized_dataset, stats


def main():
    """Main entry point for script."""
    parser = argparse.ArgumentParser(
        description="Preprocess LM data and train tokenizer"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing raw dataset",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=16000,
        help="Vocabulary size for tokenizer (default: 16000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for tokenized data",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load raw dataset
        logger.info(f"Loading dataset from {args.input}")
        dataset = load_from_disk(args.input)

        # Define special tokens
        special_tokens = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "[BOS]",
            "[EOS]",
        ]

        # Train tokenizer on training split
        train_texts = dataset["train"]["text"]
        # Filter empty texts
        train_texts = [text for text in train_texts if text.strip()]

        tokenizer = train_tokenizer(train_texts, args.vocab_size, special_tokens)

        # Convert to HuggingFace tokenizer
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="[BOS]",
            eos_token="[EOS]",
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )

        # Save tokenizer
        tokenizer_path = output_path / "tokenizer"
        hf_tokenizer.save_pretrained(str(tokenizer_path))
        logger.info(f"Tokenizer saved to {tokenizer_path}")

        # Compute tokenizer hash for reproducibility
        tokenizer_hash = compute_tokenizer_hash(tokenizer_path)
        logger.info(f"Tokenizer hash (SHA256): {tokenizer_hash}")

        # Tokenize dataset
        tokenized_dataset, stats = tokenize_dataset(
            dataset, hf_tokenizer, args.max_length
        )

        # Save tokenized dataset
        tokenized_dataset.save_to_disk(str(output_path / "tokenized_data"))
        logger.info(f"Tokenized dataset saved to {output_path / 'tokenized_data'}")

        # Add metadata to stats
        stats["tokenizer_hash"] = tokenizer_hash
        stats["seed"] = args.seed
        stats["input_path"] = args.input

        # Save statistics
        stats_file = output_path / "tokenization_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to {stats_file}")

        # Create length histogram data for plotting
        all_lengths = []
        for split_name in tokenized_dataset.keys():
            input_ids = tokenized_dataset[split_name]["input_ids"]
            all_lengths.extend([len(ids) for ids in input_ids])

        histogram_data = {
            "lengths": all_lengths,
            "bins": 50,
            "description": "Token length distribution across all splits",
        }

        histogram_file = output_path / "length_histogram.json"
        with open(histogram_file, "w") as f:
            json.dump(histogram_data, f)
        logger.info(f"Histogram data saved to {histogram_file}")

        logger.info("Preprocessing completed successfully!")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()
