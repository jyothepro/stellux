#!/usr/bin/env python3
"""Download WikiText-2 dataset for language model pretraining.

This script downloads the WikiText-2 dataset from HuggingFace and saves it
to the specified output directory. The dataset contains ~50M tokens which
is suitable for our 10M parameter experiments.

Usage:
    python scripts/download_wikitext.py --output data/wikitext-2
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_wikitext(output_dir: str, dataset_name: str = "wikitext-2-raw-v1") -> Dict:
    """Download WikiText dataset and save to disk.

    Args:
        output_dir: Directory to save the dataset
        dataset_name: Name of WikiText dataset variant (default: wikitext-2-raw-v1)

    Returns:
        Dictionary with dataset statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading WikiText dataset: {dataset_name}")

    # Download dataset from HuggingFace
    dataset = load_dataset("wikitext", dataset_name)

    logger.info("Dataset downloaded successfully")
    logger.info(f"Splits: {list(dataset.keys())}")

    # Save dataset to disk
    dataset.save_to_disk(str(output_path))
    logger.info(f"Dataset saved to {output_path}")

    # Collect statistics
    stats = {
        "dataset_name": dataset_name,
        "splits": {},
        "total_examples": 0,
        "total_characters": 0,
    }

    for split_name, split_data in dataset.items():
        num_examples = len(split_data)
        texts = split_data["text"]
        total_chars = sum(len(text) for text in texts)

        stats["splits"][split_name] = {
            "num_examples": num_examples,
            "total_characters": total_chars,
        }
        stats["total_examples"] += num_examples
        stats["total_characters"] += total_chars

        logger.info(
            f"Split '{split_name}': {num_examples} examples, {total_chars:,} characters"
        )

    # Save statistics
    stats_file = output_path / "dataset_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Statistics saved to {stats_file}")

    return stats


def main():
    """Main entry point for script."""
    parser = argparse.ArgumentParser(
        description="Download WikiText-2 dataset for LM pretraining"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/wikitext-2",
        help="Output directory for dataset (default: data/wikitext-2)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext-2-raw-v1",
        choices=["wikitext-2-raw-v1", "wikitext-2-v1", "wikitext-103-raw-v1"],
        help="WikiText dataset variant (default: wikitext-2-raw-v1)",
    )

    args = parser.parse_args()

    try:
        stats = download_wikitext(args.output, args.dataset)
        logger.info("Download completed successfully!")
        logger.info(f"Total: {stats['total_examples']} examples, "
                    f"{stats['total_characters']:,} characters")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


if __name__ == "__main__":
    main()
