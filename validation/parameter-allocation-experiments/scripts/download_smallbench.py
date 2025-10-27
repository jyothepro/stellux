#!/usr/bin/env python3
"""Download SmallBench evaluation tasks.

This script downloads a curated set of small benchmark tasks for evaluating
language model capabilities. Tasks include classification, NLI, and QA.

The tasks are kept small (100-500 examples) to allow for quick evaluation
during training without contaminating the pretraining data.

Usage:
    python scripts/download_smallbench.py --output data/smallbench
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


# Define SmallBench tasks
SMALLBENCH_TASKS = {
    "sentiment": {
        "dataset": "sst2",
        "config": None,
        "split": "validation",
        "max_samples": 500,
        "task_type": "classification",
        "description": "Stanford Sentiment Treebank (binary sentiment)",
    },
    "nli": {
        "dataset": "glue",
        "config": "rte",
        "split": "validation",
        "max_samples": 277,  # Full validation set
        "task_type": "nli",
        "description": "Recognizing Textual Entailment",
    },
    "qa": {
        "dataset": "boolq",
        "config": None,
        "split": "validation",
        "max_samples": 500,
        "task_type": "qa",
        "description": "Boolean Questions (yes/no question answering)",
    },
    "paraphrase": {
        "dataset": "glue",
        "config": "mrpc",
        "split": "validation",
        "max_samples": 408,  # Full validation set
        "task_type": "classification",
        "description": "Microsoft Research Paraphrase Corpus",
    },
}


def download_task(task_name: str, task_config: Dict, output_dir: Path) -> Dict:
    """Download a single SmallBench task.

    Args:
        task_name: Name of the task
        task_config: Task configuration dictionary
        output_dir: Directory to save the task data

    Returns:
        Dictionary with task statistics
    """
    logger.info(f"Downloading task: {task_name}")
    logger.info(f"  Description: {task_config['description']}")

    # Load dataset
    if task_config["config"]:
        dataset = load_dataset(
            task_config["dataset"],
            task_config["config"],
            split=task_config["split"],
        )
    else:
        dataset = load_dataset(task_config["dataset"], split=task_config["split"])

    # Sample if needed
    max_samples = task_config["max_samples"]
    if len(dataset) > max_samples:
        logger.info(f"  Sampling {max_samples} from {len(dataset)} examples")
        dataset = dataset.shuffle(seed=42).select(range(max_samples))
    else:
        logger.info(f"  Using all {len(dataset)} examples")

    # Save to disk
    task_output_dir = output_dir / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(task_output_dir))

    # Collect statistics
    stats = {
        "dataset": task_config["dataset"],
        "config": task_config["config"],
        "task_type": task_config["task_type"],
        "description": task_config["description"],
        "num_examples": len(dataset),
        "features": list(dataset.features.keys()),
    }

    logger.info(f"  Saved to {task_output_dir}")

    return stats


def download_smallbench(output_dir: str) -> Dict:
    """Download all SmallBench tasks.

    Args:
        output_dir: Directory to save SmallBench data

    Returns:
        Dictionary with overall statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading SmallBench tasks to {output_path}")
    logger.info(f"Total tasks: {len(SMALLBENCH_TASKS)}")

    all_stats = {"tasks": {}, "total_examples": 0}

    # Download each task
    for task_name, task_config in SMALLBENCH_TASKS.items():
        try:
            task_stats = download_task(task_name, task_config, output_path)
            all_stats["tasks"][task_name] = task_stats
            all_stats["total_examples"] += task_stats["num_examples"]
        except Exception as e:
            logger.error(f"Failed to download task '{task_name}': {e}")
            raise

    # Save overall metadata
    metadata_file = output_path / "smallbench_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"Metadata saved to {metadata_file}")

    # Create task manifest for easy reference
    manifest = {
        "version": "1.0",
        "tasks": list(SMALLBENCH_TASKS.keys()),
        "total_tasks": len(SMALLBENCH_TASKS),
        "total_examples": all_stats["total_examples"],
        "task_types": list(set(t["task_type"] for t in SMALLBENCH_TASKS.values())),
    }

    manifest_file = output_path / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved to {manifest_file}")

    return all_stats


def main():
    """Main entry point for script."""
    parser = argparse.ArgumentParser(description="Download SmallBench evaluation tasks")
    parser.add_argument(
        "--output",
        type=str,
        default="data/smallbench",
        help="Output directory for SmallBench data (default: data/smallbench)",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks and exit",
    )

    args = parser.parse_args()

    if args.list_tasks:
        print("\nAvailable SmallBench tasks:")
        print("=" * 80)
        for task_name, config in SMALLBENCH_TASKS.items():
            print(f"\n{task_name}:")
            print(f"  Dataset: {config['dataset']}")
            if config["config"]:
                print(f"  Config: {config['config']}")
            print(f"  Type: {config['task_type']}")
            print(f"  Max samples: {config['max_samples']}")
            print(f"  Description: {config['description']}")
        print("\n")
        return

    try:
        stats = download_smallbench(args.output)
        logger.info("SmallBench download completed successfully!")
        logger.info(f"Total: {len(stats['tasks'])} tasks, "
                    f"{stats['total_examples']} examples")

        # Print summary
        print("\nSmallBench Summary:")
        print("=" * 80)
        for task_name, task_stats in stats["tasks"].items():
            print(f"{task_name:15s}: {task_stats['num_examples']:4d} examples "
                  f"({task_stats['task_type']})")
        print(f"{'TOTAL':15s}: {stats['total_examples']:4d} examples")
        print("=" * 80)

    except Exception as e:
        logger.error(f"SmallBench download failed: {e}")
        raise


if __name__ == "__main__":
    main()
