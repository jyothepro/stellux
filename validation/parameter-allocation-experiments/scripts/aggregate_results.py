#!/usr/bin/env python3
"""Aggregate experiment results from multiple runs.

This script collects metrics from all experiment runs and produces a
consolidated CSV file for analysis and plotting.

Usage:
    python scripts/aggregate_results.py \\
        --log_dir logs \\
        --output results_summary.csv
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_metrics_files(log_dir: Path) -> List[Path]:
    """Find all metrics.json files in the log directory.

    Args:
        log_dir: Directory containing experiment logs

    Returns:
        List of paths to metrics.json files
    """
    metrics_files = list(log_dir.rglob("metrics.json"))
    logger.info(f"Found {len(metrics_files)} metrics files")
    return metrics_files


def load_metrics(metrics_file: Path) -> Dict:
    """Load metrics from a single experiment run.

    Args:
        metrics_file: Path to metrics.json file

    Returns:
        Dictionary with experiment metrics
    """
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    # Extract experiment name from path
    exp_name = metrics_file.parent.name

    # Add metadata
    metrics["experiment_name"] = exp_name
    metrics["log_path"] = str(metrics_file.parent)

    return metrics


def aggregate_results(log_dir: str) -> pd.DataFrame:
    """Aggregate results from all experiment runs.

    Args:
        log_dir: Directory containing experiment logs

    Returns:
        DataFrame with aggregated results
    """
    log_path = Path(log_dir)

    if not log_path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    # Find all metrics files
    metrics_files = find_metrics_files(log_path)

    if not metrics_files:
        logger.warning(f"No metrics files found in {log_dir}")
        return pd.DataFrame()

    # Load all metrics
    all_metrics = []
    for metrics_file in metrics_files:
        try:
            metrics = load_metrics(metrics_file)
            all_metrics.append(metrics)
            logger.info(f"Loaded metrics from {metrics['experiment_name']}")
        except Exception as e:
            logger.error(f"Failed to load {metrics_file}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)

    # Sort by experiment name
    if "experiment_name" in df.columns:
        df = df.sort_values("experiment_name")

    logger.info(f"Aggregated {len(df)} experiments")

    return df


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for experiments with multiple seeds.

    Args:
        df: DataFrame with experiment results

    Returns:
        DataFrame with summary statistics
    """
    if df.empty:
        return df

    # Group by experiment configuration (excluding seed)
    if "seed" in df.columns and "experiment_name" in df.columns:
        # Extract base experiment name (without seed suffix)
        df["base_experiment"] = df["experiment_name"].str.replace(
            r"_seed\d+$", "", regex=True
        )

        # Compute statistics per base experiment
        numeric_cols = df.select_dtypes(include=["number"]).columns
        summary = df.groupby("base_experiment")[numeric_cols].agg(
            ["mean", "std", "min", "max", "count"]
        )

        logger.info(f"Computed summary statistics for {len(summary)} base experiments")

        return summary

    return df


def main():
    """Main entry point for script."""
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory containing experiment logs (default: logs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_summary.csv",
        help="Output CSV file (default: results_summary.csv)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Also compute summary statistics for multi-seed experiments",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "json", "excel"],
        default="csv",
        help="Output format (default: csv)",
    )

    args = parser.parse_args()

    try:
        # Aggregate results
        df = aggregate_results(args.log_dir)

        if df.empty:
            logger.warning("No results to aggregate")
            return

        # Save main results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.format == "csv":
            df.to_csv(output_path, index=False)
        elif args.format == "json":
            df.to_json(output_path, orient="records", indent=2)
        elif args.format == "excel":
            df.to_excel(output_path, index=False)

        logger.info(f"Results saved to {output_path}")

        # Print summary
        print("\nExperiment Summary:")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

        # Compute and save summary statistics if requested
        if args.summary:
            summary_df = compute_summary_statistics(df)
            if not summary_df.empty:
                summary_path = output_path.parent / f"summary_{output_path.name}"
                summary_df.to_csv(summary_path)
                logger.info(f"Summary statistics saved to {summary_path}")

        logger.info("Aggregation completed successfully!")

    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
        raise


if __name__ == "__main__":
    main()
