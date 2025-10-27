#!/usr/bin/env python3
"""
Aggregate results from Phase 1 ranking runs.

Collects metrics from all Phase 1 experiments and selects top-1 per axis.

Usage:
    python scripts/aggregate_phase1_results.py \
        --results-dir outputs/phase1 \
        --output results/phase1_summary.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval.metrics import load_metrics_from_run

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def find_phase1_runs(results_dir: str) -> List[Path]:
    """Find all Phase 1 experiment directories.

    Args:
        results_dir: Base results directory

    Returns:
        List of experiment directories
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        logger.warning(f"Results directory not found: {results_dir}")
        return []

    # Find all directories that contain metrics.json
    run_dirs = []

    for item in results_path.rglob("metrics.json"):
        run_dir = item.parent
        run_dirs.append(run_dir)

    logger.info(f"Found {len(run_dirs)} Phase 1 runs")

    return run_dirs


def aggregate_phase1_metrics(run_dirs: List[Path]) -> pd.DataFrame:
    """Aggregate metrics from Phase 1 runs.

    Args:
        run_dirs: List of run directories

    Returns:
        DataFrame with aggregated metrics
    """
    records = []

    for run_dir in run_dirs:
        try:
            # Load metrics
            metrics = load_metrics_from_run(str(run_dir))

            if metrics is None:
                logger.warning(f"No metrics found in {run_dir}")
                continue

            # Extract key metrics
            record = {
                "run_id": metrics.run_id,
                "experiment_name": metrics.experiment_name,
                "embedding_ratio": metrics.embedding_ratio,
                "glu_expansion": metrics.glu_expansion,
                "d_model": metrics.d_model,
                "d_ff": metrics.d_ff,
                "n_layers": metrics.n_layers,
                "total_params": metrics.total_params,
                "train_loss": metrics.train_loss,
                "train_perplexity": metrics.train_perplexity,
                "val_loss": metrics.val_loss,
                "val_perplexity": metrics.val_perplexity,
                "best_train_loss": metrics.best_train_loss,
                "best_val_loss": metrics.best_val_loss,
                "total_steps": metrics.total_steps,
                "total_tokens": metrics.total_tokens,
                "training_time_seconds": metrics.training_time_seconds,
            }

            records.append(record)

        except Exception as e:
            logger.error(f"Failed to load metrics from {run_dir}: {e}")
            continue

    if not records:
        logger.warning("No metrics loaded!")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    logger.info(f"Aggregated {len(df)} experiments")

    return df


def select_top1_per_axis(df: pd.DataFrame) -> Dict[str, Dict]:
    """Select top-1 configuration per axis (embedding ratio, GLU expansion).

    Args:
        df: DataFrame with aggregated metrics

    Returns:
        Dictionary with top-1 for each axis
    """
    top1 = {}

    # Top-1 embedding ratio (best val_perplexity)
    if "embedding_ratio" in df.columns:
        emb_sweeps = df[df["experiment_name"].str.contains("emb", case=False)]

        if not emb_sweeps.empty:
            best_emb = emb_sweeps.loc[emb_sweeps["val_perplexity"].idxmin()]

            top1["embedding_ratio"] = {
                "experiment_name": best_emb["experiment_name"],
                "embedding_ratio": best_emb["embedding_ratio"],
                "val_perplexity": best_emb["val_perplexity"],
                "val_loss": best_emb["val_loss"],
                "d_model": best_emb["d_model"],
                "n_layers": best_emb["n_layers"],
            }

            logger.info(
                f"Top-1 embedding ratio: {best_emb['embedding_ratio']:.2%} "
                f"(PPL={best_emb['val_perplexity']:.2f})"
            )

    # Top-1 GLU expansion (best val_perplexity)
    if "glu_expansion" in df.columns:
        glu_sweeps = df[df["experiment_name"].str.contains("glu", case=False)]

        if not glu_sweeps.empty:
            best_glu = glu_sweeps.loc[glu_sweeps["val_perplexity"].idxmin()]

            top1["glu_expansion"] = {
                "experiment_name": best_glu["experiment_name"],
                "glu_expansion": best_glu["glu_expansion"],
                "val_perplexity": best_glu["val_perplexity"],
                "val_loss": best_glu["val_loss"],
                "d_ff": best_glu["d_ff"],
                "n_layers": best_glu["n_layers"],
            }

            logger.info(
                f"Top-1 GLU expansion: {best_glu['glu_expansion']:.2f}× "
                f"(PPL={best_glu['val_perplexity']:.2f})"
            )

    return top1


def generate_comparison_table(df: pd.DataFrame, axis: str = "embedding_ratio") -> None:
    """Generate comparison table for an axis.

    Args:
        df: DataFrame with metrics
        axis: Axis to compare ("embedding_ratio" or "glu_expansion")
    """
    if axis == "embedding_ratio":
        sweep_df = df[df["experiment_name"].str.contains("emb", case=False)]
        sort_col = "embedding_ratio"
    else:
        sweep_df = df[df["experiment_name"].str.contains("glu", case=False)]
        sort_col = "glu_expansion"

    if sweep_df.empty:
        logger.warning(f"No experiments found for {axis}")
        return

    # Sort by axis value
    sweep_df = sweep_df.sort_values(sort_col)

    # Print table
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Phase 1 Results: {axis.replace('_', ' ').title()}")
    logger.info(f"{'=' * 80}")

    logger.info(
        f"{'Experiment':<20} {axis.replace('_', ' ').title():<15} "
        f"{'Val PPL':<10} {'Val Loss':<10} {'Layers':<8}"
    )
    logger.info("-" * 80)

    for _, row in sweep_df.iterrows():
        exp_name = row["experiment_name"]
        axis_value = row[sort_col]
        val_ppl = row["val_perplexity"]
        val_loss = row["val_loss"]
        n_layers = row["n_layers"]

        if axis == "embedding_ratio":
            axis_str = f"{axis_value:.2%}"
        else:
            axis_str = f"{axis_value:.2f}×"

        logger.info(
            f"{exp_name:<20} {axis_str:<15} "
            f"{val_ppl:<10.2f} {val_loss:<10.4f} {n_layers:<8}"
        )

    logger.info("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Phase 1 ranking results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="outputs/phase1",
        help="Directory containing Phase 1 results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/phase1_summary.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--top1-json",
        type=str,
        default="results/phase1_top1.json",
        help="Output JSON for top-1 selections",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    logger.info("Aggregating Phase 1 ranking results...")

    # Find all Phase 1 runs
    run_dirs = find_phase1_runs(args.results_dir)

    if not run_dirs:
        logger.error("No Phase 1 runs found!")
        sys.exit(1)

    # Aggregate metrics
    df = aggregate_phase1_metrics(run_dirs)

    if df.empty:
        logger.error("No metrics aggregated!")
        sys.exit(1)

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Saved aggregated results to {output_path}")

    # Select top-1 per axis
    logger.info("\nSelecting top-1 configurations per axis...")
    top1 = select_top1_per_axis(df)

    # Save top-1 to JSON
    top1_path = Path(args.top1_json)
    top1_path.parent.mkdir(parents=True, exist_ok=True)

    with open(top1_path, "w") as f:
        json.dump(top1, f, indent=2)

    logger.info(f"Saved top-1 selections to {top1_path}")

    # Generate comparison tables
    generate_comparison_table(df, "embedding_ratio")
    generate_comparison_table(df, "glu_expansion")

    # Print summary
    logger.info("=" * 80)
    logger.info("Phase 1 Summary")
    logger.info("=" * 80)
    logger.info(f"Total experiments: {len(df)}")

    if "embedding_ratio" in top1:
        emb_top1 = top1["embedding_ratio"]
        logger.info(
            f"\nTop-1 Embedding Ratio: {emb_top1['embedding_ratio']:.2%}"
        )
        logger.info(f"  Experiment: {emb_top1['experiment_name']}")
        logger.info(f"  Val PPL: {emb_top1['val_perplexity']:.2f}")

    if "glu_expansion" in top1:
        glu_top1 = top1["glu_expansion"]
        logger.info(
            f"\nTop-1 GLU Expansion: {glu_top1['glu_expansion']:.2f}×"
        )
        logger.info(f"  Experiment: {glu_top1['experiment_name']}")
        logger.info(f"  Val PPL: {glu_top1['val_perplexity']:.2f}")

    logger.info("=" * 80)

    logger.info("\nPhase 1 aggregation complete!")


if __name__ == "__main__":
    main()
