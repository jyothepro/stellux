#!/usr/bin/env python3
"""Generate plots and visualizations from experiment results.

This script creates publication-ready plots for the paper, including:
- PPL vs embedding ratio
- PPL vs GLU expansion factor
- Pareto frontier (compute/latency vs perplexity)
- Token length histograms

Usage:
    python scripts/generate_plots.py \\
        --input results_summary.csv \\
        --output_dir reports/figures
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set publication-quality plotting defaults
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (6, 4),
    }
)


def plot_embedding_ratio_sweep(
    df: pd.DataFrame, output_dir: Path, metric: str = "perplexity"
) -> None:
    """Plot perplexity vs embedding ratio.

    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save plots
        metric: Metric to plot (default: perplexity)
    """
    logger.info("Generating embedding ratio sweep plot")

    # Filter for embedding experiments
    embed_df = df[df["experiment_name"].str.contains("emb", case=False, na=False)]

    if embed_df.empty:
        logger.warning("No embedding ratio experiments found")
        return

    # Extract embedding ratio from experiment name
    embed_df = embed_df.copy()
    embed_df["embedding_ratio"] = (
        embed_df["experiment_name"].str.extract(r"emb(\d+)")[0].astype(float) / 100
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        embed_df["embedding_ratio"],
        embed_df[metric],
        marker="o",
        linewidth=2,
        markersize=8,
        color="#2E86AB",
    )

    ax.set_xlabel("Embedding Parameter Ratio")
    ax.set_ylabel(f"{metric.capitalize()}")
    ax.set_title(f"{metric.capitalize()} vs Embedding Parameter Ratio")
    ax.grid(True, alpha=0.3)

    # Add value annotations
    for _, row in embed_df.iterrows():
        ax.annotate(
            f"{row[metric]:.2f}",
            (row["embedding_ratio"], row[metric]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()

    output_file = output_dir / f"embedding_ratio_{metric}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot to {output_file}")
    plt.close()


def plot_glu_expansion_sweep(
    df: pd.DataFrame, output_dir: Path, metric: str = "perplexity"
) -> None:
    """Plot perplexity vs GLU expansion factor.

    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save plots
        metric: Metric to plot (default: perplexity)
    """
    logger.info("Generating GLU expansion sweep plot")

    # Filter for GLU experiments
    glu_df = df[df["experiment_name"].str.contains("glu", case=False, na=False)]

    if glu_df.empty:
        logger.warning("No GLU expansion experiments found")
        return

    # Extract GLU expansion from experiment name
    glu_df = glu_df.copy()
    # Handle both "glu2x" and "glu266x" formats
    glu_df["glu_expansion"] = glu_df["experiment_name"].str.extract(r"glu(\d+)")[0]
    # Convert 266 -> 2.66, etc
    glu_df["glu_expansion"] = glu_df["glu_expansion"].astype(float)
    glu_df.loc[glu_df["glu_expansion"] >= 100, "glu_expansion"] /= 100

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        glu_df["glu_expansion"],
        glu_df[metric],
        marker="s",
        linewidth=2,
        markersize=8,
        color="#A23B72",
    )

    ax.set_xlabel("GLU Expansion Factor")
    ax.set_ylabel(f"{metric.capitalize()}")
    ax.set_title(f"{metric.capitalize()} vs GLU Expansion Factor")
    ax.grid(True, alpha=0.3)

    # Add value annotations
    for _, row in glu_df.iterrows():
        ax.annotate(
            f"{row[metric]:.2f}",
            (row["glu_expansion"], row[metric]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()

    output_file = output_dir / f"glu_expansion_{metric}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot to {output_file}")
    plt.close()


def plot_pareto_frontier(
    df: pd.DataFrame,
    output_dir: Path,
    x_metric: str = "latency_ms",
    y_metric: str = "perplexity",
) -> None:
    """Plot Pareto frontier of compute/latency vs perplexity.

    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save plots
        x_metric: X-axis metric (default: latency_ms)
        y_metric: Y-axis metric (default: perplexity)
    """
    logger.info("Generating Pareto frontier plot")

    if x_metric not in df.columns or y_metric not in df.columns:
        logger.warning(f"Metrics {x_metric} or {y_metric} not found in data")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4))

    # Color by experiment type
    colors = {"emb": "#2E86AB", "glu": "#A23B72"}

    for exp_type, color in colors.items():
        mask = df["experiment_name"].str.contains(exp_type, case=False, na=False)
        subset = df[mask]

        if not subset.empty:
            ax.scatter(
                subset[x_metric],
                subset[y_metric],
                label=exp_type.upper(),
                color=color,
                s=100,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
            )

            # Add labels
            for _, row in subset.iterrows():
                ax.annotate(
                    row["experiment_name"],
                    (row[x_metric], row[y_metric]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                    alpha=0.8,
                )

    ax.set_xlabel(f"{x_metric.replace('_', ' ').title()}")
    ax.set_ylabel(f"{y_metric.capitalize()}")
    ax.set_title(f"Pareto Frontier: {y_metric.capitalize()} vs {x_metric.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / f"pareto_{x_metric}_{y_metric}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot to {output_file}")
    plt.close()


def plot_token_length_histogram(
    histogram_file: Path, output_dir: Path
) -> None:
    """Plot token length distribution histogram.

    Args:
        histogram_file: Path to histogram data JSON
        output_dir: Directory to save plots
    """
    logger.info("Generating token length histogram")

    if not histogram_file.exists():
        logger.warning(f"Histogram file not found: {histogram_file}")
        return

    # Load histogram data
    with open(histogram_file, "r") as f:
        data = json.load(f)

    lengths = data["lengths"]
    bins = data.get("bins", 50)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(lengths, bins=bins, color="#18A558", alpha=0.7, edgecolor="black")

    ax.set_xlabel("Token Length")
    ax.set_ylabel("Frequency")
    ax.set_title("Token Length Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    # Add statistics
    mean_len = np.mean(lengths)
    median_len = np.median(lengths)
    ax.axvline(mean_len, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_len:.1f}")
    ax.axvline(median_len, color="blue", linestyle="--", linewidth=2, label=f"Median: {median_len:.1f}")
    ax.legend()

    plt.tight_layout()

    output_file = output_dir / "token_length_histogram.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot to {output_file}")
    plt.close()


def main():
    """Main entry point for script."""
    parser = argparse.ArgumentParser(description="Generate plots from experiment results")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file with aggregated results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/figures",
        help="Output directory for plots (default: reports/figures)",
    )
    parser.add_argument(
        "--histogram_file",
        type=str,
        help="Path to token length histogram JSON (optional)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format (default: png)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load results
        logger.info(f"Loading results from {args.input}")
        df = pd.read_csv(args.input)

        if df.empty:
            logger.warning("No data to plot")
            return

        logger.info(f"Loaded {len(df)} experiments")

        # Generate plots
        plot_embedding_ratio_sweep(df, output_dir, metric="perplexity")
        plot_glu_expansion_sweep(df, output_dir, metric="perplexity")

        # Generate Pareto plot if latency data exists
        if "latency_ms" in df.columns:
            plot_pareto_frontier(df, output_dir)

        # Generate token length histogram if file provided
        if args.histogram_file:
            histogram_path = Path(args.histogram_file)
            plot_token_length_histogram(histogram_path, output_dir)

        logger.info("Plot generation completed successfully!")
        logger.info(f"Plots saved to {output_dir}")

    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
