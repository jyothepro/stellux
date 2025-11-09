"""Per-run summary generation utilities."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .metrics import StandardMetrics

logger = logging.getLogger(__name__)


def generate_run_summary(
    metrics: StandardMetrics,
    output_path: str,
    format: str = "markdown",
) -> None:
    """Generate a human-readable summary of a run.

    Args:
        metrics: StandardMetrics object with run data
        output_path: Path to save summary
        format: Output format ("markdown" or "json" or "both")
    """
    if format in ("markdown", "both"):
        _generate_markdown_summary(metrics, output_path)

    if format in ("json", "both"):
        json_path = output_path.replace(".md", ".json")
        metrics.to_json(json_path)


def _generate_markdown_summary(
    metrics: StandardMetrics,
    output_path: str,
) -> None:
    """Generate markdown summary.

    Args:
        metrics: StandardMetrics object
        output_path: Path to save markdown file
    """
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Build markdown content
    md = []
    md.append(f"# Experiment Run Summary: {metrics.experiment_name}")
    md.append("")
    md.append(f"**Run ID:** `{metrics.run_id}`")
    md.append(f"**Timestamp:** {metrics.timestamp}")
    md.append(f"**Device:** {metrics.device}")
    md.append("")

    # Model Configuration
    md.append("## Model Configuration")
    md.append("")
    md.append("| Parameter | Value |")
    md.append("|-----------|-------|")
    md.append(f"| Total Parameters | {metrics.total_params:,} |")
    md.append(f"| d_model | {metrics.d_model} |")
    md.append(f"| d_ff | {metrics.d_ff} |")
    md.append(f"| n_layers | {metrics.n_layers} |")
    md.append(f"| n_heads | {metrics.n_heads} |")
    md.append(f"| Embedding Ratio | {metrics.embedding_ratio:.2%} |")
    md.append(f"| GLU Expansion | {metrics.glu_expansion:.2f}× |")
    md.append("")

    # Training Results
    md.append("## Training Results")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Total Steps | {metrics.total_steps:,} |")
    md.append(f"| Total Epochs | {metrics.total_epochs} |")
    md.append(f"| Total Tokens | {metrics.total_tokens:,} |")
    md.append(f"| Training Time | {_format_duration(metrics.training_time_seconds)} |")
    md.append(f"| Final Train Loss | {metrics.train_loss:.4f} |")
    md.append(f"| Final Train PPL | {metrics.train_perplexity:.2f} |")
    md.append(f"| Best Train Loss | {metrics.best_train_loss:.4f} |")
    md.append("")

    # Validation Results
    if metrics.val_loss is not None:
        md.append("## Validation Results")
        md.append("")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        md.append(f"| Val Loss | {metrics.val_loss:.4f} |")
        md.append(f"| Val Perplexity | {metrics.val_perplexity:.2f} |")
        if metrics.best_val_loss is not None:
            md.append(f"| Best Val Loss | {metrics.best_val_loss:.4f} |")
        md.append("")

    # Test Results
    if metrics.test_loss is not None:
        md.append("## Test Results")
        md.append("")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        md.append(f"| Test Loss | {metrics.test_loss:.4f} |")
        md.append(f"| Test Perplexity | {metrics.test_perplexity:.2f} |")
        md.append("")

    # SmallBench Results
    if metrics.smallbench_avg_acc is not None:
        md.append("## SmallBench Evaluation")
        md.append("")
        md.append("| Task | Accuracy |")
        md.append("|------|----------|")
        if metrics.smallbench_sentiment_acc is not None:
            md.append(f"| Sentiment (SST-2) | {metrics.smallbench_sentiment_acc:.2%} |")
        if metrics.smallbench_nli_acc is not None:
            md.append(f"| NLI (RTE) | {metrics.smallbench_nli_acc:.2%} |")
        if metrics.smallbench_qa_acc is not None:
            md.append(f"| QA (BoolQ) | {metrics.smallbench_qa_acc:.2%} |")
        if metrics.smallbench_paraphrase_acc is not None:
            md.append(f"| Paraphrase (MRPC) | {metrics.smallbench_paraphrase_acc:.2%} |")
        md.append(f"| **Average** | **{metrics.smallbench_avg_acc:.2%}** |")
        md.append("")

    # Performance Metrics
    has_perf = any([
        metrics.latency_batch1_seq128 is not None,
        metrics.latency_batch1_seq512 is not None,
        metrics.throughput_tokens_per_sec is not None,
    ])

    if has_perf:
        md.append("## Performance Metrics")
        md.append("")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        if metrics.latency_batch1_seq128 is not None:
            md.append(f"| Latency (batch=1, seq=128) | {metrics.latency_batch1_seq128:.2f} ms |")
        if metrics.latency_batch1_seq512 is not None:
            md.append(f"| Latency (batch=1, seq=512) | {metrics.latency_batch1_seq512:.2f} ms |")
        if metrics.throughput_tokens_per_sec is not None:
            md.append(f"| Throughput | {metrics.throughput_tokens_per_sec:,.0f} tokens/sec |")
        if metrics.memory_allocated_mb is not None:
            md.append(f"| Memory Allocated | {metrics.memory_allocated_mb:.1f} MB |")
        if metrics.memory_reserved_mb is not None:
            md.append(f"| Memory Reserved | {metrics.memory_reserved_mb:.1f} MB |")
        md.append("")

    # Reproducibility Info
    md.append("## Reproducibility")
    md.append("")
    md.append("| Item | Value |")
    md.append("|------|-------|")
    md.append(f"| Seed | {metrics.seed} |")
    if metrics.git_commit:
        md.append(f"| Git Commit | `{metrics.git_commit}` |")
    md.append("")

    # Notes
    if metrics.notes:
        md.append("## Notes")
        md.append("")
        md.append(metrics.notes)
        md.append("")

    # Model Config Details
    md.append("## Full Model Configuration")
    md.append("")
    md.append("```json")
    md.append(json.dumps(metrics.model_config, indent=2))
    md.append("```")
    md.append("")

    # Footer
    md.append("---")
    md.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    # Write to file
    content = "\n".join(md)
    with open(output_path, "w") as f:
        f.write(content)

    logger.info(f"Run summary saved to {output_path}")


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2h 34m 56s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    seconds = seconds % 60

    if minutes < 60:
        return f"{minutes}m {seconds:.0f}s"

    hours = minutes // 60
    minutes = minutes % 60

    return f"{hours}h {minutes}m {seconds:.0f}s"


def generate_comparison_table(
    metrics_list: list[StandardMetrics],
    output_path: str,
    sort_by: str = "val_perplexity",
) -> None:
    """Generate a comparison table of multiple runs.

    Args:
        metrics_list: List of StandardMetrics objects
        output_path: Path to save comparison table
        sort_by: Metric to sort by
    """
    if not metrics_list:
        logger.warning("No metrics to compare")
        return

    # Sort metrics
    if sort_by and hasattr(metrics_list[0], sort_by):
        metrics_list = sorted(
            metrics_list,
            key=lambda m: getattr(m, sort_by) or float("inf")
        )

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    md = []
    md.append("# Experiment Comparison")
    md.append("")
    md.append(f"**Total Runs:** {len(metrics_list)}")
    md.append(f"**Sorted by:** {sort_by}")
    md.append("")

    # Build comparison table
    md.append("| Rank | Experiment | Embed | GLU | Val PPL | Test PPL | SmallBench | Latency (ms) |")
    md.append("|------|------------|-------|-----|---------|----------|------------|--------------|")

    for rank, metrics in enumerate(metrics_list, 1):
        experiment = metrics.experiment_name[:20]  # Truncate
        embed = f"{metrics.embedding_ratio:.0%}"
        glu = f"{metrics.glu_expansion:.2f}×"
        val_ppl = f"{metrics.val_perplexity:.2f}" if metrics.val_perplexity else "N/A"
        test_ppl = f"{metrics.test_perplexity:.2f}" if metrics.test_perplexity else "N/A"
        smallbench = f"{metrics.smallbench_avg_acc:.1%}" if metrics.smallbench_avg_acc else "N/A"
        latency = f"{metrics.latency_batch1_seq512:.1f}" if metrics.latency_batch1_seq512 else "N/A"

        md.append(f"| {rank} | {experiment} | {embed} | {glu} | {val_ppl} | {test_ppl} | {smallbench} | {latency} |")

    md.append("")

    # Write to file
    content = "\n".join(md)
    with open(output_path, "w") as f:
        f.write(content)

    logger.info(f"Comparison table saved to {output_path}")
