"""Standardized metrics schema and logging utilities."""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StandardMetrics:
    """Standardized metrics schema for experiment tracking.

    This schema ensures consistent reporting across all experiments.
    """

    # Experiment identification
    run_id: str
    experiment_name: str
    timestamp: str

    # Model configuration
    model_config: Dict[str, Any]
    total_params: int
    d_model: int
    d_ff: int
    n_layers: int
    n_heads: int
    embedding_ratio: float
    glu_expansion: float

    # Training metrics
    train_loss: float
    train_perplexity: float
    best_train_loss: float

    # Validation metrics
    val_loss: Optional[float] = None
    val_perplexity: Optional[float] = None
    best_val_loss: Optional[float] = None

    # Test metrics
    test_loss: Optional[float] = None
    test_perplexity: Optional[float] = None

    # SmallBench evaluation metrics
    smallbench_sentiment_acc: Optional[float] = None
    smallbench_nli_acc: Optional[float] = None
    smallbench_qa_acc: Optional[float] = None
    smallbench_paraphrase_acc: Optional[float] = None
    smallbench_avg_acc: Optional[float] = None

    # Performance metrics
    latency_batch1_seq128: Optional[float] = None  # ms
    latency_batch1_seq512: Optional[float] = None  # ms
    throughput_tokens_per_sec: Optional[float] = None
    memory_allocated_mb: Optional[float] = None
    memory_reserved_mb: Optional[float] = None

    # Training info
    total_steps: int = 0
    total_epochs: int = 0
    total_tokens: int = 0
    training_time_seconds: float = 0.0

    # Hardware info
    device: str = "cpu"
    num_gpus: int = 0

    # Additional metadata
    seed: int = 42
    git_commit: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)

    def to_json(self, path: str, indent: int = 2) -> None:
        """Save metrics to JSON file.

        Args:
            path: Output file path
            indent: JSON indentation level
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)
        logger.info(f"Metrics saved to {path}")

    @classmethod
    def from_json(cls, path: str) -> "StandardMetrics":
        """Load metrics from JSON file.

        Args:
            path: Input file path

        Returns:
            StandardMetrics instance
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


class MetricsLogger:
    """Logger for tracking metrics during training and evaluation.

    Features:
    - Accumulates metrics over time
    - Computes statistics (mean, min, max)
    - Saves to standardized JSON format
    """

    def __init__(self, log_dir: str = "logs"):
        """Initialize metrics logger.

        Args:
            log_dir: Directory to save metrics logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Storage for accumulated metrics
        self.metrics_history: List[Dict[str, Any]] = []
        self.step_metrics: Dict[str, List[float]] = {}

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics for a single step.

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number
        """
        # Add step if provided
        if step is not None:
            metrics["step"] = step

        # Add timestamp
        metrics["timestamp"] = datetime.now().isoformat()

        # Store in history
        self.metrics_history.append(metrics.copy())

        # Accumulate for statistics
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key != "step":
                if key not in self.step_metrics:
                    self.step_metrics[key] = []
                self.step_metrics[key].append(value)

    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Dictionary with mean, min, max, last
        """
        if metric_name not in self.step_metrics:
            return {}

        values = self.step_metrics[metric_name]
        return {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "last": values[-1],
            "count": len(values),
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics.

        Returns:
            Dictionary mapping metric names to their statistics
        """
        return {
            metric: self.get_stats(metric)
            for metric in self.step_metrics.keys()
        }

    def save_history(self, filename: str = "metrics_history.jsonl") -> None:
        """Save full metrics history to JSONL file.

        Args:
            filename: Output filename
        """
        output_path = self.log_dir / filename
        with open(output_path, "w") as f:
            for entry in self.metrics_history:
                f.write(json.dumps(entry) + "\n")
        logger.info(f"Metrics history saved to {output_path}")

    def save_summary(self, filename: str = "metrics_summary.json") -> None:
        """Save statistics summary to JSON file.

        Args:
            filename: Output filename
        """
        output_path = self.log_dir / filename
        summary = self.get_all_stats()
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Metrics summary saved to {output_path}")

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.metrics_history.clear()
        self.step_metrics.clear()


def load_metrics_from_run(run_dir: str) -> Optional[StandardMetrics]:
    """Load standardized metrics from a run directory.

    Args:
        run_dir: Path to run directory containing metrics.json

    Returns:
        StandardMetrics if found, None otherwise
    """
    metrics_path = Path(run_dir) / "metrics.json"
    if not metrics_path.exists():
        logger.warning(f"Metrics file not found: {metrics_path}")
        return None

    try:
        return StandardMetrics.from_json(str(metrics_path))
    except Exception as e:
        logger.error(f"Failed to load metrics from {metrics_path}: {e}")
        return None


def aggregate_metrics(run_dirs: List[str]) -> List[StandardMetrics]:
    """Aggregate metrics from multiple run directories.

    Args:
        run_dirs: List of run directory paths

    Returns:
        List of StandardMetrics objects
    """
    metrics_list = []

    for run_dir in run_dirs:
        metrics = load_metrics_from_run(run_dir)
        if metrics is not None:
            metrics_list.append(metrics)

    logger.info(f"Loaded metrics from {len(metrics_list)}/{len(run_dirs)} runs")
    return metrics_list
