"""Tests for evaluation and metrics modules (Milestone 4)."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Import evaluation modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval.evaluate import evaluate_perplexity, compute_token_accuracy
from eval.metrics import StandardMetrics, MetricsLogger
from eval.profiler import LatencyProfiler, ThroughputProfiler, get_memory_stats
from eval.summary import generate_run_summary, _format_duration
from models.lm import LanguageModel, ModelConfig


# Fixtures
@pytest.fixture
def small_model():
    """Create a small model for testing."""
    config = ModelConfig(
        vocab_size=1000,
        total_params=100_000,
        embedding_ratio=0.35,
        glu_expansion=2.66,
        n_heads=4,
        max_seq_length=128,
    )
    model = LanguageModel(config)
    return model


@pytest.fixture
def dummy_dataloader():
    """Create a dummy dataloader for testing."""
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, 1000, (64,)),
                "labels": torch.randint(0, 1000, (64,)),
            }

    dataset = DummyDataset()
    return torch.utils.data.DataLoader(dataset, batch_size=2)


# Tests for metrics.py
class TestStandardMetrics:
    """Tests for StandardMetrics class."""

    def test_create_metrics(self):
        """Test creating StandardMetrics object."""
        metrics = StandardMetrics(
            run_id="test_run",
            experiment_name="test_exp",
            timestamp="2025-01-01T00:00:00",
            model_config={"vocab_size": 1000},
            total_params=100_000,
            d_model=256,
            d_ff=682,
            n_layers=4,
            n_heads=4,
            embedding_ratio=0.35,
            glu_expansion=2.66,
            train_loss=2.5,
            train_perplexity=12.18,
            best_train_loss=2.3,
        )

        assert metrics.run_id == "test_run"
        assert metrics.total_params == 100_000
        assert metrics.d_model == 256
        assert metrics.train_loss == 2.5

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = StandardMetrics(
            run_id="test_run",
            experiment_name="test_exp",
            timestamp="2025-01-01T00:00:00",
            model_config={},
            total_params=100_000,
            d_model=256,
            d_ff=682,
            n_layers=4,
            n_heads=4,
            embedding_ratio=0.35,
            glu_expansion=2.66,
            train_loss=2.5,
            train_perplexity=12.18,
            best_train_loss=2.3,
        )

        data = metrics.to_dict()
        assert isinstance(data, dict)
        assert data["run_id"] == "test_run"
        assert data["total_params"] == 100_000

    def test_save_load_json(self):
        """Test saving and loading metrics from JSON."""
        metrics = StandardMetrics(
            run_id="test_run",
            experiment_name="test_exp",
            timestamp="2025-01-01T00:00:00",
            model_config={},
            total_params=100_000,
            d_model=256,
            d_ff=682,
            n_layers=4,
            n_heads=4,
            embedding_ratio=0.35,
            glu_expansion=2.66,
            train_loss=2.5,
            train_perplexity=12.18,
            best_train_loss=2.3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            metrics.to_json(str(path))

            # Check file exists
            assert path.exists()

            # Load and verify
            loaded = StandardMetrics.from_json(str(path))
            assert loaded.run_id == metrics.run_id
            assert loaded.total_params == metrics.total_params
            assert loaded.train_loss == metrics.train_loss


class TestMetricsLogger:
    """Tests for MetricsLogger class."""

    def test_log_metrics(self):
        """Test logging metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir)

            # Log some metrics
            logger.log({"loss": 2.5, "accuracy": 0.8}, step=1)
            logger.log({"loss": 2.3, "accuracy": 0.82}, step=2)
            logger.log({"loss": 2.1, "accuracy": 0.85}, step=3)

            assert len(logger.metrics_history) == 3
            assert "loss" in logger.step_metrics
            assert "accuracy" in logger.step_metrics

    def test_get_stats(self):
        """Test getting statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir)

            # Log metrics
            logger.log({"loss": 2.5}, step=1)
            logger.log({"loss": 2.0}, step=2)
            logger.log({"loss": 1.5}, step=3)

            # Get stats
            stats = logger.get_stats("loss")
            assert stats["mean"] == 2.0
            assert stats["min"] == 1.5
            assert stats["max"] == 2.5
            assert stats["last"] == 1.5

    def test_save_history(self):
        """Test saving metrics history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir)

            # Log metrics
            logger.log({"loss": 2.5}, step=1)
            logger.log({"loss": 2.0}, step=2)

            # Save history
            logger.save_history()

            # Check file exists
            history_file = Path(tmpdir) / "metrics_history.jsonl"
            assert history_file.exists()

            # Verify content
            lines = history_file.read_text().strip().split("\n")
            assert len(lines) == 2

    def test_save_summary(self):
        """Test saving metrics summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir)

            # Log metrics
            logger.log({"loss": 2.5, "accuracy": 0.8}, step=1)
            logger.log({"loss": 2.0, "accuracy": 0.85}, step=2)

            # Save summary
            logger.save_summary()

            # Check file exists
            summary_file = Path(tmpdir) / "metrics_summary.json"
            assert summary_file.exists()

            # Verify content
            with open(summary_file) as f:
                summary = json.load(f)

            assert "loss" in summary
            assert "accuracy" in summary
            assert summary["loss"]["mean"] == 2.25


# Tests for evaluate.py
class TestEvaluation:
    """Tests for evaluation functions."""

    def test_evaluate_perplexity(self, small_model, dummy_dataloader):
        """Test perplexity evaluation."""
        device = "cpu"
        results = evaluate_perplexity(
            model=small_model,
            dataloader=dummy_dataloader,
            device=device,
            max_batches=5,
        )

        assert "loss" in results
        assert "perplexity" in results
        assert "num_tokens" in results
        assert "num_batches" in results
        assert results["num_batches"] <= 5

    def test_evaluate_perplexity_with_max_batches(self, small_model, dummy_dataloader):
        """Test perplexity evaluation with max_batches limit."""
        device = "cpu"
        results = evaluate_perplexity(
            model=small_model,
            dataloader=dummy_dataloader,
            device=device,
            max_batches=3,
        )

        assert results["num_batches"] == 3


# Tests for profiler.py
class TestLatencyProfiler:
    """Tests for LatencyProfiler class."""

    def test_profile_latency(self, small_model):
        """Test latency profiling."""
        device = "cpu"
        profiler = LatencyProfiler(small_model, device)

        results = profiler.profile_latency(
            batch_size=1,
            seq_length=64,
            num_runs=10,
            warmup_runs=2,
        )

        assert "mean_ms" in results
        assert "std_ms" in results
        assert "min_ms" in results
        assert "max_ms" in results
        assert "median_ms" in results
        assert results["batch_size"] == 1
        assert results["seq_length"] == 64

    def test_profile_multiple_configs(self, small_model):
        """Test profiling multiple configurations."""
        device = "cpu"
        profiler = LatencyProfiler(small_model, device)

        configs = [(1, 64), (2, 64)]
        results = profiler.profile_multiple_configs(
            configs=configs,
            num_runs=10,
        )

        assert "batch1_seq64" in results
        assert "batch2_seq64" in results


class TestThroughputProfiler:
    """Tests for ThroughputProfiler class."""

    def test_profile_throughput(self, small_model):
        """Test throughput profiling."""
        device = "cpu"
        profiler = ThroughputProfiler(small_model, device)

        results = profiler.profile_throughput(
            batch_size=2,
            seq_length=64,
            duration_seconds=1.0,
        )

        assert "tokens_per_sec" in results
        assert "samples_per_sec" in results
        assert "batches_processed" in results
        assert "total_tokens" in results
        assert results["batch_size"] == 2
        assert results["seq_length"] == 64


class TestMemoryStats:
    """Tests for memory statistics."""

    def test_get_memory_stats_cpu(self):
        """Test getting memory stats on CPU."""
        stats = get_memory_stats("cpu")

        assert "allocated_mb" in stats
        assert "reserved_mb" in stats
        assert stats["allocated_mb"] == 0.0


# Tests for summary.py
class TestSummary:
    """Tests for summary generation."""

    def test_format_duration(self):
        """Test duration formatting."""
        assert _format_duration(30.5) == "30.5s"
        assert _format_duration(90) == "1m 30s"
        assert _format_duration(3665) == "1h 1m 5s"

    def test_generate_run_summary(self):
        """Test generating run summary."""
        metrics = StandardMetrics(
            run_id="test_run",
            experiment_name="test_exp",
            timestamp="2025-01-01T00:00:00",
            model_config={},
            total_params=100_000,
            d_model=256,
            d_ff=682,
            n_layers=4,
            n_heads=4,
            embedding_ratio=0.35,
            glu_expansion=2.66,
            train_loss=2.5,
            train_perplexity=12.18,
            best_train_loss=2.3,
            val_loss=2.6,
            val_perplexity=13.46,
            training_time_seconds=3665,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "summary.md"
            generate_run_summary(metrics, str(output_path), format="markdown")

            # Check file exists
            assert output_path.exists()

            # Verify content
            content = output_path.read_text()
            assert "test_exp" in content
            assert "test_run" in content
            assert "2.5" in content  # train loss
            assert "2.6" in content  # val loss


# Integration test
class TestIntegration:
    """Integration tests for evaluation pipeline."""

    def test_end_to_end_evaluation(self, small_model, dummy_dataloader):
        """Test end-to-end evaluation pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Evaluate perplexity
            results = evaluate_perplexity(
                model=small_model,
                dataloader=dummy_dataloader,
                device="cpu",
                max_batches=5,
            )

            # Profile latency
            profiler = LatencyProfiler(small_model, "cpu")
            latency_results = profiler.profile_latency(
                batch_size=1,
                seq_length=64,
                num_runs=10,
            )

            # Create metrics
            metrics = StandardMetrics(
                run_id="integration_test",
                experiment_name="integration",
                timestamp="2025-01-01T00:00:00",
                model_config={},
                total_params=small_model.config.total_params,
                d_model=small_model.d_model,
                d_ff=small_model.d_ff,
                n_layers=small_model.n_layers,
                n_heads=small_model.config.n_heads,
                embedding_ratio=small_model.config.embedding_ratio,
                glu_expansion=small_model.config.glu_expansion,
                train_loss=results["loss"],
                train_perplexity=results["perplexity"],
                best_train_loss=results["loss"],
                latency_batch1_seq128=latency_results["mean_ms"],
            )

            # Save metrics
            metrics_path = Path(tmpdir) / "metrics.json"
            metrics.to_json(str(metrics_path))

            # Generate summary
            summary_path = Path(tmpdir) / "summary.md"
            generate_run_summary(metrics, str(summary_path))

            # Verify both files exist
            assert metrics_path.exists()
            assert summary_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
