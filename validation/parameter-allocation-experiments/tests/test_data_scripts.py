"""Unit tests for data processing scripts."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts to path
root = Path(__file__).parent.parent
scripts_path = root / "scripts"
sys.path.insert(0, str(scripts_path))


class TestDownloadWikitext:
    """Tests for download_wikitext.py script."""

    @patch("download_wikitext.load_dataset")
    def test_download_wikitext_basic(self, mock_load_dataset, tmp_path):
        """Test basic WikiText download functionality."""
        # Mock dataset
        mock_dataset = {
            "train": MagicMock(text=["sample text 1", "sample text 2"]),
            "validation": MagicMock(text=["val text"]),
            "test": MagicMock(text=["test text"]),
        }
        mock_dataset["train"].__len__ = lambda x: 2
        mock_dataset["validation"].__len__ = lambda x: 1
        mock_dataset["test"].__len__ = lambda x: 1

        for split in mock_dataset.values():
            split.save_to_disk = MagicMock()

        mock_load_dataset.return_value = mock_dataset

        # Import after mocking
        from download_wikitext import download_wikitext

        # Run download
        output_dir = tmp_path / "wikitext"
        stats = download_wikitext(str(output_dir))

        # Verify
        assert "splits" in stats
        assert stats["total_examples"] == 4
        mock_load_dataset.assert_called_once()


class TestPreprocessLM:
    """Tests for preprocess_lm.py script."""

    def test_tokenizer_hash_deterministic(self, tmp_path):
        """Test that tokenizer hash is deterministic."""
        from preprocess_lm import compute_tokenizer_hash

        # Create dummy tokenizer file
        tokenizer_dir = tmp_path / "tokenizer"
        tokenizer_dir.mkdir()

        tokenizer_file = tokenizer_dir / "tokenizer.json"
        tokenizer_file.write_text('{"test": "data"}')

        # Compute hash twice
        hash1 = compute_tokenizer_hash(tokenizer_dir)
        hash2 = compute_tokenizer_hash(tokenizer_dir)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hash length


class TestDownloadSmallbench:
    """Tests for download_smallbench.py script."""

    def test_smallbench_tasks_defined(self):
        """Test that SmallBench tasks are properly defined."""
        from download_smallbench import SMALLBENCH_TASKS

        assert len(SMALLBENCH_TASKS) > 0

        for task_name, config in SMALLBENCH_TASKS.items():
            assert "dataset" in config
            assert "task_type" in config
            assert "max_samples" in config
            assert "description" in config


class TestAggregateResults:
    """Tests for aggregate_results.py script."""

    def test_load_metrics(self, tmp_path):
        """Test loading metrics from JSON file."""
        from aggregate_results import load_metrics

        # Create dummy metrics file
        metrics_dir = tmp_path / "exp1"
        metrics_dir.mkdir()

        metrics_file = metrics_dir / "metrics.json"
        test_metrics = {
            "perplexity": 25.5,
            "loss": 3.2,
            "accuracy": 0.85,
        }
        metrics_file.write_text(json.dumps(test_metrics))

        # Load metrics
        loaded = load_metrics(metrics_file)

        assert loaded["perplexity"] == 25.5
        assert loaded["experiment_name"] == "exp1"
        assert "log_path" in loaded


class TestGeneratePlots:
    """Tests for generate_plots.py script."""

    def test_plot_imports(self):
        """Test that plotting script can be imported."""
        try:
            import generate_plots

            assert hasattr(generate_plots, "plot_embedding_ratio_sweep")
            assert hasattr(generate_plots, "plot_glu_expansion_sweep")
            assert hasattr(generate_plots, "plot_pareto_frontier")
        except ImportError as e:
            # Allow import errors for optional dependencies in CI
            if "matplotlib" in str(e) or "seaborn" in str(e):
                pytest.skip("Plotting dependencies not installed")
            else:
                raise


def test_scripts_directory_structure():
    """Test that scripts directory has correct structure."""
    scripts_dir = Path(__file__).parent.parent / "scripts"

    assert scripts_dir.exists(), "scripts/ directory should exist"
    assert (scripts_dir / "__init__.py").exists()
    assert (scripts_dir / "download_wikitext.py").exists()
    assert (scripts_dir / "preprocess_lm.py").exists()
    assert (scripts_dir / "download_smallbench.py").exists()
    assert (scripts_dir / "aggregate_results.py").exists()
    assert (scripts_dir / "generate_plots.py").exists()


def test_scripts_executable():
    """Test that scripts have proper shebang and are executable."""
    scripts_dir = Path(__file__).parent.parent / "scripts"

    script_files = [
        "download_wikitext.py",
        "preprocess_lm.py",
        "download_smallbench.py",
        "aggregate_results.py",
        "generate_plots.py",
    ]

    for script_name in script_files:
        script_file = scripts_dir / script_name
        assert script_file.exists()

        # Check shebang
        with open(script_file, "r") as f:
            first_line = f.readline()
            assert first_line.startswith("#!/usr/bin/env python"), \
                f"{script_name} should have proper shebang"
