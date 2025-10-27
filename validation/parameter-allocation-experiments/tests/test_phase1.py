"""Tests for Phase 1 ranking runs and early stopping (Milestone 5)."""

import json
import tempfile
from pathlib import Path

import pytest

# Import Phase 1 modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.early_stopping import EarlyStoppingKillRule, BaselineTracker


# Tests for EarlyStoppingKillRule
class TestEarlyStoppingKillRule:
    """Tests for EarlyStoppingKillRule class."""

    def test_initialization(self):
        """Test early stopping initialization."""
        es = EarlyStoppingKillRule(
            patience=5,
            threshold=0.5,
            min_tokens=1_000_000,
            mode="min",
        )

        assert es.patience == 5
        assert es.threshold == 0.5
        assert es.min_tokens == 1_000_000
        assert es.mode == "min"
        assert es.best_value == float("inf")
        assert es.should_stop == False

    def test_update_before_min_tokens(self):
        """Test that early stopping waits for min_tokens."""
        es = EarlyStoppingKillRule(
            patience=3,
            threshold=0.5,
            min_tokens=1_000_000,
            mode="min",
        )

        # Should not stop before min_tokens
        should_stop = es.update(
            current_value=5.0,
            current_step=100,
            tokens_processed=500_000,  # Less than 1M
        )

        assert should_stop == False
        assert es.should_stop == False

    def test_kill_rule_trigger(self):
        """Test that kill rule triggers when PPL is threshold worse."""
        es = EarlyStoppingKillRule(
            patience=5,
            threshold=0.5,
            min_tokens=1_000_000,
            mode="min",
            baseline=3.0,  # Baseline PPL
        )

        # Reach min_tokens
        es.update(3.2, 1000, tokens_processed=1_000_000)

        # Current PPL is 0.6 worse than baseline (>= 0.5 threshold)
        should_stop = es.update(3.6, 1100, tokens_processed=1_100_000)

        assert should_stop == True
        assert es.should_stop == True

    def test_no_trigger_within_threshold(self):
        """Test that kill rule doesn't trigger within threshold."""
        es = EarlyStoppingKillRule(
            patience=5,
            threshold=0.5,
            min_tokens=1_000_000,
            mode="min",
            baseline=3.0,
        )

        # Reach min_tokens
        es.update(3.2, 1000, tokens_processed=1_000_000)

        # Current PPL is 0.3 worse than baseline (< 0.5 threshold)
        should_stop = es.update(3.3, 1100, tokens_processed=1_100_000)

        assert should_stop == False
        assert es.should_stop == False

    def test_improvement_resets_patience(self):
        """Test that improvement resets patience counter."""
        es = EarlyStoppingKillRule(
            patience=3,
            threshold=0.5,
            min_tokens=1_000_000,
            mode="min",
        )

        # Reach min_tokens
        es.update(5.0, 1000, tokens_processed=1_000_000)
        assert es.wait_count == 0

        # No improvement
        es.update(5.1, 1100, tokens_processed=1_100_000)
        assert es.wait_count == 1

        es.update(5.2, 1200, tokens_processed=1_200_000)
        assert es.wait_count == 2

        # Improvement - should reset
        es.update(4.8, 1300, tokens_processed=1_300_000)
        assert es.wait_count == 0
        assert es.best_value == 4.8

    def test_patience_exhausted(self):
        """Test that patience-based stopping works."""
        es = EarlyStoppingKillRule(
            patience=3,
            threshold=0.5,
            min_tokens=1_000_000,
            mode="min",
        )

        # Reach min_tokens with initial value
        es.update(5.0, 1000, tokens_processed=1_000_000)

        # No improvement for patience steps
        es.update(5.1, 1100, tokens_processed=1_100_000)
        assert es.should_stop == False

        es.update(5.1, 1200, tokens_processed=1_200_000)
        assert es.should_stop == False

        # Patience exhausted
        should_stop = es.update(5.1, 1300, tokens_processed=1_300_000)
        assert should_stop == True
        assert es.should_stop == True

    def test_get_state(self):
        """Test state serialization."""
        es = EarlyStoppingKillRule(
            patience=5,
            threshold=0.5,
            min_tokens=1_000_000,
        )

        es.update(5.0, 1000, tokens_processed=1_000_000)

        state = es.get_state()

        assert "best_value" in state
        assert "best_step" in state
        assert "wait_count" in state
        assert "should_stop" in state
        assert "tokens_seen" in state
        assert "history" in state

        assert state["best_value"] == 5.0
        assert state["best_step"] == 1000
        assert state["tokens_seen"] == 1_000_000

    def test_load_state(self):
        """Test state deserialization."""
        es = EarlyStoppingKillRule()

        state = {
            "best_value": 4.5,
            "best_step": 2000,
            "wait_count": 2,
            "should_stop": False,
            "tokens_seen": 2_000_000,
            "history": [
                {"value": 5.0, "step": 1000, "tokens": 1_000_000},
                {"value": 4.5, "step": 2000, "tokens": 2_000_000},
            ],
        }

        es.load_state(state)

        assert es.best_value == 4.5
        assert es.best_step == 2000
        assert es.wait_count == 2
        assert es.tokens_seen == 2_000_000
        assert len(es.history) == 2

    def test_get_summary(self):
        """Test summary generation."""
        es = EarlyStoppingKillRule(
            patience=5,
            threshold=0.5,
            min_tokens=1_000_000,
        )

        es.update(5.0, 1000, tokens_processed=1_000_000)
        es.update(4.8, 1100, tokens_processed=1_100_000)

        summary = es.get_summary()

        assert summary["best_value"] == 4.8
        assert summary["best_step"] == 1100
        assert summary["total_evals"] == 2
        assert summary["tokens_seen"] == 1_100_000
        assert summary["kill_rule_threshold"] == 0.5
        assert summary["kill_rule_min_tokens"] == 1_000_000


# Tests for BaselineTracker
class TestBaselineTracker:
    """Tests for BaselineTracker class."""

    def test_initialization(self):
        """Test baseline tracker initialization."""
        tracker = BaselineTracker()

        assert tracker.baseline_ppl is None
        assert tracker.baseline_loss is None
        assert tracker.baseline_name is None

    def test_set_baseline(self):
        """Test setting baseline."""
        tracker = BaselineTracker()

        tracker.set_baseline(ppl=3.5, loss=1.25, name="baseline_model")

        assert tracker.baseline_ppl == 3.5
        assert tracker.baseline_loss == 1.25
        assert tracker.baseline_name == "baseline_model"

    def test_compare_no_baseline(self):
        """Test comparison without baseline set."""
        tracker = BaselineTracker()

        result = tracker.compare(current_ppl=4.0, threshold=0.5)

        assert result["has_baseline"] == False
        assert result["should_kill"] == False

    def test_compare_within_threshold(self):
        """Test comparison within threshold."""
        tracker = BaselineTracker()
        tracker.set_baseline(ppl=3.5)

        result = tracker.compare(current_ppl=3.8, threshold=0.5)

        assert result["has_baseline"] == True
        assert result["baseline_ppl"] == 3.5
        assert result["current_ppl"] == 3.8
        assert result["diff"] == 0.3
        assert result["should_kill"] == False

    def test_compare_exceeds_threshold(self):
        """Test comparison exceeding threshold."""
        tracker = BaselineTracker()
        tracker.set_baseline(ppl=3.5)

        result = tracker.compare(current_ppl=4.1, threshold=0.5)

        assert result["has_baseline"] == True
        assert result["diff"] == 0.6
        assert result["should_kill"] == True


# Integration tests
class TestPhase1Integration:
    """Integration tests for Phase 1 components."""

    def test_early_stopping_with_baseline(self):
        """Test early stopping with baseline comparison."""
        # Setup baseline
        baseline_tracker = BaselineTracker()
        baseline_tracker.set_baseline(ppl=3.0, name="baseline")

        # Setup early stopping
        es = EarlyStoppingKillRule(
            patience=5,
            threshold=0.5,
            min_tokens=1_000_000,
            mode="min",
            baseline=baseline_tracker.baseline_ppl,
        )

        # Simulate training
        should_stop = False

        # Step 1: 1M tokens, PPL slightly worse
        should_stop = es.update(3.3, 1000, tokens_processed=1_000_000)
        assert should_stop == False

        # Step 2: 1.1M tokens, PPL getting worse
        should_stop = es.update(3.4, 1100, tokens_processed=1_100_000)
        assert should_stop == False

        # Step 3: 1.2M tokens, PPL exceeds threshold
        should_stop = es.update(3.6, 1200, tokens_processed=1_200_000)
        assert should_stop == True

        # Verify it was the kill rule, not patience
        assert es.wait_count < es.patience

    def test_config_loading(self):
        """Test Phase 1 config file structure."""
        # This tests that config files have correct structure
        config_dir = Path(__file__).parent.parent / "configs"

        phase1_embedding = config_dir / "phase1_embedding_sweep.yaml"
        phase1_glu = config_dir / "phase1_glu_sweep.yaml"

        # Check files exist
        assert phase1_embedding.exists(), "Phase 1 embedding config not found"
        assert phase1_glu.exists(), "Phase 1 GLU config not found"

        # Basic structure validation would go here
        # (Would need PyYAML to parse, so we just check existence)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
