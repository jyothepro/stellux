"""Early stopping utilities for parameter allocation experiments.

Implements the kill rule: stop training if dev PPL is ≥0.5 worse than
baseline for 1M tokens.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EarlyStoppingKillRule:
    """Early stopping with kill rule for ranking runs.

    Tracks validation perplexity and stops training if it's consistently
    worse than a baseline or best value.

    Kill rule: Stop if dev PPL ≥threshold worse for min_tokens tokens.
    """

    def __init__(
        self,
        patience: int = 5,
        threshold: float = 0.5,
        min_tokens: int = 1_000_000,
        mode: str = "min",
        baseline: Optional[float] = None,
    ):
        """Initialize early stopping.

        Args:
            patience: Number of evaluations to wait before stopping
            threshold: PPL threshold for early stopping (e.g., 0.5)
            min_tokens: Minimum tokens before checking (e.g., 1M)
            mode: "min" or "max" for metric comparison
            baseline: Optional baseline value to compare against
        """
        self.patience = patience
        self.threshold = threshold
        self.min_tokens = min_tokens
        self.mode = mode
        self.baseline = baseline

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_step = 0
        self.wait_count = 0
        self.stopped_step = 0
        self.should_stop = False

        # Track history for kill rule
        self.history: List[Dict[str, float]] = []
        self.tokens_seen = 0

    def update(
        self,
        current_value: float,
        current_step: int,
        tokens_processed: Optional[int] = None,
    ) -> bool:
        """Update early stopping state.

        Args:
            current_value: Current metric value (e.g., validation PPL)
            current_step: Current training step
            tokens_processed: Total tokens processed so far

        Returns:
            True if training should stop, False otherwise
        """
        if tokens_processed is not None:
            self.tokens_seen = tokens_processed

        # Add to history
        self.history.append({
            "value": current_value,
            "step": current_step,
            "tokens": self.tokens_seen,
        })

        # Check if we should start monitoring
        if self.tokens_seen < self.min_tokens:
            logger.debug(
                f"Early stopping: Waiting for {self.min_tokens:,} tokens "
                f"(current: {self.tokens_seen:,})"
            )
            return False

        # Determine reference value (baseline or best)
        reference_value = self.baseline if self.baseline is not None else self.best_value

        # Check if current value is better than best
        if self.mode == "min":
            is_better = current_value < self.best_value
            is_worse_than_ref = current_value >= (reference_value + self.threshold)
        else:
            is_better = current_value > self.best_value
            is_worse_than_ref = current_value <= (reference_value - self.threshold)

        # Update best value
        if is_better:
            self.best_value = current_value
            self.best_step = current_step
            self.wait_count = 0
            logger.info(
                f"Early stopping: New best {self.mode} value: {self.best_value:.4f} "
                f"at step {current_step}"
            )
        else:
            self.wait_count += 1
            logger.debug(
                f"Early stopping: No improvement for {self.wait_count}/{self.patience} evals"
            )

        # Check kill rule: current value is threshold worse than reference
        if is_worse_than_ref:
            logger.warning(
                f"Kill rule triggered: Current PPL {current_value:.4f} is "
                f">={self.threshold} worse than reference {reference_value:.4f}"
            )
            self.should_stop = True
            self.stopped_step = current_step
            return True

        # Check patience-based stopping
        if self.wait_count >= self.patience:
            logger.info(
                f"Early stopping: Patience exhausted ({self.patience} evals without improvement)"
            )
            self.should_stop = True
            self.stopped_step = current_step
            return True

        return False

    def get_state(self) -> Dict:
        """Get current state for checkpointing."""
        return {
            "best_value": self.best_value,
            "best_step": self.best_step,
            "wait_count": self.wait_count,
            "stopped_step": self.stopped_step,
            "should_stop": self.should_stop,
            "tokens_seen": self.tokens_seen,
            "history": self.history,
        }

    def load_state(self, state: Dict) -> None:
        """Load state from checkpoint."""
        self.best_value = state.get("best_value", self.best_value)
        self.best_step = state.get("best_step", 0)
        self.wait_count = state.get("wait_count", 0)
        self.stopped_step = state.get("stopped_step", 0)
        self.should_stop = state.get("should_stop", False)
        self.tokens_seen = state.get("tokens_seen", 0)
        self.history = state.get("history", [])

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "best_value": self.best_value,
            "best_step": self.best_step,
            "stopped_early": self.should_stop,
            "stopped_step": self.stopped_step,
            "wait_count": self.wait_count,
            "total_evals": len(self.history),
            "tokens_seen": self.tokens_seen,
            "kill_rule_threshold": self.threshold,
            "kill_rule_min_tokens": self.min_tokens,
        }


class BaselineTracker:
    """Track baseline performance for comparison in kill rule.

    Useful for comparing multiple runs against a baseline configuration.
    """

    def __init__(self):
        """Initialize baseline tracker."""
        self.baseline_ppl: Optional[float] = None
        self.baseline_loss: Optional[float] = None
        self.baseline_name: Optional[str] = None

    def set_baseline(
        self,
        ppl: float,
        loss: Optional[float] = None,
        name: str = "baseline",
    ) -> None:
        """Set baseline values.

        Args:
            ppl: Baseline perplexity
            loss: Baseline loss
            name: Baseline run name
        """
        self.baseline_ppl = ppl
        self.baseline_loss = loss
        self.baseline_name = name
        logger.info(
            f"Baseline set: {name} with PPL={ppl:.4f}" +
            (f", loss={loss:.4f}" if loss else "")
        )

    def load_baseline_from_checkpoint(self, checkpoint_path: str) -> None:
        """Load baseline from checkpoint file.

        Args:
            checkpoint_path: Path to baseline checkpoint
        """
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "best_loss" in checkpoint:
            self.baseline_loss = checkpoint["best_loss"]

        if "val_perplexity" in checkpoint:
            self.baseline_ppl = checkpoint["val_perplexity"]
        elif "val_loss" in checkpoint:
            import math
            self.baseline_ppl = math.exp(checkpoint["val_loss"])

        if "run_name" in checkpoint:
            self.baseline_name = checkpoint["run_name"]

        logger.info(
            f"Loaded baseline from {checkpoint_path}: "
            f"PPL={self.baseline_ppl:.4f if self.baseline_ppl else 'N/A'}"
        )

    def compare(
        self,
        current_ppl: float,
        threshold: float = 0.5,
    ) -> Dict[str, any]:
        """Compare current perplexity to baseline.

        Args:
            current_ppl: Current perplexity
            threshold: Threshold for kill rule

        Returns:
            Dictionary with comparison results
        """
        if self.baseline_ppl is None:
            return {
                "has_baseline": False,
                "should_kill": False,
                "message": "No baseline set",
            }

        diff = current_ppl - self.baseline_ppl
        should_kill = diff >= threshold

        return {
            "has_baseline": True,
            "baseline_ppl": self.baseline_ppl,
            "current_ppl": current_ppl,
            "diff": diff,
            "threshold": threshold,
            "should_kill": should_kill,
            "message": (
                f"Current PPL {current_ppl:.4f} is {diff:+.4f} vs "
                f"baseline {self.baseline_ppl:.4f} "
                f"({'WORSE' if should_kill else 'OK'})"
            ),
        }
