"""Learning rate range finder utility.

Implements the LR range test from "Cyclical Learning Rates for Training Neural Networks"
(Smith, 2017) to find a good initial learning rate.
"""

import logging
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LRFinder:
    """Learning rate range finder.

    Gradually increases LR from a very small value to a large value and
    records the loss at each step. Helps identify optimal LR range.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
    ):
        """Initialize LR finder.

        Args:
            model: Model to train
            optimizer: Optimizer to use
            criterion: Loss function
            device: Device to use
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.lrs: List[float] = []
        self.losses: List[float] = []

    def range_test(
        self,
        train_loader: DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_iter: int = 100,
        smooth_f: float = 0.05,
        diverge_th: float = 5.0,
    ) -> Tuple[List[float], List[float]]:
        """Run LR range test.

        Args:
            train_loader: Training dataloader
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations to test
            smooth_f: Smoothing factor for loss
            diverge_th: Stop if loss > diverge_th * best_loss

        Returns:
            Tuple of (learning_rates, losses)
        """
        # Save initial model state
        model_state = {
            k: v.cpu().clone() for k, v in self.model.state_dict().items()
        }
        optim_state = self.optimizer.state_dict()

        # Set model to training mode
        self.model.train()

        # Initialize
        self.lrs = []
        self.losses = []
        best_loss = float("inf")
        smoothed_loss = 0.0

        # Calculate LR multiplier
        mult = (end_lr / start_lr) ** (1 / num_iter)
        lr = start_lr
        self.optimizer.param_groups[0]["lr"] = lr

        # Training loop
        data_iter = iter(train_loader)
        pbar = tqdm(range(num_iter), desc="LR Finder")

        for i in pbar:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Prepare inputs
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
            elif isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(self.device)
                labels = batch[1].to(self.device) if len(batch) > 1 else input_ids
            else:
                input_ids = batch.to(self.device)
                labels = input_ids

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, labels=labels)

            if isinstance(outputs, tuple):
                loss = outputs[1]  # (logits, loss)
            else:
                loss = outputs

            # Check if loss diverged
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Loss diverged at lr={lr:.2e}")
                break

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update smoothed loss
            smoothed_loss = smooth_f * loss.item() + (1 - smooth_f) * smoothed_loss

            # Record
            self.lrs.append(lr)
            self.losses.append(smoothed_loss if i > 0 else loss.item())

            # Update best loss
            if smoothed_loss < best_loss or i == 0:
                best_loss = smoothed_loss

            # Check for divergence
            if smoothed_loss > diverge_th * best_loss:
                logger.info(f"Stopping: loss diverged at lr={lr:.2e}")
                break

            # Update progress bar
            pbar.set_postfix({"lr": f"{lr:.2e}", "loss": f"{smoothed_loss:.4f}"})

            # Increase learning rate
            lr *= mult
            self.optimizer.param_groups[0]["lr"] = lr

        # Restore original model state
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optim_state)

        logger.info(f"LR range test complete. Tested {len(self.lrs)} learning rates")

        return self.lrs, self.losses

    def plot(self, output_path: str = "lr_finder.png", skip_start: int = 10, skip_end: int = 5):
        """Plot LR vs loss curve.

        Args:
            output_path: Path to save plot
            skip_start: Skip first N points
            skip_end: Skip last N points
        """
        if not self.lrs or not self.losses:
            raise ValueError("No data to plot. Run range_test() first.")

        # Skip points
        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(lrs, losses)
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate (log scale)")
        ax.set_ylabel("Loss")
        ax.set_title("Learning Rate Range Test")
        ax.grid(True, alpha=0.3)

        # Find suggested LR (steepest gradient)
        if len(losses) > 10:
            grads = np.gradient(np.array(losses))
            min_grad_idx = np.argmin(grads)
            suggested_lr = lrs[min_grad_idx]
            ax.axvline(suggested_lr, color="red", linestyle="--", alpha=0.7,
                       label=f"Suggested LR: {suggested_lr:.2e}")
            ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"LR finder plot saved to {output_path}")

        plt.close()

    def suggest_lr(self, skip_start: int = 10, skip_end: int = 5) -> float:
        """Suggest a good learning rate based on the steepest gradient.

        Args:
            skip_start: Skip first N points
            skip_end: Skip last N points

        Returns:
            Suggested learning rate
        """
        if not self.lrs or not self.losses:
            raise ValueError("No data available. Run range_test() first.")

        # Skip points
        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]

        # Find steepest gradient
        grads = np.gradient(np.array(losses))
        min_grad_idx = np.argmin(grads)
        suggested_lr = lrs[min_grad_idx]

        logger.info(f"Suggested learning rate: {suggested_lr:.2e}")

        return suggested_lr
