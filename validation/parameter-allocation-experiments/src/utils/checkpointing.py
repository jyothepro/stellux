"""Utilities for model checkpointing and resuming training."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    step: int,
    epoch: int,
    best_loss: float,
    checkpoint_dir: str,
    filename: Optional[str] = None,
) -> str:
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state
        step: Current training step
        epoch: Current epoch
        best_loss: Best validation loss so far
        checkpoint_dir: Directory to save checkpoint
        filename: Optional custom filename

    Returns:
        Path to saved checkpoint
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_step_{step}.pt"

    checkpoint_file = checkpoint_path / filename

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "epoch": epoch,
        "best_loss": best_loss,
    }

    # Save checkpoint
    torch.save(checkpoint, checkpoint_file)
    logger.info(f"Checkpoint saved to {checkpoint_file}")

    # Also save as "latest"
    latest_file = checkpoint_path / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_file)

    return str(checkpoint_file)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint on

    Returns:
        Dictionary with checkpoint info (step, epoch, best_loss)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    info = {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "best_loss": checkpoint.get("best_loss", float("inf")),
    }

    logger.info(f"Checkpoint loaded: step={info['step']}, epoch={info['epoch']}, "
                f"best_loss={info['best_loss']:.4f}")

    return info


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint or None if not found
    """
    checkpoint_path = Path(checkpoint_dir)

    # Check for "latest" checkpoint
    latest_file = checkpoint_path / "checkpoint_latest.pt"
    if latest_file.exists():
        return str(latest_file)

    # Find most recent checkpoint by step number
    checkpoints = list(checkpoint_path.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        return None

    # Extract step numbers and find max
    def get_step(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-1])
        except (ValueError, IndexError):
            return -1

    latest = max(checkpoints, key=get_step)
    return str(latest)


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_last_n: int = 3,
    keep_best: bool = True,
) -> None:
    """Remove old checkpoints, keeping only the most recent ones.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to keep the best checkpoint
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoints = list(checkpoint_path.glob("checkpoint_step_*.pt"))

    if len(checkpoints) <= keep_last_n:
        return

    # Sort by step number
    def get_step(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-1])
        except (ValueError, IndexError):
            return -1

    checkpoints_sorted = sorted(checkpoints, key=get_step, reverse=True)

    # Keep most recent
    to_keep = set(checkpoints_sorted[:keep_last_n])

    # Keep best if requested
    if keep_best and (checkpoint_path / "checkpoint_best.pt").exists():
        to_keep.add(checkpoint_path / "checkpoint_best.pt")

    # Remove old checkpoints
    for ckpt in checkpoints:
        if ckpt not in to_keep and ckpt.name != "checkpoint_latest.pt":
            try:
                ckpt.unlink()
                logger.debug(f"Removed old checkpoint: {ckpt}")
            except OSError as e:
                logger.warning(f"Failed to remove checkpoint {ckpt}: {e}")
