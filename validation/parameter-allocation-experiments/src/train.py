"""Training harness for language model experiments."""

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.lm import LanguageModel, count_parameters
from utils.checkpointing import (
    cleanup_old_checkpoints,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from utils.determinism import set_seed

logger = logging.getLogger(__name__)


class Trainer:
    """Training harness for language models.

    Features:
    - Mixed precision training (AMP)
    - Gradient clipping
    - Cosine LR scheduling with warmup
    - Checkpointing and auto-resume
    - Evaluation and metrics tracking
    - Overfit mode for sanity checking
    """

    def __init__(
        self,
        model: LanguageModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
    ):
        """Initialize trainer.

        Args:
            model: Language model to train
            train_loader: Training dataloader
            val_loader: Optional validation dataloader
            config: Training configuration dictionary
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config or {}

        # Move model to device
        self.model.to(self.device)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # Setup training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_amp()

        # Log model info
        num_params = count_parameters(self.model)
        logger.info(f"Model has {num_params:,} trainable parameters")
        logger.info(f"Training on device: {self.device}")

    def _setup_optimizer(self) -> None:
        """Setup optimizer with weight decay."""
        lr = self.config.get("learning_rate", 3e-4)
        weight_decay = self.config.get("weight_decay", 0.01)
        betas = (
            self.config.get("adam_beta1", 0.9),
            self.config.get("adam_beta2", 0.999),
        )
        eps = self.config.get("adam_epsilon", 1e-8)

        # Separate parameters with/without weight decay
        # Don't apply weight decay to LayerNorm and embeddings
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "ln" in name.lower() or "norm" in name.lower() or "embedding" in name.lower():
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = AdamW(
            optimizer_groups,
            lr=lr,
            betas=betas,
            eps=eps,
        )

        logger.info(f"Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")

    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler with warmup and cosine decay."""
        max_steps = self.config.get("max_steps", -1)
        num_epochs = self.config.get("num_train_epochs", 1)
        warmup_ratio = self.config.get("warmup_ratio", 0.05)

        # Calculate total steps
        if max_steps > 0:
            total_steps = max_steps
        else:
            total_steps = len(self.train_loader) * num_epochs

        warmup_steps = int(total_steps * warmup_ratio)
        cosine_steps = total_steps - warmup_steps

        # Create warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-10,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        # Create cosine decay scheduler
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_steps,
            eta_min=self.config.get("min_lr", 0.0),
        )

        # Combine schedulers
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        logger.info(
            f"Scheduler: Warmup ({warmup_steps} steps) + CosineAnneal ({cosine_steps} steps)"
        )

    def _setup_amp(self) -> None:
        """Setup automatic mixed precision training."""
        self.use_amp = self.config.get("fp16", False)
        self.scaler = GradScaler() if self.use_amp else None

        if self.use_amp:
            logger.info("Using automatic mixed precision (FP16)")

    def train_step(self, batch: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single training step.

        Args:
            batch: Training batch

        Returns:
            Tuple of (loss, metrics_dict)
        """
        self.model.train()

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

        # Forward pass with AMP
        self.optimizer.zero_grad()

        if self.use_amp:
            with autocast():
                logits, loss = self.model(input_ids, labels=labels)
        else:
            logits, loss = self.model(input_ids, labels=labels)

        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient clipping
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_grad_norm
        )

        # Optimizer step
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # Scheduler step
        self.scheduler.step()

        # Compute perplexity
        perplexity = torch.exp(loss).item()

        # Get current LR
        current_lr = self.scheduler.get_last_lr()[0]

        metrics = {
            "loss": loss.item(),
            "perplexity": perplexity,
            "lr": current_lr,
            "grad_norm": grad_norm.item(),
        }

        return loss, metrics

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set.

        Returns:
            Dictionary of evaluation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
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
            logits, loss = self.model(input_ids, labels=labels)

            # Accumulate loss
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_tokens += batch_size

        # Compute average metrics
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        metrics = {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
        }

        return metrics

    def train(
        self,
        num_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        eval_steps: int = 1000,
        save_steps: int = 5000,
        logging_steps: int = 100,
        checkpoint_dir: str = "checkpoints",
        resume_from_checkpoint: Optional[str] = None,
        overfit_tokens: Optional[int] = None,
    ) -> Dict[str, float]:
        """Train the model.

        Args:
            num_epochs: Number of epochs to train (default from config)
            max_steps: Maximum training steps (overrides num_epochs)
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            logging_steps: Log metrics every N steps
            checkpoint_dir: Directory to save checkpoints
            resume_from_checkpoint: Path to checkpoint to resume from
            overfit_tokens: If set, overfit on first N tokens (sanity check)

        Returns:
            Dictionary with final metrics
        """
        # Set seeds for reproducibility
        seed = self.config.get("seed", 42)
        set_seed(seed)

        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            if resume_from_checkpoint == "auto":
                resume_from_checkpoint = find_latest_checkpoint(checkpoint_dir)

            if resume_from_checkpoint:
                info = load_checkpoint(
                    resume_from_checkpoint,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.device,
                )
                self.global_step = info["step"]
                self.epoch = info["epoch"]
                self.best_loss = info["best_loss"]

        # Get training parameters
        if num_epochs is None:
            num_epochs = self.config.get("num_train_epochs", 1)

        if max_steps is None:
            max_steps = self.config.get("max_steps", -1)

        # Overfit mode (sanity check)
        if overfit_tokens:
            logger.info(f"OVERFIT MODE: Training on first {overfit_tokens} tokens only")
            # Take only first batch and repeat it
            first_batch = next(iter(self.train_loader))
            # Truncate to overfit_tokens
            if isinstance(first_batch, dict):
                first_batch["input_ids"] = first_batch["input_ids"][:, :overfit_tokens]
                if "labels" in first_batch:
                    first_batch["labels"] = first_batch["labels"][:, :overfit_tokens]
            elif isinstance(first_batch, (list, tuple)):
                first_batch = (first_batch[0][:, :overfit_tokens],)
            else:
                first_batch = first_batch[:, :overfit_tokens]

            # Create infinite iterator
            def overfit_iterator():
                while True:
                    yield first_batch

            train_iterator = overfit_iterator()
        else:
            train_iterator = None

        # Training loop
        logger.info("=" * 80)
        logger.info(f"Starting training: {num_epochs} epochs, max_steps={max_steps}")
        logger.info("=" * 80)

        start_time = time.time()
        running_loss = 0.0

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch

            # Get data iterator
            if overfit_tokens:
                data_iter = train_iterator
                pbar = tqdm(range(max_steps), desc=f"Epoch {epoch+1}/{num_epochs}")
            else:
                data_iter = iter(self.train_loader)
                pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch_idx, batch in enumerate(pbar):
                # Train step
                loss, metrics = self.train_step(batch)
                running_loss += metrics["loss"]

                self.global_step += 1

                # Logging
                if self.global_step % logging_steps == 0:
                    avg_loss = running_loss / logging_steps
                    elapsed = time.time() - start_time
                    steps_per_sec = logging_steps / elapsed

                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "ppl": f"{math.exp(avg_loss):.2f}",
                        "lr": f"{metrics['lr']:.2e}",
                        "steps/s": f"{steps_per_sec:.2f}",
                    })

                    running_loss = 0.0
                    start_time = time.time()

                # Evaluation
                if self.global_step % eval_steps == 0:
                    eval_metrics = self.evaluate()
                    if eval_metrics:
                        logger.info(
                            f"Step {self.global_step}: "
                            f"eval_loss={eval_metrics['eval_loss']:.4f}, "
                            f"eval_ppl={eval_metrics['eval_perplexity']:.2f}"
                        )

                        # Save best model
                        if eval_metrics["eval_loss"] < self.best_loss:
                            self.best_loss = eval_metrics["eval_loss"]
                            save_checkpoint(
                                self.model,
                                self.optimizer,
                                self.scheduler,
                                self.global_step,
                                epoch,
                                self.best_loss,
                                checkpoint_dir,
                                filename="checkpoint_best.pt",
                            )

                # Save checkpoint
                if self.global_step % save_steps == 0:
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.global_step,
                        epoch,
                        self.best_loss,
                        checkpoint_dir,
                    )

                    # Cleanup old checkpoints
                    cleanup_old_checkpoints(
                        checkpoint_dir,
                        keep_last_n=self.config.get("save_total_limit", 3),
                    )

                # Check max steps
                if max_steps > 0 and self.global_step >= max_steps:
                    logger.info(f"Reached max_steps={max_steps}")
                    break

            # Check max steps
            if max_steps > 0 and self.global_step >= max_steps:
                break

        # Final evaluation
        logger.info("=" * 80)
        logger.info("Training complete!")
        final_metrics = self.evaluate()
        if final_metrics:
            logger.info(f"Final metrics: {final_metrics}")

        # Save final checkpoint
        save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            self.global_step,
            self.epoch,
            self.best_loss,
            checkpoint_dir,
            filename="checkpoint_final.pt",
        )

        return final_metrics
