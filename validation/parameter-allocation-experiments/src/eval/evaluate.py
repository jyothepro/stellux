"""Evaluation utilities for language models."""

import logging
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    max_batches: Optional[int] = None,
    desc: str = "Evaluating",
) -> Dict[str, float]:
    """Evaluate perplexity on a dataset.

    Args:
        model: Language model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        max_batches: Maximum number of batches to evaluate (for quick testing)
        desc: Description for progress bar

    Returns:
        Dictionary with metrics:
        - loss: Average cross-entropy loss
        - perplexity: exp(loss)
        - num_tokens: Total tokens evaluated
        - num_batches: Number of batches processed
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    # Create progress bar
    pbar = tqdm(dataloader, desc=desc, leave=False)

    for batch_idx, batch in enumerate(pbar):
        # Check if we've reached max_batches
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Prepare inputs
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            labels = batch.get("labels", input_ids).to(device)
        elif isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(device)
            labels = batch[1].to(device) if len(batch) > 1 else input_ids
        else:
            input_ids = batch.to(device)
            labels = input_ids

        # Forward pass
        try:
            logits, loss = model(input_ids, labels=labels)
        except Exception as e:
            logger.warning(f"Error in forward pass for batch {batch_idx}: {e}")
            continue

        # Accumulate loss
        # The model returns sum of losses with reduction='sum'
        # We need to count tokens the same way: shifted labels, excluding ignore_index
        if labels is not None:
            # Count tokens in shifted labels (model predicts token t+1 from tokens 0:t)
            # So we lose 1 token per sequence due to shifting
            shift_labels = labels[:, 1:]  # Same shift as in model

            # Count non-padding tokens (ignore_index=-100 in cross_entropy)
            # Also ignore 0 if it's used as padding in the data
            non_pad_tokens = ((shift_labels != -100) & (shift_labels != 0)).sum().item()
        else:
            # If no labels, count all tokens minus 1 per sequence (due to shift)
            non_pad_tokens = input_ids[:, 1:].numel()

        total_loss += loss.item()  # loss is already a sum, don't multiply
        total_tokens += non_pad_tokens
        num_batches += 1

        # Update progress bar
        if num_batches % 10 == 0:
            avg_loss = total_loss / total_tokens
            current_ppl = math.exp(min(avg_loss, 20))  # Cap at 20 to avoid overflow
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "ppl": f"{current_ppl:.2f}",
            })

    # Compute final metrics
    if total_tokens == 0:
        logger.warning("No tokens evaluated!")
        return {
            "loss": float("inf"),
            "perplexity": float("inf"),
            "num_tokens": 0,
            "num_batches": 0,
        }

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow

    metrics = {
        "loss": avg_loss,
        "perplexity": perplexity,
        "num_tokens": total_tokens,
        "num_batches": num_batches,
    }

    logger.info(
        f"Evaluation complete: loss={avg_loss:.4f}, ppl={perplexity:.2f}, "
        f"tokens={total_tokens:,}, batches={num_batches}"
    )

    return metrics


@torch.no_grad()
def evaluate_splits(
    model: nn.Module,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model on train/val/test splits.

    Args:
        model: Language model to evaluate
        train_loader: Optional training dataloader
        val_loader: Optional validation dataloader
        test_loader: Optional test dataloader
        device: Device to run evaluation on
        max_batches: Maximum batches per split (for quick testing)

    Returns:
        Dictionary with metrics for each split:
        {
            "train": {...},
            "val": {...},
            "test": {...}
        }
    """
    results = {}

    if train_loader is not None:
        logger.info("Evaluating on training set...")
        results["train"] = evaluate_perplexity(
            model, train_loader, device, max_batches, desc="Train Eval"
        )

    if val_loader is not None:
        logger.info("Evaluating on validation set...")
        results["val"] = evaluate_perplexity(
            model, val_loader, device, max_batches, desc="Val Eval"
        )

    if test_loader is not None:
        logger.info("Evaluating on test set...")
        results["test"] = evaluate_perplexity(
            model, test_loader, device, max_batches, desc="Test Eval"
        )

    return results


def compute_token_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    top_k: int = 1,
) -> float:
    """Compute top-k token prediction accuracy.

    Args:
        model: Language model
        dataloader: DataLoader for evaluation
        device: Device to run on
        top_k: Consider top-k predictions

    Returns:
        Accuracy as float in [0, 1]
    """
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc=f"Computing top-{top_k} accuracy", leave=False):
        # Prepare inputs
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            labels = batch.get("labels", input_ids).to(device)
        else:
            input_ids = batch.to(device)
            labels = input_ids

        # Forward pass
        logits, _ = model(input_ids, labels=labels)

        # Get predictions (shift by 1 for causal LM)
        # logits: [batch, seq_len, vocab]
        # We predict token t+1 from tokens 0:t
        pred_logits = logits[:, :-1, :]  # [batch, seq_len-1, vocab]
        target_ids = labels[:, 1:]  # [batch, seq_len-1]

        # Get top-k predictions
        _, top_k_preds = pred_logits.topk(top_k, dim=-1)  # [batch, seq_len-1, k]

        # Check if target is in top-k
        target_expanded = target_ids.unsqueeze(-1)  # [batch, seq_len-1, 1]
        matches = (top_k_preds == target_expanded).any(dim=-1)  # [batch, seq_len-1]

        # Count non-padding tokens
        non_pad_mask = target_ids != 0
        correct += (matches & non_pad_mask).sum().item()
        total += non_pad_mask.sum().item()

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Top-{top_k} accuracy: {accuracy:.4f} ({correct}/{total})")

    return accuracy
