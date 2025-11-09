"""Data loading utilities."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""

    def __init__(self, data, max_length: int = 512):
        """Initialize dataset.

        Args:
            data: HuggingFace dataset or list of examples
            max_length: Maximum sequence length
        """
        self.data = data
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self.data[idx]

        # Get input_ids
        if isinstance(item, dict):
            input_ids = item["input_ids"]
        else:
            input_ids = item

        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]

        return torch.tensor(input_ids, dtype=torch.long)


def collate_fn(batch):
    """Collate function for DataLoader.

    Pads sequences to the same length within a batch.
    """
    # Find max length in batch
    max_len = max(len(x) for x in batch)

    # Pad sequences
    padded = []
    for seq in batch:
        if len(seq) < max_len:
            padding = torch.zeros(max_len - len(seq), dtype=torch.long)
            seq = torch.cat([seq, padding])
        padded.append(seq)

    # Stack into tensor
    input_ids = torch.stack(padded)

    return {"input_ids": input_ids, "labels": input_ids}


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """Create train and validation dataloaders.

    Args:
        data_dir: Directory containing tokenized data
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        seed: Random seed for shuffling

    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_path = Path(data_dir)

    # Check if tokenized data exists
    tokenized_path = data_path / "tokenized_data"
    if not tokenized_path.exists():
        logger.warning(f"Tokenized data not found at {tokenized_path}")
        return None, None

    try:
        # Load tokenized dataset
        dataset = load_from_disk(str(tokenized_path))

        # Create datasets
        train_dataset = TextDataset(dataset["train"], max_length=max_length)
        val_dataset = (
            TextDataset(dataset["validation"], max_length=max_length)
            if "validation" in dataset
            else None
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )

        logger.info(f"Train dataset: {len(train_dataset)} examples")
        if val_dataset:
            logger.info(f"Val dataset: {len(val_dataset)} examples")

        return train_loader, val_loader

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None, None
