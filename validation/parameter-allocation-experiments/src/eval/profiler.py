"""Performance profiling utilities for language models."""

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LatencyProfiler:
    """Profile model inference latency.

    Measures time for forward passes with different sequence lengths
    and batch sizes. Useful for understanding model efficiency.
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        """Initialize profiler.

        Args:
            model: Model to profile
            device: Device to run profiling on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def profile_latency(
        self,
        batch_size: int = 1,
        seq_length: int = 128,
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """Profile inference latency for given batch size and sequence length.

        Args:
            batch_size: Batch size to test
            seq_length: Sequence length to test
            num_runs: Number of runs for averaging
            warmup_runs: Number of warmup runs (not counted)

        Returns:
            Dictionary with timing statistics:
            - mean_ms: Mean latency in milliseconds
            - std_ms: Standard deviation in milliseconds
            - min_ms: Minimum latency
            - max_ms: Maximum latency
            - median_ms: Median latency
            - p95_ms: 95th percentile latency
            - p99_ms: 99th percentile latency
        """
        # Create dummy input
        input_ids = torch.randint(
            0, self.model.config.vocab_size, (batch_size, seq_length),
            device=self.device
        )

        # Warmup runs
        logger.info(f"Running {warmup_runs} warmup iterations...")
        for _ in range(warmup_runs):
            _ = self.model(input_ids)
            if self.device == "cuda":
                torch.cuda.synchronize()

        # Timed runs
        logger.info(f"Running {num_runs} timed iterations...")
        latencies = []

        for _ in range(num_runs):
            if self.device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = self.model(input_ids)

            if self.device == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

        # Compute statistics
        latencies_tensor = torch.tensor(latencies)
        mean_ms = latencies_tensor.mean().item()
        std_ms = latencies_tensor.std().item()
        min_ms = latencies_tensor.min().item()
        max_ms = latencies_tensor.max().item()
        median_ms = latencies_tensor.median().item()

        # Percentiles
        sorted_latencies = latencies_tensor.sort()[0]
        p95_idx = int(0.95 * len(sorted_latencies))
        p99_idx = int(0.99 * len(sorted_latencies))
        p95_ms = sorted_latencies[p95_idx].item()
        p99_ms = sorted_latencies[p99_idx].item()

        results = {
            "batch_size": batch_size,
            "seq_length": seq_length,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "median_ms": median_ms,
            "p95_ms": p95_ms,
            "p99_ms": p99_ms,
            "num_runs": num_runs,
        }

        logger.info(
            f"Latency (batch={batch_size}, seq={seq_length}): "
            f"mean={mean_ms:.2f}ms, std={std_ms:.2f}ms, "
            f"p95={p95_ms:.2f}ms, p99={p99_ms:.2f}ms"
        )

        return results

    def profile_multiple_configs(
        self,
        configs: Optional[List[Tuple[int, int]]] = None,
        num_runs: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        """Profile latency for multiple (batch_size, seq_length) configurations.

        Args:
            configs: List of (batch_size, seq_length) tuples.
                    If None, uses standard configs from PRD.
            num_runs: Number of runs per configuration

        Returns:
            Dictionary mapping config name to results
        """
        if configs is None:
            # Standard configs from PRD: batch=1, seq=128/512
            configs = [(1, 128), (1, 512)]

        results = {}

        for batch_size, seq_length in configs:
            config_name = f"batch{batch_size}_seq{seq_length}"
            logger.info(f"\nProfiling {config_name}...")

            results[config_name] = self.profile_latency(
                batch_size=batch_size,
                seq_length=seq_length,
                num_runs=num_runs,
            )

        return results


class ThroughputProfiler:
    """Profile model throughput (tokens/sec).

    Measures how many tokens can be processed per second
    with different batch sizes.
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        """Initialize profiler.

        Args:
            model: Model to profile
            device: Device to run profiling on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def profile_throughput(
        self,
        batch_size: int = 32,
        seq_length: int = 512,
        duration_seconds: float = 10.0,
    ) -> Dict[str, float]:
        """Profile throughput for given batch size and sequence length.

        Args:
            batch_size: Batch size to test
            seq_length: Sequence length to test
            duration_seconds: How long to run the test

        Returns:
            Dictionary with throughput metrics:
            - tokens_per_sec: Tokens processed per second
            - samples_per_sec: Samples processed per second
            - batches_processed: Number of batches processed
            - total_tokens: Total tokens processed
            - elapsed_seconds: Actual elapsed time
        """
        # Create dummy input
        input_ids = torch.randint(
            0, self.model.config.vocab_size, (batch_size, seq_length),
            device=self.device
        )

        # Warmup
        for _ in range(10):
            _ = self.model(input_ids)
            if self.device == "cuda":
                torch.cuda.synchronize()

        # Timed run
        logger.info(f"Running throughput test for {duration_seconds}s...")

        batches_processed = 0
        start_time = time.perf_counter()

        while True:
            _ = self.model(input_ids)
            if self.device == "cuda":
                torch.cuda.synchronize()

            batches_processed += 1

            elapsed = time.perf_counter() - start_time
            if elapsed >= duration_seconds:
                break

        # Compute metrics
        total_tokens = batches_processed * batch_size * seq_length
        tokens_per_sec = total_tokens / elapsed
        samples_per_sec = (batches_processed * batch_size) / elapsed

        results = {
            "batch_size": batch_size,
            "seq_length": seq_length,
            "tokens_per_sec": tokens_per_sec,
            "samples_per_sec": samples_per_sec,
            "batches_processed": batches_processed,
            "total_tokens": total_tokens,
            "elapsed_seconds": elapsed,
        }

        logger.info(
            f"Throughput (batch={batch_size}, seq={seq_length}): "
            f"{tokens_per_sec:.0f} tokens/sec, "
            f"{samples_per_sec:.2f} samples/sec"
        )

        return results

    def profile_batch_scaling(
        self,
        batch_sizes: Optional[List[int]] = None,
        seq_length: int = 512,
        duration_seconds: float = 5.0,
    ) -> Dict[int, Dict[str, float]]:
        """Profile throughput scaling with batch size.

        Args:
            batch_sizes: List of batch sizes to test
            seq_length: Sequence length
            duration_seconds: Duration per test

        Returns:
            Dictionary mapping batch size to results
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]

        results = {}

        for batch_size in batch_sizes:
            logger.info(f"\nProfiling batch_size={batch_size}...")

            try:
                results[batch_size] = self.profile_throughput(
                    batch_size=batch_size,
                    seq_length=seq_length,
                    duration_seconds=duration_seconds,
                )
            except RuntimeError as e:
                logger.warning(f"Failed at batch_size={batch_size}: {e}")
                break

        return results


def get_memory_stats(device: str = "cuda") -> Dict[str, float]:
    """Get current GPU memory statistics.

    Args:
        device: Device to check

    Returns:
        Dictionary with memory statistics in MB:
        - allocated_mb: Currently allocated memory
        - reserved_mb: Reserved memory
        - max_allocated_mb: Peak allocated memory
        - max_reserved_mb: Peak reserved memory
    """
    if device == "cpu":
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
            "max_reserved_mb": 0.0,
        }

    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
    max_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024

    return {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "max_allocated_mb": max_allocated,
        "max_reserved_mb": max_reserved,
    }


def reset_memory_stats(device: str = "cuda") -> None:
    """Reset peak memory statistics.

    Args:
        device: Device to reset
    """
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        logger.info("Memory stats reset")
