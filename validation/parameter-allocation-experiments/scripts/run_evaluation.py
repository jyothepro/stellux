#!/usr/bin/env python3
"""
Comprehensive evaluation script for parameter allocation experiments.

This script runs all evaluation components:
1. Dev/test perplexity
2. SmallBench tasks
3. Latency/throughput profiling
4. Memory profiling

Results are saved in standardized metrics.json format.
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval.evaluate import evaluate_splits
from eval.metrics import StandardMetrics
from eval.profiler import (
    LatencyProfiler,
    ThroughputProfiler,
    get_memory_stats,
    reset_memory_stats,
)
from eval.summary import generate_run_summary
from models.lm import LanguageModel, ModelConfig
from utils.checkpointing import load_checkpoint
from utils.data import get_dataloaders

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def run_comprehensive_evaluation(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str,
    run_id: str,
    experiment_name: str,
    device: str = "cuda",
    eval_perplexity: bool = True,
    eval_smallbench: bool = True,
    profile_latency: bool = True,
    profile_throughput: bool = True,
    max_eval_batches: Optional[int] = None,
    smallbench_max_samples: Optional[int] = None,
) -> StandardMetrics:
    """Run comprehensive evaluation on a trained model.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Path to data directory
        output_dir: Directory to save results
        run_id: Unique run identifier
        experiment_name: Experiment name
        device: Device to run on
        eval_perplexity: Whether to evaluate perplexity
        eval_smallbench: Whether to evaluate SmallBench
        profile_latency: Whether to profile latency
        profile_throughput: Whether to profile throughput
        max_eval_batches: Max batches for perplexity eval
        smallbench_max_samples: Max samples per SmallBench task

    Returns:
        StandardMetrics object with all results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"Running comprehensive evaluation: {experiment_name}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 80)

    # Load model
    logger.info("Loading model from checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get model config
    if "model_config" in checkpoint:
        model_config = checkpoint["model_config"]
        if isinstance(model_config, dict):
            model_config = ModelConfig(**model_config)
    else:
        logger.warning("Model config not in checkpoint, using default")
        model_config = ModelConfig()

    # Create model
    model = LanguageModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    logger.info(f"Model loaded: {model_config.total_params:,} parameters")

    # Initialize metrics
    metrics = StandardMetrics(
        run_id=run_id,
        experiment_name=experiment_name,
        timestamp=datetime.now().isoformat(),
        model_config=model_config.__dict__,
        total_params=model_config.total_params,
        d_model=model.d_model,
        d_ff=model.d_ff,
        n_layers=model.n_layers,
        n_heads=model_config.n_heads,
        embedding_ratio=model_config.embedding_ratio,
        glu_expansion=model_config.glu_expansion,
        train_loss=checkpoint.get("train_loss", 0.0),
        train_perplexity=checkpoint.get("train_perplexity", 0.0),
        best_train_loss=checkpoint.get("best_loss", 0.0),
        total_steps=checkpoint.get("step", 0),
        total_epochs=checkpoint.get("epoch", 0),
        device=device,
        num_gpus=torch.cuda.device_count() if device == "cuda" else 0,
        seed=model_config.seed if hasattr(model_config, "seed") else 42,
        git_commit=get_git_commit(),
    )

    # 1. Evaluate perplexity on dev/test splits
    if eval_perplexity:
        logger.info("\n" + "=" * 80)
        logger.info("Evaluating perplexity on dev/test splits...")
        logger.info("=" * 80)

        # Load dataloaders
        train_loader, val_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=32,
            max_length=512,
        )

        if val_loader is not None:
            results = evaluate_splits(
                model=model,
                val_loader=val_loader,
                device=device,
                max_batches=max_eval_batches,
            )

            if "val" in results:
                metrics.val_loss = results["val"]["loss"]
                metrics.val_perplexity = results["val"]["perplexity"]
                metrics.best_val_loss = results["val"]["loss"]

    # 2. Evaluate on SmallBench
    if eval_smallbench:
        logger.info("\n" + "=" * 80)
        logger.info("Evaluating on SmallBench tasks...")
        logger.info("=" * 80)

        smallbench_dir = Path(data_dir).parent / "smallbench"
        tokenizer_path = Path(data_dir) / "tokenizer.json"

        if smallbench_dir.exists() and tokenizer_path.exists():
            try:
                # Run eval_smallbench.py script
                cmd = [
                    sys.executable,
                    str(Path(__file__).parent / "eval_smallbench.py"),
                    "--checkpoint", checkpoint_path,
                    "--data-dir", str(smallbench_dir),
                    "--tokenizer", str(tokenizer_path),
                    "--output", str(output_path / "smallbench_results.json"),
                    "--device", device,
                ]

                if smallbench_max_samples:
                    cmd.extend(["--max-samples", str(smallbench_max_samples)])

                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    # Load results
                    with open(output_path / "smallbench_results.json") as f:
                        smallbench_results = json.load(f)

                    metrics.smallbench_sentiment_acc = smallbench_results.get("sentiment", {}).get("accuracy")
                    metrics.smallbench_nli_acc = smallbench_results.get("nli", {}).get("accuracy")
                    metrics.smallbench_qa_acc = smallbench_results.get("qa", {}).get("accuracy")
                    metrics.smallbench_paraphrase_acc = smallbench_results.get("paraphrase", {}).get("accuracy")
                    metrics.smallbench_avg_acc = smallbench_results.get("average", {}).get("accuracy")
                else:
                    logger.warning(f"SmallBench evaluation failed: {result.stderr}")

            except Exception as e:
                logger.warning(f"Failed to run SmallBench evaluation: {e}")
        else:
            logger.warning("SmallBench data or tokenizer not found, skipping")

    # 3. Profile latency
    if profile_latency and device == "cuda":
        logger.info("\n" + "=" * 80)
        logger.info("Profiling latency...")
        logger.info("=" * 80)

        try:
            profiler = LatencyProfiler(model, device)
            latency_results = profiler.profile_multiple_configs(
                configs=[(1, 128), (1, 512)],
                num_runs=100,
            )

            if "batch1_seq128" in latency_results:
                metrics.latency_batch1_seq128 = latency_results["batch1_seq128"]["mean_ms"]
            if "batch1_seq512" in latency_results:
                metrics.latency_batch1_seq512 = latency_results["batch1_seq512"]["mean_ms"]

            # Save detailed latency results
            with open(output_path / "latency_results.json", "w") as f:
                json.dump(latency_results, f, indent=2)

        except Exception as e:
            logger.warning(f"Latency profiling failed: {e}")

    # 4. Profile throughput
    if profile_throughput and device == "cuda":
        logger.info("\n" + "=" * 80)
        logger.info("Profiling throughput...")
        logger.info("=" * 80)

        try:
            reset_memory_stats(device)
            profiler = ThroughputProfiler(model, device)
            throughput_results = profiler.profile_throughput(
                batch_size=32,
                seq_length=512,
                duration_seconds=10.0,
            )

            metrics.throughput_tokens_per_sec = throughput_results["tokens_per_sec"]

            # Get memory stats
            mem_stats = get_memory_stats(device)
            metrics.memory_allocated_mb = mem_stats["allocated_mb"]
            metrics.memory_reserved_mb = mem_stats["reserved_mb"]

            # Save detailed throughput results
            with open(output_path / "throughput_results.json", "w") as f:
                json.dump(throughput_results, f, indent=2)

        except Exception as e:
            logger.warning(f"Throughput profiling failed: {e}")

    # Save standardized metrics
    logger.info("\n" + "=" * 80)
    logger.info("Saving results...")
    logger.info("=" * 80)

    metrics_path = output_path / "metrics.json"
    metrics.to_json(str(metrics_path))

    # Generate summary
    summary_path = output_path / "summary.md"
    generate_run_summary(metrics, str(summary_path), format="markdown")

    logger.info(f"\nResults saved to {output_dir}/")
    logger.info(f"  - metrics.json (standardized metrics)")
    logger.info(f"  - summary.md (human-readable summary)")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation on trained model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/lm_tokenized",
        help="Path to tokenized data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID (auto-generated if not provided)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="evaluation",
        help="Experiment name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Skip perplexity evaluation",
    )
    parser.add_argument(
        "--skip-smallbench",
        action="store_true",
        help="Skip SmallBench evaluation",
    )
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip latency profiling",
    )
    parser.add_argument(
        "--skip-throughput",
        action="store_true",
        help="Skip throughput profiling",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Max batches for perplexity eval (for quick testing)",
    )
    parser.add_argument(
        "--smallbench-max-samples",
        type=int,
        default=None,
        help="Max samples per SmallBench task (for quick testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Generate run ID if not provided
    if args.run_id is None:
        args.run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Run evaluation
    metrics = run_comprehensive_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        experiment_name=args.experiment_name,
        device=args.device,
        eval_perplexity=not args.skip_perplexity,
        eval_smallbench=not args.skip_smallbench,
        profile_latency=not args.skip_latency,
        profile_throughput=not args.skip_throughput,
        max_eval_batches=args.max_eval_batches,
        smallbench_max_samples=args.smallbench_max_samples,
    )

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation complete!")
    logger.info("=" * 80)

    return metrics


if __name__ == "__main__":
    main()
