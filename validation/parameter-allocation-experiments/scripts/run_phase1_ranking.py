#!/usr/bin/env python3
"""
Run Phase 1 ranking experiments for parameter allocation.

Phase 1 goals:
- Run all variants (embedding ratios and GLU expansions) to 5M tokens
- Apply kill rule: stop if dev PPL ≥0.5 worse for 1M tokens
- Aggregate results and select top-1 per axis

Usage:
    # Run embedding ratio sweep
    python scripts/run_phase1_ranking.py --sweep embedding --parallel 3

    # Run GLU expansion sweep
    python scripts/run_phase1_ranking.py --sweep glu --parallel 4

    # Run all sweeps sequentially
    python scripts/run_phase1_ranking.py --sweep all
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_config(config_path: str) -> Dict:
    """Load YAML config file.

    Args:
        config_path: Path to config file

    Returns:
        Config dictionary
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def run_single_experiment(
    config_path: str,
    exp_name: str,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict:
    """Run a single experiment.

    Args:
        config_path: Path to config file
        exp_name: Experiment name
        dry_run: If True, only print command
        verbose: Verbose logging

    Returns:
        Dictionary with experiment results
    """
    logger.info(f"Starting experiment: {exp_name}")

    # Build command
    cmd = [
        sys.executable,
        "run_experiment.py",
        "--config", config_path,
        "--exp", exp_name,
    ]

    if verbose:
        cmd.append("--verbose")

    logger.info(f"Command: {' '.join(cmd)}")

    if dry_run:
        return {
            "exp_name": exp_name,
            "status": "dry_run",
            "message": "Dry run - command not executed",
        }

    # Run experiment
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        elapsed = time.time() - start_time

        logger.info(f"Experiment {exp_name} completed in {elapsed:.1f}s")

        return {
            "exp_name": exp_name,
            "status": "success",
            "elapsed_seconds": elapsed,
            "stdout": result.stdout[-1000:],  # Last 1000 chars
            "stderr": result.stderr[-1000:] if result.stderr else "",
        }

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time

        logger.error(f"Experiment {exp_name} failed after {elapsed:.1f}s")
        logger.error(f"Error: {e.stderr}")

        return {
            "exp_name": exp_name,
            "status": "failed",
            "elapsed_seconds": elapsed,
            "error": str(e),
            "stdout": e.stdout[-1000:] if e.stdout else "",
            "stderr": e.stderr[-1000:] if e.stderr else "",
        }

    except Exception as e:
        elapsed = time.time() - start_time

        logger.error(f"Experiment {exp_name} crashed: {e}")

        return {
            "exp_name": exp_name,
            "status": "crashed",
            "elapsed_seconds": elapsed,
            "error": str(e),
        }


def run_experiments_parallel(
    config_path: str,
    exp_names: List[str],
    max_workers: int = 3,
    dry_run: bool = False,
    verbose: bool = False,
) -> List[Dict]:
    """Run multiple experiments in parallel.

    Args:
        config_path: Path to config file
        exp_names: List of experiment names
        max_workers: Maximum parallel workers
        dry_run: If True, only print commands
        verbose: Verbose logging

    Returns:
        List of experiment results
    """
    logger.info(f"Running {len(exp_names)} experiments with {max_workers} workers")

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        futures = {
            executor.submit(
                run_single_experiment,
                config_path,
                exp_name,
                dry_run,
                verbose,
            ): exp_name
            for exp_name in exp_names
        }

        # Collect results as they complete
        for future in as_completed(futures):
            exp_name = futures[future]
            try:
                result = future.result()
                results.append(result)

                logger.info(f"Collected result for {exp_name}: {result['status']}")

            except Exception as e:
                logger.error(f"Failed to get result for {exp_name}: {e}")
                results.append({
                    "exp_name": exp_name,
                    "status": "error",
                    "error": str(e),
                })

    return results


def run_experiments_sequential(
    config_path: str,
    exp_names: List[str],
    dry_run: bool = False,
    verbose: bool = False,
) -> List[Dict]:
    """Run experiments sequentially.

    Args:
        config_path: Path to config file
        exp_names: List of experiment names
        dry_run: If True, only print commands
        verbose: Verbose logging

    Returns:
        List of experiment results
    """
    logger.info(f"Running {len(exp_names)} experiments sequentially")

    results = []

    for exp_name in exp_names:
        result = run_single_experiment(config_path, exp_name, dry_run, verbose)
        results.append(result)

    return results


def save_results(results: List[Dict], output_path: str) -> None:
    """Save experiment results to JSON.

    Args:
        results: List of experiment results
        output_path: Output file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def print_summary(results: List[Dict]) -> None:
    """Print summary of experiment results.

    Args:
        results: List of experiment results
    """
    logger.info("=" * 80)
    logger.info("Phase 1 Ranking Runs - Summary")
    logger.info("=" * 80)

    total = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    crashed = sum(1 for r in results if r["status"] == "crashed")

    logger.info(f"Total experiments: {total}")
    logger.info(f"  Success: {success}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Crashed: {crashed}")

    if success > 0:
        total_time = sum(r.get("elapsed_seconds", 0) for r in results)
        avg_time = total_time / len(results)
        logger.info(f"  Average time: {avg_time:.1f}s")

    logger.info("")
    logger.info("Individual results:")
    for r in results:
        status_emoji = {
            "success": "✓",
            "failed": "✗",
            "crashed": "💥",
            "dry_run": "🔍",
        }.get(r["status"], "?")

        elapsed = r.get("elapsed_seconds", 0)
        logger.info(f"  {status_emoji} {r['exp_name']:20s} {r['status']:10s} ({elapsed:.1f}s)")

    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 1 ranking experiments"
    )
    parser.add_argument(
        "--sweep",
        type=str,
        choices=["embedding", "glu", "all"],
        default="embedding",
        help="Which sweep to run",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Directory containing config files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase1",
        help="Directory to save results",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel experiments (1 for sequential)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Define sweeps
    sweeps = {}

    if args.sweep in ["embedding", "all"]:
        sweeps["embedding"] = {
            "config": f"{args.config_dir}/phase1_embedding_sweep.yaml",
            "experiments": [
                "phase1_emb25",
                "phase1_emb35",
                "phase1_emb45",
            ],
        }

    if args.sweep in ["glu", "all"]:
        sweeps["glu"] = {
            "config": f"{args.config_dir}/phase1_glu_sweep.yaml",
            "experiments": [
                "phase1_glu2x",
                "phase1_glu266x",
                "phase1_glu3x",
                "phase1_glu4x",
            ],
        }

    # Run sweeps
    all_results = []

    for sweep_name, sweep_config in sweeps.items():
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Running {sweep_name} sweep")
        logger.info(f"Config: {sweep_config['config']}")
        logger.info(f"Experiments: {len(sweep_config['experiments'])}")
        logger.info(f"{'=' * 80}\n")

        # Run experiments
        if args.parallel > 1:
            results = run_experiments_parallel(
                config_path=sweep_config["config"],
                exp_names=sweep_config["experiments"],
                max_workers=args.parallel,
                dry_run=args.dry_run,
                verbose=args.verbose,
            )
        else:
            results = run_experiments_sequential(
                config_path=sweep_config["config"],
                exp_names=sweep_config["experiments"],
                dry_run=args.dry_run,
                verbose=args.verbose,
            )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.output_dir) / f"{sweep_name}_results_{timestamp}.json"
        save_results(results, str(output_path))

        # Print summary
        print_summary(results)

        all_results.extend(results)

    # Save combined results
    if len(sweeps) > 1:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_path = Path(args.output_dir) / f"all_results_{timestamp}.json"
        save_results(all_results, str(combined_path))

    logger.info("\nAll Phase 1 ranking runs complete!")

    # Check if any failed
    failed_count = sum(1 for r in all_results if r["status"] != "success")
    if failed_count > 0:
        logger.warning(f"{failed_count} experiments failed or crashed")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
