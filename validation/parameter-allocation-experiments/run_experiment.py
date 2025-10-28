#!/usr/bin/env python3
"""Main experiment runner for parameter allocation experiments."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str, experiment_name: Optional[str] = None) -> dict:
    """Load experiment configuration from YAML file.

    Args:
        config_path: Path to YAML config file
        experiment_name: Optional experiment name to select specific config

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if experiment_name and 'experiments' in config:
        experiments = config['experiments']

        # Handle both dict format and list format
        if isinstance(experiments, dict):
            # Dict format: experiments are keys
            if experiment_name in experiments:
                base_config = config.copy()
                base_config.update(experiments[experiment_name])
                return base_config
            else:
                raise ValueError(f"Experiment '{experiment_name}' not found in config")
        elif isinstance(experiments, list):
            # List format: find experiment with matching 'name' field
            experiment = None
            for exp in experiments:
                if exp.get('name') == experiment_name:
                    experiment = exp
                    break

            if experiment is None:
                available = [e.get('name') for e in experiments]
                raise ValueError(
                    f"Experiment '{experiment_name}' not found in config. "
                    f"Available: {available}"
                )

            # Merge base config with experiment config
            base_config = {k: v for k, v in config.items() if k != 'experiments'}

            # Deep merge experiment-specific settings
            for key in ['model', 'training', 'evaluation', 'logging']:
                if key in experiment:
                    if key in base_config:
                        # Merge nested dicts
                        base_config[key] = {**base_config.get(key, {}), **experiment[key]}
                    else:
                        base_config[key] = experiment[key]

            return base_config
        else:
            raise ValueError(f"Invalid experiments format: {type(experiments)}")

    return config


def setup_environment(config: dict) -> None:
    """Setup environment variables and paths.
    
    Args:
        config: Configuration dictionary
    """
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    if 'seed' in config:
        torch.manual_seed(config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['seed'])
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA not available, using CPU")


def run_training(config: dict) -> None:
    """Run training with given configuration.

    Args:
        config: Configuration dictionary
    """
    logger.info("Starting training with config:")
    logger.info(yaml.dump(config, default_flow_style=False))

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / "src"))

    from models.lm import LanguageModel, ModelConfig, print_param_table
    from train import Trainer
    from utils.data import get_dataloaders

    # Build model
    model_config = ModelConfig(
        vocab_size=config.get("vocab_size", 16_000),
        total_params=config.get("total_params", 10_000_000),
        embedding_ratio=config.get("embedding_ratio", 0.35),
        glu_expansion=config.get("glu_expansion", 2.66),
        n_heads=config.get("n_heads", 8),
        max_seq_length=config.get("max_seq_length", 512),
        dropout=config.get("dropout", 0.1),
        tied_lm_head=config.get("tied_lm_head", True),
    )

    model = LanguageModel(model_config)
    logger.info("Model initialized")
    print_param_table(model)

    # Load data
    data_dir = config.get("data_dir", "./data/lm_tokenized")
    batch_size = config.get("batch_size", 32)

    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        max_length=model_config.max_seq_length,
    )

    if train_loader is None:
        logger.error("Failed to load training data")
        return

    # Create trainer
    device = config.get("device", "cuda")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # Train
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    resume = config.get("resume_from_checkpoint", None)
    overfit_tokens = config.get("overfit_tokens", None)

    trainer.train(
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=resume,
        overfit_tokens=overfit_tokens,
    )


def run_evaluation(config: dict) -> None:
    """Run evaluation with given configuration.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Starting evaluation with config:")
    logger.info(yaml.dump(config, default_flow_style=False))
    
    # TODO: Implement actual evaluation logic
    logger.info("Evaluation placeholder - to be implemented")


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(description='Run parameter allocation experiments')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--exp',
        type=str,
        help='Experiment name to run from config file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'eval', 'both'],
        default='train',
        help='Run mode: train, eval, or both'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint for resuming or evaluation'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory for logs'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--lr-find',
        action='store_true',
        help='Run LR range finder before training'
    )
    parser.add_argument(
        '--overfit-tokens',
        type=int,
        help='Overfit on first N tokens (sanity check)'
    )
    
    args = parser.parse_args()
    
    # Check config file exists
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    # Load configuration
    try:
        config = load_config(args.config, args.exp)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Override config with command line arguments
    config['log_dir'] = args.log_dir
    config['device'] = args.device
    if args.checkpoint:
        config['checkpoint'] = args.checkpoint
    if args.overfit_tokens:
        config['overfit_tokens'] = args.overfit_tokens
    config['lr_find'] = args.lr_find
    
    # Setup environment
    setup_environment(config)
    
    # Run experiment
    try:
        if args.mode in ['train', 'both']:
            run_training(config)
        
        if args.mode in ['eval', 'both']:
            run_evaluation(config)
            
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()