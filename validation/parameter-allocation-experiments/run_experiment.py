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
        if experiment_name in config['experiments']:
            base_config = config.copy()
            base_config.update(config['experiments'][experiment_name])
            return base_config
        else:
            raise ValueError(f"Experiment '{experiment_name}' not found in config")
    
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
    
    # TODO: Implement actual training logic
    # This will be expanded in later milestones
    logger.info("Training placeholder - to be implemented")
    
    # Placeholder for model initialization
    # model = build_model(config)
    
    # Placeholder for data loading
    # train_loader, val_loader = get_dataloaders(config)
    
    # Placeholder for training loop
    # trainer = Trainer(model, config)
    # trainer.train(train_loader, val_loader)


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