"""Configuration loading and management utilities."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json


@dataclass
class ExperimentConfig:
    """Container for experiment configuration."""
    
    name: str
    model: Dict[str, Any]
    training: Dict[str, Any]
    evaluation: Dict[str, Any]
    logging: Dict[str, Any]
    infrastructure: Dict[str, Any]
    paths: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "model": self.model,
            "training": self.training,
            "evaluation": self.evaluation,
            "logging": self.logging,
            "infrastructure": self.infrastructure,
            "paths": self.paths,
        }
    
    def save(self, path: Path) -> None:
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == ".yaml" or path.suffix == ".yml":
            with open(path, 'w') as f:
                yaml.safe_dump(self.to_dict(), f, default_flow_style=False)
        elif path.suffix == ".json":
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")


class ConfigLoader:
    """Load and merge experiment configurations."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize config loader.
        
        Args:
            config_dir: Directory containing config files. 
                       Defaults to configs/ in project root.
        """
        if config_dir is None:
            # Find project root (contains configs directory)
            current = Path(__file__).parent
            while current != current.parent:
                if (current / "configs").exists():
                    config_dir = current / "configs"
                    break
                current = current.parent
            
            if config_dir is None:
                raise ValueError("Could not find configs directory")
        
        self.config_dir = Path(config_dir)
        self.base_config = self._load_base_config()
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration."""
        base_path = self.config_dir / "base_config.yaml"
        if not base_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_path}")
        
        with open(base_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Dictionary with values to override
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def load_experiment_config(self, config_file: str, experiment_name: str) -> ExperimentConfig:
        """Load configuration for a specific experiment.
        
        Args:
            config_file: Name of config file (e.g., "embed_ratio.yaml")
            experiment_name: Name of experiment (e.g., "emb35")
            
        Returns:
            ExperimentConfig object
        """
        # Load experiment file
        exp_path = self.config_dir / config_file
        if not exp_path.exists():
            raise FileNotFoundError(f"Config file not found: {exp_path}")
        
        with open(exp_path, 'r') as f:
            exp_config = yaml.safe_load(f)
        
        # Find specific experiment
        experiments = exp_config.get("experiments", [])
        experiment = None
        for exp in experiments:
            if exp.get("name") == experiment_name:
                experiment = exp
                break
        
        if experiment is None:
            available = [e.get("name") for e in experiments]
            raise ValueError(f"Experiment '{experiment_name}' not found. Available: {available}")
        
        # Start with base config
        merged_config = self.base_config.copy()
        
        # Apply experiment-level overrides from file
        for key in ["model", "training", "evaluation", "logging"]:
            if key in exp_config and key != "experiments":
                merged_config[key] = self._deep_merge(merged_config.get(key, {}), exp_config[key])
        
        # Apply specific experiment overrides
        for key in ["model", "training", "evaluation", "logging"]:
            if key in experiment:
                merged_config[key] = self._deep_merge(merged_config.get(key, {}), experiment[key])
        
        # Create ExperimentConfig
        return ExperimentConfig(
            name=experiment_name,
            model=merged_config.get("model", {}),
            training=merged_config.get("training", {}),
            evaluation=merged_config.get("evaluation", {}),
            logging=merged_config.get("logging", {}),
            infrastructure=merged_config.get("infrastructure", {}),
            paths=merged_config.get("paths", {}),
        )
    
    def get_all_experiments(self, config_file: str) -> List[str]:
        """Get list of all experiments in a config file.
        
        Args:
            config_file: Name of config file
            
        Returns:
            List of experiment names
        """
        exp_path = self.config_dir / config_file
        if not exp_path.exists():
            raise FileNotFoundError(f"Config file not found: {exp_path}")
        
        with open(exp_path, 'r') as f:
            exp_config = yaml.safe_load(f)
        
        experiments = exp_config.get("experiments", [])
        return [exp.get("name") for exp in experiments if "name" in exp]