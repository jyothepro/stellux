"""Basic tests for project structure and imports."""

import os
import sys
from pathlib import Path


def test_project_structure():
    """Test that required directories exist."""
    root = Path(__file__).parent.parent
    
    required_dirs = ['configs', 'data', 'scripts', 'src', 'logs', 'checkpoints']
    for dir_name in required_dirs:
        assert (root / dir_name).exists(), f"Missing required directory: {dir_name}"


def test_config_files():
    """Test that example config files exist."""
    root = Path(__file__).parent.parent
    
    config_files = [
        'configs/embed_ratio.yaml',
        'configs/glu_expansion.yaml'
    ]
    
    for config_file in config_files:
        assert (root / config_file).exists(), f"Missing config file: {config_file}"


def test_run_experiment_exists():
    """Test that main experiment runner exists."""
    root = Path(__file__).parent.parent
    assert (root / 'run_experiment.py').exists(), "Missing run_experiment.py"


def test_imports():
    """Test that main modules can be imported."""
    root = Path(__file__).parent.parent
    sys.path.insert(0, str(root))
    
    try:
        import run_experiment
        assert hasattr(run_experiment, 'main'), "run_experiment missing main function"
    except ImportError as e:
        # Allow torch import error since it might not be installed yet
        if "torch" in str(e):
            pass  # This is expected if torch isn't installed
        else:
            assert False, f"Failed to import run_experiment: {e}"


def test_environment():
    """Test basic environment setup."""
    assert sys.version_info >= (3, 8), "Python 3.8+ required"
    
    try:
        import torch
        assert torch.__version__, "PyTorch not properly installed"
    except ImportError:
        pass  # PyTorch might not be installed yet in CI