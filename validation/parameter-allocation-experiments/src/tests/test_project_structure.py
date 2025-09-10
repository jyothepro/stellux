"""Test project structure and configuration for Milestone 0."""

import os
import yaml
import pytest
from pathlib import Path


class TestProjectStructure:
    """Test that all required directories and files exist."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent.parent
    
    def test_directory_structure(self):
        """Test that all required directories exist."""
        required_dirs = [
            "configs",
            "data", 
            "scripts",
            "src",
            "src/models",
            "src/tests",
            "logs",
            "checkpoints",
            "reports",
            "reports/data",
            "reports/figures",
        ]
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            assert dir_path.exists(), f"Directory {dir_name} does not exist"
            assert dir_path.is_dir(), f"{dir_name} is not a directory"
    
    def test_config_files_exist(self):
        """Test that all configuration files exist."""
        config_files = [
            "configs/base_config.yaml",
            "configs/embed_ratio.yaml",
            "configs/glu_expansion.yaml",
        ]
        
        for config_file in config_files:
            file_path = self.project_root / config_file
            assert file_path.exists(), f"Config file {config_file} does not exist"
            assert file_path.is_file(), f"{config_file} is not a file"
    
    def test_config_files_valid_yaml(self):
        """Test that config files are valid YAML."""
        config_files = [
            "configs/base_config.yaml",
            "configs/embed_ratio.yaml", 
            "configs/glu_expansion.yaml",
        ]
        
        for config_file in config_files:
            file_path = self.project_root / config_file
            with open(file_path, 'r') as f:
                try:
                    config = yaml.safe_load(f)
                    assert config is not None, f"{config_file} is empty"
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {config_file}: {e}")
    
    def test_base_config_schema(self):
        """Test that base config has all required fields."""
        config_path = self.project_root / "configs/base_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check top-level sections
        required_sections = ["model", "training", "evaluation", "logging", "infrastructure", "paths"]
        for section in required_sections:
            assert section in config, f"Missing section '{section}' in base config"
        
        # Check model parameters
        model_params = ["total_params", "vocab_size", "max_seq_length"]
        for param in model_params:
            assert param in config["model"], f"Missing model parameter '{param}'"
        
        # Check training parameters
        training_params = ["batch_size", "learning_rate", "total_tokens"]
        for param in training_params:
            assert param in config["training"], f"Missing training parameter '{param}'"
    
    def test_embed_ratio_experiments(self):
        """Test embedding ratio experiment configuration."""
        config_path = self.project_root / "configs/embed_ratio.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert "experiments" in config, "Missing experiments section"
        experiments = config["experiments"]
        
        # Check we have 3 experiments
        assert len(experiments) == 3, f"Expected 3 experiments, got {len(experiments)}"
        
        # Check each experiment
        expected_ratios = [0.25, 0.35, 0.45]
        for exp in experiments:
            assert "name" in exp, "Experiment missing name"
            assert "model" in exp, "Experiment missing model config"
            assert "embedding_ratio" in exp["model"], "Missing embedding_ratio"
            assert exp["model"]["embedding_ratio"] in expected_ratios
    
    def test_glu_expansion_experiments(self):
        """Test GLU expansion experiment configuration."""
        config_path = self.project_root / "configs/glu_expansion.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert "experiments" in config, "Missing experiments section"
        experiments = config["experiments"]
        
        # Check we have 4 experiments
        assert len(experiments) == 4, f"Expected 4 experiments, got {len(experiments)}"
        
        # Check each experiment
        expected_factors = [2.0, 2.66, 3.0, 4.0]
        for exp in experiments:
            assert "name" in exp, "Experiment missing name"
            assert "model" in exp, "Experiment missing model config"
            assert "glu_expansion" in exp["model"], "Missing glu_expansion"
            assert exp["model"]["glu_expansion"] in expected_factors
    
    def test_project_files_exist(self):
        """Test that essential project files exist."""
        required_files = [
            "README.md",
            "requirements.txt",
            "pyproject.toml",
            ".pre-commit-config.yaml",
            ".env.example",
            "Makefile",
        ]
        
        # Check in parameter-allocation-experiments directory
        pae_root = self.project_root
        for file_name in required_files:
            if file_name == "requirements.txt":
                # This is in the parent directory
                file_path = self.project_root.parent / file_name
            else:
                file_path = pae_root / file_name
            assert file_path.exists(), f"Required file {file_name} does not exist"
    
    def test_github_workflows(self):
        """Test that GitHub Actions workflow exists."""
        workflow_path = self.project_root / ".github/workflows/ci.yml"
        assert workflow_path.exists(), "GitHub Actions CI workflow does not exist"
        
        # Validate it's valid YAML
        with open(workflow_path, 'r') as f:
            try:
                workflow = yaml.safe_load(f)
                assert "jobs" in workflow, "Workflow missing jobs section"
                assert "lint" in workflow["jobs"], "Workflow missing lint job"
                assert "test" in workflow["jobs"], "Workflow missing test job"
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML in workflow: {e}")
    
    def test_makefile_targets(self):
        """Test that Makefile has all required targets."""
        makefile_path = self.project_root / "Makefile"
        with open(makefile_path, 'r') as f:
            content = f.read()
        
        required_targets = [
            "setup", "test", "lint", "format", "data",
            "train-embedding-sweep", "train-glu-sweep",
            "eval", "reports", "clean"
        ]
        
        for target in required_targets:
            assert f"{target}:" in content, f"Makefile missing target '{target}'"


class TestDependencies:
    """Test project dependencies and setup."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent.parent
    
    def test_requirements_file(self):
        """Test that requirements.txt is valid."""
        # Check parent directory for requirements.txt
        req_path = self.project_root.parent / "requirements.txt"
        assert req_path.exists(), "requirements.txt does not exist"
        
        with open(req_path, 'r') as f:
            lines = f.readlines()
        
        # Check for essential packages
        packages = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                packages.append(line.split('==')[0].split('>=')[0])
        
        essential = [
            "accelerate", "transformers", "datasets", "tokenizers",
            "tensorboard", "pytest", "ruff", "black"
        ]
        
        for pkg in essential:
            assert pkg in packages, f"Missing essential package: {pkg}"
    
    def test_pyproject_toml(self):
        """Test pyproject.toml configuration."""
        pyproject_path = self.project_root / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml does not exist"
        
        # We can't use tomllib in Python 3.8, so just check structure
        with open(pyproject_path, 'r') as f:
            content = f.read()
        
        # Check for essential sections
        assert "[project]" in content, "Missing [project] section"
        assert "[tool.black]" in content, "Missing [tool.black] section"
        assert "[tool.ruff]" in content, "Missing [tool.ruff] section"
        assert "[tool.pytest.ini_options]" in content, "Missing pytest config"