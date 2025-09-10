.PHONY: help setup clean test lint format install-hooks data train-embedding-sweep train-glu-sweep eval reports all

# Default target
help:
	@echo "Parameter Allocation Experiments - Makefile targets"
	@echo ""
	@echo "Setup & Environment:"
	@echo "  make setup          - Complete setup (venv, deps, pre-commit)"
	@echo "  make install-hooks  - Install pre-commit hooks"
	@echo "  make clean          - Remove generated files and caches"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Run unit tests with coverage"
	@echo "  make lint           - Run linters (ruff, black, isort)"
	@echo "  make format         - Auto-format code"
	@echo ""
	@echo "Data Preparation:"
	@echo "  make data           - Download and preprocess all datasets"
	@echo ""
	@echo "Training:"
	@echo "  make train-embedding-sweep - Run embedding ratio experiments"
	@echo "  make train-glu-sweep      - Run GLU expansion experiments"
	@echo ""
	@echo "Evaluation & Reports:"
	@echo "  make eval           - Run evaluation on all checkpoints"
	@echo "  make reports        - Generate plots and summary reports"
	@echo ""
	@echo "Complete Pipeline:"
	@echo "  make all            - Run complete pipeline (setup → train → eval → reports)"

# Setup targets
setup:
	@echo "Setting up development environment..."
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt
	. .venv/bin/activate && pip install -e .
	@echo "Installing pre-commit hooks..."
	. .venv/bin/activate && pre-commit install
	@echo "Creating directories..."
	mkdir -p data logs checkpoints reports/figures reports/data
	@echo "Setup complete! Activate venv with: source .venv/bin/activate"

install-hooks:
	. .venv/bin/activate && pre-commit install
	. .venv/bin/activate && pre-commit run --all-files

# Development targets
test:
	. .venv/bin/activate && pytest src/tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint:
	. .venv/bin/activate && ruff check src/
	. .venv/bin/activate && black --check src/
	. .venv/bin/activate && isort --check-only src/

format:
	. .venv/bin/activate && ruff check --fix src/
	. .venv/bin/activate && black src/
	. .venv/bin/activate && isort src/

# Data preparation
data:
	@echo "Downloading and preprocessing datasets..."
	. .venv/bin/activate && python scripts/download_wikitext.py --output data/wikitext-2
	. .venv/bin/activate && python scripts/preprocess_lm.py \
		--input data/wikitext-2 \
		--vocab_size 16000 \
		--output data/lm_tokenized
	. .venv/bin/activate && python scripts/download_smallbench.py --output data/smallbench
	@echo "Data preparation complete!"

# Training experiments
train-embedding-sweep:
	@echo "Running embedding ratio sweep..."
	@for cfg in emb25 emb35 emb45; do \
		echo "Starting experiment: $$cfg"; \
		. .venv/bin/activate && python run_experiment.py \
			--config configs/embed_ratio.yaml \
			--exp $$cfg & \
	done; \
	wait
	@echo "Embedding sweep complete!"

train-glu-sweep:
	@echo "Running GLU expansion sweep..."
	@for cfg in glu2x glu266x glu3x glu4x; do \
		echo "Starting experiment: $$cfg"; \
		. .venv/bin/activate && python run_experiment.py \
			--config configs/glu_expansion.yaml \
			--exp $$cfg & \
	done; \
	wait
	@echo "GLU sweep complete!"

# Evaluation
eval:
	@echo "Running evaluation on all checkpoints..."
	. .venv/bin/activate && python scripts/evaluate_all.py \
		--checkpoint_dir checkpoints \
		--output_dir reports/data

# Training (generic)
train:
	@echo "Running default training..."
	. .venv/bin/activate && python run_experiment.py \
		--config configs/base_config.yaml \
		--mode train

# Reports and visualization
reports:
	@echo "Generating reports and visualizations..."
	. .venv/bin/activate && python scripts/aggregate_results.py \
		--log_dir logs \
		--output results_summary.csv
	. .venv/bin/activate && python scripts/generate_plots.py \
		--input results_summary.csv \
		--output_dir reports/figures
	@echo "Reports generated in reports/ directory"

# Generate plots
plots:
	@echo "Generating plots..."
	. .venv/bin/activate && python scripts/generate_plots.py \
		--input results_summary.csv \
		--output_dir reports/figures
	@echo "Plots saved to reports/figures/"

# Clean up
clean:
	rm -rf .venv __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf build dist *.egg-info

# Complete pipeline
all: setup data train-embedding-sweep train-glu-sweep eval reports
	@echo "Complete pipeline finished!"