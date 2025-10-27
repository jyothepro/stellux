# Parameter Allocation Experiments for 10M-Parameter LMs

This repository contains systematic experiments to optimize parameter allocation in small language models (≤100M parameters), focusing on embedding allocation and GLU expansion factors.

## Overview

We systematically sweep two critical design choices at 10M parameters:
- **Embedding parameter share**: 25%, 35%, 45% of total parameters
- **GLU expansion factors**: 2.0×, 2.66×, 3.0×, 4.0×

Our goal is to identify Pareto-efficient configurations that maximize quality per parameter while maintaining computational efficiency.

## Project Structure

```
parameter-allocation-experiments/
├── configs/              # Experiment configuration files
├── data/                 # Training and evaluation datasets
├── scripts/              # Data processing and analysis scripts
├── src/
│   ├── models/          # Model implementations
│   └── tests/           # Unit tests
├── logs/                # Training logs and metrics
├── checkpoints/         # Model checkpoints
└── reports/             # Results and visualizations
```

## Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (V100/A100 recommended)
- 16GB+ RAM

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run setup
make setup
```

## Quick Start

### 1. Data Preparation

Download and preprocess the training data:

```bash
# Download WikiText-2 dataset (~50M tokens)
python scripts/download_wikitext.py --output data/wikitext-2

# Train tokenizer and preprocess (16k vocab, BPE)
python scripts/preprocess_lm.py \
    --input data/wikitext-2 \
    --vocab_size 16000 \
    --output data/lm_tokenized

# Download SmallBench evaluation tasks (optional)
python scripts/download_smallbench.py --output data/smallbench
```

Or use the Makefile shortcut:
```bash
make data
```

### 2. Run Training

**Basic training:**
```bash
python run_experiment.py --config configs/base_config.yaml
```

**With experiment selection:**
```bash
# Run specific experiment from config
python run_experiment.py \
    --config configs/embed_ratio.yaml \
    --exp emb35
```

**LR range finder (find optimal learning rate):**
```bash
python run_experiment.py \
    --config configs/base_config.yaml \
    --lr-find
```

**Overfit sanity check:**
```bash
# Train on first 1024 tokens to verify training works
python run_experiment.py \
    --config configs/base_config.yaml \
    --overfit-tokens 1024
```

**Resume from checkpoint:**
```bash
python run_experiment.py \
    --config configs/base_config.yaml \
    --checkpoint auto
```

### 3. Run Parameter Allocation Experiments

**Embedding ratio sweep (25%, 35%, 45%):**
```bash
for exp in emb25 emb35 emb45; do
    python run_experiment.py \
        --config configs/embed_ratio.yaml \
        --exp $exp &
done
wait
```

Or use Makefile:
```bash
make train-embedding-sweep
```

**GLU expansion sweep (2.0×, 2.66×, 3.0×, 4.0×):**
```bash
for exp in glu2x glu266x glu3x glu4x; do
    python run_experiment.py \
        --config configs/glu_expansion.yaml \
        --exp $exp &
done
wait
```

Or use Makefile:
```bash
make train-glu-sweep
```

### 4. Generate Reports

```bash
# Aggregate results from all experiments
python scripts/aggregate_results.py \
    --log_dir logs \
    --output results_summary.csv

# Generate plots
python scripts/generate_plots.py \
    --input results_summary.csv \
    --output_dir reports/figures

# Or use Makefile
make reports
```

## Experiment Details

- **Total Parameters**: 10M (fixed)
- **Vocabulary Size**: 16,000 tokens
- **Training Data**: ~50M tokens (WikiText-2 or WMT subset)
- **Evaluation**: Perplexity (primary), SmallBench accuracy (secondary)
- **Early Stopping**: Kill if dev PPL ≥0.5 worse than baseline for 1M tokens

## Results

Results and analysis will be available in `reports/RESULTS.md` after experiments complete.

## Citation

If you use this code, please cite:
```bibtex
@misc{stellux2024param,
  title={Parameter Allocation Experiments for Small Language Models},
  author={Stellux Team},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.