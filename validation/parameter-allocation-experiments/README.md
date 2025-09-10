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

### Data Preparation
```bash
make data
```

### Run Experiments
```bash
# Embedding ratio sweep
make train-embedding-sweep

# GLU expansion sweep  
make train-glu-sweep
```

### Generate Reports
```bash
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