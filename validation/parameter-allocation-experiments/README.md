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
│   ├── wikitext-2/      # Raw WikiText-2 data
│   ├── lm_tokenized/    # Preprocessed training data
│   └── smallbench/      # SmallBench evaluation tasks
├── scripts/              # Data processing, training, and evaluation scripts
│   ├── download_*.py    # Data download scripts
│   ├── preprocess_lm.py # Tokenization and preprocessing
│   ├── eval_smallbench.py       # SmallBench evaluation
│   ├── run_evaluation.py        # Comprehensive evaluation
│   ├── aggregate_results.py     # Results aggregation
│   └── generate_plots.py        # Visualization
├── src/
│   ├── models/          # Model implementations (transformer, SwiGLU)
│   ├── eval/            # Evaluation framework (perplexity, profiling, metrics)
│   ├── utils/           # Training utilities (data, checkpointing, determinism)
│   └── train.py         # Training harness
├── tests/               # Unit tests and integration tests
├── logs/                # Training logs and metrics
├── checkpoints/         # Model checkpoints
├── results/             # Evaluation results (metrics.json, summary.md)
└── reports/             # Final analysis and visualizations
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

### 3. Evaluate a Trained Model

After training, evaluate your model with the comprehensive evaluation suite:

**Full evaluation (perplexity + SmallBench + profiling):**
```bash
python scripts/run_evaluation.py \
    --checkpoint checkpoints/best_model.pt \
    --data-dir data/lm_tokenized \
    --output-dir results/exp_001 \
    --experiment-name emb35_glu266
```

**Quick evaluation (skip expensive operations):**
```bash
python scripts/run_evaluation.py \
    --checkpoint checkpoints/best_model.pt \
    --data-dir data/lm_tokenized \
    --output-dir results/quick_test \
    --skip-smallbench \
    --skip-throughput \
    --max-eval-batches 50
```

**Evaluate only on SmallBench:**
```bash
python scripts/eval_smallbench.py \
    --checkpoint checkpoints/best_model.pt \
    --data-dir data/smallbench \
    --tokenizer data/lm_tokenized/tokenizer.json \
    --output results/smallbench.json
```

**Evaluation outputs:**
```
results/exp_001/
├── metrics.json              # Standardized metrics (40+ fields)
├── summary.md                # Human-readable summary
├── smallbench_results.json   # SmallBench task results
├── latency_results.json      # Latency profiling (batch=1, seq=128/512)
└── throughput_results.json   # Throughput profiling
```

**Metrics included:**
- **Perplexity**: Dev/test set perplexity
- **SmallBench**: 4 tasks (sentiment, NLI, QA, paraphrase)
- **Latency**: Inference latency at batch=1, seq=128/512 (ms)
- **Throughput**: Tokens processed per second
- **Memory**: GPU memory usage (MB)

### 4. Run Parameter Allocation Experiments

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

### 5. Analyze Results Across Experiments

After running multiple experiments, aggregate and visualize results:

**Aggregate metrics from all runs:**
```bash
python scripts/aggregate_results.py \
    --log_dir logs \
    --output results/results_summary.csv
```

**Generate visualizations:**
```bash
python scripts/generate_plots.py \
    --input results/results_summary.csv \
    --output_dir reports/figures
```

**Or use Makefile shortcuts:**
```bash
# Run all analysis
make reports

# Individual targets
make aggregate  # Aggregate results to CSV
make plots      # Generate visualizations
```

**Generated plots:**
- Perplexity vs embedding ratio
- Perplexity vs GLU expansion
- Pareto frontier (compute vs quality)
- Training curves comparison
- SmallBench task performance

## Complete Workflow

The complete experimental workflow from data preparation to results:

```bash
# 1. Setup and data preparation
make setup
make data

# 2. Run experiments (parallel execution)
for exp in emb25 emb35 emb45; do
    python run_experiment.py --config configs/embed_ratio.yaml --exp $exp &
done
wait

# 3. Evaluate each trained model
for checkpoint in checkpoints/emb*/best_model.pt; do
    exp_name=$(basename $(dirname $checkpoint))
    python scripts/run_evaluation.py \
        --checkpoint $checkpoint \
        --data-dir data/lm_tokenized \
        --output-dir results/$exp_name \
        --experiment-name $exp_name &
done
wait

# 4. Aggregate and visualize results
python scripts/aggregate_results.py --log_dir results --output results_summary.csv
python scripts/generate_plots.py --input results_summary.csv --output_dir reports/figures

# 5. Review results
cat reports/figures/RESULTS.md
```

## Experiment Details

### Model Configuration
- **Total Parameters**: 10M (fixed across all experiments)
- **Vocabulary Size**: 16,000 tokens (BPE)
- **Architecture**: Pre-norm transformer with SwiGLU activation
- **Parameter Allocation**: Automatic computation to meet ±0.5% tolerance

### Training
- **Data**: ~50M tokens (WikiText-2)
- **Optimizer**: AdamW with cosine LR schedule + warmup
- **Mixed Precision**: FP16 with gradient scaling
- **Checkpointing**: Auto-save and resume support
- **Early Stopping**: Kill if dev PPL ≥0.5 worse for 1M tokens

### Evaluation Metrics
- **Perplexity**: Dev/test set perplexity (primary metric)
- **SmallBench**: 4-task benchmark for capability assessment
  - Sentiment classification (SST-2)
  - Natural language inference (RTE)
  - Question answering (BoolQ)
  - Paraphrase detection (MRPC)
- **Latency**: Inference time at batch=1, seq=128/512
- **Throughput**: Tokens processed per second
- **Memory**: GPU memory consumption

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