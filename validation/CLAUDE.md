# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the Stellux validation repository for parameter allocation experiments on 10M-parameter language models. The goal is to systematically optimize embedding allocation and GLU expansion factors for small-scale LMs to find compute-efficient configurations.

## Project Structure

```
validation/
├── scripts/           # Infrastructure and setup scripts
│   ├── setup_vps.sh  # Bootstrap Hetzner Cloud VPS
│   ├── setup_vast.sh # Prepare Vast.ai GPU instance
│   └── ops.sh        # Sync/tunnel helpers between VPS and Vast
├── validation-prd.md # Product requirements document
├── validation-todo.md # Detailed implementation checklist
└── requirements.txt   # Python dependencies
```

## Key Commands

### Infrastructure Setup

```bash
# On Hetzner VPS (orchestration host)
bash scripts/setup_vps.sh --workspace ~/work/pae --swap 4G

# On Vast.ai GPU instance
bash scripts/setup_vast.sh --workspace /workspace/pae

# Sync data between VPS and Vast
./scripts/ops.sh to ~/work/pae/ /workspace/pae/     # Push to Vast
./scripts/ops.sh from /workspace/pae/logs/ ~/work/pae/logs/  # Pull from Vast
./scripts/ops.sh tb 16006 6006  # TensorBoard tunnel
```

### Experiment Workflows

```bash
# Data preparation
python scripts/download_wikitext.py --output data/wikitext-2
python scripts/preprocess_lm.py --input data/wikitext-2 --vocab_size 16000 --output data/lm_tokenized
python scripts/download_smallbench.py --output data/smallbench

# Run embedding sweep
for cfg in emb35 emb25 emb45; do
  python run_experiment.py --config configs/embed_ratio.yaml --exp $cfg &
done; wait

# Run GLU expansion sweep
for cfg in glu2x glu266x glu3x glu4x; do
  python run_experiment.py --config configs/glu_expansion.yaml --exp $cfg &
done; wait

# Aggregate results
python scripts/aggregate_results.py --log_dir logs --output results_summary.csv
```

### Development

```bash
# Create Python virtual environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run tests (when implemented)
pytest

# Linting (when configured)
ruff check .
black --check .
```

## Architecture Notes

### Experiment Design
- **Two-lever optimization**: embedding parameter share (25%/35%/45%) and GLU expansion factors (2.0/2.66/3.0/4.0)
- **10M total parameters** with 16k vocabulary size
- **Evaluation metrics**: perplexity (primary) and SmallBench accuracy (sanity check)
- **Early stopping**: kill rule if dev PPL ≥0.5 worse than baseline for 1M tokens

### Model Configuration
- Model builder must honor `embedding_ratio` by allocating V×d parameters and adjusting backbone accordingly
- `glu_expansion` scales FFN inner width (SwiGLU architecture)
- Support for tied/untied LM head configurations
- Total parameters must stay within ±0.5% of target

### Infrastructure
- **Hetzner VPS**: Always-on orchestration host with tmux sessions
- **Vast.ai**: Spot GPU instances for training runs
- **Docker containers**: PyTorch CUDA-enabled images for consistent environment
- **SSH ProxyJump**: Access Vast instances through VPS for stability

## Important Constraints

- Keep experiment costs under $50 for sweeps, $150 total including full training
- SmallBench must be held out from pretraining data to avoid contamination
- Use checkpointing every ~15 minutes for spot instance resilience
- Log all metrics to TensorBoard/W&B with separate directories per run
- Generate reproducible results with fixed seeds and deterministic operations

## Expected Deliverables

1. Config files for embedding and GLU sweeps
2. Results summary CSV with aggregated metrics
3. Pareto plots (PPL vs embedding ratio, PPL vs GLU expansion)
4. One-page report with best configuration and analysis
5. Reproducibility bundle (configs, tokenizer, git SHA, seeds)