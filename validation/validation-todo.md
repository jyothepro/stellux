# TODO: Parameter Allocation Experiments (10M LM)

> Use this checklist to drive the build from repo setup → experiments → report.  
> Convention: check off subtasks first; then check the milestone.

---

- [ ] **Milestone 0 — Initialize repo & infra**
  - [ ] Create a folder scaffold (`parameter-allocation-experiments/`) with folders from PRD.
  - [ ] Add `README.md`
  - [ ] Create Python env & pin deps (`pyproject.toml`/`requirements.txt`).
  - [ ] Add `pre-commit` (ruff/black/isort) and basic unit test framework (pytest).
  - [ ] Set up GitHub Actions: lint + tests on PR; build artifacts on tag.
  - [ ] Configure W&B/TensorBoard logging boilerplate (API key via secrets).
  - [ ] Define `configs/` schema and example configs; add `.env.example`.
  - [ ] Basic `Makefile` with common targets (`setup`, `train`, `eval`, `plots`).

- [ ] **Milestone 1 — Data ingestion & preprocessing**
  - [ ] Implement `scripts/download_wikitext.py` (or WMT subset) to `data/wikitext-2`.
  - [ ] Implement `scripts/preprocess_lm.py` (train tokenizer; `--vocab_size 16_000`).
  - [ ] Implement `scripts/download_smallbench.py` to `data/smallbench`.
  - [ ] Write a `DATASET_MANIFEST.json` (sources, license, counts, token stats).
  - [ ] Add unit tests for tokenization determinism (seed, special tokens).
  - [ ] Sanity plot: token length histogram → `reports/data/length_hist.png`.

- [ ] **Milestone 2 — Model builder (parameter accounting)**
  - [ ] Implement `src/models/lm.py` with knobs: `total_params`, `embedding_ratio`, `glu_expansion`, `tied_lm_head`.
  - [ ] Compute param budget per block; assert `±0.5%` of `total_params`.
  - [ ] Implement GLU/SwiGLU FFN with variable inner width.
  - [ ] Add `print_param_table()` to log counts by component (embed, attn, ffn, head).
  - [ ] Toggle `tied_lm_head` and ensure correct sharing with embeddings.
  - [ ] Unit tests: shapes, mask correctness, forward pass with tiny model.

- [ ] **Milestone 3 — Training harness**
  - [ ] Implement `src/train.py` (AMP, grad clipping, cosine decay, warmup).
  - [ ] Implement LR range test utility (`lr_find.py`) and expose CLI flag.
  - [ ] Add 1k-token overfit sanity mode (`--overfit_tokens 1024`).
  - [ ] Checkpointing & auto-resume (every N steps/minutes).
  - [ ] Determinism utilities (seed, cudnn flags) and reproducible dataloaders.
  - [ ] CLI entry: `run_experiment.py` to load YAML, spawn run dirs, log artifacts.

- [ ] **Milestone 4 — Evaluation & telemetry**
  - [ ] Implement dev/test perplexity evaluation at fixed intervals.
  - [ ] Implement `eval_smallbench.py` (100–500 prompts per task → accuracy).
  - [ ] Add latency/throughput profiler (batch=1; seq=128/512) + peak memory log.
  - [ ] Standardize `metrics.json` schema; write per-run summary.

- [ ] **Milestone 5 — Phase 1: short “ranking” runs (≤5M tokens)**
  - [ ] Generate configs for embedding share sweep: `{0.25, 0.35, 0.45}` @ `V=16k`.
  - [ ] Generate configs for GLU expansion sweep: `{2.0, 2.66, 3.0, 4.0}`.
  - [ ] Implement **kill rule**: stop if dev PPL ≥0.5 worse than baseline for 1M tokens.
  - [ ] Run all variants to 5M tokens; log PPL slope curves.
  - [ ] Aggregate early results to CSV; rank variants; select top-1 per axis.

- [ ] **Milestone 6 — Micro-ablations (cheap)**
  - [ ] Add vocab sweep: `V={8k,16k}` with/without `tied_lm_head`.
  - [ ] Run short ranking passes; record any shift in optimal embedding share.
  - [ ] Update decision notes based on ablation results.

- [ ] **Milestone 7 — Phase 2: finalists (full token budget)**
  - [ ] Promote top 1–2 configs; train to full budget with **3 seeds**.
  - [ ] Record final PPL, SmallBench accuracy, latency, memory.
  - [ ] Export checkpoints + logs; verify reproducibility with seeds.

- [ ] **Milestone 8 — Analysis & visualization**
  - [ ] Implement `scripts/aggregate_results.py` → `results_summary.csv`.
  - [ ] Plot PPL vs. embedding_ratio and PPL vs. GLU expansion.
  - [ ] Plot compute/latency vs. perplexity Pareto chart.
  - [ ] Write `reports/RESULTS.md` (methods, tables, charts).

- [ ] **Milestone 9 — Cost tracking & infra**
  - [ ] Add spot-instance launch script with auto-termination & checkpoint restore.
  - [ ] Log GPU-hours and \$ cost per run; export `COST_SUMMARY.json`.
  - [ ] Add dashboard snippet in `README` for cost + progress badges.

- [ ] **Milestone 10 — Paper (Idea draft integration)**
  - [ ] Create 1-page summary for the arXiv “idea” draft (setup, best config, results).
  - [ ] Export publication-ready figures (300 dpi PNG/SVG).
  - [ ] Draft limitations & future work section (what we didn’t tune).

- [ ] **Milestone 11 — Repro & release**
  - [ ] Bundle reproducibility artifacts (config YAMLs, tokenizer hash, git SHA, seeds).
  - [ ] Tag release (v0.1.0) and attach artifacts.
  - [ ] Publish `MODEL_CARD.md` and short “how to run” guide.
  - [ ] Update machineresearch.xyz page stub with results summary.

---

## Quick commands (for convenience)

```bash
# Setup
make setup

# Data
python scripts/download_wikitext.py --output data/wikitext-2
python scripts/preprocess_lm.py --input data/wikitext-2 --vocab_size 16000 --output data/lm_tokenized
python scripts/download_smallbench.py --output data/smallbench

# Embedding sweep (parallel)
for cfg in emb35 emb25 emb45; do
  python run_experiment.py --config configs/embed_ratio.yaml --exp $cfg &
done; wait

# GLU sweep (parallel)
for cfg in glu2x glu266x glu3x glu4x; do
  python run_experiment.py --config configs/glu_expansion.yaml --exp $cfg &
done; wait

# Aggregate results
python scripts/aggregate_results.py --log_dir logs --output results_summary.csv
```

---

## Notes
- Promote top **2** configs to finalists if early margins are thin; break ties by latency.
- Keep SmallBench fully held-out from pretraining data to avoid leakage.
- Aim to keep sweep cost **< $50**; full 10M runbook **< $150**.
