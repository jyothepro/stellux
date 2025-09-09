# PRD: Parameter Allocation Experiments for 10M-Parameter LMs

## 0) Why we’re doing this

Small LMs (≤100M) live on tight parameter/compute budgets. Two levers dominate quality-per-parameter: (a) how many parameters we park in **embeddings** vs the **backbone**, and (b) how wide we make the **FFN/GLU** layers. We will systematically sweep those levers at 10M params to find a compute-efficient recipe we can freeze for our paper’s “idea” phase and reuse at 100M.

Concretely, we will:
- Sweep embedding parameter share (25%/35%/45%) at fixed total params and vocab.
- Sweep GLU expansion factors (2.0/2.66/3.0/4.0) at fixed total params. The 2.66× target reflects our SLIM design guidance for small models.

Our evaluation uses **perplexity** (intrinsic) and a tiny **SmallBench** probe (extrinsic). A run “wins” if it beats baseline perplexity by ≥5% at equal or lower latency/memory.

---

## 1) Goals & non-goals

**Goals**
- Identify an embedding share and GLU factor that are **Pareto-efficient** (loss vs. speed/memory) at 10M params.  
- Produce a reproducible harness (configs + scripts + plots) and a one-pager of results to drop into the arXiv “idea” draft.  
- Keep total spend **< $50** for sweeps, and **< $150** including base pretrain + light instruction-tuning.

**Non-goals**
- We are **not** tuning attention variants, context length, or sparsity here; those belong in later iterations.

---

## 2) Datasets & metrics

**Pretrain corpus:** ~50M tokens (e.g., WikiText-2 or WMT news subset). Tokenize to vocab size 16k.

**Eval sets:** SmallBench (classification, NLI, etc.) as a tiny capability probe; *do not* contaminate pretraining with these examples.

**Primary metric:** **Perplexity** (lower is better). We also track SmallBench accuracy on ~100–500 prompts per task for a sanity check.

---

## 3) Hypotheses

1) **Embedding allocation**: 30–40% of params parked in embeddings is optimal for 10M with V=16k; we will test 25% / 35% / 45%.
2) **GLU expansion**: A **2.66×** FFN/GLU width is competitive with 4× at small scale, reducing params and memory without hurting quality. We’ll compare 2.0/2.66/3.0/4.0.

---

## 4) Scope of experiments (and early-exit plan)

We’ll run **short “ranking” runs** (≤5M tokens) for every variant and fully train only the top performers:

- **Phase 0 – Pre-flight sanity (single config):** optimizer sanity (memorize 1k tokens), LR range test, parameter accounting per block.
- **Phase 1 – Short-token ranking:**  
  - Embedding share: {25, 35, 45}% @ V=16k (pick top-1).  
  - GLU factor: {2.0, 2.66, 3.0, 4.0} on the chosen embed config (pick top-1–2).  
  - **Kill rule:** stop a run if dev PPL is ≥0.5 worse than baseline for 1M consecutive tokens.
- **Phase 1b – Micro-ablation (cheap):** Vocab 8k vs 16k + tied head toggle to check if it shifts optimal embedding share.
- **Phase 2 – Finish finalists:** full token budget; 3 seeds; measure PPL, SmallBench accuracy, latency (b=1 at L=128/512) and peak memory.

---

## 5) Environment, setup, and repo layout

**Spot GPUs** (V100/A100) with auto-termination; install PyTorch/Transformers/Datasets; clone the repo skeleton.

```
parameter-allocation-experiments/
├─ configs/
│  ├─ embed_ratio.yaml
│  └─ glu_expansion.yaml
├─ data/
├─ scripts/
│  ├─ download_wikitext.py
│  ├─ preprocess_lm.py
│  ├─ download_smallbench.py
│  └─ aggregate_results.py
├─ src/
│  ├─ models/ (LM definition; knobs: embedding_ratio, glu_expansion)
│  ├─ train.py
│  └─ eval_smallbench.py
└─ run_experiment.py
```

**Data prep commands** (LM + SmallBench):
```bash
python scripts/download_wikitext.py --output data/wikitext-2
python scripts/preprocess_lm.py --input data/wikitext-2 --vocab_size 16000 --output data/lm_tokenized
python scripts/download_smallbench.py --output data/smallbench
```

---

## 6) Experiment configs & how to launch

### 6.1 Embedding-ratio sweep (10M params; V=16k)
Config skeleton and launch loop are below; run in parallel on spot instances.
```yaml
# configs/embed_ratio.yaml
model:
  total_params: 10_000_000
  vocab_size: 16_000
experiments:
  - name: emb35
    embedding_ratio: 0.35
  - name: emb25
    embedding_ratio: 0.25
  - name: emb45
    embedding_ratio: 0.45
training:
  tokens: 50_000_000
  batch_size: 256
evaluation:
  metrics: ["perplexity", "smallbench_accuracy"]
```

```bash
for cfg in emb35 emb25 emb45; do
  python run_experiment.py --config configs/embed_ratio.yaml --exp $cfg &
done
wait
```

### 6.2 GLU-expansion sweep (same total params)
Config and launcher:
```yaml
# configs/glu_expansion.yaml
model:
  total_params: 10_000_000
  vocab_size: 16_000
experiments:
  - name: glu2x
    glu_expansion: 2.0
  - name: glu266x
    glu_expansion: 2.66
  - name: glu3x
    glu_expansion: 3.0
  - name: glu4x
    glu_expansion: 4.0
training:
  tokens: 50_000_000
  batch_size: 256
evaluation:
  metrics: ["perplexity", "smallbench_accuracy"]
```

```bash
for cfg in glu2x glu266x glu3x glu4x; do
  python run_experiment.py --config configs/glu_expansion.yaml --exp $cfg &
done
wait
```

**Note (intern):** Model code must honor `embedding_ratio` by allocating V×d and adjusting the backbone to keep `total_params` ≈10M. `glu_expansion` scales the FFN inner width (e.g., SwiGLU) accordingly. (SLIM architecture favors ~2.66× at small scale.)

---

## 7) Training, logging, and artifacts

- Log **train loss & perplexity**, **GPU utilization/memory**, and **eval metrics** to TensorBoard or W&B. Use separate log dirs per run (e.g., `logs/emb35`, `logs/glu266x`).  
- Save for every run: `config.yaml`, tokenizer hash, git SHA, seed, final checkpoint, and `metrics.json`.  
- For finalists only, run **3 seeds** and log mean±stdev.

---

## 8) Evaluation & analysis

**Aggregate & plot**: one script to gather logs and write a CSV; then plot PPL vs. embedding_ratio and PPL vs. glu_expansion.
```bash
python scripts/aggregate_results.py --log_dir logs --output results_summary.csv
```

**Decision criteria** (gate to “idea” paper table):  
- Winner must beat baseline PPL by **≥5%** and not regress latency/memory.  
- If two configs tie on PPL, pick the **faster** one (batch=1, seq=128/512).

---

## 9) Costing & infrastructure

Spot pricing + hours put the experiment phase **under $50**, and the whole 10M runbook **under $150** including base pretrain + light instruction-tuning. (Keep an eye on spot interruptions; use checkpointing.)

---

## 10) Step-by-step “Do this” (for the intern)

1) **Provision** one spot GPU (V100/A100) and install deps; clone repo.  
2) **Prepare data** (LM tokens + SmallBench): run the three data scripts.  
3) **Phase 0 sanity**:  
   - Overfit 1k tokens to confirm optimizer/wiring.  
   - LR range test (short warmup sweep) and pick LR.  
   - Print parameter counts by block to verify `embedding_ratio` math.  
4) **Phase 1 ranking**: run all embed/GLU variants to **5M tokens**, track **dev PPL slope**; apply **kill rule** to bad runs.  
5) **Phase 1b micro-ablation**: quick 8k vs 16k vocab + tied head toggle on the leading embed config (cheap pass).  
6) **Phase 2 finalists**: full-budget training for the top 1–2 configs; **3 seeds**; log latency & memory.  
7) **Aggregate & plot** results; produce `results_summary.csv` + two charts.  
8) **Write up** one-pager with: setup, best config, PPL, SmallBench, speed/memory, and a short discussion.

---

## 11) Deliverables

- `configs/embed_ratio.yaml`, `configs/glu_expansion.yaml` (checked in).  
- `results_summary.csv` + two PNG plots (PPL vs. ratio, PPL vs. GLU).  
- `REPORT.md` (1 page): best setting, metrics table, charts, and cost.  
- Repro bundle (config, tokenizer hash, git SHA, seed list).

---

## 12) Risks & mitigations

- **Parameter drift** (total params vary across configs) → assert counts per block; unit test the model builder.  
- **Spot interruptions** → save checkpoints every ~15 minutes; use auto-resume.  
- **Eval leakage** → keep SmallBench completely held-out from pretraining.  
- **Noisy short-run ranking** → promote **top 2** if margins are thin; confirm with full runs.

---

## 13) Appendix: reference commands (for convenience)

- **Embedding sweep launcher** (parallel):
```bash
for cfg in emb35 emb25 emb45; do
  python run_experiment.py --config configs/embed_ratio.yaml --exp $cfg &
done
wait
```

- **GLU sweep launcher** (parallel):
```bash
for cfg in glu2x glu266x glu3x glu4x; do
  python run_experiment.py --config configs/glu_expansion.yaml --exp $cfg &
done
wait
```
