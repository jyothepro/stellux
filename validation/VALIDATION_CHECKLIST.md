# Pre-Phase 2 Validation Checklist

## ⚠️ Critical Issue: PPL of 1.054 is Unrealistic

Your Phase 1 perplexity of **1.054** is unrealistically low and indicates a serious problem.

**Expected perplexity for 10M model on WikiText-2: 80-200+**

This checklist will help you diagnose and fix the issue.

---

## Quick Diagnosis (Start Here)

Run this on your Mac/Lambda box where you have the Phase 1 data:

```bash
cd validation/parameter-allocation-experiments

# Quick check (no model loading required)
python scripts/quick_check.py \
    --train path/to/your/train_data.txt \
    --val path/to/your/val_data.txt
```

This will immediately tell you:
- ✓ Validation set size (most common issue)
- ✓ Exact duplicates between train/val
- ✓ Basic statistics

---

## Three Critical Checks

### ✅ Check 1: Validation Set Size

**Most likely culprit for PPL of 1.054**

```python
# Quick manual check
with open('path/to/val_data.txt', 'r') as f:
    text = f.read()
    tokens = text.split()
    print(f"Validation tokens: {len(tokens):,}")
```

**Requirements:**
- Minimum: 50,000 tokens
- Recommended: 100,000-500,000 tokens
- WikiText-2 standard: ~200,000 tokens

**If < 50,000 tokens:**
- ❌ Your val set is too small
- ❌ Model likely memorized it
- ❌ Explains PPL of 1.054
- ✅ Solution: Use larger val set or standard WikiText-2 split

---

### ✅ Check 2: Data Leakage

**Run comprehensive check:**

```bash
python scripts/check_data_leakage.py \
    --train-data path/to/train \
    --val-data path/to/val \
    --ngram-size 8
```

**What to look for:**
- ✓ 0 exact duplicates (critical!)
- ✓ N-gram overlap < 90%
- ✓ Sequential split (not random)

**If duplicates found:**
- ❌ Data leakage detected
- ❌ Could explain low PPL
- ✅ Solution: Re-create train/val split properly

---

### ✅ Check 3: Perplexity Calculation

**Common bugs in PPL calculation:**

```python
# ❌ WRONG - Using wrong log base
ppl = 2 ** avg_loss   # Wrong!
ppl = 10 ** avg_loss  # Wrong!

# ❌ WRONG - Averaging over batches instead of tokens
avg_loss = total_loss / num_batches  # Wrong!

# ✓ CORRECT - Natural log, average over tokens
avg_loss = total_loss / total_tokens
ppl = math.exp(avg_loss)  # Correct!
```

**Verify your calculation:**

```bash
python scripts/verify_perplexity.py \
    --checkpoint path/to/checkpoint.pt \
    --data path/to/val_data \
    --vocab-size 16000
```

---

## Validation Scripts Reference

All scripts are in `validation/parameter-allocation-experiments/scripts/`

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `quick_check.py` | Fast basic checks | Start here - no model needed |
| `verify_perplexity.py` | Verify PPL calculation | If you suspect calculation bug |
| `check_data_leakage.py` | Detect train/val contamination | Check for data leakage |
| `analyze_validation_set.py` | Comprehensive val analysis | Deep dive into val set quality |
| `run_all_validations.py` | Run all checks | Complete validation |

---

## Step-by-Step Action Plan

### Step 1: Quick Diagnosis (5 minutes)
```bash
python scripts/quick_check.py \
    --train path/to/train \
    --val path/to/val
```

Look for:
- Val set size (should be 50k+ tokens)
- Exact duplicates (should be 0)

### Step 2: Full Validation (15 minutes)

If quick check shows issues, run full validation:

```bash
# Data leakage check
python scripts/check_data_leakage.py \
    --train-data path/to/train \
    --val-data path/to/val

# Validation set analysis
python scripts/analyze_validation_set.py \
    --train-data path/to/train \
    --val-data path/to/val

# Perplexity verification (if checkpoint available)
python scripts/verify_perplexity.py \
    --checkpoint path/to/checkpoint.pt \
    --data path/to/val
```

### Step 3: Fix Issues

**If val set is too small:**
```python
# Use standard WikiText-2 split
from datasets import load_dataset

wikitext = load_dataset("wikitext", "wikitext-2-raw-v1")
train_data = wikitext['train']
val_data = wikitext['validation']  # ~200k tokens

# Or create larger custom split (5-10% of data)
total_tokens = 50_000_000
val_tokens = 5_000_000  # 10%
```

**If data leakage detected:**
```python
# Use sequential split (not random)
# Example: first 90% = train, last 10% = val
split_point = int(len(data) * 0.9)
train_data = data[:split_point]
val_data = data[split_point:]

# Don't shuffle after splitting!
```

**If PPL calculation is wrong:**
```python
# Correct formula
def calculate_perplexity(model, dataloader):
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            reduction='sum'  # Use sum
        )
        total_loss += loss.item()
        total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens  # Average over tokens
    ppl = math.exp(avg_loss)  # Natural log
    return ppl
```

### Step 4: Re-run Evaluation

After fixing issues:
1. Re-run Phase 1 evaluation
2. Verify PPL is realistic (80-200+)
3. Document the fix in logs
4. Update Phase 1 results

### Step 5: Verify Before Phase 2

**DO NOT proceed to Phase 2 until:**
- [ ] Validation set has 50k+ tokens
- [ ] No data leakage detected
- [ ] PPL calculation verified correct
- [ ] PPL is realistic (80-200+)
- [ ] Re-ran evaluation with fixes

---

## Expected Results

### Healthy Setup
```
✓ Validation tokens: 200,000
✓ Val ratio: 8% of total data
✓ No exact duplicates: 0
✓ N-gram overlap: 65% (normal)
✓ Perplexity: 120 (realistic for 10M model)
```

### Problematic Setup (explains PPL of 1.054)
```
❌ Validation tokens: 500 (TOO SMALL!)
❌ Val ratio: 0.1% (TOO SMALL!)
❌ Exact duplicates: 15
❌ N-gram overlap: 95% (LEAKAGE!)
❌ Perplexity: 1.054 (unrealistic)
```

---

## What If PPL is Still ~1?

If you've verified everything and PPL is still near 1:

1. **Print debug info during evaluation:**
```python
print(f"Validation tokens: {total_tokens:,}")
print(f"Total loss: {total_loss:.4f}")
print(f"Average loss: {total_loss/total_tokens:.4f}")
print(f"Perplexity: {math.exp(total_loss/total_tokens):.4f}")
```

2. **Test with random predictions:**
```python
# Random model should give PPL ≈ vocab_size
random_ppl = vocab_size  # e.g., 16000
print(f"Expected random PPL: {random_ppl}")
```

3. **Compare with baseline:**
- Load a random untrained model
- Evaluate on same val set
- Should get PPL close to vocab_size
- If not, PPL calculation is wrong

4. **Use standard library:**
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# Use standard evaluation from transformers
# Compare with your implementation
```

---

## Common Root Causes

Ranked by likelihood:

1. **Tiny validation set (< 1k tokens)** - 70% of cases
2. **Data leakage (val in train)** - 20% of cases
3. **Wrong PPL formula (log base)** - 5% of cases
4. **Averaging over batches not tokens** - 3% of cases
5. **Other calculation bugs** - 2% of cases

---

## Next Steps After Validation

Once validation checks pass:

1. ✅ Fix any issues found
2. ✅ Re-run Phase 1 evaluation
3. ✅ Verify PPL is realistic (80-200+)
4. ✅ Document changes made
5. ✅ Proceed to Phase 2 preparation

See the main README for Phase 2 next steps.

---

## Need Help?

If stuck, share:
1. Output of `quick_check.py`
2. Number of tokens in val set
3. PPL value and how it's calculated
4. Sample of evaluation code

Most issues are solved by using a larger validation set or fixing data leakage.
