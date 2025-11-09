# Validation Scripts

Scripts to verify data quality and evaluation correctness before Phase 2.

## ⚠️ Why These Scripts Are Critical

Your Phase 1 perplexity of **1.054 is unrealistically low**. For reference:
- **Perfect perplexity = 1.0** (impossible in practice)
- **Expected for 10M model on WikiText-2: 80-200+**
- **Your result: 1.054** ← This indicates a serious problem!

These scripts help diagnose three common causes:
1. **Tiny validation set** (< 1000 tokens) that the model memorized
2. **Data leakage** (validation data contaminated the training set)
3. **Bug in perplexity calculation** (wrong formula or log base)

## Scripts Overview

### 1. `verify_perplexity.py`
Verifies that perplexity is calculated correctly.

**Checks:**
- ✓ Correct formula: `PPL = exp(cross_entropy_loss)`
- ✓ Natural log (not log2 or log10)
- ✓ Loss averaged over tokens (not batches)
- ✓ Proper handling of padding tokens
- ✓ Comparison with random model baseline

**Usage:**
```bash
python scripts/verify_perplexity.py \
    --checkpoint path/to/checkpoint.pt \
    --data path/to/val_data \
    --vocab-size 16000
```

**What to look for:**
- Perplexity should be ≥ 1.0 (mathematical requirement)
- For 10M model: expect PPL of 80-200+
- Random model should give PPL ≈ vocab_size
- If PPL ≈ 1, likely a tiny val set or data leakage

---

### 2. `check_data_leakage.py`
Detects data contamination between train and validation sets.

**Checks:**
- ✓ No exact duplicates
- ✓ N-gram overlap (< 90% is healthy)
- ✓ No file-level contamination
- ✓ Proper sequential splitting
- ✓ Token distribution comparison

**Usage:**
```bash
python scripts/check_data_leakage.py \
    --train-data path/to/train_data \
    --val-data path/to/val_data \
    --ngram-size 8
```

**What to look for:**
- 0 exact duplicates (critical!)
- N-gram overlap < 90% (some overlap is normal for language)
- Sequential splitting (not random shuffling)
- Similar but not identical token distributions

---

### 3. `analyze_validation_set.py`
Analyzes validation set size, diversity, and quality.

**Checks:**
- ✓ Size (tokens and sequences)
- ✓ Diversity (entropy, vocabulary coverage)
- ✓ Repetition (repeated sequences)
- ✓ Difficulty (sequence lengths)
- ✓ Comparison with training set

**Usage:**
```bash
python scripts/analyze_validation_set.py \
    --train-data path/to/train_data \
    --val-data path/to/val_data
```

**What to look for:**
- Val set size: **minimum 50k tokens**, ideally 100k-500k
- Val ratio: 5-10% of total data
- Low repetition: < 10% repeated sequences
- Entropy similar to training set
- WikiText-2 standard val: ~200k tokens

**Red flags:**
- < 1,000 tokens → explains PPL of 1.054!
- > 30% repetition → easy to memorize
- Very low entropy → not diverse enough

---

### 4. `run_all_validations.py`
Master script that runs all validation checks.

**Usage:**
```bash
python scripts/run_all_validations.py \
    --checkpoint path/to/checkpoint.pt \
    --train-data path/to/train_data \
    --val-data path/to/val_data \
    --output validation_report.json
```

---

## Quick Start

### Step 1: Check Validation Set Size
This is the most likely culprit for PPL of 1.054.

```python
# Quick check (adapt to your data format)
import json

# For text files
with open('path/to/val_data.txt', 'r') as f:
    text = f.read()
    tokens = text.split()
    print(f"Validation tokens: {len(tokens):,}")

# For tokenized files
# tokens = json.load(open('path/to/val_data.json'))
# print(f"Validation tokens: {len(tokens):,}")
```

**If < 50,000 tokens:** This is your problem! Create a larger val set.

---

### Step 2: Check for Duplicates

```python
# Quick duplicate check
train_lines = set(open('path/to/train.txt').readlines())
val_lines = set(open('path/to/val.txt').readlines())

duplicates = train_lines & val_lines
print(f"Duplicates: {len(duplicates)}")

if duplicates:
    print("❌ Data leakage detected!")
else:
    print("✓ No exact duplicates")
```

---

### Step 3: Verify Perplexity Calculation

```python
import torch
import torch.nn.functional as F
import math

# Your evaluation code should look like this:
def calculate_perplexity(model, dataloader):
    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']

            # Forward pass
            outputs = model(input_ids)
            logits = outputs.logits

            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            # Calculate loss (sum over tokens)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'  # Important: use sum, then divide by total
            )

            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    # Perplexity = exp(average loss)
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)  # Use natural log (math.exp)

    return ppl

# Common bugs:
# ❌ ppl = 2 ** avg_loss  # Wrong base!
# ❌ ppl = 10 ** avg_loss  # Wrong base!
# ❌ avg_loss = total_loss / num_batches  # Should be total_tokens!
# ✓ ppl = math.exp(total_loss / total_tokens)  # Correct!
```

---

## Expected Results

### Healthy Validation Setup
```
✓ Validation set size: 100,000+ tokens
✓ Val/train ratio: 5-10%
✓ No exact duplicates: 0
✓ N-gram overlap: 40-80% (normal for language)
✓ Repetition: < 10%
✓ Perplexity: 80-200+ for 10M model on WikiText-2
```

### Problematic Setup (explains PPL of 1.054)
```
❌ Validation set size: 500 tokens (TOO SMALL!)
❌ Val/train ratio: 0.1% (TOO SMALL!)
❌ High repetition: 40% repeated sequences
❌ Perplexity: 1.054 (unrealistic)
```

---

## What to Do if PPL is Still ~1

If you've verified everything and PPL is still unrealistically low:

1. **Print validation set size:**
   ```python
   print(f"Val tokens: {len(val_tokens)}")
   print(f"Val sequences: {len(val_sequences)}")
   ```

2. **Print actual loss during evaluation:**
   ```python
   print(f"Average loss: {avg_loss}")
   print(f"Perplexity: {math.exp(avg_loss)}")
   ```

3. **Test with random predictions:**
   ```python
   # Random uniform predictions should give PPL ≈ vocab_size
   random_ppl = vocab_size  # e.g., 16000
   print(f"Expected random PPL: {random_ppl}")
   print(f"Your PPL: {your_ppl}")
   ```

4. **Use standard WikiText-2 splits:**
   Don't create custom splits. Use the standard ones:
   ```python
   from datasets import load_dataset
   wikitext = load_dataset("wikitext", "wikitext-2-raw-v1")
   train = wikitext['train']
   val = wikitext['validation']  # ~200k tokens
   ```

---

## Recommendations for Phase 2

**DO NOT proceed to Phase 2 until:**
- ✓ Validation set has 50k+ tokens (ideally 100k-500k)
- ✓ No data leakage detected
- ✓ Perplexity calculation verified
- ✓ PPL is realistic (80-200+ for 10M model)

**If PPL is truly near-perfect (~1):**
- This means your model has essentially memorized the validation set
- The evaluation is unreliable
- Phase 2 results will also be unreliable
- You must fix the validation setup first

---

## Contact / Issues

If you're stuck, check:
1. How many tokens are in your validation set?
2. Did you use the standard WikiText-2 validation split?
3. Print the actual cross-entropy loss value
4. Is the loss calculation using natural log?

Most commonly, the issue is a validation set that's too small (< 10k tokens).
