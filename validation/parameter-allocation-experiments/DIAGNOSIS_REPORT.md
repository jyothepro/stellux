# Phase 1 Results Diagnosis Report

**Date:** 2025-11-08
**Issue:** PPL of ~1.054 is unrealistically low
**Expected PPL:** 80-200+ for a 10M parameter model on WikiText-2

---

## Executive Summary

✅ **PPL Calculation:** Mathematically correct (`PPL = exp(loss)`)
✅ **Validation Set Size:** Appears adequate (~30M tokens evaluated)
❌ **ROOT CAUSE:** Cross-entropy loss is ~0.053, which is **unrealistically low**

**The Problem:** The cross-entropy loss itself is wrong, not the perplexity formula.

---

## Detailed Findings

### 1. Perplexity Calculation ✅ CORRECT

```python
# All experiments use correct formula:
eval_loss = 0.053
eval_perplexity = exp(0.053) = 1.054
```

**Verification:**
- phase1_emb25: loss=0.0556, PPL=1.0571 ✓
- phase1_emb35: loss=0.0530, PPL=1.0544 ✓
- phase1_glu3x: loss=0.0520, PPL=1.0534 ✓

**Math checks out perfectly.**

---

### 2. Validation Set Size ✅ ADEQUATE

```
Evaluation batches: 118
Batch size: 512
Sequence length: ~512
Total tokens: ~30,000,000
```

**This is plenty of data** - not the issue.

---

### 3. Cross-Entropy Loss ❌ **UNREALISTICALLY LOW**

```
Observed loss: ~0.053
Expected for random predictions: ln(16000) ≈ 9.68
Your model is 175x better than random!
```

**This is impossible for a 10M parameter model on WikiText-2.**

---

## Why is the Loss So Low?

There are only a few possible explanations:

### Hypothesis 1: **Wrong Loss Function** (Most Likely)

**Possible bugs:**

#### A. Not Using Cross-Entropy
```python
# ❌ WRONG
loss = mse_loss(logits, labels)  # Mean squared error

# ✓ CORRECT
loss = F.cross_entropy(logits, labels)
```

#### B. Ignoring Most Tokens
```python
# ❌ WRONG - Only counting non-padding tokens incorrectly
loss = F.cross_entropy(logits, labels, ignore_index=-100)
# But if labels are mostly -100, you're averaging over very few tokens

# ✓ CORRECT
loss = F.cross_entropy(logits, labels, reduction='sum')
total_loss / total_tokens  # Average over ALL tokens
```

#### C. Dividing by Wrong Denominator
```python
# ❌ WRONG
avg_loss = total_loss / (batch_size * seq_length * vocab_size)  # Too big!

# ✓ CORRECT
avg_loss = total_loss / (batch_size * seq_length)  # Tokens only
```

---

### Hypothesis 2: **Data Leakage** (Less Likely)

Even with validation data in training, you wouldn't get loss this low unless:
- You trained for many epochs on the same tiny val set
- The model completely memorized it

---

### Hypothesis 3: **Wrong Labels/Logits Shape**

```python
# ❌ WRONG - Comparing wrong tokens
loss = F.cross_entropy(logits, input_ids)  # Should be shifted!

# ✓ CORRECT - Next token prediction
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = input_ids[..., 1:].contiguous()
loss = F.cross_entropy(shift_logits.view(-1, vocab_size),
                        shift_labels.view(-1))
```

---

## How to Diagnose Further

### Step 1: Check Your Evaluation Code

Look for the evaluation loop in your training code (likely in the Lambda GPU box).

**Key questions:**
1. What loss function are you using?
2. Are you shifting logits/labels for next-token prediction?
3. How are you averaging the loss?
4. Are you handling padding tokens correctly?

### Step 2: Add Debug Prints

```python
# In your evaluation loop, add:
print(f"Logits shape: {logits.shape}")  # Should be [batch, seq_len, vocab_size]
print(f"Labels shape: {labels.shape}")  # Should be [batch, seq_len]
print(f"Loss (raw): {loss.item()}")
print(f"Loss (mean): {loss.mean().item()}")
print(f"Total tokens: {total_tokens}")
print(f"Average loss: {total_loss / total_tokens}")

# Check a single prediction
print(f"Sample logit max: {logits[0, 0].max().item()}")
print(f"Sample label: {labels[0, 0].item()}")
print(f"Vocab size: {logits.shape[-1]}")
```

### Step 3: Test with Random Model

```python
# Initialize a random untrained model
random_model = LanguageModel(config)
# DON'T load checkpoint!

# Evaluate on same val set
random_ppl = evaluate(random_model, val_loader)
print(f"Random model PPL: {random_ppl}")
# Should be close to vocab_size (16000)
```

**If random PPL is also ~1.0, your loss calculation is definitely wrong!**

---

## Most Likely Fix

Based on typical bugs, the issue is probably in how the loss is calculated during evaluation.

### Check your evaluation code for this pattern:

```python
# ❌ COMMON BUG - Wrong reduction or averaging
def evaluate(model, dataloader):
    total_loss = 0
    total_batches = 0  # ❌ Should be total_tokens!

    for batch in dataloader:
        loss = model(batch['input_ids']).loss
        total_loss += loss.item()
        total_batches += 1  # ❌ WRONG!

    avg_loss = total_loss / total_batches  # ❌ WRONG!
    return math.exp(avg_loss)
```

### Should be:

```python
# ✓ CORRECT
def evaluate(model, dataloader):
    total_loss = 0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch['input_ids']
        logits = model(input_ids)

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Calculate loss (use 'sum' reduction)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum'  # Important!
        )

        total_loss += loss.item()
        total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens  # Average over tokens
    return math.exp(avg_loss)
```

---

## Expected Values

After fixing the loss calculation, you should see:

```
Random model (untrained):
  Loss: ~9.68 (ln(16000))
  PPL: ~16,000

Trained 10M model (short training):
  Loss: ~4.5 - 5.5
  PPL: ~90 - 250

Trained 10M model (full training):
  Loss: ~4.0 - 5.0
  PPL: ~55 - 150

Your current results:
  Loss: ~0.053 ❌
  PPL: ~1.054 ❌
```

---

## Action Items

### Before Phase 2:

1. **Find your evaluation code** on the Lambda GPU box
   - Look in `src/train.py` or `src/eval/evaluate.py`
   - Search for `cross_entropy` or `nll_loss`

2. **Check the loss calculation**
   - Is it using `F.cross_entropy`?
   - Is `reduction='sum'` or `'mean'`?
   - Are you dividing by total tokens?

3. **Add debug logging**
   - Print shapes of logits and labels
   - Print loss values at each step
   - Verify total_tokens count

4. **Test with random model**
   - Should get PPL ≈ 16,000
   - If not, loss calculation is wrong

5. **Re-run evaluation** with fixed code
   - Should get PPL of 80-200+
   - If still ~1, investigate further

---

## Questions to Answer

1. **Where is your evaluation code?**
   - What file contains the evaluation loop?
   - Can you share the code?

2. **What does your loss calculation look like?**
   ```python
   # Share this part of your code
   loss = F.cross_entropy(...)
   ```

3. **Did you use a pre-built training script?**
   - HuggingFace Trainer?
   - Custom training loop?

---

## Conclusion

**The PPL calculation is correct, but the underlying cross-entropy loss is wrong.**

Most likely causes (in order of probability):
1. Wrong averaging (dividing by batches instead of tokens) - 60%
2. Wrong loss function or shape mismatch - 25%
3. Ignoring too many tokens with ignore_index - 10%
4. Data leakage or other issue - 5%

**Next step:** Find and inspect your evaluation code on the Lambda GPU box.

---

## Need Help?

If you can share your evaluation code, I can pinpoint the exact issue. Look for files like:
- `src/train.py`
- `src/eval/evaluate.py`
- `scripts/run_training.py`

Or any file that contains your training/evaluation loop.
