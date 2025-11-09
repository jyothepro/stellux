import json
import re

with open('results/phase1/all_results_20251028_203855.json', 'r') as f:
    data = json.load(f)

print("\n" + "="*80)
print("PHASE 1 RANKING RESULTS")
print("="*80)

emb_results = []
glu_results = []

for exp in data:
    name = exp['exp_name']
    status = exp['status']
    
    if status != 'success':
        continue
    
    stderr = exp.get('stderr', '')
    for line in stderr.split('\n'):
        if 'Final metrics:' in line and 'eval_perplexity' in line:
            match = re.search(r"'eval_perplexity':\s*([\d.]+)", line)
            if match:
                ppl = float(match.group(1))
                if 'emb' in name:
                    emb_results.append((name, ppl))
                elif 'glu' in name:
                    glu_results.append((name, ppl))
                break

print("\n📊 EMBEDDING RATIO RESULTS:")
emb_results.sort(key=lambda x: x[1])
for i, (name, ppl) in enumerate(emb_results, 1):
    marker = "🏆" if i == 1 else "  "
    print(f"{marker} {name:20s}  PPL: {ppl:.6f}")

print("\n📊 GLU EXPANSION RESULTS:")
glu_results.sort(key=lambda x: x[1])
for i, (name, ppl) in enumerate(glu_results, 1):
    marker = "🏆" if i == 1 else "  "
    print(f"{marker} {name:20s}  PPL: {ppl:.6f}")

print("\n" + "="*80)
print("WINNERS:")
print("="*80)
if emb_results:
    winner_emb = min(emb_results, key=lambda x: x[1])
    print(f"🏆 Best Embedding Ratio: {winner_emb[0]} (PPL: {winner_emb[1]:.6f})")
    
if glu_results:
    winner_glu = min(glu_results, key=lambda x: x[1])
    print(f"🏆 Best GLU Expansion:   {winner_glu[0]} (PPL: {winner_glu[1]:.6f})")
print("="*80 + "\n")
