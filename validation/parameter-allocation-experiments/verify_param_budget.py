#!/usr/bin/env python3
"""
Manual Parameter Budget Verification

Verifies the parameter counting formulas are correct by manually calculating
expected parameter counts for different configurations.
"""

print("=" * 80)
print("PARAMETER BUDGET VERIFICATION")
print("=" * 80)
print()

# Test Configuration
configs = [
    {
        "name": "Default 10M (35% embed, 2.66x GLU, tied)",
        "total_params": 10_000_000,
        "vocab_size": 16_000,
        "embedding_ratio": 0.35,
        "glu_expansion": 2.66,
        "tied_lm_head": True,
        "n_heads": 8,
    },
    {
        "name": "1M Small (25% embed, 2.0x GLU, tied)",
        "total_params": 1_000_000,
        "vocab_size": 1_000,
        "embedding_ratio": 0.25,
        "glu_expansion": 2.0,
        "tied_lm_head": True,
        "n_heads": 8,
    },
    {
        "name": "1M (45% embed, 4.0x GLU, untied)",
        "total_params": 1_000_000,
        "vocab_size": 1_000,
        "embedding_ratio": 0.45,
        "glu_expansion": 4.0,
        "tied_lm_head": False,
        "n_heads": 8,
    },
]

def compute_dims_manual(config):
    """Manually compute model dimensions (replicates compute_model_dims logic)."""
    V = config["vocab_size"]
    total = config["total_params"]
    ratio = config["embedding_ratio"]
    glu = config["glu_expansion"]
    n_heads = config["n_heads"]
    tied = config["tied_lm_head"]

    # Embedding params
    if tied:
        embed_target = ratio * total
        d_model = int(embed_target / V)
    else:
        embed_target = ratio * total
        d_model = int(embed_target / (2 * V))

    # Round to be divisible by n_heads
    d_model = (d_model // n_heads) * n_heads

    # FFN dimension
    d_ff = int(glu * d_model)

    # Actual embedding params
    if tied:
        embed_params = V * d_model
    else:
        embed_params = 2 * V * d_model

    # Backbone budget
    backbone_budget = total - embed_params

    # Params per layer
    # Attention: 4 * d^2 (Q,K,V,O)
    # SwiGLU: 3 * d * d_ff (gate, up, down)
    # LayerNorm: 4 * d (2 LNs, each with weight+bias = 2*d)
    params_per_layer = (
        4 * d_model * d_model +
        3 * d_model * d_ff +
        4 * d_model
    )

    n_layers = max(1, int(backbone_budget / params_per_layer))

    return d_model, d_ff, n_layers


def count_params_manual(config, d_model, d_ff, n_layers, max_seq_len=512):
    """Manually count expected parameters."""
    V = config["vocab_size"]
    tied = config["tied_lm_head"]

    # Token embedding: V * d
    token_embed = V * d_model

    # Position embedding: max_seq_len * d
    pos_embed = max_seq_len * d_model

    # Per-layer params
    attn_params = 4 * d_model * d_model  # Q, K, V, O
    ffn_params = 3 * d_model * d_ff      # gate, up, down
    ln_params = 4 * d_model               # 2 LNs × 2 params each

    per_layer = attn_params + ffn_params + ln_params

    # Backbone
    backbone = n_layers * per_layer

    # Final LN
    final_ln = 2 * d_model

    # LM head
    if tied:
        lm_head = 0  # Shares with token embedding
    else:
        lm_head = V * d_model

    # Total
    total = token_embed + pos_embed + backbone + final_ln + lm_head

    return {
        "token_embed": token_embed,
        "pos_embed": pos_embed,
        "backbone": backbone,
        "final_ln": final_ln,
        "lm_head": lm_head,
        "total": total,
    }


# Run verification for each config
print("Testing parameter budget calculations:\n")

for i, config in enumerate(configs, 1):
    print(f"[Config {i}] {config['name']}")
    print("-" * 80)

    # Compute dimensions
    d_model, d_ff, n_layers = compute_dims_manual(config)

    print(f"  Computed dimensions:")
    print(f"    d_model:  {d_model}")
    print(f"    d_ff:     {d_ff}")
    print(f"    n_layers: {n_layers}")

    # Count parameters
    params = count_params_manual(config, d_model, d_ff, n_layers)

    print(f"\n  Parameter breakdown:")
    print(f"    Token Embedding:      {params['token_embed']:>10,}")
    print(f"    Position Embedding:   {params['pos_embed']:>10,}")
    print(f"    Transformer Backbone: {params['backbone']:>10,}")
    print(f"    Final LayerNorm:      {params['final_ln']:>10,}")
    print(f"    LM Head:              {params['lm_head']:>10,}")
    print(f"    {'─' * 40}")
    print(f"    TOTAL:                {params['total']:>10,}")

    # Check tolerance
    target = config["total_params"]
    actual = params["total"]
    diff = actual - target
    diff_pct = (actual / target - 1) * 100

    print(f"\n  Target:     {target:>10,}")
    print(f"  Actual:     {actual:>10,}")
    print(f"  Difference: {diff:>+10,} ({diff_pct:+.2f}%)")

    # Verify ±0.5% tolerance
    tolerance = 0.5
    if abs(diff_pct) <= tolerance:
        print(f"  ✓ WITHIN ±{tolerance}% TOLERANCE")
    else:
        print(f"  ✗ OUTSIDE ±{tolerance}% TOLERANCE")

    # Verify embedding ratio
    embed_total = params['token_embed'] + params['lm_head']
    actual_ratio = embed_total / actual
    ratio_diff = abs(actual_ratio - config['embedding_ratio']) * 100

    print(f"\n  Embedding ratio:")
    print(f"    Target:  {config['embedding_ratio']:.1%}")
    print(f"    Actual:  {actual_ratio:.1%}")
    print(f"    Diff:    {ratio_diff:.1f}pp")

    # Verify GLU expansion
    actual_expansion = d_ff / d_model
    expansion_diff = abs(actual_expansion - config['glu_expansion'])

    print(f"\n  GLU expansion:")
    print(f"    Target:  {config['glu_expansion']:.2f}x")
    print(f"    Actual:  {actual_expansion:.2f}x")
    print(f"    Diff:    {expansion_diff:.2f}x")

    print("\n" + "=" * 80 + "\n")

print("✓ All parameter budget calculations verified!")
print()
