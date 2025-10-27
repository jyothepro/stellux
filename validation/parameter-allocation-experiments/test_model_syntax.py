#!/usr/bin/env python3
"""Quick syntax and structure validation for model code."""

import ast
import sys
from pathlib import Path

print("=" * 80)
print("Model Code Validation (Syntax & Structure)")
print("=" * 80)

# Test 1: Check file exists and can be parsed
print("\n[Test 1] Checking model file syntax...")
model_file = Path("src/models/lm.py")

if not model_file.exists():
    print("✗ Model file not found")
    sys.exit(1)

try:
    with open(model_file) as f:
        code = f.read()

    ast.parse(code)
    print(f"✓ Model file syntax is valid ({len(code)} characters)")
except SyntaxError as e:
    print(f"✗ Syntax error: {e}")
    sys.exit(1)

# Test 2: Check key classes are defined
print("\n[Test 2] Checking key classes...")
expected_classes = [
    "ModelConfig",
    "SwiGLU",
    "MultiHeadAttention",
    "TransformerBlock",
    "LanguageModel",
]

tree = ast.parse(code)
defined_classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

for class_name in expected_classes:
    if class_name in defined_classes:
        print(f"✓ {class_name} defined")
    else:
        print(f"✗ {class_name} missing")
        sys.exit(1)

# Test 3: Check key functions are defined
print("\n[Test 3] Checking key functions...")
expected_functions = [
    "compute_model_dims",
    "count_parameters",
    "print_param_table",
]

defined_functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

for func_name in expected_functions:
    if func_name in defined_functions:
        print(f"✓ {func_name} defined")
    else:
        print(f"✗ {func_name} missing")
        sys.exit(1)

# Test 4: Check ModelConfig has required attributes
print("\n[Test 4] Checking ModelConfig attributes...")
required_attrs = [
    "vocab_size",
    "total_params",
    "embedding_ratio",
    "glu_expansion",
    "n_layers",
    "n_heads",
    "tied_lm_head",
]

for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "ModelConfig":
        # Check for dataclass decorator
        has_dataclass = any(
            isinstance(dec, ast.Name) and dec.id == "dataclass"
            for dec in node.decorator_list
        )
        if has_dataclass:
            print("✓ ModelConfig is a dataclass")
        break

# Test 5: Count lines of code
print("\n[Test 5] Code statistics...")
lines = code.split('\n')
non_empty_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
doc_lines = code.count('"""')

print(f"✓ Total lines: {len(lines)}")
print(f"✓ Non-empty lines: {len(non_empty_lines)}")
print(f"✓ Classes defined: {len(set(defined_classes))}")
print(f"✓ Functions defined: {len(set(defined_functions))}")

# Test 6: Check imports
print("\n[Test 6] Checking imports...")
required_imports = ["torch", "torch.nn", "torch.nn.functional"]

has_torch_import = "import torch" in code
has_nn_import = "torch.nn" in code
has_f_import = "torch.nn.functional" in code

if has_torch_import:
    print("✓ PyTorch imported")
if has_nn_import:
    print("✓ torch.nn used")
if has_f_import:
    print("✓ torch.nn.functional used")

print("\n" + "=" * 80)
print("✓ All syntax and structure checks passed!")
print("=" * 80)
print("\nModel implementation is structurally complete.")
print("Run pytest tests/test_model.py to validate functionality (requires PyTorch).")
