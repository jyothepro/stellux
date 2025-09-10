#!/usr/bin/env bash
set -euo pipefail

# ----- pretty helpers -----
pass() { printf "✅ %s\n" "$1"; }
fail() { printf "❌ %s\n" "$1"; exit 1; }
warn() { printf "⚠️  %s\n" "$1"; }

# ----- locate project root -----
# Works if you run from repo root (where run_experiment.py lives)
# or from the parent directory (where the folder is named parameter-allocation-experiments)
ROOT="parameter-allocation-experiments"
if [ -f "run_experiment.py" ] || [ -d "src" ]; then
  ROOT="."
elif [ -d "$ROOT" ]; then
  : # keep ROOT as is
else
  fail "Can't find project root. Run from repo root or ensure '$ROOT/' exists next to this script."
fi
pass "Project root detected at: $ROOT"

# 1) Scaffold (PRD layout)
for p in configs data scripts src run_experiment.py; do
  [ -e "$ROOT/$p" ] || fail "Missing $ROOT/$p"
done
pass "Repo scaffold matches PRD layout"

# 2) Example configs
for f in configs/embed_ratio.yaml configs/glu_expansion.yaml; do
  [ -f "$ROOT/$f" ] || fail "Missing $ROOT/$f"
done
pass "Example configs present"

# 3) Docs & env
[ -f "$ROOT/README.md" ] || fail "Missing $ROOT/README.md"
([ -f "$ROOT/pyproject.toml" ] || [ -f "$ROOT/requirements.txt" ]) || fail "Missing deps file (pyproject.toml or requirements.txt)"
[ -f "$ROOT/.env.example" ] || fail "Missing $ROOT/.env.example"
pass "Docs & env files present"

# 4) Quality hooks
[ -f "$ROOT/.pre-commit-config.yaml" ] || fail "Missing $ROOT/.pre-commit-config.yaml"
if command -v pre-commit >/dev/null 2>&1; then
  (cd "$ROOT" && pre-commit validate-config && pre-commit run --all-files) || warn "pre-commit hooks reported issues"
else
  warn "pre-commit not installed; skipping hook run"
fi

[ -d "$ROOT/tests" ] || warn "No tests/ directory found"
if command -v pytest >/dev/null 2>&1; then
  (cd "$ROOT" && pytest -q) || fail "pytest failed (ensure at least one passing test)"
else
  warn "pytest not installed; skipping tests"
fi
pass "Quality hooks/tests OK (or warnings noted)"

# 5) CI workflow
WF="$ROOT/.github/workflows/ci.yml"
[ -f "$WF" ] || fail "Missing $WF"
grep -q "pull_request" "$WF" || fail "CI missing pull_request trigger"
if grep -q "push:" "$WF" && grep -q "tags:" "$WF"; then
  pass "CI will also build on tags"
else
  warn "CI may not build on tag pushes (optional but recommended)"
fi
pass "CI workflow present with PR trigger"

# 6) Logging boilerplate
grep -q "WANDB" "$ROOT/.env.example" || warn ".env.example lacks WANDB_* placeholders"
if grep -Rqs "tensorboard" "$ROOT" || grep -Rqs "wandb" "$ROOT"; then
  pass "Found W&B/TensorBoard references"
else
  warn "No obvious W&B/TensorBoard init in code"
fi

# 7) Makefile targets
[ -f "$ROOT/Makefile" ] || fail "Missing $ROOT/Makefile"
for t in setup train eval plots; do
  grep -qE "^$t:" "$ROOT/Makefile" || fail "Makefile missing target: $t"
done
pass "Makefile targets present"

echo "🎯 Milestone 0 validation completed."
