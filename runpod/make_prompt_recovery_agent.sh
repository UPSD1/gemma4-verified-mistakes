#!/usr/bin/env bash
# Eval-only control arm: the base agent plus a short recovery instruction in its system prompt
# (Plan: "prompt-only recovery"). Nothing else changes: same model, tools, budgets and sampling.
set -euo pipefail
SRC=/workspace/agents/base
DST=/workspace/agents/prompt_recovery
rm -rf "$DST"; cp -r "$SRC" "$DST"
cat /workspace/repo/agents/prompt_recovery_addendum.md >> "$DST/prompts/system.md"
diff -r "$SRC" "$DST" || true
