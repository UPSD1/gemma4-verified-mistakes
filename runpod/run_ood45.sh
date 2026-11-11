#!/usr/bin/env bash
# Base-model OOD-45 runs: one harness process per repo (each repo has its own wheel dir), all in parallel.
# Usage: bash runpod/run_ood45.sh <agent_dir> <results_root> <seed> [<seed> ...]
set -uo pipefail
source /workspace/env.sh
# Sandboxes (~1 GB each) and the unpacked-wheel cache overflow the 50 GB container disk at 44 parallel
# tasks, and /workspace is a network filesystem; RAM-backed /dev/shm (117 GB) is fast and large enough.
export TMPDIR=${TMPDIR_OVERRIDE:-/dev/shm/tmp}; mkdir -p "$TMPDIR"
AGENT=$1; ROOT=$2; shift 2
OOD=/workspace/datasets/ood45
REPOS=$(python3 -c "import json; print(' '.join(sorted({json.loads(l)['repo'] for l in open('$OOD/tasks.jsonl')})))")
for SEED in "$@"; do
  OUT=$ROOT/s$SEED; mkdir -p "$OUT"
  for R in $REPOS; do
    N=${R//\//__}
    python /workspace/repo/runner/run_eval.py --ood $OOD --repo "$R" --agent "$AGENT" \
      --results "$OUT/$N" --concurrency 12 > "$OUT/$N.log" 2>&1 &
  done
  wait
  echo "seed $SEED done $(date -u +%H:%M)"
done
