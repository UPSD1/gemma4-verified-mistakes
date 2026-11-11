#!/usr/bin/env bash
# Base-model OOD-45 runs: one harness process per repo (each repo has its own wheel dir), PAR repos at a time.
# Usage: bash runpod/run_ood45.sh <agent_dir> <results_root> <seed> [<seed> ...]
set -uo pipefail
source /workspace/env.sh
source /workspace/repo/runpod/sandbox_tmp.sh
AGENT=$1; ROOT=$2; shift 2
PAR=${PAR:-8}
OOD=/workspace/datasets/ood45
REPOS=$(python3 -c "import json; print(' '.join(sorted({json.loads(l)['repo'] for l in open('$OOD/tasks.jsonl')} - {'urllib3/urllib3'})))")
for SEED in "$@"; do
  OUT=$ROOT/s$SEED; mkdir -p "$OUT"
  echo $REPOS | tr ' ' '\n' | xargs -P "$PAR" -I{} sh -c \
    'N=$(echo {} | tr / _); python /workspace/repo/runner/run_eval.py --ood '"$OOD"' --repo {} --agent '"$AGENT"' --results '"$OUT"'/$N --concurrency 12 > '"$OUT"'/$N.log 2>&1'
  echo "seed $SEED done $(date -u +%H:%M)"
done
