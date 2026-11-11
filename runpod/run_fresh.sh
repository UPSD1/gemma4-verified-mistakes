#!/usr/bin/env bash
# Base-model runs on FreshSWE-Py training tasks, queued behind any running OOD-45 job.
# Each pass covers every task once, so a stop mid-way still leaves complete passes.
# Usage: bash runpod/run_fresh.sh <agent_dir> <results_root> <tasks.jsonl> [passes=4]
set -uo pipefail
source /workspace/env.sh
source /workspace/repo/runpod/sandbox_tmp.sh
AGENT=$1; ROOT=$2; TASKS=$3; PASSES=${4:-4}
while pgrep -f "[r]un_ood45.sh" > /dev/null; do sleep 60; done
echo "start $(date -u +%H:%M)"
for P in $(seq 1 "$PASSES"); do
  python /workspace/repo/runner/run_eval.py --fresh /workspace/datasets/freshswe --tasks "$TASKS" \
    --agent "$AGENT" --results "$ROOT/r$P" --concurrency ${CONC:-28} > "$ROOT/r$P.log" 2>&1
  echo "pass $P done $(date -u +%H:%M)"
done
