#!/usr/bin/env bash
# Second GPU pod: serve the model, then run FreshSWE-Py base passes (default 3 and 4) on the shared task list.
# Expects /opt/venv, the system pytest packages, the model, FreshSWE-Py and its wheelhouse at the same paths
# as the first pod (runpod/push + Kaggle downloads).
# Usage: bash runpod/h100_passes.sh [pass ...]
set -uo pipefail
PASSES=${*:-3 4}
source /workspace/env.sh
source /workspace/repo/runpod/sandbox_tmp.sh
mkdir -p /opt/freshswe /workspace/logs
ln -sfn /workspace/freshswe_wheelhouse /opt/freshswe/wheelhouse
ln -sfn /workspace/datasets/freshswe/locks /opt/freshswe/locks

if ! curl -sf localhost:8000/v1/models > /dev/null; then
  nohup bash /workspace/serve.sh > /workspace/logs/vllm.log 2>&1 < /dev/null &
  until curl -sf localhost:8000/v1/models > /dev/null; do sleep 10; done
fi
echo "vllm ready $(date -u +%H:%M)"
for P in $PASSES; do
  python /workspace/repo/runner/run_eval.py --fresh /workspace/datasets/freshswe \
    --tasks /workspace/results/fresh_base/tasks.jsonl --agent /workspace/agents/base \
    --results /workspace/results/fresh_base/r$P --concurrency ${CONC:-48} > /workspace/results/fresh_base/r$P.log 2>&1
  echo "pass $P done $(date -u +%H:%M)"
done
echo H100_DONE
