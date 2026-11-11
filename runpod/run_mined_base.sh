#!/usr/bin/env bash
# Base-model runs on our mined tasks (tracker row 11 for the mined set): 4 passes, each over every task
# whose gold patch passes in the harness, so a stop between passes still leaves complete passes.
# Usage: bash runpod/run_mined_base.sh [passes=4]
source /workspace/env.sh
source /workspace/repo/runpod/sandbox_tmp.sh
cd /workspace/repo
SLICE=/workspace/datasets/mined_slice
ROOT=/workspace/results/mined_base
mkdir -p $ROOT
python3 - <<PY
import json
bad = {r["id"] for r in map(json.loads, open("/workspace/results/gold_mined/gold_check.jsonl"))
       if r["kind"] == "gold" and not r["resolved"]}
rows = [l for l in open("$SLICE/tasks.jsonl") if json.loads(l)["instance_id"] not in bad]
open("$ROOT/tasks.jsonl", "w").writelines(rows)
print(len(rows), "tasks (gold patch passes)")
PY
up() { curl -s -m 5 localhost:8000/v1/models | grep -q gemma; }
if ! up; then (cd /workspace && setsid nohup bash serve.sh > /workspace/logs/vllm.log 2>&1 < /dev/null &); fi
for i in $(seq 1 60); do up && break; sleep 10; done
echo "start $(date -u +%H:%M)"
for P in $(seq 1 "${1:-4}"); do
  up || { echo "vLLM down before pass $P"; exit 1; }
  python runner/run_eval.py --fresh $SLICE --tasks $ROOT/tasks.jsonl --agent /workspace/agents/base \
    --results $ROOT/r$P --concurrency ${CONC:-28} > $ROOT/r$P.log 2>&1
  echo "pass $P done $(date -u +%H:%M)"
done
echo MINED_BASE_DONE
