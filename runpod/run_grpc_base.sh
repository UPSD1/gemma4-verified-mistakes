#!/usr/bin/env bash
# Base-model runs on the gRPC slice (3 seeds), with nothing else on the GPU (fixed load),
# then the memorization check for the mined and public sets (short prompts: prompt log-probs
# over the whole vocabulary can run vLLM out of memory, so it goes last).
source /workspace/env.sh
source /workspace/repo/runpod/sandbox_tmp.sh
cd /workspace/repo
up() { curl -s -m 5 localhost:8000/v1/models | grep -q gemma; }
if ! up; then (cd /workspace && setsid nohup bash serve.sh > /workspace/logs/vllm.log 2>&1 < /dev/null &); fi
for i in $(seq 1 60); do up && break; sleep 10; done
up || { echo "vLLM did not come up"; exit 1; }
SKIP=urllib3_5279,$(cat /workspace/datasets/grpc_slice/gold_failed.txt)
mkdir -p /workspace/results/grpc_base
for S in 1 2 3; do
  up || { echo "vLLM down before seed $S"; exit 1; }
  python runner/run_eval.py --fresh /workspace/datasets/grpc_slice --tasks /workspace/datasets/grpc_slice/tasks.jsonl \
    --agent /workspace/agents/base --results /workspace/results/grpc_base/s$S --concurrency 20 --skip "$SKIP" \
    > /workspace/results/grpc_base/s$S.log 2>&1
  echo "grpc seed $S done $(date -u +%H:%M)"
done
echo GRPC_DONE
O=/workspace/results/memorization
python3 verify/memorization.py --tasks data/tasks/verified.jsonl --set mined --out $O/familiarity.jsonl --workers 2 --max-chars 6000
python3 verify/memorization.py --tasks /workspace/comp/tasks.jsonl --set public --out $O/familiarity.jsonl --workers 2 --max-chars 6000
echo MEM_DONE $(date -u +%H:%M)
