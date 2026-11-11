#!/usr/bin/env bash
# Resample every branch point (training/branch_points.jsonl) SAMPLES times, after the mined base runs
# finish (waits for MINED_BASE_DONE). Each harness run takes one branch point per task (--pick).
# Two groups run at once. Results: /workspace/results/resample/s<sample>_p<pick>/ (task_results.jsonl, traces/, branches.jsonl).
source /workspace/env.sh
source /workspace/repo/runpod/sandbox_tmp.sh
cd /workspace/repo
SAMPLES=${SAMPLES:-2}
while ! grep -q MINED_BASE_DONE /workspace/logs/mined_base.log 2>/dev/null; do sleep 120; done
echo "start $(date -u +%H:%M)"
up() { curl -s -m 5 localhost:8000/v1/models | grep -q gemma; }
run_group() {  # $1 = sample, $2 = pick
  local OUT=/workspace/results/resample/s$1_p$2
  python training/resample.py --branches data/training/branch_points.jsonl --sample $1 --pick $2 \
    --fresh /workspace/datasets/freshswe_run --agent /workspace/agents/base --results $OUT --concurrency ${CONC:-24} \
    > $OUT.log 2>&1 || true
  echo "s$1 p$2 done $(date -u +%H:%M): $(grep -c '"resolved": true' $OUT/task_results.jsonl 2>/dev/null || echo 0) resolved"
}
# Two groups at a time keep the GPU busy (one group of 24 leaves it underused).
for S in $(seq 1 $SAMPLES); do
  up || { echo "vLLM down before sample $S"; exit 1; }
  run_group $S 0 & run_group $S 1 & wait
  run_group $S 2 & run_group $S 3 & wait
done
echo RESAMPLE_DONE
