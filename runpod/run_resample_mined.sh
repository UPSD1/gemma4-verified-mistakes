#!/usr/bin/env bash
# Second resampling batch: branch points from the mined-task base runs, after the FreshSWE batch.
source /workspace/env.sh
source /workspace/repo/runpod/sandbox_tmp.sh
cd /workspace/repo
while ! grep -q RESAMPLE_DONE /workspace/logs/resample.log 2>/dev/null; do sleep 120; done
echo "start $(date -u +%H:%M)"
run_group() {
  local OUT=/workspace/results/resample_mined/s$1_p$2
  python training/resample.py --branches data/training/branch_points_mined.jsonl --sample $1 --pick $2 \
    --fresh /workspace/datasets/mined_slice --agent /workspace/agents/base --results $OUT --concurrency 24 > $OUT.log 2>&1 || true
  echo "s$1 p$2 done $(date -u +%H:%M): $(grep -c "\"resolved\": true" $OUT/task_results.jsonl 2>/dev/null || echo 0) resolved"
}
mkdir -p /workspace/results/resample_mined
for S in 1 2; do run_group $S 0 & run_group $S 1 & wait; run_group $S 2 & run_group $S 3 & wait; done
echo RESAMPLE_MINED_DONE
