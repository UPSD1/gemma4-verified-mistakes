#!/usr/bin/env bash
# Resampling, corrected design (review points 1 and 8): the prefix ends WITH the failing step and its
# error, the loaded context is only what the model saw since the last compaction, and branch points that
# would load more than 16k tokens are skipped. One branch_points.jsonl holds FreshSWE-Py and mined tasks;
# each dataset runs with its own --fresh folder (resample.py keeps only that dataset's tasks).
# SAMPLES continuations per branch point. Results: /workspace/results/resample_v2/{fresh,mined}/s<S>_p<P>/
source /workspace/env.sh
source /workspace/repo/runpod/sandbox_tmp.sh
cd /workspace/repo
SAMPLES=${SAMPLES:-2}
R=/workspace/results/resample_v2
mkdir -p $R/fresh $R/mined
echo "start $(date -u +%H:%M)"
up() { curl -s -m 5 localhost:8000/v1/models | grep -q gemma; }
run_group() {  # $1 = fresh|mined, $2 = sample, $3 = pick
  local DS=freshswe_run; [ "$1" = mined ] && DS=mined_slice
  local OUT=$R/$1/s$2_p$3
  python training/resample.py --branches data/training/branch_points.jsonl --sample $2 --pick $3 \
    --fresh /workspace/datasets/$DS --agent /workspace/agents/base --results $OUT --concurrency ${CONC:-24} \
    > $OUT.log 2>&1 || true
  echo "$1 s$2 p$3 done $(date -u +%H:%M): $(grep -c '"resolved": true' $OUT/task_results.jsonl 2>/dev/null || echo 0) resolved of $(wc -l < $OUT/branches.jsonl 2>/dev/null || echo 0)"
}
for S in $(seq 1 $SAMPLES); do
  up || { echo "vLLM down before sample $S"; exit 1; }
  run_group fresh $S 0 & run_group mined $S 0 & wait          # 49 + 34 branch points
  run_group fresh $S 1 & run_group mined $S 1 & run_group fresh $S 2 & run_group mined $S 2 &
  run_group fresh $S 3 & run_group mined $S 3 & wait          # 20 + 8 + 6 + 2 + 1 + 1
done
echo RESAMPLE_V2_DONE
