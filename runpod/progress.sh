#!/usr/bin/env bash
# One-screen progress of everything running on both pods. Run from the repo:  runpod/progress.sh
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a; source "$ROOT/.env"; set +a
O=(-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new)

echo "=== Main pod ($RUNPOD_SSH_HOST:$RUNPOD_SSH_PORT): resampling for arms (b)-(d)"
ssh "${O[@]}" -p "$RUNPOD_SSH_PORT" -i "$RUNPOD_SSH_KEY" "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST" '
  echo "time (UTC): $(date -u +%H:%M)"
  curl -s -m 5 localhost:8000/v1/models | grep -q gemma && echo "vLLM: up" || echo "vLLM: DOWN"
  for B in resample resample_mined; do
    [ -f /workspace/logs/$B.log ] || continue
    echo "-- $B: $(grep -E "^start|DONE" /workspace/logs/$B.log | tr "\n" " ")"
    for d in /workspace/results/$B/s*_p*/; do
      [ -d "$d" ] || continue
      n=$(cat $d/task_results.jsonl 2>/dev/null | wc -l); b=$(wc -l < $d/branches.jsonl 2>/dev/null || echo "?")
      r=$(grep -c "\"resolved\": true" $d/task_results.jsonl 2>/dev/null || echo 0)
      echo "   $(basename $d): $n/$b continuations graded, $r recovered"
    done
  done' 2>/dev/null || echo "   (main pod not reachable)"

echo
echo "=== Training pod (${TRAIN_SSH_HOST:-?}:${TRAIN_SSH_PORT:-?}): arm training"
[ -n "$TRAIN_SSH_HOST" ] && ssh "${O[@]}" -p "$TRAIN_SSH_PORT" -i "$TRAIN_SSH_KEY" "$TRAIN_SSH_USER@$TRAIN_SSH_HOST" '
  echo "time (UTC): $(date -u +%H:%M)"
  for L in /workspace/logs/train_arm_*.log; do
    case "$L" in *stopped*) continue;; esac
    echo "-- $(basename $L .log): $(grep -E "examples," $L | tail -1)"
    grep -E "^step" $L | tail -1 | sed "s/^/   last: /"
    grep -qE "TRAIN_DONE" $L && echo "   DONE -> $(grep -E "adapter ->" $L | tail -1)"
    grep -E "Traceback|out of memory|Error" $L | tail -2 | sed "s/^/   ERROR: /"
  done
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed "s/^/   GPU: /"' 2>/dev/null || echo "   (training pod not reachable)"
