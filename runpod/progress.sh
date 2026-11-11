#!/usr/bin/env bash
# One-screen progress of everything running on both pods. Run from the repo:  runpod/progress.sh
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a; source "$ROOT/.env"; set +a
O=(-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new)

echo "=== Main pod ($RUNPOD_SSH_HOST:$RUNPOD_SSH_PORT): arm training"
ssh "${O[@]}" -p "$RUNPOD_SSH_PORT" -i "$RUNPOD_SSH_KEY" "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST" '
  echo "time (UTC): $(date -u +%H:%M)"
  curl -s -m 5 localhost:8000/v1/models | grep -q gemma && echo "vLLM: up" || echo "vLLM: down"
  for L in /workspace/logs/train_arm_*.log; do
    [ -f "$L" ] || continue
    case "$L" in *stopped*|*precleanup*) continue;; esac
    echo "-- $(basename $L .log): $(grep -E "examples," $L | tail -1)"
    grep -E "^step" $L | tail -1 | sed "s/^/   last: /"
    grep -qE "TRAIN_DONE" $L && echo "   DONE -> $(grep -E "adapter ->" $L | tail -1)"
    grep -E "Traceback|out of memory|Error" $L | tail -2 | sed "s/^/   ERROR: /"
  done
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed "s/^/   GPU: /"' 2>/dev/null || echo "   (main pod not reachable)"

echo
echo "=== Training pod (${TRAIN_SSH_HOST:-?}:${TRAIN_SSH_PORT:-?}): arm training"
[ -n "$TRAIN_SSH_HOST" ] && ssh "${O[@]}" -p "$TRAIN_SSH_PORT" -i "$TRAIN_SSH_KEY" "$TRAIN_SSH_USER@$TRAIN_SSH_HOST" '
  echo "time (UTC): $(date -u +%H:%M)"
  for L in /workspace/logs/train_arm_*.log; do
    case "$L" in *stopped*|*precleanup*) continue;; esac
    echo "-- $(basename $L .log): $(grep -E "examples," $L | tail -1)"
    grep -E "^step" $L | tail -1 | sed "s/^/   last: /"
    grep -qE "TRAIN_DONE" $L && echo "   DONE -> $(grep -E "adapter ->" $L | tail -1)"
    grep -E "Traceback|out of memory|Error" $L | tail -2 | sed "s/^/   ERROR: /"
  done
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed "s/^/   GPU: /"' 2>/dev/null || echo "   (training pod not reachable)"
