#!/usr/bin/env bash
# One-screen progress of everything running on both pods. Run from the repo:  runpod/progress.sh
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a; source "$ROOT/.env"; set +a
O=(-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new)

echo "=== Main pod ($RUNPOD_SSH_HOST:$RUNPOD_SSH_PORT): arm training and OOD-45 eval"
ssh "${O[@]}" -p "$RUNPOD_SSH_PORT" -i "$RUNPOD_SSH_KEY" "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST" '
  echo "time (UTC): $(date -u +%H:%M)"
  curl -s -m 5 localhost:8000/v1/models | grep -q gemma && echo "vLLM: up" || echo "vLLM: down"
  for L in /workspace/logs/train_arm_*.log; do
    [ -f "$L" ] || continue
    case "$L" in *stopped*|*precleanup*) continue;; esac
    echo "-- $(basename $L .log): $(grep -E "examples,|pairs \(" $L | tail -1)"
    grep -E "^step" $L | tail -1 | sed "s/^/   last: /"
    grep -qE "TRAIN_DONE" $L && echo "   DONE -> $(grep -E "adapter ->" $L | tail -1)"
    grep -E "Traceback|out of memory|Error" $L | tail -2 | sed "s/^/   ERROR: /"
  done
  for R in /root/results/eval_*/ /workspace/results/eval_*/; do
    [ -f "$R/eval_config.txt" ] || continue
    E=$(basename $R)
    echo "-- $E ($(grep -E "^seeds" $R/eval_config.txt | sed "s/  */ /g")):"
    [ -f /workspace/logs/$E.log ] && grep -E " start | done |vllm for" /workspace/logs/$E.log | tail -1 | sed "s/^/   now: /"
    for a in $R*/; do for d in $a/s*/; do
      [ -d "$d" ] || continue
      n=$(cat $d/*/task_results.jsonl 2>/dev/null | wc -l); r=$(cat $d/*/task_results.jsonl 2>/dev/null | grep -c "\"resolved\": true")
      echo "   $(basename $a) $(basename $d): $n/45 graded, $r resolved$([ -f ${d%/}.done ] && echo " (done)")"
    done; done
  done
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed "s/^/   GPU: /"' 2>/dev/null || echo "   (main pod not reachable)"

echo
echo "=== Training pod (${TRAIN_SSH_HOST:-?}:${TRAIN_SSH_PORT:-?}): arm training"
[ -n "$TRAIN_SSH_HOST" ] && ssh "${O[@]}" -p "$TRAIN_SSH_PORT" -i "$TRAIN_SSH_KEY" "$TRAIN_SSH_USER@$TRAIN_SSH_HOST" '
  echo "time (UTC): $(date -u +%H:%M)"
  for L in /workspace/logs/train_arm_*.log; do
    case "$L" in *stopped*|*precleanup*) continue;; esac
    echo "-- $(basename $L .log): $(grep -E "examples,|pairs \(" $L | tail -1)"
    grep -E "^step" $L | tail -1 | sed "s/^/   last: /"
    grep -qE "TRAIN_DONE" $L && echo "   DONE -> $(grep -E "adapter ->" $L | tail -1)"
    grep -E "Traceback|out of memory|Error" $L | tail -2 | sed "s/^/   ERROR: /"
  done
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed "s/^/   GPU: /"' 2>/dev/null || echo "   (training pod not reachable)"

# David's v3 retraining pods (training/fix_examples.py data, runpod/retrain_v3.sh): logs in /root/rt/logs.
v3_pod() {  # $1 = title, $2 = .env prefix
  local H=${2}_SSH_HOST P=${2}_SSH_PORT U=${2}_SSH_USER K=${2}_SSH_KEY
  [ -n "${!H}" ] || return
  echo; echo "=== $1 (${!H}:${!P})"
  ssh "${O[@]}" -p "${!P}" -i "${!K/#\~/$HOME}" "${!U}@${!H}" '
    echo "time (UTC): $(date -u +%H:%M)"
    for L in /root/rt/logs/train_arm_*.log; do
      [ -f "$L" ] || continue
      echo "-- $(basename $L .log): $(grep -E "examples,|pairs \(" $L | tail -1)"
      grep -E "^step" $L | tail -1 | sed "s/^/   last: /"
      grep -qE "adapter ->" $L && echo "   DONE -> $(grep -E "adapter ->" $L | tail -1)"
      grep -E "Traceback|out of memory|Error" $L | tail -2 | sed "s/^/   ERROR: /"
    done
    for Q in /root/rt/logs/queue_*.log; do [ -s "$Q" ] && echo "-- $(basename $Q .log): $(tail -1 $Q)"; done
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed "s/^/   GPU /"' 2>/dev/null || echo "   (not reachable)"
}
v3_pod "v3 retrain pod, arm (a)" V3A
v3_pod "v3 retrain pod, arms (b), (c), then (d)" V3BC
