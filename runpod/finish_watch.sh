#!/usr/bin/env bash
# Run on the laptop. Wait until a pod's work is done, copy its results here, then stop the pod.
# Exits early (without stopping the pod) if results stop growing for STALL_MIN minutes.
# Usage: bash runpod/finish_watch.sh <name> <port> <host> "<done-check command run on the pod>"
set -uo pipefail
NAME=$1; PORT=$2; HOST=$3; DONE_CHECK=$4
STALL_MIN=${STALL_MIN:-40}
POD="ssh -o ConnectTimeout=20 -p $PORT root@$HOST"
DEST=$(cd "$(dirname "$0")/.." && pwd)/data/runs/$NAME
count () { $POD 'cat /workspace/results/*/*/task_results.jsonl /workspace/results/*/*/*/task_results.jsonl 2>/dev/null | wc -l' 2>/dev/null || echo -1; }
last=$(count); still=0
until $POD "$DONE_CHECK" 2>/dev/null; do
  sleep 600
  now=$(count)
  if [ "$now" = "$last" ]; then still=$((still + 10)); else still=0; last=$now; fi
  if [ "$still" -ge "$STALL_MIN" ]; then echo "STALL $NAME: $now results, unchanged for $still min"; exit 2; fi
done
echo "DONE $NAME $(date -u +%H:%M) results=$(count)"
mkdir -p "$DEST"
rsync -a -e "ssh -p $PORT" --exclude '_wheels' --exclude 'gold_fresh' --exclude 'ood_probe' \
  root@"$HOST":/workspace/results/ "$DEST/" && echo "COPIED $(du -sh "$DEST" | cut -f1)" || { echo "COPY FAILED"; exit 3; }
$POD 'cp -r /workspace/logs /workspace/results/_logs 2>/dev/null; true'
rsync -a -e "ssh -p $PORT" root@"$HOST":/workspace/results/_logs/ "$DEST/_logs/" 2>/dev/null
# Stop the pod with its own pod-scoped key (volume and results stay; billing drops to storage only).
$POD 'export RUNPOD_API_KEY=$(tr "\0" "\n" < /proc/1/environ | sed -n "s/^RUNPOD_API_KEY=//p"); runpodctl stop pod "$RUNPOD_POD_ID"' \
  && echo "STOPPED $NAME" || echo "STOP FAILED $NAME"
