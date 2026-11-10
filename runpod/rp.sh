#!/usr/bin/env bash
# Work with the RunPod pod from this machine, using the RUNPOD_* settings in .env.
#
#   runpod/rp.sh ssh                         interactive shell on the pod
#   runpod/rp.sh run "<command>"             run a command there (env.sh is sourced first)
#   runpod/rp.sh pull <remote> <local>       copy a file or folder from the pod
#   runpod/rp.sh push <local> <remote>       copy a file or folder to the pod
#   runpod/rp.sh tunnel                      forward the pod's vLLM to localhost (Ctrl-C to stop)
#   runpod/rp.sh check                       test the connection and show GPU / disk
#
# Remote paths may use the pod path variables, e.g. '$RUNPOD_FRESHSWE_DIR/tasks.jsonl'
# (single quotes, so they expand from .env here, not in your shell).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/.env" ] || { echo "missing $ROOT/.env (copy .env.example)"; exit 1; }
set -a; source "$ROOT/.env"; set +a
: "${RUNPOD_SSH_HOST:?set RUNPOD_SSH_HOST in .env}" "${RUNPOD_SSH_PORT:?set RUNPOD_SSH_PORT in .env}"

SSH_OPTS=(-o ConnectTimeout=15 -o ServerAliveInterval=30 -p "$RUNPOD_SSH_PORT" -i "$RUNPOD_SSH_KEY")
SCP_OPTS=(-o ConnectTimeout=15 -P "$RUNPOD_SSH_PORT" -i "$RUNPOD_SSH_KEY")
TARGET="$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST"
expand() { eval "printf '%s' \"$1\""; }   # expand $RUNPOD_* in a path
# Pod path variables, exported on the pod too so '$RUNPOD_FRESHSWE_DIR' works in "run".
POD_VARS=""
for v in RUNPOD_WORKSPACE RUNPOD_ENV_SCRIPT RUNPOD_REPO_DIR RUNPOD_COMP_DIR RUNPOD_FRESHSWE_DIR \
         RUNPOD_OOD45_DIR RUNPOD_MODELS_DIR RUNPOD_RESULTS_DIR RUNPOD_LOGS_DIR RUNPOD_GRPC_WHEELS_DIR \
         RUNPOD_VLLM_PORT RUNPOD_VLLM_MODEL; do
  [ -n "${!v:-}" ] && POD_VARS+="export $v=$(printf '%q' "${!v}"); "
done

cmd="${1:-}"; shift || true
case "$cmd" in
  ssh)    exec ssh -t "${SSH_OPTS[@]}" "$TARGET" ;;
  run)    exec ssh "${SSH_OPTS[@]}" "$TARGET" "source ${RUNPOD_ENV_SCRIPT:-/workspace/env.sh} >/dev/null 2>&1; $POD_VARS $*" ;;
  pull)   [ $# -eq 2 ] || { echo "usage: rp.sh pull <remote> <local>"; exit 2; }
          mkdir -p "$(dirname "$2")"; exec scp -r "${SCP_OPTS[@]}" "$TARGET:$(expand "$1")" "$2" ;;
  push)   [ $# -eq 2 ] || { echo "usage: rp.sh push <local> <remote>"; exit 2; }
          r="$(expand "$2")"; ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p \"$(dirname "$r")\""
          exec scp -r "${SCP_OPTS[@]}" "$1" "$TARGET:$r" ;;
  tunnel) echo "vLLM on the pod -> http://localhost:${RUNPOD_VLLM_LOCAL_PORT}/v1 (Ctrl-C to stop)"
          exec ssh -N "${SSH_OPTS[@]}" -L "${RUNPOD_VLLM_LOCAL_PORT}:localhost:${RUNPOD_VLLM_PORT}" "$TARGET" ;;
  check)  exec ssh "${SSH_OPTS[@]}" "$TARGET" 'hostname; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader; df -h /workspace | tail -1' ;;
  *)      sed -n '2,12p' "$0"; exit 2 ;;
esac
