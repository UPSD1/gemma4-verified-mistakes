#!/usr/bin/env bash
# OOD-45 eval of several agents (base, prompt-only control, LoRA arms) under one fixed load, so arms
# differ only in the agent. Each agent gets a fresh vLLM (its adapter mounted, nothing else loaded) and
# the same PAR repos x CONC tasks per repo, seed by seed. Refuses to start while other harness work runs
# on this GPU (resampling, base runs): GPU load changed the base solve rate by up to 18 points.
#
#   bash runpod/eval_arms.sh <results_root> <agent_dir> [<agent_dir> ...]
#   SEEDS="1 2 3" PAR=8 CONC=12 bash runpod/eval_arms.sh /workspace/results/eval_v1 \
#       /workspace/agents/base /workspace/agents/prompt_recovery /workspace/agents/arm_a ...
#
# Agent dirs with a LoRA come from runpod/make_arm_agent.py. Results: <root>/<agent>/s<seed>/<repo>/.
# Seeds are interleaved across agents (s1 for all agents, then s2, ...) so a late failure still leaves
# every agent with the same number of seeds. Each <root>/<agent>/s<seed>.done marks a finished seed;
# rerunning skips them.
set -uo pipefail
source /workspace/env.sh
source /workspace/repo/runpod/sandbox_tmp.sh
ROOT=$1; shift
AGENTS=("$@")
SEEDS=${SEEDS:-1 2 3}
PAR=${PAR:-8}
CONC=${CONC:-12}
OOD=/workspace/datasets/ood45
LOG=/workspace/logs
mkdir -p "$ROOT" "$LOG"

busy=$(ps -eo args | grep -E "[r]un_eval.py|[r]esample.py|[t]rain_sft.py" | grep -v -- "--results $ROOT" | head -3)
if [ -n "$busy" ] && [ -z "${FORCE:-}" ]; then
  echo "Other GPU work is running; arms would not share the same load:"; echo "$busy"; exit 1
fi

REPOS=$(python3 -c "import json; print(' '.join(sorted({json.loads(l)['repo'] for l in open('$OOD/tasks.jsonl')} - {'urllib3/urllib3'})))")
cat > "$ROOT/eval_config.txt" <<EOF
started $(date -u +%FT%TZ)
gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
seeds $SEEDS  par $PAR  conc $CONC  repos $(echo $REPOS | wc -w)
agents ${AGENTS[*]}
repo $(git -C /workspace/repo rev-parse --short HEAD)
EOF

stop_vllm() {
  pkill -f "[v]llm.entrypoints.openai.api_server" || true
  while pgrep -f "[v]llm.entrypoints.openai.api_server" > /dev/null; do sleep 5; done
  sleep 10  # let the GPU memory free up
}

start_vllm() {  # $1 = agent dir
  local mods
  mods=$(python3 - "$1" <<'PY'
import sys
from pathlib import Path
from adk_submission import discover_adapters
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
for n, i in discover_adapters(sys.argv[1], adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS).adapters.items():
    p = Path(i.path)
    print(f"{n}={p.parent if (p.parent / 'adapter_config.json').is_file() else p}")
PY
)
  stop_vllm
  local args=()
  [ -n "$mods" ] && args=(--lora-modules $mods)
  echo "vllm for $(basename "$1"): ${mods:-no adapter}"
  nohup bash /workspace/serve.sh "${args[@]}" > "$LOG/vllm_$(basename "$1").log" 2>&1 < /dev/null &
  local t=0
  until curl -sf localhost:8000/v1/models > /dev/null; do
    sleep 10; t=$((t + 10))
    [ $t -ge 1500 ] && { echo "vLLM did not start for $1"; tail -20 "$LOG/vllm_$(basename "$1").log"; return 1; }
  done
  if [ -n "$mods" ]; then  # the adapter must be served, not silently dropped
    curl -s localhost:8000/v1/models | grep -q "\"${mods%%=*}\"" || { echo "adapter ${mods%%=*} not served"; return 1; }
  fi
}

for SEED in $SEEDS; do
  for A in "${AGENTS[@]}"; do
    NAME=$(basename "$A"); OUT=$ROOT/$NAME/s$SEED
    [ -f "$OUT.done" ] && { echo "skip $NAME s$SEED"; continue; }
    start_vllm "$A" || exit 1
    mkdir -p "$OUT"
    echo "$NAME s$SEED start $(date -u +%H:%M)"
    echo $REPOS | tr ' ' '\n' | xargs -P "$PAR" -I{} sh -c \
      'N=$(echo {} | tr / _); python /workspace/repo/runner/run_eval.py --ood '"$OOD"' --repo {} --agent '"$A"' --results '"$OUT"'/$N --concurrency '"$CONC"' > '"$OUT"'/$N.log 2>&1'
    n=$(cat "$OUT"/*/task_results.jsonl 2>/dev/null | grep -c '"resolved": true')
    t=$(cat "$OUT"/*/task_results.jsonl 2>/dev/null | wc -l)
    echo "$NAME s$SEED done $(date -u +%H:%M): $n/$t resolved" | tee -a "$ROOT/progress.txt"
    touch "$OUT.done"
  done
done
echo EVAL_ARMS_DONE
