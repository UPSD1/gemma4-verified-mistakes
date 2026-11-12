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
# Default 0 = stock vLLM 0.19.1, exactly as the competition serves it. The harness (adk_submission,
# resolvers/generation.py:66) already copies reasoning_content into `reasoning`, so the agent's earlier
# thoughts reach the model either way; prompt_tokens in our traces match renderings with thoughts kept.
# 1 = also accept bare reasoning_content (runpod/patch_vllm_reasoning.sh); only for clients that bypass
# the harness. The check below sends bare reasoning_content, so it verifies the server setting itself.
KEEP_REASONING=${KEEP_REASONING:-0}
export VLLM_KEEP_REASONING_CONTENT=$KEEP_REASONING
if [ "$KEEP_REASONING" = 1 ] && ! grep -q VLLM_KEEP_REASONING_CONTENT \
     "$(python3 -c 'import vllm.entrypoints.chat_utils as m; print(m.__file__)')"; then
  echo "KEEP_REASONING=1 but vLLM is not patched: run runpod/patch_vllm_reasoning.sh first"; exit 1
fi
OOD=/workspace/datasets/ood45
LOG=/workspace/logs
mkdir -p "$ROOT" "$LOG"

busy=$(ps -eo args | grep -E "[r]un_eval.py|[r]esample.py|[t]rain_sft.py|[t]rain_dpo.py|[l]ora_smoke.py|[v]erify_gold.py|[m]emorization.py" | grep -v -- "--results $ROOT" | head -3)
if [ -n "$busy" ] && [ -z "${FORCE:-}" ]; then
  echo "Other GPU work is running; arms would not share the same load:"; echo "$busy"; exit 1
fi

REPOS=$(python3 -c "import json; print(' '.join(sorted({json.loads(l)['repo'] for l in open('$OOD/tasks.jsonl')} - {'urllib3/urllib3'})))")
cat > "$ROOT/eval_config.txt" <<EOF
started $(date -u +%FT%TZ)
gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
seeds $SEEDS  par $PAR  conc $CONC  repos $(echo $REPOS | wc -w)  keep_reasoning $KEEP_REASONING
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
  # The agent's earlier thought must reach the prompt the way the harness sends it (adk_submission
  # mirrors reasoning_content into reasoning), whatever KEEP_REASONING is.
  python3 - <<'PY' || return 1
import json, sys, urllib.request
def post(path, body):
    req = urllib.request.Request("http://localhost:8000" + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))
model = json.load(urllib.request.urlopen("http://localhost:8000/v1/models"))["data"][0]["id"]
msgs = [{"role": "user", "content": "What time is it?"},
        {"role": "assistant", "content": "", "reasoning_content": "Secret word PAPAYAQUOKKA.",
         "reasoning": "Secret word PAPAYAQUOKKA.",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "12:34"}]
toks = post("/tokenize", {"model": model, "messages": msgs, "add_generation_prompt": True,
                          "chat_template_kwargs": {"enable_thinking": True}})["tokens"]
kept = "PAPAYAQUOKKA" in post("/detokenize", {"model": model, "tokens": toks})["prompt"]
print(f"reasoning check: earlier thought in prompt = {kept} (must be True)")
sys.exit(0 if kept else 1)
PY
}

for SEED in $SEEDS; do
  for A in "${AGENTS[@]}"; do
    NAME=$(basename "$A"); OUT=$ROOT/$NAME/s$SEED
    [ -f "$OUT.done" ] && { echo "skip $NAME s$SEED"; continue; }
    start_vllm "$A" || exit 1
    rm -rf "$OUT"; mkdir -p "$OUT"  # a half-finished seed is rerun from scratch (results files append)
    echo "$NAME s$SEED start $(date -u +%H:%M)"
    echo $REPOS | tr ' ' '\n' | xargs -P "$PAR" -I{} sh -c \
      'N=$(echo {} | tr / _); python /workspace/repo/runner/run_eval.py --ood '"$OOD"' --repo {} --agent '"$A"' --results '"$OUT"'/$N --concurrency '"$CONC"' > '"$OUT"'/$N.log 2>&1'
    n=$(cat "$OUT"/*/task_results.jsonl 2>/dev/null | grep -c '"resolved": true')
    t=$(cat "$OUT"/*/task_results.jsonl 2>/dev/null | wc -l)
    echo "$NAME s$SEED done $(date -u +%H:%M): $n/$t resolved" | tee -a "$ROOT/progress.txt"
    NT=$(cat "$OOD/tasks.jsonl" | grep -vc '"urllib3/urllib3"')
    [ "$t" -eq "$NT" ] || { echo "$NAME s$SEED incomplete: $t of $NT tasks graded; not marking done"; exit 1; }
    touch "$OUT.done"
  done
done
echo EVAL_ARMS_DONE
