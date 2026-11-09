#!/usr/bin/env bash
# RunPod GPU pod setup: competition data + vLLM serving Gemma 4.
# Usage on the pod (PyTorch template, >=200 GB volume at /workspace):
#   export KAGGLE_API_TOKEN=...   # from kaggle.com/settings -> API (after accepting the competition rules)
#   export MODEL=...              # path or HF id of the Gemma 4 checkpoint (see step 3)
#   bash runpod/setup.sh
set -euo pipefail
WS=${WS:-/workspace}
cd "$WS"

# 1. Tools. vLLM 0.19.1 is the version public competition write-ups used with the gemma4 tool parser.
pip install -q uv
uv pip install --system -q "vllm==0.19.1" kaggle huggingface_hub

# 2. Competition data (22 GB). Requires that the account behind the token accepted the rules.
mkdir -p "$WS/comp"
if [ ! -f "$WS/comp/tasks.jsonl" ]; then
  kaggle competitions download -c gemma-4-developer-agent -p "$WS/comp"
  (cd "$WS/comp" && unzip -q -o ./*.zip && rm -f ./*.zip)
fi
echo "Read $WS/comp/HARNESS_README.md before running the harness."

# 3. Model. The competition serves gemma-4-{31b,12b}-it-qat-w4a16-ct (4-bit).
#    Use the same checkpoint for data collection so runs match competition behaviour.
: "${MODEL:?set MODEL to the Gemma 4 checkpoint path or id}"
TP=${TP:-$(nvidia-smi -L | wc -l)}

# 4. Serve. Thinking on + gemma4 tool-call parser; LoRA enabled so we can evaluate adapters.
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name gemma \
  --tensor-parallel-size "$TP" --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser gemma4 \
  --enable-prefix-caching --enable-lora --max-lora-rank 64 \
  --port 8000 > "$WS/vllm.log" 2>&1 &
echo "vLLM starting (log: $WS/vllm.log). Check: curl localhost:8000/v1/models"
