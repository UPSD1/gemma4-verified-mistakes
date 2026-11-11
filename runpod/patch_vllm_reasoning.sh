#!/usr/bin/env bash
# Switchable fix for the dropped-reasoning bug in vLLM 0.19.1 (Kaggle discussion 744354, vllm#38488).
#
# ADK sends the agent's previous thought back as `reasoning_content`; vLLM 0.19.1 only reads `reasoning`
# (entrypoints/chat_utils.py, _parse_chat_message_content), so earlier thoughts never reach the chat
# template. This adds a fallback to `reasoning_content`, active only when the vLLM server runs with
# VLLM_KEEP_REASONING_CONTENT=1. Without that variable the server behaves exactly as before, so the
# patched file is safe to leave in place for unpatched runs.
#
#   bash runpod/patch_vllm_reasoning.sh            # apply (idempotent; keeps a .orig backup), then check
#   bash runpod/patch_vllm_reasoning.sh check      # check only: renders one history both ways, no GPU
set -euo pipefail
PY=${PY:-/opt/venv/bin/python}
MODEL=${MODEL:-/workspace/models/kaggle-12b}
F=$($PY -c "import vllm.entrypoints.chat_utils as m; print(m.__file__)")
MARK="VLLM_KEEP_REASONING_CONTENT"

if [ "${1:-apply}" = apply ]; then
  if grep -q "$MARK" "$F"; then
    echo "already patched: $F"
  else
    [ -f "$F.orig" ] || cp "$F" "$F.orig"
    n=$(grep -c '^    reasoning = message.get("reasoning")$' "$F")
    [ "$n" = 1 ] || { echo "expected exactly one target line in $F, found $n; not patching"; exit 1; }
    sed -i 's/^    reasoning = message.get("reasoning")$/    reasoning = message.get("reasoning")\n    if reasoning is None and __import__("os").environ.get("VLLM_KEEP_REASONING_CONTENT") == "1":\n        reasoning = message.get("reasoning_content")  # ADK sends thoughts under this key/' "$F"
    echo "patched: $F (backup $F.orig)"
  fi
fi

# Check: the same tool-call history, thought under reasoning_content (what ADK sends), parsed by vLLM's
# own function and rendered with the served model's chat template, with the switch off and on.
for v in 0 1; do
  VLLM_KEEP_REASONING_CONTENT=$v $PY - "$MODEL" <<'PY'
import os, sys
from transformers import AutoTokenizer
from vllm.entrypoints import chat_utils as cu
tok = AutoTokenizer.from_pretrained(sys.argv[1])
thought = "I will call get_time. Secret word PAPAYAQUOKKA."
msgs = [{"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": "What time is it?"},
        {"role": "assistant", "content": "", "reasoning_content": thought,
         "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"time": "12:34"}'}]
class _Parser:  # text-only messages: no media, so every multimodal query answers "nothing"
    def __getattr__(self, name):
        return lambda *a, **k: {}
class _Tracker:
    def create_parser(self, **kw):
        return _Parser()
conv = []
for m in msgs:
    conv += cu._parse_chat_message_content(m, _Tracker(), "string", interleave_strings=False)
for m in conv:  # the template expects tool-call arguments as objects, as vLLM passes them on
    for c in m.get("tool_calls") or []:
        if isinstance(c["function"].get("arguments"), str):
            import json; c["function"]["arguments"] = json.loads(c["function"]["arguments"])
tools = [{"type": "function", "function": {"name": "get_time", "description": "Returns the time.",
                                           "parameters": {"type": "object", "properties": {}}}}]
text = tok.apply_chat_template(conv, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=True)
print(f"VLLM_KEEP_REASONING_CONTENT={os.environ['VLLM_KEEP_REASONING_CONTENT']}: thought in prompt: {'PAPAYAQUOKKA' in text}")
PY
done
