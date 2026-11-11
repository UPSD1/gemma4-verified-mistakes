source /workspace/env.sh >/dev/null 2>&1; deactivate 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=1
/workspace/trainenv/bin/python - <<PY
import os
from huggingface_hub import snapshot_download
p = snapshot_download("google/gemma-4-12B-it-qat-q4_0-unquantized",
                      local_dir="/workspace/models/gemma-4-12b-it-qat-unquantized")
print("WEIGHTS_OK", p, flush=True)
PY
