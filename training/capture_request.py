"""Capture the exact request the agent sends to the model (tool declarations, message format), so
training examples can be rendered with the same tools the model sees at inference.

Wraps litellm.acompletion, saves the first request's `tools` and `messages` to --out, then lets the
harness run continue. Run on one task with runner/run_eval.py arguments; stop it once the file exists.

    python training/capture_request.py --out /workspace/results/agent_request.json \\
        --fresh /workspace/datasets/freshswe_run --tasks <one-task.jsonl> --agent /workspace/agents/base --results /tmp/cap
"""
import json
import sys
from pathlib import Path

import litellm

OUT = Path(sys.argv[sys.argv.index("--out") + 1])
del sys.argv[sys.argv.index("--out"):sys.argv.index("--out") + 2]
_orig = litellm.acompletion


async def acompletion(*args, **kw):
    if not OUT.exists():
        OUT.write_text(json.dumps({"tools": kw.get("tools"), "messages": kw.get("messages"),
                                   "extra": {k: v for k, v in kw.items() if k not in ("tools", "messages", "api_key")}},
                                  default=str, indent=1))
        print(f"captured request -> {OUT}", flush=True)
    return await _orig(*args, **kw)


litellm.acompletion = acompletion
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner"))
import run_eval  # noqa: E402

sys.argv = ["run_eval.py", *sys.argv[1:]]
run_eval.main()
