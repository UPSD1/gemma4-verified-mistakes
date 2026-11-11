"""Make a submission dir for one trained arm: a copy of the base agent plus its LoRA adapter.

    python runpod/make_arm_agent.py --base /workspace/agents/base --adapter /workspace/adapters/arm_a \
        --out /workspace/agents/arm_a [--sub-agents]

The adapter dir is the PEFT output (adapter_config.json + adapter_model.safetensors). It is copied to
<out>/adapters/<arm>/, and agent.yaml gets `adapter: <name>` with the name the harness discovers.
Prints `<name>=<path>` for vLLM's --lora-modules. By default only the main agent uses the adapter
(the code_analyzer sub-agent stays on the base model); --sub-agents applies it there too.
"""
import argparse
import re
import shutil
from pathlib import Path

from adk_submission import discover_adapters
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS


def set_adapter(yaml_path, name):
    text = yaml_path.read_text()
    text = re.sub(r"(?m)^adapter:.*\n", "", text)
    if not re.search(r"(?m)^model:", text):
        raise SystemExit(f"{yaml_path}: no top-level model: line")
    yaml_path.write_text(re.sub(r"(?m)^(model:.*\n)", rf"\1adapter: {name}\n", text, count=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True, help="PEFT adapter dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sub-agents", action="store_true")
    a = ap.parse_args()
    src, out = Path(a.adapter), Path(a.out)
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        if not (src / f).is_file():
            raise SystemExit(f"missing {src / f}")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(a.base, out, ignore=shutil.ignore_patterns("adapters"))
    dst = out / "adapters" / out.name
    dst.mkdir(parents=True)
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(src / f, dst / f)
    found = discover_adapters(str(out), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS).adapters
    if len(found) != 1:
        raise SystemExit(f"expected one adapter in {out}, found {list(found)}")
    name, info = next(iter(found.items()))
    set_adapter(out / "agent.yaml", name)
    if a.sub_agents:
        for y in (out / "sub_agents").glob("*.yaml"):
            set_adapter(y, name)
    print(f"{name}={Path(info.path).parent}")


if __name__ == "__main__":
    main()
