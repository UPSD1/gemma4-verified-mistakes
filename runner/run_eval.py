"""Run the competition harness exactly as the official starter notebook configures it,
against a vLLM server that is already running (runpod/setup.sh or /workspace/serve.sh).

The `swegemma eval` CLI leaves out context caching, event compaction (14,336-token
threshold), the code-graph data and the test wheels. Without compaction a 12B run hits the
32k context limit after ~16k tokens, so every run fails. This script passes all of them.

    python runner/run_eval.py --tasks /workspace/comp/tasks.jsonl --agent /workspace/agents/base \
        --results /workspace/results/smoke --concurrency 8 --limit 5
"""
import argparse
import asyncio
import json
import os
import random
import tempfile
from pathlib import Path

import litellm
import yaml
from adk_submission import VllmConfig, VllmServer, discover_adapters
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator

litellm.drop_params = True
MODEL = "gemma-4-12b-it-qat-w4a16-ct"


def load_budget(agent_dir):
    path = Path(agent_dir) / "eval_config.yaml"
    raw = yaml.safe_load(path.read_text()) if path.is_file() else {}
    ev = (raw or {}).get("evaluation", raw or {})
    turns = ev.get("max_turns", ev.get("max_llm_calls"))
    return dict(
        timeout_seconds=int(ev.get("timeout_seconds", 300)),
        verification_timeout_seconds=int(ev.get("verification_timeout_seconds", 300)),
        max_tool_calls=int(ev.get("max_tool_calls", 100)),
        max_time_minutes=float(ev.get("max_time_minutes", 60.0)),
        max_turns=int(turns) if turns is not None else None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks")
    ap.add_argument("--agent", required=True, help="submission directory (agent.yaml, prompts, adapters/)")
    ap.add_argument("--results", required=True)
    ap.add_argument("--comp", default="/workspace/comp", help="competition data (snapshots, graphs, embeddings, wheels)")
    ap.add_argument("--snapshots", help="snapshots dir if not under --comp (FreshSWE-Py, OOD-45)")
    ap.add_argument("--wheels", help="wheels dir if not under --comp")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0, help="task-order shuffle seed (only with --limit)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ood", help="OOD-45 dataset dir: use its tools/ood45_harness.py (per-repo wheels, "
                                  "subprocess dependency install); --tasks is ignored, pick a repo with --repo")
    ap.add_argument("--repo", help="with --ood: run only this repo (e.g. kedro-org/kedro)")
    ap.add_argument("--skip", default="urllib3_5279", help="comma-separated instance ids to leave out")
    a = ap.parse_args()

    comp = Path(a.comp)
    if a.ood:
        import sys
        sys.path.insert(0, str(Path(a.ood) / "tools"))
        import ood45_harness as ood
        run = ood.setup(a.ood, wheels_tmp=Path(a.results) / "_wheels")
        skip = set(filter(None, a.skip.split(",")))
        rows = [r for i, r in run.meta.items() if i not in skip and (not a.repo or r["repo"] == a.repo)]
        if not rows:
            raise SystemExit(f"no OOD-45 tasks for repo {a.repo}")
        tmp = Path(tempfile.mkdtemp()) / "tasks.jsonl"
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        a.tasks, a.snapshots = str(tmp), str(run.snapshots)
        a.wheels = str(run.wheels_for(rows[0]["repo"]))
    tasks_path = Path(a.tasks)
    if a.limit:
        rows = [json.loads(l) for l in open(tasks_path)]
        random.Random(a.seed).shuffle(rows)
        tmp = Path(tempfile.mkdtemp()) / "tasks.jsonl"
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows[: a.limit]))
        tasks_path = tmp

    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    adapters = discover_adapters(str(a.agent), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    # Not started: only used to build LiteLLM clients pointing at the running server.
    server = VllmServer(VllmConfig(model=MODEL, port=a.port, host="127.0.0.1"), adapter_manifest=adapters)
    models = server.create_model_registry(aliases=[MODEL], model_prefix="openai/", api_key="EMPTY")
    limits, gen = build_submission_limits()

    cfg = EvalConfig(
        tasks_path=tasks_path,
        snapshots_dir=Path(a.snapshots) if a.snapshots else comp / "snapshots",
        results_dir=Path(a.results),
        submission_dir=Path(a.agent),
        models=models,
        sandbox="subprocess",
        limits=limits,
        generation_constraints=gen,
        adapter_manifest=adapters,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
        # OOD-45 ships no code graphs; point at the competition dirs, which have none for its repos.
        graph_dir=str(comp / "graphs"),
        embeddings_dir=str(comp / "embeddings"),
        wheels_dir=Path(a.wheels) if a.wheels else comp / "wheels",
        concurrency=a.concurrency,
        display_mode="quiet",
        **load_budget(a.agent),
    )
    result = Evaluator(cfg).run()
    if asyncio.iscoroutine(result):
        asyncio.run(result)


if __name__ == "__main__":
    main()
