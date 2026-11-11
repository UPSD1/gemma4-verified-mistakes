"""Check that tasks resolve with their gold patch (and fail with an empty one) in the subprocess sandbox,
using the harness's own verify_task. Run before collecting trajectories on a new task set.

    python runner/verify_gold.py --fresh /workspace/datasets/freshswe --agent /workspace/agents/base \
        --results /workspace/results/gold_fresh --per-repo 1
"""
import argparse
import asyncio
import dataclasses
import json
import random
import sys
from pathlib import Path

from swegemma.config import EvalConfig
from swegemma.evaluate import Evaluator
from swegemma.harness.verification import verify_task
from swegemma.models import Task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--ids", help="comma-separated instance ids (default: sample)")
    ap.add_argument("--per-repo", type=int, default=1, help="tasks sampled per repo")
    ap.add_argument("--empty", action="store_true", help="also check that an empty patch fails")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    import freshswe_env
    freshswe_env.patch(a.fresh)

    fresh = Path(a.fresh)
    rows = [json.loads(l) for l in open(fresh / "tasks.jsonl")]
    if a.ids:
        want = set(a.ids.split(","))
        rows = [r for r in rows if r["instance_id"] in want]
    else:
        random.Random(a.seed).shuffle(rows)
        seen, pick = {}, []
        for r in rows:
            if seen.get(r["repo"], 0) < a.per_repo:
                seen[r["repo"]] = seen.get(r["repo"], 0) + 1
                pick.append(r)
        rows = pick

    fields = {f.name for f in dataclasses.fields(Task)}
    results = Path(a.results)
    results.mkdir(parents=True, exist_ok=True)
    from adk_submission import VllmConfig, VllmServer
    models = VllmServer(VllmConfig(model="unused", port=8000, host="127.0.0.1")).create_model_registry(
        aliases=["unused"], model_prefix="openai/", api_key="EMPTY")  # no model call during verification
    cfg = EvalConfig(models=models, tasks_path=fresh / "tasks.jsonl", snapshots_dir=fresh / "snapshots", results_dir=results,
                     submission_dir=Path(a.agent), sandbox="subprocess", graph_dir=str(fresh / "graphs"),
                     embeddings_dir=str(fresh / "embeddings"))
    ev = Evaluator(cfg)
    sem = asyncio.Semaphore(a.concurrency)

    async def one(r, patch, label):
        task = Task(**{k: v for k, v in r.items() if k in fields})
        async with sem:
            try:
                res = await verify_task(ev.docker, cfg, task, fresh / "snapshots" / f"{r['instance_id']}.tgz",
                                        agent_patch=patch)
                out = dict(id=r["instance_id"], kind=label, resolved=res.resolved, error=res.error,
                           exit=res.test_exit_code, tail=(res.test_output or "")[-600:])
            except Exception as e:  # noqa: BLE001
                out = dict(id=r["instance_id"], kind=label, resolved=None, error=repr(e)[:600])
        print(json.dumps({k: out[k] for k in ("id", "kind", "resolved", "error")}), flush=True)
        return out

    async def run():
        jobs = [one(r, r["patch"], "gold") for r in rows]
        if a.empty:
            jobs += [one(r, "", "empty") for r in rows]
        return await asyncio.gather(*jobs)

    out = asyncio.run(run())
    (results / "gold_check.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))
    gold = [o for o in out if o["kind"] == "gold"]
    print(f"gold resolved {sum(bool(o['resolved']) for o in gold)}/{len(gold)}")
    if a.empty:
        emp = [o for o in out if o["kind"] == "empty"]
        print(f"empty resolved {sum(bool(o['resolved']) for o in emp)}/{len(emp)} (should be 0)")


if __name__ == "__main__":
    main()
