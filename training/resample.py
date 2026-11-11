"""Resample failed runs from their branch points (recovery splices for arms (b)-(d)).

For each branch point (training/build_data.py -> branch_points.jsonl), in the competition harness:
  1. start a fresh sandbox for the task exactly as a normal run does;
  2. replay the run's recorded tool calls up to the branch point with the agent's own tool
     functions (swegemma.tools), so files and environment match what the agent had done;
  3. load those recorded steps into the agent's ADK session, then let the base model continue
     from there with the normal budgets;
  4. the harness grades the final patch as usual (task_results.jsonl, traces/).

A continuation that resolves the task, joined to the recorded prefix, is a recovery splice.
Run it once per sample (--sample k) to get several continuations per branch point.

    python training/resample.py --branches data/training/branch_points.jsonl --sample 1 --pick 0 \\
        --fresh /workspace/datasets/freshswe --agent /workspace/agents/base \\
        --results /workspace/results/resample/s1 --concurrency 16 [--limit 2]

Traces referenced in branch_points.jsonl are looked up by file name under --trace-root
(default /workspace/results/*/fresh_base or /workspace/results/fresh_base).
"""
import argparse
import contextvars
import glob
import json
import logging
import sys
import tempfile
from pathlib import Path

log = logging.getLogger("resample")
BRANCH = {}  # instance_id -> recorded steps before the branch point
CURRENT = contextvars.ContextVar("resample_task", default=None)
NUDGE = ("Please continue your work using the available tools, or call submit_patch when you have "
         "completed and verified your changes.")  # the harness's continuation nudge, verbatim


def prefix_steps(trace_path, branch_step_id):
    """Recorded agent steps (with their tool calls and results) before the branch step."""
    steps = json.load(open(trace_path))["steps"]
    out = []
    for st in steps:
        if st.get("source") != "agent":
            continue
        if st.get("step_id") == branch_step_id:
            break
        out.append(st)
    return out


def replay(ctx, steps):
    """Redo the recorded state-changing tool calls in the fresh sandbox, with the agent's own tools."""
    from swegemma.tools import execution, workspace
    n = 0
    for st in steps:
        for call in st.get("tool_calls") or []:
            name, args = call.get("function_name"), call.get("arguments") or {}
            try:
                if name == "write_file":
                    workspace.write_file(ctx, args["filepath"], args["content"])
                elif name == "edit_file":
                    workspace.edit_file(ctx, args["filepath"], args["old_string"], args["new_string"],
                                        args.get("allow_multiple", False))
                elif name == "run_command":
                    execution.run_command(ctx, args["command"])
                else:
                    continue  # read-only tools need no replay
                n += 1
            except Exception as e:  # noqa: BLE001 - a failed recorded call failed then too
                log.debug("replay %s failed: %s", name, e)
    return n


def to_events(steps, agent_name):
    """Recorded steps -> ADK session events (model turn with its tool calls, then the tool results)."""
    from google.adk.events import Event
    from google.genai import types as gt
    events = []
    for st in steps:
        calls = st.get("tool_calls") or []
        parts = []
        if st.get("message"):
            thought = (st.get("extra") or {}).get("event_type") == "thinking"
            parts.append(gt.Part(text=st["message"], thought=thought or None))
        for c in calls:
            parts.append(gt.Part(function_call=gt.FunctionCall(id=c.get("tool_call_id"), name=c.get("function_name"),
                                                               args=c.get("arguments") or {})))
        if parts:
            events.append(Event(author=agent_name, content=gt.Content(role="model", parts=parts)))
        obs = st.get("observation")
        if calls and obs:
            raw = obs.get("content") if isinstance(obs, dict) else obs
            try:
                resp = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, ValueError):
                resp = {"result": raw}
            if not isinstance(resp, dict):
                resp = {"result": resp}
            c = calls[0]
            events.append(Event(author=agent_name, content=gt.Content(role="user", parts=[gt.Part(
                function_response=gt.FunctionResponse(id=c.get("tool_call_id"), name=c.get("function_name"),
                                                      response=resp))])))
    return events


def install_patches():
    from swegemma import context as sc
    from swegemma.harness import agent_runner as ar

    original_start = sc.SwegemmaContext.start_agent_session

    def start_agent_session(self):
        tid = getattr(getattr(self, "task", None), "instance_id", None)
        if tid in BRANCH:
            n = replay(self, BRANCH[tid])
            log.warning("resample %s: replayed %d tool calls from %d steps", tid, n, len(BRANCH[tid]))
            CURRENT.set(tid)
        return original_start(self)

    sc.SwegemmaContext.start_agent_session = start_agent_session

    BaseRunner = ar.Runner

    class BranchRunner(BaseRunner):
        _primed = set()

        async def run_async(self, *, user_id, session_id, new_message=None, **kw):
            tid = CURRENT.get()
            if tid in BRANCH and session_id not in self._primed:
                self._primed.add(session_id)
                session = await self.session_service.get_session(app_name=self.app_name, user_id=user_id,
                                                                 session_id=session_id)
                from google.adk.events import Event
                agent_name = getattr(self.agent, "name", "agent")
                await self.session_service.append_event(session, Event(author="user", content=new_message))
                for ev in to_events(BRANCH[tid], agent_name):
                    await self.session_service.append_event(session, ev)
                # ADK needs a new message to run; the harness's own mid-run nudge keeps this in-distribution.
                from google.genai import types as gt
                new_message = gt.Content(role="user", parts=[gt.Part(text=NUDGE)])
            async for ev in super().run_async(user_id=user_id, session_id=session_id, new_message=new_message, **kw):
                yield ev

    ar.Runner = BranchRunner


def find_trace(path, roots):
    p = Path(path)
    if p.exists():
        return p
    parts = p.parts
    anchor = next((x for x in ("fresh_base", "mined_base") if x in parts), None)
    rel = Path(*parts[parts.index(anchor) + 1:]) if anchor else Path(p.name)
    for root in roots:
        for cand in glob.glob(str(Path(root) / rel)):
            return Path(cand)
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", required=True)
    ap.add_argument("--sample", type=int, default=1, help="sample number (a label; each sample is a fresh continuation)")
    ap.add_argument("--pick", type=int, default=0,
                    help="which of a task's branch points to run (0-based; a task can have one per base pass). "
                         "The harness needs unique task ids per run, so run --pick 0..3 to cover them all")
    ap.add_argument("--trace-root", action="append",
                    default=["/workspace/results/fresh_base", "/workspace/results/fresh_base_h100",
                             "/workspace/results/mined_base"])
    ap.add_argument("--limit", type=int)
    a, rest = ap.parse_known_args()
    logging.basicConfig(level=logging.WARNING)

    branches = [json.loads(l) for l in open(a.branches) if l.strip()]
    # One branch per task per harness run (instance ids must be unique): pick per sample round-robin.
    by_task = {}
    for b in branches:
        by_task.setdefault(b["instance_id"], []).append(b)
    chosen, missing = [], 0
    for tid, bs in sorted(by_task.items()):
        bs = sorted(bs, key=lambda x: x["pass"])
        if a.pick >= len(bs):
            continue
        b = bs[a.pick]
        tr = find_trace(b["trace"], a.trace_root)
        if tr is None:
            missing += 1
            continue
        BRANCH[tid] = prefix_steps(tr, b["branch_step_id"])
        chosen.append({**b, "trace": str(tr)})
    if a.limit:
        chosen = chosen[: a.limit]
        BRANCH = {b["instance_id"]: BRANCH[b["instance_id"]] for b in chosen}
    print(f"sample {a.sample}, pick {a.pick}: {len(chosen)} branch points ({missing} traces not found)", flush=True)

    # Task file for the harness: the chosen tasks only.
    fresh = rest[rest.index("--fresh") + 1]
    tasks = {json.loads(l)["instance_id"]: l for l in open(Path(fresh) / "tasks.jsonl")}
    tmp = Path(tempfile.mkdtemp()) / "tasks.jsonl"
    tmp.write_text("".join(tasks[b["instance_id"]] for b in chosen if b["instance_id"] in tasks))
    results = Path(rest[rest.index("--results") + 1])
    results.mkdir(parents=True, exist_ok=True)
    (results / "branches.jsonl").write_text("".join(json.dumps(b) + "\n" for b in chosen))

    install_patches()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner"))
    import run_eval
    sys.argv = ["run_eval.py", "--tasks", str(tmp), *rest]
    run_eval.main()
