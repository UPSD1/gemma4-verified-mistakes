"""Resample failed runs from their branch points (recovery splices for arms (b)-(d)).

A branch point (training/build_data.py -> branch_points.jsonl) is the agent step whose tool call failed
(a rejected edit or a failing command) after its first edit. For each one, in the competition harness:
  1. start a fresh sandbox for the task exactly as a normal run does;
  2. replay every state-changing call the agent made up to AND INCLUDING the failing step, with the
     agent's own tool functions (swegemma.tools), so files and environment are as they were right
     after the mistake;
  3. load into the agent's session what the model actually had in context at that point: the task
     prompt, then (if the harness had compacted the conversation by then) a compaction note and the
     steps since the last compaction, ending with the failing step and its error output. This keeps
     the loaded context within what the model saw (compaction keeps it under ~16k tokens);
  4. let the base model continue (the harness's own continuation nudge starts the turn), with the
     normal budgets; the harness grades the final patch as usual (task_results.jsonl, traces/).

A continuation that resolves the task, joined to the recorded prefix, is a mistake -> recovery splice
(training/build_splices.py). Run once per sample (--sample k) for several continuations per branch point.

    python training/resample.py --branches data/training/branch_points.jsonl --sample 1 --pick 0 \\
        --fresh /workspace/datasets/freshswe_run --agent /workspace/agents/base \\
        --results /workspace/results/resample/s1_p0 --concurrency 24 [--limit 2]
"""
import argparse
import contextvars
import glob
import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_data import COMPACTION_NOTE, MAIN_AGENT, author, scrub, scrub_args  # noqa: E402

log = logging.getLogger("resample")
BRANCH = {}  # instance_id -> {"replay": steps, "context": steps, "compacted": bool}
CURRENT = contextvars.ContextVar("resample_task", default=None)
NUDGE = ("Please continue your work using the available tools, or call submit_patch when you have "
         "completed and verified your changes.")  # the harness's continuation nudge, verbatim
CARRY = 2  # model calls kept from before a compaction cut (same as training/build_data.py segments)


def prefix_steps(trace_path, branch_step_id):
    """(replay steps, context steps, compacted) for a branch point.

    replay: every main-agent step up to and including the failing step (and the steps of the same model
    call that follow it without their own metrics). context: the part of that the model had in context
    at the failing step: everything if no compaction happened yet, else the steps since the last
    compaction cut plus CARRY earlier model calls."""
    trace = trace_path if isinstance(trace_path, dict) else json.load(open(trace_path))
    steps = [st for st in trace["steps"]
             if st.get("source") == "agent" and author(st) == MAIN_AGENT]
    b = next((i for i, st in enumerate(steps) if st.get("step_id") == branch_step_id), None)
    if b is None:
        raise ValueError(f"branch step {branch_step_id} not in {trace_path}")
    e = b + 1
    while e < len(steps) and not steps[e].get("metrics"):
        e += 1
    replay = steps[:e]
    calls = [i for i, st in enumerate(replay) if st.get("metrics")]
    cuts, prev = [], None
    for i in calls:
        cur = replay[i]["metrics"].get("prompt_tokens")
        if cur and prev and cur < 0.75 * prev:
            cuts.append(i)
        prev = cur or prev
    if not cuts:
        return replay, replay, False
    earlier = [i for i in calls if i < cuts[-1]]
    start = earlier[-CARRY] if len(earlier) >= CARRY else (earlier[0] if earlier else cuts[-1])
    return replay, replay[start:], True


def context_tokens(steps):
    """Estimated tokens of the steps loaded as context (text, tool calls and their results)."""
    return int(sum(len(s.get("message") or "") + len(json.dumps(s.get("tool_calls") or []))
                   + len(json.dumps(s.get("observation") or "")) for s in steps) / 3.5)


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
    """Recorded steps -> ADK session events: one model event per model call (its thinking, text and
    tool calls), followed by the tool results. Leaked control tokens are removed from the text."""
    from google.adk.events import Event
    from google.genai import types as gt
    events, parts, responses = [], [], []

    def flush():
        if parts:
            events.append(Event(author=agent_name, content=gt.Content(role="model", parts=list(parts))))
        for r in responses:
            events.append(Event(author=agent_name, content=gt.Content(role="user", parts=[r])))
        parts.clear()
        responses.clear()

    for st in steps:
        if st.get("metrics") and parts:
            flush()
        calls = st.get("tool_calls") or []
        text = scrub(st.get("message") or "")
        if text:
            thought = (st.get("extra") or {}).get("event_type") == "thinking"
            parts.append(gt.Part(text=text, thought=thought or None))
        for c in calls:
            parts.append(gt.Part(function_call=gt.FunctionCall(id=c.get("tool_call_id"), name=c.get("function_name"),
                                                               args=scrub_args(c.get("arguments") or {}))))
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
            responses.append(gt.Part(function_response=gt.FunctionResponse(
                id=c.get("tool_call_id"), name=c.get("function_name"), response=resp)))
    flush()
    return events


def install_patches():
    from swegemma import context as sc
    from swegemma.harness import agent_runner as ar

    original_start = sc.SwegemmaContext.start_agent_session

    def start_agent_session(self):
        tid = getattr(getattr(self, "task", None), "instance_id", None)
        if tid in BRANCH:
            n = replay(self, BRANCH[tid]["replay"])
            log.warning("resample %s: replayed %d tool calls through the failing step; %d of %d steps loaded "
                        "as context%s", tid, n, len(BRANCH[tid]["context"]), len(BRANCH[tid]["replay"]),
                        " (after compaction)" if BRANCH[tid]["compacted"] else "")
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
                from google.genai import types as gt
                await self.session_service.append_event(session, Event(author="user", content=new_message))
                if BRANCH[tid]["compacted"]:
                    await self.session_service.append_event(session, Event(author="user", content=gt.Content(
                        role="user", parts=[gt.Part(text=COMPACTION_NOTE)])))
                for ev in to_events(BRANCH[tid]["context"], agent_name):
                    await self.session_service.append_event(session, ev)
                # ADK needs a new message to run; the harness's own mid-run nudge keeps this in-distribution.
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
    ap.add_argument("--max-context", type=int, default=16000,
                    help="skip branch points whose loaded context is longer than this (estimated tokens). The "
                         "harness compacts at ~14k, so the model never saw more; longer replays could overflow")
    ap.add_argument("--limit", type=int)
    a, rest = ap.parse_known_args()
    logging.basicConfig(level=logging.WARNING)

    # Only branch points whose task is in this dataset (branch_points.jsonl holds FreshSWE-Py and mined
    # tasks; each dataset is resampled with its own --fresh folder).
    fresh = rest[rest.index("--fresh") + 1]
    tasks = {json.loads(l)["instance_id"]: l for l in open(Path(fresh) / "tasks.jsonl")}
    branches = [b for b in (json.loads(l) for l in open(a.branches) if l.strip()) if b["instance_id"] in tasks]
    # One branch per task per harness run (instance ids must be unique): pick per sample round-robin.
    by_task = {}
    for b in branches:
        by_task.setdefault(b["instance_id"], []).append(b)
    chosen, missing, too_long = [], 0, 0
    for tid, bs in sorted(by_task.items()):
        bs = sorted(bs, key=lambda x: x["pass"])
        if a.pick >= len(bs):
            continue
        b = bs[a.pick]
        tr = find_trace(b["trace"], a.trace_root)
        if tr is None:
            missing += 1
            continue
        rp, cx, comp = prefix_steps(tr, b["branch_step_id"])
        if context_tokens(cx) > a.max_context:
            too_long += 1
            continue
        BRANCH[tid] = {"replay": rp, "context": cx, "compacted": comp}
        chosen.append({**b, "trace": str(tr)})
    if a.limit:
        chosen = chosen[: a.limit]
        BRANCH = {b["instance_id"]: BRANCH[b["instance_id"]] for b in chosen}
    print(f"sample {a.sample}, pick {a.pick}: {len(chosen)} branch points ({missing} traces not found, "
          f"{too_long} over --max-context)", flush=True)
    if not chosen:
        sys.exit(0)

    # Task file for the harness: the chosen tasks only.
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
