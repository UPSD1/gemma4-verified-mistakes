"""Build training data for the arms from the base-model runs (tracker rows 12-13).

Reads the harness output of every base run (task_results.jsonl + traces/trace_<id>.json, format
ATIF-v1.7) and writes, under --out (default data/training):

  runs.jsonl            one row per run: task, pass, resolved, grade, size, fairness flags
  arm_a_sft.jsonl       arm (a): clean successes as chat examples, split at the harness's context
                        compaction points so each example is what the model actually saw (option 2)
  dpo_same_task.jsonl   preliminary pairs for arm (d): success vs failure on the same task,
                        graded; the final arm (d) pairs come from shared branch points
  branch_points.jsonl   where to resample failed runs for the recovery splices of arms (b)/(c)/(d):
                        the first failure after the agent's first edit, within --branch-window tokens
  summary.json          counts

Chat examples use the OpenAI message format the trainer expects: system, user (bug report),
then assistant turns (reasoning text + tool calls) and tool results. Only assistant turns are
trained ("loss": true); system, user and tool messages are masked, as in every arm.

Splitting at compaction (default; --no-split keeps whole runs and drops those over --max-tokens).
The harness compacts the conversation once it passes ~14k tokens, keeping the system prompt, the
bug report, a summary and the last few events. A compaction shows in the trace as a drop in the
next step's prompt_tokens. Each segment becomes one example: system + bug report + a masked note
that earlier steps were summarized + the last --carry steps before the cut (masked, as context)
+ the segment's own steps (trained). The harness's summary text is not in the trace, so the note
stands in for it.

Tasks are kept out when their environment is broken (--exclude-prefix, default black_ and isort_)
or when both fairness raters flag them (they are almost never solved, so their failures teach
noise); pass --keep-flagged to keep them.

    python3 training/build_data.py --runs ~/Downloads/base_runs_2026-11-11 --runs data/pod_results/mined_base
"""
import argparse
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mining"))
from common import read_jsonl, write_jsonl  # noqa: E402

CHARS_PER_TOKEN = 3.5
ERROR_RE = re.compile(r"Traceback \(most recent call last\)|\bFAILED\b|\bError\b|exit_code\"?: ?[1-9]|"
                      r"\"status\": \"error\"|No such file|command not found", re.I)


def tokens(text):
    return int(len(text or "") / CHARS_PER_TOKEN)


def msg_text(m):
    return (m.get("content") or "") + (m.get("reasoning") or "")


def grade(row):
    """Graded outcome of one run (higher is better), from the harness's task_results row.
    3 solved; 2 made a patch, tests ran and failed; 1 made a patch but hit the time or turn cap;
    0 no patch. (Finer grades need per-test results: see verify/ and the plan's ranking.)"""
    if row.get("resolved"):
        return 3
    if row.get("agent_patch_size", 0) > 0:
        return 2 if not row.get("error") else 1
    return 0


def to_messages(trace):
    """ATIF steps -> chat messages with a per-message loss flag, plus a running token count."""
    msgs, total = [], 0

    def count(m):
        nonlocal total
        total += tokens(msg_text(m)) + sum(tokens(c["function"]["arguments"]) for c in m.get("tool_calls", []))
        m["tokens_so_far"] = total

    for st in trace["steps"]:
        start = len(msgs)
        src, text = st.get("source"), st.get("message") or ""
        ev = (st.get("extra") or {}).get("event_type")
        if src == "system" and ev in ("system_instruction", None) and not msgs:
            msgs.append({"role": "system", "content": text, "loss": False})
        elif src == "user":
            msgs.append({"role": "user", "content": text, "loss": False})
        elif src == "system":  # harness nudges mid-run
            msgs.append({"role": "user", "content": text, "loss": False})
        elif src == "agent":
            calls = st.get("tool_calls") or []
            # Thinking stays thinking: Gemma's template renders `reasoning` as its thought channel.
            body = {"content": "", "reasoning": text} if ev == "thinking" else {"content": text}
            msgs.append({"role": "assistant", **body, "loss": True, "step_id": st.get("step_id"),
                         "tool_calls": [{"id": c.get("tool_call_id"), "type": "function",
                                         "function": {"name": c.get("function_name"),
                                                      "arguments": json.dumps(c.get("arguments", {}))}}
                                        for c in calls]})
            obs = st.get("observation")
            if obs:
                content = obs.get("content") if isinstance(obs, dict) else obs
                content = content if isinstance(content, str) else json.dumps(content)
                msgs.append({"role": "tool", "content": content, "loss": False,
                             "tool_call_id": calls[0].get("tool_call_id") if calls else None})
        else:
            continue
        for m in msgs[start:]:
            count(m)
    return msgs, total


COMPACTION_NOTE = ("[Earlier steps of this session were compacted into a summary by the harness. "
                   "The most recent steps follow.]")


def segments(trace, msgs, carry):
    """Split a run's messages at compaction points (prompt_tokens drops by >25% between agent steps)."""
    pt = {st.get("step_id"): (st.get("metrics") or {}).get("prompt_tokens") for st in trace["steps"]}
    asst = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    cuts, prev = [], None
    for i in asst:
        cur = pt.get(msgs[i].get("step_id"))
        if cur and prev and cur < 0.75 * prev:
            cuts.append(i)
        prev = cur or prev
    head = [m for m in msgs[:asst[0]]] if asst else msgs
    bounds = [asst[0] if asst else len(msgs)] + cuts + [len(msgs)]
    out = []
    for n, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        if n == 0:
            ex = head + msgs[lo:hi]
        else:
            ctx_start = lo
            for _ in range(carry):  # carry the previous `carry` assistant steps (and their tool results)
                earlier = [i for i in asst if i < ctx_start]
                if earlier:
                    ctx_start = earlier[-1]
            ctx = [{**m, "loss": False} for m in msgs[ctx_start:lo]]
            ex = head + [{"role": "user", "content": COMPACTION_NOTE, "loss": False}] + ctx + msgs[lo:hi]
        if any(m["loss"] for m in ex):
            out.append(ex)
    return out


EDIT_TOOLS = {"edit_file", "write_file"}


def branch_point(msgs, window):
    """As in the plan: the first tool result showing a failure (failing test, traceback, rejected edit)
    AFTER the agent's first edit, within the first `window` tokens. Failures before any edit are
    mostly exploration (a missing file, an empty grep), not mistakes to recover from. Branching
    resamples the assistant step that issued the failing call, onward."""
    first_edit = next((i for i, m in enumerate(msgs) if m["role"] == "assistant"
                       and any(c["function"]["name"] in EDIT_TOOLS for c in m.get("tool_calls", []))), None)
    if first_edit is None:
        return None, "no_edit"
    for i, m in enumerate(msgs):
        if m["tokens_so_far"] > window:
            break
        if i > first_edit and m["role"] == "tool" and ERROR_RE.search(m["content"]):
            # the assistant message that issued this call
            j = max(k for k in range(i) if msgs[k]["role"] == "assistant")
            return j, "tool_error_or_failing_tests"
    # No failure after the first edit inside the window: branch at the middle of the window, after the edit.
    mids = [k for k, m in enumerate(msgs) if m["role"] == "assistant" and k >= first_edit
            and m["tokens_so_far"] <= window]
    return (mids[len(mids) // 2], "window_midpoint") if mids else (None, "none")


def load_flags():
    flags = defaultdict(int)
    for screen in ("train_sets", "mined_sets"):  # FreshSWE-Py and our mined tasks
        for r in ("opus", "sol"):
            p = ROOT / "data" / "tasks" / "screen" / f"{screen}.{r}.fairness_log.jsonl"
            if p.exists():
                for x in read_jsonl(p):
                    flags[x["instance_id"]] += x["status"] == "flagged"
    return flags


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, action="append",
                    help="base-run folder(s); every task_results.jsonl below it with a traces/ folder next to it "
                         "is one pass. Repeat for several sources, e.g. FreshSWE-Py and the mined tasks")
    ap.add_argument("--out", default=str(ROOT / "data" / "training"))
    ap.add_argument("--max-tokens", type=int, default=28000, help="drop SFT examples longer than this")
    ap.add_argument("--branch-window", type=int, default=32000,
                    help="branch only within the first N tokens (32k: splitting at compaction keeps examples short)")
    ap.add_argument("--exclude-prefix", default="black_,isort_")
    ap.add_argument("--keep-flagged", action="store_true", help="keep tasks both fairness raters flag")
    ap.add_argument("--no-split", action="store_true", help="keep whole runs (drop those over --max-tokens)")
    ap.add_argument("--carry", type=int, default=2, help="steps carried over as context after a compaction")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    skip = tuple(x for x in a.exclude_prefix.split(",") if x)
    flags = load_flags()

    runs = []
    # Training runs only: FreshSWE-Py (fresh_base) and our mined tasks (mined_base). Test-set runs
    # (OOD-45, gRPC, public) can sit in the same folders and must never become training data.
    TRAIN_SOURCES = {"fresh_base": "freshswe", "mined_base": "mined"}
    results = []
    for root in a.runs:
        for res in sorted(Path(root).expanduser().rglob("task_results.jsonl")):
            src = next((v for k, v in TRAIN_SOURCES.items() if k in res.parts), None)
            if src and (res.parent / "traces").is_dir():
                results.append((src, res))
    for src, res in results:
        pass_id = f"{src}/{res.parent.name}"
        for row in read_jsonl(res):
            tid = row["instance_id"]
            trace = res.parent / "traces" / f"trace_{tid}.json"
            runs.append({"instance_id": tid, "pass": pass_id, "resolved": bool(row["resolved"]),
                         "grade": grade(row), "error": row.get("error"), "patch_size": row.get("agent_patch_size"),
                         "flags": flags.get(tid, 0), "excluded_env": tid.startswith(skip),
                         "trace": str(trace) if trace.exists() else None})

    by_task = defaultdict(list)
    for r in runs:
        by_task[r["instance_id"]].append(r)
    kind = {t: ("always" if all(x["resolved"] for x in rs) else "never" if not any(x["resolved"] for x in rs)
                else "mixed") for t, rs in by_task.items()}
    usable = lambda r: not r["excluded_env"] and (a.keep_flagged or r["flags"] < 2)  # noqa: E731

    cache = {}

    def msgs_of(r):
        if r["trace"] not in cache:
            tr = json.load(open(r["trace"]))
            cache[r["trace"]] = (*to_messages(tr), tr)
        return cache[r["trace"]][:2]

    def trace_of(r):
        msgs_of(r)
        return cache[r["trace"]][2]

    clean = lambda ms: [{k: v for k, v in m.items() if k not in ("tokens_so_far", "step_id")} for m in ms]  # noqa: E731
    size = lambda ms: sum(tokens(msg_text(m)) + sum(tokens(c["function"]["arguments"]) for c in m.get("tool_calls", []))  # noqa: E731
                          for m in ms)

    arm_a, dropped_long = [], 0
    for r in runs:
        r["task_kind"] = kind[r["instance_id"]]
        if not r["trace"]:
            continue
        msgs, n = msgs_of(r)
        r["tokens"] = n
        if r["resolved"] and not r["excluded_env"]:
            if a.no_split:
                if n > a.max_tokens:
                    dropped_long += 1
                    continue
                arm_a.append({"instance_id": r["instance_id"], "pass": r["pass"], "segment": 0, "tokens": n,
                              "messages": clean(msgs)})
                continue
            for k, seg in enumerate(segments(trace_of(r), msgs, a.carry)):
                m = size(seg)
                if m > a.max_tokens:
                    dropped_long += 1
                    continue
                arm_a.append({"instance_id": r["instance_id"], "pass": r["pass"], "segment": k, "tokens": m,
                              "messages": clean(seg)})

    pairs = []
    for t, rs in by_task.items():
        rs = [r for r in rs if r["trace"] and usable(r)]
        for x, y in itertools.combinations(rs, 2):
            if x["grade"] == y["grade"] or 3 not in (x["grade"], y["grade"]):
                continue
            ch, rj = (x, y) if x["grade"] > y["grade"] else (y, x)
            pairs.append({"instance_id": t, "chosen": {"pass": ch["pass"], "grade": ch["grade"], "tokens": ch["tokens"]},
                          "rejected": {"pass": rj["pass"], "grade": rj["grade"], "tokens": rj["tokens"]},
                          "fits": max(ch["tokens"], rj["tokens"]) <= a.max_tokens})

    branches = []
    for r in runs:
        if r["resolved"] or not r["trace"] or not usable(r) or r["task_kind"] == "always":
            continue
        msgs, _ = msgs_of(r)
        j, why = branch_point(msgs, a.branch_window)
        if j is None:
            continue
        branches.append({"instance_id": r["instance_id"], "pass": r["pass"], "task_kind": r["task_kind"],
                         "grade": r["grade"], "flags": r["flags"], "branch_message_index": j,
                         "branch_step_id": msgs[j].get("step_id"), "prefix_tokens": msgs[j]["tokens_so_far"],
                         "reason": why, "trace": r["trace"]})

    write_jsonl(out / "runs.jsonl", runs)
    write_jsonl(out / "arm_a_sft.jsonl", arm_a)
    write_jsonl(out / "dpo_same_task.jsonl", pairs)
    write_jsonl(out / "branch_points.jsonl", branches)
    summary = {
        "runs": len(runs), "tasks": len(by_task), "task_kinds": dict(Counter(kind.values())),
        "excluded_env_tasks": sum(1 for t in by_task if t.startswith(skip)),
        "flagged_by_both_tasks": sum(1 for t in by_task if flags.get(t, 0) >= 2),
        "arm_a_examples": len(arm_a), "arm_a_tasks": len({x["instance_id"] for x in arm_a}),
        "arm_a_runs": len({(x["instance_id"], x["pass"]) for x in arm_a}),
        "arm_a_split_at_compaction": not a.no_split,
        "arm_a_dropped_over_max_tokens": dropped_long,
        "arm_a_trained_tokens": sum(size([m for m in x["messages"] if m["loss"]]) for x in arm_a),
        "dpo_same_task_pairs": len(pairs), "dpo_pairs_fitting_max_tokens": sum(p["fits"] for p in pairs),
        "branch_points": len(branches), "branch_points_by_task_kind": dict(Counter(b["task_kind"] for b in branches)),
        "branch_reasons": dict(Counter(b["reason"] for b in branches)),
        "grades": dict(Counter(r["grade"] for r in runs)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
