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
                        (only the context since the last compaction is loaded when resampling)
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
MAIN_AGENT = "swe_baseline_agent"  # the agent's own steps; sub-agent (code_analyzer_agent) steps are left out
# Gemma control tokens that leaked into trace text as literals. vLLM's gemma4 reasoning parser splits on
# them, so a model trained to write them could have its answer or tool call swallowed as reasoning.
CTRL_SPAN = re.compile(r"<\|channel>.*?<channel\|>", re.S)
CTRL_TOK = re.compile(r"<\|[a-z_]+>|<[a-z_]+\|>")
# ADK appends this to the system instruction at runtime (seen in a captured request); traces omit it.
SYSTEM_SUFFIX = '\n\n\nYou are an agent. Your internal name is "swe_baseline_agent".'
MAX_MARKERS_PER_RUN = 4  # runs that repeat control tokens more than this are dropped
SEARCH_CMDS = {"grep", "rg", "egrep", "find", "ls", "cat", "head", "tail", "wc", "git", "which", "test", "["}
EDIT_TOOLS = {"edit_file", "write_file"}


def tokens(text):
    return int(len(text or "") / CHARS_PER_TOKEN)


def msg_text(m):
    return (m.get("content") or "") + (m.get("reasoning") or "")


def scrub(text):
    """Remove leaked control tokens (and any thought span they enclose) from trace text."""
    return CTRL_TOK.sub("", CTRL_SPAN.sub("", text or ""))


def scrub_args(v):
    """scrub() applied to every string inside tool-call arguments."""
    if isinstance(v, str):
        return scrub(v)
    if isinstance(v, dict):
        return {k: scrub_args(x) for k, x in v.items()}
    if isinstance(v, list):
        return [scrub_args(x) for x in v]
    return v


def marker_count(trace):
    """Leaked control tokens in the main agent's text and tool-call arguments."""
    n = 0
    for st in trace["steps"]:
        if st.get("source") == "agent" and author(st) == MAIN_AGENT:
            n += len(CTRL_TOK.findall(st.get("message") or ""))
            n += sum(len(CTRL_TOK.findall(json.dumps(c.get("arguments", {})))) for c in st.get("tool_calls") or [])
    return n


def author(st):
    return (st.get("extra") or {}).get("author") or MAIN_AGENT


def grade(row):
    """Graded outcome of one run (higher is better), from the harness's task_results row.
    3 solved; 2 made a patch, tests ran and failed; 1 made a patch but hit the time or turn cap;
    0 no patch. (Finer grades need per-test results: see verify/ and the plan's ranking.)"""
    if row.get("resolved"):
        return 3
    if row.get("agent_patch_size", 0) > 0:
        return 2 if not row.get("error") else 1
    return 0


def parse_obs(content):
    try:
        d = json.loads(content) if isinstance(content, str) else content
    except ValueError:
        return {}
    return d if isinstance(d, dict) else {}


def is_failure(call, content):
    """A mistake the agent should recover from, read from the tool result's own fields:
    a rejected edit, or a command that failed (not a search that found nothing).
    A missing file in read_file is exploration, not a mistake."""
    d = parse_obs(content)
    name = (call or {}).get("function", {}).get("name")
    failed = d.get("status") == "error" or "error" in d
    if name in EDIT_TOOLS:
        return failed
    if name == "run_command":
        try:
            cmd = json.loads(call["function"]["arguments"]).get("command", "")
        except (ValueError, KeyError, AttributeError):
            cmd = ""
        # Test runs: read the result itself, since `pytest ... | tail` exits 0 even when tests fail.
        out = json.dumps(d.get("details") or {}) + str(d.get("stdout") or "") + str(d.get("stderr") or "")
        if re.search(r"\bpytest\b|\bunittest\b|\btox\b|\bnox\b", cmd):
            return bool(failed or re.search(r"\b\d+ (failed|errors?)\b|^FAILED |^ERROR |Traceback \(most recent", out, re.M))
        # Other commands: the exit code comes from the last command of the last pipeline
        # (`cd x && pip list | grep y` exits with grep's code: a search that found nothing).
        last = re.split(r"&&|;|\|\|", cmd)[-1].split("|")[-1].strip()
        first = last.split()[0] if last else ""
        if first in SEARCH_CMDS:
            return False
        if failed:
            return True
        return d.get("exit_code") not in (0, None)
    return False


def to_messages(trace):
    """ATIF steps -> chat messages with a per-message loss flag, plus a running token count.

    One assistant message per model call: a step without its own metrics (a text or final event, or an
    extra tool call) belongs to the call before it and is merged into it, with its tool results after.
    Only the main agent's steps are used; control tokens leaked into the text are removed."""
    msgs, total, pending = [], 0, None

    def count(m):
        nonlocal total
        total += tokens(msg_text(m)) + sum(tokens(c["function"]["arguments"]) for c in m.get("tool_calls", []))
        m["tokens_so_far"] = total

    def flush():
        nonlocal pending
        if pending is None:
            return
        a, results = pending
        if not a.get("reasoning"):
            a.pop("reasoning", None)
        msgs.append(a)
        count(a)
        for call_id, content in results:
            t = {"role": "tool", "content": content, "loss": False, "tool_call_id": call_id}
            msgs.append(t)
            count(t)
        pending = None

    for st in trace["steps"]:
        src, text = st.get("source"), st.get("message") or ""
        ev = (st.get("extra") or {}).get("event_type")
        if src == "agent":
            if author(st) != MAIN_AGENT:
                continue  # sub-agent turns ran in their own context
            text = scrub(text)
            calls = [{"id": c.get("tool_call_id"), "type": "function",
                      "function": {"name": c.get("function_name"),
                                   "arguments": json.dumps(scrub_args(c.get("arguments", {})))}}
                     for c in st.get("tool_calls") or []]
            results = []
            obs = st.get("observation")
            if obs and calls:
                content = obs.get("content") if isinstance(obs, dict) else obs
                results.append((calls[0]["id"], content if isinstance(content, str) else json.dumps(content)))
            if pending is not None and not st.get("metrics"):
                a, res = pending  # same model call as the previous step
                if ev == "thinking":
                    a["reasoning"] = ((a.get("reasoning") or "") + "\n" + text).strip()
                else:
                    a["content"] = ((a.get("content") or "") + "\n" + text).strip()
                a["tool_calls"] += calls
                res += results
                continue
            flush()
            body = {"content": "", "reasoning": text} if ev == "thinking" else {"content": text}
            pending = ({"role": "assistant", **body, "loss": True, "step_id": st.get("step_id"),
                        "tool_calls": calls}, results)
            continue
        flush()
        if src == "system" and ev in ("system_instruction", None) and not msgs:
            sysmsg = text if text.rstrip().endswith(SYSTEM_SUFFIX.strip()) else text.rstrip("\n") + SYSTEM_SUFFIX
            msgs.append({"role": "system", "content": sysmsg, "loss": False})
        elif src in ("user", "system"):  # task prompt; harness nudges mid-run
            msgs.append({"role": "user", "content": text, "loss": False})
        else:
            continue
        count(msgs[-1])
    flush()
    return msgs, total


COMPACTION_NOTE = ("[Earlier steps of this session were compacted into a summary by the harness. "
                   "The most recent steps follow.]")


def segments(trace, msgs, carry):
    """Split a run's messages at compaction points (prompt_tokens drops by >25% between model calls).

    After a cut the harness's summary is not in the trace, so the first agent turn of that segment
    (which may rely on facts only the summary held) is kept as context but not trained."""
    pt = {st.get("step_id"): (st.get("metrics") or {}).get("prompt_tokens") for st in trace["steps"]
          if st.get("source") == "agent" and author(st) == MAIN_AGENT}
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
            body = [dict(m) for m in msgs[lo:hi]]
            first = next((m for m in body if m["role"] == "assistant"), None)
            if first is not None:
                first["loss"] = False
            ex = head + [{"role": "user", "content": COMPACTION_NOTE, "loss": False}] + ctx + body
        if any(m["loss"] for m in ex):
            out.append(ex)
    return out


def cap_segments(segs, k):
    """At most k segments per run: the first, the last (where the run finishes) and evenly spaced ones."""
    if len(segs) <= k:
        return list(enumerate(segs))
    idx = sorted({0, len(segs) - 1} | {round(i * (len(segs) - 1) / (k - 1)) for i in range(k)})[:k]
    return [(i, segs[i]) for i in idx]


def branch_point(msgs, window):
    """As in the plan: the first tool result showing a failure (a rejected edit or a failing command,
    not a search that found nothing) AFTER the agent's first edit, within the first `window` tokens.
    Returns the index of the assistant message whose call failed; the splice keeps that step and its
    error, and the resampled continuation recovers from it. No failure in the window: no branch point."""
    first_edit = next((i for i, m in enumerate(msgs) if m["role"] == "assistant"
                       and any(c["function"]["name"] in EDIT_TOOLS for c in m.get("tool_calls", []))), None)
    if first_edit is None:
        return None, "no_edit"
    for i, m in enumerate(msgs):
        if m["tokens_so_far"] > window:
            break
        if i > first_edit and m["role"] == "tool":
            j = max(k for k in range(i) if msgs[k]["role"] == "assistant")
            call = next((c for c in msgs[j].get("tool_calls", []) if c["id"] == m.get("tool_call_id")),
                        (msgs[j].get("tool_calls") or [None])[0])
            if is_failure(call, m["content"]):
                return j, "failure_after_edit"
    return None, "no_failure_in_window"


def excluded_test_ids():
    """Task ids that must never be training data: competition public tasks, OOD-45, the gRPC slice."""
    ids = set()
    p = ROOT / "data" / "comp" / "public_task_ids.txt"
    if p.exists():
        ids |= set(p.read_text().split())
    p = ROOT / "data" / "ood45" / "tasks.jsonl"
    if p.exists():
        ids |= {json.loads(l)["instance_id"] for l in open(p) if l.strip()}
    p = ROOT / "progress" / "grpc_task_ids.json"
    if p.exists():
        ids |= set(json.load(open(p))["verified"]["ids"])
    if len(ids) < 129 + 45:
        raise SystemExit("test-set id lists missing (data/comp/public_task_ids.txt, data/ood45/tasks.jsonl)")
    return ids


def load_flags():
    """Fairness flags per task (0-2 raters). Fails loudly if a rater's file is missing, so a missing
    file can never silently mean "no task is flagged"."""
    flags = defaultdict(int)
    for screen in ("train_sets", "mined_sets"):  # FreshSWE-Py and our mined tasks
        for r in ("opus", "sol"):
            p = ROOT / "data" / "tasks" / "screen" / f"{screen}.{r}.fairness_log.jsonl"
            if not p.exists():
                raise SystemExit(f"missing fairness ratings: {p}")
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
                    help="branch only within the first N tokens of the run (early failures; the context loaded "
                         "for resampling is only what the model saw since the last compaction, see resample.py)")
    ap.add_argument("--max-segments-per-run", type=int, default=3,
                    help="cap examples per run so long runs do not dominate (first, last, evenly spaced)")
    ap.add_argument("--exclude-prefix", default="black_,isort_")
    ap.add_argument("--keep-flagged", action="store_true", help="keep tasks both fairness raters flag")
    ap.add_argument("--no-split", action="store_true", help="keep whole runs (drop those over --max-tokens)")
    ap.add_argument("--carry", type=int, default=2, help="steps carried over as context after a compaction")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    skip = tuple(x for x in a.exclude_prefix.split(",") if x)
    flags = load_flags()
    test_ids = excluded_test_ids()

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
            if tid in test_ids:
                raise SystemExit(f"{tid} is a test-set task ({res}): refusing to build training data from it")
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

    arm_a, dropped_long, dropped_markers, capped = [], 0, 0, 0
    for r in runs:
        r["task_kind"] = kind[r["instance_id"]]
        if not r["trace"]:
            continue
        msgs, n = msgs_of(r)
        r["tokens"] = n
        r["markers"] = marker_count(trace_of(r))
        if r["markers"] > MAX_MARKERS_PER_RUN:
            dropped_markers += r["resolved"]
            continue
        if r["resolved"] and not r["excluded_env"]:
            if a.no_split:
                if n > a.max_tokens:
                    dropped_long += 1
                    continue
                arm_a.append({"instance_id": r["instance_id"], "pass": r["pass"], "segment": 0, "tokens": n,
                              "messages": clean(msgs)})
                continue
            segs = segments(trace_of(r), msgs, a.carry)
            kept = cap_segments(segs, a.max_segments_per_run)
            capped += len(segs) - len(kept)
            for k, seg in kept:
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
        if (r["resolved"] or not r["trace"] or not usable(r) or r["task_kind"] == "always"
                or r.get("markers", 0) > MAX_MARKERS_PER_RUN):
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
        "successful_runs_dropped_for_control_tokens": dropped_markers,
        "segments_dropped_by_per_run_cap": capped,
        "arm_a_trained_tokens": sum(size([m for m in x["messages"] if m["loss"]]) for x in arm_a),
        "dpo_same_task_pairs": len(pairs), "dpo_pairs_fitting_max_tokens": sum(p["fits"] for p in pairs),
        "branch_points": len(branches), "branch_points_by_task_kind": dict(Counter(b["task_kind"] for b in branches)),
        "branch_reasons": dict(Counter(b["reason"] for b in branches)),
        "grades": dict(Counter(r["grade"] for r in runs)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
