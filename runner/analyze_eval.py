"""Per-agent OOD-45 numbers for the paper (Section 5), from eval_arms.sh results and their ATIF traces.

For each agent dir <root>/<agent>/s<seed>/ (several roots allowed):
  solved, timeouts, other outcomes;
  paired comparison with base on the same tasks: tasks gained / lost and a 95% paired bootstrap interval of the
    difference in solve rate (resampling tasks);
  behaviour per model call: output tokens, thinking characters, visible-text characters, tool-argument characters;
  recovery after a mid-run test failure: among runs in which a test command failed after the agent's first edit,
    the share that were solved;
  repeated failing calls: per run, how often the agent repeats exactly the tool call (same tool, same arguments)
    that has just failed.

    python runner/analyze_eval.py --root /root/results/eval_v3 /root/results/eval_v2_e2 --seed 2 --base base \\
        --out eval_summary.json
"""
import argparse
import glob
import json
import random
import re
import statistics as st
from pathlib import Path

MAIN = "swe_baseline_agent"
TEST_CMD = re.compile(r"\b(pytest|py\.test|tox|nox|unittest|runtests)\b")
EDIT_TOOLS = {"edit_file", "write_file"}


def obs_failed(obs):
    """True if a tool result reports a failure (tool error or non-zero exit code)."""
    raw = obs.get("content") if isinstance(obs, dict) else obs
    try:
        r = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return False
    if not isinstance(r, dict):
        return False
    if r.get("status") == "error":
        return True
    res = r.get("result") if isinstance(r.get("result"), dict) else r
    code = res.get("exit_code") if isinstance(res, dict) else None
    return isinstance(code, int) and code != 0


def trace_stats(path):
    steps = [s for s in json.load(open(path))["steps"]
             if s.get("source") == "agent" and (s.get("extra") or {}).get("author", MAIN) == MAIN]
    calls, cur = [], None
    for s in steps:
        if s.get("metrics"):
            if cur:
                calls.append(cur)
            cur = {"out": (s["metrics"] or {}).get("completion_tokens") or 0, "think": 0, "text": 0, "args": 0}
        if cur is None:
            continue
        msg = s.get("message") or ""
        if (s.get("extra") or {}).get("event_type") == "thinking":
            cur["think"] += len(msg)
        else:
            cur["text"] += len(msg)
        for c in s.get("tool_calls") or []:
            cur["args"] += len(json.dumps(c.get("arguments") or {}))
    if cur:
        calls.append(cur)
    edited, test_fail_after_edit, repeats, prev_failed_call = False, False, 0, None
    for s in steps:
        for c in s.get("tool_calls") or []:
            name, args = c.get("function_name"), c.get("arguments") or {}
            key = (name, json.dumps(args, sort_keys=True))
            failed = obs_failed(s.get("observation") or {})
            if key == prev_failed_call:
                repeats += 1
            prev_failed_call = key if failed else None
            if name in EDIT_TOOLS:
                edited = True
            elif edited and name == "run_command" and TEST_CMD.search(str(args.get("command", ""))) and failed:
                test_fail_after_edit = True
    return calls, test_fail_after_edit, repeats


def outcome(r):
    e = r.get("error") or ""
    if r["resolved"]:
        return "solved"
    if "timeout" in e:
        return "timeout"
    if "ContextWindow" in e:
        return "context_overflow"
    if "No space left" in e:
        return "no_space"
    if e:
        return "other_error"
    return "patch_tests_fail" if r.get("agent_patch_size") else "no_patch"


def agent_summary(d):
    rows = [json.loads(l) for f in glob.glob(f"{d}/*/task_results.jsonl") for l in open(f)]
    calls, tf_runs, tf_solved, rep = [], 0, 0, []
    for r in rows:
        tr = glob.glob(f"{d}/*/traces/trace_{r['instance_id']}.json")
        if not tr:
            continue
        c, tf, k = trace_stats(tr[0])
        calls += c
        rep.append(k)
        if tf:
            tf_runs += 1
            tf_solved += r["resolved"]
    out = {"tasks": len(rows), "solved": sum(r["resolved"] for r in rows), "outcomes": {}}
    for r in rows:
        out["outcomes"][outcome(r)] = out["outcomes"].get(outcome(r), 0) + 1
    if calls:
        out["per_call"] = {k: round(st.mean(c[k] for c in calls), 1) for k in ("out", "think", "text", "args")}
        out["per_call"]["out_median"] = st.median(c["out"] for c in calls)
        out["calls_per_task_median"] = st.median(r.get("total_llm_calls") or 0 for r in rows)
        out["seconds_per_call_median"] = round(st.median(r["duration_seconds"] / r["total_llm_calls"]
                                                         for r in rows if r.get("total_llm_calls")), 1)
    out["test_failure_runs"] = tf_runs
    out["recovered_after_test_failure"] = round(tf_solved / tf_runs, 3) if tf_runs else None
    out["repeated_failing_calls_per_run"] = round(st.mean(rep), 2) if rep else None
    out["by_task"] = {r["instance_id"]: bool(r["resolved"]) for r in rows}
    return out


def paired(a, b, n=10000, seed=0):
    """a vs b on shared tasks: (gained, lost, diff in solve rate, 95% bootstrap interval)."""
    tasks = sorted(set(a) & set(b))
    d = [int(a[t]) - int(b[t]) for t in tasks]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return {"tasks": len(tasks), "gained": sum(x > 0 for x in d), "lost": sum(x < 0 for x in d),
            "diff": round(sum(d) / len(d), 3), "ci95": [round(boots[int(0.025 * n)], 3), round(boots[int(0.975 * n)], 3)]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="+", required=True)
    ap.add_argument("--seed", default="2")
    ap.add_argument("--base", default="base")
    ap.add_argument("--out")
    a = ap.parse_args()
    agents = {}
    for root in a.root:
        for d in sorted(Path(root).glob(f"*/s{a.seed}")):
            if Path(f"{d}.done").exists():
                agents[d.parent.name] = agent_summary(d)
    base = agents.get(a.base)
    for name, s in agents.items():
        if base and name != a.base:
            s["vs_base"] = paired(s["by_task"], base["by_task"])
    summary = {k: {x: v for x, v in s.items() if x != "by_task"} for k, s in agents.items()}
    if a.out:
        Path(a.out).write_text(json.dumps({"agents": summary, "by_task": {k: s["by_task"] for k, s in agents.items()}},
                                          indent=1))
    print(f"{'agent':18}{'solved':>7}{'timeout':>8}{'vs base [95% CI]':>26}{'think/call':>11}{'out tok':>8}"
          f"{'rec.after test fail':>21}{'repeats/run':>12}")
    for k, s in summary.items():
        vb = s.get("vs_base")
        vbs = f"{vb['diff']:+.2f} [{vb['ci95'][0]:+.2f},{vb['ci95'][1]:+.2f}] +{vb['gained']}/-{vb['lost']}" if vb else "-"
        rec = s["recovered_after_test_failure"]
        print(f"{k:18}{s['solved']:>4}/{s['tasks']:<3}{s['outcomes'].get('timeout', 0):>6}{vbs:>27}"
              f"{s.get('per_call', {}).get('think', 0):>11.0f}{s.get('per_call', {}).get('out_median', 0):>8.0f}"
              f"{(f'{rec:.0%} of ' + str(s['test_failure_runs'])) if rec is not None else '-':>21}"
              f"{s['repeated_failing_calls_per_run'] or 0:>12.2f}")
