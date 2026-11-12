"""Extra arm (d) pairs: each verified recovery (chosen) against the ORIGINAL run's own ending from the
same branch point (rejected), as Enoch and David agreed on Nov 12.

Both sides share the prompt of training/build_splices.py's pairs (the segment holding the branch point,
up to and including the failing step's tool result) and stop at their first later user message. The
original ending is a genuine base-model sample from that state, but it only had the time left in its
run, while the resampled continuation got a fresh budget. So an original ending is used only when it
ended with a real attempt (Enoch's rule): a patch whose tests failed without hitting a cap (grade 2), or
no patch for a reason other than the session timeout; or, when it timed out, if the recovery solved the
task in no more time than the original run still had at the branch point ("time_matched"). Same filters as every other example: at most
MAX_MARKERS_PER_RUN leaked control tokens, no lost sub-agent result (an empty user message). Every pair
is tagged "source": "original". Run training/fix_examples.py --dpo on the output afterwards (nudge on
both sides, broken calls masked), then concatenate with the resample pairs (tagged "resample").

    python3 training/build_orig_pairs.py --resample /workspace/results/resample_v2/fresh \\
        /workspace/results/resample_v2/mined --trace-root /workspace/results/fresh_base \\
        --trace-root /workspace/results/fresh_base_h100 --trace-root /workspace/results/mined_base \\
        --out data/dpo_pairs_original.jsonl
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_data import MAIN_AGENT, MAX_MARKERS_PER_RUN, author, grade, marker_count, segments, to_messages  # noqa: E402
from build_splices import OFFSET, boundary_no_cut, load_trace, splice  # noqa: E402


BUDGET_S = 15 * 60  # the harness's session timeout


def elapsed_at(orig, step_id):
    st = next((s for s in orig["steps"] if s.get("step_id") == step_id), {})
    return float((st.get("extra") or {}).get("elapsed_s") or 0)


def side(orig, branch_step_id, cont_steps, carry):
    """(messages, prompt_len) for one side, or (None, reason). Same construction as build_splices' dpo_side."""
    tr, _ = splice(orig, branch_step_id, {"steps": cont_steps})
    if marker_count(tr) > MAX_MARKERS_PER_RUN:
        return None, "markers"
    msgs, _ = to_messages(tr)
    if any(m["role"] == "user" and not (m.get("content") or "").strip() for m in msgs):
        return None, "lost_subagent_result"
    for seg in segments(tr, msgs, carry, boundary_no_cut(orig, branch_step_id, tr)):
        first = next((i for i, m in enumerate(seg) if (m.get("step_id") or 0) >= OFFSET), None)
        if first is None:
            continue
        ms = [{**{x: v for x, v in m.items() if x not in ("tokens_so_far", "step_id")},
               "loss": bool(m["loss"]) and (m.get("step_id") or 0) >= OFFSET} for m in seg]
        nudge = next((i for i in range(first, len(ms)) if ms[i]["role"] == "user"), None)
        ms = ms[:nudge] if nudge is not None else ms
        if not any(m["loss"] for m in ms):
            return None, "no_trained_turns"
        return {"messages": ms, "prompt_len": first}, None
    return None, "no_continuation"


def original_ending(orig, branch_step_id):
    """The original run's steps after the failing step's model call (main agent and harness nudges)."""
    main = [s for s in orig["steps"] if s.get("source") == "agent" and author(s) == MAIN_AGENT]
    b = next(i for i, s in enumerate(main) if s.get("step_id") == branch_step_id)
    e = b + 1
    while e < len(main) and not main[e].get("metrics"):
        e += 1
    after = {s.get("step_id") for s in main[e:]}
    first_after = main[e].get("step_id") if e < len(main) else None
    out, started = [], False
    for s in orig["steps"]:
        if s.get("step_id") == first_after:
            started = True
        if started and (s.get("step_id") in after or (s.get("extra") or {}).get("event_type") == "continuation_nudge"):
            out.append(s)
    return out


def original_row(orig_trace_path, tid, roots):
    """The original run's task_results row (next to its traces/ folder)."""
    p = Path(orig_trace_path)
    cands = [p.parent.parent / "task_results.jsonl"]
    parts = p.parts
    i = next((k for k, x in enumerate(parts) if x.startswith(("fresh_base", "mined_base"))), None)
    if i is not None:
        rel = Path(*parts[i + 1:]).parent.parent / "task_results.jsonl"
        cands += [Path(r) / rel for r in roots]
    for c in cands:
        if c.exists():
            for l in open(c):
                row = json.loads(l)
                if row["instance_id"] == tid:
                    return row
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resample", nargs="+", required=True)
    ap.add_argument("--trace-root", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--carry", type=int, default=2)
    a = ap.parse_args()

    stats, pairs, seen = Counter(), [], set()
    for root in a.resample:
        for d in sorted(Path(root).glob("s*_p*")):
            if not (d / "branches.jsonl").exists() or not (d / "task_results.jsonl").exists():
                continue
            br = {json.loads(l)["instance_id"]: json.loads(l) for l in open(d / "branches.jsonl")}
            for row in map(json.loads, open(d / "task_results.jsonl")):
                tid = row["instance_id"]
                tr = d / "traces" / f"trace_{tid}.json"
                if not row["resolved"] or tid not in br or not tr.exists():
                    continue
                stats["recoveries"] += 1
                b = br[tid]
                orig = load_trace(b["trace"], a.trace_root)
                orow = original_row(b["trace"], tid, a.trace_root)
                if orig is None or orow is None:
                    stats["drop_missing_original"] += 1
                    continue
                og = grade(orow)
                timeout = "session timeout" in (orow.get("error") or "")
                real_attempt = og == 2 or (og == 0 and not timeout and bool(orow.get("error")))
                # Time-matched: the recovery finished within the time the original run still had at the
                # branch point, so a longer budget cannot be why it won.
                left = BUDGET_S - elapsed_at(orig, b["branch_step_id"])
                used = max((float((s.get("extra") or {}).get("elapsed_s") or 0) for s in json.load(open(tr))["steps"]),
                           default=BUDGET_S)
                time_matched = timeout and used <= left
                if not (real_attempt or time_matched):
                    stats[f"drop_original_grade{og}{'_timeout' if timeout else ''}"] += 1
                    continue
                rule = "real_attempt" if real_attempt else "time_matched"
                sc, why_c = side(orig, b["branch_step_id"], json.load(open(tr))["steps"], a.carry)
                sr, why_r = side(orig, b["branch_step_id"], original_ending(orig, b["branch_step_id"]), a.carry)
                if sc is None or sr is None:
                    stats["drop_" + str(why_c or why_r)] += 1
                    continue
                if sc["messages"][: sc["prompt_len"]] != sr["messages"][: sr["prompt_len"]]:
                    stats["drop_prompt_mismatch"] += 1
                    continue
                key = json.dumps(sc["messages"], sort_keys=True)
                if key in seen:
                    stats["drop_duplicate"] += 1
                    continue
                seen.add(key)
                pairs.append({"instance_id": tid, "branch": [tid, b["pass"], b["branch_step_id"]],
                              "branch_step_id": b["branch_step_id"], "source": "original", "rule": rule,
                              "time_left_s": round(left), "recovery_time_s": round(used),
                              "chosen": {"trace": str(tr), "grade": 3, **sc},
                              "rejected": {"trace": b["trace"], "grade": og, **sr}})
    Path(a.out).write_text("".join(json.dumps(p) + "\n" for p in pairs))
    stats.update(Counter("pairs_" + p["rule"] for p in pairs))
    print(json.dumps(dict(stats, pairs=len(pairs))), "->", a.out)
