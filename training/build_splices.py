"""Build training data for arms (b), (c) and (d) from the resampling results (training/resample.py).

A recovery splice is a failed base run up to and including its failing step (with the error the agent
saw), joined to a resampled continuation that the harness grades as resolved. Splices are split at the harness's compaction points, exactly
like arm (a)'s clean runs (training/build_data.py), so every example is what the model saw.

  arm_b_sft.jsonl   (b) Recovery: clean examples + splices, every agent turn trained
  arm_c_sft.jsonl   (c) Recovery-masked: same examples, but the mistaken step (the failing model call
                    at the branch point) is masked from the loss; the recovery after it is trained
  dpo_pairs.jsonl   (d) pairs between continuations of the SAME branch point with different grades,
                    each side as chat messages (shared prompt, loss on the continuation only)
                    (graded scale from the plan, coarse version: resolved > patch with failing
                    tests > patch but hit a budget > no patch); prompt = the recorded prefix
  summary.json      counts

Splices are marked "priority": training/train_sft.py keeps them all before filling the equal
token budget with clean examples, so every arm sees the same number of trained tokens.

    python3 training/build_splices.py --resample data/pod_results/resample_v2/fresh data/pod_results/resample_v2/mined \\
        --clean data/training/arm_a_sft.jsonl --out data/training
"""
import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_data import MAIN_AGENT, MAX_MARKERS_PER_RUN, author, grade, marker_count, segments, to_messages  # noqa: E402

OFFSET = 100000  # continuation step ids are shifted so they never collide with the prefix's


def load_trace(path, roots):
    p = Path(path)
    if p.exists():
        return json.load(open(p))
    parts = p.parts
    # Pod paths: .../fresh_base/r1/..., .../fresh_base_h100/r3/..., .../mined_base/r2/...
    i = next((k for k, x in enumerate(parts) if x.startswith(("fresh_base", "mined_base"))), None)
    rel = Path(*parts[i + 1:]) if i is not None else Path(p.name)
    for root in roots:
        cand = Path(root) / rel
        if cand.exists():
            return json.load(open(cand))
    return None


def splice(original, branch_step_id, continuation):
    """Combined trace: the original system prompt and task prompt, the main agent's steps up to and
    including the failing step (with its error output), then the continuation's own main-agent steps
    (and harness nudges). Returns (trace, mistaken step ids): the failing model call, which arm (c)
    masks from the loss and arm (b) trains."""
    head = [s for s in original["steps"] if s.get("source") in ("system", "user")][:2]
    main = [s for s in original["steps"] if s.get("source") == "agent" and author(s) == MAIN_AGENT]
    b = next((i for i, s in enumerate(main) if s.get("step_id") == branch_step_id), None)
    if b is None:
        raise ValueError(f"branch step {branch_step_id} not found")
    e = b + 1
    while e < len(main) and not main[e].get("metrics"):
        e += 1  # steps of the same model call as the failing step
    prefix = main[:e]
    mistaken = {s.get("step_id") for s in main[b:e]}
    cont = []
    for s in continuation["steps"]:
        ev = (s.get("extra") or {}).get("event_type")
        if (s.get("source") == "agent" and author(s) == MAIN_AGENT) or \
                (s.get("source") == "system" and ev == "continuation_nudge"):
            cont.append({**s, "step_id": OFFSET + int(s.get("step_id") or 0)})
    return {"steps": head + prefix + cont}, mistaken


def boundary_no_cut(original, branch_step_id, trace):
    """Step ids of the spliced trace where a prompt-size drop is NOT a compaction.

    The resampled run starts with the harness's continuation nudge, a user message, and Gemma's template
    drops every thought before the last user message. So the first resampled call's prompt is smaller than
    the original's at the failing step by about the thoughts in the loaded context (measured: median drop
    ~2.1k tokens vs ~2.8-3.0k thought tokens), which segments() would read as a compaction (inserting a
    compaction note, cutting context, masking the first recovery turn). The drop is still a cut if it is
    larger than 25% after the thoughts are taken out (a real compaction on the first call)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from resample import prefix_steps
    replay, context, _ = prefix_steps(original, branch_step_id)
    pts = [s["metrics"]["prompt_tokens"] for s in replay if (s.get("metrics") or {}).get("prompt_tokens")]
    thoughts = sum(len(s.get("message") or "") for s in context
                   if (s.get("extra") or {}).get("event_type") == "thinking") / 3.5
    first = next((s for s in trace["steps"] if (s.get("step_id") or 0) >= OFFSET and s.get("source") == "agent"
                  and (s.get("metrics") or {}).get("prompt_tokens")), None)
    if not pts or first is None or first["metrics"]["prompt_tokens"] < 0.75 * (pts[-1] - thoughts):
        return set()
    return {first["step_id"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resample", nargs="+", required=True, help="resample result roots (contain s*_p*/)")
    ap.add_argument("--clean", default="data/training/arm_a_sft.jsonl")
    ap.add_argument("--trace-root", action="append", default=[], help="where the original base-run traces are")
    ap.add_argument("--out", default="data/training")
    ap.add_argument("--carry", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=28000)
    a = ap.parse_args()
    roots = a.trace_root or [str(Path.home() / "Downloads/base_runs_2026-11-11/rtx/fresh_base"),
                             str(Path.home() / "Downloads/base_runs_2026-11-11/h100/fresh_base"),
                             "data/pod_results/mined_base"]

    conts = []  # one per resampled continuation
    for root in a.resample:
        for d in sorted(Path(root).glob("s*_p*")):
            if not (d / "branches.jsonl").exists() or not (d / "task_results.jsonl").exists():
                continue
            br = {json.loads(l)["instance_id"]: json.loads(l) for l in open(d / "branches.jsonl")}
            for row in map(json.loads, open(d / "task_results.jsonl")):
                tid = row["instance_id"]
                tr = d / "traces" / f"trace_{tid}.json"
                if tid in br and tr.exists():
                    b = br[tid]
                    conts.append({"instance_id": tid, "branch": (tid, b["pass"], b["branch_step_id"]),
                                  "orig_trace": b["trace"], "branch_step_id": b["branch_step_id"],
                                  "cont_trace": str(tr), "resolved": bool(row["resolved"]), "grade": grade(row),
                                  "sample_dir": d.name})

    out = Path(a.out)
    splices_b, splices_c, missing, long_drop, marker_drop, dup_drop, seen = [], [], 0, 0, 0, 0, set()
    boundary_kept = 0  # recoveries whose first resampled call was NOT a compaction (only thoughts dropped)
    for c in conts:
        if not c["resolved"]:
            continue
        orig = load_trace(c["orig_trace"], roots)
        if orig is None:
            missing += 1
            continue
        tr, mistaken = splice(orig, c["branch_step_id"], json.load(open(c["cont_trace"])))
        if marker_count(tr) > MAX_MARKERS_PER_RUN:  # same rule as arm (a)'s clean runs
            marker_drop += 1
            continue
        msgs, _ = to_messages(tr)
        keep = boundary_no_cut(orig, c["branch_step_id"], tr)
        boundary_kept += bool(keep)
        for k, seg in enumerate(segments(tr, msgs, a.carry, keep)):
            size = sum(len((m.get("content") or "") + (m.get("reasoning") or "")) for m in seg) / 3.5
            if size > a.max_tokens:
                long_drop += 1
                continue
            clean = lambda ms, mask: [  # noqa: E731
                {**{x: v for x, v in m.items() if x not in ("tokens_so_far", "step_id")},
                 "loss": m["loss"] and not (mask and m.get("step_id") in mistaken)} for m in ms]
            meta = {"instance_id": c["instance_id"], "branch": list(c["branch"]), "segment": k,
                    "kind": "splice", "priority": True}
            ex_b, ex_c = {**meta, "messages": clean(seg, False)}, {**meta, "messages": clean(seg, True)}
            # Two recoveries from the same branch point share their prefix segments: keep one copy.
            key = json.dumps(ex_c["messages"], sort_keys=True)
            if key in seen:
                dup_drop += 1
                continue
            seen.add(key)
            if any(m["loss"] for m in ex_b["messages"]):
                splices_b.append(ex_b)
            if any(m["loss"] for m in ex_c["messages"]):
                splices_c.append(ex_c)

    clean_rows = [{**json.loads(l), "kind": "clean"} for l in open(a.clean)]
    for name, spl in (("arm_b_sft.jsonl", splices_b), ("arm_c_sft.jsonl", splices_c)):
        with open(out / name, "w") as f:
            for r in spl + clean_rows:
                f.write(json.dumps(r) + "\n")

    # Arm (d): pairs between continuations of the same branch point with different grades. Each side is
    # the segment that holds the branch point (the context the model had there, cut at compaction points
    # like every other example) followed by that continuation; only the continuation's agent turns carry
    # loss, so the two sides share an identical prompt and differ only in what came after the mistake.
    def dpo_side(c):
        orig = load_trace(c["orig_trace"], roots)
        if orig is None:
            return None, "missing"
        tr, _ = splice(orig, c["branch_step_id"], json.load(open(c["cont_trace"])))
        if marker_count(tr) > MAX_MARKERS_PER_RUN:  # same rule as the SFT data
            return None, "markers"
        msgs, _ = to_messages(tr)
        for seg in segments(tr, msgs, a.carry, boundary_no_cut(orig, c["branch_step_id"], tr)):
            first = next((i for i, m in enumerate(seg) if (m.get("step_id") or 0) >= OFFSET), None)
            if first is None:
                continue
            ms = [{**{x: v for x, v in m.items() if x not in ("tokens_so_far", "step_id")},
                   "loss": bool(m["loss"]) and (m.get("step_id") or 0) >= OFFSET} for m in seg]
            # End the side at the continuation's first harness nudge: in Gemma's template a later user message
            # drops every earlier thought, the shared prompt's included, so the two sides would no longer
            # be compared from the same rendered state. What is compared is what came right after the mistake.
            nudge = next((i for i in range(first, len(ms)) if ms[i]["role"] == "user"), None)
            ms = ms[:nudge] if nudge is not None else ms
            if not any(m["loss"] for m in ms):
                return None, "no_trained_turns"
            return {"messages": ms, "prompt_len": first}, None
        return None, "no_continuation"

    by_branch = defaultdict(list)
    for c in conts:
        by_branch[c["branch"]].append(c)
    pairs, pair_drop = [], Counter()
    for key, cs in by_branch.items():
        for x, y in itertools.combinations(cs, 2):
            if x["grade"] == y["grade"]:
                continue
            ch, rj = (x, y) if x["grade"] > y["grade"] else (y, x)
            (sc, why_c), (sr, why_r) = dpo_side(ch), dpo_side(rj)
            if sc is None or sr is None:
                pair_drop[why_c or why_r] += 1
                continue
            if sc["messages"][: sc["prompt_len"]] != sr["messages"][: sr["prompt_len"]]:
                pair_drop["prompt_mismatch"] += 1
                continue
            pairs.append({"instance_id": key[0], "branch": list(key), "branch_step_id": ch["branch_step_id"],
                          "chosen": {"trace": ch["cont_trace"], "grade": ch["grade"], **sc},
                          "rejected": {"trace": rj["cont_trace"], "grade": rj["grade"], **sr}})
    with open(out / "dpo_pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    summary = {"continuations": len(conts), "resolved": sum(c["resolved"] for c in conts),
               "branch_points": len(by_branch),
               "branch_points_recovered": len({c["branch"] for c in conts if c["resolved"]}),
               "grades": dict(Counter(c["grade"] for c in conts)),
               "splice_examples_b": len(splices_b), "splice_examples_c": len(splices_c),
               "clean_examples": len(clean_rows), "missing_original_traces": missing,
               "dropped_over_max_tokens": long_drop, "dropped_over_max_markers": marker_drop,
               "duplicate_segments_dropped": dup_drop, "recoveries_without_cut_at_branch": boundary_kept,
               "branch_points_in_splices": len({tuple(r["branch"]) for r in splices_b}),
               "dpo_pairs": len(pairs), "dpo_pairs_dropped": dict(pair_drop),
               "dpo_pairs_with_resolved_chosen": sum(p["chosen"]["grade"] == 3 for p in pairs),
               "dpo_grade_pairs": {f"{k[0]}>{k[1]}": v for k, v in
                                   Counter((p["chosen"]["grade"], p["rejected"]["grade"]) for p in pairs).items()}}
    (out / "splices_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
