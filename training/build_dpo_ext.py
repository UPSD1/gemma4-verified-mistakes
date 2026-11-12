"""Arm (d) ablation: add "verified recovery vs. the original run's own ending" pairs to the DPO data.

The main arm (d) uses only pairs between resampled continuations of the same branch point
(training/build_splices.py -> dpo_pairs.jsonl). Here every resampled continuation that the harness resolved
is also paired with what the ORIGINAL base run did after the same failing step (that run was not resolved,
which is why it has a branch point). Both sides are built by the same code (build_splices.dpo_side): the
original's remainder is passed as the continuation, so prompt, segmenting at compaction points, the leaked-
token rule and the loss mask are identical; a pair is kept only if both prompts are identical.

Known bias, reported, not hidden: the original ending only had the time left in its run, while a resample got
a fresh budget. Every pair carries `orig_grade` (2 = patch with failing tests, 1 = patch but hit a budget,
0 = no patch) and `orig_time_left_s` so results can be split. --min-time-left 300 (the ablation's setting)
keeps only originals that still had 5 minutes at the failing step: 16 of 23, median 9 min left; the 7 dropped
had 2.8-4.2 min. --min-orig-grade 2 (failed on a real test attempt) would leave only 2 pairs.

    python3 training/build_dpo_ext.py --resample data/pod_results/resample_v2/fresh data/pod_results/resample_v2/mined \\
        --min-time-left 300
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_data import MAIN_AGENT, author, grade  # noqa: E402
from build_splices import dpo_side, load_trace  # noqa: E402


SESSION_S = 900  # the harness's session timeout (15 min)


def elapsed_at(orig, branch_step_id):
    """Seconds into the original run when the failing step's tool call was made."""
    st = next(s for s in orig["steps"] if s.get("step_id") == branch_step_id and author(s) == MAIN_AGENT)
    for c in st.get("tool_calls") or []:
        if (c.get("extra") or {}).get("elapsed_s"):
            return c["extra"]["elapsed_s"]
    return (st.get("extra") or {}).get("elapsed_s") or 0


def original_remainder(orig, branch_step_id):
    """The original run's steps after the failing model call, as a continuation trace for splice()."""
    steps = orig["steps"]
    main = [i for i, s in enumerate(steps) if s.get("source") == "agent" and author(s) == MAIN_AGENT]
    b = next(k for k, i in enumerate(main) if steps[i].get("step_id") == branch_step_id)
    e = b + 1
    while e < len(main) and not steps[main[e]].get("metrics"):
        e += 1  # same model call as the failing step
    last = main[e - 1]
    return {"steps": steps[last + 1:]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resample", nargs="+", required=True)
    ap.add_argument("--base-pairs", default="data/training/dpo_pairs.jsonl", help="the main arm (d) pairs, kept as is")
    ap.add_argument("--out", default="data/training/dpo_pairs_ext.jsonl")
    ap.add_argument("--carry", type=int, default=2)
    ap.add_argument("--min-orig-grade", type=int, default=0,
                    help="keep only original endings with at least this grade (2 = failed on a real test attempt)")
    ap.add_argument("--min-time-left", type=float, default=0,
                    help="keep only originals with at least this many seconds of the 900 s session left at the failing "
                         "step (300: the original had a real chance to recover; under that it mostly ran out of time)")
    ap.add_argument("--trace-root", action="append", default=[])
    a = ap.parse_args()
    roots = a.trace_root or [str(Path.home() / "Downloads/base_runs_2026-11-11/rtx/fresh_base"),
                             str(Path.home() / "Downloads/base_runs_2026-11-11/h100/fresh_base"),
                             "data/pod_results/mined_base"]

    base = [{**json.loads(l), "source": "resample"} for l in open(a.base_pairs) if l.strip()]
    new, drop = [], Counter()
    for root in a.resample:
        for d in sorted(Path(root).glob("s*_p*")):
            if not (d / "branches.jsonl").exists():
                continue
            br = {json.loads(l)["instance_id"]: json.loads(l) for l in open(d / "branches.jsonl")}
            for row in map(json.loads, open(d / "task_results.jsonl")):
                if not row["resolved"]:
                    continue
                b = br[row["instance_id"]]
                c = {"instance_id": b["instance_id"], "orig_trace": b["trace"], "branch_step_id": b["branch_step_id"],
                     "cont_trace": str(d / "traces" / f"trace_{b['instance_id']}.json")}
                if b["grade"] < a.min_orig_grade:
                    drop[f"orig_grade_{b['grade']}_below_min"] += 1
                    continue
                orig = load_trace(b["trace"], roots)
                left = SESSION_S - elapsed_at(orig, b["branch_step_id"])
                if left < a.min_time_left:
                    drop["original_under_min_time_left"] += 1
                    continue
                sc, why_c = dpo_side(c, roots, a.carry)
                so, why_o = dpo_side(c, roots, a.carry, cont=original_remainder(orig, b["branch_step_id"]), no_cut=set())
                if sc is None or so is None:
                    drop[f"{'chosen' if sc is None else 'original'}_{why_c or why_o}"] += 1
                    continue
                if sc["messages"][: sc["prompt_len"]] != so["messages"][: so["prompt_len"]]:
                    drop["prompt_mismatch"] += 1
                    continue
                new.append({"instance_id": b["instance_id"], "branch": [b["instance_id"], b["pass"], b["branch_step_id"]],
                            "branch_step_id": b["branch_step_id"], "source": "original", "orig_grade": b["grade"],
                            "orig_time_left_s": round(left),
                            "sample_dir": d.name,
                            "chosen": {"trace": c["cont_trace"], "grade": grade(row), **sc},
                            "rejected": {"trace": b["trace"], "grade": b["grade"], **so}})
    Path(a.out).write_text("".join(json.dumps(p) + "\n" for p in base + new))
    tok = lambda side: sum(len((m.get("content") or "") + (m.get("reasoning") or "")) for m in side["messages"] if m["loss"]) / 3.5  # noqa: E731
    summary = {"base_pairs": len(base), "new_pairs": len(new), "total": len(base) + len(new), "dropped": dict(drop),
               "new_by_orig_grade": dict(Counter(p["orig_grade"] for p in new)),
               "new_time_left_s_min": min((p["orig_time_left_s"] for p in new), default=None),
               "new_branch_points": len({tuple(p["branch"]) for p in new}),
               "new_overlapping_base_branches": len({tuple(p["branch"]) for p in new} & {tuple(p["branch"]) for p in base}),
               "est_trained_tokens_chosen_median": sorted(int(tok(p["chosen"])) for p in new)[len(new) // 2] if new else 0,
               "est_trained_tokens_original_median": sorted(int(tok(p["rejected"])) for p in new)[len(new) // 2] if new else 0}
    print(json.dumps(summary, indent=2), "->", a.out)
