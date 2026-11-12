"""Submit-only clean data (round 2): arm (a)'s v3 examples restricted to runs that ended with a submit_patch call.

Most "solved" base runs never chose to stop: the harness graded their workspace when time ran out. This keeps only
the runs the agent itself ended, to test whether the not-stopping habit comes from the training data.

    python training/build_submit_only.py --clean training_data/v3/arm_a_sft_v3.jsonl --runs base_runs \\
        --out training_data/v3_submit_only/arm_a_submit_only_v3.jsonl

Train it at the same 2.25M-token budget as arms (a)-(c) by repeating it (one pass is 1,078,640 trained tokens):
    python training/train_sft.py --data <out> --out <adapter dir> --epochs 2.086 --seed 0
"""
import argparse
import glob
import json
from pathlib import Path


def trace_of(runs, pass_name, iid):
    """The base-run trace an example came from: pass is 'freshswe/r3' or 'mined/r2'."""
    src, run = pass_name.split("/")
    pat = f"{runs}/freshswe/*/fresh_base/{run}/traces/trace_{iid}.json" if src == "freshswe" \
        else f"{runs}/mined/{run}/traces/trace_{iid}.json"
    hits = glob.glob(pat)
    return hits[0] if hits else None


def submitted(trace):
    return any(c.get("function_name") == "submit_patch"
               for s in json.load(open(trace))["steps"] for c in s.get("tool_calls") or [])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True, help="arm (a) examples (arm_a_sft_v3.jsonl)")
    ap.add_argument("--runs", required=True, help="folder with freshswe/ and mined/ base runs and their traces")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.clean) if l.strip()]
    status, keep = {}, []
    for r in rows:
        key = (r["pass"], r["instance_id"])
        if key not in status:
            tr = trace_of(a.runs, *key)
            status[key] = submitted(tr) if tr else None
        if status[key]:
            keep.append(r)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    kept_runs = sum(1 for v in status.values() if v)
    print(f"{len(keep)} of {len(rows)} examples from {kept_runs} of {len(status)} runs that called submit_patch "
          f"({sum(1 for v in status.values() if v is None)} without a trace) -> {a.out}")
