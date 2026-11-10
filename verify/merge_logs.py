"""Merge verify_log.jsonl files from several machines (verify_tasks.py --shard K/N)
into one log and one verified.jsonl.

    python3 verify/merge_logs.py data/tasks/verify_log.jsonl ~/from_other_machine/verify_log.jsonl
    python3 verify/merge_logs.py progress/verify_log.jsonl --ids progress/train_task_ids.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mining"))
from common import ROOT, TASK_DIR, read_jsonl, write_jsonl  # noqa: E402

IDS_FILE = ROOT / "progress" / "train_task_ids.json"


def load_ids(path):
    """Training task IDs: progress/train_task_ids.json (mined_verified.ids) or a JSON list."""
    d = json.loads(Path(path).read_text())
    return set(d["mined_verified"]["ids"] if isinstance(d, dict) else d)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="verify_log.jsonl files to merge")
    ap.add_argument("--tasks", default=str(TASK_DIR / "candidates.jsonl"))
    ap.add_argument("--out", default=str(TASK_DIR / "verified.jsonl"))
    ap.add_argument("--log-out", default=str(TASK_DIR / "verify_log.merged.jsonl"))
    ap.add_argument("--ids", default=str(IDS_FILE) if IDS_FILE.exists() else None,
                    help="keep only these verified tasks (default: progress/train_task_ids.json, "
                         "which drops competition public tasks and shared base commits)")
    a = ap.parse_args()

    done = {}
    for path in a.logs:
        for r in read_jsonl(path):
            prev = done.get(r["instance_id"])
            # A task verified anywhere counts as verified; otherwise keep the latest status.
            if prev is None or prev["status"] != "verified":
                done[r["instance_id"]] = r
    by_id = {t["instance_id"]: t for t in read_jsonl(a.tasks)}
    if a.ids:
        keep = load_ids(a.ids)
        by_id = {k: v for k, v in by_id.items() if k in keep}
    verified = [{**by_id[tid], "FAIL_TO_PASS": r["FAIL_TO_PASS"], "PASS_TO_PASS": r["PASS_TO_PASS"],
                 "fail_log": r.get("fail_log", "")}
                for tid, r in done.items() if r["status"] == "verified" and tid in by_id]
    write_jsonl(a.out, verified)
    with open(a.log_out, "w") as f:
        for r in done.values():
            f.write(json.dumps(r) + "\n")
    print(f"{len(done)} tasks logged, {len(verified)} verified" +
          (f" and in {a.ids}" if a.ids else "") + f" -> {a.out}")
    print(dict(Counter(r["status"].split(":")[0] for r in done.values())))
