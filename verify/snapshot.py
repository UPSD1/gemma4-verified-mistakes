"""Copy the verification log into progress/ (tracked in git) with a STATUS.md summary,
so teammates can see where the run is without the data/ folder.

Verified rows (with their fail_logs) are published only for tasks in
progress/train_task_ids.json; others (competition public tasks, shared base commits)
are left out of the published log.

    python3 verify/snapshot.py
    python3 verify/snapshot.py --since 2021-01-01 data/tasks/verify_log.jsonl other/verify_log.jsonl
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mining"))
from common import ROOT, TASK_DIR, read_jsonl  # noqa: E402

OUT = ROOT / "progress"
IDS_FILE = OUT / "train_task_ids.json"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", default=[str(TASK_DIR / "verify_log.jsonl")])
    ap.add_argument("--tasks", default=str(TASK_DIR / "candidates.jsonl"))
    ap.add_argument("--since", default="2021-01-01", help="same filter as the verify run")
    a = ap.parse_args()

    done = {}
    for path in a.logs:
        for r in read_jsonl(path):
            prev = done.get(r["instance_id"])
            if prev is None or prev["status"] != "verified":
                done[r["instance_id"]] = r
    keep = set(json.loads(IDS_FILE.read_text())["mined_verified"]["ids"]) if IDS_FILE.exists() else None
    excluded = {tid for tid, r in done.items() if keep is not None and r["status"] == "verified" and tid not in keep}
    tasks = [t for t in read_jsonl(a.tasks) if t["created_at"] >= a.since]
    total = len(tasks)
    repo_of = {t["instance_id"]: t["repo"] for t in tasks}

    OUT.mkdir(exist_ok=True)
    with open(OUT / "verify_log.jsonl", "w") as f:
        for tid, r in done.items():
            if tid not in excluded:
                f.write(json.dumps(r) + "\n")

    status = Counter(r["status"].split(":")[0] for r in done.values())
    by_repo = {}
    for tid, r in done.items():
        repo = repo_of.get(tid, "?")
        c = by_repo.setdefault(repo, Counter())
        c["logged"] += 1
        c["verified"] += r["status"] == "verified"
        c["clean"] += r["status"] == "verified" and tid not in excluded
    n_ver = status.get("verified", 0)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Verification progress",
        "",
        f"Snapshot: {now}. Tasks created on/after {a.since}: **{total}**.",
        "",
        f"- Logged: **{len(done)} / {total}** ({100 * len(done) / max(total, 1):.0f}%)",
        f"- Verified: **{n_ver}** ({100 * n_ver / max(len(done), 1):.0f}% of logged)",
        f"- Remaining: **{max(total - len(done), 0)}**",
        *([f"- Clean for training: **{n_ver - len(excluded)}** (in `progress/train_task_ids.json`); "
           f"{len(excluded)} verified tasks excluded as competition public tasks or shared base commits, "
           "and left out of the published log"] if keep is not None else []),
        "",
        "## By status",
        "",
        "| Status | Tasks |",
        "|---|---:|",
        *[f"| {s} | {n} |" for s, n in status.most_common()],
        "",
        "## By repo",
        "",
        "| Repo | Logged | Verified | Clean |",
        "|---|---:|---:|---:|",
        *[f"| {repo} | {c['logged']} | {c['verified']} | {c['clean']} |"
          for repo, c in sorted(by_repo.items(), key=lambda kv: -kv[1]["logged"])],
        "",
        "Per-task results: `progress/verify_log.jsonl` (one line per task; `fail_log` on verified rows).",
        "To skip tasks already done here, copy it to `data/tasks/verify_log.jsonl` before running",
        "`verify/verify_tasks.py --shard 1/2`.",
        "",
    ]
    (OUT / "STATUS.md").write_text("\n".join(lines))
    print(f"{len(done)}/{total} logged, {n_ver} verified -> {OUT}/STATUS.md")
