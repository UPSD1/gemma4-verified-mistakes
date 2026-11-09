"""Fail-to-pass verification of mined tasks, the same check the organisers used.

For each task, in a fresh Docker container at base_commit:
  1. apply test_patch only        -> at least one test in the changed test files must FAIL
  2. apply the fix (patch)        -> those tests must PASS, and nothing that passed before may fail
  3. repeat step 2 (--repeats)    -> results must be identical (drop flaky tasks)

Writes data/tasks/verified.jsonl (tasks with FAIL_TO_PASS / PASS_TO_PASS lists)
and data/tasks/verify_log.jsonl (one status line per task).

    python3 verify/verify_tasks.py --limit 5 --workers 2
    python3 verify/verify_tasks.py --workers 4
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mining"))
from common import ROOT, TASK_DIR, is_test_file, read_jsonl, write_jsonl  # noqa: E402

MIRRORS = ROOT / "data" / "cache" / "mirrors"
PIP_CACHE = ROOT / "data" / "cache" / "pip"
SCRIPT = Path(__file__).resolve().parent / "in_container.sh"
RESULT_RE = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS) (\S+)")
_mirror_lock = threading.Lock()


def mirror(repo):
    """One local clone per repo; tasks clone from it with --shared (fast, no network)."""
    path = MIRRORS / repo.replace("/", "__")
    with _mirror_lock:
        if not path.exists():
            MIRRORS.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "-q", "--mirror", f"https://github.com/{repo}", str(path)], check=True)
    return path


def changed_test_files(test_patch):
    files = re.findall(r"^diff --git a/\S+ b/(\S+)", test_patch, re.M)
    return [f for f in files if f.endswith(".py") and is_test_file(f) and not f.endswith("conftest.py")]


def parse(output, label):
    m = re.search(rf"=== RUN {label}\n(.*?)=== END {label}", output, re.S)
    if not m:
        return None
    res = {}
    for line in m.group(1).splitlines():
        r = RESULT_RE.match(line)
        if r:
            res[r.group(2)] = r.group(1)
    return res


def verify(task, image, repeats, timeout):
    tid = task["instance_id"]
    tests = changed_test_files(task["test_patch"])
    if not tests:
        return {"instance_id": tid, "status": "no_test_files"}
    with tempfile.TemporaryDirectory(prefix=f"v_{tid}_") as tmp:
        tmp = Path(tmp)
        work, tdir = tmp / "work", tmp / "task"
        tdir.mkdir()
        try:
            src = mirror(task["repo"])
            subprocess.run(["git", "clone", "-q", "--shared", str(src), str(work)], check=True)
            subprocess.run(["git", "-C", str(work), "checkout", "-q", task["base_commit"]], check=True)
        except subprocess.CalledProcessError:
            return {"instance_id": tid, "status": "checkout_failed"}
        (tdir / "test.patch").write_text(task["test_patch"])
        (tdir / "fix.patch").write_text(task["patch"])
        (tdir / "test_files.txt").write_text(" ".join(tests))
        cmd = ["docker", "run", "--rm", "--network", "bridge",
               "-e", f"REPEATS={repeats}",
               "-v", f"{work}:/work", "-v", f"{tdir}:/task:ro",
               "-v", f"{SCRIPT}:/run.sh:ro", "-v", f"{PIP_CACHE}:/root/.cache/pip",
               image, "bash", "/run.sh"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"instance_id": tid, "status": "timeout"}
    out = proc.stdout + proc.stderr
    if "SETUP_FAILED" in out:
        reason = re.search(r"SETUP_FAILED (\S+)", out).group(1)
        return {"instance_id": tid, "status": f"setup_failed:{reason}", "tail": out[-600:]}

    before = parse(out, "before")
    afters = [parse(out, f"after{i}") for i in range(1, repeats + 1)]
    if before is None or any(a is None for a in afters) or not afters[0]:
        return {"instance_id": tid, "status": "no_results", "tail": out[-600:]}
    if any(a != afters[0] for a in afters[1:]):
        return {"instance_id": tid, "status": "flaky"}
    after = afters[0]
    # Tests that fail before (or don't exist / error) and pass after.
    f2p = sorted(t for t, s in after.items() if s == "PASSED" and before.get(t) != "PASSED")
    p2p = sorted(t for t, s in after.items() if s == "PASSED" and before.get(t) == "PASSED")
    broke = sorted(t for t, s in before.items() if s == "PASSED" and after.get(t) in ("FAILED", "ERROR"))
    still_failing = sorted(t for t, s in after.items() if s in ("FAILED", "ERROR"))
    if not f2p:
        return {"instance_id": tid, "status": "no_fail_to_pass"}
    if broke:
        return {"instance_id": tid, "status": "fix_breaks_tests", "broke": broke[:5]}
    if still_failing:
        return {"instance_id": tid, "status": "tests_fail_after_fix", "failing": still_failing[:5]}
    return {"instance_id": tid, "status": "verified", "FAIL_TO_PASS": f2p, "PASS_TO_PASS": p2p}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(TASK_DIR / "candidates.jsonl"))
    ap.add_argument("--out", default=str(TASK_DIR / "verified.jsonl"))
    ap.add_argument("--image", default="python:3.12-slim-git")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", help="comma-separated instance_ids")
    a = ap.parse_args()

    PIP_CACHE.mkdir(parents=True, exist_ok=True)
    tasks = read_jsonl(a.tasks)
    if a.only:
        keep = set(a.only.split(","))
        tasks = [t for t in tasks if t["instance_id"] in keep]
    # Fixed-seed shuffle so partial runs are balanced across repos.
    import random
    random.Random(0).shuffle(tasks)
    if a.limit:
        tasks = tasks[: a.limit]
    log_path = Path(a.out).parent / "verify_log.jsonl"
    done = {r["instance_id"]: r for r in read_jsonl(log_path)} if log_path.exists() else {}
    todo = [t for t in tasks if t["instance_id"] not in done]
    print(f"{len(tasks)} tasks, {len(todo)} to verify ({len(done)} already logged)")

    lock = threading.Lock()

    def run(t):
        r = verify(t, a.image, a.repeats, a.timeout)
        with lock:
            done[t["instance_id"]] = r
            with open(log_path, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"  {t['instance_id']}: {r['status']}", flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(run, todo))

    by_id = {t["instance_id"]: t for t in tasks}
    verified = []
    for tid, r in done.items():
        if r["status"] == "verified" and tid in by_id:
            verified.append({**by_id[tid], "FAIL_TO_PASS": r["FAIL_TO_PASS"], "PASS_TO_PASS": r["PASS_TO_PASS"]})
    write_jsonl(a.out, verified)
    counts = {}
    for r in done.values():
        counts[r["status"].split(":")[0]] = counts.get(r["status"].split(":")[0], 0) + 1
    print(f"verified {len(verified)} -> {a.out}")
    print(f"status counts: {counts}")
