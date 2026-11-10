"""Stage 4: log-based bug reports.

Tasks without a linked issue were mined with the PR title and body as their bug report,
and a PR description often explains the fix. For those tasks (meta.statement_leak_risk),
replace the bug report with what a CI job would show: the failing tests and their short
tracebacks from the pre-fix run (fail_log, written by verify_tasks.py). The original text
is kept in meta.original_problem_statement.

Every task with a fail_log also gets `problem_statement_log`, so any set (e.g. OOD-45)
can later be run with log-only bug reports.

    python3 verify/log_statements.py
    python3 verify/log_statements.py --in data/tasks/verified.jsonl --out data/tasks/train_tasks.jsonl
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mining"))
from common import TASK_DIR, read_jsonl, write_jsonl  # noqa: E402


def test_files(test_patch):
    return sorted(set(re.findall(r"^diff --git a/\S+ b/(\S+\.py)", test_patch, re.M)))


def log_statement(task):
    files = " ".join(test_files(task["test_patch"]))
    n = len(task["FAIL_TO_PASS"])
    return (
        f"CI is failing: {n} test{'s fail' if n != 1 else ' fails'} on this commit.\n\n"
        f"$ python -m pytest {files}\n"
        f"```\n{task['fail_log'].strip()}\n```\n\n"
        "Find the cause in the source code and fix it. The tests describe the intended behaviour."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(TASK_DIR / "verified.jsonl"))
    ap.add_argument("--out", default=str(TASK_DIR / "train_tasks.jsonl"))
    ap.add_argument("--replace-all", action="store_true",
                    help="use the log-based bug report for every task, not only leak-risk ones")
    a = ap.parse_args()

    rows, replaced, no_log = [], 0, 0
    for t in read_jsonl(a.inp):
        t = dict(t)
        meta = dict(t.get("meta", {}))
        if not t.get("fail_log"):
            no_log += 1
        else:
            t["problem_statement_log"] = log_statement(t)
            if a.replace_all or meta.get("statement_leak_risk"):
                meta["original_problem_statement"] = t["problem_statement"]
                meta["problem_source"] = "test_log"
                t["problem_statement"] = t["problem_statement_log"]
                t["hints_text"] = ""
                replaced += 1
        t["meta"] = meta
        rows.append(t)
    write_jsonl(a.out, rows)
    print(f"{len(rows)} tasks -> {a.out}: {replaced} bug reports replaced with test logs, "
          f"{no_log} without a fail_log (kept as is)")
