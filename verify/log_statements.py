"""Stage 4: bug-report format for the mined training tasks.

The competition's own bug reports are PR descriptions, so that stays the main format:
tasks keep their linked issue or PR description. A labelled subset (at most --log-share,
default 25%, drawn from tasks with no linked issue) gets a CI-style failure log instead,
built from the pre-fix test run (fail_log, written by verify_tasks.py).

The log must not reveal the hidden tests, so test names and test-file paths are removed:
  - pytest section titles  "____ test_wrap_file ____"   -> "____ failing test 1 ____"
  - frames in test files    "tests/test_x.py:617: in test_y" and the test code under them
  - summary lines           "FAILED tests/x.py::test_y - Err" -> "FAILED <test> - Err"
  - any other test path or test name                    -> "<test>"
Error messages ("E   AssertionError: ...") and frames in the project's own code stay.

Every task with a fail_log also gets `problem_statement_log` (the stripped log report), so
results can be compared across formats. meta.statement_format records what the agent sees:
"issue", "pr_body" or "ci_log"; for ci_log tasks the original text is kept in
meta.original_problem_statement.

    python3 verify/log_statements.py
    python3 verify/log_statements.py --log-share 0.25 --in data/tasks/verified.jsonl
"""
import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mining"))
from common import TASK_DIR, is_test_file, read_jsonl, write_jsonl  # noqa: E402

SECTION_RE = re.compile(r"^(_{3,}) (.+?) (_{3,})$")
FRAME_RE = re.compile(r"^(\S+\.py):\d+: in (\S+)")
PATH_RE = re.compile(r"[\w./-]*\.py")


def strip_log(log, f2p):
    """Remove test names and test-file paths from a pytest failure log."""
    test_names = set()
    for node in f2p:
        parts = node.split("::")
        test_names.update(p.split("[")[0] for p in parts[1:])
    out, n_section, skip_code = [], 0, False
    for line in log.splitlines():
        m = SECTION_RE.match(line)
        if m and m.group(2) not in ("short test summary info",) and not m.group(2).startswith("Captured"):
            n_section += 1
            out.append(f"{m.group(1)} failing test {n_section} {m.group(3)}")
            skip_code = False
            continue
        fr = FRAME_RE.match(line)
        if fr:
            # Frames in test files show the test's code: drop the frame and its code lines.
            skip_code = is_test_file(fr.group(1)) or fr.group(2) in test_names
            if not skip_code:
                out.append(line)
            continue
        if skip_code and line.startswith("    "):
            continue
        skip_code = False
        if line.startswith(("FAILED ", "ERROR ")):
            kind, _, rest = line.partition(" ")
            msg = rest.split(" - ", 1)[1] if " - " in rest else ""
            out.append(f"{kind} <test>" + (f" - {msg}" if msg else ""))
            continue
        out.append(line)
    text = "\n".join(out)
    # Anything left that names a test file or a test.
    text = PATH_RE.sub(lambda m: "<test>" if is_test_file(m.group(0)) else m.group(0), text)
    for name in sorted(test_names, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(name)}\b", "<test>", text)
    return text.strip()


def log_statement(task):
    n = len(task["FAIL_TO_PASS"])
    return (
        f"CI is failing: {n} test{'s fail' if n != 1 else ' fails'} on this commit.\n\n"
        f"```\n{strip_log(task['fail_log'], task['FAIL_TO_PASS'])}\n```\n\n"
        "Find the cause in the source code and fix it."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(TASK_DIR / "verified.jsonl"))
    ap.add_argument("--out", default=str(TASK_DIR / "train_tasks.jsonl"))
    ap.add_argument("--log-share", type=float, default=0.25, help="max share of tasks given a CI-log report")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tasks = [dict(t) for t in read_jsonl(a.inp)]
    # Candidates for the log subset: no linked issue (their text is the PR description) and a log.
    pool = sorted(t["instance_id"] for t in tasks if t.get("meta", {}).get("statement_leak_risk") and t.get("fail_log"))
    random.Random(a.seed).shuffle(pool)
    chosen = set(pool[: int(a.log_share * len(tasks))])

    for t in tasks:
        meta = dict(t.get("meta", {}))
        if t.get("fail_log"):
            t["problem_statement_log"] = log_statement(t)
        if t["instance_id"] in chosen:
            meta["original_problem_statement"] = t["problem_statement"]
            meta["statement_format"] = "ci_log"
            t["problem_statement"] = t["problem_statement_log"]
            t["hints_text"] = ""
        else:
            meta["statement_format"] = "pr_body" if meta.get("statement_leak_risk") else "issue"
        t["meta"] = meta
    write_jsonl(a.out, tasks)
    fmt = Counter(t["meta"]["statement_format"] for t in tasks)
    print(f"{len(tasks)} tasks -> {a.out}: {dict(fmt)} "
          f"(ci_log {100 * fmt['ci_log'] / max(len(tasks), 1):.0f}%, cap {a.log_share:.0%})")
