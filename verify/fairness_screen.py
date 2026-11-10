"""Stage 5: fairness screen, adapted from SWE-bench Verified.

Rates whether an agent can reasonably solve each task from what it is shown, on two 0-3
scales. Used as a REPORT, not a filter: every task is kept, with its ratings in
meta.fairness and meta.fairness_flag (either score >= --threshold), so results can be
sliced by them. --filter drops flagged tasks instead. With --ratings N > 1 the worst
(highest) rating counts.

  clarity   0 = clear what to fix, 1 = some blanks but a sensible reading exists,
            2 = vague, several readings, 3 = impossible to know what is asked
  test_fit  0 = tests check only what the bug report asks for, 1 = small extra details
            a good fix would likely match, 2 = tests need details the report never gives
            (exact message, new function name), 3 = tests check something unrelated

The rater sees what the agent sees (problem_statement, after verify/log_statements.py),
plus the tests and the gold fix. Any OpenAI-compatible endpoint works (vLLM on RunPod,
OpenRouter, ...); set RATER_BASE_URL, RATER_MODEL and RATER_API_KEY (in .env).

    python3 verify/fairness_screen.py --in data/tasks/grpc/verified.jsonl --dry-run
    python3 verify/fairness_screen.py --in data/tasks/grpc/verified.jsonl
    python3 verify/fairness_screen.py --in data/ood45.jsonl --out data/ood45.rated.jsonl
"""
import argparse
import json
import os
import re
import sys
import threading
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mining"))
from common import read_jsonl, write_jsonl  # noqa: E402

MAX_CHARS = {"problem_statement": 6000, "test_patch": 8000, "patch": 6000}

PROMPT = """You are screening software-engineering tasks for an AI coding agent benchmark.
The agent sees ONLY the bug report below and the repository at its parent commit.
It must change the source code so that hidden tests pass. Judge whether that is fair.

Rate two things from 0 to 3.

clarity: is it clear what needs to change?
  0 = clear what the problem is and what a fix must do
  1 = some blanks, but there is a sensible reading of what a fix must do
  2 = vague; several readings are possible and it is unclear which one is wanted
  3 = almost impossible to tell what is asked without more information

test_fit: do the hidden tests check only what the bug report asks for?
  0 = the tests accept any reasonable fix of the problem described
  1 = the tests rely on small details a good fix would very likely match
  2 = the tests need details the report never gives (an exact error message,
      a new function or argument name, a specific output format), so many
      reasonable fixes would fail
  3 = the tests check something different from, or far beyond, what is described

Repository: {repo}

## Bug report shown to the agent
{problem_statement}

## Hidden tests (diff)
{test_patch}

## Reference fix (diff; not shown to the agent)
{patch}

Answer with one JSON object and nothing else:
{{"clarity": <0-3>, "test_fit": <0-3>, "reason": "<one sentence on the main problem, or 'fine'>"}}"""


def clip(text, n):
    return text if len(text) <= n else text[:n] + f"\n[... {len(text) - n} more characters cut]"


def build_prompt(task):
    return PROMPT.format(repo=task["repo"], **{k: clip(task.get(k) or "", n) for k, n in MAX_CHARS.items()})


def parse_rating(text):
    """Pull {"clarity", "test_fit", "reason"} out of a reply; None if it isn't there."""
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, re.S))):
        try:
            r = json.loads(m.group(0))
            c, t = int(r["clarity"]), int(r["test_fit"])
        except (ValueError, KeyError, TypeError):
            continue
        if 0 <= c <= 3 and 0 <= t <= 3:
            return {"clarity": c, "test_fit": t, "reason": str(r.get("reason", ""))[:300]}
    return None


def chat(base_url, model, api_key, prompt, temperature, timeout=180):
    body = json.dumps({"model": model, "temperature": temperature, "max_tokens": 400,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def screen(task, ask, ratings, threshold):
    """ask(prompt) -> reply text. Collects `ratings` valid ratings (up to 2 extra tries)."""
    prompt, got, errors = build_prompt(task), [], 0
    while len(got) < ratings and errors < ratings + 2:
        try:
            r = parse_rating(ask(prompt))
        except Exception:  # network or server error: count it and retry
            r = None
        if r:
            got.append(r)
        else:
            errors += 1
    row = {"instance_id": task["instance_id"], "ratings": got}
    if len(got) < ratings:
        row["status"] = "rater_failed"
        return row
    row["clarity"] = max(r["clarity"] for r in got)
    row["test_fit"] = max(r["test_fit"] for r in got)
    row["status"] = "ok" if row["clarity"] < threshold and row["test_fit"] < threshold else "flagged"
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="tasks.jsonl to rate (test sets, gRPC slice)")
    ap.add_argument("--out", help="default: <in>.rated.jsonl")
    ap.add_argument("--log", help="default: <in>.fairness_log.jsonl (resumable)")
    ap.add_argument("--ratings", type=int, default=1, help="ratings per task; the worst counts")
    ap.add_argument("--threshold", type=int, default=2, help="flag if either score >= this")
    ap.add_argument("--filter", action="store_true", help="drop flagged tasks instead of only reporting")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and exit")
    a = ap.parse_args()

    stem = str(Path(a.inp).with_suffix(""))
    a.out, a.log = a.out or stem + ".rated.jsonl", a.log or stem + ".fairness_log.jsonl"
    tasks = read_jsonl(a.inp)[: a.limit] if a.limit else read_jsonl(a.inp)
    if a.dry_run:
        print(build_prompt(tasks[0]))
        sys.exit(0)
    base_url, model, key = (os.environ.get(k) for k in ("RATER_BASE_URL", "RATER_MODEL", "RATER_API_KEY"))
    if not base_url or not model:
        sys.exit("Set RATER_BASE_URL and RATER_MODEL (and RATER_API_KEY) in .env, then: set -a; source .env; set +a")

    log_path = Path(a.log)
    done = {r["instance_id"]: r for r in read_jsonl(log_path) if r["status"] != "rater_failed"} \
        if log_path.exists() else {}
    todo = [t for t in tasks if t["instance_id"] not in done]
    print(f"{len(tasks)} tasks, {len(todo)} to rate ({len(done)} already rated) with {model}")
    lock = threading.Lock()

    def run(t):
        r = screen(t, lambda p: chat(base_url, model, key or "none", p, a.temperature), a.ratings, a.threshold)
        with lock:
            done[t["instance_id"]] = r
            with open(log_path, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"  {t['instance_id']}: {r['status']} {r.get('clarity', '')}/{r.get('test_fit', '')}", flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(run, todo))

    kept = []
    for t in tasks:
        r = done.get(t["instance_id"], {"status": "rater_failed"})
        if a.filter and r["status"] != "ok":
            continue
        fair = {k: r[k] for k in ("clarity", "test_fit") if k in r}
        fair["reasons"] = [x["reason"] for x in r.get("ratings", [])]
        kept.append({**t, "meta": {**t.get("meta", {}), "fairness": fair,
                                   "fairness_flag": r["status"] != "ok"}})
    write_jsonl(a.out, kept)
    by_source = Counter(("test_log" if t.get("meta", {}).get("problem_source") == "test_log" else "issue",
                         done.get(t["instance_id"], {}).get("status", "unrated")) for t in tasks)
    print(f"{'kept' if a.filter else 'wrote'} {len(kept)} of {len(tasks)} -> {a.out}")
    print("status:", dict(Counter(r["status"] for r in done.values())))
    print("by bug-report source:", {f"{s}/{st}": n for (s, st), n in sorted(by_source.items())})
