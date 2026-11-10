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
plus the tests and the gold fix. Any OpenAI-compatible endpoint works (Anthropic, OpenAI,
vLLM on RunPod, OpenRouter). Each rater is a prefix of settings in .env, picked with
--rater (default RATER):

  <P>_BASE_URL  <P>_MODEL                   endpoint, model id
  key: ANTHROPIC_API_KEY for api.anthropic.com, OPENAI_API_KEY for api.openai.com,
       otherwise <P>_API_KEY (e.g. vLLM or OpenRouter)
  <P>_PRICE_IN  <P>_PRICE_OUT  <P>_MAX_COST  USD per 1M tokens and spend cap (optional)
  <P>_REASONING_EFFORT                       set for reasoning models (low|medium|high);
                                             sends max_completion_tokens and no temperature

Outputs are named per rater, so several raters can rate the same file and be compared
with verify/fairness_compare.py.

Competition data is never sent to outside APIs: tasks whose instance_id is in
data/comp/public_task_ids.txt are skipped (the 129 public tasks). Rate test sets that are
public GitHub content (OOD-45, the gRPC slice); verify/build_screen_set.py combines them.

    python3 verify/fairness_screen.py --in data/tasks/grpc/verified.jsonl --dry-run
    python3 verify/fairness_screen.py --in data/tasks/grpc/verified.jsonl
    python3 verify/fairness_screen.py --in data/ood45.jsonl --out data/ood45.rated.jsonl
    python3 verify/fairness_screen.py --in data/tasks/screen/test_sets.jsonl --rater OPUS
    python3 verify/fairness_screen.py --in data/tasks/screen/test_sets.jsonl --rater SOL
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
from common import ROOT, read_jsonl, write_jsonl  # noqa: E402

PUBLIC_IDS = ROOT / "data" / "comp" / "public_task_ids.txt"

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


def chat(base_url, model, api_key, prompt, temperature, max_tokens=400, reasoning_effort=None, timeout=300):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if reasoning_effort:  # reasoning models: thinking counts toward the limit, temperature is fixed
        body.update(reasoning_effort=reasoning_effort, max_completion_tokens=max_tokens)
    else:
        body.update(temperature=temperature, max_tokens=max_tokens)
    body = json.dumps(body).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    usage = data.get("usage") or {}
    text = data["choices"][0]["message"].get("content") or ""
    return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def screen(task, ask, ratings, threshold, on_error=None):
    """ask(prompt) -> reply text. Collects `ratings` valid ratings (up to 2 extra tries)."""
    prompt, got, errors = build_prompt(task), [], 0
    while len(got) < ratings and errors < ratings + 2:
        try:
            r = parse_rating(ask(prompt))
        except Exception as e:  # network or server error: count it and retry
            r = None
            if on_error:
                on_error(e)
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
    ap.add_argument("--rater", required=True, help="prefix of this rater's settings in .env: OPUS or SOL")
    ap.add_argument("--out", help="default: <in>.<rater>.rated.jsonl")
    ap.add_argument("--log", help="default: <in>.<rater>.fairness_log.jsonl (resumable)")
    ap.add_argument("--ratings", type=int, default=1, help="ratings per task; the worst counts")
    ap.add_argument("--threshold", type=int, default=2, help="flag if either score >= this")
    ap.add_argument("--filter", action="store_true", help="drop flagged tasks instead of only reporting")
    ap.add_argument("--temperature", type=float, default=0.0, help="ignored for reasoning models")
    ap.add_argument("--max-tokens", type=int, help="reply limit; default 400, or 4000 for reasoning models")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--price-in", type=float, help="USD per 1M input tokens (default <P>_PRICE_IN)")
    ap.add_argument("--price-out", type=float, help="USD per 1M output tokens (default <P>_PRICE_OUT)")
    ap.add_argument("--max-cost", type=float, help="stop starting new tasks at this spend, USD (default <P>_MAX_COST)")
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and exit")
    a = ap.parse_args()

    P = a.rater.upper()
    env = lambda k: os.environ.get(f"{P}_{k}") or None  # noqa: E731
    num = lambda v: float(v) if v else None  # noqa: E731
    a.price_in = a.price_in if a.price_in is not None else num(env("PRICE_IN")) or 0.0
    a.price_out = a.price_out if a.price_out is not None else num(env("PRICE_OUT")) or 0.0
    a.max_cost = a.max_cost if a.max_cost is not None else num(env("MAX_COST"))
    effort = env("REASONING_EFFORT")
    a.max_tokens = a.max_tokens or (4000 if effort else 400)
    stem = str(Path(a.inp).with_suffix("")) + "." + P.lower()
    a.out, a.log = a.out or stem + ".rated.jsonl", a.log or stem + ".fairness_log.jsonl"
    tasks = read_jsonl(a.inp)
    if PUBLIC_IDS.exists():
        public = set(PUBLIC_IDS.read_text().split())
        n = len(tasks)
        tasks = [t for t in tasks if t["instance_id"] not in public]
        if len(tasks) < n:
            print(f"skipped {n - len(tasks)} competition public tasks (not sent to outside APIs)")
    else:
        print(f"WARNING: {PUBLIC_IDS} missing; cannot check for competition tasks")
    tasks = tasks[: a.limit] if a.limit else tasks
    if a.dry_run:
        print(build_prompt(tasks[0]))
        sys.exit(0)
    base_url, model = env("BASE_URL"), env("MODEL")
    key_var = ("ANTHROPIC_API_KEY" if "anthropic.com" in base_url else
               "OPENAI_API_KEY" if "openai.com" in base_url else f"{P}_API_KEY") if base_url else ""
    key = os.environ.get(key_var)
    if not base_url or not model:
        sys.exit(f"Set {P}_BASE_URL and {P}_MODEL in .env, then: set -a; source .env; set +a")
    if not key and "127.0.0.1" not in base_url and "localhost" not in base_url:
        sys.exit(f"{key_var} is empty: put the key in .env, then: set -a; source .env; set +a")

    log_path = Path(a.log)
    done = {r["instance_id"]: r for r in read_jsonl(log_path) if r["status"] != "rater_failed"} \
        if log_path.exists() else {}
    todo = [t for t in tasks if t["instance_id"] not in done]
    est_in = sum(len(build_prompt(t)) for t in todo) / 3.5 * a.ratings
    est_out = 150 + (1500 if effort else 0)  # reasoning models think before answering
    est = (est_in * a.price_in + len(todo) * a.ratings * est_out * a.price_out) / 1e6
    print(f"{len(tasks)} tasks, {len(todo)} to rate ({len(done)} already rated) with {model}; "
          f"estimated cost ${est:.2f}" + (f", cap ${a.max_cost:.2f}" if a.max_cost else ""))
    lock = threading.Lock()
    spent, errors_seen, successes = [0.0], [], [0]

    def on_error(e):
        with lock:
            if len(errors_seen) < 3:  # show the first few so a bad key or model id is obvious
                detail = e.read().decode(errors="replace")[:300] if hasattr(e, "read") else ""
                print(f"  rater error: {e} {detail}", flush=True)
            errors_seen.append(e)

    def ask(prompt):
        text, tin, tout = chat(base_url, model, key or "none", prompt, a.temperature, a.max_tokens, effort)
        with lock:
            successes[0] += 1
            spent[0] += (tin * a.price_in + tout * a.price_out) / 1e6
        return text

    def run(t):
        if a.max_cost and spent[0] >= a.max_cost:
            return  # over budget: left unrated, so a later run resumes here
        if len(errors_seen) >= 10 and not successes[0]:
            return  # every call so far failed (bad key, model id or URL): stop instead of retrying all
        r = screen(t, ask, a.ratings, a.threshold, on_error)
        r["rater"] = model
        if r["status"] == "rater_failed":
            return  # not logged, so a rerun retries it
        with lock:
            done[t["instance_id"]] = r
            with open(log_path, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"  {t['instance_id']}: {r['status']} {r.get('clarity', '')}/{r.get('test_fit', '')}", flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(run, todo))
    if len(errors_seen) >= 10 and not successes[0]:
        sys.exit("stopped: the first 10 calls all failed; fix the error above and rerun")
    print(f"spent ${spent[0]:.2f}" + (f" of ${a.max_cost:.2f} cap" if a.max_cost else "")
          + (" (cap reached; rerun later to finish)" if a.max_cost and spent[0] >= a.max_cost else ""))

    kept = []
    for t in tasks:
        r = done.get(t["instance_id"], {"status": "rater_failed"})
        if a.filter and r["status"] != "ok":
            continue
        fair = {k: r[k] for k in ("clarity", "test_fit") if k in r}
        fair["reasons"] = [x["reason"] for x in r.get("ratings", [])]
        fair["rater"] = r.get("rater", model)
        kept.append({**t, "meta": {**t.get("meta", {}), "fairness": fair,
                                   "fairness_flag": r["status"] != "ok"}})
    write_jsonl(a.out, kept)
    by_source = Counter(("test_log" if t.get("meta", {}).get("problem_source") == "test_log" else "issue",
                         done.get(t["instance_id"], {}).get("status", "unrated")) for t in tasks)
    print(f"{'kept' if a.filter else 'wrote'} {len(kept)} of {len(tasks)} -> {a.out}")
    print("status:", dict(Counter(r["status"] for r in done.values())))
    print("by bug-report source:", {f"{s}/{st}": n for (s, st), n in sorted(by_source.items())})
