"""Stage 2: download each candidate PR's diff and split it into `patch` and `test_patch`.

Output rows use the competition's tasks.jsonl fields (instance_id, repo, base_commit,
problem_statement, hints_text, patch, test_patch, created_at) plus our own metadata.

    python3 mining/build_tasks.py                       # all data/raw/*_candidates.jsonl
    python3 mining/build_tasks.py --exclude data/public_tasks.jsonl
"""
import argparse
import re
from concurrent.futures import ThreadPoolExecutor

from common import RAW_DIR, ROOT, TASK_DIR, gh, is_test_file, read_jsonl, write_jsonl

CACHE = ROOT / "data" / "cache" / "diffs"
FILE_SPLIT_RE = re.compile(r"(?=^diff --git )", re.M)
PATH_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)
BINARY_RE = re.compile(r"^(Binary files|GIT binary patch)", re.M)


def fetch_diff(repo, number):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{repo.replace('/', '__')}_{number}.diff"
    if path.exists():
        return path.read_text()
    diff = gh(["api", "-H", "Accept: application/vnd.github.v3.diff", f"repos/{repo}/pulls/{number}"])
    path.write_text(diff)
    return diff


def split_diff(diff):
    """Return (code_patch, test_patch). Non-test files (code, config) go in the fix."""
    code, test = [], []
    for chunk in FILE_SPLIT_RE.split(diff):
        if not chunk.strip():
            continue
        m = PATH_RE.match(chunk)
        if not m:
            continue
        if BINARY_RE.search(chunk):
            return None, None  # binary files can't be applied as text patches; skip task
        (test if is_test_file(m.group(2)) else code).append(chunk)
    return "".join(code), "".join(test)


def load_excludes(path):
    """Public competition tasks: exclude by instance_id and by base commit."""
    if not path:
        return set(), set()
    rows = read_jsonl(path)
    return {r["instance_id"] for r in rows}, {r["base_commit"] for r in rows}


def build(cand, excl_ids, excl_commits):
    if cand["instance_id"] in excl_ids or cand["base_commit"] in excl_commits:
        return None, "overlaps_public"
    try:
        diff = fetch_diff(cand["repo"], cand["pr_number"])
    except RuntimeError as e:
        return None, f"diff_error: {e}"[:80]
    patch, test_patch = split_diff(diff)
    if patch is None:
        return None, "binary"
    if not patch.strip() or not test_patch.strip():
        return None, "empty_side"
    row = {k: cand[k] for k in ("instance_id", "repo", "base_commit", "problem_statement",
                                "hints_text", "created_at")}
    row.update(patch=patch, test_patch=test_patch,
               meta={k: cand[k] for k in ("pr_number", "pr_url", "problem_source", "merged_at")})
    return row, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", help="competition tasks.jsonl to exclude")
    ap.add_argument("--out", default=str(TASK_DIR / "candidates.jsonl"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    excl_ids, excl_commits = load_excludes(a.exclude)
    if not a.exclude:
        print("WARNING: no --exclude file; public-task overlap NOT removed yet")
    cands = [c for p in sorted(RAW_DIR.glob("*_candidates.jsonl")) for c in read_jsonl(p)]
    with ThreadPoolExecutor(a.workers) as pool:
        results = list(pool.map(lambda c: build(c, excl_ids, excl_commits), cands))
    rows = [r for r, _ in results if r]
    drops = {}
    for _, why in results:
        if why:
            drops[why.split(":")[0]] = drops.get(why.split(":")[0], 0) + 1
    write_jsonl(a.out, rows)
    print(f"{len(cands)} candidates -> {len(rows)} tasks -> {a.out}")
    print(f"dropped: {drops}")
