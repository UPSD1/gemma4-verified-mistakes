"""Stage 1: list merged PRs that change both source code and tests.

Mirrors the organisers' pipeline: keep PRs that jointly modify functional .py code
and unit tests; drop docs-only, test-only and very large changes.

    python3 mining/mine_prs.py --repo psf/requests --limit 200
    python3 mining/mine_prs.py --all
"""
import argparse
import re

from common import REPOS, RAW_DIR, gh_graphql, is_code_file, is_test_file, write_jsonl

QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: MERGED, first: 50, after: $cursor,
                 orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title body url createdAt mergedAt baseRefOid
        changedFiles additions deletions
        files(first: 100) { nodes { path } }
        closingIssuesReferences(first: 3) {
          nodes { number title body createdAt
                  comments(first: 10) { nodes { body } } }
        }
      }
    }
  }
}
"""

MAX_FILES = 15          # larger PRs are usually refactors (organisers drop "large-scale changes")
MAX_LINES = 600
ISSUE_REF_RE = re.compile(r"\b(?:fix(?:es|ed)?|close[sd]?|resolve[sd]?)\s+#(\d+)", re.I)


def keep(pr):
    """Return a reason string if the PR should be dropped, else None."""
    paths = [f["path"] for f in pr["files"]["nodes"]]
    if pr["changedFiles"] > MAX_FILES:
        return "too_many_files"
    if pr["additions"] + pr["deletions"] > MAX_LINES:
        return "too_many_lines"
    if not any(is_test_file(p) and p.endswith(".py") for p in paths):
        return "no_test_change"
    if not any(is_code_file(p) for p in paths):
        return "no_code_change"
    return None


def to_candidate(repo, short, pr):
    issues = pr["closingIssuesReferences"]["nodes"]
    if issues:
        iss = issues[0]
        problem = f"{iss['title']}\n\n{iss['body'] or ''}".strip()
        hints = "\n\n".join(c["body"] for c in iss["comments"]["nodes"] if c["body"])
        source = f"issue#{iss['number']}"
    else:
        # No linked issue: fall back to the PR description, like the organisers'
        # "merged-PR titles" tasks. Flag it so we can filter or analyse later.
        problem = f"{pr['title']}\n\n{pr['body'] or ''}".strip()
        hints = ""
        source = "pr_body"
    return {
        "instance_id": f"{short}_{pr['number']}",
        "repo": repo,
        "pr_number": pr["number"],
        "pr_url": pr["url"],
        "base_commit": pr["baseRefOid"],
        "problem_statement": problem,
        "hints_text": hints,
        "problem_source": source,
        "created_at": pr["createdAt"],
        "merged_at": pr["mergedAt"],
        "files": [f["path"] for f in pr["files"]["nodes"]],
    }


def mine(repo, limit=None):
    owner, name = repo.split("/")
    short = REPOS[repo]
    cursor, seen, kept, drops = None, 0, [], {}
    while True:
        data = gh_graphql(QUERY, owner=owner, name=name, cursor=cursor)
        page = data["repository"]["pullRequests"]
        for pr in page["nodes"]:
            seen += 1
            reason = keep(pr)
            if reason:
                drops[reason] = drops.get(reason, 0) + 1
            else:
                kept.append(to_candidate(repo, short, pr))
            if limit and seen >= limit:
                break
        print(f"  {repo}: scanned {seen}, kept {len(kept)}", flush=True)
        if (limit and seen >= limit) or not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]
    out = RAW_DIR / f"{short}_candidates.jsonl"
    write_jsonl(out, kept)
    print(f"{repo}: {seen} merged PRs scanned, {len(kept)} kept -> {out}")
    print(f"  dropped: {drops}")
    no_issue = sum(c["problem_source"] == "pr_body" for c in kept)
    print(f"  without a linked issue: {no_issue}")
    return kept


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", choices=list(REPOS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, help="max merged PRs to scan per repo")
    a = ap.parse_args()
    repos = list(REPOS) if a.all else [a.repo]
    if repos == [None]:
        ap.error("pass --repo or --all")
    for r in repos:
        mine(r, a.limit)
