# Gemma 4 Paper Track — Learning from Verified Mistakes

Plan doc: https://claude.ai/artifact/T3JFGy1JPmAwAZDaA1ghKD
Deadline: 2026-11-12 23:59 UTC (target submit 18:00 UTC).

## Layout

| Folder | What |
| --- | --- |
| `mining/` | Find candidate bug-fix PRs in fastapi, rich, requests, httpx and split them into tasks |
| `verify/` | Fail-before / pass-after check of each task inside Docker |
| `runpod/` | GPU pod setup: vLLM + competition harness |
| `data/raw/` | Raw PR metadata from GitHub (not committed) |
| `data/tasks/` | Verified tasks in the competition's `tasks.jsonl` format |

## Step 1: mine candidates

Needs the GitHub CLI (`gh auth login`). No other dependencies.

```bash
python3 mining/mine_prs.py --repo psf/requests --limit 200
python3 mining/mine_prs.py --all
```

Output: `data/raw/<repo>_candidates.jsonl`, then

```bash
python3 mining/build_tasks.py --exclude data/public_tasks.jsonl
```

writes `data/tasks/candidates.jsonl` with `patch` / `test_patch` split.
## Step 2: verify (Docker)

```bash
docker build -t python:3.12-slim-git verify/
python3 verify/verify_tasks.py --workers 4
```

Writes `data/tasks/verified.jsonl` with `FAIL_TO_PASS` / `PASS_TO_PASS` test lists. Resumable: already-checked tasks are skipped via `verify_log.jsonl`.
Tested on 3 requests PRs: all 3 verified.

## Step 3: GPU pod

`runpod/setup.sh` downloads the competition data and starts vLLM. Open issue: the harness runs agent sandboxes in Docker, and RunPod pods are themselves containers without Docker. Read `HARNESS_README.md` first to see whether it has a no-Docker mode.

`data/public_tasks.jsonl` is the competition's `tasks.jsonl` (download after accepting the rules); any PR whose number or base commit matches a public task is dropped.
