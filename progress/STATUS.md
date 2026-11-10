# Verification progress

Snapshot: 2026-11-10 08:18 UTC. Tasks created on/after 2021-01-01: **552**.

- Logged: **207 / 552** (38%)
- Verified: **96** (46% of logged)
- Remaining: **345**

## By status

| Status | Tasks |
|---|---:|
| verified | 96 |
| no_fail_to_pass | 67 |
| no_results | 24 |
| setup_failed | 9 |
| tests_fail_after_fix | 9 |
| timeout | 2 |

## By repo

| Repo | Logged | Verified |
|---|---:|---:|
| Textualize/rich | 72 | 41 |
| fastapi/fastapi | 68 | 25 |
| encode/httpx | 54 | 26 |
| psf/requests | 13 | 4 |

Per-task results: `progress/verify_log.jsonl` (one line per task; `fail_log` on verified rows).
To skip tasks already done here, copy it to `data/tasks/verify_log.jsonl` before running
`verify/verify_tasks.py --shard 1/2`.
