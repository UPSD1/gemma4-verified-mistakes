# Verification progress

Snapshot: 2026-11-10 10:24 UTC. Tasks created on/after 2021-01-01: **552**.

- Logged: **552 / 552** (100%)
- Verified: **268** (49% of logged)
- Remaining: **0**

## By status

| Status | Tasks |
|---|---:|
| verified | 268 |
| no_fail_to_pass | 156 |
| no_results | 57 |
| tests_fail_after_fix | 35 |
| setup_failed | 29 |
| timeout | 5 |
| no_test_files | 1 |
| flaky | 1 |

## By repo

| Repo | Logged | Verified |
|---|---:|---:|
| Textualize/rich | 206 | 125 |
| fastapi/fastapi | 184 | 66 |
| encode/httpx | 128 | 66 |
| psf/requests | 34 | 11 |

Per-task results: `progress/verify_log.jsonl` (one line per task; `fail_log` on verified rows).
To skip tasks already done here, copy it to `data/tasks/verify_log.jsonl` before running
`verify/verify_tasks.py --shard 1/2`.
