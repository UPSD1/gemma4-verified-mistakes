# Fairness report

184 tasks rated by all 2 raters: mined 184.

## 1. Flagged tasks per rater

| Rater | mined | All |
|---|---:|---:|
| claude-opus-5-5 | 46% (85/184) | 46% (85/184) |
| gpt-6.1-sol | 45% (83/184) | 45% (83/184) |

By bug-report format (what the agent sees):

| Format | Tasks | flagged by claude-opus-5-5 | flagged by gpt-6.1-sol |
|---|---:|---:|---:|
| ci_log | 46 | 52% (24/46) | 50% (23/46) |
| issue | 57 | 47% (27/57) | 49% (28/57) |
| pr_body | 81 | 42% (34/81) | 40% (32/81) |

## 2. Agreement between raters

| Pair | Scale | Exact | Kappa | Weighted kappa |
|---|---|---:|---:|---:|
| claude-opus-5-5 vs gpt-6.1-sol | clarity | 51% | 0.30 | 0.71 |
| claude-opus-5-5 vs gpt-6.1-sol | test_fit | 53% | 0.39 | 0.80 |
| claude-opus-5-5 vs gpt-6.1-sol | flag | 95% | 0.89 | - |

Tasks flagged by: 0 raters 95, 1 raters 10, 2 raters 79

(No --results given: sections 3-4, solve rate by rating, appear once runs exist.)
