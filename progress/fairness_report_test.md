# Fairness report

85 tasks rated by all 2 raters: grpc 40, ood45 45.

## 1. Flagged tasks per rater

| Rater | grpc | ood45 | All |
|---|---:|---:|---:|
| claude-opus-5-5 | 60% (24/40) | 0% (0/45) | 28% (24/85) |
| gpt-6.1-sol | 50% (20/40) | 0% (0/45) | 24% (20/85) |

gRPC slice by tag (automatic; check by hand before quoting):

| Tag | Tasks | flagged by claude-opus-5-5 | flagged by gpt-6.1-sol |
|---|---:|---:|---:|
| grpc_adjacent | 10 | 40% (4/10) | 30% (3/10) |
| grpc_path | 12 | 50% (6/12) | 42% (5/12) |
| other | 18 | 78% (14/18) | 67% (12/18) |

## 2. Agreement between raters

| Pair | Scale | Exact | Kappa | Weighted kappa |
|---|---|---:|---:|---:|
| claude-opus-5-5 vs gpt-6.1-sol | clarity | 73% | 0.44 | 0.76 |
| claude-opus-5-5 vs gpt-6.1-sol | test_fit | 71% | 0.50 | 0.85 |
| claude-opus-5-5 vs gpt-6.1-sol | flag | 93% | 0.82 | - |

Tasks flagged by: 0 raters 60, 1 raters 6, 2 raters 19

(No --results given: sections 3-4, solve rate by rating, appear once runs exist.)
