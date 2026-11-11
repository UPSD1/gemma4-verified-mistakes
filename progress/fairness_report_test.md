# Fairness report

85 tasks rated by all 2 raters: grpc 40, ood45 45.

## 1. Flagged tasks per rater

| Rater | grpc | ood45 | All |
|---|---:|---:|---:|
| claude-opus-5-5 | 60% (24/40) | 0% (0/45) | 28% (24/85) |
| gpt-6.1-sol | 50% (20/40) | 0% (0/45) | 24% (20/85) |

gRPC slice by tag (hand-checked labels in progress/grpc_tags.json when present):

| Tag | Tasks | flagged by claude-opus-5-5 | flagged by gpt-6.1-sol |
|---|---:|---:|---:|
| grpc_path | 16 | 44% (7/16) | 31% (5/16) |
| other | 24 | 71% (17/24) | 62% (15/24) |

## 2. Agreement between raters

| Pair | Scale | Exact | Kappa | Weighted kappa |
|---|---|---:|---:|---:|
| claude-opus-5-5 vs gpt-6.1-sol | clarity | 73% | 0.44 | 0.76 |
| claude-opus-5-5 vs gpt-6.1-sol | test_fit | 71% | 0.50 | 0.85 |
| claude-opus-5-5 vs gpt-6.1-sol | flag | 93% | 0.82 | - |

Tasks flagged by: 0 raters 60, 1 raters 6, 2 raters 19

(No --results given: sections 3-4, solve rate by rating, appear once runs exist.)
