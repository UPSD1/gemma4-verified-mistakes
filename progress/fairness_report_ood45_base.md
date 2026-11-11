# Fairness report

45 tasks rated by all 2 raters: ood45 45.

## 1. Flagged tasks per rater

| Rater | ood45 | All |
|---|---:|---:|
| claude-opus-5-5 | 0% (0/45) | 0% (0/45) |
| gpt-6.1-sol | 0% (0/45) | 0% (0/45) |

## 2. Agreement between raters

| Pair | Scale | Exact | Kappa | Weighted kappa |
|---|---|---:|---:|---:|
| claude-opus-5-5 vs gpt-6.1-sol | clarity | 100% | 1.00 | 1.00 |
| claude-opus-5-5 vs gpt-6.1-sol | test_fit | 82% | 0.28 | 0.28 |
| claude-opus-5-5 vs gpt-6.1-sol | flag | 100% | 1.00 | - |

Tasks flagged by: 0 raters 45

## 3. Solve rate by rating


ood45:

| Arm | no rater flagged | one flagged | all flagged | worst score 0-1 | worst score 2-3 |
|---|---:|---:|---:|---:|---:|
| base | 67% (89/132) | - | - | 67% (89/132) | - |

all:

| Arm | no rater flagged | one flagged | all flagged | worst score 0-1 | worst score 2-3 |
|---|---:|---:|---:|---:|---:|
| base | 67% (89/132) | - | - | 67% (89/132) | - |

## 4. Does the rating predict difficulty? (all arms pooled)

| Worst score | Tasks with results | Solve rate (runs) |
|---:|---:|---:|
| 0 | 35 | 71% (75/105) |
| 1 | 9 | 52% (14/27) |
| 2 | 0 | - |
| 3 | 0 | - |

Rated tasks with no result, per arm: base 1
