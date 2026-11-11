# Fairness report

647 tasks rated by all 2 raters: freshswe 647.

## 1. Flagged tasks per rater

| Rater | freshswe | All |
|---|---:|---:|
| claude-opus-5-5 | 36% (230/647) | 36% (230/647) |
| gpt-6.1-sol | 31% (203/647) | 31% (203/647) |

## 2. Agreement between raters

| Pair | Scale | Exact | Kappa | Weighted kappa |
|---|---|---:|---:|---:|
| claude-opus-5-5 vs gpt-6.1-sol | clarity | 67% | 0.34 | 0.62 |
| claude-opus-5-5 vs gpt-6.1-sol | test_fit | 64% | 0.46 | 0.80 |
| claude-opus-5-5 vs gpt-6.1-sol | flag | 93% | 0.85 | - |

Tasks flagged by: 0 raters 409, 1 raters 43, 2 raters 195

## 3. Solve rate by rating


freshswe:

| Arm | no rater flagged | one flagged | all flagged | worst score 0-1 | worst score 2-3 |
|---|---:|---:|---:|---:|---:|
| base | 44% (213/484) | 13% (9/68) | 2% (6/248) | 44% (213/484) | 5% (15/316) |

all:

| Arm | no rater flagged | one flagged | all flagged | worst score 0-1 | worst score 2-3 |
|---|---:|---:|---:|---:|---:|
| base | 44% (213/484) | 13% (9/68) | 2% (6/248) | 44% (213/484) | 5% (15/316) |

## 4. Does the rating predict difficulty? (all arms pooled)

| Worst score | Tasks with results | Solve rate (runs) |
|---:|---:|---:|
| 0 | 71 | 60% (171/284) |
| 1 | 50 | 21% (42/200) |
| 2 | 54 | 7% (15/216) |
| 3 | 25 | 0% (0/100) |

Rated tasks with no result, per arm: base 447
