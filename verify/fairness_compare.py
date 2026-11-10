"""Compare fairness ratings from several raters (verify/fairness_screen.py --rater X).

    python3 verify/fairness_compare.py data/tasks/grpc/verified.opus.fairness_log.jsonl \
        data/tasks/grpc/verified.sol.fairness_log.jsonl

Prints, per pair of raters: exact agreement and Cohen's kappa (weighted and unweighted)
on each 0-3 scale, agreement on the flag (either score >= 2), and the tasks they disagree on.
"""
import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mining"))
from common import read_jsonl  # noqa: E402


def kappa(x, y, weighted=False, k=4):
    """Cohen's kappa for two lists of 0..k-1 labels; quadratic weights if weighted."""
    n = len(x)
    if n == 0:
        return float("nan")
    w = (lambda i, j: (i - j) ** 2 / (k - 1) ** 2) if weighted else (lambda i, j: float(i != j))
    px, py = Counter(x), Counter(y)
    observed = sum(w(a, b) for a, b in zip(x, y)) / n
    expected = sum(px[i] * py[j] * w(i, j) for i in range(k) for j in range(k)) / n ** 2
    return 1.0 if expected == 0 else 1 - observed / expected


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="fairness_log.jsonl files, one per rater")
    ap.add_argument("--threshold", type=int, default=2)
    a = ap.parse_args()

    raters = {}
    for path in a.logs:
        rows = [r for r in read_jsonl(path) if r.get("status") in ("ok", "flagged")]
        name = rows[0].get("rater", Path(path).stem) if rows else Path(path).stem
        raters[name] = {r["instance_id"]: r for r in rows}
        flagged = sum(r["status"] == "flagged" for r in rows)
        print(f"{name}: {len(rows)} rated, {flagged} flagged ({100 * flagged / max(len(rows), 1):.0f}%)")

    for (na, ra), (nb, rb) in itertools.combinations(raters.items(), 2):
        common = sorted(set(ra) & set(rb))
        print(f"\n{na} vs {nb}: {len(common)} tasks rated by both")
        for scale in ("clarity", "test_fit"):
            x, y = [ra[t][scale] for t in common], [rb[t][scale] for t in common]
            exact = sum(i == j for i, j in zip(x, y)) / max(len(x), 1)
            print(f"  {scale:9s} exact {exact:.0%}  kappa {kappa(x, y):.2f}  weighted kappa {kappa(x, y, True):.2f}")
        fa = [ra[t]["status"] == "flagged" for t in common]
        fb = [rb[t]["status"] == "flagged" for t in common]
        print(f"  flag      agree {sum(i == j for i, j in zip(fa, fb)) / max(len(fa), 1):.0%}  "
              f"kappa {kappa([int(v) for v in fa], [int(v) for v in fb], k=2):.2f}  "
              f"flagged by both {sum(i and j for i, j in zip(fa, fb))}, by either {sum(i or j for i, j in zip(fa, fb))}")
        diff = [t for t, i, j in zip(common, fa, fb) if i != j]
        for t in diff[:15]:
            print(f"    {t}: {na} {ra[t]['clarity']}/{ra[t]['test_fit']}, {nb} {rb[t]['clarity']}/{rb[t]['test_fit']}")
