"""Fairness report: the comparisons David asked for, once the raters have run.

1. Raters: how many tasks each flags, per test set (OOD-45, gRPC) and per gRPC tag.
2. Agreement between raters: exact agreement and Cohen's kappa on each scale and on the
   flag (same numbers as verify/fairness_compare.py).
3. Results by rating (when --results is given): solve rate of each arm on tasks no rater
   flagged, one rater flagged, both flagged; and by worst score (0-1 vs 2-3). Shows
   whether gains hold on tasks both raters call fair.
4. Does a rating predict difficulty: solve rate per worst-score level, pooled over arms.

Results come from the harness: each run folder has task_results.jsonl with one row per
task ({"instance_id", "resolved", ...}). Pass one folder per arm and seed; seeds of the
same arm are pooled. Folders on the pod can be copied first with runpod/rp.sh pull.

    python3 verify/fairness_report.py data/tasks/screen/test_sets.opus.rated.jsonl \\
        data/tasks/screen/test_sets.sol.rated.jsonl
    python3 verify/fairness_report.py data/tasks/screen/test_sets.*.rated.jsonl \\
        --results base=results/base_ood45_s1 --results base=results/base_ood45_s2 \\
        --results recovery=results/rec_ood45_s1 --out data/tasks/screen/fairness_report.md
"""
import argparse
import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mining"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_jsonl  # noqa: E402
from fairness_compare import kappa  # noqa: E402


def load_rated(path):
    rows = read_jsonl(path)
    rater = next((r["meta"]["fairness"].get("rater") for r in rows if r["meta"].get("fairness")), None)
    name = rater or Path(path).name.split(".")[-3]
    return name, {r["instance_id"]: r for r in rows if r["meta"].get("fairness", {}).get("clarity") is not None}


def load_results(specs):
    """arm -> instance_id -> [resolved, ...] (one entry per seed)."""
    out = defaultdict(lambda: defaultdict(list))
    for spec in specs:
        arm, _, path = spec.partition("=")
        p = Path(path)
        f = p / "task_results.jsonl" if p.is_dir() else p
        for r in read_jsonl(f):
            out[arm][r["instance_id"]].append(bool(r.get("resolved")))
    return out


def rate(vals):
    return f"{100 * sum(vals) / len(vals):.0f}% ({sum(vals)}/{len(vals)})" if vals else "-"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rated", nargs="+", help="*.rated.jsonl from fairness_screen.py, one per rater")
    ap.add_argument("--results", action="append", default=[], help="ARM=run folder or task_results.jsonl")
    ap.add_argument("--threshold", type=int, default=2)
    ap.add_argument("--out", help="also write the report to this .md file")
    a = ap.parse_args()

    lines = []
    say = lines.append
    raters = dict(load_rated(p) for p in a.rated)
    ids = sorted(set.intersection(*(set(r) for r in raters.values())))
    any_rows = next(iter(raters.values()))
    set_of = {t: any_rows[t]["meta"].get("set", "?") for t in ids}
    tag_of = {t: any_rows[t]["meta"].get("grpc_tag") for t in ids}
    flag = {n: {t: r[t]["meta"]["fairness_flag"] for t in ids} for n, r in raters.items()}
    worst = {t: max(max(r[t]["meta"]["fairness"]["clarity"], r[t]["meta"]["fairness"]["test_fit"])
                    for r in raters.values()) for t in ids}
    n_flags = {t: sum(flag[n][t] for n in raters) for t in ids}

    say(f"# Fairness report\n\n{len(ids)} tasks rated by all {len(raters)} raters: "
        + ", ".join(f"{k} {v}" for k, v in sorted(Counter(set_of.values()).items())) + ".\n")

    say("## 1. Flagged tasks per rater\n")
    sets = sorted(set(set_of.values()))
    say("| Rater | " + " | ".join(sets) + " | All |")
    say("|---|" + "---:|" * (len(sets) + 1))
    for n in raters:
        cells = [rate([flag[n][t] for t in ids if set_of[t] == s]) for s in sets]
        say(f"| {n} | " + " | ".join(cells) + f" | {rate([flag[n][t] for t in ids])} |")
    tags = sorted({v for v in tag_of.values() if v})
    if tags:
        say("\ngRPC slice by tag (automatic; check by hand before quoting):\n")
        say("| Tag | Tasks | " + " | ".join(f"flagged by {n}" for n in raters) + " |")
        say("|---|---:|" + "---:|" * len(raters))
        for g in tags:
            tt = [t for t in ids if tag_of[t] == g]
            say(f"| {g} | {len(tt)} | " + " | ".join(rate([flag[n][t] for t in tt]) for n in raters) + " |")

    say("\n## 2. Agreement between raters\n")
    say("| Pair | Scale | Exact | Kappa | Weighted kappa |")
    say("|---|---|---:|---:|---:|")
    for (na, ra), (nb, rb) in itertools.combinations(raters.items(), 2):
        for scale in ("clarity", "test_fit"):
            x = [ra[t]["meta"]["fairness"][scale] for t in ids]
            y = [rb[t]["meta"]["fairness"][scale] for t in ids]
            exact = sum(i == j for i, j in zip(x, y)) / max(len(x), 1)
            say(f"| {na} vs {nb} | {scale} | {exact:.0%} | {kappa(x, y):.2f} | {kappa(x, y, True):.2f} |")
        fx, fy = [int(flag[na][t]) for t in ids], [int(flag[nb][t]) for t in ids]
        say(f"| {na} vs {nb} | flag | {sum(i == j for i, j in zip(fx, fy)) / max(len(fx), 1):.0%} "
            f"| {kappa(fx, fy, k=2):.2f} | - |")
    say("\nTasks flagged by: " + ", ".join(f"{k} raters {v}" for k, v in sorted(Counter(n_flags.values()).items())))

    results = load_results(a.results)
    if results:
        say("\n## 3. Solve rate by rating\n")
        groups = [("no rater flagged", lambda t: n_flags[t] == 0), ("one flagged", lambda t: n_flags[t] == 1),
                  ("all flagged", lambda t: n_flags[t] == len(raters) and len(raters) > 1),
                  ("worst score 0-1", lambda t: worst[t] < a.threshold),
                  ("worst score 2-3", lambda t: worst[t] >= a.threshold)]
        for s in sets + ["all"]:
            say(f"\n{s}:\n")
            say("| Arm | " + " | ".join(g for g, _ in groups) + " |")
            say("|---|" + "---:|" * len(groups))
            for arm, res in results.items():
                cells = []
                for _, keep in groups:
                    vals = [v for t in ids if (s == "all" or set_of[t] == s) and keep(t) for v in res.get(t, [])]
                    cells.append(rate(vals))
                say(f"| {arm} | " + " | ".join(cells) + " |")
        say("\n## 4. Does the rating predict difficulty? (all arms pooled)\n")
        say("| Worst score | Tasks | Solve rate |")
        say("|---:|---:|---:|")
        for level in range(4):
            tt = [t for t in ids if worst[t] == level]
            vals = [v for res in results.values() for t in tt for v in res.get(t, [])]
            say(f"| {level} | {len(tt)} | {rate(vals)} |")
        missing = {arm: sum(1 for t in ids if t not in res) for arm, res in results.items()}
        say("\nRated tasks with no result, per arm: " + ", ".join(f"{k} {v}" for k, v in missing.items()))
    else:
        say("\n(No --results given: sections 3-4, solve rate by rating, appear once runs exist.)")

    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text)
        print(f"-> {a.out}")
