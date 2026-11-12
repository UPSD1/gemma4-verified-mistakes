"""Summarize base-model runs: FreshSWE-Py passes (training-data yield) and OOD-45 seeds (baseline eval).

    python runner/summarize_runs.py --fresh data/runs/rtx/fresh_base data/runs/h100/fresh_base \
        --ood data/runs/rtx/ood45_base --ood-tasks /path/to/ood45/tasks.jsonl --out progress/base_runs_summary.json

Each results dir holds task_results.jsonl (one row per task; a rerun of a task keeps the last row).
"""
import argparse
import collections
import glob
import json
import statistics
from math import comb
from pathlib import Path


def load_rows(path):
    rows = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            rows[r["instance_id"]] = r
    return rows


def kind(r):
    e = r.get("error") or ""
    if r.get("resolved"):
        return "resolved"
    if "session timeout" in e:
        return "timeout"
    if "turns budget" in e:
        return "turn_cap"
    if "ContextWindow" in e:
        return "context_overflow"
    if e:
        return "error"
    # task_results.jsonl rows carry agent_patch_size, not the patch text.
    has_patch = (r.get("agent_patch_size") or 0) > 0 or bool((r.get("agent_patch") or "").strip())
    return "wrong_patch" if has_patch else "empty_patch"


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def summarize_passes(passes):
    """passes: {label: {iid: row}} -> per-pass and per-task stats."""
    per_pass = {}
    for label, rows in sorted(passes.items()):
        ks = collections.Counter(kind(r) for r in rows.values())
        mins = [r.get("duration_seconds", 0) / 60 for r in rows.values()]
        per_pass[label] = dict(n=len(rows), resolved=ks["resolved"], rate=round(ks["resolved"] / max(len(rows), 1), 3),
                               outcomes=dict(ks), median_min=round(statistics.median(mins), 1) if mins else None)
    tasks = collections.defaultdict(list)
    for rows in passes.values():
        for iid, r in rows.items():
            tasks[iid].append(r)
    ids = sorted(tasks)
    full = [i for i in ids if len(tasks[i]) == len(passes)]
    k_max = len(passes)
    pk = {f"pass@{k}": round(statistics.mean(pass_at_k(len(tasks[i]), sum(bool(r.get("resolved")) for r in tasks[i]), k)
                                             for i in full), 3) for k in range(1, k_max + 1)} if full else {}
    solved_counts = collections.Counter(sum(bool(r.get("resolved")) for r in tasks[i]) for i in full)
    mixed = [i for i in full if 0 < sum(bool(r.get("resolved")) for r in tasks[i]) < len(tasks[i])]
    never = [i for i in full if not any(r.get("resolved") for r in tasks[i])]
    always = [i for i in full if all(r.get("resolved") for r in tasks[i])]
    by_repo = collections.defaultdict(lambda: [0, 0])
    for i in full:
        repo = tasks[i][0].get("repo") or i.rsplit("_", 1)[0]
        by_repo[repo][0] += sum(bool(r.get("resolved")) for r in tasks[i])
        by_repo[repo][1] += len(tasks[i])
    return dict(per_pass=per_pass, tasks_with_all_passes=len(full), pass_at_k=pk,
                solved_in_k_of_n={str(k): solved_counts[k] for k in range(k_max + 1)},
                mixed_tasks=len(mixed), never_solved=len(never), always_solved=len(always),
                by_repo={k: dict(resolved=v[0], runs=v[1], rate=round(v[0] / v[1], 3)) for k, v in sorted(by_repo.items())},
                mixed_ids=mixed, never_ids=never)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", nargs="*", default=[], help="fresh_base dirs containing r1..r4")
    ap.add_argument("--ood", help="ood45_base dir containing s1..s3/<repo>/")
    ap.add_argument("--ood-tasks", help="OOD-45 tasks.jsonl (for env_overlay / interface_hint cuts)")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = {}

    passes = {}
    for d in a.fresh:
        for f in sorted(glob.glob(f"{d}/r*/task_results.jsonl")):
            passes[Path(f).parent.name] = load_rows(f)
    if passes:
        out["freshswe"] = summarize_passes(passes)

    if a.ood:
        seeds = {}
        for s in sorted(glob.glob(f"{a.ood}/s*")):
            rows = {}
            for f in glob.glob(f"{s}/*/task_results.jsonl"):
                rows.update(load_rows(f))
            if rows:
                seeds[Path(s).name] = rows
        ood = summarize_passes(seeds)
        if a.ood_tasks:
            meta = {json.loads(l)["instance_id"]: json.loads(l) for l in open(a.ood_tasks) if l.strip()}
            def rate(filt):
                c = n = 0
                for rows in seeds.values():
                    for i, r in rows.items():
                        if filt(meta.get(i, {})):
                            n += 1; c += bool(r.get("resolved"))
                return dict(resolved=c, runs=n, rate=round(c / n, 3) if n else None)
            ood["cuts"] = dict(
                all=rate(lambda m: True),
                no_env_overlay=rate(lambda m: not m.get("env_overlay")),
                no_hint_or_spec=rate(lambda m: not (m.get("interface_hint") or m.get("spec_added"))),
            )
            repo_rates = [v["rate"] for v in ood["by_repo"].values()]
            ood["cuts"]["macro_by_repo"] = round(statistics.mean(repo_rates), 3) if repo_rates else None
        out["ood45"] = ood

    text = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n")
    for name, s in out.items():
        print(f"== {name}")
        for label, p in s["per_pass"].items():
            print(f"  {label}: {p['resolved']}/{p['n']} ({p['rate']:.0%})  {p['outcomes']}  median {p['median_min']} min")
        print(f"  tasks with all runs: {s['tasks_with_all_passes']}  {s['pass_at_k']}")
        print(f"  solved k of n: {s['solved_in_k_of_n']}  mixed={s['mixed_tasks']} never={s['never_solved']} always={s['always_solved']}")
        if "cuts" in s:
            print(f"  cuts: {s['cuts']}")


if __name__ == "__main__":
    main()
