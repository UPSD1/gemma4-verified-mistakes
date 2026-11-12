"""Every Section 5 number that is not in analyze_eval.py, from the released data folder (CPU only).

    python runner/paper_analyses.py --data <dataset folder> [--out paper_numbers.json]

<dataset folder> is the Kaggle dataset (or gemma4_release/): eval/, base_runs/, training_data/, training_logs/.
Prints and returns:
  table3      solved per run and the mean, and each agent vs base over both v3 runs (paired bootstrap over tasks)
  churn       tasks whose outcome changes between two base runs, and between each agent and base
  table5      reasoning characters per call and seconds per call (median, quartiles), compactions per run
  curves      tasks solved by each minute of the run (agent clock from the trace), Figure 4
  never       tasks the base model never solved in its five OOD-45 runs, and who solved them
  train_data  reasoning characters per trained step in the training examples (clean vs splices)
  train_loss  mean loss of the first and last 10 steps of every SFT run (each batch is unseen when scored)
  training_pool  the solved FreshSWE-Py base runs we trained on: submitted or cut off at the deadline, nudges
  behaviour   per agent and seed: submits (and how many passed), deadline solves, nudges and how many were followed
              by another tool call, share of calls at the 4,096-token thinking cap, repeated lines in those capped
              thoughts, generation speed
  early_samples  cap share and submits on five random 20-task samples
"""
import argparse
import glob
import itertools
import json
import random
import re
import statistics as st
from pathlib import Path

MAIN = "swe_baseline_agent"
V3 = {"base": "base", "prompt": "prompt_recovery", "a": "arm_a_v3", "b": "arm_b_v3", "c": "arm_c_v3",
      "d": "arm_d17_v3", "d_ablation": "arm_d23_v3"}
V2 = {"a": "arm_a_v1", "b": "arm_b_v1", "c": "arm_c_v1", "d": "arm_d_v1"}
ROUND2 = {"b_second_pass": "arm_b_v3_ep2", "submit_only": "arm_a_submit_only_v3"}


def rows_of(d):
    return [json.loads(l) for f in sorted(glob.glob(f"{d}/*/task_results.jsonl")) for l in open(f)]


def solved(d):
    return {r["instance_id"]: bool(r["resolved"]) for r in rows_of(d)}


def runs(data):
    """agent -> {seed: run folder}. The v3 agents ran with seed 2 (eval_v3) and seed 1 (eval_v3_seed1), each seed on its
    own machine; the v2 build once with seed 2; the round-2 agents (second pass of (b), submit-only clean arm) with both
    seeds on the same two machines (eval_round2). Comparisons with base always pair runs of the same seed."""
    out = {}
    for k, v in V3.items():
        out[k] = {s: p for s, p in (("2", f"{data}/eval/eval_v3/{v}/s2"), ("1", f"{data}/eval/eval_v3_seed1/{v}/s1"))
                  if Path(p).is_dir()}
    for k, v in V2.items():
        out[f"{k}_v2"] = {"2": f"{data}/eval/eval_v2_e2/{v}/s2"}
    for k, v in ROUND2.items():
        found = {s: f"{data}/eval/eval_round2/{v}/s{s}" for s in ("2", "1")
                 if glob.glob(f"{data}/eval/eval_round2/{v}/s{s}/*/task_results.jsonl")}
        if found:
            out[k] = found
    return out


def main_run(ps):
    """The run used for one-run measures: seed 2 when there is one (Tables 1-2 and Figure 2 use it)."""
    return ps.get("2") or next(iter(ps.values()))


def boot(d, n=10000, seed=0):
    rng = random.Random(seed)
    b = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return [round(b[int(0.025 * n)], 3), round(b[int(0.975 * n)], 3)]


def table3(R):
    base = {s: solved(p) for s, p in R["base"].items()}
    out = {}
    for k, ps in R.items():
        res = {s: solved(p) for s, p in ps.items()}
        row = {"solved": {s: sum(r.values()) for s, r in res.items()},
               "mean": round(sum(sum(r.values()) for r in res.values()) / len(res), 1),
               "timeouts": {s: sum("timeout" in (r.get("error") or "") for r in rows_of(p)) for s, p in ps.items()}}
        seeds = [s for s in res if s in base]
        if k != "base" and seeds:
            tasks = sorted(base[seeds[0]])
            d = [sum(int(res[s][t]) - int(base[s][t]) for s in seeds) / len(seeds) for t in tasks]
            row["vs_base_seeds"] = seeds
            row["vs_base"] = round(sum(d) / len(d), 3)
            row["ci95"] = boot(d)
        out[k] = row
    return out


def churn(R, data):
    base_by_seed = {s: solved(p) for s, p in R["base"].items()}
    base = list(base_by_seed.values()) + [solved(f"{data}/base_runs/ood45/{s}") for s in ("s1", "s2", "s3")]
    tasks = sorted(base[0])
    flips = lambda a, b: (sum(a[t] and not b[t] for t in tasks), sum(b[t] and not a[t] for t in tasks))
    out = {"base_vs_base": [sum(flips(a, b)) for a, b in itertools.combinations(base, 2)]}
    for k, ps in R.items():
        if k != "base":
            s0 = "2" if "2" in ps else next(iter(ps))
            g, l = flips(solved(ps[s0]), base_by_seed[s0])
            out[k] = {"gained": g, "lost": l}
    never = [t for t in tasks if not any(b.get(t) for b in base)]
    solvers = {t: [k for k, ps in R.items() if k not in ("base",) for p in ps.values() if solved(p).get(t)] for t in never}
    return out, {"base_runs": len(base), "per_run_solved": [sum(b.values()) for b in base], "never": solvers}


def per_call(d):
    think, comp, n = [], 0, 0
    for f in glob.glob(f"{d}/*/traces/trace_*.json"):
        steps = [s for s in json.load(open(f))["steps"]
                 if s.get("source") == "agent" and (s.get("extra") or {}).get("author", MAIN) == MAIN]
        cur, prev, n = None, None, n + 1
        for s in steps:
            m = s.get("metrics") or {}
            if m:
                if cur is not None:
                    think.append(cur)
                cur = 0
                pt = m.get("prompt_tokens")
                if pt and prev and pt < 0.75 * prev:  # prompt shrank by a quarter: the harness compacted
                    comp += 1
                prev = pt or prev
            if cur is not None and (s.get("extra") or {}).get("event_type") == "thinking":
                cur += len(s.get("message") or "")
        if cur is not None:
            think.append(cur)
    rows = rows_of(d)
    spc = [r["duration_seconds"] / r["total_llm_calls"] for r in rows if r.get("total_llm_calls")]
    q = lambda xs: [round(x, 1) for x in st.quantiles(xs, n=4)]
    return {"reasoning_chars_q1_med_q3": q(think), "seconds_per_call_q1_med_q3": q(spc),
            "calls_per_run_median": st.median(r.get("total_llm_calls") or 0 for r in rows),
            "compactions_per_run": round(comp / max(1, n), 2)}


def solve_minutes(d):
    """Agent-clock minute at which each solved run ended (largest elapsed_s in its trace)."""
    out = []
    for r in rows_of(d):
        if not r["resolved"]:
            continue
        tr = glob.glob(f"{d}/*/traces/trace_{r['instance_id']}.json")
        if tr:
            m = 0
            for s in json.load(open(tr[0]))["steps"]:
                for x in [s.get("extra") or {}] + [c.get("extra") or {} for c in s.get("tool_calls") or []]:
                    m = max(m, x.get("elapsed_s") or 0)
            out.append(m / 60)
    return out


def curves(R, minutes=16):
    out = {}
    for k, ps in R.items():
        mins = solve_minutes(main_run(ps))
        out[k] = [sum(m <= t for m in mins) for t in range(minutes)] + [len(mins)]
    return out


CAP = 4090  # the server caps a thought at 4,096 tokens; a call this long hit the cap


def run_behaviour(trace, row=None):
    """How one run spent its time: submitted?, nudges and what followed, calls at the thinking cap, repeated lines
    in those capped thoughts, generation speed (completion tokens over the time between a step and the one before)."""
    st = json.load(open(trace))["steps"]
    out = {"submitted": False, "nudges": 0, "nudge_continued": 0, "calls": 0, "cap": 0, "cap_lines": 0,
           "cap_repeat_lines": 0, "gen_s": 0.0, "gen_tok": 0}
    prev_ts, think, after_nudge = None, [], False
    for s in st:
        e = s.get("extra") or {}
        calls = s.get("tool_calls") or []
        if any(c.get("function_name") == "submit_patch" for c in calls):
            out["submitted"] = True
        if e.get("event_type") == "continuation_nudge":
            out["nudges"] += 1
            after_nudge = True
            continue
        if after_nudge and s.get("source") == "agent" and (calls or e.get("event_type") == "final"):
            out["nudge_continued"] += any(c.get("function_name") != "submit_patch" for c in calls)
            after_nudge = False
        if s.get("source") == "agent" and e.get("author", MAIN) == MAIN:
            if e.get("event_type") == "thinking":
                think.append(s.get("message") or "")
            m = s.get("metrics") or {}
            if m:
                tok = m.get("completion_tokens") or 0
                out["calls"] += 1
                if tok >= CAP:
                    out["cap"] += 1
                    lines = [l.strip() for l in "\n".join(think).splitlines() if l.strip()]
                    out["cap_lines"] += len(lines)
                    out["cap_repeat_lines"] += len(lines) - len(set(lines))
                think = []
                ts = e.get("timestamp") or next(((c.get("extra") or {}).get("timestamp") for c in calls), None)
                if ts and prev_ts and 0 < ts - prev_ts < 600:
                    out["gen_s"] += ts - prev_ts
                    out["gen_tok"] += tok
        ts = e.get("timestamp") or next(((c.get("extra") or {}).get("timestamp") for c in calls), None)
        prev_ts = ts or prev_ts
    if row is not None:
        out["solved"] = bool(row["resolved"])
        out["deadline_solve"] = out["solved"] and not out["submitted"] and row["duration_seconds"] >= 14.5 * 60
    return out


def behaviour(d, tasks=None):
    """Sum of run_behaviour over an agent's runs (optionally only `tasks`)."""
    tot = {}
    for f in glob.glob(f"{d}/**/task_results.jsonl", recursive=True):
        for l in open(f):
            r = json.loads(l)
            tr = Path(f).parent / "traces" / f"trace_{r['instance_id']}.json"
            if (tasks is None or r["instance_id"] in tasks) and tr.exists():
                for k, v in run_behaviour(tr, r).items():
                    tot[k] = tot.get(k, 0) + v
    out = {"submits": tot.get("submitted", 0),
           "submits_passed": None,
           "deadline_solves": tot.get("deadline_solve", 0),
           "nudges": tot.get("nudges", 0), "nudge_continued": tot.get("nudge_continued", 0),
           "cap_share_pct": round(100 * tot.get("cap", 0) / max(1, tot.get("calls", 0)), 1),
           "cap_repeat_line_pct": round(100 * tot.get("cap_repeat_lines", 0) / max(1, tot.get("cap_lines", 0)), 1),
           "tokens_per_s": round(tot.get("gen_tok", 0) / max(1e-9, tot.get("gen_s", 0)), 1)}
    passed = 0
    for f in glob.glob(f"{d}/**/task_results.jsonl", recursive=True):
        for l in open(f):
            r = json.loads(l)
            tr = Path(f).parent / "traces" / f"trace_{r['instance_id']}.json"
            if r["resolved"] and tr.exists() and (tasks is None or r["instance_id"] in tasks):
                passed += any(c.get("function_name") == "submit_patch"
                              for s in json.load(open(tr))["steps"] for c in s.get("tool_calls") or [])
    out["submits_passed"] = passed
    return out


def training_pool(data):
    """The solved base runs we trained on (FreshSWE-Py, 4 runs): how they ended, nudges, thinking-cap share."""
    tot = {"solved": 0, "submitted": 0, "deadline": 0, "nudges": 0, "nudge_continued": 0, "calls": 0, "cap": 0}
    for f in glob.glob(f"{data}/base_runs/freshswe/*/fresh_base/r*/task_results.jsonl"):
        for l in open(f):
            r = json.loads(l)
            tr = Path(f).parent / "traces" / f"trace_{r['instance_id']}.json"
            if not (r["resolved"] and tr.exists()):
                continue
            b = run_behaviour(tr, r)
            tot["solved"] += 1
            tot["submitted"] += b["submitted"]
            tot["deadline"] += b["deadline_solve"]
            for k in ("nudges", "nudge_continued", "calls", "cap"):
                tot[k] += b[k]
    tot["cap_share_pct"] = round(100 * tot.pop("cap") / max(1, tot.pop("calls")), 1)
    return tot


def early_samples(R, n=5, size=20, seed=0):
    """Cap share and submit rate on random 20-task samples of the seed-2 run, against base on the same tasks."""
    tasks = sorted(solved(R["base"]["2"]))
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        pick = set(rng.sample(tasks, size))
        row = {}
        for k, ps in R.items():
            if not k.endswith("_v2") and "2" in ps:
                b = behaviour(ps["2"], pick)
                row[k] = {"cap_share_pct": b["cap_share_pct"], "submits": b["submits"]}
        out.append(row)
    return out


def train_data(data):
    out = {}
    for f in sorted(glob.glob(f"{data}/training_data/v*/arm_[abc]_sft*.jsonl")):
        v = [len(x.get("reasoning") or "") for l in open(f) for x in json.loads(l)["messages"]
             if x["role"] == "assistant" and x.get("loss", True)]
        out[str(Path(f).relative_to(data))] = {"trained_steps": len(v), "q1_med_q3": [round(x) for x in st.quantiles(v, n=4)]}
    return out


def train_loss(data):
    out = {}
    for f in sorted(glob.glob(f"{data}/training_logs/v*/**/train_arm_[abc]*.log", recursive=True)):
        l = [float(x) for x in re.findall(r"step \d+/\d+ loss ([0-9.]+)", open(f).read())]
        if l:
            out[Path(f).name] = {"steps": len(l), "first10": round(sum(l[:10]) / 10, 3), "last10": round(sum(l[-10:]) / 10, 3)}
    return out


def main(data, out=None):
    R = runs(data)
    res = {"table3": table3(R)}
    res["churn"], res["never"] = churn(R, data)
    res["table5"] = {k: per_call(main_run(ps)) for k, ps in R.items()}
    res["curves"] = curves(R)
    res["train_data"] = train_data(data)
    res["train_loss"] = train_loss(data)
    res["training_pool"] = training_pool(data)
    res["behaviour"] = {f"{k} seed{s}": behaviour(p) for k, ps in R.items() for s, p in ps.items()}
    res["early_samples"] = early_samples(R)
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = main(a.data, a.out)
    for k, v in res.items():
        print(f"== {k}")
        for kk, vv in (v.items() if isinstance(v, dict) else [(None, v)]):
            print(f"  {kk}: {json.dumps(vv)}")
