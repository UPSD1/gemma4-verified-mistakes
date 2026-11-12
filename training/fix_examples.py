"""Rewrite built SFT examples so every trained token was produced in exactly the context it is shown in
(audit of Nov 12). Applies to arm_a_sft.jsonl, arm_b_sft.jsonl and arm_c_sft.jsonl as built by
build_data.py / build_splices.py; writes *_v3.jsonl next to them.

Fixes:
  1. Recovery splices (rows with "priority"): the resampled continuation was started with the harness's
     continuation nudge (training/resample.py NUDGE) after the recorded prefix, and that user message hides
     the prefix's thoughts. The nudge is inserted right after the failing step's tool result. The failing
     step is found as the one assistant turn that arm (c) masks and arm (b) trains (--pair gives arm (b)
     for arm (c) and vice versa).
  2. Turns before a mid-run user message (a nudge): Gemma's template hides an assistant turn's reasoning
     once a later user message follows, so the model never saw those thoughts afterwards, but it did
     produce them. An example with trained turns before a later user message is split in two:
       - the part up to that user message, trained as produced (thoughts visible: no later user yet);
       - the whole example with those earlier turns masked (their thoughts hidden by the template).
     train_sft.encode also drops the reasoning of turns before the last user message, as the template
     does, so loss ownership is computed on the text the model really saw.
  3. Lost sub-agent results: build_data turned each code_analyzer_agent result into an empty user message
     (the result itself is missing from the trace step it is read from). Such examples are dropped.
  4. Malformed tool calls written as text ("...text <|channel>call:run_command{...}"): the template strips
     everything from <|channel> on, so the model saw only the leading text. The turn is cut there and
     masked, so the adapter does not learn the broken call.

    python3 training/fix_examples.py --data arm_b_sft.jsonl --pair arm_c_sft.jsonl
    python3 training/fix_examples.py --data arm_c_sft.jsonl --pair arm_b_sft.jsonl
    python3 training/fix_examples.py --data arm_a_sft.jsonl
"""
import argparse
import collections
import json
import re
from pathlib import Path

NUDGE = ("Please continue your work using the available tools, or call submit_patch when you have "
         "completed and verified your changes.")  # training/resample.py, verbatim
BROKEN_CALL = re.compile(r"(<\|channel>\s*)*call:[A-Za-z_]+\{")


def key(r):
    return (r["instance_id"], json.dumps(r.get("branch")), r.get("segment"), r.get("kind"))


def fix_broken_calls(msgs, stats):
    for m in msgs:
        if m["role"] != "assistant" or m.get("tool_calls"):
            continue
        hit = BROKEN_CALL.search(m.get("content") or "")
        if hit:
            m["content"] = m["content"][: hit.start()].rstrip()
            m["loss"] = False
            stats["broken_call_turns_masked"] += 1


def insert_nudge(msgs, other, stats):
    """Insert the resampler's nudge after the failing step's tool result. The failing step is the one
    assistant turn whose loss flag differs between arm (b) and arm (c)."""
    diff = [i for i, (x, y) in enumerate(zip(msgs, other))
            if x["role"] == "assistant" and bool(x.get("loss")) != bool(y.get("loss"))]
    if len(diff) != 1:
        stats[f"splice_boundary_not_found_{len(diff)}"] += 1
        return msgs
    i = diff[0] + 1
    while i < len(msgs) and msgs[i]["role"] == "tool":
        i += 1
    stats["nudge_inserted"] += 1
    return msgs[:i] + [{"role": "user", "content": NUDGE, "loss": False}] + msgs[i:]


def split_at_users(msgs, stats):
    """Examples in which no trained assistant turn is followed by a user message (see fix 2)."""
    out, cur = [], [dict(m) for m in msgs]
    first_user = next((i for i, m in enumerate(cur) if m["role"] == "user"), 0)
    while True:
        users = [i for i, m in enumerate(cur) if m["role"] == "user" and i > first_user]
        bad = next((u for u in users if any(m["role"] == "assistant" and m.get("loss") for m in cur[:u])), None)
        if bad is None:
            out.append(cur)
            return out
        out.append([dict(m) for m in cur[:bad]])  # trained as produced: no later user message
        stats["split_at_user"] += 1
        for m in cur[:bad]:
            if m["role"] == "assistant":
                m["loss"] = False
                m.pop("reasoning", None)  # hidden by the template once the user message follows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pair", help="the other recovery arm's file (b for c, c for b), to locate the failing step")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    other = {}
    if a.pair:
        other = {key(r): r["messages"] for r in (json.loads(l) for l in open(a.pair) if l.strip())}
    stats = collections.Counter(rows_in=len(rows))
    out = []
    for r in rows:
        msgs = [dict(m) for m in r["messages"]]
        if any(m["role"] == "user" and not (m.get("content") or "").strip() for m in msgs):
            stats["dropped_lost_subagent_result"] += 1
            continue
        if r.get("priority"):
            if key(r) not in other:
                stats["splice_without_pair"] += 1
            else:
                msgs = insert_nudge(msgs, other[key(r)], stats)
        fix_broken_calls(msgs, stats)
        parts = split_at_users(msgs, stats)
        for j, p in enumerate(parts):
            if not any(m["role"] == "assistant" and m.get("loss") for m in p):
                stats["dropped_part_without_trained_turn"] += 1
                continue
            out.append({**r, "messages": p, "part": j})
    dst = Path(a.out or Path(a.data).with_name(Path(a.data).stem + "_v3.jsonl"))
    dst.write_text("".join(json.dumps(r) + "\n" for r in out))
    stats["rows_out"] = len(out)
    print(json.dumps(dict(stats)), "->", dst)


if __name__ == "__main__":
    main()
