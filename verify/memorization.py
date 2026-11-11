"""Memorization check (tracker row 6c), after MLE-bench's familiarity measure.

For each task, ask the served base model (vLLM, OpenAI-compatible) for the log-probability of
every token of "<bug report>\\n\\n<gold patch>" and report the mean log-probability of the gold
patch tokens given the bug report, and of the bug report itself. High familiarity (mean log-prob
close to 0) on the patch suggests the fix was seen in pretraining. Compare it with solve rate:
if familiar tasks are solved more often, part of the score may be memory.

Runs against the model on the pod; nothing leaves the pod.

    python3 verify/memorization.py --tasks <tasks.jsonl> --set freshswe --out <dir>/familiarity_freshswe.jsonl
"""
import argparse
import json
import statistics
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAX_CHARS = 16000  # prompt log-probs cover the whole vocabulary per token: keep prompts short


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat_ids(base, model, user, reply):
    msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
    return post(f"{base}/tokenize", {"model": model, "messages": msgs, "add_generation_prompt": False,
                                     "continue_final_message": True})["tokens"]


def familiarity(base, model, task, max_chars=MAX_CHARS):
    """Mean log-probability of the gold patch as the model's reply to the bug report, under the
    chat template (raw-text log-probs from this instruction-tuned server are not meaningful).
    The first two reply tokens are skipped: they score the start of any reply, not the fix."""
    issue = (task.get("problem_statement") or "")[: max_chars // 2]
    patch = (task.get("patch") or "")[: max(max_chars - len(issue), 2000)]
    user = f"Bug report:\n{issue}\n\nWrite the fix as a unified diff."
    ids = chat_ids(base, model, user, patch)
    k = len(chat_ids(base, model, user, ""))
    out = post(f"{base}/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1,
                                         "temperature": 0, "prompt_logprobs": 0})
    vals = [next(iter(d.values()))["logprob"] if d else None for d in out["choices"][0].get("prompt_logprobs") or []]
    patch_lp = [v for v in vals[k + 2:] if v is not None]
    return {"instance_id": task["instance_id"], "repo": task.get("repo"), "patch_tokens": len(patch_lp),
            "patch_mean_logprob": statistics.mean(patch_lp) if patch_lp else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--set", required=True, help="label stored with each row, e.g. freshswe, ood45, grpc, mined, public")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="gemma-4-12b-it-qat-w4a16-ct")
    ap.add_argument("--ids", help="optional file with one instance_id per line to keep")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS,
                    help="cap on bug report + patch characters; log-probs over the whole vocabulary need "
                         "about 1 GB of GPU memory per 1,000 tokens")
    a = ap.parse_args()

    tasks = [json.loads(l) for l in open(a.tasks) if l.strip()]
    if a.ids:
        keep = set(Path(a.ids).read_text().split())
        tasks = [t for t in tasks if t["instance_id"] in keep]
    out = Path(a.out)
    # Rows that errored (e.g. the server was down) are retried.
    done = {r["instance_id"] for r in map(json.loads, open(out)) if r.get("patch_mean_logprob") is not None} \
        if out.exists() else set()
    todo = [t for t in tasks if t["instance_id"] not in done]
    print(f"{len(tasks)} tasks, {len(todo)} to score", flush=True)

    def one(t):
        try:
            return {**familiarity(a.base, a.model, t, a.max_chars), "set": a.set, "max_chars": a.max_chars}
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            return {"instance_id": t["instance_id"], "set": a.set, "error": str(e)[:300]}

    with open(out, "a") as f, ThreadPoolExecutor(a.workers) as pool:
        for r in pool.map(one, todo):
            f.write(json.dumps(r) + "\n")
            f.flush()
    latest = {}
    for r in map(json.loads, open(out)):  # keep the last row per task
        latest[r["instance_id"]] = r
    rows = list(latest.values())
    ok = [r for r in rows if r.get("patch_mean_logprob") is not None]
    print(f"{len(ok)}/{len(rows)} scored; median patch log-prob "
          f"{statistics.median(r['patch_mean_logprob'] for r in ok):.3f}" if ok else "none scored")
