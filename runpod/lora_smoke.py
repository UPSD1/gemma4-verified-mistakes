"""Smoke test: does a trained LoRA arm load and act on the competition's 4-bit model in vLLM?

Run on a pod where vLLM serves the base model with the adapter mounted, e.g.
    serve.sh --lora-modules arm_a=/path/to/adapters/arm_a
then
    python runpod/lora_smoke.py --data arm_a_sample.jsonl --adapter arm_a \
        --tokenizer-dir /workspace/models/kaggle-12b \
        --train-template /workspace/models/gemma-4-12b-it-qat-unquantized/chat_template.jinja

Checks (each prints PASS/FAIL):
  1. served:     the adapter is listed by /v1/models next to the base model.
  2. applied:    on held-in training examples, the adapter's mean log-probability of the agent's own
                 tokens is higher than the base model's (an adapter that silently fails to load gives
                 identical numbers; one whose weights do not match the quantized layers gives worse ones).
  3. generation: from real mid-trajectory prompts, the adapter returns parseable tool calls with no
                 leaked Gemma control tokens, at least as often as the base model.
  4. template:   how many training examples render to different tokens under the serving chat template
                 (the model dir vLLM uses) than under the template training used. Informational.
"""
import argparse
import json
import re
import statistics
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from train_sft import TEMPLATE_KW, TOOLS, encode, to_template  # noqa: E402

CONTROL = re.compile(r"<\|channel>|<channel\|>|<\|turn>|<turn\|>|<\|tool_call>|<tool_call\|>")


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def mean_logp(base_url, model, ids, labels):
    """Mean log-prob of the trained (label != -100) tokens, from vLLM prompt_logprobs."""
    out = post(f"{base_url}/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1,
                                             "temperature": 0, "prompt_logprobs": 0})
    plp = out["choices"][0]["prompt_logprobs"]  # one entry per prompt token; first is None
    vals = []
    for i, lab in enumerate(labels):
        if lab == -100 or i == 0 or plp[i] is None:
            continue
        entry = plp[i].get(str(ids[i]))
        if entry is None:
            raise SystemExit(f"prompt_logprobs missing the actual token at position {i}")
        vals.append(entry["logprob"])
    return sum(vals) / len(vals), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--base", default="gemma-4-12b-it-qat-w4a16-ct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True, help="SFT jsonl rows with messages + loss flags")
    ap.add_argument("--tokenizer-dir", required=True, help="serving model dir (tokenizer + chat template)")
    ap.add_argument("--train-template", help="chat_template.jinja used in training, for check 4")
    ap.add_argument("--n-logp", type=int, default=8)
    ap.add_argument("--n-gen", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4000, help="score only the first N tokens (logprob memory)")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    tools = TOOLS  # the agent's tool declarations, as in training
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    ok = True

    # 1. served
    models = [m["id"] for m in json.load(urllib.request.urlopen(f"{a.url}/v1/models"))["data"]]
    served = a.adapter in models and a.base in models
    print(f"[{'PASS' if served else 'FAIL'}] served: {models}")
    ok &= served

    # 2. applied: training-time rendering (serving tokenizer), same ids sent to both models.
    enc = []
    for r in rows:
        ids, labels = encode(tok, r["messages"], a.max_tokens)  # serving template, as vLLM will; first max_tokens
        if sum(l != -100 for l in labels) >= 20:
            enc.append((ids, labels))
    enc = enc[: a.n_logp]
    diffs = []
    for ids, labels in enc:
        b, n = mean_logp(a.url, a.base, ids, labels)
        t, _ = mean_logp(a.url, a.adapter, ids, labels)
        diffs.append(t - b)
        print(f"   {len(ids):6d} tokens, {n:5d} trained: base {b:.4f}  adapter {t:.4f}  diff {t - b:+.4f}")
    applied = bool(diffs) and statistics.mean(diffs) > 0.01 and sum(d > 0 for d in diffs) >= 0.75 * len(diffs)
    print(f"[{'PASS' if applied else 'FAIL'}] applied: mean log-prob gain {statistics.mean(diffs) if diffs else 0:+.4f} "
          f"nats/token on {len(diffs)} examples ({sum(d > 0 for d in diffs)} positive)")
    ok &= applied

    # 3. generation from the prefix before a trained assistant turn.
    stats = {a.base: [0, 0], a.adapter: [0, 0]}  # [well-formed tool call, leaked control tokens]
    used = 0
    for r in rows:
        msgs = r["messages"]
        cut = next((i for i in range(len(msgs) - 1, 0, -1) if msgs[i]["role"] == "assistant" and msgs[i].get("loss")), None)
        if cut is None:
            continue
        if len(tok.apply_chat_template(to_template(json.loads(json.dumps(msgs[:cut]))), tools=tools, tokenize=True,
                                       add_generation_prompt=True, enable_thinking=True)) > 24000:
            continue
        prompt = [{k: v for k, v in m.items() if k != "loss"} for m in msgs[:cut]]  # API form: string arguments
        for m in stats:
            out = post(f"{a.url}/v1/chat/completions", {
                "model": m, "messages": prompt, "tools": tools,
                "max_tokens": 2048, "temperature": 1.0, "top_p": 0.95, "seed": 0,
                "chat_template_kwargs": {"enable_thinking": True}, "skip_special_tokens": False})
            ch = out["choices"][0]
            msg = ch["message"]
            calls = msg.get("tool_calls") or []
            good = bool(calls) and all(_json_ok(c["function"].get("arguments")) for c in calls)
            leak = bool(CONTROL.search((msg.get("content") or ""))) or ch.get("finish_reason") == "length" \
                or (not calls and not (msg.get("content") or "").strip())
            stats[m][0] += good
            stats[m][1] += leak
        used += 1
        if used >= a.n_gen:
            break
    print(f"   from {used} prompts: well-formed tool calls base {stats[a.base][0]}, adapter {stats[a.adapter][0]}; "
          f"leaked control tokens base {stats[a.base][1]}, adapter {stats[a.adapter][1]}")
    gen = used > 0 and stats[a.adapter][0] >= stats[a.base][0] - 1 and stats[a.adapter][1] <= stats[a.base][1]
    print(f"[{'PASS' if gen else 'FAIL'}] generation")
    ok &= gen

    # 4. template mismatch (informational)
    if a.train_template:
        train_tpl = Path(a.train_template).read_text()
        kw = dict(tokenize=True, tools=tools, **TEMPLATE_KW)
        diff_n, first = 0, []
        for r in rows[:50]:
            msgs = to_template(json.loads(json.dumps(r["messages"])))
            s = tok.apply_chat_template(msgs, **kw)
            t = tok.apply_chat_template(msgs, chat_template=train_tpl, **kw)
            if s != t:
                diff_n += 1
                k = next((i for i, (x, y) in enumerate(zip(s, t)) if x != y), min(len(s), len(t)))
                first.append((len(s), len(t), k))
        print(f"[INFO] template: {diff_n}/{min(50, len(rows))} examples render differently under the serving template "
              f"(serving len, training len, first differing token): {first[:5]}")

    print("SMOKE_OK" if ok else "SMOKE_FAIL")


def _json_ok(args):
    if isinstance(args, dict):
        return True
    try:
        json.loads(args)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
