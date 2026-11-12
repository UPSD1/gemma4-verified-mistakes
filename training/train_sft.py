"""LoRA SFT for arms (a), (b) and (c) on Gemma 4 12B (tracker rows 14-15).

Follows the Plan's training settings:
  - bf16 QAT weights (models/gemma-4-12b-it-qat-unquantized); the adapter is then served on the
    competition's 4-bit model by vLLM (--enable-lora, max rank 128);
  - LoRA rank 16-32 on attention and MLP projections, low learning rate, 1-2 epochs;
  - Gemma's thinking text kept: assistant `reasoning` renders as the thought channel;
  - loss on the agent's own tokens only: every message carries a `loss` flag (training/build_data.py).
    System prompt, bug report and tool outputs are False. Arm (c) also sets the mistaken steps False,
    so the same script trains (a), (b) and (c);
  - equal budget across arms: --max-trained-tokens caps the agent tokens seen per epoch (pass the
    smallest arm's total so every arm sees the same amount, +-5%).

The loss is computed only at trained positions, in chunks (Gemma's 262k vocabulary makes full-sequence
logits too large at 27k tokens), with Gemma's final logit soft-capping applied.

    PYTHONPATH=/workspace/train_pkgs python training/train_sft.py --data data/training/arm_a_sft.jsonl \\
        --out /workspace/adapters/arm_a --rank 32 --epochs 1
    # smoke test on CPU: --device cpu --limit 1 --max-len 512 --steps 1
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

MODEL = "/workspace/models/gemma-4-12b-it-qat-unquantized"
# The agent's 10 tool declarations, captured from a real request (training/capture_request.py), so the
# rendered system turn matches what the model sees at inference.
TOOLS = json.loads((Path(__file__).resolve().parent / "agent_tools.json").read_text())
# The agent's request sets enable_thinking and not preserve_thinking; render training text the same way.
TEMPLATE_KW = dict(add_generation_prompt=False, enable_thinking=True)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def to_template(messages):
    """Our message format -> the chat template's: tool-call arguments as objects, no training flags."""
    out = []
    for m in messages:
        t = {k: v for k, v in m.items() if k not in ("loss",)}
        for c in t.get("tool_calls") or []:
            args = c["function"].get("arguments")
            if isinstance(args, str):
                try:
                    c["function"]["arguments"] = json.loads(args)
                except ValueError:
                    c["function"]["arguments"] = {"raw": args}
        if not t.get("tool_calls"):
            t.pop("tool_calls", None)
        out.append(t)
    return out


def _lcp(a, b):
    """Length of the longest common prefix (binary search on C-level slice comparisons)."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def render(tok, msgs):
    return tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, **TEMPLATE_KW)


def encode(tok, messages, max_len):
    """Token ids and labels (-100 = no loss) for one example.

    The example is rendered once in its final form, which is what the model sees at inference (in Gemma's
    template the agent loop is one model turn; a turn's visible text is placed after its tool result, and
    a later user message drops earlier thoughts). Each character of that final text is owned by the message
    whose addition first produced it in its final place (the common prefix between rendering messages
    [0..i] and the final text grows when message i is added). Characters owned by a message with
    loss=True are trained; everything else (system, user, tool results, text a later message moved) is
    masked. A trained message can therefore never be given tool output or prompt text."""
    msgs = to_template(json.loads(json.dumps(messages)))
    # Gemma's template hides the reasoning of every assistant turn that a later user message follows.
    # Drop it here too, so the prefix renderings below agree with the final one: otherwise such a turn's
    # text first appears in its final place only when the user message is added, and its tool call is
    # silently owned by that (masked) message (audit of Nov 12). training/fix_examples.py keeps trained
    # turns after the last user message, so this never removes a trained thought.
    last_user = max((i for i, m in enumerate(msgs) if m["role"] == "user"), default=-1)
    for i, m in enumerate(msgs[:last_user]):
        if m["role"] == "assistant":
            m.pop("reasoning", None)
    final = render(tok, msgs)
    trained = bytearray(len(final))
    settled = 0
    for i, m in enumerate(messages):
        upto = _lcp(render(tok, msgs[: i + 1]), final)
        if upto > settled:
            if m.get("loss"):
                trained[settled:upto] = b"\x01" * (upto - settled)
            settled = upto
    enc = tok(final, add_special_tokens=False, return_offsets_mapping=True)
    ids, labels = enc["input_ids"], []
    for tid, (s0, e0) in zip(ids, enc["offset_mapping"]):
        # A token belongs to whoever produced its LAST character: an ownership boundary can fall inside a
        # special token's text (renderings may share a "<" before diverging), never past its end.
        labels.append(tid if e0 > s0 and trained[e0 - 1] else -100)
    return ids[:max_len], labels[:max_len]


def load_model(path, device, rank, alpha, dropout, init_adapter=None):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=None)
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    # Language model only: target the projections inside the text decoder layers.
    target = r".*language_model.*\.(" + "|".join(TARGETS) + r")$"
    if not any("language_model" in n for n, _ in model.named_modules()):
        target = r".*layers.*\.(" + "|".join(TARGETS) + r")$"
    if init_adapter:  # continue training an existing adapter (a further pass over the same data)
        return PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=target, task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def lm_head_and_cap(model):
    base = model.get_base_model()
    head = base.get_output_embeddings()
    cfg = getattr(base.config, "text_config", base.config)
    return head, getattr(cfg, "final_logit_softcapping", None)


def final_hidden(model, x):
    """Last hidden state (after the final norm) without keeping every layer's hidden state."""
    base = model.get_base_model()
    inner = getattr(base, "model", None)
    if inner is not None:
        out = inner(input_ids=x, use_cache=False, return_dict=True)
        return out.last_hidden_state[0]
    return model(input_ids=x, output_hidden_states=True, return_dict=True, logits_to_keep=1).hidden_states[-1][0]


def _chunk_loss(h, y, head_w, cap):
    logits = torch.nn.functional.linear(h, head_w).float()
    if cap:
        logits = cap * torch.tanh(logits / cap)
    return torch.nn.functional.cross_entropy(logits, y, reduction="sum")


def loss_on_trained(model, ids, labels, device, chunk=1024):
    """Cross-entropy at trained positions only (next-token). Each chunk's logits over the 262k vocabulary
    are recomputed in the backward pass (checkpointed) instead of stored, which keeps memory flat."""
    from torch.utils.checkpoint import checkpoint
    x = torch.tensor([ids], device=device)
    y = torch.tensor(labels[1:] + [-100], device=device)
    h = final_hidden(model, x)
    pos = (y != -100).nonzero(as_tuple=True)[0]
    head, cap = lm_head_and_cap(model)
    total, n = h.new_zeros((), dtype=torch.float32), len(pos)
    for s in range(0, n, chunk):
        p = pos[s:s + chunk]
        total = total + checkpoint(_chunk_loss, h[p], y[p], head.weight, cap, use_reentrant=False)
    return total / max(n, 1), n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=28000)
    ap.add_argument("--max-trained-tokens", type=int, help="cap on agent tokens per epoch (equal budget across arms)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--steps", type=int, help="stop after this many optimizer steps (smoke tests)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-adapter", help="start from this adapter instead of a fresh LoRA (a further pass)")
    ap.add_argument("--order-offset", type=int, default=0,
                    help="example order of pass k uses seed + offset + k; 1 gives the second pass the order a "
                         "two-epoch run would have used")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    random.Random(a.seed).shuffle(rows)
    # Rows marked "priority" (recovery splices) are taken first, then the budget is filled with the rest,
    # so arms (b)/(c) keep every splice and all arms see the same number of trained tokens.
    rows = [r for r in rows if r.get("priority")] + [r for r in rows if not r.get("priority")]
    rows = rows[: a.limit] if a.limit else rows

    t0 = time.time()
    data, trained = [], 0
    for r in rows:
        ids, labels = encode(tok, r["messages"], a.max_len)
        n = sum(l != -100 for l in labels)
        if n == 0:
            continue
        if a.max_trained_tokens and trained + n > a.max_trained_tokens:
            continue
        data.append((ids, labels))
        trained += n
    random.Random(a.seed + 1).shuffle(data)  # training order: splices and clean examples mixed
    print(f"{len(data)} examples, {trained:,} trained tokens, {sum(len(i) for i, _ in data):,} tokens in all "
          f"(encoded in {time.time() - t0:.0f}s)", flush=True)

    model = load_model(a.model, a.device, a.rank, a.alpha, a.dropout, a.init_adapter)
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total_steps = max(1, math.ceil(len(data) * a.epochs / a.grad_accum))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, total_steps // 20)) * 0.5 * (1 + math.cos(math.pi * s / total_steps)))
    model.train()
    order = [i for _ in range(math.ceil(a.epochs))
             for i in random.Random(a.seed + a.order_offset + _).sample(range(len(data)), len(data))]
    order = order[: int(len(data) * a.epochs)]
    step, acc, t0 = 0, [], time.time()
    for k, i in enumerate(order):
        ids, labels = data[i]
        loss, n = loss_on_trained(model, ids, labels, a.device)
        (loss / a.grad_accum).backward()
        acc.append(loss.item())
        if (k + 1) % a.grad_accum == 0 or k + 1 == len(order):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
            step += 1
            print(f"step {step}/{total_steps} loss {sum(acc) / len(acc):.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{time.time() - t0:.0f}s", flush=True)
            acc = []
            if a.steps and step >= a.steps:
                break
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    (Path(a.out) / "train_args.json").write_text(json.dumps({**vars(a), "examples": len(data),
                                                             "trained_tokens": trained}, indent=2))
    print(f"adapter -> {a.out}", flush=True)
