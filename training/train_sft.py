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


def encode(tok, messages, max_len):
    """Token ids and labels (-100 = no loss). Each message's tokens are the difference between the
    rendering up to and including it and the rendering before it; a message is trained if loss=True."""
    msgs = to_template(json.loads(json.dumps(messages)))
    kw = dict(tokenize=False, add_generation_prompt=False, enable_thinking=True, preserve_thinking=True)
    ids, labels, prev = [], [], ""
    for i, m in enumerate(messages):
        text = tok.apply_chat_template(msgs[: i + 1], **kw)
        if not text.startswith(prev):  # template rewrote earlier text: fall back to whole-render
            prev = ""
            ids, labels = [], []
            piece = text
        else:
            piece = text[len(prev):]
        p_ids = tok(piece, add_special_tokens=not ids)["input_ids"]
        ids += p_ids
        labels += p_ids if m.get("loss") else [-100] * len(p_ids)
        prev = text
    return ids[:max_len], labels[:max_len]


def load_model(path, device, rank, alpha, dropout):
    from peft import LoraConfig, get_peft_model
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
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=target, task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def lm_head_and_cap(model):
    base = model.get_base_model()
    head = base.get_output_embeddings()
    cfg = getattr(base.config, "text_config", base.config)
    return head, getattr(cfg, "final_logit_softcapping", None)


def loss_on_trained(model, ids, labels, device, chunk=2048):
    """Cross-entropy at trained positions only (next-token), in chunks over those positions."""
    x = torch.tensor([ids], device=device)
    y = torch.tensor(labels[1:] + [-100], device=device)
    out = model(input_ids=x, output_hidden_states=True, return_dict=True, logits_to_keep=1)
    h = out.hidden_states[-1][0]
    pos = (y != -100).nonzero(as_tuple=True)[0]
    head, cap = lm_head_and_cap(model)
    total, n = 0.0, len(pos)
    for s in range(0, n, chunk):
        p = pos[s:s + chunk]
        logits = head(h[p]).float()
        if cap:
            logits = cap * torch.tanh(logits / cap)
        total = total + torch.nn.functional.cross_entropy(logits, y[p], reduction="sum")
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
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    random.Random(a.seed).shuffle(rows)
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
    print(f"{len(data)} examples, {trained:,} trained tokens, {sum(len(i) for i, _ in data):,} tokens in all "
          f"(encoded in {time.time() - t0:.0f}s)", flush=True)

    model = load_model(a.model, a.device, a.rank, a.alpha, a.dropout)
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total_steps = max(1, math.ceil(len(data) * a.epochs / a.grad_accum))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, total_steps // 20)) * 0.5 * (1 + math.cos(math.pi * s / total_steps)))
    model.train()
    order = [i for _ in range(math.ceil(a.epochs)) for i in random.Random(a.seed + _).sample(range(len(data)), len(data))]
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
