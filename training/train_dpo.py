"""Arm (d): graded DPO + mu * cross-entropy on the chosen side, starting from adapter (b) (tracker row 16).

Data: data/training/dpo_pairs.jsonl from training/build_splices.py. Each pair is two continuations from the
SAME branch point with different harness grades (resolved > patch with failing tests > patch but hit a
budget > no patch). Both sides are chat messages with an identical prompt (the context the model had at
the branch point, cut at compaction points like the SFT data); only the continuation's agent turns carry
loss, so the comparison is purely about what the model did after the mistake.

Objective, per pair (x = average log-probability per trained token, i.e. length-averaged, so long and
short continuations are compared fairly):

    L = -log sigmoid(beta * ((x_c - ref_c) - (x_r - ref_r)))  +  mu * (-x_c)

The second term is the language-modelling loss on the chosen side the Plan asks for (as in iterative
reasoning preference optimization / RPO, which normalises it by length the same way). The reference
log-probabilities are adapter (b)'s, computed once before any update.

Memory: each side can be ~28k tokens, so the two sides are never in the graph together. A no-grad pass
gives x_c and x_r; dL/dx_c and dL/dx_r are then exact scalars, and each side is re-run with gradients and
back-propagated with its own scalar weight. Dropout is set to 0 so the two passes see the same function.

    /workspace/trainenv/bin/python training/train_dpo.py --data data/dpo_pairs.jsonl \\
        --init /workspace/adapters/arm_b --out /workspace/adapters/arm_d
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from train_sft import MODEL, encode, loss_on_trained


def avg_logp(model, ids, labels, device):
    """Average log-probability of the trained tokens (minus the SFT loss), and how many there are."""
    loss, n = loss_on_trained(model, ids, labels, device)
    return -loss, n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--init", required=True, help="adapter (b), the starting point and the DPO reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--beta", type=float, default=2.0,
                    help="DPO temperature on length-averaged log-probs (SimPO-scale; summed log-probs use ~0.1)")
    ap.add_argument("--mu", type=float, default=1.0, help="weight of the chosen side's cross-entropy (RPO uses 1)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=28000)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--steps", type=int, help="stop after this many optimizer steps (smoke tests)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l) for l in open(a.data) if l.strip()][: a.limit]

    t0, pairs, skipped = time.time(), [], 0
    for r in rows:
        c = encode(tok, r["chosen"]["messages"], a.max_len)
        j = encode(tok, r["rejected"]["messages"], a.max_len)
        nc, nr = (sum(l != -100 for l in x[1]) for x in (c, j))
        if nc == 0 or nr == 0:  # the continuation fell past --max-len
            skipped += 1
            continue
        pairs.append({"c": c, "r": j, "id": r["instance_id"], "grades": (r["chosen"]["grade"], r["rejected"]["grade"])})
    print(f"{len(pairs)} pairs ({skipped} skipped: continuation past --max-len), "
          f"{sum(len(p['c'][0]) + len(p['r'][0]) for p in pairs):,} tokens in all (encoded in {time.time() - t0:.0f}s)",
          flush=True)

    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=None)
    base.to(a.device)
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, a.init, is_trainable=True)
    for m in model.modules():  # identical function in the no-grad and the gradient pass
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    model.print_trainable_parameters()

    # Reference: adapter (b) before any update.
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for p in pairs:
            p["ref_c"] = avg_logp(model, *p["c"], a.device)[0].item()
            p["ref_r"] = avg_logp(model, *p["r"], a.device)[0].item()
    print(f"reference log-probs in {time.time() - t0:.0f}s; adapter (b) prefers the chosen side on "
          f"{sum(p['ref_c'] > p['ref_r'] for p in pairs)}/{len(pairs)} pairs", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    order = [i for e in range(a.epochs) for i in random.Random(a.seed + e).sample(range(len(pairs)), len(pairs))]
    total_steps = max(1, math.ceil(len(order) / a.grad_accum))
    warm = max(1, total_steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    model.train()
    step, log, t0 = 0, [], time.time()
    for k, i in enumerate(order):
        p = pairs[i]
        with torch.no_grad():
            xc = avg_logp(model, *p["c"], a.device)[0].float()
            xr = avg_logp(model, *p["r"], a.device)[0].float()
        xc_, xr_ = xc.clone().requires_grad_(), xr.clone().requires_grad_()
        margin = a.beta * ((xc_ - p["ref_c"]) - (xr_ - p["ref_r"]))
        loss = -torch.nn.functional.logsigmoid(margin) - a.mu * xc_
        loss.backward()
        gc, gr = xc_.grad.item(), xr_.grad.item()
        for side, g in (("c", gc), ("r", gr)):
            x, _ = avg_logp(model, *p[side], a.device)
            (g * x / a.grad_accum).backward()
        log.append((loss.item(), margin.item(), xc.item(), xr.item()))
        if (k + 1) % a.grad_accum == 0 or k + 1 == len(order):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
            step += 1
            n = len(log)
            print(f"step {step}/{total_steps} loss {sum(x[0] for x in log) / n:.4f} "
                  f"margin {sum(x[1] for x in log) / n:+.4f} acc {sum(x[1] > 0 for x in log) / n:.2f} "
                  f"logp chosen {sum(x[2] for x in log) / n:.4f} rejected {sum(x[3] for x in log) / n:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
            log = []
            if a.steps and step >= a.steps:
                break
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    (Path(a.out) / "train_args.json").write_text(json.dumps({**vars(a), "pairs": len(pairs)}, indent=2))
    print(f"adapter -> {a.out}", flush=True)
