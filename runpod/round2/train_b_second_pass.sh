#!/usr/bin/env bash
# Round 2, arm (b) second pass: continue the v3 (b) adapter for one more pass over the same data and budget, in the
# example order a two-epoch run would have used, with a fresh warmup and cosine decay at lr 1e-4. About 3.6 h on one
# RTX PRO 6000.
source /workspace/env.sh >/dev/null 2>&1; cd /workspace/repo
python training/train_sft.py --data data/training/arm_b_sft_v3.jsonl --out /root/rt/adapters/arm_b_v3_ep2 \
  --model /workspace/models/gemma-4-12b-it-qat-unquantized --init-adapter /root/agents/arm_b_v3/adapters/arm_b_v3 \
  --order-offset 1 --max-trained-tokens 2250000 --seed 0 > /workspace/logs/train_arm_b_v3_ep2.log 2>&1
