#!/usr/bin/env bash
# Submit-only arm, compute-matched: 1,078,640 trained tokens per pass x 2.086 passes = 2.25M, the budget of arms (a)-(c).
source /workspace/env.sh >/dev/null 2>&1; cd /workspace/repo
L=/workspace/logs/round2.log; say() { echo "[$(date -u +%H:%M)] $*" >> $L; }
say "arm_a_submit_only_v3, compute-matched: 2.086 passes over the 282 examples (= 2.25M trained tokens), rank 32, lr 1e-4, seed 0"
python training/train_sft.py --data data/training/arm_a_submit_only_v3.jsonl --out /root/rt/adapters/arm_a_submit_only_v3 \
  --model /workspace/models/gemma-4-12b-it-qat-unquantized --epochs 2.086 --seed 0 > /workspace/logs/train_arm_a_submit_only_v3.log 2>&1
say "arm_a_submit_only_v3 training exited ($?): $(tail -1 /workspace/logs/train_arm_a_submit_only_v3.log)"
