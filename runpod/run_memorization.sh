#!/usr/bin/env bash
# Memorization check (6c) for every task set, against the vLLM server on this pod.
source /workspace/env.sh
cd /workspace/repo
O=/workspace/results/memorization; mkdir -p $O
python3 verify/memorization.py --tasks data/tasks/freshswe_train.jsonl --ids data/tasks/fresh200_ids.txt --set freshswe --out $O/familiarity.jsonl
python3 verify/memorization.py --tasks /workspace/datasets/ood45/tasks.jsonl --set ood45 --out $O/familiarity.jsonl
python3 verify/memorization.py --tasks data/tasks/grpc/verified.jsonl --set grpc --out $O/familiarity.jsonl
python3 verify/memorization.py --tasks data/tasks/verified.jsonl --set mined --out $O/familiarity.jsonl
python3 verify/memorization.py --tasks /workspace/comp/tasks.jsonl --set public --out $O/familiarity.jsonl
echo MEM_DONE
