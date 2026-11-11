"""Pick N FreshSWE-Py training tasks for base-model runs: round-robin over repos (seeded), so no repo dominates.
Usage: python runpod/pick_fresh.py <freshswe_dir> <train_task_ids.json> <out.jsonl> [N]"""
import json
import random
import sys
from collections import defaultdict

fresh, ids_path, out = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 200
train = set(json.load(open(ids_path))["freshswe_train"]["ids"])
by_repo = defaultdict(list)
for line in open(f"{fresh}/tasks.jsonl"):
    r = json.loads(line)
    if r["instance_id"] in train:
        by_repo[r["repo"]].append(r)
rng = random.Random(0)
for v in by_repo.values():
    rng.shuffle(v)
picked, i = [], 0
while len(picked) < n and any(by_repo.values()):
    for repo in sorted(by_repo):
        if by_repo[repo] and len(picked) < n:
            picked.append(by_repo[repo].pop())
with open(out, "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in picked)
print(len(picked), "tasks from", len({r["repo"] for r in picked}), "repos")
