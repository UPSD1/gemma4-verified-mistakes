"""Combine the test sets that may be sent to outside raters into one file for the
fairness screen: OOD-45 (CC BY 4.0) and the gRPC slice (python-api-core, Apache-2.0).
The 129 competition public tasks are left out on purpose (competition data).

Adds meta.set ("ood45" | "grpc") and, for gRPC tasks, meta.grpc_tag:
  grpc_path      fix changes gRPC-path modules (retry, bidi, grpc_helpers, timeout, ...)
                 and the changed lines use gRPC/retry code
  grpc_adjacent  one of the two holds
  other          general client code (client_options, path_template, ...)
The tag is automatic (file names + keywords); check it by hand before quoting it.

    python3 verify/build_screen_set.py               # test sets -> data/tasks/screen/test_sets.jsonl
    python3 verify/build_screen_set.py --train       # FreshSWE-Py training tasks -> train_sets.jsonl
    python3 verify/build_screen_set.py --mined       # our 184 mined tasks -> mined_sets.jsonl

--train rates the 647 FreshSWE-Py training tasks (data/freshswe/train_tasks.jsonl, the IDs in
progress/train_task_ids.json). Report only: ratings never drop training tasks. Our 184 mined
tasks (--mined) are rated from data/tasks/train_tasks.jsonl, i.e. with the bug report the
agent sees (issue, PR description or stripped CI log; see verify/log_statements.py).
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mining"))
from common import ROOT, TASK_DIR, read_jsonl, write_jsonl  # noqa: E402

GRPC_FILE = re.compile(r"grpc|bidi|retry|timeout|streaming|operations_v1/(operations_client|abstract|transports/grpc)"
                       r"|operation_async|operation\.py", re.I)
GRPC_CODE = re.compile(r"\bgrpc\.|StatusCode|aio\.|grpc_helpers|bidi|from_grpc|RpcError|_retry|retry_target|deadline",
                       re.I)


HAND_TAGS_FILE = ROOT / "progress" / "grpc_tags.json"
HAND_TAGS = json.loads(HAND_TAGS_FILE.read_text()) if HAND_TAGS_FILE.exists() else None


def grpc_tag(task):
    files = [f for f in re.findall(r"^diff --git a/(\S+)", task["patch"], re.M) if f.endswith(".py")]
    changed = "\n".join(l for l in task["patch"].splitlines()
                        if l[:1] in "+-" and not l.startswith(("+++", "---")))
    f_hit, c_hit = any(GRPC_FILE.search(f) for f in files), bool(GRPC_CODE.search(changed))
    return "grpc_path" if f_hit and c_hit else "grpc_adjacent" if f_hit or c_hit else "other"


if __name__ == "__main__":
    train, mined = "--train" in sys.argv[1:], "--mined" in sys.argv[1:]
    if mined:
        sets = [("mined", TASK_DIR / "train_tasks.jsonl")]
    elif train:
        sets = [("freshswe", ROOT / "data" / "freshswe" / "train_tasks.jsonl")]
    else:
        sets = [("ood45", ROOT / "data" / "ood45" / "tasks.jsonl"), ("grpc", TASK_DIR / "grpc" / "verified.jsonl")]
    rows = []
    for name, path in sets:
        if not path.exists():
            sys.exit(f"missing {path}")
        for t in read_jsonl(path):
            meta = {**t.get("meta", {}), "set": name}
            if name == "grpc":
                meta["grpc_tag"] = grpc_tag(t)
                if HAND_TAGS:  # hand-checked labels win over the automatic tag
                    meta["grpc_tag"] = "grpc_path" if t["instance_id"] in HAND_TAGS["grpc_path"]["ids"] else "other"
            rows.append({**t, "meta": meta})
    out = TASK_DIR / "screen" / ("mined_sets.jsonl" if mined else "train_sets.jsonl" if train else "test_sets.jsonl")
    write_jsonl(out, rows)
    print(f"{len(rows)} tasks -> {out}: {dict(Counter(r['meta']['set'] for r in rows))}")
    if not (train or mined):
        print("gRPC tags:", dict(Counter(r["meta"]["grpc_tag"] for r in rows if "grpc_tag" in r["meta"])))
