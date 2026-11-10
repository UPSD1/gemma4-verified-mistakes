"""Combine the test sets that may be sent to outside raters into one file for the
fairness screen: OOD-45 (CC BY 4.0) and the gRPC slice (python-api-core, Apache-2.0).
The 129 competition public tasks are left out on purpose (competition data).

Adds meta.set ("ood45" | "grpc") and, for gRPC tasks, meta.grpc_tag:
  grpc_path      fix changes gRPC-path modules (retry, bidi, grpc_helpers, timeout, ...)
                 and the changed lines use gRPC/retry code
  grpc_adjacent  one of the two holds
  other          general client code (client_options, path_template, ...)
The tag is automatic (file names + keywords); check it by hand before quoting it.

    python3 verify/build_screen_set.py
"""
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


def grpc_tag(task):
    files = [f for f in re.findall(r"^diff --git a/(\S+)", task["patch"], re.M) if f.endswith(".py")]
    changed = "\n".join(l for l in task["patch"].splitlines()
                        if l[:1] in "+-" and not l.startswith(("+++", "---")))
    f_hit, c_hit = any(GRPC_FILE.search(f) for f in files), bool(GRPC_CODE.search(changed))
    return "grpc_path" if f_hit and c_hit else "grpc_adjacent" if f_hit or c_hit else "other"


if __name__ == "__main__":
    sets = [("ood45", ROOT / "data" / "ood45" / "tasks.jsonl"), ("grpc", TASK_DIR / "grpc" / "verified.jsonl")]
    rows = []
    for name, path in sets:
        if not path.exists():
            sys.exit(f"missing {path}")
        for t in read_jsonl(path):
            meta = {**t.get("meta", {}), "set": name}
            if name == "grpc":
                meta["grpc_tag"] = grpc_tag(t)
            rows.append({**t, "meta": meta})
    out = TASK_DIR / "screen" / "test_sets.jsonl"
    write_jsonl(out, rows)
    print(f"{len(rows)} tasks -> {out}: {dict(Counter(r['meta']['set'] for r in rows))}")
    print("gRPC tags:", dict(Counter(r["meta"]["grpc_tag"] for r in rows if "grpc_tag" in r["meta"])))
