"""Shared helpers: repo list, GitHub calls through the gh CLI, file classification."""
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
TASK_DIR = ROOT / "data" / "tasks"

# Same four repos as the competition; short names match its instance_id prefixes.
REPOS = {
    "fastapi/fastapi": "fastapi",
    "Textualize/rich": "rich",
    "psf/requests": "requests",
    "encode/httpx": "httpx",
}

# gRPC slice: evaluation only, never training (Problem areas tab). Pure-Python libraries
# built on gRPC whose repo root installs with pip; grpc/grpc itself needs a C++ build.
GRPC_REPOS = {
    "googleapis/python-api-core": "api_core",
    "vmagamedov/grpclib": "grpclib",
    "d5h-foss/grpc-interceptor": "grpc_interceptor",
    "python-trio/purerpc": "purerpc",
}
GRPC_RAW_DIR = RAW_DIR / "grpc"
GRPC_TASK_DIR = TASK_DIR / "grpc"

TEST_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")
# Paths that are never the "fix": docs, doc examples, CI, scripts.
NON_CODE_RE = re.compile(r"^(docs|docs_src|doc|scripts|\.github|benchmarks?|examples?)/")


def is_test_file(path: str) -> bool:
    return bool(TEST_RE.search(path))


def is_code_file(path: str) -> bool:
    return path.endswith(".py") and not is_test_file(path) and not NON_CODE_RE.search(path)


def gh(args, stdin=None, retries=4):
    """Run `gh` and return stdout. Retries on rate limits / transient errors."""
    for attempt in range(retries):
        proc = subprocess.run(["gh", *args], input=stdin, capture_output=True, text=True)
        if proc.returncode == 0:
            return proc.stdout
        err = proc.stderr.lower()
        if any(s in err for s in ("rate limit", "502", "503", "timeout", "secondary")):
            time.sleep(20 * (attempt + 1))
            continue
        raise RuntimeError(f"gh {' '.join(args[:3])} failed: {proc.stderr.strip()[:500]}")
    raise RuntimeError(f"gh {' '.join(args[:3])} failed after {retries} retries")


def gh_graphql(query: str, **variables):
    args = ["api", "graphql", "-f", f"query={query}"]
    for k, v in variables.items():
        if v is None:
            continue
        args += ["-F" if isinstance(v, int) else "-f", f"{k}={v}"]
    data = json.loads(gh(args))
    if data.get("errors"):
        raise RuntimeError(data["errors"])
    return data["data"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
