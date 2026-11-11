"""Package mined tasks so the competition harness runs them like FreshSWE-Py (tracker row 8).

Runs ON THE POD (it needs internet and Linux wheels). For each task it builds, under --out:

  snapshots/<instance_id>.tgz  the repo at base_commit, as a fresh git repo with one commit
                               "base <sha>" by "snapshot" (FreshSWE-Py's layout; its sandbox/setup.py
                               finds the lock from that message)
  locks/<sha>.txt              pinned environment: the package (with --extras) and its test
                               dependencies, installed in a clean Python 3.12 venv, then pip freeze
  wheelhouse/                  wheels for every pin, so the sandbox installs offline
  sandbox/setup.py             FreshSWE-Py's setup script (installs the lock, then the workspace)
  tasks.jsonl                  the harness's task fields

Run the harness on it with runner/run_eval.py --fresh <out> --tasks <out>/tasks.jsonl, after
pointing FRESHSWE_WHEELHOUSE and FRESHSWE_LOCKS at <out>/wheelhouse and <out>/locks.

    python3 packaging/build_slice.py --tasks data/tasks/grpc/verified.jsonl \\
        --out /workspace/datasets/grpc_slice --extras grpc --extra-pip "pytest-asyncio mock"
"""
import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HARNESS_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement", "hints_text", "patch",
                  "test_patch", "created_at")
# Same requirement files, in the same order, as verify/in_container.sh, so the packaged environment
# matches the one the tasks were verified in.
TEST_REQS = ("requirements-tests.txt", "requirements-dev.txt", "dev-requirements.txt", "requirements.txt",
             "requirements/testing.txt")


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def mirror(repo, cache):
    path = cache / repo.replace("/", "__")
    if not path.exists():
        sh(["git", "clone", "-q", "--mirror", f"https://github.com/{repo}", str(path)])
    return path


def snapshot(task, mirror_path, out):
    dst = out / "snapshots" / f"{task['instance_id']}.tgz"
    if dst.exists():
        return
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "ws"
        ws.mkdir()
        arc = subprocess.run(["git", "-C", str(mirror_path), "archive", task["base_commit"]],
                             check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(ws)], input=arc, check=True)
        env = {"GIT_AUTHOR_NAME": "snapshot", "GIT_AUTHOR_EMAIL": "snapshot@local",
               "GIT_COMMITTER_NAME": "snapshot", "GIT_COMMITTER_EMAIL": "snapshot@local", "PATH": "/usr/bin:/bin"}
        sh(["git", "init", "-q"], cwd=ws)
        sh(["git", "add", "-A"], cwd=ws)
        sh(["git", "commit", "-q", "-m", f"base {task['base_commit']}"], cwd=ws, env=env)
        with tarfile.open(dst, "w:gz") as tf:
            tf.add(ws, arcname=".")


def lock(task, mirror_path, out, python, extras, extra_pip):
    sha = task["base_commit"]
    dst = out / "locks" / f"{sha}.txt"
    if dst.exists():
        return "cached"
    with tempfile.TemporaryDirectory() as tmp:
        ws, venv = Path(tmp) / "ws", Path(tmp) / "venv"
        ws.mkdir()
        arc = subprocess.run(["git", "-C", str(mirror_path), "archive", sha], check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(ws)], input=arc, check=True)
        sh([python, "-m", "venv", str(venv)])
        pip = [str(venv / "bin" / "python"), "-m", "pip", "install", "-q"]
        target = f".[{extras}]" if extras else "."
        sh(pip + [target], cwd=ws)
        for req in TEST_REQS:
            if (ws / req).exists():
                subprocess.run(pip + ["-r", req], cwd=ws, capture_output=True)
        sh(pip + ["pytest", "pytest-timeout", "anyio", *extra_pip.split()])
        frozen = sh([str(venv / "bin" / "python"), "-m", "pip", "freeze", "--exclude-editable"]).stdout
        # Package name from pyproject: PEP 621 [project] or Poetry's [tool.poetry] (rich); empty if neither.
        name = sh([str(venv / "bin" / "python"), "-c",
                   "import tomllib;d=tomllib.load(open('pyproject.toml','rb'));"
                   "print(d.get('project',{}).get('name') or d.get('tool',{}).get('poetry',{}).get('name') or '')"],
                  cwd=ws).stdout.strip() if (ws / "pyproject.toml").exists() else ""
        pins = [l for l in frozen.splitlines() if l and not l.startswith("-e ")
                and " @ " not in l and not (name and l.lower().startswith(name.lower().replace("_", "-") + "=="))]
        # Drop the package itself (installed from the workspace by setup.py), however it is spelled.
        own = {name.lower().replace("_", "-")} if name else set()
        own |= {"google-api-core"} if (ws / "google" / "api_core").exists() else set()
        own |= {Path(task["repo"]).name.lower()}
        pins = [l for l in pins if l.split("==")[0].lower().replace("_", "-") not in own]
        dst.write_text("\n".join(pins) + "\n")
        subprocess.run([str(venv / "bin" / "python"), "-m", "pip", "wheel", "-q", "--no-deps",
                        "-w", str(out / "wheelhouse"), "-r", str(dst)], capture_output=True)
    return "built"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extras", default="")
    ap.add_argument("--extra-pip", default="")
    ap.add_argument("--python", default="python3.12")
    ap.add_argument("--setup-py", default="/workspace/datasets/freshswe/sandbox/setup.py")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    out = Path(a.out)
    for d in ("snapshots", "locks", "wheelhouse", "sandbox"):
        (out / d).mkdir(parents=True, exist_ok=True)
    # The sandbox does not inherit FRESHSWE_WHEELHOUSE / FRESHSWE_LOCKS, so bake this slice's paths
    # in as the defaults; otherwise setup.py finds no lock and silently installs nothing.
    setup = Path(a.setup_py).read_text()
    setup = setup.replace('"/opt/freshswe/wheelhouse"', f'"{out.resolve() / "wheelhouse"}"')
    setup = setup.replace('"/opt/freshswe/locks"', f'"{out.resolve() / "locks"}"')
    (out / "sandbox" / "setup.py").write_text(setup)
    tasks = [json.loads(l) for l in open(a.tasks) if l.strip()]
    cache = out / "_mirrors"
    cache.mkdir(exist_ok=True)
    mirrors = {r: mirror(r, cache) for r in sorted({t["repo"] for t in tasks})}

    def one(t):
        try:
            snapshot(t, mirrors[t["repo"]], out)
            how = lock(t, mirrors[t["repo"]], out, a.python, a.extras, a.extra_pip)
            return t["instance_id"], "ok", how
        except subprocess.CalledProcessError as e:
            return t["instance_id"], "failed", (e.stderr or "")[-300:]

    results = list(ThreadPoolExecutor(a.workers).map(one, tasks))
    for r in results:
        print(*r, flush=True)
    ok = [t for t, (_, s, _) in zip(tasks, results) if s == "ok"]
    with open(out / "tasks.jsonl", "w") as f:
        for t in ok:
            f.write(json.dumps({k: t.get(k, "") for k in HARNESS_FIELDS}) + "\n")
    print(f"{len(ok)}/{len(tasks)} tasks packaged -> {out}; "
          f"{len(list((out / 'wheelhouse').glob('*.whl')))} wheels, {len(list((out / 'locks').glob('*.txt')))} locks")
    sys.exit(0 if ok else 1)
