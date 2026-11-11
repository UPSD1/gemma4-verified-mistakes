"""Give FreshSWE-Py tasks their pinned environments in the harness's subprocess sandbox.

In subprocess mode swegemma 0.2.11 installs nothing (install_test_dependencies returns early), so the sandbox
only sees the host's packages. This routes that call to FreshSWE-Py's own sandbox/setup.py, which installs the
task's lock offline from the wheelhouse, then the workspace in editable mode, then the competition's pytest.ini
and conftest fast path. Same approach as OOD-45's tools/ood45_harness.py.

Expects /opt/freshswe/wheelhouse (cp312 wheels built from every lock) and /opt/freshswe/locks.
"""
import importlib
import logging
from pathlib import Path

log = logging.getLogger("freshswe_env")
SUBPROCESS = {"SubprocessManager", "SubprocessSandbox"}


def patch(fresh_dir):
    from swegemma.harness import container_setup as cs

    setup_py = Path(fresh_dir) / "sandbox" / "setup.py"
    original = cs.install_test_dependencies
    if getattr(original, "_fresh_wrapper", False):
        return

    def install(mgr, cid, repo="", *, fast_path=True, config=None):
        if type(mgr).__name__ not in SUBPROCESS:
            return original(mgr, cid, repo, fast_path=fast_path, config=config)
        # The sandbox venv also sees the host interpreter's site-packages (_host_env.pth): on the pod that is the
        # vLLM environment. Cut it off before installing, or pip treats lock pins as already satisfied by the host
        # and installs nothing. The task's lock plus the system Python (pytest, build backends, as in the Docker
        # image) is the intended environment.
        mgr.exec(cid, 'rm -f "$VIRTUAL_ENV"/lib/python3*/site-packages/_host_env.pth')
        mgr.copy_to(cid, setup_py, "/tmp/")
        res = mgr.exec(cid, "python3 /tmp/setup.py --workspace /workspace", timeout=900)
        if res.exit_code != 0:
            log.warning("freshswe setup failed for %s: %s", repo, (res.stdout + res.stderr)[-1500:])

    install._fresh_wrapper = True
    for name in ("swegemma.harness.container_setup", "swegemma.harness.verification",
                 "swegemma.harness.agent_runner", "swegemma.evaluate"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if getattr(mod, "install_test_dependencies", None) is original:
            mod.install_test_dependencies = install
