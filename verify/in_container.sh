#!/usr/bin/env bash
# Runs inside the container. /work = repo at base_commit, /task = patch files.
# Prints marker lines the host script parses. Exit code is always 0 unless setup breaks.
set -u
cd /work
export PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
git config --global user.email v@v && git config --global user.name v
git config --global --add safe.directory /work

echo "=== SETUP"
# EXTRAS (e.g. "grpc") installs the package's optional extras; empty by default, so the main
# verification is unchanged. Used for the gRPC slice, whose tests skip without grpcio.
TARGET="."; [ -n "${EXTRAS:-}" ] && TARGET=".[${EXTRAS}]"
pip install -q -e "$TARGET" >/tmp/install.log 2>&1 || pip install -q "$TARGET" >>/tmp/install.log 2>&1 || { echo "SETUP_FAILED install"; tail -20 /tmp/install.log; exit 3; }
for req in requirements-tests.txt requirements-dev.txt dev-requirements.txt requirements.txt requirements/testing.txt; do
  if [ -f "$req" ]; then pip install -q -r "$req" >>/tmp/install.log 2>&1 || true; fi
done
pip install -q pytest pytest-timeout anyio ${EXTRA_PIP:-} >>/tmp/install.log 2>&1
# Requirement files can pull the package itself from PyPI (already containing the fix).
# Reinstall the checked-out code last so the repo at base_commit is what gets tested.
pip install -q --no-deps --force-reinstall -e . >>/tmp/install.log 2>&1
python -c "import importlib.metadata as m, sys; d=m.distribution('$(basename $(python -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['name'])" 2>/dev/null || echo x))'); print('INSTALLED_FROM', d.read_text('direct_url.json'))" 2>/dev/null | head -1
git add -A >/dev/null 2>&1; git commit -qm baseline --allow-empty >/dev/null 2>&1

TEST_FILES=$(cat /task/test_files.txt)

run_tests () {  # $1 = label
  echo "=== RUN $1"
  timeout 600 python -m pytest $TEST_FILES -o addopts="" -o filterwarnings="" -rA --tb=no -q -p no:cacheprovider 2>&1 | grep -E '^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS) ' || true
  echo "=== END $1"
}

# Full pytest output with short tracebacks, as a CI job would show it. Used to write
# log-based bug reports for tasks whose only description is the PR body (which can leak the fix).
run_log () {
  echo "=== LOG"
  timeout 600 python -m pytest $TEST_FILES -o addopts="" -o filterwarnings="" -rf --tb=short -q -p no:cacheprovider 2>&1 | tail -c 20000 || true
  echo "=== ENDLOG"
}

git apply /task/test.patch || { echo "SETUP_FAILED test_patch_apply"; exit 3; }
# Run the pre-fix tests more than once: a test that fails by chance before the fix
# would otherwise be counted as fail-to-pass.
for i in $(seq 1 "${BEFORE_REPEATS:-2}"); do run_tests "before$i"; done
run_log
git apply /task/fix.patch || { echo "SETUP_FAILED fix_patch_apply"; exit 3; }
for i in $(seq 1 "${REPEATS:-3}"); do run_tests "after$i"; done
