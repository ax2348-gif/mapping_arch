"""Test-suite isolation for run provenance (mapping/provenance.py).

Every run a test launches makes a snapshot and a ledger record, as every
run must. So that a suite run does not fill this repository's
``refs/runs/`` and ``experiments/ledger/`` with test runs, the whole
session (and every subprocess it starts) writes them elsewhere:
- ``XP_SNAPSHOT_GIT_DIR``: a scratch bare repository that borrows this
  repository's objects (``objects/info/alternates``), so snapshot commits,
  their new objects and their refs land there; the working tree, HEAD and
  the index snapshotted are this checkout's, as in a real run;
- ``XP_LEDGER_DIR``: a scratch ledger directory;
- ``XP_HOST=test``.
tests/test_provenance.py checks the real thing, without these, on
throwaway repositories.
"""
import os
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRATCH = []


def pytest_configure(config):
    if os.environ.get('XP_SNAPSHOT_GIT_DIR'):       # an outer session set it up
        return
    scratch = tempfile.mkdtemp(prefix='mapping-tests-provenance-')
    _SCRATCH.append(scratch)
    store = os.path.join(scratch, 'store.git')
    subprocess.run(['git', 'init', '-q', '--bare', store], check=True)
    common = subprocess.run(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'],
                            cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    with open(os.path.join(store, 'objects', 'info', 'alternates'), 'w') as f:
        f.write(os.path.join(common, 'objects') + '\n')
    os.environ['XP_SNAPSHOT_GIT_DIR'] = store
    os.environ['XP_LEDGER_DIR'] = os.path.join(scratch, 'ledger')
    os.environ['XP_HOST'] = 'test'


def pytest_unconfigure(config):
    for d in _SCRATCH:
        shutil.rmtree(d, ignore_errors=True)
