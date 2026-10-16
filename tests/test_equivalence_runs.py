"""Whole runs are unchanged by the config refactor.

Each case trains a short run three ways and compares everything it writes:
  old    the pre-refactor checkout (``MAPPING_REF``, e.g. a worktree of
         master before refactor/hydra), where se2d was still ``--task se2``
  flags  this checkout's ``train.py --flags``
  run    this checkout's ``run.py`` with Hydra overrides written by hand
On CPU the runs must be bit-identical: log.jsonl, final.json and the
checkpoints' tensors and args (timing and the output path aside, and the
task's old name ``se2`` read as ``se2d``), and grid.png byte for byte. The
GPU case (a Triton delta-rule model) is not bit-reproducible even run to
run (about 1e-6), so it is compared within that noise.

    MAPPING_REF=/path/to/old/checkout python -m pytest tests/test_equivalence_runs.py
Skipped without MAPPING_REF.
"""
import json
import os
import subprocess
import sys

import pytest
import torch

REF = os.environ.get('MAPPING_REF')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(not REF, reason='set MAPPING_REF to a pre-refactor checkout')

COMMON = ['--steps', '20', '--eval-every', '10', '--log-every', '5', '--eval-batches', '2',
          '--batch', '8', '--query-times', 'uniform', '--save-every', '10']
COMMON_RUN = ['run.steps=20', 'eval.every=10', 'eval.log_every=5', 'eval.batches=2',
              'optim.batch=8', 'task.query_times=uniform', 'run.save_every=10']
CASES = {
    'z2': (['--model', 'vanilla4', '--size', 'xs', '--workers', '2', '--device', 'cpu'],
           ['model=z2/vanilla4', 'task.generator.workers=2', 'run.device=cpu']),
    'se2d_wsd': (['--task', 'se2d', '--model', 'vanilla4', '--size', 's', '--workers', '2',
                  '--device', 'cpu', '--schedule', 'wsd', '--cooldown', '0.5', '--seed', '3'],
                 ['model=se2d/vanilla4', 'model/size=s', 'task.generator.workers=2',
                  'run.device=cpu', 'schedule=wsd', 'schedule.cooldown=0.5', 'run.seed=3']),
    'se2c': (['--task', 'se2c', '--model', 'vanilla4c', '--size', 'xs', '--device', 'cpu',
              '--turn-n', '8'],
             ['model=se2c/vanilla4c', 'run.device=cpu', 'task.generator.turn_n=8']),
}
GPU_CASE = (['--task', 'se2d', '--model', 'hybrid4', '--size', 'xs', '--workers', '2'],
            ['model=se2d/hybrid4', 'task.generator.workers=2'])
TIMING = ('elapsed_s', 'steps_per_s')


def launch(cwd, script, argv, out):
    r = subprocess.run([sys.executable, script, *argv, *(['--out', out] if script == 'train.py'
                                                          else [f'run.out={out}'])],
                       cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]


def three_runs(tmp_path, flags, overrides):
    old_flags = ['se2' if f == 'se2d' else f for f in flags]   # the old task name
    dirs = {k: str(tmp_path / k) for k in ('old', 'flags', 'run')}
    launch(REF, 'train.py', old_flags + COMMON, dirs['old'])
    launch(HERE, 'train.py', flags + COMMON, dirs['flags'])
    launch(HERE, 'run.py', overrides + COMMON_RUN, dirs['run'])
    return dirs


def norm(x):
    """Timing and the output path dropped; the old task name read as new."""
    if isinstance(x, dict):
        return {k: norm(v) for k, v in x.items()
                if k not in TIMING and k != 'out'}
    if isinstance(x, list):
        return [norm(v) for v in x]
    return 'se2d' if x == 'se2' else x


def records(d):
    log = [norm(json.loads(line)) for line in open(os.path.join(d, 'log.jsonl'))]
    final = norm(json.load(open(os.path.join(d, 'final.json'))))
    return log, final


def checkpoints(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith('.pt'):
            out[f] = torch.load(os.path.join(d, f), map_location='cpu', weights_only=False)
    return out


def same_tensors(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        keep = [k for k in a if k not in TIMING and k != 'out']
        return a.keys() == b.keys() and all(same_tensors(a[k], b[k]) for k in keep)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_tensors(x, y) for x, y in zip(a, b))
    return norm(a) == norm(b)


def same_grid(a, b):
    """Byte-identical, or identical pixels below the title line (the title
    names the task, so an se2d run's differs from the old se2 one's)."""
    if open(a, 'rb').read() == open(b, 'rb').read():
        return True
    import numpy as np
    from PIL import Image
    x, y = np.asarray(Image.open(a)), np.asarray(Image.open(b))
    return x.shape == y.shape and np.array_equal(x[TITLE_ROWS:], y[TITLE_ROWS:])


TITLE_ROWS = 22      # mapping.render: the title is drawn at y = 4 in a 13 px font


@pytest.mark.parametrize('case', sorted(CASES))
def test_cpu_runs_are_bit_identical(tmp_path, case):
    dirs = three_runs(tmp_path, *CASES[case])
    old_log, old_final = records(dirs['old'])
    old_ckpt = checkpoints(dirs['old'])
    old_grid = os.path.join(dirs['old'], 'grid.png')
    for k in ('flags', 'run'):
        log, final = records(dirs[k])
        assert log == old_log, k
        assert final == old_final, k
        ckpt = checkpoints(dirs[k])
        assert ckpt.keys() == old_ckpt.keys(), k
        for name in ckpt:
            assert same_tensors(ckpt[name], old_ckpt[name]), (k, name)
        assert same_grid(os.path.join(dirs[k], 'grid.png'), old_grid), k
        assert os.path.exists(os.path.join(dirs[k], 'config.yaml')), k


def gap(log_a, log_b):
    """Largest difference of any logged number (train and eval metrics)."""
    worst = 0.0
    for a, b in zip(log_a[1:], log_b[1:]):
        worst = max(worst, abs(a['train_mse'] - b['train_mse']))
        for key in ('eval', 'eval_shuffled'):
            if key in b:
                assert a[key].keys() == b[key].keys()
                worst = max(worst, max(abs(a[key][m] - b[key][m]) for m in b[key]))
    return worst


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_gpu_run_matches_within_noise(tmp_path):
    """The old code run twice sets the noise; new runs must stay inside it."""
    dirs = three_runs(tmp_path, *GPU_CASE)
    launch(REF, 'train.py', ['se2' if f == 'se2d' else f for f in GPU_CASE[0]] + COMMON,
           str(tmp_path / 'old2'))
    old_log, old_final = records(dirs['old'])
    noise = gap(records(str(tmp_path / 'old2'))[0], old_log)
    for k in ('flags', 'run'):
        log, final = records(dirs[k])
        assert final['args'] == old_final['args'] and final['params'] == old_final['params'], k
        assert len(log) == len(old_log)
        # count metrics (within_0.1) move in steps of one pixel's share; a
        # single old pair underestimates the noise, so allow 3x it, or 1e-3
        assert gap(log, old_log) <= max(3 * noise, 1e-3), (k, gap(log, old_log), noise)
