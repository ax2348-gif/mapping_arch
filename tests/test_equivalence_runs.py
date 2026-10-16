"""Whole runs are unchanged by the config refactor.

Each case trains a short run three ways and compares everything it writes:
  old    the pre-refactor checkout (``MAPPING_REF``, e.g. a worktree of
         master before refactor/hydra), where se2d was still ``--task se2``
  flags  this checkout's ``train.py --flags``
  run    this checkout's ``run.py`` with Hydra overrides written by hand
On CPU the runs must be bit-identical: log.jsonl, final.json and the
checkpoints' tensors and args (timing and the output path aside, and the
task's old name ``se2`` read as ``se2d``), and grid.png byte for byte. New
checkpoints add one entry, ``arch`` (the class and constructor arguments
the model was built with); it must rebuild the checkpoint's model, and
everything else in them must be identical. The
GPU case (a Triton delta-rule model) is not bit-reproducible even run to
run (about 1e-6), so it is compared within that noise.

    MAPPING_REF=/path/to/old/checkout python -m pytest tests/test_equivalence_runs.py
Skipped without MAPPING_REF.

The se2c cases need a reference with the se2c odometry-dropout knob
(master 038f19c or later, before the merge of refactor/hydra): from that
commit on, every se2c run records ``generator.odometry_dropout`` and
``--odometry-dropout`` acts on se2c. With an older reference they are
skipped, never compared against the wrong code. The z2 and se2d cases run
against either.
"""
import json
import os
import shutil
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
                 ['model=se2d/vanilla4', 'model.size=s', 'task.generator.workers=2',
                  'run.device=cpu', 'schedule=wsd', 'schedule.cooldown=0.5', 'run.seed=3']),
    'se2c': (['--task', 'se2c', '--model', 'vanilla4c', '--size', 'xs', '--device', 'cpu',
              '--turn-n', '8'],
             ['model=se2c/vanilla4c', 'run.device=cpu', 'task.generator.turn_n=8']),
    # partial dropout: the draw itself (RNG position, zeroed rows) must match
    'se2c_dropout': (['--task', 'se2c', '--model', 'vanilla4c', '--size', 'xs', '--device', 'cpu',
                      '--odometry-dropout', '0.5'],
                     ['model=se2c/vanilla4c', 'run.device=cpu',
                      'task.generator.odometry_dropout=0.5']),
    # visual odometry, as the Torch run results/se2c/hybrid4c/m/vo_120k_qt
    'se2c_vo': (['--task', 'se2c', '--model', 'vanilla4c', '--size', 'xs', '--device', 'cpu',
                 '--odometry-dropout', '1.0'],
                ['model=se2c/vanilla4c', 'run.device=cpu', 'task.generator.odometry_dropout=1.0']),
}


def ref_has_se2c_dropout():
    try:
        src = open(os.path.join(REF, 'mapping', 'se2c', 'generator.py')).read()
    except (OSError, TypeError):
        return False
    return 'odometry_dropout' in src
GPU_CASE = (['--task', 'se2d', '--model', 'hybrid4', '--size', 'xs', '--workers', '2'],
            ['model=se2d/hybrid4', 'task.generator.workers=2'])
TIMING = ('elapsed_s', 'steps_per_s')
NOTE = []          # run.py launch arguments every run needs besides its overrides


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


def rebuilds(arch, state):
    """``arch`` builds a module with exactly the checkpoint's parameters."""
    import mapping.tasks as tasks
    from mapping.checkpoint import build_from_arch
    from mapping.task_se2c import TaskSpec as Se2cSpec
    for spec in (tasks.config_and_spec('z2', 32, 8, 0)[1],
                 tasks.config_and_spec('se2d', 32, 8, 0)[1], Se2cSpec()):
        try:
            got = build_from_arch(arch, spec).state_dict()
        except Exception:
            continue
        if got.keys() == state.keys() and all(got[k].shape == state[k].shape for k in got):
            return True
    return False


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
    if case.startswith('se2c') and not ref_has_se2c_dropout():
        pytest.skip('MAPPING_REF predates the se2c odometry-dropout knob (master 038f19c)')
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
            arch = ckpt[name].pop('arch')        # new, additive
            assert same_tensors(ckpt[name], old_ckpt[name]), (k, name)
            assert rebuilds(arch, ckpt[name]['model']), (k, name)
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


# Default run directories, against the reference. Every case above writes
# through an explicit run.out / --out, so the path is compared here, and
# the one change is stated, not normalised away: the run directory
# template (docs/config.md, "Run directories") puts z2 runs under a task
# segment, results/z2/, where the reference wrote results/<model>/. The
# reference's se2 is se2d (the rename); se2c is unchanged.
PATH_CHANGE = {
    'z2': lambda old: os.path.join('z2', old),
    'se2d_wsd': lambda old: 'se2d' + old[len('se2'):] if old.startswith('se2' + os.sep) else None,
    'se2c': lambda old: old,
}


def _run_dir_under(root):
    found = [dp for dp, _, fs in os.walk(root) if 'final.json' in fs]
    assert len(found) == 1, found
    return os.path.relpath(found[0], root)


@pytest.mark.parametrize('case', sorted(PATH_CHANGE))
def test_default_run_directory_is_the_reference_one_with_the_task_segment(tmp_path, case):
    if case.startswith('se2c') and not ref_has_se2c_dropout():
        pytest.skip('MAPPING_REF predates the se2c odometry-dropout knob (master 038f19c)')
    flags, overrides = CASES[case]
    name = f'equiv_path_{os.getpid()}_{case}'
    ref_results = os.path.join(REF, 'results')
    before, existed = set(), os.path.isdir(ref_results)
    if existed:
        before = {os.path.join(dp, d) for dp, ds, _ in os.walk(ref_results) for d in ds}
    old_flags = ['se2' if f == 'se2d' else f for f in flags]
    r = subprocess.run([sys.executable, 'train.py', *old_flags, *COMMON, '--run', name],
                       cwd=REF, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    try:
        old_rel = next(os.path.relpath(dp, ref_results) for dp, _, fs in os.walk(ref_results)
                       if os.path.basename(dp) == name and 'final.json' in fs)
    finally:
        for dp, ds, _ in sorted(os.walk(ref_results), reverse=True):
            if os.path.basename(dp) == name:
                shutil.rmtree(dp)
        for d in sorted({os.path.join(dp, d) for dp, ds, _ in os.walk(ref_results) for d in ds}
                        - before, reverse=True):
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
        if not existed and os.path.isdir(ref_results) and not os.listdir(ref_results):
            os.rmdir(ref_results)
    root = tmp_path / 'root'
    r = subprocess.run([sys.executable, 'run.py', *overrides, *COMMON_RUN, f'run.root={root}',
                        f'run.name={name}', *NOTE], cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    new_rel = _run_dir_under(root)
    assert new_rel == PATH_CHANGE[case](old_rel), (old_rel, new_rel)
