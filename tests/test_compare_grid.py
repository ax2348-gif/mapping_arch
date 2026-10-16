"""tools/compare_grid.py: checkpoints whose task settings differ (odometry
dropout included) are compared only with a loud warning, on stderr and in
the image; matched ones without it."""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tiny_run(out, *overrides):
    cmd = [sys.executable, 'run.py', 'model=se2c/vanilla4c', 'model.size=xs', 'run.device=cpu',
           'run.steps=2', 'eval.every=2', 'eval.log_every=2', 'eval.batches=1', 'optim.batch=2',
           'task.query_times=uniform', f'run.out={out}', *overrides]
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return os.path.join(out, 'final.pt')


def compare(tmp_path, a, b, name):
    out = str(tmp_path / name)
    r = subprocess.run([sys.executable, 'tools/compare_grid.py', a, b, '--labels', 'A', 'B',
                        '--device', 'cpu', '--rows', '2', '--batches', '1', '--out', out],
                       cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stderr, np.asarray(Image.open(out))


def test_unmatched_settings_warn_on_stderr_and_in_the_image(tmp_path):
    a = tiny_run(tmp_path / 'a')
    b = tiny_run(tmp_path / 'b', 'task.generator.odometry_dropout=1.0')
    c = tiny_run(tmp_path / 'c', 'run.seed=0')          # same settings as a
    err, img = compare(tmp_path, a, b, 'unmatched.png')
    assert 'NOT A MATCHED COMPARISON' in err and 'odometry_dropout' in err
    assert any(set(line) == {'#'} and len(line) > 20 for line in err.splitlines())  # boxed
    assert (img[:20] == 255).mean() > 0.5                 # the inverted banner on top
    err, img = compare(tmp_path, a, c, 'matched.png')
    assert 'NOT A MATCHED' not in err
    assert (img[:20] == 255).mean() < 0.1
