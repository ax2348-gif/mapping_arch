"""The legacy argparse training path is sealed (stage 1 of
docs/hydra_merge_review.md, section 5).

- ``train.py --flags`` is frozen: its flags and their defaults are the
  snapshot in tests/legacy_flags.json. A new setting is a config field
  (``mapping.config``), which reaches the args record through
  ``Experiment.legacy_args`` with no flag. Do not update the snapshot to
  add a flag.
- Only run.py (and train.py's own frozen translator) start training: no
  other module calls ``train.main``, and no other harness module builds an
  argparse trainer (``--model`` and ``--steps`` together).
- ``train.py --flags`` says it is deprecated, on stderr, before training.

Stage 2 (deleting the CLI) is a documented, reversible next step in the
review.
"""
import ast
import glob
import json
import os

import train

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# separate tools with their own argparse loops, not harness training runs
NOT_HARNESS = ('experiments/', 'tests/', 'tools/bench.py', 'tools/se2c_bench.py',
               'shelved/', 'quarantine/', 'results/')


def test_train_py_flags_are_frozen():
    want = json.load(open(os.path.join(HERE, 'tests', 'legacy_flags.json')))
    have = {a.option_strings[0]: (None if a.dest == 'device' else a.default)
            for a in train.build_parser()._actions if a.option_strings and a.dest != 'help'}
    have = json.loads(json.dumps(have))           # as JSON stores it
    added, removed = sorted(set(have) - set(want)), sorted(set(want) - set(have))
    assert not added, (f'train.py gained flags {added}: the argparse path is frozen. Add the '
                       f'setting as a config field instead (docs/config.md)')
    assert not removed, f'train.py lost flags {removed}: old job files would stop translating'
    changed = {k: (want[k], have[k]) for k in want if want[k] != have[k]}
    assert not changed, f'train.py flag defaults changed (snapshot, now): {changed}'


def harness_modules():
    for path in sorted(glob.glob(os.path.join(HERE, '**', '*.py'), recursive=True)):
        rel = os.path.relpath(path, HERE)
        if not rel.startswith(NOT_HARNESS):
            yield rel, ast.parse(open(path).read(), rel)


def calls_train_main(tree):
    """Calls ``train.main(...)``, or ``main(...)`` after ``from train import main``."""
    imported = any(isinstance(n, ast.ImportFrom) and n.module == 'train'
                   and any(a.name == 'main' for a in n.names) for n in ast.walk(tree))
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr == 'main' and \
                    isinstance(f.value, ast.Name) and f.value.id in ('train', 'trainer'):
                return True
            if imported and isinstance(f, ast.Name) and f.id == 'main':
                return True
    return False


def argparse_trainer(tree):
    """A training CLI: an argument parser with both --model and --steps, in a
    module that trains in-process (calls ``.backward()``). Sweep and job
    generators (scaling.py, tools/branch_jobs.py) take the same flags but
    launch run.py."""
    if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == 'backward' for n in ast.walk(tree)):
        return False
    flags = {a.value for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
             for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str)}
    return {'--model', '--steps'} <= flags


def test_only_run_py_trains():
    starters = [rel for rel, tree in harness_modules() if calls_train_main(tree)]
    assert starters == ['run.py'], f'train.main is called outside run.py: {starters}'
    clis = [rel for rel, tree in harness_modules() if argparse_trainer(tree)]
    assert clis == ['train.py'], (f'argparse training entry points other than the frozen '
                                  f'train.py: {clis}. Runs go through run.py')


def test_the_scan_sees_what_it_should():
    """The checks above would catch a new entry point (not vacuous)."""
    tree = ast.parse("import train\np.add_argument('--model')\np.add_argument('--steps')\n"
                     "loss.backward()\ntrain.main(e)\n")
    assert calls_train_main(tree) and argparse_trainer(tree)
    assert calls_train_main(ast.parse('from train import main\nmain(e)\n'))


def test_flags_path_says_it_is_deprecated(capsys, tmp_path):
    import pytest
    with pytest.raises(BaseException):          # stops at the model (CUDA-only on CPU) or runs
        train.main_from_flags(['--note', 'test', '--task', 'se2d', '--model', 'hybrid4', '--size', 'xs',
                               '--device', 'cpu', '--steps', '1', '--out', str(tmp_path / 'r')])
    err = capsys.readouterr().err
    assert 'deprecated and frozen' in err and 'python run.py model=se2d/hybrid4' in err
