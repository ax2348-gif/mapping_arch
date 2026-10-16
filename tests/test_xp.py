"""The xp porcelain (mapping/xp/; docs/provenance_design.md §9, §15).

Each subcommand runs as a user would run it (``python -m mapping.xp ...``,
the same entry as the ``xp`` console script) in a throwaway clone of this
harness with real provenance (tests/test_provenance.py's fixtures): ``ls``
(newest first, filters, retroactive records marked, ``--json``), ``show``,
``compare``, ``run`` (a note is required; it launches run.py), a real tiny
CPU ``reproduce`` whose verdict must be ``bitwise``, ``sync`` (one tidy
ledger commit by explicit path, idempotent) and ``tag``. xp never imports
the trainer or torch: run.py stays the only training entry.
"""
import ast
import glob
import json
import os
import subprocess
import sys

import pytest

from tests.test_provenance import TINY, clone, git, real_env, template  # noqa: F401 (fixtures)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY_SAVE = [*TINY, 'run.save_every=1']


def xp(repo, *args, check=True):
    r = subprocess.run([sys.executable, '-m', 'mapping.xp', *args], cwd=repo,
                       capture_output=True, text=True, env=real_env())
    if check:
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return r


def launch(repo, *argv):
    r = subprocess.run([sys.executable, 'run.py', *argv], cwd=repo, capture_output=True,
                       text=True, env=real_env())
    assert r.returncode == 0, r.stderr[-3000:]
    d = next(ln.split(' -> ')[1].split(' (')[0] for ln in r.stdout.splitlines() if ' -> ' in ln)
    return json.load(open(os.path.join(repo, d, 'provenance.json')))


def retro_record(repo):
    """A retroactive record, as tools/import_retro.py writes them."""
    rec = {'schema': 1, 'run_id': 'legacy-erid-2-z2-mlp-xs-old', 'host': 'erid-2',
           'retroactive': True, 'status': 'completed', 'created': '2020-01-01T00:00:00Z',
           'code': {'snapshot': None, 'ref': None, 'commit': 'inferred:abc1234',
                    'basis': 'a test', 'dirty': None},
           'config': {'resolved': None}, 'run': {'task': 'z2', 'model': 'mlp', 'size': 'xs',
                                                 'note': 'an old run', 'tags': ['major', 'old']},
           'results': {'dir': '/nowhere/results/z2/mlp/xs/old', 'rel': 'z2/mlp/xs/old'},
           'end': {'final_metrics': {'mse': 0.05}}}
    path = os.path.join(repo, 'experiments', 'ledger', 'erid-2', rec['run_id'] + '.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(rec, open(path, 'w'))
    return rec


def test_ls_newest_first_with_filters_and_retro_marked(clone):
    first = launch(clone, *TINY, 'run.name=a', "run.note='the first run'", 'run.tag=[smoke]')
    second = launch(clone, *TINY, 'run.name=b', "run.note='the second run, newer'")
    retro_record(clone)
    out = xp(clone, 'ls').stdout.splitlines()
    assert out[0].split()[:4] == ['START', 'STATUS', 'NOTE', 'PATH']
    body = out[1:-1]
    assert [ln.split()[-1] for ln in body] == [second['run_id'][-6:], first['run_id'][-6:], 'retro']
    assert 'the second run, newer' in body[0] and 'z2/vanilla4/xs/b' in body[0]
    assert ' done ' in body[0] and ' test ' in body[0]
    assert 'smoke' in body[1] and f'{first["end"]["final_metrics"]["mse"]:.4g}' in body[1]
    assert 'done*' in body[2] and 'retroactive' in out[-1] and '3 of 3 runs' in out[-1]
    assert len(xp(clone, 'ls', '--latest', '1').stdout.splitlines()) == 3
    assert [r['run_id'] for r in json.loads(xp(clone, 'ls', '--json').stdout)] == \
        [second['run_id'], first['run_id'], 'legacy-erid-2-z2-mlp-xs-old']
    ids = lambda *f: [r['run_id'] for r in json.loads(xp(clone, 'ls', '--json', *f).stdout)]  # noqa
    assert ids('--tag', 'smoke') == [first['run_id']]
    assert ids('--tag', 'old') == ['legacy-erid-2-z2-mlp-xs-old']
    assert ids('--since', '2d') == [second['run_id'], first['run_id']]
    assert ids('--host', 'erid-2') == ['legacy-erid-2-z2-mlp-xs-old']
    assert ids('--model', 'mlp', '--task', 'z2') == ['legacy-erid-2-z2-mlp-xs-old']
    assert ids('--running') == []


def test_show_and_compare(clone):
    a = launch(clone, *TINY_SAVE, 'run.name=a', 'run.note=one')
    (clone / 'README.md').write_text('a change\n')                      # a dirty snapshot
    b = launch(clone, *TINY, 'run.name=b', 'run.seed=1', 'run.note=two')
    out = xp(clone, 'show', a['run_id'][-6:]).stdout
    assert a['run_id'] in out and 'note     one' in out and 'clean' in out
    assert 'checkpoints: final.pt, step1.pt' in out
    j = json.loads(xp(clone, 'show', 'z2/vanilla4/xs/b', '--json').stdout)
    assert j['record']['run_id'] == b['run_id'] and j['record']['code']['dirty']
    assert 'README.md' in j['dirty_diff_stat'] and j['snapshot_stat'].startswith(
        b['code']['snapshot'])
    c = json.loads(xp(clone, 'compare', a['run_id'], b['run_id'], '--json').stdout)
    assert [m['run_id'] for m in c['metrics']] == [a['run_id'], b['run_id']]
    diff = c['config'][0]['differs']
    assert diff['run.seed'] == [0, 1] and 'run.note' not in diff
    assert c['code'][0]['same'] is False and 'README.md' in c['code'][0]['stat']
    assert 'run.seed: 0 -> 1' in xp(clone, 'compare', a['run_id'], b['run_id']).stdout
    assert xp(clone, 'show', 'nope', check=False).returncode != 0


def test_run_needs_a_note_and_launches_run_py(clone):
    r = xp(clone, 'run', *TINY, check=False)
    assert r.returncode != 0 and 'run.note is required' in r.stderr
    r = xp(clone, 'run', *TINY, 'run.name=x', '-m', "it's a test, with: punctuation",
           '--tag', 'smoke', '--explain')
    assert '+ cd ' in r.stderr and 'run.py' in r.stderr
    rec = json.load(open(clone / 'results' / 'z2' / 'vanilla4' / 'xs' / 'x' / 'provenance.json'))
    assert rec['run']['note'] == "it's a test, with: punctuation"
    assert rec['run']['tags'] == ['smoke'] and rec['entry']['via'] == 'xp run'
    assert rec['entry']['argv'][0].endswith('run.py')
    r = xp(clone, 'run', *TINY, 'run.name=y', '-m', 'dry', '--dry-run')
    assert 'run.py' in r.stderr and not os.path.exists(clone / 'results' / 'z2' / 'vanilla4' /
                                                       'xs' / 'y')


def test_reproduce_a_cpu_run_bit_for_bit(clone):
    orig = launch(clone, *TINY_SAVE, 'run.name=orig', "run.note='to be reproduced'")
    d = clone / 'results' / 'z2' / 'vanilla4' / 'xs' / 'orig'
    r = xp(clone, 'reproduce', orig['run_id'][-6:])
    assert 'VERDICT: BITWISE' in r.stdout
    reps = glob.glob(str(d / 'reproductions' / '*'))
    assert len(reps) == 1
    rep = reps[0]
    report = json.load(open(os.path.join(rep, 'reproduce_report.json')))
    assert report['verdict'] == 'bitwise' and report['checkpoint_bitwise'] is True
    assert report['code'] == {'original': orig['code']['snapshot'],
                              'reproduction': orig['code']['snapshot'], 'same': True}
    assert report['config_diff'] == [] and report['data_diff'] == {}
    assert all(m['abs_diff'] == 0 for m in report['metrics'].values())
    new = json.load(open(os.path.join(rep, 'provenance.json')))
    assert os.path.basename(rep) == new['run_id'].split('-')[0] + '_' + new['run_id']
    assert new['reproduces'] == orig['run_id'] and new['run']['note'] == \
        f'reproduce {orig["run_id"]}'
    assert new['entry']['via'] == 'xp reproduce' and new['code']['dirty'] is False
    assert os.path.exists(clone / 'experiments' / 'ledger' / 'test' / f'{new["run_id"]}.json')
    # the worktree is gone; show links the two
    assert git(clone, 'worktree', 'list').count('\n') == 0
    assert f'reproduced by {new["run_id"]}' in xp(clone, 'show', orig['run_id']).stdout
    # aggregation never counts a reproduction as a run
    from mapping.results import load_runs
    from mapping.provenance import iter_run_dirs
    size_dir = str(clone / 'results' / 'z2' / 'vanilla4' / 'xs')
    assert glob.glob(os.path.join(size_dir, '*', '*', '*', 'final.json'))   # the glob would see it
    assert load_runs(size_dir) == []
    assert list(iter_run_dirs(str(clone / 'results'))) == [str(d)]
    r = subprocess.run([sys.executable, 'summarize.py', '--root', size_dir], cwd=clone,
                       capture_output=True, text=True)
    assert r.stdout.strip() == 'no runs found'
    # a retroactive record has no snapshot: refused, not approximated
    retro_record(clone)
    r = xp(clone, 'reproduce', 'legacy-erid-2-z2-mlp-xs-old', check=False)
    assert r.returncode != 0 and 'no code snapshot' in r.stderr


def test_sync_commits_the_ledger_only_and_is_idempotent(clone):
    launch(clone, *TINY, 'run.name=a', 'run.note=one')
    launch(clone, *TINY, 'run.name=b', 'run.note=two')
    (clone / 'README.md').write_text('staged, not ledger\n')
    git(clone, 'add', 'README.md')
    head = git(clone, 'rev-parse', 'HEAD')
    r = xp(clone, 'sync', '--dry-run', '--explain')
    assert 'git -C' in r.stderr and 'add' in r.stderr and git(clone, 'rev-parse', 'HEAD') == head
    out = json.loads(xp(clone, 'sync', '--json').stdout)
    assert out['added'] == 2 and out['message'] == 'ledger: sync (test +2 ~0)'
    files = git(clone, 'show', '--name-only', '--format=', 'HEAD').split()
    assert len(files) == 2 and all(f.startswith('experiments/ledger/test/') for f in files)
    assert git(clone, 'diff', '--cached', '--name-only') == 'README.md'   # still staged, not swept in
    assert 'nothing to sync' in xp(clone, 'sync').stdout
    assert git(clone, 'rev-list', '--count', f'{head}..HEAD') == '1'


def test_tag_curates_the_registry(clone):
    rec = launch(clone, *TINY, 'run.name=a', 'run.note=one')
    xp(clone, 'tag', rec['run_id'], 'major', '--note', 'the reference')
    import yaml
    reg = yaml.safe_load(open(clone / 'experiments' / 'registry.yaml'))
    assert reg['tagged'][-1]['id'] == rec['run_id'] and reg['tagged'][-1]['tag'] == 'major'
    assert 'already tagged' in xp(clone, 'tag', rec['run_id'][-6:], 'major').stdout
    assert len(yaml.safe_load(open(clone / 'experiments' / 'registry.yaml'))['tagged']) == 1
    ids = [r['run_id'] for r in json.loads(xp(clone, 'ls', '--json', '--tag', 'major').stdout)]
    assert rec['run_id'] in ids                     # launch tags and registry tags, unioned
    ledger = json.load(open(clone / 'experiments' / 'ledger' / 'test' / f'{rec["run_id"]}.json'))
    assert ledger['run']['tags'] == []              # the ledger is written only by its run


def test_xp_never_imports_the_trainer_or_torch():
    for path in glob.glob(os.path.join(HERE, 'mapping', 'xp', '*.py')):
        tree = ast.parse(open(path).read())
        for n in ast.walk(tree):
            names = ([a.name for a in n.names] if isinstance(n, ast.Import) else
                     [n.module or ''] if isinstance(n, ast.ImportFrom) else [])
            for name in names:
                assert name.split('.')[0] not in ('torch', 'train', 'run'), (path, name)
                assert name not in ('mapping.config', 'mapping.models'), (path, name)
    r = subprocess.run([sys.executable, '-c', 'import sys, mapping.xp.cli; '
                        'print("torch" in sys.modules)'], cwd=HERE, capture_output=True, text=True)
    assert r.stdout.strip() == 'False', r.stderr
