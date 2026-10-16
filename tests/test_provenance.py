"""Run provenance (mapping/provenance.py; docs/provenance_design.md §2, §3,
§8, §14, §15).

- No run and no checkpoint exists without provenance: ``train.main`` pins
  the code and opens the ledger record before the run directory or any
  step, and every checkpoint goes through ``mapping.checkpoint.save``,
  which needs that live record and stamps the run id into the file.
- The snapshot round-trips: checking out ``refs/runs/<id>`` gives the exact
  working tree, untracked files included, ignored files and the ledger
  left out. HEAD, the branch, the index and ``git status`` are untouched.
- ``experiments/ledger/`` is in neither the snapshot nor the dirty check.
- Concurrent runs never write the same file (run ids, refs, ledger files,
  private indexes, run directories).
- ``run.note`` is required, non-empty, from the launcher only, and is
  metadata: it reaches neither the args record nor any training input.

Snapshot and run tests use throwaway clones of this harness (the suite's
own isolation, tests/conftest.py, is switched off for them), so they test
the real mechanism: refs in the repository, the ledger in its main
worktree.
"""
import ast
import glob
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest
from omegaconf import OmegaConf

import mapping.provenance as provenance
import run
import train

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIT_ID = {'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@t', 'GIT_COMMITTER_NAME': 't',
          'GIT_COMMITTER_EMAIL': 't@t'}
TINY = ['model=z2/vanilla4', 'task.generator.workers=0', 'run.device=cpu', 'run.steps=2',
        'eval.every=2', 'eval.log_every=2', 'eval.batches=1', 'optim.batch=2']


def git(cwd, *args, env=None):
    return subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, check=True,
                          env={**os.environ, **GIT_ID, **(env or {})}).stdout.strip()


@pytest.fixture(scope='module')
def template(tmp_path_factory):
    """A git repository holding this harness as it is on disk (tracked and
    untracked, .gitignore respected; no results, no ledger), one commit."""
    root = tmp_path_factory.mktemp('harness')
    files = subprocess.run(['git', 'ls-files', '-z', '-co', '--exclude-standard'], cwd=HERE,
                           capture_output=True, text=True, check=True).stdout.split('\0')
    for f in files:
        if not f or f.startswith(('experiments/ledger/', 'results/')) or \
                not os.path.isfile(os.path.join(HERE, f)):
            continue
        os.makedirs(os.path.join(root, os.path.dirname(f)), exist_ok=True)
        shutil.copy2(os.path.join(HERE, f), os.path.join(root, f))
    git(root, 'init', '-q', '-b', 'master')
    git(root, 'add', '-A')
    git(root, 'commit', '-q', '-m', 'harness')
    return root


@pytest.fixture
def clone(template, tmp_path, monkeypatch):
    """A fresh clone of the template, with data/ linked in, and the suite's
    provenance isolation off (snapshots and ledger go where a real run's go)."""
    d = tmp_path / 'repo'
    git(tmp_path, 'clone', '-q', str(template), str(d))
    os.makedirs(d / 'data', exist_ok=True)
    for f in glob.glob(os.path.join(HERE, 'data', '*.npz')):
        os.symlink(f, d / 'data' / os.path.basename(f))
    monkeypatch.delenv('XP_SNAPSHOT_GIT_DIR', raising=False)
    monkeypatch.delenv('XP_LEDGER_DIR', raising=False)
    monkeypatch.setenv('XP_HOST', 'test')
    return d


def real_env():
    env = {k: v for k, v in os.environ.items()
           if k not in ('XP_SNAPSHOT_GIT_DIR', 'XP_LEDGER_DIR')}
    return {**env, 'XP_HOST': 'test'}


def state(repo):
    """What a snapshot must not change."""
    gd = git(repo, 'rev-parse', '--absolute-git-dir')
    index = open(os.path.join(gd, 'index'), 'rb').read()
    return (git(repo, 'rev-parse', 'HEAD'), git(repo, 'symbolic-ref', 'HEAD'),
            hashlib.sha1(index).hexdigest(), git(repo, 'status', '--porcelain', '--untracked-files=all'))


def tree_files(repo, rev):
    names = git(repo, 'ls-tree', '-r', '--name-only', '-z', rev).split('\0')
    return {n: subprocess.run(['git', 'show', f'{rev}:{n}'], cwd=repo, capture_output=True,
                              check=True).stdout for n in names if n}


def disk_files(repo):
    names = git(repo, 'ls-files', '-z', '-co', '--exclude-standard').split('\0')
    out = {}
    for n in names:
        if n and not n.startswith('experiments/ledger/') and os.path.isfile(os.path.join(repo, n)):
            out[n] = open(os.path.join(repo, n), 'rb').read()
    return out


# ---------------------------------------------------------------- snapshots

def test_snapshot_round_trips_the_exact_tree(clone):
    (clone / 'run.py').write_text((clone / 'run.py').read_text() + '\n# an edit\n')
    os.makedirs(clone / 'notes')
    (clone / 'notes' / 'new.txt').write_text('untracked, not ignored\n')
    os.makedirs(clone / 'results', exist_ok=True)
    (clone / 'results' / 'ignored.txt').write_text('ignored\n')
    os.makedirs(clone / 'experiments' / 'ledger' / 'test')
    (clone / 'experiments' / 'ledger' / 'test' / 'other.json').write_text('{}')
    before = state(clone)
    code = provenance.snapshot('t1', str(clone))
    assert state(clone) == before                    # HEAD, branch, index, status untouched
    assert code['dirty'] and code['base'] == before[0] and code['base_branch'] == 'master'
    assert sorted(code['dirty_paths']) == ['notes/new.txt', 'run.py']
    assert git(clone, 'rev-parse', 'refs/runs/t1') == code['snapshot']
    assert git(clone, 'rev-parse', code['snapshot'] + '^') == code['base']
    snap = tree_files(clone, code['snapshot'])
    assert snap == disk_files(clone)                 # byte for byte, untracked included
    assert 'results/ignored.txt' not in snap
    assert not any(n.startswith('experiments/ledger/') for n in snap)
    # a checkout of the ref reproduces the tree
    wt = clone.parent / 'wt'
    git(clone, 'worktree', 'add', '-q', '--detach', str(wt), 'refs/runs/t1')
    assert open(wt / 'notes' / 'new.txt').read() == 'untracked, not ignored\n'
    assert open(wt / 'run.py').read().endswith('# an edit\n')


def test_a_clean_tree_is_its_head_and_the_ledger_never_makes_it_dirty(clone):
    head = git(clone, 'rev-parse', 'HEAD')
    os.makedirs(clone / 'experiments' / 'ledger' / 'test')
    (clone / 'experiments' / 'ledger' / 'test' / 'x.json').write_text('{"a": 1}')
    code = provenance.snapshot('t2', str(clone))
    assert (code['dirty'], code['snapshot'], code['dirty_count']) == (False, head, 0)
    # a committed ledger file, then changed on disk: still clean, and left out
    git(clone, 'add', 'experiments/ledger')
    git(clone, 'commit', '-q', '-m', 'ledger')
    (clone / 'experiments' / 'ledger' / 'test' / 'x.json').write_text('{"a": 2}')
    (clone / 'experiments' / 'ledger' / 'test' / 'y.json').write_text('{}')
    code = provenance.snapshot('t3', str(clone))
    assert code['dirty'] is False and code['snapshot'] == git(clone, 'rev-parse', 'HEAD')
    # a dirty snapshot carries no ledger file at all, even a committed one
    (clone / 'README.md').write_text('changed\n')
    code = provenance.snapshot('t4', str(clone))
    assert code['dirty'] and not any(n.startswith('experiments/ledger')
                                     for n in tree_files(clone, code['snapshot']))


def test_refs_are_immutable_and_refusals_are_loud(clone, tmp_path):
    provenance.snapshot('same', str(clone))
    with pytest.raises(SystemExit, match='git failed .*update-ref'):
        provenance.snapshot('same', str(clone))
    plain = tmp_path / 'plain'
    os.makedirs(plain)
    with pytest.raises(SystemExit, match='not a git checkout'):
        provenance.snapshot('x', str(plain))
    git(plain, 'init', '-q')
    with pytest.raises(SystemExit, match='no HEAD commit'):
        provenance.snapshot('x', str(plain))


def test_an_untracked_file_over_the_size_guard_is_refused(clone):
    (clone / 'dump.bin').write_bytes(os.urandom(2 * 2 ** 20 + 1))
    objects = git(clone, 'count-objects', '-v')
    with pytest.raises(SystemExit, match=r'dump.bin \(2.0 MB\)'):
        provenance.snapshot('big', str(clone), max_untracked_mb=2)
    assert git(clone, 'for-each-ref', 'refs/runs') == ''
    assert git(clone, 'count-objects', '-v') == objects     # nothing written
    assert provenance.snapshot('ok', str(clone), max_untracked_mb=3)['dirty']   # no strict mode


def test_concurrent_snapshots_never_share_a_file(clone):
    code = ('import json, sys; sys.path.insert(0, sys.argv[1]); import mapping.provenance as p; '
            'rid = p.new_run_id(); print(json.dumps([rid, p.snapshot(rid, sys.argv[1])]))')
    (clone / 'scratch.txt').write_text('dirty, so each snapshot is a new commit\n')
    procs = [subprocess.Popen([sys.executable, '-c', code, str(clone)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, env=real_env())
             for _ in range(8)]
    outs = [p.communicate() for p in procs]
    assert all(p.returncode == 0 for p in procs), [e for _, e in outs]
    got = [json.loads(o) for o, _ in outs]
    ids = [rid for rid, _ in got]
    assert len(set(ids)) == 8
    trees = {git(clone, 'rev-parse', c['snapshot'] + '^{tree}') for _, c in got}
    assert len(trees) == 1                            # the same bytes, eight runs
    assert sorted(git(clone, 'for-each-ref', '--format=%(refname)', 'refs/runs').split()) == \
        sorted(f'refs/runs/{i}' for i in ids)
    gd = git(clone, 'rev-parse', '--absolute-git-dir')
    assert not glob.glob(os.path.join(gd, 'xp-index-*'))   # private indexes cleaned up


# ---------------------------------------------------------------- whole runs

def launch(repo, *argv, cwd=None):
    return subprocess.run([sys.executable, os.path.join(repo, 'run.py'), *argv],
                          cwd=cwd or repo, capture_output=True, text=True, env=real_env())


def ledger_of(repo, run_dir):
    prov = json.load(open(os.path.join(run_dir, 'provenance.json')))
    path = os.path.join(repo, 'experiments', 'ledger', prov['host'], prov['run_id'] + '.json')
    return prov, json.load(open(path))


SCHEMA = {'schema': int, 'run_id': str, 'host': str, 'status': str, 'created': str,
          'updated': str, 'code': dict, 'entry': dict, 'config': dict, 'run': dict,
          'results': dict, 'parent': dict, 'env': dict, 'data': dict, 'hardware': dict,
          'progress': dict}
CODE_KEYS = {'snapshot', 'ref', 'base', 'base_branch', 'dirty', 'dirty_paths', 'dirty_count',
             'worktree'}


def test_a_run_is_pinned_and_recorded_and_every_checkpoint_leads_back(clone):
    before = state(clone)
    r = launch(clone, *TINY, 'run.save_every=1', 'run.name=a', "run.note='  first run, test  '")
    assert r.returncode == 0, r.stderr[-3000:]
    d = clone / 'results' / 'z2' / 'vanilla4' / 'xs' / 'a'
    prov, rec = ledger_of(clone, d)
    assert prov == rec                                   # provenance.json mirrors the ledger
    for k, t in SCHEMA.items():
        assert isinstance(rec[k], t), k
    assert CODE_KEYS <= set(rec['code'])
    assert rec['schema'] == 1 and rec['host'] == 'test' and rec['status'] == 'completed'
    assert rec['run']['note'] == 'first run, test'       # stripped
    assert rec['run_id'].endswith(rec['run_id'][-6:]) and '-test-' in rec['run_id']
    assert (d / 'RUN_ID').read_text().strip() == rec['run_id']
    assert rec['code']['dirty'] is False and rec['code']['snapshot'] == before[0]
    assert git(clone, 'rev-parse', f'refs/runs/{rec["run_id"]}') == rec['code']['snapshot']
    final = json.load(open(d / 'final.json'))
    assert rec['end']['exit_status'] == 0
    assert rec['end']['final_metrics']['mse'] == final['final_eval']['mse']
    assert rec['results'] == {'dir': str(d), 'root': str(clone / 'results'),
                              'rel': 'z2/vanilla4/xs/a'}
    assert rec['data']['data/mnist.npz']['bytes'] == os.path.getsize(clone / 'data' / 'mnist.npz')
    assert rec['config']['resolved']['run']['note'] == 'first run, test'
    assert hashlib.sha256((d / 'pip_freeze.txt').read_bytes()).hexdigest() == \
        rec['env']['pip_freeze_sha256']
    import torch
    pts = sorted(glob.glob(str(d / '*.pt')))
    assert [os.path.basename(p) for p in pts] == ['final.pt', 'step1.pt']
    for p in pts:
        stamp = torch.load(p, map_location='cpu', weights_only=False)['provenance']
        assert stamp == {'run_id': rec['run_id'], 'snapshot': rec['code']['snapshot']}
        git(clone, 'cat-file', '-e', stamp['snapshot'] + '^{commit}')
    assert sorted(rec['progress']['checkpoints']) == ['final.pt', 'step1.pt']
    # the checkout is as it was, apart from the new (untracked) ledger record
    after = state(clone)
    assert after[:3] == before[:3]
    assert after[3] == f'?? experiments/ledger/test/{rec["run_id"]}.json'
    # a second run, now with that ledger file on disk, is still clean
    r = launch(clone, *TINY, 'run.name=b', 'run.note=test')
    assert r.returncode == 0, r.stderr[-3000:]
    assert ledger_of(clone, clone / 'results' / 'z2' / 'vanilla4' / 'xs' / 'b')[1]['code'][
        'dirty'] is False


def test_a_run_from_a_linked_worktree_records_into_the_main_ledger(clone):
    wt = clone.parent / 'linked'
    git(clone, 'worktree', 'add', '-q', '--detach', str(wt), 'HEAD')
    os.symlink(clone / 'data', wt / 'data')      # ignored? no: a symlink named data is untracked
    r = launch(wt, *TINY, 'run.name=w', 'run.note=test')
    assert r.returncode == 0, r.stderr[-3000:]
    prov = json.load(open(wt / 'results' / 'z2' / 'vanilla4' / 'xs' / 'w' / 'provenance.json'))
    assert os.path.exists(clone / 'experiments' / 'ledger' / 'test' / f'{prov["run_id"]}.json')
    assert not os.path.exists(wt / 'experiments' / 'ledger')
    assert prov['code']['worktree'] == str(wt)
    assert prov['code']['dirty'] and prov['code']['dirty_paths'] == ['data']   # the link, recorded


def test_concurrent_runs_never_write_the_same_file(clone):
    def start(*argv):
        return subprocess.Popen([sys.executable, 'run.py', *TINY, 'run.note=test', *argv],
                                cwd=clone, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, env=real_env())
    distinct = [start(f'run.name=c{i}') for i in range(3)]
    same = [start(f'run.out={clone}/shared') for _ in range(2)]
    for p in distinct:
        p.communicate()
    outs_same = [p.communicate() for p in same]
    assert all(p.returncode == 0 for p in distinct)
    assert sorted(p.returncode for p in same) == [0, 1]      # one claims it, the other is refused
    ids = [(clone / 'results' / 'z2' / 'vanilla4' / 'xs' / f'c{i}' / 'RUN_ID').read_text().strip()
           for i in range(3)]
    ledger = sorted(os.listdir(clone / 'experiments' / 'ledger' / 'test'))
    recs = [json.load(open(clone / 'experiments' / 'ledger' / 'test' / f)) for f in ledger]
    assert len(set(ids)) == 3 and set(ids) <= {r['run_id'] for r in recs}
    assert len(ledger) == len({r['run_id'] for r in recs})    # one file per run id
    shared = [r for r in recs if r['results']['dir'] == f'{clone}/shared']
    winner = [r for r in shared if r['status'] == 'completed']
    assert len(winner) == 1
    assert (clone / 'shared' / 'RUN_ID').read_text().strip() == winner[0]['run_id']
    # the other one was refused, either at the non-empty check (before any
    # record) or at the RUN_ID claim (its record ends as failed, saying why)
    for r in shared:
        if r is not winner[0]:
            assert r['status'] == 'failed' and 'is claimed by run' in r['end']['error']
    loser_err = next(e for p, (_, e) in zip(same, outs_same) if p.returncode)
    assert 'is claimed by run' in loser_err or 'is not empty' in loser_err


def test_sigterm_ends_the_record_as_terminated(clone):
    long = [a for a in TINY if not a.startswith('run.steps=')] + ['run.steps=1000000']
    p = subprocess.Popen([sys.executable, 'run.py', *long, 'run.name=t',
                          'run.note=test'], cwd=clone, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, env=real_env())
    d = clone / 'results' / 'z2' / 'vanilla4' / 'xs' / 't'
    deadline = time.time() + 120
    while not (d / 'log.jsonl').exists() and time.time() < deadline:
        time.sleep(0.2)
    time.sleep(1.0)
    p.send_signal(signal.SIGTERM)
    p.communicate(timeout=120)
    assert p.returncode == 143
    _, rec = ledger_of(clone, d)
    assert (rec['status'], rec['end']['exit_status'], rec['end']['error']) == (
        'terminated', 143, 'SIGTERM')


# ---------------------------------------------------------------- in-process

def test_no_run_and_no_checkpoint_without_provenance(tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise SystemExit('provenance refused')
    monkeypatch.setattr(provenance, 'begin', refuse)
    steps = []
    monkeypatch.setattr(train, 'call', lambda *a, **k: steps.append(1))
    with pytest.raises(SystemExit, match='provenance refused'):
        run.main([*TINY, f'run.out={tmp_path}/r', 'run.save_every=1', 'run.note=test'])
    assert not os.path.exists(tmp_path / 'r') and steps == []


def test_a_failed_run_is_recorded_as_failed(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError('evaluation exploded')
    monkeypatch.setattr(train, 'evaluate', boom)
    with pytest.raises(RuntimeError):
        run.main([*TINY, f'run.out={tmp_path}/r', 'run.note=test'])
    rec = json.load(open(tmp_path / 'r' / 'provenance.json'))
    assert rec['status'] == 'failed' and rec['end']['error'] == 'RuntimeError: evaluation exploded'
    ledger = os.path.join(os.environ['XP_LEDGER_DIR'], rec['host'], rec['run_id'] + '.json')
    assert json.load(open(ledger)) == rec


def test_the_checkpoint_writer_needs_a_live_record(tmp_path):
    from mapping.checkpoint import save
    with pytest.raises(RuntimeError, match='live provenance record'):
        save(None, str(tmp_path / 'x.pt'), {'step': 1})
    assert not os.path.exists(tmp_path / 'x.pt')


def test_only_the_checkpoint_writer_calls_torch_save():
    """In the training path (run.py, train.py, mapping/), ``torch.save``
    is called only inside ``mapping.checkpoint.save``."""
    found = []
    for path in ['run.py', 'train.py', *glob.glob('mapping/**/*.py', recursive=True)]:
        tree = ast.parse(open(os.path.join(HERE, path)).read())

        def visit(node, owner):
            for child in ast.iter_child_nodes(node):
                inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    else owner
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and \
                        child.func.attr == 'save' and isinstance(child.func.value, ast.Name) and \
                        child.func.value.id == 'torch':
                    found.append((path, owner))
                visit(child, inner)
        visit(tree, '<module>')
    assert found == [('mapping/checkpoint.py', 'save')], found


# ---------------------------------------------------------------- the note

def test_the_note_is_required_and_given_at_launch(tmp_path):
    for bad in ([], ['run.note='], ["run.note='   '"]):
        with pytest.raises(SystemExit, match='run.note is required'):
            run.main([*TINY, f'run.out={tmp_path}/r', *bad])
    assert not os.path.exists(tmp_path / 'r')
    exp = tmp_path / 'e.yaml'
    exp.write_text('# @package _global_\nrun:\n  note: from a file\n')
    with pytest.raises(SystemExit, match='comes from a file'):
        run.main([*TINY, f'run.out={tmp_path}/r', '--exp', str(exp)])
    with pytest.raises(SystemExit, match='run.note is required'):
        train.main_from_flags(['--model', 'vanilla4', '--size', 'xs', '--device', 'cpu',
                               '--out', str(tmp_path / 'r')])
    assert not os.path.exists(tmp_path / 'r')


def test_no_file_under_conf_sets_the_note():
    """The note always comes from the launcher: conf/config.yaml declares
    ``run.note: null`` and no other file under conf/ mentions it."""
    import yaml
    hits = []
    for f in glob.glob(os.path.join(HERE, 'conf', '**', '*.yaml'), recursive=True):
        doc = yaml.safe_load(open(f)) or {}
        node = doc.get('run') if isinstance(doc, dict) else None
        if isinstance(node, dict) and 'note' in node:
            hits.append((os.path.relpath(f, HERE), node['note']))
    assert hits == [('conf/config.yaml', None)]


def test_the_note_and_tags_are_metadata_only(tmp_path):
    """Changing the note or the tags changes no training input and not the
    args record, and two runs differing only in them write identical
    logs, finals and checkpoint tensors."""
    from tests.test_config import trainer_inputs
    import torch
    a = run.build(run.resolved(run.load([*TINY, 'run.note=one'])))
    b = run.build(run.resolved(run.load([*TINY, 'run.note=two', 'run.tag=[x,y]'])))
    assert vars(a.legacy_args()) == vars(b.legacy_args())
    ia, ib = trainer_inputs(a), trainer_inputs(b)
    assert ia[:5] == ib[:5] and ia[6:] == ib[6:]          # all but asdict(run)
    ra, rb = dict(ia[5]), dict(ib[5])
    for k in ('note', 'tag'):
        ra.pop(k), rb.pop(k)
    assert ra == rb
    for name, note in (('one', 'one'), ('two', 'something else entirely')):
        run.main([*TINY, f'run.out={tmp_path}/{name}', 'run.save_every=1', f"run.note='{note}'"])

    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items()            # timing, the output path
                    if k not in ('elapsed_s', 'steps_per_s', 'out')}
        return x
    for f in ('log.jsonl',):
        la = [strip(json.loads(ln)) for ln in open(tmp_path / 'one' / f)]
        lb = [strip(json.loads(ln)) for ln in open(tmp_path / 'two' / f)]
        assert la == lb
    assert strip(json.load(open(tmp_path / 'one' / 'final.json'))) == \
        strip(json.load(open(tmp_path / 'two' / 'final.json')))
    for pt in ('step1.pt', 'final.pt'):
        x = torch.load(tmp_path / 'one' / pt, weights_only=False)
        y = torch.load(tmp_path / 'two' / pt, weights_only=False)
        assert x.pop('provenance') != y.pop('provenance')
        assert strip(x['args']) == strip(y['args'])
        assert all(torch.equal(x['model'][k], y['model'][k]) for k in x['model'])
    assert (tmp_path / 'one' / 'grid.png').read_bytes() == (tmp_path / 'two' / 'grid.png').read_bytes()


def test_a_rerun_takes_a_new_note_not_the_saved_one(tmp_path):
    run.main([*TINY, f'run.out={tmp_path}/r', 'run.note=original'])
    saved = OmegaConf.load(tmp_path / 'r' / 'overrides.yaml')
    assert not any(o.startswith('run.note') for o in saved.overrides)
    assert OmegaConf.load(tmp_path / 'r' / 'config.yaml').run.note == 'original'
    with pytest.raises(SystemExit, match='run.note is required'):
        run.main(['--from', str(tmp_path / 'r' / 'config.yaml'), f'run.out={tmp_path}/r2'])
    run.main(['--from', str(tmp_path / 'r' / 'config.yaml'), f'run.out={tmp_path}/r2',
              'run.note=again'])
    assert json.load(open(tmp_path / 'r2' / 'provenance.json'))['run']['note'] == 'again'
