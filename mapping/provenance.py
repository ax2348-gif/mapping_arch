"""Run provenance: the exact code of every run, and one ledger record per run.

docs/provenance_design.md (§1-§3, the user's decisions in §14, what was
built in §15). ``train.main`` calls ``begin`` once everything is checked
and before the run directory or any training step exists, so no run, and
no checkpoint, exists without:

- **a snapshot commit** of the exact working tree, at ``refs/runs/<run_id>``.
  Tracked and untracked files go in (``.gitignore`` respected), through a
  private index (``GIT_INDEX_FILE``): the branch, ``HEAD`` and the real
  index are never touched. ``experiments/ledger/`` is left out of the
  snapshot's content and of the dirty check. A clean tree's snapshot is
  ``HEAD`` itself. An untracked file over ``run.provenance.max_untracked_mb``
  (20 MB) is refused, so no stray dump lands in ``.git/objects`` for good.
  A run is refused only when no snapshot can be made; a dirty tree is fine
  (there is no strict mode: the snapshot holds the exact bytes).
- **a ledger record**, ``experiments/ledger/<host>/<run_id>.json`` in the
  main worktree (schema 1, §3), mirrored as ``provenance.json`` in the run
  directory. It is written atomically at the start, at every checkpoint and
  at the end (completed, failed with the error, or terminated by SIGTERM),
  only by its run. Jobs never commit it; ``xp sync`` does.
- **``run.note``**, required and non-empty, like ``git commit -m``. It comes
  from the launcher only, never from a preset (``check_note``).

Environment (all optional):
    XP_HOST               the logical host (``torch``); else conf/hosts.yaml
    XP_LEDGER_DIR         where ledger records go (default: main worktree)
    XP_VIA                what launched run.py (``xp run``, ``xp reproduce``)
    XP_REPRODUCES         the run id a run reproduces (``xp reproduce``)
    XP_SUBMIT_ID          the submission that started the job (``xp submit``)
    XP_SNAPSHOT_GIT_DIR   tests only: a git dir (borrowing this repository's
                          objects) that receives snapshot objects and refs,
                          so a test suite run leaves this repository's refs
                          alone. tests/conftest.py sets it; tests of the
                          snapshot itself use throwaway repositories.
"""
import contextlib
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys

SCHEMA = 1
LEDGER = os.path.join('experiments', 'ledger')
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_LEDGER = ':(exclude)experiments/ledger'
IDENTITY = {'GIT_AUTHOR_NAME': 'run.py', 'GIT_AUTHOR_EMAIL': 'run@localhost',
            'GIT_COMMITTER_NAME': 'run.py', 'GIT_COMMITTER_EMAIL': 'run@localhost'}
MAX_DIRTY_PATHS = 50

NOTE_HELP = (
    'run.note is required: one line saying what this run is for, like a commit message '
    '(one word is enough). Give it on the command line, quoted for Hydra when it has '
    "spaces or commas: python run.py <overrides> \"run.note='baseline, seed 2'\", or "
    "xp run <preset> -m 'baseline, seed 2' (train.py --flags: --note '...'). It never "
    'comes from a preset or an experiment file.')


class Terminated(BaseException):
    """SIGTERM arrived (Slurm sends it before SIGKILL)."""


def check_note(note):
    """The run note, whitespace stripped; refused when missing or empty."""
    if note is None or not isinstance(note, str) or not note.strip():
        raise SystemExit(NOTE_HELP)
    return note.strip()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def iso(t):
    return t.strftime('%Y-%m-%dT%H:%M:%SZ')


def git(args, cwd=None, env=None, check=True):
    r = subprocess.run(['git', *args], cwd=cwd, env=env, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise subprocess.CalledProcessError(r.returncode, ['git', *args], r.stdout, r.stderr)
    return r.stdout.strip() if check else r


# ---------------------------------------------------------------- identity

def hosts_table():
    """conf/hosts.yaml: logical hosts and the machine names that map to them."""
    path = os.path.join(CODE_ROOT, 'conf', 'hosts.yaml')
    if not os.path.exists(path):
        return {}
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(path))


def host():
    """The logical host: ``XP_HOST``, else the machine name mapped through
    conf/hosts.yaml (``machines``: name or glob -> host), else the machine
    name. One ledger directory per logical host keeps one writer per
    directory."""
    import fnmatch
    name = os.environ.get('XP_HOST')
    if not name:
        machine = socket.gethostname().split('.')[0]
        name = machine
        for pattern, logical in (hosts_table().get('machines') or {}).items():
            if fnmatch.fnmatch(machine, pattern):
                name = logical
                break
    name = re.sub(r'[^A-Za-z0-9_.-]', '-', name).strip('.-') or 'host'
    return name


def new_run_id(host_name=None, now=None):
    """``<UTC yyyymmddTHHMMSSZ>-<host>-<6 hex>``: sorts by time; the random
    part (os.urandom) keeps runs started in the same second apart."""
    now = now or utc_now()
    return f'{now:%Y%m%dT%H%M%SZ}-{host_name or host()}-{os.urandom(3).hex()}'


# ---------------------------------------------------------------- snapshot

def refuse(message):
    raise SystemExit(f'provenance: {message}. No run was started (a run must be pinned to '
                     f'an exact snapshot of its code; docs/provenance_design.md §2)')


def _rm_ledger(env, top):
    git(['rm', '-r', '--cached', '-q', '--ignore-unmatch', '--', LEDGER.replace(os.sep, '/')],
        cwd=top, env=env)


def snapshot(run_id, top=CODE_ROOT, max_untracked_mb=20):
    """Commit the working tree at ``top`` to ``refs/runs/<run_id>`` without
    touching the branch, HEAD or the index. Returns the ledger's ``code``
    block. Refuses (SystemExit) only when no snapshot can be made."""
    try:
        top = git(['rev-parse', '--show-toplevel'], cwd=top)
        real_git_dir = git(['rev-parse', '--absolute-git-dir'], cwd=top)
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        refuse(f'{top} is not a git checkout')
    try:
        base = git(['rev-parse', '--verify', '-q', 'HEAD^{commit}'], cwd=top)
    except subprocess.CalledProcessError:
        refuse(f'{top} has no HEAD commit yet')
    branch = git(['symbolic-ref', '-q', '--short', 'HEAD'], cwd=top, check=False).stdout.strip() or None
    store = os.environ.get('XP_SNAPSHOT_GIT_DIR')
    gd = os.path.abspath(store) if store else real_git_dir

    # the size guard, before anything is written into the object store
    out = git(['ls-files', '-z', '--others', '--exclude-standard', '--', '.', EXCLUDE_LEDGER],
              cwd=top)
    limit = max_untracked_mb * 1024 * 1024
    big = []
    for p in filter(None, out.split('\0')):
        f = os.path.join(top, p)
        if os.path.isfile(f) and not os.path.islink(f) and os.path.getsize(f) > limit:
            big.append(f'{p} ({os.path.getsize(f) / 2 ** 20:.1f} MB)')
    if big:
        refuse(f'untracked files over run.provenance.max_untracked_mb={max_untracked_mb} would '
               f'be committed into the snapshot: {", ".join(big)}. Add them to .gitignore, move '
               f'them out of the repository, or raise the limit')

    idx = os.path.join(gd, f'xp-index-{run_id}')
    base_idx = idx + '-base'
    try:
        real_index = os.path.join(real_git_dir, 'index')
        if os.path.exists(real_index):        # seeds the stat cache; contents replaced below
            shutil.copyfile(real_index, idx)
        env = {**os.environ, 'GIT_DIR': gd, 'GIT_WORK_TREE': top, 'GIT_INDEX_FILE': idx}
        git(['add', '-A', '--', '.', EXCLUDE_LEDGER], cwd=top, env=env)
        _rm_ledger(env, top)
        tree = git(['write-tree'], cwd=top, env=env)
        benv = {**env, 'GIT_INDEX_FILE': base_idx}
        git(['read-tree', base], cwd=top, env=benv)
        _rm_ledger(benv, top)
        base_tree = git(['write-tree'], cwd=top, env=benv)
        dirty = tree != base_tree
        paths = [p for p in git(['diff-tree', '-r', '--name-only', '-z', base_tree, tree],
                                cwd=top, env=env).split('\0') if p]
        if dirty:
            snap = git(['commit-tree', tree, '-p', base, '-m',
                        f'run {run_id} (dirty snapshot of {base})'],
                       cwd=top, env={**env, **IDENTITY})
        else:
            snap = base                       # clean: the run is its base commit
        git(['update-ref', '-m', f'run {run_id}', f'refs/runs/{run_id}', snap, ''],
            cwd=top, env=env)
    except subprocess.CalledProcessError as e:
        refuse(f'git failed while making the snapshot ({" ".join(e.cmd)}): '
               f'{(e.stderr or "").strip()}')
    finally:
        for f in (idx, base_idx, idx + '.lock', base_idx + '.lock'):
            with contextlib.suppress(FileNotFoundError):
                os.remove(f)
    return {'snapshot': snap, 'ref': f'refs/runs/{run_id}', 'base': base,
            'base_branch': branch, 'dirty': dirty, 'dirty_paths': paths[:MAX_DIRTY_PATHS],
            'dirty_count': len(paths), 'worktree': top,
            **({'store': gd} if store else {})}


# ---------------------------------------------------------------- ledger

def ledger_dir(top=CODE_ROOT):
    """``XP_LEDGER_DIR``, else ``experiments/ledger`` of the MAIN worktree
    (the parent of the common git dir), so runs from linked worktrees land
    in one place."""
    if os.environ.get('XP_LEDGER_DIR'):
        return os.path.abspath(os.environ['XP_LEDGER_DIR'])
    common = git(['rev-parse', '--path-format=absolute', '--git-common-dir'], cwd=top)
    return os.path.join(os.path.dirname(common), LEDGER)


def write_json(path, data):
    """Atomic: a reader sees the old file or the new one, never a part."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=1, sort_keys=False)
        f.write('\n')
    os.replace(tmp, path)


_HASHES = {}


def file_hash(path):
    st = os.stat(path)
    key = (os.path.realpath(path), st.st_size, st.st_mtime_ns)
    if key not in _HASHES:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        _HASHES[key] = {'sha256': h.hexdigest(), 'bytes': st.st_size}
    return dict(_HASHES[key])


def pip_freeze():
    """``name==version`` of every installed distribution, sorted (the
    content of ``pip freeze``, read through importlib.metadata)."""
    from importlib import metadata
    lines = set()
    for d in metadata.distributions():
        name = d.metadata.get('Name')
        if name:
            lines.add(f'{name}=={d.version}')
    return '\n'.join(sorted(lines, key=str.lower)) + '\n'


def _version(dist):
    from importlib import metadata
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def environment(freeze):
    import torch
    cudnn = None
    with contextlib.suppress(Exception):
        cudnn = torch.backends.cudnn.version()
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'cudnn': cudnn,
            'fla': _version('flash-linear-attention') or _version('fla'),
            'hydra': _version('hydra-core'), 'omegaconf': _version('omegaconf'),
            'numpy': _version('numpy'),
            'pip_freeze_sha256': hashlib.sha256(freeze.encode()).hexdigest(),
            'executable': sys.executable, 'platform': platform.platform(),
            'node': socket.gethostname()}


def hardware(device):
    import torch
    if getattr(device, 'type', str(device)) != 'cuda' or not torch.cuda.is_available():
        return {'gpu': None, 'gpu_count': 0, 'driver': None, 'device': str(device)}
    driver = None
    with contextlib.suppress(Exception):
        driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=10).stdout.split('\n')[0].strip() or None
    return {'gpu': torch.cuda.get_device_name(device), 'gpu_count': torch.cuda.device_count(),
            'driver': driver, 'device': str(device)}


def slurm():
    e = os.environ
    if 'SLURM_JOB_ID' not in e:
        return None
    task = e.get('SLURM_ARRAY_TASK_ID')
    job = (f'{e["SLURM_ARRAY_JOB_ID"]}_{task}' if e.get('SLURM_ARRAY_JOB_ID') and task is not None
           else e['SLURM_JOB_ID'])
    packed = [x for x in e.get('XP_PACK_GROUP', '').split(',') if x]
    return {'job_id': job, 'array_job_id': e.get('SLURM_ARRAY_JOB_ID'),
            'array_task_id': int(task) if task is not None and task.isdigit() else task,
            'job_name': e.get('SLURM_JOB_NAME'), 'partition': e.get('SLURM_JOB_PARTITION'),
            'packed_with': packed or None}


def submit():
    sid = os.environ.get('XP_SUBMIT_ID')
    if not sid:
        return None
    return {'id': sid, 'script': os.environ.get('XP_SUBMIT_SCRIPT'),
            'script_sha256': os.environ.get('XP_SUBMIT_SCRIPT_SHA256')}


class Record:
    """A run's ledger record: the only writer of its file and of the run
    directory's provenance.json."""

    def __init__(self, path, data, freeze):
        self.path, self.data, self.freeze = path, data, freeze
        self.out = None
        self.ended = False

    @property
    def run_id(self):
        return self.data['run_id']

    @property
    def snapshot(self):
        return self.data['code']['snapshot']

    def stamp(self):
        """What a checkpoint carries, so a .pt found anywhere leads back here."""
        return {'run_id': self.run_id, 'snapshot': self.snapshot}

    def write(self):
        self.data['updated'] = iso(utc_now())
        write_json(self.path, self.data)
        if self.out is not None:
            write_json(os.path.join(self.out, 'provenance.json'), self.data)

    def attach(self, out, overwrite=False):
        """The run directory exists: claim it with RUN_ID (exclusively, unless
        the run replaces an old one), then write provenance.json and
        pip_freeze.txt. A second run racing for the same directory is
        refused here."""
        claim = os.path.join(out, 'RUN_ID')
        try:
            fd = os.open(claim, os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if overwrite else os.O_EXCL),
                         0o644)
        except FileExistsError:
            other = open(claim).read().strip()
            raise SystemExit(f'{out} is claimed by run {other} (its RUN_ID file). Write elsewhere: '
                             f'run.name=<new name>, or run.overwrite=true to replace it')
        with os.fdopen(fd, 'w') as f:
            f.write(self.run_id + '\n')
        self.out = out
        with open(os.path.join(out, 'pip_freeze.txt'), 'w') as f:
            f.write(self.freeze)
        self.write()

    def checkpoint(self, step, path):
        prog = self.data['progress']
        prog['last_step'] = max(prog['last_step'], int(step))
        rel = os.path.relpath(path, self.out) if self.out else path
        if rel not in prog['checkpoints']:
            prog['checkpoints'].append(rel)
        self.write()

    def end(self, status, exit_status, error=None, final=None):
        if self.ended:
            return
        self.ended = True
        self.data['status'] = status
        self.data['ended'] = iso(utc_now())
        self.data['end'] = {'exit_status': exit_status, 'reason': status, 'error': error,
                            'elapsed_s': (final or {}).get('elapsed_s'),
                            'final_metrics': final_metrics(final) if final else None}
        if final:
            self.data['progress']['last_step'] = max(self.data['progress']['last_step'],
                                                     int(final.get('steps', 0)))
        self.write()

    @contextlib.contextmanager
    def lifetime(self):
        """Ends the record however the run ends: completed (``end`` called
        by the trainer), failed (an exception, with its text), or terminated
        (SIGTERM, Ctrl-C). SIGTERM is turned into an exception here so the
        record is written before the process goes."""
        old = None

        def on_term(signum, frame):
            raise Terminated()
        main = hasattr(signal, 'SIGTERM') and \
            __import__('threading').current_thread() is __import__('threading').main_thread()
        if main:
            old = signal.signal(signal.SIGTERM, on_term)
        try:
            yield self
            if not self.ended:
                self.end('failed', 1, error='the run returned without an end record')
        except Terminated:
            self.end('terminated', 143, error='SIGTERM')
            raise SystemExit(143)
        except KeyboardInterrupt:
            self.end('terminated', 130, error='KeyboardInterrupt')
            raise
        except SystemExit as e:
            if e.code not in (None, 0):
                self.end('failed', e.code if isinstance(e.code, int) else 1, error=str(e.code))
            raise
        except BaseException as e:
            self.end('failed', 1, error=f'{type(e).__name__}: {e}'[:4000])
            raise
        finally:
            if main:
                signal.signal(signal.SIGTERM, old)


def final_metrics(final):
    e = final.get('final_eval') or {}
    s = final.get('final_eval_shuffled') or {}
    best = final.get('best_eval') or {}
    return {'mse': e.get('mse'), 'mse_last': e.get('mse_last'),
            'shuffled_mse': s.get('mse'), 'floor_mse': (final.get('floor') or {}).get('mse'),
            'best_step': final.get('best_step'), 'best_mse': best.get('mse')}


def begin(experiment, config, sources, location, data_files=(), device='cpu', parent=None,
          top=CODE_ROOT):
    """Snapshot the code and write the ledger's start record. Called by
    ``train.main`` after every check and before the run directory or any
    training step exists. ``location``: ``Run.location()``. ``parent``: the
    loaded ``run.branch`` checkpoint, if any. Returns the ``Record``."""
    from omegaconf import OmegaConf
    r, m, t = experiment.run, experiment.model, experiment.task
    note = check_note(r.note)
    now = utc_now()
    h = host()
    run_id = new_run_id(h, now)
    code = snapshot(run_id, top, r.provenance.max_untracked_mb)
    freeze = pip_freeze()
    resolved = OmegaConf.to_container(config) if config is not None else None
    text = None
    exp = (sources or {}).get('experiment')
    if exp is not None and os.path.exists(exp):
        text = open(exp).read()
    data = {}
    for f in data_files:
        if f and os.path.exists(f):
            data[f] = file_hash(f)
    parent_prov = (parent or {}).get('provenance') or {}
    loc = dict(location)
    data_ = {
        'schema': SCHEMA, 'run_id': run_id, 'host': h, 'status': 'running',
        'created': iso(now), 'updated': iso(now), 'ended': None,
        'code': code,
        'entry': {'argv': list(sys.argv), 'cwd': os.getcwd(),
                  'via': os.environ.get('XP_VIA') or os.path.basename(sys.argv[0] or 'python')},
        'config': {'resolved': resolved,
                   'overrides': list((sources or {}).get('overrides') or []),
                   'experiment_file': exp, 'experiment_text': text,
                   'sha256': hashlib.sha256(OmegaConf.to_yaml(config).encode()).hexdigest()
                   if config is not None else None},
        'run': {'task': t.name, 'model': m.name, 'size': m.size,
                'name': r.name or f'seed{r.seed}', 'seed': r.seed, 'steps': r.steps,
                'note': note, 'tags': list(r.tag or ())},
        'results': {'dir': os.path.abspath(loc['dir']), 'root': os.path.abspath(loc['root']),
                    'rel': loc['rel']},
        'parent': {'branch_from': r.branch, 'parent_run_id': parent_prov.get('run_id')},
        'reproduces': os.environ.get('XP_REPRODUCES') or None,
        'env': environment(freeze),
        'data': data,
        'hardware': hardware(device),
        'slurm': slurm(),
        'submit': submit(),
        'progress': {'last_step': int((parent or {}).get('step', 0)), 'checkpoints': []},
        'end': None,
    }
    record = Record(os.path.join(ledger_dir(top), h, run_id + '.json'), data_, freeze)
    record.write()
    print(f'provenance: run {run_id}, code {code["snapshot"][:12]} '
          f'({"dirty snapshot of " + code["base"][:12] if code["dirty"] else "clean"}), '
          f'ledger {record.path}', flush=True)
    return record


# ---------------------------------------------------------------- reading

def read_ledger(root=None):
    """Every record under the ledger directory, as dicts (a corrupt file is
    skipped with a warning on stderr)."""
    root = root or ledger_dir()
    out = []
    if not os.path.isdir(root):
        return out
    for dp, _, fs in os.walk(root):
        for f in sorted(fs):
            if not f.endswith('.json'):
                continue
            p = os.path.join(dp, f)
            try:
                rec = json.load(open(p))
            except (OSError, ValueError) as e:
                print(f'warning: skipping unreadable ledger file {p}: {e}', file=sys.stderr)
                continue
            if isinstance(rec, dict) and 'run_id' in rec:
                rec['_path'] = p
                out.append(rec)
    out.sort(key=lambda r: r.get('created') or '')
    return out


def find_run(prefix, root=None):
    """The one record whose id starts with ``prefix`` (or equals it)."""
    recs = read_ledger(root)
    exact = [r for r in recs if r['run_id'] == prefix]
    if exact:
        return exact[0]
    hits = [r for r in recs if r['run_id'].startswith(prefix)]
    if not hits:
        raise KeyError(f'no run {prefix!r} in the ledger ({root or ledger_dir()})')
    if len(hits) > 1:
        raise KeyError(f'{prefix!r} is ambiguous: ' + ', '.join(r['run_id'] for r in hits[:10]))
    return hits[0]


REPRODUCTIONS = 'reproductions'


def iter_run_dirs(root):
    """Every run directory (one holding final.json) under ``root``, never
    descending into a run's ``reproductions/`` (those are reproductions of
    the run, not runs of the study)."""
    for dp, ds, fs in os.walk(root):
        if REPRODUCTIONS in ds:
            ds.remove(REPRODUCTIONS)
        ds.sort()
        if 'final.json' in fs:
            yield dp
