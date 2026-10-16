"""xp: run, list, show, compare, reproduce, sync and tag runs.

    xp ls                       the latest 20 runs, newest first (start, status, note,
                                path, host, tag, final mse); the first thing to run
    xp ls --all --running --tag T --task T --model M --host H --since 2d --json
    xp show <run>               one run: record, code, files, registry entry
    xp compare <run> <run> ...  metrics, config differences, code differences
    xp run <preset> -m NOTE [overrides]      python run.py +experiment=<preset> ...
    xp run --exp FILE -m NOTE [overrides]
    xp reproduce <run>          rerun its exact code into <run>/reproductions/<ts>_<id>/,
                                and say whether it is bitwise identical
    xp sync                     commit the local ledger (local git only, for now)
    xp tag <run> <tag> [--note TEXT]         curate: append to experiments/registry.yaml

A <run> is a run id, a unique prefix or 6-hex suffix of one, or its results
path (``z2/vanilla4/xs/a``, ``torch:se2d/...``). Every command takes
``--explain`` (print each underlying command first) and ``--dry-run``
(print the commands that would change something, run none of them);
queries take ``--json``. docs/provenance_design.md §9 and §15.
"""
import argparse
import datetime
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile

from mapping import provenance as P
from mapping.xp import ledger_query as Q

REGISTRY = os.path.join('experiments', 'registry.yaml')


# ---------------------------------------------------------------- plumbing

class Runner:
    """Runs (and with --explain prints) every underlying command."""

    def __init__(self, explain=False, dry_run=False):
        self.explain, self.dry_run = explain, dry_run

    def show(self, cmd, cwd=None, env=None):
        pre = f'cd {shlex.quote(cwd)} && ' if cwd else ''
        envs = ''.join(f'{k}={shlex.quote(v)} ' for k, v in (env or {}).items())
        print('+ ' + pre + envs + ' '.join(shlex.quote(str(c)) for c in cmd),
              file=sys.stderr, flush=True)

    def run(self, cmd, cwd=None, env=None, mutate=False, check=True, capture=True,
            stdout=None):
        if self.explain or (self.dry_run and mutate):
            self.show(cmd, cwd, env)
        if self.dry_run and mutate:
            return subprocess.CompletedProcess(cmd, 0, '', '')
        r = subprocess.run([str(c) for c in cmd], cwd=cwd, env={**os.environ, **(env or {})},
                           capture_output=capture, text=True,
                           **({} if capture else {'stdout': stdout}))
        if check and r.returncode != 0:
            err = (r.stderr or '').strip() if capture else ''
            raise SystemExit(f'xp: {" ".join(map(str, cmd))} failed ({r.returncode})'
                             + (f': {err}' if err else ''))
        return r

    def note(self, text):
        """A step that is not a command (a file written by xp itself)."""
        if self.explain or self.dry_run:
            print('# ' + text, file=sys.stderr, flush=True)


def repo_root(start=None):
    """The harness checkout xp acts on: the git checkout around ``start``
    (default: the working directory) when it is this harness, else the one
    xp is installed from."""
    r = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=start or os.getcwd(),
                       capture_output=True, text=True)
    top = r.stdout.strip()
    if r.returncode == 0 and os.path.exists(os.path.join(top, 'run.py')) and \
            os.path.exists(os.path.join(top, 'mapping', 'provenance.py')):
        return top
    return P.CODE_ROOT


def git_objects(top):
    """git arguments that see run snapshots: the test suite's snapshot store
    (tests/conftest.py) when set, else the repository itself."""
    store = os.environ.get('XP_SNAPSHOT_GIT_DIR')
    return ['git', '--git-dir', store] if store else ['git', '-C', top]


def load_registry(top):
    path = os.path.join(top, REGISTRY)
    if not os.path.exists(path):
        return {}
    import yaml
    return yaml.safe_load(open(path)) or {}


def registry_tags(reg):
    out = {}
    for t in reg.get('tagged') or []:
        out.setdefault(t['id'], set()).add(t['tag'])
    return out


def registry_entry(reg, rec):
    """The registry experiment that lists this run (by audit id or path)."""
    keys = {(rec.get('audit') or {}).get('row'), (rec.get('results') or {}).get('dir'),
            rec['run_id']}
    for e in reg.get('experiments') or []:
        for r in e.get('runs') or []:
            if r.get('id') in keys or r.get('path') in keys:
                return {k: e[k] for k in ('id', 'title', 'status', 'decision') if k in e}
    for t in reg.get('tagged') or []:
        if t['id'] == rec['run_id']:
            return {'tagged': t}
    return None


def records(top):
    return P.read_ledger(P.ledger_dir(top))


def strip_private(rec):
    return {k: v for k, v in rec.items() if not k.startswith('_')}


def emit_json(obj):
    print(json.dumps(obj, indent=1, default=str))


def hydra_quote(s):
    return "'" + str(s).replace('\\', '\\\\').replace("'", "\\'") + "'"


def flat_diff(a, b, prefix=''):
    """Dotted keys whose values differ between two plain dicts."""
    out = []
    a, b = a or {}, b or {}
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k, '<missing>'), b.get(k, '<missing>')
        if isinstance(x, dict) and isinstance(y, dict):
            out += flat_diff(x, y, f'{prefix}{k}.')
        elif x != y:
            out.append(f'{prefix}{k}')
    return out


def dotted(d, key):
    for k in key.split('.'):
        if not isinstance(d, dict) or k not in d:
            return '<missing>'
        d = d[k]
    return d


# ---------------------------------------------------------------- ls

def cmd_ls(a, run, top):
    reg = load_registry(top)
    rtags = registry_tags(reg)
    recs = Q.select(records(top), a, rtags)
    shown = recs if a.all else recs[:a.latest]
    if a.json:
        emit_json([strip_private(r) for r in shown])
        return 0
    if not recs:
        print('no runs in the ledger match' + (' (xp ls --all for everything)' if not a.all else ''))
        return 0
    print(Q.header())
    for r in shown:
        print(Q.line(r, rtags))
    foot = [f'{len(shown)} of {len(recs)} runs']
    if len(shown) < len(recs):
        foot.append(f'xp ls --latest N or --all for more')
    if any(r.get('retroactive') for r in shown):
        foot.append('* retroactive record (audit of 2026-10-16; start time estimated)')
    print('  '.join(foot))
    return 0


# ---------------------------------------------------------------- show

def run_files(d):
    if not d or not os.path.isdir(d):
        return None
    out = sorted(os.listdir(d))
    return out


def cmd_show(a, run, top):
    recs = records(top)
    rec = Q.find(recs, a.run)
    reg = load_registry(top)
    entry = registry_entry(reg, rec)
    repros = [r['run_id'] for r in recs if r.get('reproduces') == rec['run_id']]
    code = rec.get('code') or {}
    g = git_objects(top)
    stat = diff = None
    if code.get('snapshot'):
        r = run.run(g + ['show', '--stat', '--format=%H %s', code['snapshot']], check=False)
        stat = r.stdout.strip() if r.returncode == 0 else f'(snapshot not here: {r.stderr.strip()})'
        if code.get('dirty'):
            r = run.run(g + ['diff', '--stat', code['base'], code['snapshot']], check=False)
            diff = r.stdout.strip() if r.returncode == 0 else None
    d = (rec.get('results') or {}).get('dir')
    files = run_files(d) if rec.get('host') == P.host() or os.path.isdir(d or '') else None
    if a.json:
        emit_json({'record': strip_private(rec), 'ledger_file': rec['_path'],
                   'reproductions': repros, 'registry': entry, 'snapshot_stat': stat,
                   'dirty_diff_stat': diff, 'files': files})
        return 0
    run_ = rec.get('run') or {}
    fm = (rec.get('end') or {}).get('final_metrics') or {}
    lines = [
        f'run      {rec["run_id"]}' + ('   (retroactive record)' if rec.get('retroactive') else ''),
        f'note     {run_.get("note")}',
        f'status   {Q.STATUS_LABEL.get(rec.get("status"), rec.get("status"))}'
        + (f' ({rec.get("status_as_found")}, as of {rec.get("as_of")})' if rec.get('retroactive') else '')
        + (f'   exit {rec["end"]["exit_status"]}' if (rec.get('end') or {}).get('exit_status') is not None else '')
        + (f'   error: {rec["end"]["error"]}' if (rec.get('end') or {}).get('error') else ''),
        f'started  {rec.get("created")}   ended {rec.get("ended")}',
        f'what     {run_.get("task")} {run_.get("model")}/{run_.get("size")} {run_.get("name")} '
        f'seed {run_.get("seed")}, {run_.get("steps")} steps',
        f'where    {rec.get("host")}:{(rec.get("results") or {}).get("dir")}'
        + (f'\n         local mirror {rec["results"]["local_mirror"]}'
           if (rec.get('results') or {}).get('local_mirror') else ''),
        f'tags     {", ".join(Q.tags(rec, registry_tags(reg))) or "-"}',
        f'metrics  ' + (', '.join(f'{k} {v:.6g}' for k, v in fm.items() if isinstance(v, float))
                        or '-'),
    ]
    if rec.get('retroactive'):
        lines.append(f'code     {code.get("commit")} ({code.get("basis")}); no snapshot')
        if code.get('tree_verified'):
            lines.append(f'         {code["tree_verified"]}')
    else:
        lines.append(f'code     {code.get("snapshot")} ({code.get("ref")})')
        lines.append(f'         base {code.get("base")} on {code.get("base_branch")}, '
                     + (f'dirty: {code.get("dirty_count")} paths' if code.get('dirty') else 'clean'))
    if rec.get('reproduces'):
        lines.append(f'repro of {rec["reproduces"]}')
    if repros:
        lines.append('reproduced by ' + ', '.join(repros))
    if entry:
        lines.append(f'registry {json.dumps(entry)}')
    lines.append(f'ledger   {rec["_path"]}')
    print('\n'.join(lines))
    if stat:
        print('\n' + stat)
    if diff:
        print('\nchanges against its base:\n' + diff)
    if files is not None:
        pts = [f for f in files if f.endswith('.pt')]
        print(f'\nfiles    {len(files)} in {d}; checkpoints: {", ".join(pts) or "none"}')
    return 0


# ---------------------------------------------------------------- compare

COMPARE_SKIP = ('run.out', 'run.note', 'run.overwrite')


def cmd_compare(a, run, top):
    recs = records(top)
    sel = [Q.find(recs, x) for x in a.runs]
    keys = [k.strip() for k in a.metric.split(',')]
    table = []
    for r in sel:
        fm = (r.get('end') or {}).get('final_metrics') or {}
        table.append({'run_id': r['run_id'], 'note': (r.get('run') or {}).get('note'),
                      'path': Q.path_of(r), **{k: fm.get(k) for k in keys}})
    base = sel[0]
    cfg0 = (base.get('config') or {}).get('resolved')
    configs, codes = [], []
    g = git_objects(top)
    for r in sel[1:]:
        cfg = (r.get('config') or {}).get('resolved')
        if cfg0 is None or cfg is None:
            configs.append({'run_id': r['run_id'], 'differs': None,
                            'why': 'no resolved config (retroactive legacy record)'})
        else:
            diff = [k for k in flat_diff(cfg0, cfg) if not k.startswith(COMPARE_SKIP)]
            configs.append({'run_id': r['run_id'],
                            'differs': {k: [dotted(cfg0, k), dotted(cfg, k)] for k in diff}})
        sa, sb = (base.get('code') or {}).get('snapshot'), (r.get('code') or {}).get('snapshot')
        if sa and sb:
            res = run.run(g + ['diff', '--stat', sa, sb], check=False)
            codes.append({'run_id': r['run_id'], 'same': sa == sb,
                          'stat': res.stdout.strip() if res.returncode == 0 else None})
        else:
            codes.append({'run_id': r['run_id'], 'same': None,
                          'stat': 'no snapshot on one side (retroactive): commits '
                                  f'{(base.get("code") or {}).get("commit") or sa} vs '
                                  f'{(r.get("code") or {}).get("commit") or sb}'})
    if a.grid:
        cks = [os.path.join((r.get('results') or {}).get('dir') or '', 'final.pt') for r in sel]
        run.run([sys.executable, os.path.join(top, 'tools', 'compare_grid.py'), *cks,
                 '--labels', *[Q.short_id(r) for r in sel]], cwd=top, capture=False,
                stdout=sys.stderr if a.json else None)
    if a.json:
        emit_json({'metrics': table, 'config': configs, 'code': codes})
        return 0
    w = max(len(Q.clip(t['note'] or '', 40)) for t in table)
    print('  '.join(['id'.ljust(6), 'note'.ljust(w)] + [k.ljust(10) for k in keys]))
    for t, r in zip(table, sel):
        print('  '.join([Q.short_id(r).ljust(6), Q.clip(t['note'] or '', 40).ljust(w)]
                        + [(f'{t[k]:.5g}' if isinstance(t[k], (int, float)) else '-').ljust(10)
                           for k in keys]))
    for c, k in zip(configs, codes):
        print(f'\nvs {c["run_id"]} (first: {base["run_id"]})')
        if c['differs'] is None:
            print(f'  config: {c["why"]}')
        else:
            print('  config: ' + ('identical' if not c['differs'] else ''))
            for key, (x, y) in c['differs'].items():
                print(f'    {key}: {x!r} -> {y!r}')
        print('  code: ' + ('same snapshot' if k['same'] else (k['stat'] or 'differs')))
    return 0


# ---------------------------------------------------------------- run

def gpu_lock(top):
    h = P.host()
    entry = ((P.hosts_table().get('hosts') or {}).get(h)) or {}
    return entry.get('gpu_lock')


def cmd_run(a, run, top):
    note = P.check_note(a.message)
    target = ['--exp', a.exp] if a.exp else ([f'+experiment={a.preset}'] if a.preset else [])
    if not target and not a.overrides:
        raise SystemExit('xp run: give a preset (conf/experiment/<name>), --exp FILE, or '
                         'model=<task>/<name> overrides')
    if a.preset and '=' in a.preset:              # no preset: the first override
        target, a.overrides = [], [a.preset, *a.overrides]
    cmd = [sys.executable, os.path.join(top, 'run.py'), *target, *a.overrides,
           f'run.note={hydra_quote(note)}']
    if a.tag:
        cmd.append('run.tag=[' + ','.join(a.tag) + ']')
    env = {'XP_VIA': 'xp run'}
    lock = gpu_lock(top)
    if lock:
        cmd = ['flock', lock, *cmd]
    if a.bg:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        log = os.path.join(top, 'results', 'jobs', 'logs', f'local-{stamp}.out')
        if not a.dry_run:
            os.makedirs(os.path.dirname(log), exist_ok=True)
        run.note(f'nohup {" ".join(map(shlex.quote, cmd))} > {log} 2>&1 &')
        if a.dry_run:
            return 0
        if a.explain:
            run.show(['nohup', *cmd], cwd=top, env=env)
        with open(log, 'w') as f:
            subprocess.Popen(cmd, cwd=top, env={**os.environ, **env}, stdout=f,
                             stderr=subprocess.STDOUT, start_new_session=True)
        print(f'started in the background; log {log} (its first line names the run id)')
        return 0
    r = run.run(cmd, cwd=top, env=env, mutate=True, check=False, capture=False)
    return r.returncode


# ---------------------------------------------------------------- reproduce

def write_sources(src, rec):
    """config.yaml, overrides.yaml and experiment.yaml of the original run, from
    its ledger record, for ``run.py --from`` in the reproduction worktree."""
    from omegaconf import OmegaConf
    c = rec['config']
    os.makedirs(src, exist_ok=True)
    OmegaConf.save(OmegaConf.create(c['resolved']), os.path.join(src, 'config.yaml'))
    overrides = [o for o in c.get('overrides') or []
                 if o.split('=', 1)[0].lstrip('+~') not in ('run.overwrite', 'run.note')]
    exp = None
    if c.get('experiment_text') is not None:
        exp = 'experiment.yaml'
        with open(os.path.join(src, exp), 'w') as f:
            f.write(c['experiment_text'])
    OmegaConf.save(OmegaConf.create({'overrides': overrides, 'experiment': exp}),
                   os.path.join(src, 'overrides.yaml'))


def link_data(rec, top, wt, allow_drift):
    """Each data file the run read, linked into the worktree under the same
    relative path, its sha256 checked against the ledger's."""
    drift = {}
    for rel, info in (rec.get('data') or {}).items():
        src = rel if os.path.isabs(rel) else os.path.join(top, rel)
        if not os.path.exists(src):
            drift[rel] = {'original': info.get('sha256'), 'here': None}
            continue
        here = P.file_hash(src)['sha256']
        if here != info.get('sha256'):
            drift[rel] = {'original': info.get('sha256'), 'here': here}
        if not os.path.isabs(rel):
            dst = os.path.join(wt, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.lexists(dst):
                os.symlink(os.path.realpath(src), dst)
    if drift and not allow_drift:
        raise SystemExit('xp reproduce: the data differs from what the run read '
                         f'{json.dumps(drift)}; --allow-data-drift to reproduce anyway')
    return drift


def compare_metrics(a, b, tol):
    out = {}
    for k in sorted(set(a or {}) | set(b or {})):
        x, y = (a or {}).get(k), (b or {}).get(k)
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            out[k] = {'original': x, 'reproduction': y, 'abs_diff': None, 'rel_diff': None}
            continue
        d = abs(x - y)
        out[k] = {'original': x, 'reproduction': y, 'abs_diff': d,
                  'rel_diff': d / abs(x) if x else (0.0 if d == 0 else None)}
    return out


def verdict(metrics, ckpt_bitwise, tol):
    diffs = [m['abs_diff'] for m in metrics.values() if m['abs_diff'] is not None]
    if not diffs:
        return 'differs'
    if ckpt_bitwise and all(d == 0 for d in diffs):
        return 'bitwise'
    if all(d <= tol for d in diffs):
        return 'within_tolerance'
    return 'differs'


def cmd_reproduce(a, run, top):
    recs = records(top)
    rec = Q.find(recs, a.run)
    rid = rec['run_id']
    if rec.get('retroactive') or not (rec.get('code') or {}).get('snapshot'):
        raise SystemExit(f'xp reproduce: {rid} has no code snapshot (a retroactive record: '
                         f'{(rec.get("code") or {}).get("commit")}); an exact reproduction is '
                         f'impossible. Rerun it by hand at that commit and say so')
    if (rec.get('config') or {}).get('resolved') is None:
        raise SystemExit(f'xp reproduce: {rid} has no saved config in its record')
    if os.environ.get('XP_SNAPSHOT_GIT_DIR'):
        raise SystemExit('xp reproduce needs the snapshot in this repository '
                         '(XP_SNAPSHOT_GIT_DIR is set: a test session)')
    snap = rec['code']['snapshot']
    if run.run(['git', '-C', top, 'cat-file', '-e', snap + '^{commit}'], check=False).returncode:
        raise SystemExit(f'xp reproduce: snapshot {snap} of {rid} is not in this repository '
                         f'({top}); fetch refs/runs/{rid} from where it ran first')
    new_id = P.new_run_id(P.host())
    stamp = new_id.split('-', 1)[0]
    orig_dir = (rec.get('results') or {}).get('dir')
    stub = not (orig_dir and os.path.isdir(orig_dir))
    if stub:                           # the original lives elsewhere: a marked stand-in
        rel = (rec.get('results') or {}).get('rel') or os.path.join('_elsewhere', rec['host'], rid)
        parent = os.path.join(top, 'results', rel)
    else:
        parent = orig_dir
    out = os.path.join(parent, P.REPRODUCTIONS, f'{stamp}_{new_id}')
    wt = os.path.join(tempfile.gettempdir(), f'xp-repro-{new_id}')
    src = os.path.join(tempfile.gettempdir(), f'xp-repro-src-{new_id}')
    run.run(['git', '-C', top, 'worktree', 'add', '--detach', wt, snap], mutate=True)
    try:
        if a.dry_run:
            run.note(f'link data files into {wt}; write the run sources into {src}')
        else:
            drift = link_data(rec, top, wt, a.allow_data_drift)
            write_sources(src, rec)
            if stub:
                os.makedirs(parent, exist_ok=True)
                with open(os.path.join(parent, 'STUB'), 'w') as f:
                    f.write(f'Not a run: a stand-in for {rid}, which ran on {rec["host"]} at '
                            f'{orig_dir}. Its reproductions are below.\n')
                P.write_json(os.path.join(parent, 'LEDGER.json'), strip_private(rec))
        cmd = [sys.executable, os.path.join(wt, 'run.py'), '--from',
               os.path.join(src, 'config.yaml'), f'run.out={out}',
               f'run.note={hydra_quote("reproduce " + rid)}']
        env = {'XP_VIA': 'xp reproduce', 'XP_REPRODUCES': rid, 'XP_RUN_ID': new_id}
        r = run.run(cmd, cwd=wt, env=env, mutate=True, check=False, capture=False,
                    stdout=sys.stderr if a.json else None)
        if a.dry_run:
            return 0
        if r.returncode != 0:
            print(f'xp reproduce: the reproduction run failed (exit {r.returncode}); '
                  f'its record: {out}/provenance.json', file=sys.stderr)
            return 2
        new = json.load(open(os.path.join(out, 'provenance.json')))
        metrics = compare_metrics((rec.get('end') or {}).get('final_metrics'),
                                  (new.get('end') or {}).get('final_metrics'), a.tolerance)
        ck = None
        a_pt, b_pt = os.path.join(orig_dir or '', 'final.pt'), os.path.join(out, 'final.pt')
        if not stub and os.path.exists(a_pt) and os.path.exists(b_pt):
            res = run.run([sys.executable, os.path.join(top, 'tools', 'compare_checkpoints.py'),
                           a_pt, b_pt], cwd=top)
            ck = json.loads(res.stdout)
        env_diff = {k: [(rec.get('env') or {}).get(k), (new.get('env') or {}).get(k)]
                    for k in sorted(set(rec.get('env') or {}) | set(new.get('env') or {}))
                    if (rec.get('env') or {}).get(k) != (new.get('env') or {}).get(k)}
        cfg_diff = [k for k in flat_diff((rec.get('config') or {}).get('resolved'),
                                         (new.get('config') or {}).get('resolved'))
                    if not k.startswith(COMPARE_SKIP)]
        report = {
            'original': rid, 'reproduction': new['run_id'],
            'verdict': verdict(metrics, (ck or {}).get('bitwise'), a.tolerance),
            'tolerance': a.tolerance, 'metrics': metrics,
            'checkpoint_bitwise': None if ck is None else ck['bitwise'],
            'checkpoint': ck,
            'code': {'original': snap, 'reproduction': new['code']['snapshot'],
                     'same': snap == new['code']['snapshot']},
            'env_diff': env_diff, 'data_diff': drift, 'config_diff': cfg_diff,
            'original_dir': orig_dir, 'original_is_here': not stub,
            'reproduction_dir': out, 'created': P.iso(P.utc_now()),
        }
        P.write_json(os.path.join(out, 'reproduce_report.json'), report)
        if a.json:
            emit_json(report)
        else:
            print(f'\nVERDICT: {report["verdict"].upper()}   {rid} -> {new["run_id"]}')
            print(f'  checkpoint tensors identical: {report["checkpoint_bitwise"]}; same code: '
                  f'{report["code"]["same"]}; config differences: {cfg_diff or "none"}')
            for k, m in metrics.items():
                if m['abs_diff'] is not None:
                    print(f'  {k}: {m["original"]:.6g} -> {m["reproduction"]:.6g} '
                          f'(|diff| {m["abs_diff"]:.3g})')
            if env_diff:
                print(f'  environment differences: {", ".join(env_diff)}')
            print(f'  report {os.path.join(out, "reproduce_report.json")}')
        return 0 if report['verdict'] in ('bitwise', 'within_tolerance') else 1
    finally:
        if not a.keep:
            run.run(['git', '-C', top, 'worktree', 'remove', '--force', wt], mutate=True,
                    check=False)
            if not a.dry_run:
                shutil.rmtree(src, ignore_errors=True)


# ---------------------------------------------------------------- sync

def cmd_sync(a, run, top):
    """Commit the ledger, by explicit path, in one tidy commit. Local only:
    pushing refs and pulling Torch's ledger come with the Torch migration
    (design §6, §9)."""
    import fcntl
    ledger = P.ledger_dir(top)
    rel = os.path.relpath(ledger, top)
    if rel.startswith(os.pardir):
        raise SystemExit(f'xp sync: the ledger ({ledger}, XP_LEDGER_DIR) is outside this '
                         f'repository ({top}); nothing to commit here')
    common = run.run(['git', '-C', top, 'rev-parse', '--path-format=absolute',
                      '--git-common-dir']).stdout.strip()
    with open(os.path.join(common, 'xp-sync.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        missing = [r['run_id'] for r in records(top) if not r.get('retroactive')
                   and r.get('host') == P.host() and (r.get('code') or {}).get('ref')
                   and run.run(['git', '-C', top, 'rev-parse', '-q', '--verify',
                                r['code']['ref']], check=False).returncode]
        run.run(['git', '-C', top, 'add', '--', rel], mutate=True)
        st = run.run(['git', '-C', top, 'diff', '--cached', '--name-status', '--', rel])
        changes = [ln.split('\t') for ln in st.stdout.splitlines() if ln.strip()]
        if a.dry_run:
            print('xp sync --dry-run: would commit the ledger changes now in the working tree')
            return 0
        if not changes:
            out = {'committed': None, 'added': 0, 'modified': 0, 'missing_refs': missing}
            emit_json(out) if a.json else print('nothing to sync: the ledger is committed')
            return 0
        by_host = {}
        for status, path in ((c[0], c[-1]) for c in changes):
            h = os.path.relpath(path, rel).split(os.sep)[0]
            n = by_host.setdefault(h, [0, 0])
            n[0 if status.startswith('A') else 1] += 1
        msg = 'ledger: sync (' + ', '.join(f'{h} +{n[0]} ~{n[1]}' for h, n in sorted(by_host.items())) + ')'
        run.run(['git', '-C', top, 'commit', '-q', '-m', msg, '--', rel], mutate=True)
        sha = run.run(['git', '-C', top, 'rev-parse', 'HEAD']).stdout.strip()
    out = {'committed': sha, 'message': msg,
           'added': sum(n[0] for n in by_host.values()),
           'modified': sum(n[1] for n in by_host.values()), 'missing_refs': missing}
    if a.json:
        emit_json(out)
    else:
        print(f'{msg}: {sha[:12]}')
        if missing:
            print('warning: no snapshot ref here for ' + ', '.join(missing), file=sys.stderr)
    return 0


# ---------------------------------------------------------------- tag

def cmd_tag(a, run, top):
    """Curate: append ``{id, tag, note, date}`` under ``tagged:`` in
    experiments/registry.yaml. The ledger is written only by runs, so a tag
    added later lives in the human-curated registry; ``xp ls --tag`` reads both."""
    rec = Q.find(records(top), a.run)
    path = os.path.join(top, REGISTRY)
    reg = load_registry(top)
    if any(t['id'] == rec['run_id'] and t['tag'] == a.tag for t in reg.get('tagged') or []):
        print(f'{rec["run_id"]} is already tagged {a.tag}')
        return 0
    text = open(path).read() if os.path.exists(path) else ''
    if 'tagged' in reg:
        tail = text[text.index('\ntagged:') + 1:] if '\ntagged:' in text else text
        if any(ln and not ln.startswith((' ', '#', 'tagged:')) for ln in tail.splitlines()):
            raise SystemExit(f'{path}: tagged: is not the last section; add the tag by hand')
    entry = (f'  - {{id: {json.dumps(rec["run_id"])}, tag: {json.dumps(a.tag)}, '
             f'note: {json.dumps(a.note or "")}, date: '
             f'"{datetime.date.today().isoformat()}"}}\n')
    add = ('' if text.endswith('\n') or not text else '\n')
    if 'tagged' not in reg:
        add += ('\n# Runs tagged after the fact (xp tag): ledger run ids, the tag, why, when.\n'
                'tagged:\n')
    run.note(f'append to {path}: {entry.strip()}')
    if a.dry_run:
        return 0
    with open(path, 'a') as f:
        f.write(add + entry)
    print(f'tagged {rec["run_id"]} {a.tag} in {REGISTRY}')
    return 0


# ---------------------------------------------------------------- main

def parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--explain', action='store_true',
                        help='print every underlying command before running it')
    common.add_argument('--dry-run', action='store_true',
                        help='print the commands that would change something; run none')
    common.add_argument('-C', dest='repo', default=None, help='the harness checkout (default: here)')
    p = argparse.ArgumentParser(prog='xp', description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split('\n\n', 1)[1])
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('ls', parents=[common], help='runs, newest first')
    s.add_argument('--latest', type=int, default=20, metavar='N')
    s.add_argument('--all', action='store_true')
    s.add_argument('--running', action='store_true')
    s.add_argument('--tag')
    s.add_argument('--task')
    s.add_argument('--model')
    s.add_argument('--host')
    s.add_argument('--since', help='2d, 12h, 30m or an ISO date')
    s.add_argument('--json', action='store_true')

    s = sub.add_parser('show', parents=[common], help='one run')
    s.add_argument('run')
    s.add_argument('--json', action='store_true')

    s = sub.add_parser('compare', parents=[common], help='runs side by side')
    s.add_argument('runs', nargs='+')
    s.add_argument('--metric', default='mse,mse_last,shuffled_mse')
    s.add_argument('--grid', action='store_true', help='tools/compare_grid.py on their final.pt')
    s.add_argument('--json', action='store_true')

    s = sub.add_parser('run', parents=[common], help='launch run.py with a note')
    s.add_argument('preset', nargs='?', help='conf/experiment/<preset>, or the first override')
    s.add_argument('overrides', nargs='*')
    s.add_argument('--exp')
    s.add_argument('-m', '--message', help='the run note (required)')
    s.add_argument('--tag', action='append')
    s.add_argument('--bg', action='store_true')

    s = sub.add_parser('reproduce', parents=[common], help='rerun a run\'s exact code')
    s.add_argument('run')
    s.add_argument('--tolerance', type=float, default=1e-6)
    s.add_argument('--keep', action='store_true', help='keep the worktree')
    s.add_argument('--allow-data-drift', action='store_true')
    s.add_argument('--json', action='store_true')

    s = sub.add_parser('sync', parents=[common], help='commit the ledger (local only)')
    s.add_argument('--json', action='store_true')

    s = sub.add_parser('tag', parents=[common], help='tag a run in the registry')
    s.add_argument('run')
    s.add_argument('tag')
    s.add_argument('--note', default='')
    return p


COMMANDS = {'ls': cmd_ls, 'show': cmd_show, 'compare': cmd_compare, 'run': cmd_run,
            'reproduce': cmd_reproduce, 'sync': cmd_sync, 'tag': cmd_tag}


def main(argv=None):
    a = parser().parse_args(argv)
    top = repo_root(a.repo)
    runner = Runner(a.explain, a.dry_run)
    sys.exit(COMMANDS[a.cmd](a, runner, top) or 0)
