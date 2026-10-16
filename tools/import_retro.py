"""Import the MAJOR experiments' runs into the ledger, retroactively.

    python tools/import_retro.py            write experiments/ledger/<host>/legacy-*.json
    python tools/import_retro.py --check    write nothing; exit 1 if a file would change

docs/provenance_design.md §11 and the user's decisions (§14): only runs of
experiments under ``experiments:`` in experiments/registry.yaml (status
major) are imported, with the borderline calls recorded there (pre-freeze
ablations and the time-recurrence pair major, the blind probe not).

Every record is marked ``"retroactive": true``. Nothing is made more exact
than the audit found it:
- ``code.snapshot`` and ``code.ref`` are null: no snapshot exists, so no
  ``refs/runs/`` ref is created. ``code.commit`` is the audit's string
  verbatim (``<sha>`` only where a commit was recorded at launch,
  ``inferred:<sha>``, or ``unknown``), with its basis; ``code.dirty`` is null.
- status, step and metrics are the audit's (experiments/audit/runs.csv,
  2026-10-16 16:05 EDT): runs then running stay ``running`` "as of" it.
- times: the audit's own strings are kept (``created_as_found``); ``created``
  converts them to UTC assuming America/New_York, where both machines are
  (``time_basis`` says so; local starts are estimates, last write minus
  elapsed).
- config: a Hydra run's saved config.yaml and overrides.yaml when they can
  be read on this machine; a legacy run's ``args`` record verbatim (from
  its final.json or log.jsonl, here or in the local mirror), never a config
  recomposed with today's conf/. Nothing on Torch is read.
- the note is the experiment's title in the registry; the tags are
  ``major`` and the experiment id.

The importer is these files' only writer; runs never touch them. Re-running
it rewrites them from the same sources (``--refresh`` in the design).
"""
import argparse
import csv
import datetime
import json
import os
import sys
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from mapping.provenance import SCHEMA, write_json  # noqa: E402

AUDIT = os.path.join(HERE, 'experiments', 'audit', 'runs.csv')
REGISTRY = os.path.join(HERE, 'experiments', 'registry.yaml')
MIGRATION = os.path.join(HERE, 'experiments', 'audit', 'z2_results_migration.tsv')
AUDIT_AS_OF = '2026-10-16T16:05 America/New_York'
TZ = ZoneInfo('America/New_York')
TREE_VERIFIED = {   # the audit's blob-hash cross-check (experiments/audit/README.md, Method 4)
    '/scratch/ax2348/mapping_hydra': ('50341e4', '2026-10-16 audit: every file of /scratch/ax2348/'
                                      'mapping_hydra hash-identical to 50341e4 (checked after '
                                      'launch; says nothing about edits between launch and the check)'),
}
STATUS = {'completed': 'completed', 'running': 'running', 'stopped': 'terminated'}


def legacy_id(audit_id):
    """``erid-2:se2d/hybrid/m/seed0_20k`` -> ``legacy-erid-2-se2d-hybrid-m-seed0_20k``."""
    host, rest = audit_id.split(':', 1)
    return 'legacy-' + host + '-' + rest.replace('#', '-').replace('/', '-')


def to_utc(found):
    """``2026-10-16T12:56 (est. ...)`` (America/New_York) -> ISO UTC, or None."""
    if not found:
        return None
    stamp = found.split(' ')[0]
    for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M'):
        try:
            t = datetime.datetime.strptime(stamp, fmt).replace(tzinfo=TZ)
            return t.astimezone(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            continue
    return None


def moved_paths():
    out = {}
    if os.path.exists(MIGRATION):
        for line in open(MIGRATION):
            parts = line.rstrip('\n').split('\t')
            if parts[0] == 'run':
                out[os.path.join(HERE, parts[1])] = os.path.join(HERE, parts[2])
    return out


def mirror_path(row, moved):
    m = row.get('local_mirror') or ''
    if not m:
        return None
    path = m.split(' (')[0]
    return moved.get(path, path)


def read_json_head(d):
    """(``args``, source file) from a run directory's final.json, else the
    first record of its log.jsonl."""
    if not d or not os.path.isdir(d):
        return None, None
    f = os.path.join(d, 'final.json')
    if os.path.exists(f):
        rec = json.load(open(f))
        if 'args' in rec:
            return rec['args'], f
    f = os.path.join(d, 'log.jsonl')
    if os.path.exists(f):
        with open(f) as fh:
            first = json.loads(fh.readline() or '{}')
        if 'args' in first:
            return first['args'], f
    return None, None


def config_block(row, path, mirror):
    from omegaconf import OmegaConf
    source = row['config_source']
    out = {'source': source, 'resolved': None, 'overrides': None, 'experiment_file': None,
           'experiment_text': None, 'legacy_args': None, 'read_from': None, 'missing': None}
    for d in (path, mirror):
        if d and os.path.exists(os.path.join(d, 'config.yaml')):
            out['resolved'] = OmegaConf.to_container(OmegaConf.load(os.path.join(d, 'config.yaml')))
            ov = os.path.join(d, 'overrides.yaml')
            if os.path.exists(ov):
                rec = OmegaConf.to_container(OmegaConf.load(ov))
                out['overrides'] = rec.get('overrides')
                if rec.get('experiment'):
                    e = os.path.join(d, rec['experiment'])
                    out['experiment_file'] = e
                    out['experiment_text'] = open(e).read() if os.path.exists(e) else None
            out['read_from'] = os.path.join(d, 'config.yaml')
            return out
    for d in (path, mirror):
        args, f = read_json_head(d)
        if args is not None:
            out['legacy_args'], out['read_from'] = args, f
            return out
    out['missing'] = ('on Torch and not mirrored here (the import reads nothing on Torch)'
                      if row['host'] == 'torch' else 'no config.yaml, final.json or log.jsonl args found')
    return out


def code_block(row):
    c = row['commit'] or 'unknown'
    recorded = inferred = None
    if c.startswith('inferred:'):
        inferred = c.split(':', 1)[1]
    elif c != 'unknown':
        recorded = c
    sha = recorded or inferred
    resolved = None
    if sha:
        import subprocess
        r = subprocess.run(['git', 'rev-parse', '-q', '--verify', sha + '^{commit}'], cwd=HERE,
                           capture_output=True, text=True)
        resolved = r.stdout.strip() or None
    tree_root = (row.get('tree') or '').split(' (')[0]
    verified = None
    if tree_root in TREE_VERIFIED and TREE_VERIFIED[tree_root][0] == recorded:
        verified = TREE_VERIFIED[tree_root][1]
    return {'snapshot': None, 'ref': None, 'commit': c, 'recorded': recorded,
            'inferred': inferred, 'resolved': resolved, 'basis': row['commit_basis'] or None,
            'tree': row.get('tree') or None, 'tree_verified': verified, 'dirty': None}


def number(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def record(row, exp, entry, moved):
    rid = legacy_id(row['run_id'])
    path = entry['path']
    mirror = mirror_path(row, moved)
    status = STATUS[row['status'].split(' ')[0]]
    if row['status'].startswith('stopped') and 'FAILED' in (row.get('job_state') or '') + row['status']:
        status = 'failed'
    root, _, rel = path.partition('/results/')
    task = row['task_canonical']
    final = None
    if status != 'running':
        final = {'mse': number(row['final_mse']), 'mse_last': number(row['final_mse_last']),
                 'shuffled_mse': None, 'floor_mse': None, 'best_step': None, 'best_mse': None}
    job = row.get('job_id') or None
    return {
        'schema': SCHEMA, 'run_id': rid, 'host': row['host'],
        'retroactive': True, 'provenance': 'retro',
        'status': status, 'status_as_found': row['status'],
        'created': to_utc(row['start']), 'created_as_found': row['start'] or None,
        'updated': None, 'ended': to_utc(row['end']) if status != 'running' else None,
        'ended_as_found': row['end'] or None,
        'time_basis': 'the audit\'s times (sacct on Torch; local: last write minus elapsed_s), '
                      'read as America/New_York and converted to UTC',
        'as_of': AUDIT_AS_OF,
        'code': code_block(row),
        'entry': {'argv': None, 'cwd': None, 'via': 'legacy train.py' if row['config_source']
                  .startswith('legacy') else 'run.py',
                  'job_script': row.get('job_script') or None},
        'config': config_block(row, path if row['host'] != 'torch' else None, mirror),
        'run': {'task': task, 'model': row['model'], 'size': row['size'], 'name': row['run'],
                'seed': int(row['seed']) if row['seed'].isdigit() else None,
                'steps': int(row['steps_planned']) if row['steps_planned'].isdigit() else None,
                'note': exp['title'], 'tags': ['major', exp['id']],
                'task_recorded': row['task']},
        'results': {'dir': path, 'root': root + '/results', 'rel': rel or None,
                    'local_mirror': mirror, 'path_at_audit': row['path'] if row['path'] != path
                    else None},
        'parent': {'branch_from': None, 'parent_run_id': None},
        'reproduces': None,
        'env': None, 'data': None,
        'hardware': {'gpu': row.get('gpu') or None, 'node': row.get('node') or None},
        'slurm': ({'job_id': job, 'state': row.get('job_state') or None,
                   'log': row.get('job_log') or None} if job else None),
        'submit': None,
        'gpu_hours': number(row['gpu_hours']), 'gpu_hours_basis': row['gpu_hours_basis'] or None,
        'progress': {'last_step': int(row['last_step']) if row['last_step'].isdigit() else None,
                     'checkpoints': None, 'n_checkpoints': int(row['n_ckpt'])
                     if row['n_ckpt'].isdigit() else None},
        'end': (None if status == 'running' else
                {'exit_status': None, 'reason': row['status'], 'error': None, 'elapsed_s': None,
                 'final_metrics': final,
                 'metrics_basis': 'the audit (final.json, or the last logged eval of a run '
                                  'without one), as of ' + AUDIT_AS_OF}),
        'audit': {'row': row['run_id'], 'experiment': exp['id'], 'gaps': row['gaps'] or None,
                  'notes': row['notes'] or None, 'source': 'experiments/audit/runs.csv'},
    }


def records():
    import yaml
    rows = {r['run_id']: r for r in csv.DictReader(open(AUDIT))}
    reg = yaml.safe_load(open(REGISTRY))
    moved = moved_paths()
    out = []
    for exp in reg['experiments']:
        if exp.get('status') != 'major':
            continue
        for entry in exp['runs']:
            if entry['id'] not in rows:
                raise SystemExit(f'{entry["id"]} ({exp["id"]}) is not in {AUDIT}')
            out.append(record(rows[entry['id']], exp, entry, moved))
    ids = [r['run_id'] for r in out]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise SystemExit(f'two registry runs map to one ledger id: {dup}')
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ledger', default=os.path.join(HERE, 'experiments', 'ledger'))
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    changed = 0
    recs = records()
    for rec in recs:
        path = os.path.join(a.ledger, rec['host'], rec['run_id'] + '.json')
        new = json.dumps(rec, indent=1) + '\n'
        old = open(path).read() if os.path.exists(path) else None
        if old != new:
            changed += 1
            if not a.check:
                write_json(path, rec)
    print(f'{len(recs)} retroactive records; {changed} {"would change" if a.check else "written"}')
    if a.check and changed:
        sys.exit(1)


if __name__ == '__main__':
    main()
