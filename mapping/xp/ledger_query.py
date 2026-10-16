"""Reading the ledger for ``xp ls`` / ``xp show``: lookups, filters, and the
one-line-per-run listing (newest first)."""
import datetime
import os
import re

STATUS_LABEL = {'running': 'running', 'completed': 'done', 'failed': 'failed',
                'terminated': 'stopped'}


def parse_time(s):
    if not s:
        return None
    try:
        return datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))
    except ValueError:
        return None


def started(rec):
    """The best-known start time (retroactive records: the audit's estimate)."""
    return parse_time(rec.get('created'))


def since(spec, now=None):
    """``2d``, ``12h``, ``30m`` or an ISO date -> an aware datetime."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    m = re.fullmatch(r'(\d+(?:\.\d+)?)([dhm])', spec)
    if m:
        n, unit = float(m.group(1)), m.group(2)
        units = {'d': 'days', 'h': 'hours', 'm': 'minutes'}
        return now - datetime.timedelta(**{units[unit]: n})
    t = parse_time(spec)
    if t is None:
        raise SystemExit(f'--since {spec!r}: give 2d, 12h, 30m or an ISO date')
    return t if t.tzinfo else t.astimezone()


def tags(rec, registry_tags):
    return list(dict.fromkeys(list((rec.get('run') or {}).get('tags') or [])
                              + sorted(registry_tags.get(rec['run_id'], ()))))


def path_of(rec):
    r = rec.get('results') or {}
    return r.get('rel') or r.get('dir') or '-'


def matches(rec, ref):
    """``ref`` names this record: its id, an id prefix or suffix, its results
    path (absolute, or relative to the run root), or ``host:path``."""
    rid = rec['run_id']
    if ref == rid or rid.startswith(ref) or (len(ref) >= 6 and rid.endswith(ref)):
        return True
    r = rec.get('results') or {}
    cands = {r.get('dir'), r.get('rel')}
    if r.get('rel'):
        cands.add(f'{rec.get("host")}:{r["rel"]}')
        cands.add(os.path.join('results', r['rel']))
    audit = (rec.get('audit') or {}).get('row')
    if audit:
        cands.add(audit)
    ref_n = ref.rstrip('/')
    return ref_n in cands or os.path.abspath(ref_n) in cands


def find(records, ref):
    hits = [r for r in records if r['run_id'] == ref] or [r for r in records if matches(r, ref)]
    if not hits:
        raise SystemExit(f'no run {ref!r} in the ledger (xp ls lists them)')
    if len(hits) > 1:
        raise SystemExit(f'{ref!r} names {len(hits)} runs: ' +
                         ', '.join(h['run_id'] for h in hits[:8]) + '. Give more of the id')
    return hits[0]


def select(records, a, registry_tags):
    out = []
    t0 = since(a.since) if getattr(a, 'since', None) else None
    for r in records:
        run = r.get('run') or {}
        if a.running and r.get('status') != 'running':
            continue
        if a.tag and a.tag not in tags(r, registry_tags):
            continue
        if a.task and run.get('task') != a.task:
            continue
        if a.model and run.get('model') != a.model:
            continue
        if a.host and r.get('host') != a.host:
            continue
        if t0 is not None and (started(r) is None or started(r) < t0):
            continue
        out.append(r)
    far_past = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    out.sort(key=lambda r: (started(r) or far_past, r['run_id']), reverse=True)
    return out


def clip(s, n):
    s = ' '.join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + '…'


def clip_left(s, n):
    """Keep the end (a path's run name)."""
    s = str(s)
    return s if len(s) <= n else '…' + s[-(n - 1):]


def metric(rec):
    fm = ((rec.get('end') or {}).get('final_metrics')) or {}
    if fm.get('mse') is None:
        return ''
    out = f'{fm["mse"]:.4g}'
    if fm.get('mse_last') is not None and f'{fm["mse_last"]:.4g}' != out:
        out += f' last {fm["mse_last"]:.4g}'
    return out


def short_id(rec):
    return 'retro' if rec.get('retroactive') else rec['run_id'].rsplit('-', 1)[-1]


COLUMNS = (('START', 11), ('STATUS', 8), ('NOTE', 32), ('PATH', 36), ('HOST', 6), ('TAG', 12),
           ('MSE', 18), ('ID', 6))


def line(rec, registry_tags):
    t = started(rec)
    when = t.astimezone().strftime('%m-%d %H:%M') if t else '?'
    status = STATUS_LABEL.get(rec.get('status'), rec.get('status') or '?')
    if rec.get('retroactive'):
        status += '*'
    tag = ','.join(x for x in tags(rec, registry_tags) if x != 'major') or \
        ('major' if 'major' in tags(rec, registry_tags) else '')
    cells = [when, status, clip((rec.get('run') or {}).get('note') or '', 32),
             clip_left(path_of(rec), 36), rec.get('host') or '', clip(tag, 12), metric(rec),
             short_id(rec)]
    return '  '.join(c.ljust(w) for c, (_, w) in zip(cells, COLUMNS)).rstrip()


def header():
    return '  '.join(n.ljust(w) for n, w in COLUMNS).rstrip()
