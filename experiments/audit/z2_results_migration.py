"""Move the local z2 run directories results/<model>/... -> results/z2/<model>/...
(provenance design §14: full z2 path refactor). results/ is untracked, so a
plain move; every moved entry and every run directory in it is logged."""
import datetime, hashlib, json, os, sys
ROOT = 'results'
Z2 = ['hybrid', 'hybrid2', 'hybrid3', 'hybrid3_5', 'hybrid4', 'lstm', 'mlp',
      'vanilla', 'vanilla2', 'vanilla_depth', 'curves.png']
NOT_Z2 = {'se2', 'se2c', 'se2d', 'z2', 'ocean_media', 'session_archive_2026-10-16', 'jobs'}
manifest = sys.argv[1]
top = sorted(os.listdir(ROOT))
unknown = [t for t in top if t not in Z2 and t not in NOT_Z2]
assert not unknown, f'unclassified entries under results/: {unknown}'
# every run in a z2 entry must be a z2 run (task z2 or no task: pre-task runs were z2)
for d in Z2:
    for dp, _, fs in os.walk(os.path.join(ROOT, d)):
        for f in fs:
            if f in ('final.json', 'log.jsonl'):
                p = os.path.join(dp, f)
                rec = json.loads(open(p).readline()) if f == 'log.jsonl' else json.load(open(p))
                t = rec.get('task', rec.get('args', {}).get('task', 'z2'))
                assert t == 'z2', (p, t)
os.makedirs(os.path.join(ROOT, 'z2'), exist_ok=True)
assert not set(os.listdir(os.path.join(ROOT, 'z2'))) & set(Z2)
now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
rows = []
for d in Z2:
    src = os.path.join(ROOT, d)
    if not os.path.exists(src):
        continue
    runs = sorted(os.path.relpath(dp, ROOT) for dp, _, fs in os.walk(src)
                  if {'final.json', 'log.jsonl', 'config.yaml'} & set(fs))
    nfiles = sum(len(fs) for _, _, fs in os.walk(src)) if os.path.isdir(src) else 1
    dst = os.path.join(ROOT, 'z2', d)
    os.rename(src, dst)          # same filesystem: atomic, no copy
    rows.append((d, src, dst, nfiles, runs))
with open(manifest, 'w') as f:
    f.write(f'# z2 results migration, {now}, host erid-2, ~/code/mapping\n')
    f.write('# plain os.rename of each entry (results/ is untracked); runs moved with their entry\n')
    f.write('kind\told\tnew\tfiles\n')
    for d, src, dst, n, runs in rows:
        f.write(f'entry\t{src}\t{dst}\t{n}\n')
        for r in runs:
            f.write(f'run\t{ROOT}/{r}\t{ROOT}/z2/{r}\t\n')
print(len(rows), 'entries;', sum(len(r[4]) for r in rows), 'run directories')
