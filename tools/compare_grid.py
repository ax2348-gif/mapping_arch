"""Matched grids: several se2c checkpoints on the SAME held-out queries,
optionally filtered to one query kind (e.g. far only).

    python tools/compare_grid.py A.pt B.pt --labels hybrid4c vanilla4c \
        --kind far --rows 16 --out cmp_far.png
    python tools/compare_grid.py A.pt --kind far --out far_only.png

Each row is one query: the observation at the query's step, the target, then
per checkpoint its prediction its prediction with the episode's odometry
permuted in time (the use-of-odometry check; --no-shuffled omits it) and |error|, each labelled with its mse.
Rows are drawn from the evaluation set of the FIRST checkpoint's settings;
every checkpoint must share those task settings (refused otherwise), so the
comparison is matched. --sort dist orders rows by distance from the anchor
(farthest first) instead of a seeded random draw. Query kinds are
mapping.task_se2c.QUERY_KINDS: fallback, near (< 7.5 px), mid (< 15 px),
far (>= 15 px).
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, '.')
import mapping.tasks as tasks  # noqa: E402
from mapping.checkpoint import load_checkpoint  # noqa: E402
from mapping.render import GAP, SCALE, _font, _to_image  # noqa: E402

# settings that must agree for a matched comparison (the model may differ)
TASK_KEYS = ('task', 'episode_steps', 'queries', 'query_times', 'seed',
             'se2c_texture', 'turn_n', 'motion', 'directions', 'v_min',
             'v_max', 'omega_max', 'holdout', 'holdout_mode', 'query_coverage')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoints', nargs='+')
    p.add_argument('--labels', nargs='+', default=None)
    p.add_argument('--kind', default=None,
                   help='fallback | near | mid | far (default: all kinds)')
    p.add_argument('--rows', type=int, default=16)
    p.add_argument('--sort', default='random', choices=('random', 'dist'))
    p.add_argument('--no-shuffled', dest='shuffled', action='store_false',
                   help='omit the column with each model\'s odometry permuted in time (shown by default)')
    p.add_argument('--batches', type=int, default=4, help='evaluation batches to draw from')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    labels = a.labels or [os.path.basename(os.path.dirname(c)) for c in a.checkpoints]
    assert len(labels) == len(a.checkpoints), '--labels must match the checkpoints'

    loaded = [load_checkpoint(c, a.device) for c in a.checkpoints]
    _, args0, _, task = loaded[0]
    T = tasks.get(task)['task']
    if not hasattr(T, 'query_kinds'):
        raise SystemExit(f'{task}: compare_grid needs query kinds (se2c)')
    for (_, args, _, t), c in zip(loaded[1:], a.checkpoints[1:]):
        diff = {k: (args0.get(k), args.get(k)) for k in TASK_KEYS if args0.get(k) != args.get(k)}
        if t != task or diff:
            raise SystemExit(f'{c}: task settings differ from {a.checkpoints[0]}: {diff or (task, t)}')

    batches = T.data(args0, a.device, eval_batches=a.batches, batch=64)['held_out']
    gen = torch.Generator().manual_seed(0)
    rows = []   # (obs, target, [pred], [shuffled], offset, t, kind, dist)
    for b in batches:
        obs, odo, qp, tgt, qt = T.prepare(b, a.device)
        perm = torch.argsort(torch.rand(odo.shape[:2], generator=gen), 1).to(a.device)
        odo_s = torch.gather(odo, 1, perm[..., None].expand_as(odo))
        preds, shufs = [], []
        with torch.no_grad():
            for model, *_ in loaded:
                preds.append(model(obs, odo, qp, query_time=qt).float().cpu())
                shufs.append(model(obs, odo_s, qp, query_time=qt).float().cpu())
        kinds = T.query_kinds(b)
        for i in range(obs.shape[0]):
            for q in range(qp.shape[1]):
                if a.kind and kinds[i][q] != a.kind:
                    continue
                t = int(qt[i, q])
                rows.append((obs[i, t].cpu(), tgt[i, q].cpu(), [x[i, q] for x in preds],
                             [x[i, q] for x in shufs], qp[i, q].cpu(), t, kinds[i][q],
                             float(qp[i, q, :2].norm())))
    if not rows:
        raise SystemExit(f'no queries of kind {a.kind!r} in {a.batches} batches')
    if a.sort == 'dist':
        pick = sorted(range(len(rows)), key=lambda k: -rows[k][-1])[:a.rows]
    else:
        pick = np.random.default_rng(a.seed).choice(len(rows), size=min(a.rows, len(rows)), replace=False)

    # Layout: [label | obs | target] then one GROUP per model: prediction,
    # odometry shuffled (default), |error|. Groups are separated by a wider gap
    # and headed by the model's name; each cell carries its mse underneath; the
    # lower mse on each row is drawn in bold white, the others in grey.
    cell = 15 * SCALE
    sub = (('prediction', 'odometry shuffled', '|error|') if a.shuffled
           else ('prediction', '|error|'))
    group_gap, cap = 22, 18                      # gap between groups; caption height
    label_w, header = 200, 92
    big, mid, small = _font(16), _font(13), _font(12)
    base_w = 2 * (cell + GAP)
    group_w = len(sub) * (cell + GAP)
    width = label_w + base_w + len(labels) * (group_gap + group_w)
    row_h = cell + cap + GAP
    canvas = Image.new('L', (width, header + len(pick) * row_h + 8), 32)
    d = ImageDraw.Draw(canvas)
    # mean mse per model over ALL filtered queries, not just the drawn rows
    means = [np.mean([float((r[2][m] - r[1]).pow(2).mean()) for r in rows]) for m in range(len(labels))]
    kind_txt = a.kind or 'all'
    d.text((8, 6), f"{' vs '.join(labels)}  —  {kind_txt} queries", fill=255, font=big)
    d.text((8, 30), f"mean mse over all {len(rows)} {kind_txt} queries: " +
           "   ".join(f"{lab} {mu:.4f}" for lab, mu in zip(labels, means)) +
           f"     (rows shown: {len(pick)}, {'farthest first' if a.sort == 'dist' else 'random sample'})",
           fill=225, font=mid)
    x0 = label_w
    for c, name in enumerate(('observation at t', 'target')):
        d.text((x0 + c * (cell + GAP), header - 18), name, fill=210, font=small)
    gx = []
    for m, lab in enumerate(labels):
        x = label_w + base_w + group_gap + m * (group_gap + group_w)
        gx.append(x)
        d.rectangle([x - 4, header - 44, x + group_w - GAP + 4, header - 24], fill=70)
        d.text((x, header - 42), lab, fill=255, font=mid)
        for c, name in enumerate(sub):
            d.text((x + c * (cell + GAP), header - 18), name, fill=210, font=small)
    for r, k in enumerate(pick):
        o, tg, prs, shs, off, t, kind, dist = rows[k]
        y = header + r * row_h
        errs = [float((prs[m] - tg).pow(2).mean()) for m in range(len(labels))]
        best = int(np.argmin(errs))
        d.text((8, y + 4), f"step {t}", fill=235, font=mid)
        d.text((8, y + 24), f"offset ({float(off[0]):+.1f}, {float(off[1]):+.1f})", fill=210, font=small)
        d.text((8, y + 40), f"turn {math.degrees(float(off[2])):+.0f} deg", fill=210, font=small)
        d.text((8, y + 56), f"{kind}, {dist:.1f} px from anchor", fill=210, font=small)
        canvas.paste(_to_image(o), (x0, y))
        canvas.paste(_to_image(tg), (x0 + cell + GAP, y))
        for m in range(len(labels)):
            imgs = [prs[m]] + ([shs[m]] if a.shuffled else []) + [(prs[m] - tg).abs()]
            mses = [errs[m]] + ([float((shs[m] - tg).pow(2).mean())] if a.shuffled else []) + [None]
            for c, (im, e) in enumerate(zip(imgs, mses)):
                cx = gx[m] + c * (cell + GAP)
                canvas.paste(_to_image(im), (cx, y))
                if e is not None:
                    win = (c == 0 and m == best)
                    d.text((cx + 2, y + cell + 2), f"mse {e:.3f}", fill=255 if win else 150,
                           font=mid if win else small)
    canvas.save(a.out)
    print(f'wrote {a.out}: {len(pick)} rows; mean mse ' + ', '.join(f'{l} {m:.4f}' for l, m in zip(labels, means)))


if __name__ == '__main__':
    main()
