"""inert vs mixed query mode on se2d: eval curves at matched steps (from the
runs' log.jsonl) and, at the end, eval mse by query distance on one
held-out set, the first run's (final.pt of each run, fp32 eval).

    python experiments/inert_queries/compare.py REF_DIR [REF_DIR ...] --inert RUN_DIR

Distance split (se2d has none in its logs; se2c's near/mid/far is in px of a
15 px view): offsets on se2d are multiples of the 9 px step, so by distance
in steps d = |offset| / step_px: near d = 0 (the anchor cell, a turn at
most), mid 0 < d <= sqrt 2 (an adjacent cell), far d > sqrt 2.
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, '.')
import mapping.tasks as tasks  # noqa: E402
from mapping.checkpoint import load_checkpoint  # noqa: E402
from mapping.loader import eval_set, load_banks  # noqa: E402

KEYS = ('mse', 'mse_last', 'mse_t0_8', 'mse_t8_16', 'mse_t16_24', 'mse_t24_32')


def curve(d):
    out, rates = {}, []
    for line in open(os.path.join(d, 'log.jsonl')):
        r = json.loads(line)
        if 'eval' in r:
            out[r['step']] = {k: r['eval'].get(k) for k in KEYS}
            out[r['step']]['shuffled'] = r.get('eval_shuffled', {}).get('mse')
        if 'steps_per_s' in r and r['step'] >= 5000:
            rates.append(r['steps_per_s'])
    fin = json.load(open(os.path.join(d, 'final.json')))
    return out, fin.get('elapsed_s'), fin.get('steps')


@torch.no_grad()
def by_distance(ckpt, batches_cache={}):
    model, args, spec, task = load_checkpoint(ckpt, 'cuda')
    key = 'first'            # every checkpoint on the FIRST run's held-out set
    if key not in batches_cache:
        config = tasks.config_from_args(task, spec, args)
        _, bank = load_banks(args.get('bank') or 'data/mnist.npz', config, args.get('texture', 'mnist'))
        batches_cache[key] = (eval_set(config, bank, 64, 16, args['seed']), spec)
    batches, spec0 = batches_cache[key]
    prepare = tasks.get(task)['task'].prepare
    errs, dists = [], []
    for b in batches:
        obs, odo, pose, target, qt = prepare(b, 'cuda')
        pred = model(obs, odo, pose, query_time=qt).float()
        errs.append((pred - target).pow(2).mean((-1, -2)).flatten())
        dists.append((pose[..., :2].float().norm(dim=-1) / spec0.step_px).flatten())
    e, d = torch.cat(errs), torch.cat(dists)
    masks = {'near': d == 0, 'mid': (d > 0) & (d <= math.sqrt(2) + 1e-6), 'far': d > math.sqrt(2) + 1e-6}
    return {'mse': e.mean().item(),
            **{k: (e[m].mean().item(), m.float().mean().item()) for k, m in masks.items()}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('refs', nargs='+')
    p.add_argument('--inert', required=True)
    a = p.parse_args()
    runs = {os.path.basename(r.rstrip('/')) + ' (mixed)': r for r in a.refs}
    runs['inert'] = a.inert
    curves = {k: curve(v) for k, v in runs.items()}
    names = list(runs)
    print('eval mse at matched steps (mse_last in brackets)')
    print('step   ' + '  '.join(f'{n:>28s}' for n in names))
    steps = sorted(set.intersection(*(set(c[0]) for c in curves.values())))
    for s in steps:
        print(f'{s:6d} ' + '  '.join(f'{curves[n][0][s]["mse"]:.5f} [{curves[n][0][s]["mse_last"]:.5f}]'.rjust(28)
                                     for n in names))
    last = steps[-1]
    print(f'\nby query time at step {last}')
    for k in KEYS[2:] + ('shuffled',):
        print(f'{k:12s} ' + '  '.join(f'{curves[n][0][last][k]:.5f}'.rjust(12) for n in names))
    print('\nwall-clock (incl. eval): ' + '  '.join(
        f'{n}: {c[2] / c[1]:.1f} it/s' for n, c in curves.items()))
    print('\nfinal.pt on the same held-out set (fp32), mse by distance (share of queries)')
    for n in names:
        r = by_distance(os.path.join(runs[n], 'final.pt'))
        print(f'{n:28s} all {r["mse"]:.5f}  ' + '  '.join(
            f'{k} {r[k][0]:.5f} ({r[k][1]:.2f})' for k in ('near', 'mid', 'far')))


if __name__ == '__main__':
    main()
