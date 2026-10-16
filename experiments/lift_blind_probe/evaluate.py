"""Held-out-angle evaluation with a mirror-matched control (post hoc).

    python experiments/lift_blind_probe/evaluate.py RUN_DIR [RUN_DIR ...]

The in-training 'band' vs 'ctrl' sets are confounded: walks whose every
turn is in [50, 70] deg accumulate ~2x the unwrapped heading of walks in
[20, 40], and the gdn turned out to carry heading as an unwrapped real
number (chart.py). The held-out band is one-sided, so its mirror
[-70, -50] is trained and has identical statistics by symmetry: that is
the matched control here.

Sets (fixed seeds, 2048 episodes x 16 queries):
  const_band    every turn uniform in [50, 70]
  const_mirror  every turn uniform in [-70, -50]
  mix_band      each turn: with p = 0.25 uniform in [50, 70], else the
                run's training distribution
  mix_mirror    the same with [-70, -50]
For mix sets, queries are also split by whether their span (s, t] holds
any band turn. Writes heldout.json in the run directory.
"""
import json
import math
import os
import sys

import torch

sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(__file__))
import walk                                                  # noqa: E402
import probe                                                 # noqa: E402
import importlib.util                                         # noqa: E402
_spec = importlib.util.spec_from_file_location('bp_train', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py'))
_train = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_train)
feats_of, LAG_BINS = _train.feats_of, _train.LAG_BINS

T, Q = 32, 16
MIRROR = (-walk.BAND_DEG[1], -walk.BAND_DEG[0])


def make(B, n, band, mode, lohi, gen):
    bt = walk.batch(B, T, Q, n, band, gen, 'cuda')
    w = bt['w']
    special = torch.deg2rad(lohi[0] + (lohi[1] - lohi[0]) * torch.rand(B, T, generator=gen, device='cuda'))
    hit = torch.ones_like(w, dtype=torch.bool) if mode == 'const' else (
        torch.rand(B, T, generator=gen, device='cuda') < 0.25)
    w = torch.where(hit, special, w)
    v = walk.V_MIN + (walk.V_MAX - walk.V_MIN) * torch.rand(B, T, generator=gen, device='cuda')
    dx, dy, dth = walk.step_pose(v, w)
    p, th = walk.integrate(dx, dy, dth)
    s, t = bt['s'], bt['t']
    rx, ry, rth = walk.relative(p, th, s, t)
    scale = walk.STEP_RMS * (t - s).float().sqrt()
    # does span (s, t] contain a special turn? turn of step i is w[:, i-1]
    c = torch.cat([torch.zeros(B, 1, device='cuda'), hit.float().cumsum(1)], 1)
    span_hit = (torch.gather(c, 1, t) - torch.gather(c, 1, s)) > 0
    return dict(odo=torch.stack([dx, dy, dth], -1), s=s, t=t, scale=scale,
                tgt=torch.stack([rx / scale, ry / scale, rth.cos(), rth.sin()], -1),
                rel=torch.stack([rx, ry, rth], -1), span_hit=span_hit)


@torch.no_grad()
def errors(model, bt, mask=None):
    preds = []
    for i in range(0, bt['odo'].shape[0], 512):
        sl = slice(i, i + 512)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            preds.append(model(feats_of(bt['odo'][sl]), bt['s'][sl], bt['t'][sl]).float())
    pred = torch.cat(preds)
    lag = bt['t'] - bt['s']
    pos = ((pred[..., :2] - bt['tgt'][..., :2]) * bt['scale'][..., None]).pow(2).sum(-1)
    hd = torch.rad2deg(walk.wrap(torch.atan2(pred[..., 3], pred[..., 2]) - bt['rel'][..., 2])).abs()
    m0 = torch.ones_like(lag, dtype=torch.bool) if mask is None else mask
    r = {}
    for lo, hi in LAG_BINS:
        m = m0 & (lag >= lo) & (lag <= hi)
        r[f'pos_{lo}_{hi}'] = float(pos[m].mean().sqrt())
        r[f'head_{lo}_{hi}'] = float(hd[m].median())
        r[f'head90_{lo}_{hi}'] = float(hd[m].quantile(0.9))
        r[f'n_{lo}_{hi}'] = int(m.sum())
    return r


def main(runs):
    for run in runs:
        model, cfg = probe.load(run)
        n = cfg['n'] or None
        band = tuple(cfg['band']) if cfg['band'] else None
        out = {}
        for k, (mode, lohi) in enumerate((('const', walk.BAND_DEG), ('const', MIRROR),
                                          ('mix', walk.BAND_DEG), ('mix', MIRROR))):
            name = f"{mode}_{'band' if lohi == walk.BAND_DEG else 'mirror'}"
            gen = torch.Generator(device='cuda').manual_seed(500 + (k >= 2))   # band and its mirror share a seed
            bt = make(2048, n, band, mode, lohi, gen)
            out[name] = errors(model, bt)
            if mode == 'mix':
                out[name + '_spanhit'] = errors(model, bt, bt['span_hit'])
        json.dump(out, open(f'{run}/heldout.json', 'w'), indent=1)
        print(run)
        for name, r in out.items():
            print(f"  {name:22s} " + '  '.join(f"lag{b}: pos {r['pos_' + b]:.3f} head {r['head_' + b]:.2f}"
                                               for b in ('1_1', '5_8', '17_32')))


if __name__ == '__main__':
    main(sys.argv[1:])
