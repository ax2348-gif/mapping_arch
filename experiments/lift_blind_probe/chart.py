"""Counterfactual angle-chart probe (added after the main gdn result; post hoc).

    python experiments/lift_blind_probe/chart.py RUN_DIR [RUN_DIR ...]

The eigenangle probe found no rotating plane in the trained gdn, yet the
model composes continuous SE(2) on the training distribution. The residual
stream after block 0 at anchor tokens decodes the UNWRAPPED cumulative
heading sum(w) linearly (held-out R^2 ~0.95) and not its cos / sin: the model
lifts the circle to the line and composes heading as an abelian prefix sum.
This probe measures the chart w -> R that such a model uses:

1. Decoder. Ridge from the block-0 output at anchor_j to the unwrapped
   heading theta_j = w_1 + ... + w_j, fitted on walks from the model's own
   training distribution (granularity n, held-out band excluded), and
   scored on held-out episodes of that distribution.
2. Sweep. On base walks from that distribution, replace the turn of one
   step i (i = 8 and 16; speed kept, so dx, dy follow the SE(2) exp) by
   w' in a 1-degree grid over [-180, 180), band included, and read the
   decoded heading at anchor_j for j = i (the step itself) and j = i + 4
   (persistence). chart(w') = mean over base walks of
   decoded(w') - decoded(w' = 0).
   A homomorphism has chart(w') = w' (slope 1); a lookup is a staircase.
   The topology argument in the README says an exact continuous chart
   S^1 -> R does not exist: a jump (seam) of ~360 deg must sit somewhere.

Writes chart.npz / chart.json in the run directory.
"""
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(__file__))
import walk                                                  # noqa: E402
import probe                                                 # noqa: E402
from models import BlindProbe                                # noqa: E402
from encoding import encode                                  # noqa: E402

T, Q = 32, 16


@torch.no_grad()
def block0_at_anchors(model, odo, s, t):
    """Residual stream after block 0 at anchor_0..anchor_T: (B, T+1, w)."""
    cap = {}
    hk = model.blocks[0].register_forward_hook(lambda m, i, o: cap.__setitem__('h', o))
    model(encode(odo[..., :2], odo[..., 2]), s, t)
    hk.remove()
    _, _, kind, step = BlindProbe.layout(t, T)
    order = model.embed(encode(odo[..., :2], odo[..., 2]), s, t)[2]
    kind, step = torch.gather(kind, 1, order), torch.gather(step, 1, order)
    B = odo.shape[0]
    idx = torch.nonzero(kind == 0)[:, 1].view(B, T + 1)          # anchors, in step order
    assert (torch.gather(step, 1, idx) == torch.arange(T + 1, device=idx.device)).all()
    return torch.gather(cap['h'].float(), 1, idx[..., None].expand(-1, -1, cap['h'].shape[-1]))


def fit_decoder(model, n, band, episodes=4096, seed=31):
    g = torch.Generator(device='cuda').manual_seed(seed)
    bt = walk.batch(episodes, T, Q, n, band, g, 'cuda')
    H = block0_at_anchors(model, bt['odo'], bt['s'], bt['t']).cpu().numpy().astype(np.float64)
    th = np.concatenate([np.zeros((episodes, 1)), np.cumsum(bt['w'].cpu().numpy().astype(np.float64), 1)], 1)
    X = np.c_[H.reshape(-1, H.shape[-1]), np.ones(H.shape[0] * H.shape[1])]
    Y = np.stack([th.ravel(), np.cos(th).ravel(), np.sin(th).ravel()], 1)
    half = X.shape[0] // 2
    W = np.linalg.solve(X[:half].T @ X[:half] + 1e-3 * np.eye(X.shape[1]), X[:half].T @ Y[:half])
    P = X[half:] @ W
    r2 = 1 - ((P - Y[half:]) ** 2).sum(0) / ((Y[half:] - Y[half:].mean(0)) ** 2).sum(0)
    return W, dict(r2_unwrapped=float(r2[0]), r2_cos=float(r2[1]), r2_sin=float(r2[2]))


@torch.no_grad()
def sweep(model, W, n, band, base=128, seed=32):
    g = torch.Generator(device='cuda').manual_seed(seed)
    bt = walk.batch(base, T, Q, n, band, g, 'cuda')
    v = bt['odo'][..., :2].norm(dim=-1)                          # chord != v, recover v below
    w0 = bt['w']
    # recover the speed from the chord: |chord| = v |sinc(w/2)|
    sinc = torch.where(w0.abs() < 1e-6, torch.ones_like(w0), torch.sin(w0 / 2) / (w0 / 2)).abs()
    v = v / sinc
    grid = torch.deg2rad(torch.arange(-180, 180, 1.0, device='cuda'))
    out = {}
    for i in (8, 16):
        dec = []
        for wp in grid:
            w = w0.clone(); w[:, i - 1] = wp
            dx, dy, dth = walk.step_pose(v, w)
            odo = torch.stack([dx, dy, dth], -1)
            H = block0_at_anchors(model, odo, bt['s'], bt['t']).cpu().numpy().astype(np.float64)
            Hj = H[:, [i, i + 4]]
            dec.append(np.c_[Hj.reshape(-1, Hj.shape[-1]), np.ones(Hj.shape[0] * 2)] @ W[:, 0])
        dec = np.array(dec).reshape(len(grid), base, 2)               # (grid, base, j)
        ref = dec[np.argmin(np.abs(grid.cpu().numpy()))]
        out[i] = dec - ref[None]
    return np.degrees(grid.cpu().numpy()), out


def analyse(gd, ch):
    """Slope / residual of the mean chart at j = i, out of and in band, and
    the location and size of the largest jump."""
    c = np.degrees(ch.mean(1))                                     # (grid, j)
    res = {}
    for j, name in ((0, 'same_step'), (1, 'plus4')):
        y = c[:, j]
        inb = (gd >= walk.BAND_DEG[0]) & (gd <= walk.BAND_DEG[1])
        core = np.abs(gd) < 160
        r = {}
        for nm, m in (('out_of_band', ~inb & core), ('in_band', inb)):
            A = np.stack([gd[m], np.ones(m.sum())], 1)
            coef = np.linalg.lstsq(A, y[m], rcond=None)[0]
            r[nm] = dict(slope=float(coef[0]), intercept=float(coef[1]),
                         resid_rms_vs_identity=float(np.sqrt(np.mean((y[m] - gd[m]) ** 2))),
                         resid_mean_vs_identity=float(np.mean(y[m] - gd[m])))
        d = np.diff(y)
        k = int(np.argmax(np.abs(d)))
        r['largest_step_deg'] = float(d[k]); r['largest_step_at'] = float(gd[k] + 0.5)
        r['wrap_jump'] = float(y[0] - y[-1] + 1)                      # chart(-180) vs chart(179) + 1 deg
        res[name] = r
    return res


def main(runs):
    for run in runs:
        model, cfg = probe.load(run)
        n = cfg['n'] or None
        band = tuple(cfg['band']) if cfg['band'] else None
        W, fit = fit_decoder(model, n, band)
        gd, ch = sweep(model, W, n, band)
        summary = {'decoder': fit}
        for i, arr in ch.items():
            summary[f'step{i}'] = analyse(gd, arr)
        np.savez_compressed(f'{run}/chart.npz', grid=gd, **{f'step{i}': a for i, a in ch.items()})
        json.dump(summary, open(f'{run}/chart.json', 'w'), indent=1)
        print(run, json.dumps(summary, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
