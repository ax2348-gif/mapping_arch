"""How well-posed is visual odometry on the SE(2) task? With the odometry
symbols hidden (``--odometry-dropout 1``), the relative pose between two
consecutive frames can only come from their overlapping pixels, and MNIST
maps are mostly black. This counts, per consecutive pair, how many of the
16 (move, turn) hypotheses agree with both observations on their overlap,
and from that how many per-step queries are determined by the
observations alone. The numbers are a ceiling for any model.

    python tools/registrability.py --step-px 1 2 3 --episodes 4000

Definitions (all on the TEST bank, dihedral augmentation on, headings as
the SE(2) generator draws them, per-step queries):

- Hypothesis (m, k): frame t+1 lies at body move m (forward/right/back/
  left of frame t, ``step`` pixels) and is turned by k. Under it, frame
  t+1 seen in frame t's orientation is ``rot90(obs[t+1], -k)``, placed at
  ``step * MOVES[m]`` from frame t. It is *consistent* if every pixel of
  the overlap agrees EXACTLY (uint8 crops of one map, so the true
  hypothesis always agrees exactly; no tolerance is needed or used). All
  16 hypotheses compare an overlap of the same size, (9 - step) x 9.
- *Uniquely registrable* step: exactly one consistent hypothesis. Uses the
  pair alone, no walk prior; the "with no-reversal prior" line also drops
  the one move the walk forbids (immediate reversal), as if the previous
  step were known.
- *Blank overlap*: the true overlap has no stroke pixel (then every
  hypothesis whose overlap is also blank in both frames is consistent).
- A query asked at step t is *determined* when every pixel of its target
  was observed by some frame joined to frame t by a chain of uniquely
  registrable steps. Two versions: "offline" lets the chain run forward
  past t (the model sees the whole episode); "causal" uses frames <= t
  only. This is an approximation and, as a ceiling, a LOWER bound on what
  an ideal model can determine: a step that is ambiguous pairwise can
  still be resolved by a third frame overlapping both (frames t and t+2
  overlap whenever two steps move less than 9 px), and a query whose
  undetermined pixels happen to be blank under every hypothesis is still
  counted as undetermined. "anchor-only" queries lie inside frame t's own
  window: no registration needed at all.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mapping.data import MOVES, Config, load_bank, make_batch  # noqa: E402
from mapping.loader import EVAL_SPLIT  # noqa: E402


def consistent(obs, step):
    """``(B, T, 4 moves, 4 turns)`` bool: hypothesis agrees with frames t
    and t+1 on their overlap. ``obs`` is ``(B, T + 1, S, S)`` faced."""
    a, b = obs[:, :-1], obs[:, 1:]
    s = obs.shape[-1]
    out = np.empty(a.shape[:2] + (4, 4), bool)
    for k in range(4):
        bk = np.rot90(b, -k, axes=(-2, -1))
        for m in range(4):
            dy, dx = MOVES[m] * step
            # frame t+1 pixel (i, j) sits at frame t pixel (i + dy, j + dx)
            ya, yb = slice(max(0, dy), s + min(0, dy)), slice(max(0, -dy), s + min(0, -dy))
            xa, xb = slice(max(0, dx), s + min(0, dx)), slice(max(0, -dx), s + min(0, -dx))
            out[:, :, m, k] = (a[:, :, ya, xa] == bk[:, :, yb, xb]).all((-1, -2))
    return out


def true_overlap_blank(obs, move, step):
    a = obs[:, :-1]
    s = obs.shape[-1]
    blank = np.empty(move.shape, bool)
    for m in range(4):
        dy, dx = MOVES[m] * step
        ya = slice(max(0, dy), s + min(0, dy))
        xa = slice(max(0, dx), s + min(0, dx))
        nz = (a[:, :, ya, xa] == 0).all((-1, -2))
        blank[move == m] = nz[move == m]
    return blank


def segments(unique):
    """For each step t in 0..T, the first and last frame joined to t by
    uniquely registrable transitions: ``lo, hi (B, T + 1)``."""
    b, T = unique.shape
    lo = np.zeros((b, T + 1), int)
    hi = np.full((b, T + 1), T)
    for t in range(1, T + 1):
        lo[:, t] = np.where(unique[:, t - 1], lo[:, t - 1], t)
    for t in range(T - 1, -1, -1):
        hi[:, t] = np.where(unique[:, t], hi[:, t + 1], t)
    return lo, hi


def measure(step, episodes, batch, bank, seed):
    c = Config(heading=True, step=step, query_times='uniform', dihedral=True)
    rng = np.random.default_rng([seed, 2027, step])
    acc = {k: [] for k in ('n_cons', 'n_cons_prior', 'blank', 'det_off',
                           'det_causal', 'anchor', 'tblank', 'qtime',
                           'n_ep_all')}
    done = 0
    while done < episodes:
        n = min(batch, episodes - done)
        bt = make_batch(rng, c, bank, n)
        obs, odo = bt['observations'], bt['odometry']
        move, turn = odo % 4, odo // 4
        cons = consistent(obs, step)
        rows = np.arange(n)[:, None]
        cols = np.arange(c.steps)[None]
        assert cons[rows, cols, move, turn].all(), 'true hypothesis rejected'
        n_cons = cons.sum((-1, -2))
        # no-reversal prior: world dir (m + h_t) % 4 != reverse of d_{t-1}
        h = bt['headings']
        pos = bt['positions']
        d = np.argmax((np.sign(np.diff(pos, axis=1))[:, :, None, :]
                       == MOVES[None, None]).all(-1), -1)          # (n, T)
        forb = (d[:, :-1] + 2 - h[:, 1:-1]) % 4                     # body move
        prior = cons.copy()
        prior[rows[:, :1], np.arange(1, c.steps)[None], forb] = False
        acc['n_cons'].append(n_cons)
        acc['n_cons_prior'].append(prior.sum((-1, -2)))
        acc['blank'].append(true_overlap_blank(obs, move, step))
        unique = n_cons == 1
        acc['n_ep_all'].append(unique.all(1))
        lo, hi = segments(unique)
        # queries: which frames observe each target pixel
        for i, e in enumerate(bt['episodes']):
            p = e.positions                                         # (T+1, 2)
            for q, t in zip(e.queries, e.query_times):
                yy, xx = np.meshgrid(np.arange(c.obs) + q[0],
                                     np.arange(c.obs) + q[1], indexing='ij')
                px = np.stack([yy.ravel(), xx.ravel()], -1)         # (81, 2)
                inside = ((px[:, None] >= p[None]) &
                          (px[:, None] < p[None] + c.obs)).all(-1)  # (81, T+1)
                frames = np.arange(c.steps + 1)
                off = inside[:, (frames >= lo[i, t]) & (frames <= hi[i, t])]
                cau = inside[:, (frames >= lo[i, t]) & (frames <= t)]
                acc['det_off'].append(off.any(1).all())
                acc['det_causal'].append(cau.any(1).all())
                acc['anchor'].append(inside[:, t].all())
                acc['qtime'].append(t)
            acc['tblank'].append((bt['target'][i] == 0).all((-1, -2)))
        done += n
    return {k: np.concatenate([np.ravel(x) for x in v]) for k, v in acc.items()}


def report(step, r):
    n = r['n_cons']
    hist = {'1': (n == 1).mean(), '2': (n == 2).mean(),
            '3-15': ((n > 2) & (n < 16)).mean(), '16': (n == 16).mean()}
    q = r['qtime']
    lines = [
        f'step {step} px (overlap {9 - step} x 9): {len(r["n_ep_all"])} '
        f'episodes, {n.size} steps, {q.size} queries',
        f'  uniquely registrable steps      {np.mean(n == 1):.3f}'
        f'   (with no-reversal prior {np.mean(r["n_cons_prior"] == 1):.3f})',
        f'  blank overlap                   {r["blank"].mean():.3f}',
        f'  episodes with every step unique {r["n_ep_all"].mean():.4f}',
        '  consistent hypotheses per step  ' + ', '.join(
            f'{k}: {v:.3f}' for k, v in hist.items()) +
        f'; mean {n.mean():.2f}',
        f'  queries determined (offline)    {r["det_off"].mean():.3f}'
        f'   (causal, frames <= t: {r["det_causal"].mean():.3f})',
        f'  queries anchor-only             {r["anchor"].mean():.3f}'
        f'   (target entirely blank: {r["tblank"].mean():.3f})',
        f'  determined-or-blank (offline)   '
        f'{(r["det_off"] | r["tblank"]).mean():.3f}',
        '  determined (offline) by query-step bin: ' + ', '.join(
            f't{lo}-{hi}: {r["det_off"][(q >= lo) & (q < hi)].mean():.3f}'
            for lo, hi in ((0, 8), (8, 16), (16, 24), (24, 33))),
    ]
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--step-px', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--episodes', type=int, default=4000)
    p.add_argument('--batch', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--bank', default='data/mnist.npz')
    a = p.parse_args()
    bank = load_bank(a.bank, Config().cutoff, EVAL_SPLIT, True)[0]
    for step in a.step_px:
        if not 1 <= step < 9:
            raise SystemExit('step must be in 1..8 for frames to overlap')
        print(report(step, measure(step, a.episodes, a.batch, bank, a.seed)),
              flush=True)


if __name__ == '__main__':
    main()
