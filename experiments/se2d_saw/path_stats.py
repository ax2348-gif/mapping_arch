"""Loop-closure risk of the se2d walk, and what the constrained paths change.

    python experiments/se2d_saw/path_stats.py OUT.json [EPISODES]

For each path (mapping.data.PATHS) and each query-time rule, over EPISODES
se2d episodes (default 10240, batches of 64, seeds [1000, i]):

- revisits: steps whose lattice cell was visited before; the fraction of
  episodes with at least one, and the mean count.
- late_contact: episodes where some footprint overlaps or shares an edge
  with the footprint of a step more than view_gap earlier (what
  'no_revisit_view' forbids).
- per query (the step-9 lattice: each pixel lies in exactly one cell, and
  the window at step s covers exactly its cell), over the cells the target
  overlaps and their visits up to the query step tq:
    reobserved   some target cell was visited twice by tq: the target
                 ground was re-observed after a gap, a loop closure
    seam_far     the target spans cells whose latest visits are more than
                 one step apart: its pieces abut across a seam between
                 non-consecutive steps (parallel segments)
    span_needed  tq minus the oldest of the cells' latest visits: the
                 longest composition chain needed if every piece is read
                 from its most recent view
    span_first   the same with first visits: the chain with no re-anchoring
    dist         Euclidean |(dy, dx)| of the query pose, px
- the walk itself: world-frame turn histogram (straight, right, left),
  end-to-end distance in steps, and the body-frame odometry move and turn
  marginals.
- exactness: the turn histogram of the constrained sampler against plain
  walks filtered by the same acceptance test (total variation).
- acceptance rate per path (plain walks passing the test) and generator
  time per batch of 64 (one process).
"""
import json
import sys
import time

import numpy as np

from spans import BINS, query_stats
from mapping.data import (PATHS, Config, accept, constrained_walk,
                          load_bank, make_batch, random_walk, view_gap)

OUT = sys.argv[1]
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 10240
B = 64


def walk_stats(directions, displacement, step):
    rel = (directions[:, 1:] - directions[:, :-1]) % 4      # 0 straight, 1 right, 3 left
    turn = np.array([(rel == k).mean() for k in (0, 1, 3)])
    end = np.abs(displacement[:, -1]).sum(-1) / step
    return turn, end


def summarize(q):
    s = {'reobserved': q['reobserved'].mean(), 'seam_far': q['seam_far'].mean(),
         'span_needed_mean': q['span_needed'].mean(),
         'span_first_mean': q['span_first'].mean(),
         'dist_mean': q['dist'].mean(),
         'dist_q50_q90': np.quantile(q['dist'], [0.5, 0.9]).tolist(),
         'cells_hist': [(q['cells'] == k).mean() for k in (1, 2, 4)]}
    for name in ('span_needed', 'span_first'):
        h = np.histogram(q[name], BINS)[0] / len(q[name])
        s[name + '_near_mid_far'] = h.tolist()
    return {k: (float(v) if np.isscalar(v) else v) for k, v in s.items()}


bank = load_bank('data/mnist.npz', 32, 'test', True)[0]
res = {'episodes': EPISODES, 'span_bins': BINS}
# acceptance of plain walks, 2M of them
rng = np.random.default_rng(7)
d, disp = random_walk(rng, 2_000_000, 32, 9)
for path in PATHS[1:]:
    res[f'acceptance_{path}'] = float(accept(disp, path, 9, 9).mean())
res['view_gap_k'] = view_gap(9, 9)
# exactness of the per-step marginals: sampler vs filtered plain walks
for path in PATHS[1:]:
    ok = accept(disp, path, 9, 9)
    t_filter, e_filter = walk_stats(d[ok], disp[ok], 9)
    ds, dps = constrained_walk(np.random.default_rng(8), int(ok.sum()), 32, 9, 9, path)
    t_samp, e_samp = walk_stats(ds, dps, 9)
    res[f'exactness_{path}'] = {
        'n': int(ok.sum()), 'turn_filtered': t_filter.tolist(),
        'turn_sampled': t_samp.tolist(),
        'turn_tv': float(0.5 * np.abs(t_filter - t_samp).sum()),
        'end_mean_filtered': float(e_filter.mean()),
        'end_mean_sampled': float(e_samp.mean())}

for path in PATHS:
    for qt in ('uniform', 'last'):
        c = Config(heading=True, query_times=qt, path=path)
        qs, rev, late, tgen = [], [], [], []
        dirs, disps, odo = [], [], []
        for i in range(EPISODES // B):
            t0 = time.perf_counter()
            b = make_batch(np.random.default_rng([1000, i]), c, bank, B)
            tgen.append(time.perf_counter() - t0)
            pos = b['positions'].astype(np.int64)
            d_ = np.array([e.positions for e in b['episodes']])
            keys = [[tuple(x) for x in p.tolist()] for p in pos]
            rev.extend(sum(1 for t in range(len(k)) if k[t] in k[:t]) for k in keys)
            late.extend((~accept(pos - pos[:, :1], 'no_revisit_view', 9, 9)
                         | ~accept(pos - pos[:, :1], 'saw', 9, 9)).tolist())
            qs.append(query_stats(b, 9, 9))
            disps.append(pos - pos[:, :1])
            dirs.append(np.argmax((np.diff(pos, axis=1)[..., None, :] // 9
                                   == np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])).all(-1), -1))
            odo.append(b['odometry'])
        q = {k: np.concatenate([x[k] for x in qs]) for k in qs[0]}
        rev = np.array(rev)
        turn, end = walk_stats(np.concatenate(dirs), np.concatenate(disps), 9)
        o = np.concatenate(odo)
        res[f'{path}/{qt}'] = {
            'revisit_episode_frac': float((rev > 0).mean()),
            'revisits_mean': float(rev.mean()),
            'late_contact_episode_frac': float(np.mean(late)),
            'turn_straight_right_left': turn.tolist(),
            'end_to_end_steps_mean': float(end.mean()),
            'end_to_end_steps_q10_q50_q90': np.quantile(end, [.1, .5, .9]).tolist(),
            'odometry_move_hist': np.bincount((o % 4).ravel(), minlength=4).tolist(),
            'odometry_turn_hist': np.bincount((o // 4).ravel(), minlength=4).tolist(),
            'gen_ms_per_batch_mean': 1000 * float(np.mean(tgen[1:])),
            'gen_ms_per_batch_p95': 1000 * float(np.quantile(tgen[1:], .95)),
            **summarize(q)}
        print(path, qt, json.dumps(res[f'{path}/{qt}']), flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if '/' not in k}, indent=1))
