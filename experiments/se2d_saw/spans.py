"""Per-query loop-closure and composition-span bookkeeping on the step-9
lattice (path_stats.py, eval_split.py). With step == obs each pixel lies in
exactly one lattice cell and the window at step s covers exactly its cell.
For a query at step tq over the cells its target overlaps:

    reobserved   some target cell was visited twice by tq (loop closure)
    seam_far     the cells' latest visits are more than one step apart
    span_needed  tq minus the oldest of the cells' latest visits: the
                 longest chain to compose if each piece is read from its
                 most recent view
    span_first   the same with first visits (no re-anchoring)
    dist         Euclidean |(dy, dx)| of the query pose, px
    cells        number of cells the target overlaps (1, 2 or 4)
"""
import numpy as np

BINS = (0, 4, 12, 33)          # span bins: near < 4, mid 4..11, far >= 12
NAMES = ('near', 'mid', 'far')


def query_stats(b, step, obs):
    assert step == obs, 'the cell bookkeeping needs step == obs'
    out = {k: [] for k in ('reobserved', 'seam_far', 'span_needed',
                           'span_first', 'dist', 'cells')}
    for i, e in enumerate(b['episodes']):
        p0 = e.positions[0]
        cells = [tuple(c) for c in ((e.positions - p0) // step).tolist()]
        for q, tq in zip(e.queries, e.query_times):
            r = q - p0
            ys = sorted({r[0] // step, (r[0] + obs - 1) // step})
            xs = sorted({r[1] // step, (r[1] + obs - 1) // step})
            visits = {(y, x): [s for s in range(tq + 1) if cells[s] == (y, x)]
                      for y in ys for x in xs}
            assert all(visits.values()), 'a target cell was never seen'
            latest = [max(v) for v in visits.values()]
            first = [min(v) for v in visits.values()]
            out['reobserved'].append(any(len(v) > 1 for v in visits.values()))
            out['seam_far'].append(max(latest) - min(latest) > 1)
            out['span_needed'].append(tq - min(latest))
            out['span_first'].append(tq - min(first))
            out['cells'].append(len(visits))
        pose = b['query_pose'][i, :, :2].astype(float)
        out['dist'].extend(np.hypot(pose[:, 0], pose[:, 1]).tolist())
    return {k: np.array(v) for k, v in out.items()}
