"""Query coverage, the contract's hidden mine: every target sample point
of a query asked at step t lies in the footprints of observations 0..t
(causal) or 0..T (global), checked point by point against the exact
NumPy reference; the raster is conservative."""
import math

import numpy as np
import pytest
import torch

from mapping.se2c import geometry as geo
from mapping.se2c import reference as ref
from mapping.se2c.coverage import NEVER, first_seen
from mapping.se2c.generator import Config, make_batch, walk, generator

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def exact_check(batch, config):
    pos = batch['poses'][..., :2].cpu().numpy().astype(np.float64)
    th = batch['poses'][..., 2].cpu().numpy().astype(np.float64)
    qw = batch['query_world'].cpu().numpy().astype(np.float64)
    qt = batch['query_time'].cpu().numpy()
    bad = 0
    for b in range(pos.shape[0]):
        for q in range(qw.shape[1]):
            lim = qt[b, q] if config.query_coverage == 'causal' else config.steps
            pts = ref.view_points(qw[b, q, :2], qw[b, q, 2], config.view)
            hit = ref.covered(pts.reshape(-1, 2), pos[b, :lim + 1],
                              th[b, :lim + 1], config.half)
            bad += (~hit).sum()
    return bad


@pytest.mark.parametrize('kw', [dict(), dict(turn_n=8),
                                dict(query_coverage='global'),
                                dict(motion='arc'), dict(motion='arc', v_min=3.75, v_max=7.5),
                                dict(motion='decoupled', directions='lattice', turn_n=4, v_min=9.0, v_max=9.0),
                                dict(v_min=1.0, v_max=2.0)])
def test_every_target_point_is_covered(kw):
    c = Config(**kw)
    for index in range(2):
        batch = make_batch(c, index, DEV, batch=16)
        assert exact_check(batch, c) == 0


def test_causal_is_stricter_than_global():
    """A causal query is never outside the footprints up to its step, and
    some global queries are (else the variants would not differ)."""
    c = Config(query_coverage='global')
    batch = make_batch(c, 0, DEV, batch=32)
    assert exact_check(batch, Config()) > 0


def test_raster_is_conservative_and_tight():
    c = Config()
    g = generator(c, 0, DEV)
    pos, th = walk(g, c, 4, DEV)
    r = first_seen(pos, th, c.half, c.cell)
    lo, hi = pos.amin(1) - 12, pos.amax(1) + 12
    u = torch.rand(4, 20000, 2, generator=g, device=DEV)
    pts = lo[:, None] + u * (hi - lo)[:, None]
    seen = r.lookup(pts).cpu().numpy()
    P, T = pos.cpu().double().numpy(), th.cpu().double().numpy()
    pts = pts.cpu().double().numpy()
    missed = covered_total = 0
    for b in range(4):
        for t in (0, 5, 32):
            exact = ref.covered(pts[b], P[b, :t + 1], T[b, :t + 1], c.half,
                                eps=0)
            rast = seen[b] <= t
            assert not (rast & ~exact).any()            # never optimistic
            missed += (exact & ~rast).sum()
            covered_total += exact.sum()
    # the conservative margin loses well under 5% of the covered area
    assert missed / covered_total < 0.05


def test_fallbacks_are_rare_after_the_first_steps():
    c = Config()
    found, qt = [], []
    for i in range(4):
        b = make_batch(c, i, DEV)
        found.append(b['stats']['found']), qt.append(b['query_time'])
    found, qt = torch.cat(found), torch.cat(qt)
    assert (~found[qt >= 12]).float().mean() < 0.05
