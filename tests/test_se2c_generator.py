"""The se2c generator: every batch field agrees with the world poses and
the texture; turn granularity, held-out bands and determinism; and the
decoupled mode at n = 4 with grid steps reproduces the discrete SE(2)
walk's statistics."""
import math

import numpy as np
import pytest
import torch

from mapping import se2
from mapping.data import MOVES, Config as DConfig, load_bank, make_batch as dbatch
from mapping.se2c import geometry as geo
from mapping.se2c import texture as tex
from mapping.se2c.generator import Config, lattice, make_batch

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_fields_agree_with_poses_and_texture():
    c = Config()
    b = make_batch(c, 0, DEV, batch=8)
    pos, th = b['poses'][..., :2], b['poses'][..., 2]
    assert b['observations'].shape == (8, 33, 15, 15)
    assert b['odometry'].shape == (8, 32, 3) and b['query_pose'].shape == (8, 8, 3)
    # odometry integrates back to the poses
    p2, t2 = geo.integrate(b['odometry'][..., :2], b['odometry'][..., 2],
                           pos[:, 0], th[:, 0])
    assert torch.allclose(p2, pos, atol=1e-3)
    assert (b['odometry'][..., 2] >= -math.pi).all()
    assert (b['odometry'][..., 2] < math.pi).all()
    # query pose is the query relative to the pose at its step
    qt = b['query_time']
    ap = torch.gather(pos, 1, qt[..., None].expand(-1, -1, 2))
    ah = torch.gather(th, 1, qt)
    qw = b['query_world']
    back = ap + geo.unrotate(b['query_pose'][..., :2], ah)
    assert torch.allclose(back, qw[..., :2], atol=1e-3)
    assert torch.allclose(geo.wrap(ah + b['query_pose'][..., 2] - qw[..., 2]),
                          torch.zeros_like(ah), atol=1e-4)
    # views are the texture at the poses
    p = b['texture_params']
    assert torch.allclose(b['observations'],
                          tex.render(p, pos, th, 15, c.texture), atol=1e-5)
    assert torch.allclose(b['target'], tex.render(p, qw[..., :2], qw[..., 2], 15,
                                                  c.texture), atol=1e-5)
    # speeds within range
    speed = b['odometry'][..., :2].norm(dim=-1)
    assert speed.max() <= c.v_max + 1e-3


def test_deterministic_and_streams_differ():
    c = Config()
    a, b = make_batch(c, 3, DEV, batch=4), make_batch(c, 3, DEV, batch=4)
    for k in ('observations', 'odometry', 'query_pose', 'target'):
        assert torch.equal(a[k], b[k])
    e = make_batch(c, 3, DEV, batch=4, eval_set=True)
    assert not torch.equal(a['odometry'], e['odometry'])
    assert not torch.equal(a['odometry'], make_batch(c, 4, DEV, batch=4)['odometry'])


@pytest.mark.parametrize('n', [4, 8, 16, 32])
def test_granularity(n):
    c = Config(turn_n=n)
    b = make_batch(c, 0, DEV, batch=16)
    turn = b['odometry'][..., 2] * n / (2 * math.pi)
    assert (turn - turn.round()).abs().max() < 1e-3
    assert len(torch.unique(turn.round())) == n
    dq = b['query_pose'][..., 2] * n / (2 * math.pi)
    assert (dq - dq.round()).abs().max() < 1e-3
    assert len(lattice(n)) == n


def test_held_out_band():
    band = (0.5, 1.0)
    for n in (0, 32):
        tr = make_batch(Config(turn_n=n, holdout=band), 0, DEV)
        ev = make_batch(Config(turn_n=n, holdout=band, holdout_mode='only'), 0, DEV)
        inc = make_batch(Config(turn_n=n, holdout=band, holdout_mode='include'), 0, DEV)
        a = tr['odometry'][..., 2].abs()
        assert not ((a >= band[0] - 1e-5) & (a < band[1] - 1e-5)).any()
        a = ev['odometry'][..., 2].abs()
        assert ((a >= band[0] - 1e-5) & (a < band[1] + 1e-5)).all()
        a = inc['odometry'][..., 2].abs()
        assert ((a >= band[0]) & (a < band[1])).any()


def symbols_from_continuous(odo, step):
    body = odo[..., :2] / step
    k = torch.round(body).long().cpu().numpy()
    assert np.abs(body.cpu().numpy() - k).max() < 1e-3
    move = np.full(k.shape[:-1], -1)
    for d in range(4):
        move[(k == MOVES[d]).all(-1)] = d
    assert (move >= 0).all()
    q = odo[..., 2] / (math.pi / 2)
    turn = torch.round(q).long() % 4
    assert (q - torch.round(q)).abs().max() < 1e-4
    return 4 * turn.cpu().numpy() + move


def test_decoupled_grid_mode_reproduces_discrete_statistics():
    c = Config(motion='decoupled', turn_n=4, v_min=9.0, v_max=9.0)
    cont = np.concatenate([symbols_from_continuous(
        make_batch(c, i, DEV)['odometry'], 9.0).ravel() for i in range(4)])
    bank, _ = load_bank('data/mnist.npz', 32, 'train', True)
    disc = np.concatenate([dbatch(np.random.default_rng(i), DConfig(heading=True),
                                  bank, 64)['odometry'].ravel() for i in range(4)])
    hc = np.bincount(cont, minlength=16) / cont.size
    hd = np.bincount(disc, minlength=16) / disc.size
    assert 0.5 * np.abs(hc - hd).sum() < 0.03          # total variation
    # world directions: no immediate reversal, as in mapping.data.random_walk
    b = make_batch(c, 0, DEV)
    d = b['poses'][:, 1:, :2] - b['poses'][:, :-1, :2]
    rev = ((d[:, 1:] + d[:, :-1]).norm(dim=-1) < 1e-3)
    assert not rev.any()
    # headings are quarter turns, i.i.d. uniform
    h = torch.round(b['poses'][..., 2] / (math.pi / 2)).long() % 4
    freq = torch.bincount(h.flatten(), minlength=4).float() / h.numel()
    assert (freq - 0.25).abs().max() < 0.03
    # and every odometry symbol composes back to the poses, as in se2.compose
    sym = symbols_from_continuous(b['odometry'], 9.0)
    disp, hh = se2.compose(sym, 9, h0=h[:, 0].cpu().numpy())
    assert (hh == h.cpu().numpy()).all()
    p = (b['poses'][..., :2] - b['poses'][:, :1, :2]).cpu().numpy()
    assert np.abs(disp - p).max() < 1e-3


def test_bank_option_runs():
    from mapping.se2c import bank as tb
    c = Config(bank='mnist')
    b = make_batch(c, 0, DEV, batch=4, bank=tb.load('mnist', 'test', DEV))
    assert b['observations'].shape == (4, 33, 15, 15)
    assert 0 <= b['observations'].min() and b['observations'].max() <= 1
