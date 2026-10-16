"""se2c models (``mapping/models/se2c``): the contract's shape, episode
isolation, query_time None equal to all-last-step, the shared encoding
at the input, and the placement of query tokens (after observation t,
before odometry t, RoPE step t)."""
import math

import pytest
import torch

import mapping.tasks as tasks
from mapping.models.se2c.hybrid4c import (patchify, token_order,
                                          unpatchify)
from mapping.se2c.generator import Config, make_batch

CUDA = torch.cuda.is_available()
DEV = 'cuda' if CUDA else 'cpu'
T = tasks.get('se2c')
NAMES = sorted(T['models'].REGISTRY)


def build(name):
    module = T['models'].get(name)
    torch.manual_seed(0)
    net = module.build(next(iter(module.SIZES)), T['task'].TaskSpec())
    torch.nn.init.normal_(net.readout.weight, std=0.5)   # zero at init
    return net


@pytest.fixture(scope='module')
def batch():
    return make_batch(Config(), 0, DEV, batch=4)


@pytest.mark.parametrize('name', NAMES)
def test_shape_isolation_and_default_time(name, batch):
    if not CUDA and 'hybrid' in name:
        pytest.skip('the delta-rule kernel needs CUDA')
    net = build(name).to(DEV).eval()
    obs, odo, qp = batch['observations'], batch['odometry'], batch['query_pose']
    qt = batch['query_time']
    with torch.no_grad():
        out = net(obs, odo, qp, query_time=qt).float()
        assert out.shape == (4, 8, 15, 15)
        obs2, odo2 = obs.clone(), odo.clone()
        obs2[0] = torch.rand_like(obs2[0])
        odo2[0, :, 2] = -odo2[0, :, 2]
        out2 = net(obs2, odo2, qp, query_time=qt).float()
        assert (out[1:] - out2[1:]).abs().max() < 1e-4, 'episodes leak'
        assert (out[0] - out2[0]).abs().max() > 1e-3
        last = torch.full_like(qt, 32)
        a = net(obs, odo, qp).float()
        b = net(obs, odo, qp, query_time=last).float()
        assert (a - b).abs().max() < 1e-5


def test_patchify_roundtrip_and_layout():
    x = torch.arange(2 * 225).float().view(2, 15, 15)
    p = patchify(x)
    assert p.shape == (2, 25, 9)
    assert torch.equal(p[0, 1], x[0, 0:3, 3:6].reshape(-1))
    assert torch.equal(p[0, 5], x[0, 3:6, 0:3].reshape(-1))
    assert torch.equal(unpatchify(p, 15), x)


def test_placement():
    t1, q, P = 33, 3, 25
    L = t1 * (P + 1) - 1
    last = torch.full((1, q), 32)
    order = token_order(last, t1, q, P)
    assert torch.equal(order[0], torch.arange(L + q * P)), \
        'all queries at the last step: the original layout'
    qt = torch.tensor([[5, 0, 5]])
    order = token_order(qt, t1, q, P)[0]
    # layout indices: obs t patch i at t*(P+1)+i, odo t at t*(P+1)+P,
    # query k patch i at L + k*P + i
    pos_of = torch.empty_like(order)
    pos_of[order] = torch.arange(len(order))
    for k, t in enumerate(qt[0].tolist()):
        first = pos_of[L + k * P]
        assert torch.equal(pos_of[L + k * P: L + (k + 1) * P],
                           first + torch.arange(P)), 'a query is contiguous'
        last_obs = pos_of[t * (P + 1) + P - 1]
        odo = pos_of[t * (P + 1) + P]
        assert last_obs < first and pos_of[L + k * P + P - 1] < odo
    # two queries at one step keep query order, right after observation 5
    assert pos_of[L + 2 * P] == pos_of[L + P - 1] + 1
    assert pos_of[L] == pos_of[5 * (P + 1) + P - 1] + 1


@pytest.mark.parametrize('name', NAMES)
def test_rope_step_of_query_tokens(name, batch, monkeypatch):
    """The RoPE positions the blocks receive: a query's tokens carry its
    step, every context token its own step."""
    if not CUDA and 'hybrid' in name:
        pytest.skip('the delta-rule kernel needs CUDA')
    import mapping.models.se2c.hybrid4c as h
    seen = {}
    real = h.rope_cos_sin

    def spy(pos, dim):
        seen['pos'] = pos.clone()
        return real(pos, dim)
    monkeypatch.setattr(h, 'rope_cos_sin', spy)
    net = build(name).to(DEV).eval()
    qt = torch.tensor([[3, 17, 32, 0, 3, 9, 9, 30]] * 4, device=DEV)
    with torch.no_grad():
        net(batch['observations'], batch['odometry'], batch['query_pose'],
            query_time=qt)
    pos = seen['pos'][0]
    if getattr(net, 'sequence', 'sorted') == 'layout':
        # layout order [observations (33 x 25), odometry (32), queries (8 x 25)]
        steps = torch.arange(33, device=DEV)
        assert torch.equal(pos[:33 * 25], steps.repeat_interleave(25))
        assert torch.equal(pos[33 * 25:33 * 25 + 32], steps[:-1])
        assert torch.equal(pos[33 * 25 + 32:], qt[0].repeat_interleave(25))
        return
    order = token_order(qt, 33, 8, 25)[0]
    L = 33 * 26 - 1
    pos_of = torch.empty_like(order)
    pos_of[order] = torch.arange(len(order), device=DEV)
    for k, t in enumerate(qt[0].tolist()):
        assert (pos[pos_of[L + k * 25: L + (k + 1) * 25]] == t).all()
    for t in (0, 12, 32):
        assert (pos[pos_of[t * 26: t * 26 + 25]] == t).all()
    assert (pos.diff() >= 0).all(), 'the sorted sequence is in step order'


@pytest.mark.parametrize('name', NAMES)
def test_odometry_enters_through_the_shared_encoding(name, batch, monkeypatch):
    if not CUDA and 'hybrid' in name:
        pytest.skip('the delta-rule kernel needs CUDA')
    import mapping.models.se2c.hybrid4c as h
    calls = []
    real = h.encode_odometry
    monkeypatch.setattr(h, 'encode_odometry',
                        lambda d, t: calls.append(1) or real(d, t))
    net = build(name).to(DEV).eval()
    with torch.no_grad():
        a = net(batch['observations'], batch['odometry'], batch['query_pose'])
        odo = batch['odometry'].clone()
        odo[..., 2] += 2 * math.pi          # the encoding is 2 pi periodic
        b = net(batch['observations'], odo, batch['query_pose'])
    scale = a.float().abs().max()
    assert calls and (a.float() - b.float()).abs().max() < 1e-3 * scale
    with torch.no_grad():
        odo[..., 2] += 0.5
        c = net(batch['observations'], odo, batch['query_pose'])
    assert (a.float() - c.float()).abs().max() > 1e-3 * scale, 'turn used'
