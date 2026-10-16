"""Mechanism trace for the time-only recurrence (hybrid4_time and
hybrid4_time_full on se2; hybrid4c_time and hybrid4c_time_full on se2c)."""
import math

import pytest
import torch

import mapping.tasks as tasks
from mapping.models.se2d.hybrid4 import HEAD_DIM
from mapping.models.se2d.hybrid4_time import (TimeScan, delta_product,
                                             time_context, time_layout)

CUDA = torch.cuda.is_available()


def spec(task):
    return tasks.config_and_spec(task, 32, 8, 0, 'uniform')[1]


def test_layout_places_reader_queries_odometry_per_step():
    qt = torch.tensor([[0, 5, 5, 32, 31, 2, 2, 7]])
    order, step, kind = time_layout(qt, 33)
    assert step.shape == (1, 33 + 8 + 32)
    key = step[0] * 3 + kind[0]
    assert (key[1:] >= key[:-1]).all()                    # sorted by (step, kind)
    assert (kind[0] == 2).sum() == 32 and (kind[0] == 0).sum() == 33
    assert kind[0, -1] == 0 or kind[0, -1] == 1           # nothing writes after step 32


def test_readers_and_queries_cannot_change_the_state():
    torch.manual_seed(0)
    ts = TimeScan(64, 2, 4.0, 0.06, 0.03)
    qt = torch.randint(0, 33, (2, 8))
    _, _, kind = time_layout(qt, 33)
    write = kind == 2
    x = torch.randn(2, kind.shape[1], 64)
    y = ts(x, {'write': write})
    x2 = x.clone()
    x2[~write] += torch.randn_like(x2[~write]) * 3        # perturb every read-only token
    y2 = ts(x2, {'write': write})
    # odometry tokens' outputs depend only on the state and their own query
    assert torch.allclose(y[write], y2[write], atol=1e-5)
    q, k, v, g, beta = ts.quantities(x, write)
    assert (beta[~write] == 0).all() and (g[~write] == 0).all()
    assert (beta[write] > 0).all() and (beta[write] < 2).all()


def test_injection_maps_each_time_token_to_its_tokens():
    t1, P, q = 33, 9, 8
    n_ctx = t1 * 10 - 1
    steps = torch.arange(t1)
    obs_idx = steps[:, None] * 10 + torch.arange(P)
    odo_idx = steps[:-1] * 10 + 9
    qry_idx = n_ctx + torch.arange(q * P).view(q, P)
    qt = torch.randint(0, 33, (1, q))
    ctx = time_context(obs_idx, odo_idx, qry_idx, qt, n_ctx + q * P)
    hn = torch.zeros(1, n_ctx + q * P, 1)
    hn[0, obs_idx.reshape(-1), 0] = steps.repeat_interleave(P).float()   # patch = its step
    hn[0, odo_idx, 0] = 100 + steps[:-1].float()
    hn[0, qry_idx.reshape(-1), 0] = 1000 + torch.arange(q).repeat_interleave(P).float()
    tt = ctx['gather'](hn)[0, :, 0]
    kind, step = ctx['tkind'][0], ctx['tstep'][0]
    assert torch.equal(tt[kind == 0], step[kind == 0].float())          # reader = mean of its patches
    assert torch.equal(tt[kind == 2], 100 + step[kind == 2].float())
    back = ctx['scatter'](tt[None, :, None])[0, :, 0]
    assert torch.equal(back[obs_idx.reshape(-1)], steps.repeat_interleave(P).float())
    assert torch.equal(back[odo_idx], 100 + steps[:-1].float())
    assert torch.equal(back[qry_idx.reshape(-1)],
                       1000 + torch.arange(q).repeat_interleave(P).float())


@pytest.mark.parametrize('phi', [0.3, math.pi / 4, 1.2])
def test_delta_product_pair_is_a_rotation_by_twice_the_key_angle(phi):
    d = HEAD_DIM
    k1 = torch.zeros(d); k1[0] = 1
    k2 = torch.zeros(d); k2[0], k2[1] = math.cos(phi), math.sin(phi)
    k = torch.stack([k1, k2])[None, :, None]                 # (1, 2, 1, d): one token, n=2
    v = torch.zeros(1, 2, 1, d)
    beta = torch.full((1, 2, 1), 2.0)
    g = torch.zeros(1, 1, 1)
    q = torch.zeros(1, 1, 1, d)
    from fla.ops.gated_delta_product.naive import naive_recurrent_gated_delta_product
    _, h = naive_recurrent_gated_delta_product(q, k, v, g, beta, 1.0, None,
                                               initial_state=torch.eye(d)[None, None],
                                               num_householder=2)
    T = h[0, 0]
    assert torch.allclose(torch.linalg.det(T), torch.tensor(1.0), atol=1e-5)
    ang = torch.angle(torch.linalg.eigvals(T[:2, :2])).abs().max()
    assert abs(ang.item() - 2 * phi) < 1e-5
    assert torch.allclose(T[2:, 2:], torch.eye(d - 2), atol=1e-6)    # identity off the plane


@pytest.mark.skipif(not CUDA, reason='FLA kernel needs CUDA')
def test_kernel_matches_reference():
    torch.manual_seed(0)
    b, l, h, d, n = 2, 73, 4, HEAD_DIM, 2
    q = torch.nn.functional.normalize(torch.randn(b, l, h, d), dim=-1) * d ** -0.5
    k = torch.nn.functional.normalize(torch.randn(b, l * n, h, d), dim=-1)
    v = torch.randn(b, l * n, h, d)
    g = -torch.rand(b, l, h) * 0.05
    beta = torch.rand(b, l * n, h) * 2
    ref = delta_product(q, k, v, g, beta, n)
    out = delta_product(*(t.cuda().bfloat16() if t.dtype == torch.float32 and t is not g
                          else t.cuda() for t in (q, k, v)), g.cuda(), beta.cuda(), n)
    assert (out.float().cpu() - ref).abs().max() < 0.05 * ref.abs().max()


@pytest.mark.parametrize('task,name', [("se2d", "hybrid4_time"), ('se2d', 'hybrid4_time_full'),
                                       ('se2c', 'hybrid4c_time'), ('se2c', 'hybrid4c_time_full')])
def test_models_run_and_keep_the_layer_count(task, name):
    torch.manual_seed(0)
    t = tasks.get(task)
    sp = spec(task)
    base = 'hybrid4' if task == 'se2d' else 'hybrid4c'
    for size in ('xs', 's'):
        m = t['models'].get(name).build(size, sp)
        ref = t['models'].get(base).build(size, sp)
        assert len(m.blocks) == len(ref.blocks)
        kinds = [type(b_).__name__ for b_ in m.blocks]
        assert kinds[-1] == ('FullBlock' if name.endswith('_full') else 'AttnBlock')
        assert kinds[0] == 'TimeBlock'
    dev = 'cuda' if CUDA else 'cpu'
    if name.endswith('_full') and not CUDA:
        pytest.skip("the full GDN layer is hybrid4's Triton kernel: CUDA only")
    m = t['models'].get(name).build('xs', sp).to(dev)
    b, q = 2, 8
    if task == 'se2d':
        obs = torch.rand(b, 33, 9, 9)
        odo = torch.randint(0, 16, (b, 32))
        pose = torch.cat([torch.randint(-50, 50, (b, q, 2)), torch.randint(0, 4, (b, q, 1))], -1)
    else:
        obs = torch.rand(b, 33, 15, 15)
        odo = torch.randn(b, 32, 3)
        pose = torch.randn(b, q, 3)
    qt = torch.randint(0, 33, (b, q))
    obs, odo, pose, qt = (z.to(dev) for z in (obs, odo, pose, qt))
    with torch.no_grad():
        y = m(obs, odo, pose, query_time=qt)
    assert y.shape == (b, q, sp.view, sp.view) and torch.isfinite(y).all()


def test_full_layer_sees_hybrid4_order():
    """The full GDN layer of hybrid4_time_full gathers the layout into
    exactly hybrid4's token order."""
    from mapping.models.se2d.hybrid4 import Hybrid4
    qt = torch.randint(0, 33, (3, 8))
    order = Hybrid4.token_order(qt, 33, 8)
    # layout kinds/steps: obs 0, query 1, direction 2 per step
    t1 = 33
    steps = torch.arange(t1)
    lay_step = torch.cat([steps[:, None].expand(t1, 10).reshape(-1)[:-1][None].expand(3, -1),
                          qt[:, :, None].expand(-1, -1, 9).reshape(3, -1)], 1)
    kind_ctx = torch.cat([torch.zeros(t1, 9), torch.full((t1, 1), 2.)], 1).reshape(-1)[:-1]
    lay_kind = torch.cat([kind_ctx[None].expand(3, -1), torch.ones(3, 72)], 1)
    key = torch.gather(lay_step * 3 + lay_kind, 1, order)
    assert (key[:, 1:] >= key[:, :-1]).all()


@pytest.mark.skipif(not CUDA, reason="the full GDN layer is hybrid4's Triton kernel")
def test_se2c_full_layer_order_matches_hybrid4c():
    """hybrid4c_time_full's full GDN layer reproduces hybrid4c's sequence."""
    torch.manual_seed(0)
    t = tasks.get('se2c'); sp = spec('se2c')
    full = t['models'].get('hybrid4c_time_full').build('xs', sp)
    ref = t['models'].get('hybrid4c').build('xs', sp)
    ref.load_state_dict({k_: v_ for k_, v_ in full.state_dict().items()
                         if not k_.startswith('blocks')}, strict=False)
    b, q = 2, 8
    obs, odo, pose = torch.rand(b, 33, 15, 15), torch.randn(b, 32, 3), torch.randn(b, q, 3)
    qt = torch.randint(0, 33, (b, q))
    full, ref = full.cuda(), ref.cuda()
    obs, odo, pose, qt = obs.cuda(), odo.cuda(), pose.cuda(), qt.cuda()
    seen = {}
    full.blocks[-1].block.register_forward_pre_hook(lambda mod, a: seen.__setitem__('full', a[0]))
    ref.blocks[0].register_forward_pre_hook(lambda mod, a: seen.__setitem__('ref', a[0]))
    # zero the time block so the full layer sees the raw embeddings, as hybrid4c's first layer
    for blk in full.blocks[:-1]:
        for p_ in blk.parameters():
            torch.nn.init.zeros_(p_)
    with torch.no_grad():
        full(obs, odo, pose, query_time=qt)
        ref(obs, odo, pose, query_time=qt)
    assert torch.allclose(seen['full'], seen['ref'], atol=1e-6)
