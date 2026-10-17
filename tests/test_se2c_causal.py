"""Mechanism trace for hybrid4c_causal and vanilla4c_causal (se2c, H1e)."""
import math

import pytest
import torch

import mapping.tasks as tasks
from mapping.models.se2d.hybrid4 import HEAD_DIM, rope_cos_sin

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="the delta-rule kernel needs CUDA")
NAMES = ['hybrid4c_causal', 'vanilla4c_causal']
T = tasks.get('se2c')
SPEC = tasks.config_and_spec('se2c', 32, 8, 0, 'uniform')[1]


def build(name, live=False):
    torch.manual_seed(0)
    net = T['models'].get(name).build('xs', SPEC).cuda().eval()
    if live:             # the readout starts at zero (muP); dependence tests need it live
        for p_ in net.readout.parameters():
            torch.nn.init.normal_(p_, std=0.5)
    return net


def inputs(qt, seed=1):
    g = torch.Generator().manual_seed(seed)
    b, q = qt.shape
    obs = torch.rand(b, 33, 15, 15, generator=g)
    odo = torch.cat([torch.rand(b, 32, 2, generator=g) * 9,
                     (torch.rand(b, 32, 1, generator=g) * 2 - 1) * math.pi], -1)
    pose = torch.cat([torch.rand(b, q, 2, generator=g) * 40 - 20,
                      (torch.rand(b, q, 1, generator=g) * 2 - 1) * math.pi], -1)
    return [z.cuda() for z in (obs, odo, pose, qt)]


@pytest.mark.parametrize('name', NAMES)
def test_whole_stack_is_causal_in_token_order(name):
    net = build(name)
    torch.manual_seed(2)
    L = 33 * 26 - 1 + 8 * 25
    pos = torch.randint(0, 33, (2, L), device='cuda').sort(1).values
    cos, sin = rope_cos_sin(pos, HEAD_DIM)
    x = torch.randn(2, L, net.width, device='cuda')

    def run(z):
        with torch.no_grad():
            for blk in net.blocks:
                z = blk(z, cos, sin)
        return z
    y = run(x)
    for i in (0, 300, 900, L - 2):
        x2 = x.clone()
        x2[:, i + 1:] += torch.randn_like(x2[:, i + 1:])
        assert torch.allclose(y[:, :i + 1], run(x2)[:, :i + 1], atol=1e-5), i


@pytest.mark.parametrize('name', NAMES)
def test_query_sees_its_past_and_not_its_future(name):
    """All queries at t = 10: predictions ignore observations > 10 and
    odometry >= 10, and use observation 10 and odometry 9."""
    net = build(name, live=True)
    obs, odo, pose, qt = inputs(torch.full((2, 8), 10))
    with torch.no_grad():
        y = net(obs, odo, pose, query_time=qt)
        o2, d2 = obs.clone(), odo.clone()
        o2[:, 11:] = torch.rand_like(o2[:, 11:])
        d2[:, 10:] = d2[:, 10:] + 0.7
        assert torch.allclose(y, net(o2, d2, pose, query_time=qt), atol=1e-5)
        o3 = obs.clone(); o3[:, 10] = torch.rand_like(o3[:, 10])
        assert not torch.allclose(y, net(o3, odo, pose, query_time=qt), atol=1e-4)
        d3 = odo.clone(); d3[:, 9] = d3[:, 9] + 0.7
        assert not torch.allclose(y, net(obs, d3, pose, query_time=qt), atol=1e-4)


@pytest.mark.parametrize('name', NAMES)
def test_query_ignores_later_queries(name):
    net = build(name, live=True)
    obs, odo, pose, qt = inputs(torch.tensor([[5] + [20] * 7] * 2))
    with torch.no_grad():
        y = net(obs, odo, pose, query_time=qt)
        p2 = pose.clone(); p2[:, 1:, :2] += 7
        assert torch.allclose(y[:, 0], net(obs, odo, p2, query_time=qt)[:, 0], atol=1e-5)


def test_same_modules_and_initial_values_as_parents():
    for name, parent in (('hybrid4c_causal', 'hybrid4c'), ('vanilla4c_causal', 'vanilla4c')):
        a, b = build(name).state_dict(), build(parent).state_dict()
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
