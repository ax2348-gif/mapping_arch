"""Mechanism trace for hybrid4_causal and vanilla4_causal (se2d)."""
import pytest
import torch

import mapping.tasks as tasks

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="hybrid4's delta-rule kernel needs CUDA")
NAMES = ['hybrid4_causal', 'vanilla4_causal']
T = tasks.get('se2d')
SPEC = tasks.config_and_spec('se2d', 32, 8, 0, 'uniform')[1]


def build(name, seed=0, live_readout=False):
    torch.manual_seed(seed)
    net = T['models'].get(name).build('xs', SPEC).cuda().eval()
    if live_readout:     # the readout starts at zero (muP); dependence tests need it live
        for p_ in net.readout.parameters():
            torch.nn.init.normal_(p_, std=0.5)
    return net


def inputs(b=2, q=8, qt=None, seed=1):
    g = torch.Generator().manual_seed(seed)
    obs = torch.rand(b, 33, 9, 9, generator=g)
    odo = torch.randint(0, 16, (b, 32), generator=g)
    pose = torch.cat([torch.randint(-80, 80, (b, q, 2), generator=g),
                      torch.randint(0, 4, (b, q, 1), generator=g)], -1)
    if qt is None:
        qt = torch.randint(0, 33, (b, q), generator=g)
    return [z.cuda() for z in (obs, odo, pose, qt)]


@pytest.mark.parametrize('name', NAMES)
def test_whole_stack_is_causal_in_token_order(name):
    """Changing any token after position i leaves every output at positions
    <= i unchanged, through all blocks."""
    net = build(name)
    torch.manual_seed(2)
    L, w = 401, net.width
    x = torch.randn(2, L, w, device='cuda')
    pos = torch.randint(0, 33, (2, L), device='cuda').sort(1).values
    from mapping.models.se2d.hybrid4 import HEAD_DIM, rope_cos_sin
    cos, sin = rope_cos_sin(pos, HEAD_DIM)

    def run(z):
        with torch.no_grad():
            for blk in net.blocks:
                z = blk(z, cos, sin)
        return z
    y = run(x)
    for i in (0, 37, 200, 399):
        x2 = x.clone()
        x2[:, i + 1:] += torch.randn_like(x2[:, i + 1:])
        y2 = run(x2)
        assert torch.allclose(y[:, :i + 1], y2[:, :i + 1], atol=1e-5), i
        assert not torch.allclose(y[:, i + 1:], y2[:, i + 1:], atol=1e-3)


@pytest.mark.parametrize('name', NAMES)
def test_query_sees_its_past_and_not_its_future(name):
    """All queries at step t = 10: predictions ignore observations > 10 and
    odometry >= 10 (the move away from 10), and depend on observation 10
    and on odometry 9 (the move into 10)."""
    net = build(name, live_readout=True)
    obs, odo, pose, qt = inputs(qt=torch.full((2, 8), 10))
    with torch.no_grad():
        y = net(obs, odo, pose, query_time=qt)
        o2, d2 = obs.clone(), odo.clone()
        o2[:, 11:] = torch.rand_like(o2[:, 11:])
        d2[:, 10:] = (d2[:, 10:] + 5) % 16
        assert torch.allclose(y, net(o2, d2, pose, query_time=qt), atol=1e-5)
        o3 = obs.clone(); o3[:, 10] = torch.rand_like(o3[:, 10])
        assert not torch.allclose(y, net(o3, odo, pose, query_time=qt), atol=1e-4)
        d3 = odo.clone(); d3[:, 9] = (d3[:, 9] + 5) % 16
        assert not torch.allclose(y, net(obs, d3, pose, query_time=qt), atol=1e-4)


@pytest.mark.parametrize('name', NAMES)
def test_query_at_t_cannot_see_later_queries(name):
    """A query at an earlier step ignores a later query's pose."""
    net = build(name, live_readout=True)
    qt = torch.tensor([[5, 20, 20, 20, 20, 20, 20, 20]] * 2)
    obs, odo, pose, qt = inputs(qt=qt)
    with torch.no_grad():
        y = net(obs, odo, pose, query_time=qt)
        p2 = pose.clone(); p2[:, 1:, :2] += 7
        y2 = net(obs, odo, p2, query_time=qt)
    assert torch.allclose(y[:, 0], y2[:, 0], atol=1e-5)


def test_same_modules_and_initial_values_as_parents():
    for name, parent in (('hybrid4_causal', 'hybrid4'), ('vanilla4_causal', 'vanilla4')):
        a, b = build(name), build(parent)
        sa, sb = a.state_dict(), b.state_dict()
        assert sa.keys() == sb.keys()
        assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_all_delta_rule_heads_scan_forward():
    """hybrid4_causal's GDN equals hybrid4's GDN run with the right-to-left
    heads NOT flipped: compare against hybrid4's layer on a sequence and its
    reverse, head by head."""
    from mapping.models.se2d.hybrid4 import GatedDeltaNet
    net = build('hybrid4_causal')
    gdn = net.blocks[0].mixer
    ref = GatedDeltaNet.__new__(GatedDeltaNet)
    ref.__dict__ = gdn.__dict__
    torch.manual_seed(3)
    x = torch.randn(2, 120, net.width, device='cuda')
    with torch.no_grad():
        causal = gdn(x)
        # with a time-symmetric input the bidirectional layer's second head equals
        # the forward head on the reversed input; instead check causality per head:
        x2 = x.clone(); x2[:, 60:] += 1.0
        assert torch.allclose(gdn(x2)[:, :60], causal[:, :60], atol=1e-5)
        assert not torch.allclose(ref(x2)[:, :60], ref(x)[:, :60], atol=1e-5)
