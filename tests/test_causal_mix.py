"""hybrid4_mix: the causal/bidirectional fields and their masking structure."""
import pytest
import torch

import mapping.tasks as tasks
from mapping.models.se2d.hybrid4 import HEAD_DIM, rope_cos_sin
from mapping.models.se2d.hybrid4_mix import VARIANTS, Hybrid4Mix

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="hybrid4's delta-rule kernel needs CUDA")
T = tasks.get('se2d')
SPEC = tasks.config_and_spec('se2d', 32, 8, 0, 'uniform')[1]
SIZES = T['models'].get('hybrid4').SIZES


def mix(size, **kw):
    torch.manual_seed(0)
    return Hybrid4Mix(SPEC, **SIZES[size], **kw).cuda().eval()


def named(name, size):
    torch.manual_seed(0)
    return T['models'].get(name).build(size, SPEC).cuda().eval()


def batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    obs = torch.rand(2, 33, 9, 9, generator=g)
    odo = torch.randint(0, 16, (2, 32), generator=g)
    pose = torch.cat([torch.randint(-80, 80, (2, 8, 2), generator=g),
                      torch.randint(0, 4, (2, 8, 1), generator=g)], -1)
    qt = torch.randint(0, 33, (2, 8), generator=g)
    return [z.cuda() for z in (obs, odo, pose, qt)]


def live(net):
    for p_ in net.readout.parameters():
        torch.nn.init.normal_(p_, std=0.5)
    return net


@pytest.mark.parametrize('size', ['xs', 's'])
def test_corners_equal_hybrid4_and_hybrid4_causal(size):
    b = batch()
    for kw, ref in ((dict(delta='both', attn='bidir', top=0), 'hybrid4'),
                    (dict(delta='l2r', attn='causal', top=0), 'hybrid4_causal')):
        a, r = live(mix(size, **kw)), live(named(ref, size))
        assert a.state_dict().keys() == r.state_dict().keys()
        assert all(torch.equal(a.state_dict()[k], r.state_dict()[k]) for k in a.state_dict())
        with torch.no_grad():
            assert torch.equal(a(*b[:3], query_time=b[3]), r(*b[:3], query_time=b[3]))


def test_all_layers_bidirectional_is_hybrid4():
    a, r = live(mix('s', delta='l2r', attn='causal', top=4)), live(named('hybrid4', 's'))
    b = batch()
    with torch.no_grad():
        assert torch.equal(a(*b[:3], query_time=b[3]), r(*b[:3], query_time=b[3]))


EXPECT = {
    ('hybrid4_mix_gdnl2r', 'xs'): ['gdn:l2r', 'attn:bidir'],
    ('hybrid4_mix_attncausal', 'xs'): ['gdn:both', 'attn:causal'],
    ('hybrid4_mix_gdnr2l', 'xs'): ['gdn:r2l', 'attn:bidir'],
    ('hybrid4_causal_top1', 's'): ['gdn:l2r', 'attn:causal', 'gdn:l2r', 'attn:bidir'],
    ('hybrid4_causal_top2', 's'): ['gdn:l2r', 'attn:causal', 'gdn:both', 'attn:bidir'],
    ('hybrid4_causal_top3', 's'): ['gdn:l2r', 'attn:bidir', 'gdn:both', 'attn:bidir'],
    ('hybrid4_causal_top1gdn', 's'): ['gdn:l2r', 'attn:causal', 'gdn:l2r', 'gdn:both'],
}


@pytest.mark.parametrize('name,size', sorted(EXPECT))
def test_layer_kinds(name, size):
    assert named(name, size).layer_kinds() == EXPECT[(name, size)]


def stack(net, upto):
    """The first ``upto`` blocks as a function of a token sequence."""
    torch.manual_seed(2)
    pos = torch.randint(0, 33, (2, 401), device='cuda').sort(1).values
    cos, sin = rope_cos_sin(pos, HEAD_DIM)

    def run(z):
        with torch.no_grad():
            for blk in net.blocks[:upto]:
                z = blk(z, cos, sin)
        return z
    return run


def depends_on_later(run, w, i=200):
    x = torch.randn(2, 401, w, device='cuda')
    x2 = x.clone(); x2[:, i + 1:] += torch.randn_like(x2[:, i + 1:])
    return not torch.allclose(run(x)[:, :i + 1], run(x2)[:, :i + 1], atol=1e-5)


def depends_on_earlier(run, w, i=200):
    x = torch.randn(2, 401, w, device='cuda')
    x2 = x.clone(); x2[:, :i] += torch.randn_like(x2[:, :i])
    return not torch.allclose(run(x)[:, i:], run(x2)[:, i:], atol=1e-5)


@pytest.mark.parametrize('name,size', sorted(EXPECT))
def test_masking_structure(name, size):
    """Each block's causality is what its kind says, and the causal trunk
    (every block below the top k) is causal as a whole."""
    net = named(name, size)
    kinds = net.layer_kinds()
    w = net.width
    for j, kind in enumerate(kinds):
        one = stack(type('S', (), {'blocks': net.blocks[j:j + 1]})(), 1)
        later, earlier = depends_on_later(one, w), depends_on_earlier(one, w)
        if kind in ('gdn:l2r', 'attn:causal'):
            assert not later and earlier, (j, kind)
        elif kind == 'gdn:r2l':
            assert later and not earlier, (j, kind)
        else:
            assert later and earlier, (j, kind)
    trunk = len(kinds) - net.top
    if trunk and all(k in ('gdn:l2r', 'attn:causal') for k in kinds[:trunk]):
        assert not depends_on_later(stack(net, trunk), w)
    assert depends_on_later(stack(net, len(kinds)), w) == any(
        k not in ('gdn:l2r', 'attn:causal') for k in kinds)


def test_variants_are_registered_with_their_fields():
    for name, arch in VARIANTS.items():
        mod = T['models'].get(name)
        assert mod.ARCH == arch
        m = named(name, 'xs' if 'mix' in name else 's')
        assert (m.delta, m.attn, m.top, m.top_op) == tuple(arch[k] for k in ('delta', 'attn', 'top', 'top_op'))
