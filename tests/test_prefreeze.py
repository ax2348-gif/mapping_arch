"""Pre-freeze ablation models (docs/lift/prefreeze_ablations.md):
hybrid4_muconv, hybrid4.5b and its placements, hybrid4_convrace."""
import math

import pytest
import torch

import mapping.tasks as tasks
from tests.test_models import inputs

CUDA = torch.cuda.is_available()
T = tasks.get('se2')
SPEC = T['task'].TaskSpec()
R = T['models']


def build(name, size, seed=0):
    torch.manual_seed(seed)
    return R.get(name).build(size, SPEC)


@pytest.mark.parametrize('size', ['xs', 's', 'm'])
def test_nocopro_is_vanilla4(size):
    """The minimal-pair control is vanilla4: same names, same initial values
    at a seed, same outputs."""
    a, b = build('vanilla4', size), build('hybrid45b_nocopro', size)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    obs, odo, q = inputs('se2')
    qt = torch.randint(0, 33, q.shape[:2])
    with torch.no_grad():
        assert torch.equal(a(obs, odo, q, query_time=qt),
                           b(obs, odo, q, query_time=qt))


@pytest.mark.parametrize('name', ['hybrid45b', 'hybrid45b_mid',
                                  'hybrid45b_last', 'hybrid45b_two'])
def test_trunk_init_is_vanilla4(name):
    """Coprocessors are built after the trunk, so the trunk's initial values
    at a seed are vanilla4's: the pair differs only by the coprocessor."""
    sa, sb = build('vanilla4', 's').state_dict(), build(name, 's').state_dict()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    assert all(k.startswith('copros.') for k in sb if k not in sa)


def test_coprocessor_is_fixed_across_rungs():
    """The width-64 block has the same shapes and muP multipliers at every
    rung; only the projections and the trunk grow."""
    ref = None
    for size in ['xs', 's', 'm', 'l']:
        m = build('hybrid45b', size)
        c = m.copros[0]
        sig = [(n, tuple(p.shape)) for n, p in c.block.named_parameters()]
        mults = [(n, mod.mult) for n, mod in c.block.named_modules()
                 if hasattr(mod, 'mult')]
        if ref is None:
            ref = (sig, mults)
        assert (sig, mults) == ref
        assert c.down.mult == 256 / m.width and c.up.mult == 4.0
        assert c.block.mixer.heads == 2 and c.block.mixer.conv == 4


def test_coprocessor_block_matches_hybrid4_xs_layout():
    """The coprocessor block is hybrid4 xs's first block (muconv init):
    same parameter names and shapes, same multipliers and init stds."""
    h = build('hybrid4_muconv', 'xs').blocks[0]
    c = build('hybrid45b', 's').copros[0].block
    assert ([(n, p.shape) for n, p in h.named_parameters()]
            == [(n, p.shape) for n, p in c.named_parameters()])
    for (n1, m1), (n2, m2) in zip(h.named_modules(), c.named_modules()):
        assert n1 == n2 and getattr(m1, 'mult', None) == getattr(m2, 'mult', None)


@pytest.mark.parametrize('name,expected', [
    ('hybrid45b', ['c0', 'b0', 'b1', 'b2', 'b3']),
    ('hybrid45b_mid', ['b0', 'b1', 'c0', 'b2', 'b3']),
    ('hybrid45b_last', ['b0', 'b1', 'b2', 'b3', 'c0']),
    ('hybrid45b_two', ['c0', 'b0', 'b1', 'c1', 'b2', 'b3']),
    ('hybrid45b_nocopro', ['b0', 'b1', 'b2', 'b3'])])
def test_placement_order(name, expected):
    """At s (4 trunk blocks) the forward runs blocks in the stated order."""
    if not CUDA and name != 'hybrid45b_nocopro':
        pytest.skip('the delta-rule kernel needs CUDA')
    dev = 'cuda' if CUDA else 'cpu'
    m = build(name, 's').to(dev).eval()
    seen = []
    for i, blk in enumerate(m.blocks):
        blk.register_forward_hook(lambda *a, i=i: seen.append(f'b{i}'))
    for j, c in enumerate(getattr(m, 'copros', [])):
        c.register_forward_hook(lambda *a, j=j: seen.append(f'c{j}'))
    obs, odo, q = inputs('se2', device=dev)
    with torch.no_grad():
        m(obs, odo, q, query_time=torch.randint(0, 33, q.shape[:2], device=dev))
    assert seen == expected


def test_muconv_init_rule():
    for size in ['xs', 's', 'l']:
        w = build('hybrid4_muconv', size).blocks[0].mixer.short.weight
        assert w.shape[-1] == 4
        assert abs(w.std().item() - 0.5 / math.sqrt(4)) < 0.02, size


@pytest.mark.skipif(not CUDA, reason='the delta-rule kernel needs CUDA')
def test_muconv_forward_is_hybrid4():
    a = build('hybrid4', 'xs').cuda().eval()
    b = build('hybrid4_muconv', 'xs').cuda().eval()
    b.load_state_dict(a.state_dict())
    obs, odo, q = inputs('se2', device='cuda')
    qt = torch.randint(0, 33, q.shape[:2], device='cuda')
    with torch.no_grad():
        assert torch.equal(a(obs, odo, q, query_time=qt),
                           b(obs, odo, q, query_time=qt))


def test_convrace_receptive_fields():
    """Head of width k: its q, k, v at position t (in its scan order) depend
    on exactly the inputs t-k+1..t of that order."""
    g = build('hybrid4_convrace', 'xs').blocks[0].mixer
    assert g.head_widths == [1, 4, 10, 11, 1, 4, 10, 11]
    L, t = 40, 25
    x = torch.randn(1, L, 64, requires_grad=True)
    q, k, v, gh, bh = g.scan_inputs(x)
    half = g.heads // 2
    for h, kw in enumerate(g.head_widths):
        for out in (q, k, v):
            grad, = torch.autograd.grad(out[0, t, h].sum(), x, retain_graph=True)
            dep = (grad[0].abs().sum(-1) > 0).nonzero().flatten().tolist()
            if h < half:                       # left to right: inputs t-k+1..t
                want = list(range(t - kw + 1, t + 1))
            else:                              # right to left: scan position t
                want = sorted(L - 1 - s for s in range(t - kw + 1, t + 1))
            assert dep == want, (h, kw, dep)


def test_convrace_mask_holds_after_a_step():
    """Masked taps get no gradient, so an optimiser step keeps them zero
    (weight decay of a zero is zero)."""
    m = build('hybrid4_convrace', 'xs')
    g = m.blocks[0].mixer
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.1)
    x = torch.randn(2, 30, 64)
    sum(t.float().pow(2).sum() for t in g.scan_inputs(x)[:3]).backward()
    opt.step()
    assert torch.all(g.short.weight[g.conv_mask.expand_as(g.short.weight) == 0] == 0)


@pytest.mark.skipif(not CUDA, reason='the delta-rule kernel needs CUDA')
def test_convrace_layer_reduces_to_hybrid4():
    """With a single width 4 and hybrid4's head count, the race layer is
    hybrid4_muconv's layer: same weights, identical outputs."""
    from mapping.models.se2 import hybrid4_convrace as cr
    from mapping.models.se2 import hybrid4_muconv as mu
    torch.manual_seed(0)
    a = mu.GatedDeltaNet(64, 2, 4.0, 0.0625, 0.03125).cuda()
    b = cr.GatedDeltaNet(64, 2, 4.0, 0.0625, 0.03125, widths=(4,)).cuda()
    b.load_state_dict(a.state_dict())
    x = torch.randn(3, 401, 64, device='cuda')
    with torch.no_grad():
        assert torch.equal(a(x), b(x))
