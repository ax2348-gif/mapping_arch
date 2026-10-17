"""``model.query_mode`` (mapping/models/query_mode.py).

inert, asserted at every layer (block output), for every model that
implements it, bidirectional and causal:
- history invariance: the history tokens' activations are the same with 0,
  8 or 64 queries present (0 = the queries deleted), and equal to a mixed
  model's with no queries;
- query independence: perturbing query j's pose or time changes nothing in
  any other query's tokens at any layer (only the same query's tokens may
  interact); a query's prediction is the same alone, among 8 or among 64,
  at any position among them, so all-at-once equals chunked.
The checks run on CPU in fp32 (the models cast some inputs to fp32, so
not float64), with the delta-rule scan replaced by its sequential
reference (``delta_scan``, the loop the fused kernel is checked against),
to 1e-5; and on CUDA on the fused kernel in fp32, to its tolerance. Mixed mode controls: the same perturbations DO leak
there (the tests are not vacuous).

mixed: bit-identical outputs to the code before the option (``REF``, the
master commit it branched from), for every se2d and se2c model.
"""
import os
import subprocess
import sys
import tarfile
import io

import pytest
import torch

import mapping.tasks as tasks
from mapping.models.query_mode import QueryLayout

CUDA = torch.cuda.is_available()
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.environ.get('QUERY_MODE_REF', '8b2e419')     # master before query_mode
SE2D = tasks.get('se2d')
SE2C = tasks.get('se2c')
INERT = [('se2d', 'hybrid4'), ('se2d', 'vanilla4'), ('se2d', 'hybrid4_causal'),
         ('se2d', 'vanilla4_causal'), ('se2c', 'hybrid4c'), ('se2c', 'vanilla4c')]


def spec(task):
    if task == 'se2d':
        return tasks.config_and_spec('se2d', 32, 8, 0, 'uniform')[1]
    return SE2C['task'].TaskSpec()


def build(task, name, mode, device='cpu', dtype=torch.float32, reference_scan=True):
    module = tasks.get(task)['models'].get(name)
    torch.manual_seed(0)
    cls = module.build(next(iter(module.SIZES)), spec(task)).__class__
    torch.manual_seed(0)
    kw = dict(module.SIZES[next(iter(module.SIZES))])
    net = cls(spec(task), **kw, query_mode=mode)
    torch.manual_seed(1)
    torch.nn.init.normal_(net.readout.weight, std=0.5)        # zero at init (muP)
    net = net.to(device, dtype).eval()
    if reference_scan:
        from mapping.models.se2d.hybrid4 import GatedDeltaNet
        for m in net.modules():
            if isinstance(m, GatedDeltaNet):
                m.scan = scan64
    return net


def scan64(q, k, v, g, beta):
    """``mapping.models.se2d.hybrid4.delta_scan`` (the sequential reference
    the fused kernel is checked against), in the inputs' dtype."""
    b, l, h, dk = k.shape
    S = q.new_zeros(b, h, dk, v.shape[-1])
    out = []
    for t in range(l):
        S = S * g[:, t].exp().to(S.dtype)[..., None, None]
        kS = torch.einsum('bhk,bhkv->bhv', k[:, t], S)
        S = S + torch.einsum('bhk,bhv->bhkv', k[:, t] * beta[:, t, :, None].to(S.dtype),
                             v[:, t] - kS)
        out.append(torch.einsum('bhk,bhkv->bhv', q[:, t], S))
    return torch.stack(out, 1)


def inputs(task, q, b=2, seed=0, device='cpu', dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    if task == 'se2d':
        obs = torch.rand(b, 33, 9, 9, generator=g)
        odo = torch.randint(0, 16, (b, 32), generator=g)
        pose = torch.cat([torch.randint(-80, 80, (b, q, 2), generator=g),
                          torch.randint(0, 4, (b, q, 1), generator=g)], -1)
    else:
        obs = torch.rand(b, 33, 15, 15, generator=g)
        odo = torch.cat([torch.randn(b, 32, 2, generator=g) * 6,
                         (torch.rand(b, 32, 1, generator=g) * 2 - 1) * 3.14], -1)
        pose = torch.cat([torch.randn(b, q, 2, generator=g) * 20,
                          (torch.rand(b, q, 1, generator=g) * 2 - 1) * 3.14], -1)
    qt = torch.randint(0, 33, (b, q), generator=g)
    obs = obs.to(device, dtype)
    if task == 'se2c':
        odo, pose = odo.to(device, dtype), pose.to(device, dtype)
    return obs, odo.to(device), pose.to(device), qt.to(device)


def per_query(net):
    return 9 if net.spec.view == 9 else net.P


def order_of(net, qt):
    q = qt.shape[1]
    if hasattr(net, 'token_order'):
        return net.token_order(qt, 33, q)
    from mapping.models.se2c.hybrid4c import token_order
    return token_order(qt, 33, q, net.P)


def run(net, obs, odo, pose, qt):
    """``(pred, layers)``: each block's output, back in layout order
    ``(b, L, w)`` [history tokens, then query 0's tokens, query 1's ...]."""
    outs = []
    hooks = [blk.register_forward_hook(lambda m, i, o: outs.append(o)) for blk in net.blocks]
    try:
        with torch.no_grad():
            pred = net(obs, odo, pose, query_time=qt)
    finally:
        for h in hooks:
            h.remove()
    inv = torch.argsort(order_of(net, qt), 1)
    layers = [torch.gather(o, 1, inv[..., None].expand(-1, -1, o.shape[-1])) for o in outs]
    return pred, layers


def n_hist(net):
    return 33 * (per_query(net) + 1) - 1


def query_tokens(net, layer, j):
    P = per_query(net)
    s = n_hist(net) + j * P
    return layer[:, s:s + P]


def close(a, b, exact):
    if exact:
        return torch.allclose(a, b, atol=1e-5, rtol=1e-5)
    # the fused kernel's own error against the reference scan is ~1e-3 of
    # the activations' scale (its dots run in TF32); a mixed-mode leak is
    # far above this (the controls below must still fail it)
    return (a.float() - b.float()).abs().max() <= 5e-3 * max(a.float().abs().max(), 1e-6)


def settings():
    out = [pytest.param('cpu', torch.float32, True, id='cpu-reference')]
    out.append(pytest.param('cuda', torch.float32, False, id='cuda-f32-kernel',
                            marks=pytest.mark.skipif(not CUDA, reason='needs CUDA')))
    return out


def skip_cpu_kernel(name, device, reference_scan):
    if device == 'cpu' and not reference_scan and 'hybrid' in name:
        pytest.skip('the delta-rule kernel needs CUDA')


@pytest.fixture(autouse=True)
def no_tf32():
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


@pytest.mark.parametrize('device,dtype,exact', settings())
@pytest.mark.parametrize('task,name', INERT)
def test_history_is_invariant_to_queries(task, name, device, dtype, exact):
    net = build(task, name, 'inert', device, dtype, reference_scan=exact)
    mixed = build(task, name, 'mixed', device, dtype, reference_scan=exact)
    H = n_hist(net)
    base = None
    for q, seed in ((0, 0), (8, 1), (64, 2)):
        obs, odo, pose, qt = inputs(task, q, device=device, dtype=dtype)
        _, _, pose, qt = inputs(task, q, seed=seed + 10, device=device, dtype=dtype)
        _, layers = run(net, obs, odo, pose, qt)
        hist = [x[:, :H] for x in layers]
        if base is None:
            base = hist
            _, ref = run(mixed, obs, odo, pose, qt)          # mixed, queries deleted
            for i, (a, r) in enumerate(zip(hist, ref)):
                assert close(a, r[:, :H], exact), (name, 'inert != mixed without queries', i)
            continue
        for i, (a, b) in enumerate(zip(hist, base)):
            assert close(a, b, exact), (name, q, 'layer', i, (a - b).abs().max().item())
    # control: in mixed mode the history does depend on the queries
    obs, odo, pose, qt = inputs(task, 8, device=device, dtype=dtype)
    _, with_q = run(mixed, obs, odo, pose, qt)
    assert not close(with_q[-1][:, :H], ref[-1][:, :H], exact)


@pytest.mark.parametrize('device,dtype,exact', settings())
@pytest.mark.parametrize('task,name', INERT)
def test_queries_are_independent_at_every_layer(task, name, device, dtype, exact):
    """Perturbing query j's pose (and, separately, its time) changes no
    other query's tokens at any layer; query j's own tokens do change."""
    for mode in ('inert', 'mixed'):
        net = build(task, name, mode, device, dtype, reference_scan=exact)
        obs, odo, pose, qt = inputs(task, 8, device=device, dtype=dtype)
        _, base = run(net, obs, odo, pose, qt)
        for j, what in ((3, 'pose'), (5, 'time')):
            p2, t2 = pose.clone(), qt.clone()
            if what == 'pose':
                p2[:, j] = p2[:, j] + (7 if task == 'se2d' else 3.5)
                if task == 'se2d':
                    p2[:, j, 2] = (pose[:, j, 2] + 1) % 4
            else:
                t2[:, j] = (qt[:, j] + 11) % 33
            _, pert = run(net, obs, odo, p2, t2)
            leaked = False
            for li, (a, b) in enumerate(zip(base, pert)):
                for i in range(8):
                    same = close(query_tokens(net, a, i), query_tokens(net, b, i), exact)
                    if i == j:          # the perturbation reaches query j itself
                        assert not torch.equal(query_tokens(net, a, i), query_tokens(net, b, i)), (
                            name, mode, what, 'query j unchanged', li)
                    elif mode == 'inert':
                        assert same, (name, what, 'query', i, 'saw query', j, 'layer', li)
                    else:
                        leaked |= not same
                if mode == 'inert':      # history untouched too
                    assert close(a[:, :n_hist(net)], b[:, :n_hist(net)], exact), (name, what, li)
            if mode == 'mixed':
                assert leaked, (name, what, 'mixed control: no leak between queries')


@pytest.mark.parametrize('device,dtype,exact', settings())
@pytest.mark.parametrize('task,name', INERT)
def test_prediction_alone_equals_among_others(task, name, device, dtype, exact):
    """64 queries at once == each in chunks of 8 == each alone, and a
    permutation of the queries permutes the predictions."""
    net = build(task, name, 'inert', device, dtype, reference_scan=exact)
    obs, odo, pose, qt = inputs(task, 64, device=device, dtype=dtype)
    with torch.no_grad():
        full = net(obs, odo, pose, query_time=qt)
        chunks = torch.cat([net(obs, odo, pose[:, i:i + 8], query_time=qt[:, i:i + 8])
                            for i in range(0, 64, 8)], 1)
        assert close(full, chunks, exact), (name, (full - chunks).abs().max().item())
        for i in (0, 17, 63):
            alone = net(obs, odo, pose[:, i:i + 1], query_time=qt[:, i:i + 1])
            assert close(full[:, i:i + 1], alone, exact), (name, i)
        perm = torch.randperm(64, generator=torch.Generator().manual_seed(3)).to(pose.device)
        shuffled = net(obs, odo, pose[:, perm], query_time=qt[:, perm])
        assert close(full[:, perm], shuffled, exact), name
        # a query among 7 others at every position among them
        for k in range(8):
            idx = torch.tensor([*range(20, 20 + k), 5, *range(30, 37 - k)], device=pose.device)
            out = net(obs, odo, pose[:, idx], query_time=qt[:, idx])
            assert close(out[:, k], full[:, 5], exact), (name, k)


def test_attention_mask_structure():
    """history -> history; query -> history + itself; never another query;
    causal intersects with token order."""
    order = torch.tensor([[0, 1, 5, 6, 2, 3, 7, 8, 4]])    # 5 history, 2 queries of 2
    lay = QueryLayout(order, 5, 2, 2)
    assert lay.group.tolist() == [[0, 0, 1, 1, 0, 0, 2, 2, 0]]
    assert lay.hist_idx.tolist() == [[0, 1, 4, 5, 8]]
    m = lay.attn_mask()[0, 0]
    g = lay.group[0]
    for i in range(9):
        for k in range(9):
            want = bool(g[k] == 0) or bool(g[i] == g[k])
            assert bool(m[i, k]) == want
    c = lay.attn_mask(causal=True)[0, 0]
    assert torch.equal(c, m & torch.ones(9, 9, dtype=torch.bool).tril())


def test_a_subclass_does_not_inherit_inert():
    from mapping.models.query_mode import check_query_mode
    from mapping.models.se2d.hybrid4 import Hybrid4

    class Sub(Hybrid4):             # inherits Hybrid4's __init__, not its declaration
        pass
    with pytest.raises(ValueError, match='does not implement'):
        Sub(spec('se2d'), 64, query_mode='inert')
    with pytest.raises(ValueError, match='must be one of'):
        check_query_mode(Hybrid4, 'mixedx')


@pytest.mark.parametrize('model', ['se2d/hybrid4_time', 'se2d/hybrid4_causal_top1',
                                   'se2d/hybrid45b', 'se2d/hybrid4_noconv',
                                   'se2c/hybrid4c_time', 'z2/hybrid4'])
def test_inert_is_refused_where_not_implemented(model):
    import run as entry
    entry.build(entry.resolved(entry.load([f'model={model}'])))            # mixed is fine
    with pytest.raises(Exception, match='does not implement'):
        entry.build(entry.resolved(entry.load([f'model={model}', 'model.query_mode=inert'])))


def test_config_field_reaches_model_record_and_args():
    import run as entry

    def build_cfg(*o):
        return entry.build(entry.resolved(entry.load(['model=se2d/hybrid4', *o])))
    mixed, inert = build_cfg(), build_cfg('model.query_mode=inert')
    with pytest.raises(Exception, match='must be one of'):
        build_cfg('model.query_mode=x')
    assert 'query_mode' not in mixed.model.record()['kwargs']          # as before the field
    assert not hasattr(mixed.legacy_args(), 'query_mode')
    assert inert.model.record()['kwargs']['query_mode'] == 'inert'
    assert inert.legacy_args().query_mode == 'inert'
    assert inert.model.build(spec('se2d')).query_mode == 'inert'


# ---- mixed: bit-identical to the code before the option -------------------

REF_SCRIPT = r'''
import sys, torch
sys.path.insert(0, sys.argv[1])
import mapping.tasks as tasks
dev = sys.argv[2]
out = {}
for task in ('se2d', 'se2c'):
    reg = tasks.get(task)['models']
    sp = (tasks.config_and_spec('se2d', 32, 8, 0, 'uniform')[1] if task == 'se2d'
          else tasks.get('se2c')['task'].TaskSpec())
    for name in sorted(reg.REGISTRY):
        module = reg.get(name)
        if dev == 'cpu' and 'hybrid' in name:
            continue
        torch.manual_seed(0)
        net = module.build(next(iter(module.SIZES)), sp)
        torch.manual_seed(1)
        torch.nn.init.normal_(net.readout.weight, std=0.5)
        net = net.to(dev).eval()
        g = torch.Generator().manual_seed(7)
        b, q = 2, 8
        if task == 'se2d':
            obs = torch.rand(b, 33, 9, 9, generator=g)
            odo = torch.randint(0, 16, (b, 32), generator=g)
            pose = torch.cat([torch.randint(-80, 80, (b, q, 2), generator=g),
                              torch.randint(0, 4, (b, q, 1), generator=g)], -1)
        else:
            obs = torch.rand(b, 33, 15, 15, generator=g)
            odo = torch.cat([torch.randn(b, 32, 2, generator=g) * 6,
                             (torch.rand(b, 32, 1, generator=g) * 2 - 1) * 3.14], -1)
            pose = torch.cat([torch.randn(b, q, 2, generator=g) * 20,
                              (torch.rand(b, q, 1, generator=g) * 2 - 1) * 3.14], -1)
        qt = torch.randint(0, 33, (b, q), generator=g)
        with torch.no_grad():
            try:
                y = net(obs.to(dev), odo.to(dev), pose.to(dev), query_time=qt.to(dev))
            except TypeError:
                y = net(obs.to(dev), odo.to(dev), pose.to(dev))
        out[f'{task}/{name}'] = y.float().cpu()
torch.save(out, sys.argv[3])
'''


def outputs(root, device, tmp_path, tag):
    script = tmp_path / 'ref_outputs.py'
    script.write_text(REF_SCRIPT)
    dst = tmp_path / f'{tag}.pt'
    env = dict(os.environ, PYTHONPATH=str(root), CUBLAS_WORKSPACE_CONFIG=':4096:8')
    subprocess.run([sys.executable, str(script), str(root), device, str(dst)],
                   cwd=str(root), env=env, check=True)
    return torch.load(dst)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not CUDA, reason='needs CUDA'))])
def test_mixed_is_bit_identical_to_master(device, tmp_path):
    """Every se2d and se2c model, built and run in a fresh process on the
    reference commit's code and on this checkout's (default query_mode,
    i.e. mixed): identical outputs, bit for bit."""
    try:
        blob = subprocess.run(['git', '-C', HERE, 'archive', REF, 'mapping', 'conf'],
                              capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        pytest.skip(f'reference commit {REF} not available')
    ref_root = tmp_path / 'ref'
    ref_root.mkdir()
    tarfile.open(fileobj=io.BytesIO(blob)).extractall(ref_root, filter='data')
    os.symlink(os.path.join(HERE, 'data'), ref_root / 'data')
    ref = outputs(ref_root, device, tmp_path, 'ref')
    new = outputs(HERE, device, tmp_path, 'new')
    assert ref.keys() == new.keys() and len(ref) >= (3 if device == 'cpu' else 20)
    diff = {k: (ref[k] - new[k]).abs().max().item() for k in ref if not torch.equal(ref[k], new[k])}
    assert not diff, f'mixed outputs differ from {REF}: {diff}'
