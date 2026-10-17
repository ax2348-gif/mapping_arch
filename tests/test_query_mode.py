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
from mapping.models.query_mode import QueryLayout, attend

CUDA = torch.cuda.is_available()
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.environ.get('QUERY_MODE_REF', '8b2e419')     # master before query_mode
SDPA_REF = os.environ.get('QUERY_MODE_SDPA_REF', 'c55e2ee')   # last commit with SDPA masks
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
    causal intersects with token order. The mask_mod evaluated on every
    (i, k), and the block mask built from it, which must agree."""
    order = torch.tensor([[0, 1, 5, 6, 2, 3, 7, 8, 4]])    # 5 history, 2 queries of 2
    lay = QueryLayout(order, 5, 2, 2)
    assert lay.qid.tolist() == [[0, 0, 1, 1, 0, 0, 2, 2, 0]]
    assert lay.hist_idx.tolist() == [[0, 1, 4, 5, 8]]
    i, k = torch.meshgrid(torch.arange(9), torch.arange(9), indexing='ij')
    z = torch.zeros_like(i)
    g = lay.qid[0]
    want = (g[k] == 0) | (g[i] == g[k])
    assert torch.equal(lay.mask_mod()(z, z, i, k), want)
    assert torch.equal(lay.mask_mod(True)(z, z, i, k), want & (k <= i))
    # through flex: one-hot values read back which keys each row attended
    v = torch.eye(9)[None, None]
    for causal in (False, True):
        q = torch.zeros(1, 1, 9, 9)                        # uniform scores
        o = attend(q, q, v, lay.block_mask(causal))[0, 0]
        m = want & (k <= i) if causal else want
        assert torch.allclose(o, m.float() / m.sum(1, keepdim=True)), causal


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


# ---- the code before: master (mixed) and the SDPA-mask path (every mode) ---

REF_SCRIPT = r"""
import sys, torch
sys.path.insert(0, sys.argv[1])
import mapping.tasks as tasks
dev, modes = sys.argv[2], sys.argv[4].split(',')
torch.backends.cuda.matmul.allow_tf32 = False
out = {}
for task in ('se2d', 'se2c'):
    reg = tasks.get(task)['models']
    sp = (tasks.config_and_spec('se2d', 32, 8, 0, 'uniform')[1] if task == 'se2d'
          else tasks.get('se2c')['task'].TaskSpec())
    for name in sorted(reg.REGISTRY):
        module = reg.get(name)
        if dev == 'cpu' and 'hybrid' in name:
            continue
        size = next(iter(module.SIZES))
        for mode in modes:
            torch.manual_seed(0)
            net = module.build(size, sp)
            if mode != 'mixed':
                if mode not in type(net).__dict__.get('QUERY_MODES', ()):
                    continue
                torch.manual_seed(0)
                net = type(net)(sp, **module.SIZES[size], query_mode=mode)
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
            key = f'{task}/{name}/{mode}'
            out[key] = y.float().cpu()
            # attention under a mask (flex now): causal attention or inert
            out[key + '#masked'] = mode != 'mixed' or any(
                type(m).__name__.endswith('Attention') and getattr(m, 'causal', False)
                for m in net.modules())
torch.save(out, sys.argv[3])
"""


def outputs(root, device, tmp_path, tag, modes):
    script = tmp_path / 'ref_outputs.py'
    script.write_text(REF_SCRIPT)
    dst = tmp_path / f'{tag}.pt'
    env = dict(os.environ, PYTHONPATH=str(root), CUBLAS_WORKSPACE_CONFIG=':4096:8')
    subprocess.run([sys.executable, str(script), str(root), device, str(dst), ','.join(modes)],
                   cwd=str(root), env=env, check=True, stderr=subprocess.DEVNULL)
    return torch.load(dst)


def checkout(ref, tmp_path):
    try:
        blob = subprocess.run(['git', '-C', HERE, 'archive', ref, 'mapping', 'conf'],
                              capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        pytest.skip(f'reference commit {ref} not available')
    root = tmp_path / f'ref_{ref}'
    root.mkdir()
    tarfile.open(fileobj=io.BytesIO(blob)).extractall(root, filter='data')
    os.symlink(os.path.join(HERE, 'data'), root / 'data')
    return root


def compare(ref, new):
    """``{model/mode: (masked, max |diff|, max |diff| / max |ref|)}``."""
    keys = sorted(k for k in ref if not k.endswith('#masked'))
    assert keys == sorted(k for k in new if not k.endswith('#masked'))
    return {k: (new[k + '#masked'], (ref[k] - new[k]).abs().max().item(),
                ((ref[k] - new[k]).abs().max() / ref[k].abs().max()).item()) for k in keys}


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not CUDA, reason='needs CUDA'))]


@pytest.mark.parametrize('device', DEVICES)
def test_mixed_matches_master(device, tmp_path):
    """Every se2d and se2c model in mixed mode against master before the
    option, in fresh processes: bit-identical where attention is unmasked
    (plain SDPA, unchanged); where it is causal (SDPA is_causal then, flex
    now), within fp32 tolerance."""
    ref = outputs(checkout(REF, tmp_path), device, tmp_path, 'ref', ['mixed'])
    new = outputs(HERE, device, tmp_path, 'new', ['mixed'])
    res = compare(ref, new)
    assert len(res) >= (3 if device == 'cpu' else 20)
    unmasked = {k: v for k, v in res.items() if not v[0] and v[1] != 0}
    assert not unmasked, f'unmasked models changed: {unmasked}'
    masked = {k: v for k, v in res.items() if v[0]}
    assert masked or device == 'cpu'
    assert all(v[2] < 1e-4 for v in masked.values()), masked


@pytest.mark.parametrize('device', DEVICES)
def test_flex_matches_the_sdpa_mask_path(device, tmp_path):
    """Every model and mode against the last commit with SDPA masks
    (boolean inert mask, is_causal): unmasked bit-identical, masked within
    fp32 tolerance."""
    modes = ['mixed', 'inert']
    ref = outputs(checkout(SDPA_REF, tmp_path), device, tmp_path, 'ref', modes)
    new = outputs(HERE, device, tmp_path, 'new', modes)
    res = compare(ref, new)
    assert sum(k.endswith('/inert') for k in res) >= (2 if device == 'cpu' else 6)
    assert not {k: v for k, v in res.items() if not v[0] and v[1] != 0}
    bad = {k: v for k, v in res.items() if v[0] and v[2] >= 1e-4}
    assert not bad, bad
    worst = max((v for v in res.values() if v[0]), key=lambda v: v[2])
    print(f'max deviation over masked models: |d| {worst[1]:.2e}, relative {worst[2]:.2e}')


def test_no_recompile_when_queries_or_steps_change():
    """After one warm-up forward and backward at Q = 8, T = 32: other query
    counts and episode lengths compile nothing (the mask reads metadata
    tensors; flex_attention and create_block_mask are compiled dynamic)."""
    if not CUDA:
        pytest.skip('compiled flex runs on CUDA')
    from torch._dynamo.utils import counters
    for task, name, mode in [('se2d', 'hybrid4', 'inert'), ('se2d', 'vanilla4_causal', 'inert'),
                             ('se2d', 'vanilla4_causal', 'mixed')]:
        net = build(task, name, mode, 'cuda', torch.float32, reference_scan=False)

        def go(q, T=33):
            obs, odo, pose, qt = inputs(task, q, device='cuda')
            obs, odo, qt = obs[:, :T], odo[:, :T - 1], qt.clamp(max=T - 1)
            with torch.no_grad():
                net(obs, odo, pose, query_time=qt)
            net(obs, odo, pose, query_time=qt).pow(2).mean().backward()
        go(8)
        before = counters['stats']['unique_graphs']
        for q, T in ((1, 33), (3, 33), (64, 33), (300, 33), (8, 17), (64, 25)):
            go(q, T)
        assert counters['stats']['unique_graphs'] == before, (name, mode)


@pytest.mark.parametrize('name', ['hybrid4', 'vanilla4'])
def test_token_order_is_exact_at_any_length(name):
    """token_order is a stable sort by (step, kind, layout index) at any
    sequence length: the same permutation as before for training lengths,
    and queries stay between their step's observation and odometry tokens
    beyond 1e5 tokens (where the old float tie-break broke)."""
    import numpy as np
    cls = tasks.get('se2d')['models'].get(name).build('xs', spec('se2d')).__class__
    for q in (8, 64, 12000):
        qt = torch.randint(0, 33, (2, q), generator=torch.Generator().manual_seed(q))
        order = cls.token_order(qt, 33, q)
        steps = np.arange(33)
        ctx = np.concatenate([np.stack([steps * 3] * 9, 1), (steps * 3 + 2)[:, None]], 1).reshape(-1)[:-1]
        for r in range(2):
            key = np.concatenate([ctx, np.repeat(qt[r].numpy() * 3 + 1, 9)])
            want = np.lexsort((np.arange(len(key)), key))
            assert np.array_equal(order[r].numpy(), want), (name, q)
