"""``model.scan_precision`` (mapping/models/scan_precision.py).

- ``ieee`` is the exact recurrence to float32: on a hybrid4c_causal inert
  model (CUDA), predictions within 1e-6 relative and parameter gradients
  within 1e-5 relative of the same model with the scan replaced by the float64
  sequential recurrence; at the scan alone, on inputs where TF32 fails
  (beta near 2, decay near 1, near-parallel keys), ~1e-6 forward and
  gradients, while the tf32 kernel is off by orders of magnitude more (the
  control: the test is not vacuous). Also under bf16 autocast (fp32 into
  the kernel, output cast back).
- ``tf32`` (the default) is the code before the option: every se2d and
  se2c model, mixed and inert, bit-identical to ``REF`` in fresh
  processes; and the config field leaves the arch and args record as
  before.
- The per-call ``TRITON_F32_DEFAULT`` never leaks: in one process, tf32
  before and after ieee calls (forward and backward) gives bit-identical
  results, ieee gives the same bits whichever ran first, and the variable
  is absent afterwards.
- ``ieee`` is refused by models that do not declare it.
"""
import json
import os
import subprocess
import sys

import pytest
import torch

import mapping.tasks as tasks
from tests.test_query_mode import (CUDA, DEVICES, HERE, checkout, compare, inputs,
                                   outputs, spec)

REF = os.environ.get('SCAN_PRECISION_REF', '99849b8')   # exp/se2c-causal-inert before the option
needs_cuda = pytest.mark.skipif(not CUDA, reason='the delta-rule kernel needs CUDA')


@pytest.fixture(autouse=True)
def no_tf32():
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def exact64(q, k, v, g, beta):
    """The gated delta rule as the sequential recurrence in float64, the
    output in v's dtype."""
    dt = v.dtype
    q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))
    b, l, h, dk = k.shape
    S = q.new_zeros(b, h, dk, v.shape[-1])
    out = []
    for t in range(l):
        S = S * g[:, t].exp()[..., None, None]
        kS = torch.einsum('bhk,bhkv->bhv', k[:, t], S)
        S = S + torch.einsum('bhk,bhv->bhkv', k[:, t] * beta[:, t, :, None], v[:, t] - kS)
        out.append(torch.einsum('bhk,bhkv->bhv', q[:, t], S))
    return torch.stack(out, 1).to(dt)


def hard_inputs(b=2, l=300, h=2, d=32, seed=0, dtype=torch.float32):
    """Where TF32 fails: beta in (1.99, 2), log decay in (-1e-3, 0), keys
    drawn around 4 directions (near-parallel within a chunk)."""
    g_ = torch.Generator(device='cuda').manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g_, device='cuda')
    centres = r(h, 4, d)
    pick = torch.randint(0, 4, (b, l, h), generator=g_, device='cuda')
    k = centres[torch.arange(h, device='cuda')[None, None], pick] + 0.3 * r(b, l, h, d)
    k = torch.nn.functional.normalize(k, dim=-1)
    q = torch.nn.functional.normalize(r(b, l, h, d), dim=-1) * d ** -0.5
    v = r(b, l, h, d)
    g = -1e-3 * torch.rand(b, l, h, generator=g_, device='cuda')
    beta = 2 - 1e-2 * torch.rand(b, l, h, generator=g_, device='cuda')
    return [x.to(dtype) for x in (q, k, v)] + [g, beta]


def grads(f, ins, R):
    xs = [x.detach().clone().requires_grad_(True) for x in ins]
    o = f(*xs)
    (o.double() * R).sum().backward()
    return o.detach().double(), [x.grad.double() for x in xs]


def tf32_scan(q, k, v, g, beta):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    return chunk_gated_delta_rule(q.contiguous(), k.contiguous(), v.contiguous(),
                                  g.contiguous(), beta.contiguous(), scale=1.0)[0]


@needs_cuda
def test_ieee_scan_is_exact_where_tf32_is_not():
    from mapping.models.scan_precision import ieee_scan
    ins = hard_inputs()
    R = torch.randn(ins[2].shape, device='cuda', dtype=torch.float64,
                    generator=torch.Generator(device='cuda').manual_seed(1))
    o_ref, g_ref = grads(exact64, [x.double() for x in ins], R)
    o_ieee, g_ieee = grads(ieee_scan, ins, R)
    o_tf32, g_tf32 = grads(tf32_scan, ins, R)
    scale = o_ref.abs().max()
    err_ieee = (o_ieee - o_ref).abs().max() / scale
    err_tf32 = (o_tf32 - o_ref).abs().max() / scale
    rel = lambda a, b: max(((x - y).norm() / y.norm()).item() for x, y in zip(a, b))
    print(f'forward rel err: ieee {err_ieee:.2e}, tf32 {err_tf32:.2e}; '
          f'gradients rel: ieee {rel(g_ieee, g_ref):.2e}, tf32 {rel(g_tf32, g_ref):.2e}')
    assert err_ieee < 1e-5 and rel(g_ieee, g_ref) < 1e-5
    assert err_tf32 > 100 * err_ieee and rel(g_tf32, g_ref) > 100 * rel(g_ieee, g_ref)
    assert 'TRITON_F32_DEFAULT' not in os.environ


@needs_cuda
def test_ieee_scan_under_autocast_takes_fp32_and_returns_the_input_dtype():
    from mapping.models.scan_precision import ieee_scan
    ins = hard_inputs(dtype=torch.bfloat16)
    xs = [x.detach().clone().requires_grad_(True) for x in ins]
    with torch.autocast('cuda', torch.bfloat16):
        o = ieee_scan(*xs)
    o.float().sum().backward()
    assert o.dtype == torch.bfloat16
    assert [x.grad.dtype for x in xs] == [x.dtype for x in ins]
    ref = exact64(*[x.double() for x in ins]).float()      # the same bf16 values, exactly
    err = ((o.float() - ref).abs().max() / ref.abs().max()).item()
    assert err < 8e-3, err              # bf16 rounding of the output only (2^-8)


def build(name, precision, task='se2c', mode='inert'):
    module = tasks.get(task)['models'].get(name)
    size = next(iter(module.SIZES))
    cls = module.build(size, spec(task)).__class__
    torch.manual_seed(0)
    kw = {} if precision == 'tf32' else {'scan_precision': precision}
    net = cls(spec(task), **module.SIZES[size], query_mode=mode, **kw)
    torch.manual_seed(1)
    torch.nn.init.normal_(net.readout.weight, std=0.5)       # zero at init (muP)
    return net.cuda()


@needs_cuda
@pytest.mark.parametrize('task,name', [('se2c', 'hybrid4c_causal'), ('se2d', 'hybrid4')])
def test_ieee_model_matches_the_float64_recurrence(task, name):
    """A whole inert model: ieee against the same weights with every scan
    replaced by the float64 recurrence; predictions and every parameter's
    gradient."""
    from mapping.models.se2d.hybrid4 import GatedDeltaNet
    nets = {'ieee': build(name, 'ieee', task), 'exact': build(name, 'tf32', task)}
    for m in nets['exact'].modules():
        if isinstance(m, GatedDeltaNet):
            m.scan = exact64
    assert all(m.scan_precision == 'ieee' for m in nets['ieee'].modules()
               if isinstance(m, GatedDeltaNet))
    obs, odo, pose, qt = inputs(task, 8, b=2, seed=3, device='cuda')
    res = {}
    for k, net in nets.items():
        net.zero_grad()
        y = net(obs, odo, pose, query_time=qt)
        R = torch.randn(y.shape, device='cuda', generator=torch.Generator(device='cuda').manual_seed(2))
        (y * R).sum().backward()
        res[k] = (y.detach().double(), {n: p.grad.double() for n, p in net.named_parameters()
                                        if p.grad is not None})
    (y_i, g_i), (y_x, g_x) = res['ieee'], res['exact']
    assert set(g_i) == set(g_x)
    fwd = (y_i - y_x).abs().max().item()
    rel = {n: ((g_i[n] - g_x[n]).norm() / g_x[n].norm().clamp(min=1e-30)).item() for n in g_x}
    worst = max(rel, key=rel.get)
    print(f'{name}: forward max |d| {fwd:.2e} (|y| max {y_x.abs().max():.2f}); '
          f'worst gradient {worst} rel {rel[worst]:.2e}')
    assert fwd < 1e-6 * y_x.abs().max()     # |y| is ~50 here (readout std 0.5)
    assert rel[worst] < 1e-5


# ---- tf32: the code before the option -------------------------------------

@pytest.mark.parametrize('device', DEVICES)
def test_tf32_matches_the_commit_before(device, tmp_path):
    """Every se2d and se2c model, mixed and inert, at the default: bit-
    identical to REF (fresh processes, fp32, TF32 matmuls off)."""
    modes = ['mixed', 'inert']
    ref = outputs(checkout(REF, tmp_path), device, tmp_path, 'ref', modes)
    new = outputs(HERE, device, tmp_path, 'new', modes)
    res = compare(ref, new)
    assert len(res) >= (6 if device == 'cpu' else 30), sorted(res)
    assert any('hybrid4c_causal/inert' in k for k in res) or device == 'cpu'
    changed = {k: v for k, v in res.items() if v[1] != 0}
    assert not changed, changed


def test_config_field_reaches_model_record_and_args():
    import run as entry

    def build_cfg(*o):
        return entry.build(entry.resolved(entry.load(['model=se2c/hybrid4c_causal', *o])))
    tf32, ieee = build_cfg(), build_cfg('model.scan_precision=ieee', 'model.query_mode=inert')
    with pytest.raises(Exception, match='must be one of'):
        build_cfg('model.scan_precision=fp64')
    assert 'scan_precision' not in tf32.model.record()['kwargs']       # as before the field
    assert not hasattr(tf32.legacy_args(), 'scan_precision')
    assert ieee.model.record()['kwargs']['scan_precision'] == 'ieee'
    assert ieee.legacy_args().scan_precision == 'ieee'
    net = ieee.model.build(spec('se2c'))
    from mapping.models.se2d.hybrid4 import GatedDeltaNet
    gdn = [m for m in net.modules() if isinstance(m, GatedDeltaNet)]
    assert gdn and all(m.scan_precision == 'ieee' for m in gdn)
    # a checkpoint's arch record rebuilds it
    from mapping.checkpoint import build_from_arch
    again = build_from_arch(ieee.model.record(), spec('se2c'))
    assert again.scan_precision == 'ieee' and again.query_mode == 'inert'


@pytest.mark.parametrize('model', ['se2c/hybrid4c', 'se2c/hybrid4c_causal', 'se2d/hybrid4',
                                   'se2d/hybrid4_causal'])
def test_declared_by_the_gdn_models(model):
    import run as entry
    m = entry.build(entry.resolved(entry.load([f'model={model}', 'model.scan_precision=ieee'])))
    assert m.model.record()['kwargs']['scan_precision'] == 'ieee'


@pytest.mark.parametrize('model', ['se2c/vanilla4c', 'se2c/vanilla4c_causal', 'se2d/vanilla4',
                                   'se2c/hybrid4c_time', 'se2d/hybrid4_time', 'se2d/hybrid4_mix_gdnl2r',
                                   'se2d/hybrid4_causal_top1',
                                   'se2d/hybrid4_noconv', 'z2/hybrid4'])
def test_ieee_is_refused_where_not_declared(model):
    import run as entry
    entry.build(entry.resolved(entry.load([f'model={model}'])))           # tf32 is fine
    with pytest.raises(Exception, match='does not implement'):
        entry.build(entry.resolved(entry.load([f'model={model}', 'model.scan_precision=ieee'])))


def test_a_subclass_does_not_inherit_ieee():
    from mapping.models.se2c.hybrid4c import Hybrid4c

    class Sub(Hybrid4c):
        pass
    with pytest.raises(ValueError, match='does not implement'):
        Sub(spec('se2c'), 64, scan_precision='ieee')


def test_from_a_run_saved_before_the_field(tmp_path):
    """A config.yaml without model.scan_precision reruns as tf32 (with its
    sources: no drift; an older run: the field is added), and takes
    model.scan_precision=ieee as an override."""
    import run as entry
    import train
    from omegaconf import OmegaConf
    d = tmp_path / 'r'
    cfg = entry.resolved(entry.load(['model=se2c/hybrid4c_causal', f'run.out={d}']))
    os.makedirs(d)
    train.write_sources(str(d), cfg, {'overrides': ['model=se2c/hybrid4c_causal', f'run.out={d}'],
                                      'experiment': None})
    saved = OmegaConf.load(d / 'config.yaml')
    del saved.model['scan_precision']
    OmegaConf.save(saved, d / 'config.yaml')
    again, _ = entry.rerun(str(d / 'config.yaml'), [])
    assert again.model.scan_precision == 'tf32'
    again, _ = entry.rerun(str(d / 'config.yaml'), ['model.scan_precision=ieee'])
    assert entry.build(again).model.record()['kwargs']['scan_precision'] == 'ieee'
    os.remove(d / 'overrides.yaml')                       # an older run: no sources
    again, _ = entry.rerun(str(d / 'config.yaml'), [])
    assert again.model.scan_precision == 'tf32'
    again, _ = entry.rerun(str(d / 'config.yaml'), ['model.scan_precision=ieee'])
    assert entry.build(again).model.scan_precision == 'ieee'


# ---- the environment setting never leaks ----------------------------------

LEAK_SCRIPT = r"""
import json, os, sys, torch
sys.path.insert(0, sys.argv[1])
from tests.test_scan_precision import hard_inputs, tf32_scan
from mapping.models.scan_precision import ieee_scan
order = sys.argv[2].split(',')
ins = hard_inputs(l=200)
out = {}
for i, name in enumerate(order):
    f = ieee_scan if name.startswith('ieee') else tf32_scan
    xs = [x.detach().clone().requires_grad_(True) for x in ins]
    o = f(*xs)
    o.sum().backward()
    out[name] = [o.detach().double().sum().item(), o.detach().double().abs().sum().item()] + [
        x.grad.double().abs().sum().item() for x in xs]
    out[name + '#env'] = os.environ.get('TRITON_F32_DEFAULT')
print(json.dumps(out))
"""


@needs_cuda
def test_the_f32_setting_does_not_leak(tmp_path):
    script = tmp_path / 'leak.py'
    script.write_text(LEAK_SCRIPT)
    env = {k: v for k, v in os.environ.items() if k != 'TRITON_F32_DEFAULT'}

    def run(order):
        r = subprocess.run([sys.executable, str(script), HERE, ','.join(order)], cwd=HERE,
                           env=env, capture_output=True, text=True, check=True)
        return json.loads(r.stdout.strip().splitlines()[-1])
    a = run(['tf32_a', 'ieee', 'tf32_b'])          # tf32 compiled first
    b = run(['ieee', 'tf32_a'])                    # ieee compiled first
    assert a['tf32_a'] == a['tf32_b'] == b['tf32_a']
    assert a['ieee'] == b['ieee']
    assert a['ieee'] != a['tf32_a']
    assert all(v is None for k, v in {**a, **b}.items() if k.endswith('#env'))
