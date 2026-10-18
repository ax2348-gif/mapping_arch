"""Small CPU geometry/wiring checks, with the existing reference delta scan.

Nonzero readout makes causality and inert-query checks non-vacuous. CPU
backward uses SDPA with the actual BlockMask's predicate: this torch version
does not support FlexAttention CPU backward. These do not validate fused GPU
kernels or establish that the model learns the task.
"""
import math

import pytest
import torch

from mapping.hybrid6_geometry import (patch_calibration, patchify, unpatchify,
                                     planar_pose, pose_features, sensor_features)
from mapping.models.se2d.hybrid4 import GatedDeltaNet, delta_scan
from mapping.models.se2d.hybrid6 import Hybrid6, token_order
from mapping.models.se2c.hybrid6c import Hybrid6c
from mapping.task_se2d import TaskSpec
from mapping.task_se2c import TaskSpec as ContinuousSpec


@pytest.fixture(autouse=True)
def cpu_threads(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    import mapping.models.se2d.hybrid4 as backbone
    def dense_attention(q, k, v, block_mask):
        b, heads, length, _ = q.shape
        mask = block_mask.mask_mod(
            torch.arange(b)[:, None, None, None],
            torch.arange(heads)[None, :, None, None],
            torch.arange(length)[None, None, :, None],
            torch.arange(k.shape[-2])[None, None, None, :])
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    monkeypatch.setattr(backbone, 'attend', dense_attention)
    yield
    torch.set_num_threads(previous)


def model(continuous=False, causal=False, mode='inert'):
    torch.manual_seed(11)
    cls = Hybrid6c if continuous else Hybrid6
    spec = ContinuousSpec() if continuous else TaskSpec(odometry=17)
    net = cls(spec, 64, query_mode=mode, causal=causal)
    for layer in net.modules():
        if isinstance(layer, GatedDeltaNet):
            layer.scan = delta_scan
    torch.nn.init.normal_(net.readout.weight, std=0.1)
    return net


def inputs(continuous=False):
    torch.manual_seed(2)
    obs = torch.rand(1, 3, 3, 6)
    odo = torch.tensor([[3, 10]])
    pose = torch.tensor([[[2, -3, 1], [-4, 2, 3]]])
    if continuous:
        odo = torch.tensor([[[1.2, -2.1, .3], [-3., 2., -1.2]]])
        pose = pose.float()
        pose[..., 2] *= math.pi / 2
    return obs, odo, pose, torch.tensor([[0, 2]])


@pytest.mark.parametrize('shape', [(9, 9), (15, 15), (3, 12), (6, 9)])
def test_patch_roundtrip(shape):
    x = torch.arange(2 * math.prod(shape)).reshape(2, *shape)
    torch.testing.assert_close(unpatchify(patchify(x), *shape), x)


@pytest.mark.parametrize('angle', [0., math.pi / 2, -.73, math.pi])
@pytest.mark.parametrize('focal', [.5, 1., 2.])
def test_camera_samples_match_planar_renderer(angle, focal):
    cal = patch_calibration(9, 15, 'cpu', focal)
    a = -cal[:, 2:4]
    u = torch.tensor([[x, y] for y in [-1., 0., 1.] for x in [-1., 0., 1.]])
    K = torch.eye(3).expand(len(cal), 3, 3).clone()
    K[:, 0, 0], K[:, 1, 1] = cal[:, 0], cal[:, 1]
    K[:, :2, 2] = cal[:, 2:4]
    p = planar_pose(torch.tensor([4., -7., angle]))
    homogeneous = torch.cat((u, torch.ones(9, 1)), -1)
    rays = torch.einsum('ij,pjk,nk->pni', p[:3, :3], K.inverse(), homogeneous)
    hits = p[:3, 3] + rays / rays[..., 2:3]
    # Independent expected body->anchor rotation in repository (dy,dx) order.
    from mapping.se2c.geometry import unrotate
    local_yx = ((a[:, None] + u) / focal).flip(-1)
    expected_yx = unrotate(local_yx, torch.tensor(angle)) + torch.tensor([4., -7.])
    torch.testing.assert_close(hits[..., :2].flip(-1), expected_yx, atol=3e-6, rtol=1e-6)


def test_three_roles_and_shared_encoders():
    net = model()
    tables = [m for m in net.modules() if isinstance(m, torch.nn.Embedding)]
    assert tables == [net.roles] and net.roles.num_embeddings == 3
    assert not hasattr(net, 'patch_pos') and not hasattr(net, 'query_pos')
    # The same calibration means the same features, independent of grid size.
    small = sensor_features(patch_calibration(9, 9, 'cpu'))
    large = sensor_features(patch_calibration(15, 15, 'cpu')).reshape(5, 5, -1)
    torch.testing.assert_close(small.reshape(3, 3, -1), large[1:4, 1:4])


def test_local_odometry_decoder_and_unknown():
    net = model()
    symbols = torch.arange(17)[None]
    poses, _, known = net._task_poses(symbols, torch.zeros(1, 1, 3, dtype=torch.long))
    for i in range(16):
        move = [(-9., 0.), (0., 9.), (9., 0.), (0., -9.)][i % 4]
        torch.testing.assert_close(poses[0, i, :2, 3], torch.tensor(move).flip(0))
    assert known[0, :16].all() and not known[0, 16]
    f = pose_features(poses, known)
    assert torch.count_nonzero(f[0, 16]) == 0
    assert not torch.equal(f[0, 16], pose_features(torch.eye(4)))


@pytest.mark.parametrize('continuous', [False, True])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('mode', ['mixed', 'inert'])
def test_forward_backward_and_runtime_grids(continuous, causal, mode):
    net = model(continuous, causal, mode)
    x = inputs(continuous)
    y = net(*x, query_shape=(9, 3))
    assert y.shape == (1, 2, 9, 3) and torch.isfinite(y).all()
    y.square().mean().backward()
    for name in ['roles', 'pose_embed', 'sensor_embed', 'patch_embed']:
        grad = getattr(net, name).weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


@pytest.mark.parametrize('causal', [False, True])
def test_inert_query_set_and_history_invariance(causal):
    net = model(causal=causal)
    obs, odo, pose, qt = inputs()
    histories = []
    handle = net.blocks[-1].register_forward_hook(lambda m, i, out: histories.append(out.detach()))
    with torch.no_grad():
        full = net(obs, odo, pose, qt)
        alone = net(obs, odo, pose[:, 1:], qt[:, 1:])
        empty = net(obs, odo, pose[:, :0], qt[:, :0])
    handle.remove()
    torch.testing.assert_close(full[:, 1:], alone, atol=3e-6, rtol=1e-5)
    assert empty.shape == (1, 0, 3, 6)
    n_hist = 3 * 3 - 1
    restored = []
    for x, time in zip(histories, [qt, qt[:, 1:], qt[:, :0]]):
        order = token_order(time, 3, 2, 2)
        restored.append(x.gather(1, order.argsort(1)[..., None].expand_as(x))[:, :n_hist])
    torch.testing.assert_close(restored[0], restored[1], atol=3e-6, rtol=1e-5)
    torch.testing.assert_close(restored[0], restored[2], atol=3e-6, rtol=1e-5)


def test_causal_future_invariance_with_nonvacuous_control():
    net = model(causal=True)
    obs, odo, pose, qt = inputs()
    with torch.no_grad():
        before = net(obs, odo, pose, qt)
        changed = obs.clone()
        changed[:, 1:] = 1 - changed[:, 1:]
        after = net(changed, (odo + 1) % 16, pose, qt)
        past = obs.clone()
        past[:, 0] = 1 - past[:, 0]
        affected = net(past, odo, pose, qt)
    torch.testing.assert_close(before[:, 0], after[:, 0], atol=1e-6, rtol=1e-5)
    assert (before[:, 0] - affected[:, 0]).abs().max() > 1e-5


def test_explicit_camera_interface_has_full_pose_and_calibration_gradients():
    net = model(causal=True)
    pixels = torch.rand(1, 2, 2, 9)
    obs_pose = torch.eye(4).expand(1, 2, 2, 4, 4).clone().requires_grad_()
    obs_cal = patch_calibration(3, 6, 'cpu').expand(1, 2, 2, 5).clone().requires_grad_()
    odo = torch.eye(4).expand(1, 1, 4, 4).clone().requires_grad_()
    # A tilted, translated query camera, not restricted to the planar adapter.
    query = torch.eye(4).expand(1, 1, 1, 4, 4).clone()
    angle = .3
    query[..., 1, 1] = query[..., 2, 2] = math.cos(angle)
    query[..., 1, 2], query[..., 2, 1] = -math.sin(angle), math.sin(angle)
    query[..., :3, 3] = torch.tensor([1., 2., 3.])
    query.requires_grad_()
    cal = patch_calibration(3, 3, 'cpu', 2.).reshape(1, 1, 1, 5).requires_grad_()
    y = net.forward_cameras(pixels, obs_pose, obs_cal, odo, query, cal)
    y.square().sum().backward()
    for value in [obs_pose, obs_cal, odo, query, cal]:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0
    assert query.grad[..., 2, 3].abs().sum() > 0  # depth translation is not discarded
    assert query.grad[..., 1, 2].abs().sum() > 0  # pitch is not discarded


def test_invalid_inputs_rejected():
    net = model()
    obs, odo, pose, qt = inputs()
    with pytest.raises(ValueError, match='query_time'):
        net(obs, odo, pose, qt + 3)
    with pytest.raises(ValueError, match='multiples'):
        net(obs, odo, pose, qt, query_shape=(4, 3))
    bad = patch_calibration(3, 3, 'cpu')
    bad[:, 0] = 0
    with pytest.raises(ValueError, match='positive'):
        sensor_features(bad)


@pytest.mark.parametrize('name', ['se2d/hybrid6', 'se2d/hybrid6_causal',
                                 'se2c/hybrid6c', 'se2c/hybrid6c_causal'])
def test_harness_config(name):
    import run
    exp = run.build(run.resolved(run.load([
        f'model={name}', 'model.query_mode=inert', 'model.scan_precision=ieee'])))
    spec = ContinuousSpec() if name.startswith('se2c/') else exp.task.generator.data(exp.task, 0)[1]
    net = exp.model.build(spec)
    assert net.causal == name.endswith('_causal')
    assert net.query_mode == 'inert'
    assert all(m.scan_precision == 'ieee' for m in net.modules() if isinstance(m, GatedDeltaNet))


def test_cpu_attention_reference_matches_flex(monkeypatch):
    import mapping.models.se2d.hybrid4 as backbone
    from mapping.models.query_mode import attend
    net = model(causal=True)
    with torch.no_grad():
        dense = net(*inputs())
        monkeypatch.setattr(backbone, 'attend', attend)
        actual = net(*inputs())
    torch.testing.assert_close(dense, actual, atol=3e-6, rtol=1e-5)


def test_initial_proposal_matches_parent_schedule_prefix():
    import run
    from mapping.config import WSD
    exp = run.build(run.resolved(run.load(['+experiment=hybrid6_initial'])))
    assert exp.task.generator.path == 'saw'
    assert exp.model.query_mode == 'inert' and exp.model.cls.__name__.endswith('Causal')
    parent = WSD(warmup=100, cooldown=.2)
    for step in [0, 99, 100, 50000, 99999]:
        assert exp.schedule.lr(step, exp.run.steps, exp.optim.lr) == parent.lr(step, 400000, 3e-4)


def test_submission_arguments_parse():
    import shlex
    from pathlib import Path
    import run
    script = Path('jobs/hybrid6/initial.sbatch').read_text().replace('\\\n', '')
    command = next(line for line in script.splitlines() if line.startswith('python run.py '))
    cfg = run.resolved(run.load(shlex.split(command)[2:]))
    exp = run.build(cfg)
    assert exp.run.steps == 100000 and 'approved initial Torch run' in exp.run.note
