"""Two generator settings of the existing tasks (not new tasks): the walk
step (``Config.step``, ``--step-px``) and odometry dropout
(``Config.odometry_dropout``, ``--odometry-dropout``).

With both at their defaults every batch is byte-identical to the code
before they existed (``test_contract_z2.py`` for Z^2, ``SE2D_GOLDEN`` here
for SE(2), hashes taken from commit 6eeb869). With dropout above 0 the
model-facing symbols are dropped, the true ones ride along under a
forbidden key, and every other field equals the dropout-0 batch."""
import hashlib
import inspect

import numpy as np
import pytest
import torch

import mapping.task
import mapping.task_se2d
import mapping.tasks as tasks
from mapping import se2d
from mapping.data import (UNKNOWN, Config, coverage, crop_windows, load_bank,
                          make_batch, render_map)
from mapping.loader import eval_set

BANK = 'data/mnist.npz'
CUDA = torch.cuda.is_available()


def digest(batch):
    """Every field but the episodes, query_time included."""
    h = hashlib.sha256()
    for k in sorted(batch):
        if k == 'episodes':
            continue
        v = np.ascontiguousarray(batch[k])
        h.update(k.encode()); h.update(str(v.dtype).encode())
        h.update(str(v.shape).encode()); h.update(v.tobytes())
    return h.hexdigest()[:16]


# from commit 6eeb869, before step_px / odometry_dropout were options
SE2D_GOLDEN = {
    'se2d train last s0': '2712dadb0800538f',
    'se2d train last s1': 'a4074a53c1755b69',
    'se2d eval last': '82358a16207d40e4',
    'z2 step3 last': '570e925a218a15c5',
    'se2d step3 last': 'c8d1853fa9655c54',
    'se2d train uniform s0': 'c039c07a762cb611',
    'se2d train uniform s1': 'a8920aa645ff5b3f',
    'se2d eval uniform': '8e7d3ce6159655a6',
    'z2 step3 uniform': '45ad1bc73b960a8a',
    'se2d step3 uniform': '931dd2dc132e9cde',
}


@pytest.fixture(scope='module')
def bank():
    return load_bank(BANK, 32, 'train', True)[0]


def test_se2d_and_step_batches_unchanged(bank):
    out = {}
    for qt in ('last', 'uniform'):
        for seed in (0, 1):
            c = Config(seed=seed, heading=True, query_times=qt)
            out[f'se2d train {qt} s{seed}'] = digest(
                make_batch(np.random.default_rng([c.seed, 0]), c, bank, 8))
        ev = eval_set(Config(heading=True, query_times=qt), bank, 8, 1, 0)[0]
        out[f'se2d eval {qt}'] = digest({k: v.numpy() for k, v in ev.items()})
        for heading in (False, True):
            c = Config(query_times=qt, step=3, heading=heading)
            out[f'{"se2d" if heading else "z2"} step3 {qt}'] = digest(
                make_batch(np.random.default_rng([0, 0]), c, bank, 8))
    assert out == SE2D_GOLDEN


def test_default_spec_unchanged():
    for name in tasks.TASKS:
        config, spec = tasks.config_and_spec(name, 32, 8, 0)
        assert spec == tasks.get(name)['task'].TaskSpec()
        assert config.step == 9 and config.odometry_dropout == 0.0
        _, spec0 = tasks.config_and_spec(name, 32, 8, 0, 'uniform', 9, 0.0)
        assert spec0 == spec


def test_spec_follows_options():
    _, s = tasks.config_and_spec('z2', 32, 8, 0, 'uniform', 3, 1.0)
    assert (s.directions, s.step_px, s.offset_max) == (5, 3, 96)
    _, s = tasks.config_and_spec('se2d', 32, 8, 0, 'uniform', 3, 0.5)
    assert (s.odometry, s.turns, s.step_px, s.offset_max) == (17, 4, 3, 96)
    assert UNKNOWN == {False: 4, True: 16}


def test_forbidden_keys():
    assert 'directions_true' in mapping.task.FORBIDDEN_KEYS
    assert 'odometry_true' in mapping.task_se2d.FORBIDDEN_KEYS


@pytest.mark.parametrize('heading', [False, True])
@pytest.mark.parametrize('step', [9, 3])
def test_dropout_keeps_everything_else(bank, heading, step):
    key = 'odometry' if heading else 'directions'
    base = make_batch(np.random.default_rng([5, 0]),
                      Config(heading=heading, step=step,
                             query_times='uniform'), bank, 64)
    assert key + '_true' not in base
    for p in (1.0, 0.5):
        b = make_batch(np.random.default_rng([5, 0]),
                       Config(heading=heading, step=step,
                              query_times='uniform', odometry_dropout=p),
                       bank, 64)
        assert set(b) == set(base) | {key + '_true'}
        for k in base:
            if k not in ('episodes', key):
                assert np.array_equal(b[k], base[k]), k
        assert np.array_equal(b[key + '_true'], base[key])
        dropped = b[key] == UNKNOWN[heading]
        assert np.array_equal(b[key][~dropped], base[key][~dropped])
        assert b[key].dtype == base[key].dtype
        if p == 1.0:
            assert dropped.all()
        else:
            assert 0.45 < dropped.mean() < 0.55
        assert (base[key] != UNKNOWN[heading]).all()


def test_loader_carries_true_odometry(bank):
    ev = eval_set(Config(heading=True, odometry_dropout=1.0), bank, 4, 1, 0)[0]
    assert (ev['odometry'] == 16).all() and ev['odometry_true'].max() < 16


@pytest.mark.parametrize('step', [1, 2, 3])
def test_se2d_small_step_consistent(bank, step):
    """Overlapping windows: the SE(2) checks of test_se2d.py, and per-step
    queries fully observed by their step and within offset_max."""
    for qt in ('last', 'uniform'):
        c = Config(heading=True, step=step, query_times=qt,
                   odometry_dropout=1.0)
        b = make_batch(np.random.default_rng(3), c, bank, 6)
        disp, hh = se2d.compose(b['odometry_true'], c.step, b['headings'][:, 0])
        assert (hh == b['headings']).all()
        assert (b['positions'][:, 0:1] + disp == b['positions']).all()
        assert (np.abs(b['query_pose'][..., :2]) <= c.steps * c.step).all()
        for i, e in enumerate(b['episodes']):
            image = render_map(bank, e.tile_grid, c.tile, e.tile_sym)
            world_obs = crop_windows(image, e.positions, c.obs)
            assert (se2d.face(world_obs, b['headings'][i])
                    == b['observations'][i]).all()
            q, t = b['query_pose'][i], b['query_time'][i]
            corners = e.positions[t] + se2d.unrotate(q[:, :2],
                                                    b['headings'][i, t])
            assert (corners == e.queries).all()
            qh = (q[:, 2] + b['headings'][i, t]) % 4
            assert (se2d.face(crop_windows(image, e.queries, c.obs), qh)
                    == b['target'][i]).all()
            h, w = np.array(e.tile_grid.shape) * c.tile
            seen = coverage(e.positions, c.obs, h, w, first=True)
            for corner, tq in zip(e.queries, e.query_times):
                win = seen[corner[0]:corner[0] + c.obs,
                           corner[1]:corner[1] + c.obs]
                assert ((win >= 0) & (win <= tq)).all()


def test_config_from_old_args():
    _, spec = tasks.config_and_spec('se2d', 32, 8, 0)
    old = {'seed': 0, 'query_times': 'uniform'}          # pre-option checkpoint
    c = tasks.config_from_args('se2d', spec, old)
    assert (c.step, c.odometry_dropout, c.query_times) == (9, 0.0, 'uniform')
    c = tasks.config_from_args('se2d', spec, dict(old, step_px=3,
                                                 odometry_dropout=1.0))
    assert (c.step, c.odometry_dropout) == (3, 1.0)


def test_checkpoint_roundtrip(tmp_path):
    from mapping.checkpoint import load_checkpoint, save_checkpoint
    from mapping.provenance import Record
    _, spec = tasks.config_and_spec('z2', 32, 8, 0, 'last', 3, 1.0)
    net = tasks.get('z2')['models'].get('mlp').build('xs', spec)
    record = Record(str(tmp_path / 'ledger.json'),          # a live record, as a run has
                    {'run_id': 'r', 'code': {'snapshot': '0' * 40},
                     'progress': {'last_step': 0, 'checkpoints': []}}, '')
    record.out = str(tmp_path)
    save_checkpoint(record, tmp_path / 'final.pt', net,
                    {'model': 'mlp', 'size': 'xs'}, spec, 'z2')
    _, _, spec2, _ = load_checkpoint(tmp_path / 'final.pt')
    assert spec2 == spec


def cases():
    for task in sorted(tasks.TASKS):
        for name in sorted(tasks.get(task)['models'].REGISTRY):
            yield task, name


@pytest.mark.parametrize('task,name', list(cases()))
def test_models_with_dropout_and_small_step(task, name, bank):
    """Every registered model builds with the grown vocabulary and the
    small offset_max, and runs on a real batch whose odometry is partly the
    unknown symbol; episodes stay isolated."""
    t = tasks.get(task)
    config, spec = tasks.config_and_spec(task, 32, 8, 0, 'uniform', 3, 0.5)
    module = t['models'].get(name)
    try:
        net = module.build(next(iter(module.SIZES)), spec)
    except Exception as e:                       # e.g. a kernel needing CUDA
        pytest.skip(f'cannot build on this machine: {e}')
    device = 'cuda' if CUDA else 'cpu'
    net = net.to(device).eval()
    b = make_batch(np.random.default_rng(0), config, bank, 4)
    b.pop('episodes')
    b = {k: torch.from_numpy(v) for k, v in b.items()}
    obs, odo, query, _, qtime = t['task'].prepare(b, device)
    unknown = UNKNOWN[t['heading']]
    assert (odo == unknown).any() and (odo != unknown).any()
    takes_time = 'query_time' in inspect.signature(net.forward).parameters
    kw = {'query_time': qtime} if takes_time else {}
    try:
        with torch.no_grad():
            out = net(obs, odo, query, **kw).float()
    except Exception as e:
        if not CUDA:
            pytest.skip(f'needs CUDA: {e}')
        raise
    assert out.shape == (4, 8, 9, 9) and torch.isfinite(out).all()
    odo2 = odo.clone()
    odo2[0] = unknown                                # all dropped in ep. 0
    with torch.no_grad():
        out2 = net(obs, odo2, query, **kw).float()
    assert (out[1:] - out2[1:]).abs().max().item() < 1e-4, 'episodes leak'
