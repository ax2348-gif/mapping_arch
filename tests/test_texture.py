"""The texture setting (``mapping.data.TEXTURES``): the natural-image bank
(Imagenette, 64 x 64 grayscale tiles, ``scripts/get_imagenette.py``) as a
generator setting of both tasks, not a new task.

The MNIST default is pinned elsewhere (golden hashes in
``test_contract_z2.py`` and ``test_generator_options.py``). Here: the
texture fixes tile, cutoff and start window together; a bank whose tile
side disagrees with the config is refused; the spec does not depend on the
texture; and SE(2) batches on the natural bank at a small step with all
odometry dropped pass the same consistency checks as MNIST ones. Skipped
when data/imagenette64.npz is absent."""
import inspect
import os

import numpy as np
import pytest
import torch

import mapping.tasks as tasks
from mapping import se2d
from mapping.data import (TEXTURES, UNKNOWN, Config, apply_dihedral,
                          coverage, crop_windows, load_bank, make_batch,
                          render_map, texture_settings)
from mapping.loader import load_banks

NATURAL = TEXTURES['imagenette']['path']
MNIST = TEXTURES['mnist']['path']
CUDA = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not os.path.exists(NATURAL),
    reason=f'{NATURAL} absent (python -I scripts/get_imagenette.py)')


@pytest.fixture(scope='module')
def raw():
    with np.load(NATURAL) as f:
        return {k: f[k] for k in f}


@pytest.fixture(scope='module')
def bank():
    return load_bank(NATURAL, 0, 'train', True)[0]


def test_bank_file(raw):
    assert set(raw) == {'train_images', 'train_labels', 'test_images',
                        'test_labels'}
    assert raw['train_images'].shape == (9469, 64, 64)
    assert raw['test_images'].shape == (3925, 64, 64)
    for split in ('train', 'test'):
        assert raw[f'{split}_images'].dtype == np.uint8
        assert sorted(np.unique(raw[f'{split}_labels'])) == list(range(10))


def test_texture_settings():
    assert texture_settings('mnist') == dict(tile=28, cutoff=32,
                                             start_window=28)
    assert texture_settings('imagenette') == dict(tile=64, cutoff=0,
                                                  start_window=64)
    assert Config() == Config(**texture_settings('mnist'))
    with pytest.raises(ValueError):
        texture_settings('cifar')
    with pytest.raises(ValueError):          # start window below one tile
        Config(tile=64)


def test_config_and_spec_follow_texture():
    for task in tasks.TASKS:
        c0, s0 = tasks.config_and_spec(task, 32, 8, 0)
        c, s = tasks.config_and_spec(task, 32, 8, 0, 'uniform', 3, 1.0,
                                     'imagenette')
        _, s3 = tasks.config_and_spec(task, 32, 8, 0, 'uniform', 3, 1.0)
        assert (c0.tile, c0.cutoff, c0.start_window) == (28, 32, 28)
        assert (c.tile, c.cutoff, c.start_window) == (64, 0, 64)
        assert s == s3                       # the spec ignores the texture
        assert s0 == tasks.get(task)['task'].TaskSpec()


def test_config_from_args_texture():
    _, spec = tasks.config_and_spec('se2d', 32, 8, 0)
    old = {'seed': 0, 'query_times': 'uniform', 'bank': MNIST}
    assert tasks.config_from_args('se2d', spec, old).tile == 28
    c = tasks.config_from_args('se2d', spec, dict(old, texture='imagenette'))
    assert (c.tile, c.cutoff, c.start_window) == (64, 0, 64)


def test_cutoff_zero_keeps_pixels(raw):
    plain = load_bank(NATURAL, 0, 'test', False)[0]
    assert np.array_equal(plain, raw['test_images'])
    assert (plain == 0).mean() < 0.05        # natural images are not black


def test_tile_inferred_and_validated():
    c, _ = tasks.config_and_spec('se2d', 32, 8, 0, texture='imagenette')
    train, test = load_banks(NATURAL, c)
    assert train.shape[-2:] == test.shape[-2:] == (c.tile, c.tile) == (64, 64)


def test_mismatched_tile_raises(bank):
    mnist_cfg, _ = tasks.config_and_spec('se2d', 32, 8, 0)
    with pytest.raises(ValueError, match='tile'):
        load_banks(NATURAL, mnist_cfg)
    with pytest.raises(ValueError, match='tile'):
        make_batch(np.random.default_rng(0), mnist_cfg, bank, 2)
    if os.path.exists(MNIST):
        nat_cfg, _ = tasks.config_and_spec('se2d', 32, 8, 0,
                                           texture='imagenette')
        with pytest.raises(ValueError, match='tile'):
            load_banks(MNIST, nat_cfg)


def test_dihedral_64(raw, bank):
    """The precomputed (8, N, 64, 64) bank and the render agree with the
    symmetry definition: k quarter turns, plus 4 to mirror first."""
    tiles = raw['train_images'][:5]
    for s in range(8):
        want = tiles[:, :, ::-1] if s >= 4 else tiles
        want = np.rot90(want, s % 4, axes=(-2, -1))
        assert np.array_equal(bank[s, :5], want)
        assert np.array_equal(apply_dihedral(tiles, np.full(5, s)), want)
    c = Config(**texture_settings('imagenette'), heading=True)
    e = make_batch(np.random.default_rng(1), c, bank, 1)['episodes'][0]
    image = render_map(bank, e.tile_grid, c.tile, e.tile_sym)
    h, w = e.tile_grid.shape
    assert image.shape == (h * 64, w * 64)
    for i in range(h):
        for j in range(w):
            block = image[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64]
            assert np.array_equal(block, bank[e.tile_sym[i, j],
                                              e.tile_grid[i, j]])


@pytest.mark.parametrize('step', [3, 1, 9])
@pytest.mark.parametrize('qt', ['uniform', 'last'])
def test_se2d_natural_consistent(bank, step, qt):
    """SE(2) on the natural bank, all odometry dropped: composing the true
    odometry recovers the poses; faced observations, query corners and
    faced targets match a re-rendered episode; every query is observed by
    its step and within offset_max; the start phase spans one 64-px tile."""
    c, spec = tasks.config_and_spec('se2d', 32, 8, 0, qt, step, 1.0,
                                    'imagenette')
    b = make_batch(np.random.default_rng(3), c, bank, 16)
    assert (b['odometry'] == UNKNOWN[True]).all()
    disp, hh = se2d.compose(b['odometry_true'], c.step, b['headings'][:, 0])
    assert (hh == b['headings']).all()
    assert (b['positions'][:, 0:1] + disp == b['positions']).all()
    assert (np.abs(b['query_pose'][..., :2]) <= spec.offset_max).all()
    for i, e in enumerate(b['episodes']):
        image = render_map(bank, e.tile_grid, c.tile, e.tile_sym)
        world_obs = crop_windows(image, e.positions, c.obs)
        assert (se2d.face(world_obs, b['headings'][i])
                == b['observations'][i]).all()
        q, t = b['query_pose'][i], b['query_time'][i]
        corners = e.positions[t] + se2d.unrotate(q[:, :2], b['headings'][i, t])
        assert (corners == e.queries).all()
        qh = (q[:, 2] + b['headings'][i, t]) % 4
        assert (se2d.face(crop_windows(image, e.queries, c.obs), qh)
                == b['target'][i]).all()
        h, w = np.array(e.tile_grid.shape) * c.tile
        assert (e.positions >= 0).all() and (e.positions[:, 0] + c.obs <= h).all()
        assert (e.positions[:, 1] + c.obs <= w).all()
        seen = coverage(e.positions, c.obs, h, w, first=True)
        for corner, tq in zip(e.queries, e.query_times):
            win = seen[corner[0]:corner[0] + c.obs,
                       corner[1]:corner[1] + c.obs]
            assert ((win >= 0) & (win <= tq)).all()
        if qt == 'last':
            assert (t == c.steps).all()


def test_start_phase_spans_tile(bank):
    """Local maps start on a tile boundary, so the start corner modulo the
    tile is the phase; with start_window 64 it covers the whole tile."""
    c, _ = tasks.config_and_spec('z2', 32, 8, 0, texture='imagenette')
    b = make_batch(np.random.default_rng(0), c, bank, 400)
    phase = np.concatenate([e.positions[0] % c.tile for e in b['episodes']])
    assert phase.min() == 0 and 60 <= phase.max() < 64


def test_z2_natural_offsets(bank):
    c, _ = tasks.config_and_spec('z2', 32, 8, 0, 'uniform', 3, 1.0,
                                 'imagenette')
    b = make_batch(np.random.default_rng(4), c, bank, 8)
    assert (b['directions'] == UNKNOWN[False]).all()
    for i, e in enumerate(b['episodes']):
        image = render_map(bank, e.tile_grid, c.tile, e.tile_sym)
        assert (crop_windows(image, e.positions, c.obs)
                == b['observations'][i]).all()
        assert (b['query_offset'][i]
                == e.queries - e.positions[e.query_times]).all()
        assert (crop_windows(image, e.queries, c.obs) == b['target'][i]).all()


def cases():
    for task in sorted(tasks.TASKS):
        for name in sorted(tasks.get(task)['models'].REGISTRY):
            yield task, name


@pytest.mark.parametrize('task,name', list(cases()))
def test_models_on_natural_bank(task, name, bank):
    """Every registered model builds from the spec (which has no tile) and
    runs on a natural-texture batch, step 3, all odometry dropped."""
    t = tasks.get(task)
    config, spec = tasks.config_and_spec(task, 32, 8, 0, 'uniform', 3, 1.0,
                                         'imagenette')
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
