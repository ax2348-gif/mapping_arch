"""The high-passed natural-image texture ``imagenette_hp``
(``scripts/get_imagenette_hp.py``: data/imagenette64.npz minus its
Gaussian blur, sigma 2 px, one fixed affine map back to uint8).

Checks: the source bank and the ``imagenette`` / ``mnist`` settings are
untouched; the derived file has the source's keys, counts and labels plus
its identity tag and filter parameters, and is reproduced exactly by the
script's filter; the filter commutes with the dihedral symmetries; a bank
whose identity disagrees with the texture is refused; SE(2) batches at
step 3 with all odometry dropped and per-step queries pass the same
consistency checks as the other textures. Skipped per test when a bank
file is absent."""
import hashlib
import importlib.util
import json
import os

import numpy as np
import pytest

import mapping.tasks as tasks
from mapping import se2
from mapping.data import (TEXTURES, UNKNOWN, apply_dihedral, check_texture,
                          coverage, crop_windows, load_bank, make_batch,
                          render_map, texture_settings)
from mapping.loader import load_banks

HP = TEXTURES['imagenette_hp']['path']
NATURAL = TEXTURES['imagenette']['path']
MNIST = TEXTURES['mnist']['path']
# array contents of data/imagenette64.npz (key, dtype, shape, bytes), as
# pinned in scripts/get_imagenette_hp.py; the high-pass work must not touch it
NATURAL_CONTENT_SHA256 = ('c6d54bf729f1261b1e9fae42d95b0dd3'
                          '989bc9005cd1c58ba115ab85a85b009a')

need_hp = pytest.mark.skipif(
    not os.path.exists(HP),
    reason=f'{HP} absent (python -I scripts/get_imagenette_hp.py)')
need_natural = pytest.mark.skipif(
    not os.path.exists(NATURAL),
    reason=f'{NATURAL} absent (python -I scripts/get_imagenette.py)')


def script():
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'scripts', 'get_imagenette_hp.py')
    spec = importlib.util.spec_from_file_location('get_imagenette_hp', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


@pytest.fixture(scope='module')
def hp_raw():
    return load(HP)


@pytest.fixture(scope='module')
def hp_bank():
    return load_bank(HP, 0, 'train', True)[0]


def test_existing_settings_unchanged():
    assert TEXTURES['mnist'] == dict(path='data/mnist.npz', tile=28,
                                     cutoff=32, start_window=28)
    assert TEXTURES['imagenette'] == dict(path='data/imagenette64.npz',
                                          tile=64, cutoff=0, start_window=64)
    assert texture_settings('imagenette_hp') == dict(tile=64, cutoff=0,
                                                     start_window=64)


@need_natural
def test_source_bank_unchanged():
    src = load(NATURAL)
    assert script().content_sha256(src) == NATURAL_CONTENT_SHA256
    assert script().SRC_CONTENT_SHA256 == NATURAL_CONTENT_SHA256
    assert 'texture' not in src                  # untagged, as before
    assert check_texture(NATURAL, 'imagenette') is None


@need_hp
def test_hp_bank_file(hp_raw):
    assert set(hp_raw) == {'train_images', 'train_labels', 'test_images',
                           'test_labels', 'texture', 'filter',
                           'filter_scale'}
    assert hp_raw['train_images'].shape == (9469, 64, 64)
    assert hp_raw['test_images'].shape == (3925, 64, 64)
    assert hp_raw['train_images'].dtype == np.uint8
    assert str(hp_raw['texture']) == 'imagenette_hp'
    params = json.loads(str(hp_raw['filter']))
    assert params['sigma_px'] == 2.0 and params['percentile'] == 99.8
    assert params['source_content_sha256'] == NATURAL_CONTENT_SHA256
    assert params['scale'] == float(hp_raw['filter_scale'])
    assert check_texture(HP, 'imagenette_hp') == str(hp_raw['filter'])
    # no local mean left: the bank sits at mid-gray, little saturation
    im = hp_raw['test_images']
    assert abs(im.mean() - 128) < 1
    assert np.mean((im == 0) | (im == 255)) < 0.005


@need_hp
@need_natural
def test_hp_reproduced_from_source(hp_raw):
    """Labels are the source's; the stored test split is exactly the
    script's filter of the source with the recorded train-split scale."""
    s, src = script(), load(NATURAL)
    for split in ('train', 'test'):
        assert np.array_equal(hp_raw[f'{split}_labels'],
                              src[f'{split}_labels'])
    scale = float(hp_raw['filter_scale'])
    assert np.array_equal(s.quantise(s.highpass(src['test_images']), scale),
                          hp_raw['test_images'])


def test_filter_commutes_with_dihedral():
    """Isotropic kernel, symmetric border: filtering a tile under any of
    its 8 symmetries is the symmetry of the filtered tile."""
    s = script()
    x = np.random.default_rng(0).integers(0, 256, (6, 64, 64)).astype(
        np.uint8)
    hx = s.highpass(x)
    for sym in range(8):
        k = np.full(len(x), sym, np.int8)
        assert np.allclose(s.highpass(apply_dihedral(x, k)),
                           apply_dihedral(hx, k), atol=1e-9)


def test_kernel_matches_reference():
    """The numpy Gaussian is scipy.ndimage's (reflect, truncate 4), when
    scipy is installed; it is not a dependency."""
    ndimage = pytest.importorskip('scipy.ndimage')
    s = script()
    x = np.random.default_rng(1).random((3, 64, 64)) * 255
    ref = ndimage.gaussian_filter(x, sigma=(0, 2, 2), mode='reflect',
                                  truncate=4.0)
    assert np.allclose(s.blur(x), ref, atol=1e-9)


@need_hp
def test_mismatched_bank_raises(hp_bank):
    cfg, _ = tasks.config_and_spec('se2', 32, 8, 0, texture='imagenette_hp')
    train, test = load_banks(HP, cfg, 'imagenette_hp')
    assert train.shape[-2:] == test.shape[-2:] == (64, 64)
    with pytest.raises(ValueError, match='tag'):  # hp file as raw imagenette
        load_banks(HP, cfg, 'imagenette')
    if os.path.exists(NATURAL):                    # raw file as hp
        with pytest.raises(ValueError, match='tag'):
            load_banks(NATURAL, cfg, 'imagenette_hp')
    if os.path.exists(MNIST):
        with pytest.raises(ValueError):
            load_banks(MNIST, cfg, 'imagenette_hp')
    mnist_cfg, _ = tasks.config_and_spec('se2', 32, 8, 0)
    with pytest.raises(ValueError, match='tile'):
        load_banks(HP, mnist_cfg)
    with pytest.raises(ValueError, match='tile'):
        make_batch(np.random.default_rng(0), mnist_cfg, hp_bank, 2)


@need_hp
@pytest.mark.parametrize('qt', ['uniform', 'last'])
def test_se2_hp_consistent(hp_bank, qt):
    """As ``test_texture.test_se2_natural_consistent``, on the high-passed
    bank at step 3 with every odometry symbol dropped."""
    c, spec = tasks.config_and_spec('se2', 32, 8, 0, qt, 3, 1.0,
                                    'imagenette_hp')
    assert (c.tile, c.cutoff, c.start_window) == (64, 0, 64)
    b = make_batch(np.random.default_rng(3), c, hp_bank, 16)
    assert (b['odometry'] == UNKNOWN[True]).all()
    disp, hh = se2.compose(b['odometry_true'], c.step, b['headings'][:, 0])
    assert (hh == b['headings']).all()
    assert (b['positions'][:, 0:1] + disp == b['positions']).all()
    assert (np.abs(b['query_pose'][..., :2]) <= spec.offset_max).all()
    for i, e in enumerate(b['episodes']):
        image = render_map(hp_bank, e.tile_grid, c.tile, e.tile_sym)
        world_obs = crop_windows(image, e.positions, c.obs)
        assert (se2.face(world_obs, b['headings'][i])
                == b['observations'][i]).all()
        q, t = b['query_pose'][i], b['query_time'][i]
        corners = e.positions[t] + se2.unrotate(q[:, :2], b['headings'][i, t])
        assert (corners == e.queries).all()
        qh = (q[:, 2] + b['headings'][i, t]) % 4
        assert (se2.face(crop_windows(image, e.queries, c.obs), qh)
                == b['target'][i]).all()
        h, w = np.array(e.tile_grid.shape) * c.tile
        seen = coverage(e.positions, c.obs, h, w, first=True)
        for corner, tq in zip(e.queries, e.query_times):
            win = seen[corner[0]:corner[0] + c.obs,
                       corner[1]:corner[1] + c.obs]
            assert ((win >= 0) & (win <= tq)).all()
        if qt == 'last':
            assert (t == c.steps).all()


@need_hp
def test_config_from_args_hp():
    _, spec = tasks.config_and_spec('se2', 32, 8, 0)
    c = tasks.config_from_args('se2', spec, {'seed': 0, 'bank': HP,
                                             'query_times': 'uniform',
                                             'texture': 'imagenette_hp'})
    assert (c.tile, c.cutoff, c.start_window) == (64, 0, 64)
