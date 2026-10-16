"""The SE(2) task: conventions, composition, and batch consistency."""
import numpy as np

from mapping import se2d
from mapping.data import (MOVES, Config, crop_windows, load_bank,
                          make_batch, render_map)
from mapping.loader import eval_set


def test_rotate_matches_move_indices():
    # seen from heading h, world move d is body move (d - h) % 4
    for h in range(4):
        for d in range(4):
            assert (se2d.rotate(MOVES[d], h) == MOVES[(d - h) % 4]).all()
    v = np.array([[3, -7], [0, 2]])
    for k in range(4):
        assert (se2d.unrotate(se2d.rotate(v, k), k) == v).all()


def test_face_is_rot90_by_heading():
    w = np.arange(81).reshape(1, 9, 9)
    for h in range(4):
        assert (se2d.face(w, [h])[0] == np.rot90(w[0], h)).all()


def test_face_agrees_with_rotate():
    # a feature at body-frame offset b from the window centre sits at world
    # offset unrotate(b, h): rotating the image and rotating the vector agree
    img = np.zeros((9, 9), np.uint8)
    img[1, 6] = 1                                      # world offset (-3, +2)
    for h in range(4):
        faced = se2d.face(img[None], [h])[0]
        y, x = np.argwhere(faced)[0] - 4
        assert (np.array([y, x]) == se2d.rotate(np.array([-3, 2]), h)).all()


def test_compose_recovers_poses():
    rng = np.random.default_rng(0)
    d = rng.integers(0, 4, size=(16, 32))
    h = rng.integers(0, 4, size=(16, 33))
    odo = se2d.encode(d, h)
    disp, hh = se2d.compose(odo, 9, h0=h[:, 0])
    assert (hh == h).all()
    assert (disp == np.concatenate(
        [np.zeros((16, 1, 2), np.int64),
         np.cumsum(MOVES[d] * 9, axis=1)], axis=1)).all()


def test_batch_consistent():
    bank, _ = load_bank('data/mnist.npz', 32, 'train', True)
    c = Config(heading=True)
    b = make_batch(np.random.default_rng(3), c, bank, 6)
    assert set(b) == {'observations', 'odometry', 'positions', 'headings',
                      'query_pose', 'query_time', 'target', 'episodes'}
    assert b['odometry'].min() >= 0 and b['odometry'].max() < 16
    disp, hh = se2d.compose(b['odometry'], c.step, b['headings'][:, 0])
    assert (hh == b['headings']).all()
    assert (b['positions'][:, 0:1] + disp == b['positions']).all()
    for i, e in enumerate(b['episodes']):
        image = render_map(bank, e.tile_grid, c.tile, e.tile_sym)
        world_obs = crop_windows(image, e.positions, c.obs)
        assert (se2d.face(world_obs, b['headings'][i]) == b['observations'][i]).all()
        q = b['query_pose'][i]
        t = b['query_time'][i]
        corners = e.positions[t] + se2d.unrotate(q[:, :2], b['headings'][i, t])
        assert (corners == e.queries).all()
        qh = (q[:, 2] + b['headings'][i, t]) % 4
        assert (se2d.face(crop_windows(image, e.queries, c.obs), qh)
                == b['target'][i]).all()


def test_query_times_uniform_both_tasks():
    """Each query's pixels were all observed by its step, and offsets are
    measured from the window at that step."""
    from mapping.data import coverage
    bank, _ = load_bank('data/mnist.npz', 32, 'train', True)
    for heading in (False, True):
        c = Config(heading=heading, query_times='uniform')
        b = make_batch(np.random.default_rng(11), c, bank, 8)
        t = b['query_time']
        assert t.min() >= 0 and t.max() <= c.steps and len(np.unique(t)) > 4
        for i, e in enumerate(b['episodes']):
            h, w = (np.array(e.tile_grid.shape) * c.tile)
            seen = coverage(e.positions, c.obs, h, w, first=True)
            for q, tq in zip(e.queries, e.query_times):
                win = seen[q[0]:q[0] + c.obs, q[1]:q[1] + c.obs]
                assert ((win >= 0) & (win <= tq)).all()
            if not heading:
                assert (b['query_offset'][i]
                        == e.queries - e.positions[e.query_times]).all()


def test_loader_passes_se2d_batches():
    bank, _ = load_bank('data/mnist.npz', 32, 'test', True)
    ev = eval_set(Config(heading=True), bank, 4, 1, 0)[0]
    assert ev['query_pose'].shape == (4, 8, 3) and ev['odometry'].shape == (4, 32)
