"""Walk paths (``Config.path``, ``task.generator.path``): the walk as
always ('walk'), and the walk conditioned on no revisit ('saw') or on no
late view contact ('no_revisit_view'), sampled by exact rejection
(``mapping.data.constrained_walk``).

- 'walk' is the default and changes nothing (``SE2D_GOLDEN`` in
  test_generator_options.py, and the batch equals one made without the
  field).
- Batches are deterministic in the seed.
- The constraints hold exactly, checked here on pixel footprints, not
  through the functions that enforce them.
- The odometry integrates back to the poses.
- Every query is still answerable (its pixels observed by its step).
- The sampler is the exact conditional: on short walks, where every
  non-reversing walk can be listed, the sampled path frequencies match
  the uniform distribution over the accepted walks (chi-square)."""
import itertools

import numpy as np
import pytest
from scipy import stats

from mapping import se2d
from mapping.data import (MOVES, PATHS, Config, accept, constrained_walk,
                          load_bank, make_batch, view_gap)
from tests.test_generator_options import digest

BANK = 'data/mnist.npz'


@pytest.fixture(scope='module')
def bank():
    return load_bank(BANK, 32, 'train', True)[0]


def batch(bank, path, seed=0, index=0, qt='uniform', n=64):
    c = Config(seed=seed, heading=True, query_times=qt, path=path)
    return make_batch(np.random.default_rng([seed, index]), c, bank, n)


def test_walk_is_the_default(bank):
    assert Config().path == 'walk'
    for qt in ('last', 'uniform'):
        c0 = Config(heading=True, query_times=qt)
        c1 = Config(heading=True, query_times=qt, path='walk')
        assert digest(make_batch(np.random.default_rng([0, 0]), c0, bank, 16)) \
            == digest(make_batch(np.random.default_rng([0, 0]), c1, bank, 16))
    with pytest.raises(ValueError):
        Config(path='spiral')


@pytest.mark.parametrize('path', PATHS[1:])
def test_deterministic(bank, path):
    a, b = batch(bank, path), batch(bank, path)
    assert digest(a) == digest(b)
    assert digest(a) != digest(batch(bank, path, index=1))
    assert digest(a) != digest(batch(bank, 'walk'))


def footprints(positions, obs):
    """``(T + 1, H, W)`` bool pixel masks of each window, and the same
    grown by one pixel along the axes (edge contact)."""
    p = positions - positions.min(0) + 1
    h, w = p.max(0) + obs + 1
    m = np.zeros((len(p), h, w), bool)
    for t, (y, x) in enumerate(p):
        m[t, y:y + obs, x:x + obs] = True
    g = m.copy()
    g[:, 1:] |= m[:, :-1]
    g[:, :-1] |= m[:, 1:]
    g[:, :, 1:] |= m[:, :, :-1]
    g[:, :, :-1] |= m[:, :, 1:]
    return m, g


@pytest.mark.parametrize('path', PATHS[1:])
def test_constraint_holds_exactly(bank, path):
    obs, k = 9, view_gap(9, 9)
    for index in range(4):
        b = batch(bank, path, index=index)
        for pos in b['positions']:
            cells = {tuple(p) for p in pos}
            assert len(cells) == len(pos)            # no revisit
            if path == 'no_revisit_view':
                m, g = footprints(pos, obs)
                for t in range(len(pos)):
                    for s in range(t - k):           # s < t - k
                        assert not (g[t] & m[s]).any(), (t, s)


def test_walk_has_revisits(bank):
    """The control: the plain walk does revisit, and touches late."""
    pos = batch(bank, 'walk')['positions']
    assert not accept(pos[:, :, :].astype(np.int64), 'saw', 9, 9).all()


@pytest.mark.parametrize('path', PATHS)
def test_odometry_integrates_to_poses(bank, path):
    b = batch(bank, path)
    disp, head = se2d.compose(b['odometry'], 9, b['headings'][:, 0])
    assert (disp == b['positions'] - b['positions'][:, :1]).all()
    assert (head == b['headings']).all()


@pytest.mark.parametrize('path', PATHS)
def test_queries_answerable(bank, path):
    b = batch(bank, path)
    for e in b['episodes']:
        for q, t in zip(e.queries, e.query_times):
            seen = np.zeros((q[0] + 9 + 1, q[1] + 9 + 1), bool)
            for y, x in e.positions[:t + 1]:
                seen[y:y + 9, x:x + 9] = True
            assert seen[q[0]:q[0] + 9, q[1]:q[1] + 9].all()


def all_walks(steps):
    """Every non-reversing walk of ``steps`` moves, as direction tuples."""
    out = []
    for first in range(4):
        for rest in itertools.product(range(3), repeat=steps - 1):
            d = [first]
            for r in rest:
                d.append((d[-1] + 2 + 1 + r) % 4)
            out.append(tuple(d))
    return out


@pytest.mark.parametrize('path,steps', [('saw', 7), ('no_revisit_view', 7)])
def test_exact_conditional(path, steps):
    """random_walk is uniform over non-reversing walks, so the walk given
    the constraint is uniform over the accepted ones: compare the sampled
    frequencies with that, by chi-square."""
    walks = all_walks(steps)
    d = np.array(walks)
    disp = np.zeros((len(d), steps + 1, 2), np.int64)
    np.cumsum(MOVES[d] * 9, axis=1, out=disp[:, 1:])
    ok = accept(disp, path, 9, 9)
    allowed = {tuple(w): i for i, w in enumerate(d[ok].tolist())}
    assert 0 < len(allowed) < len(walks)
    rng = np.random.default_rng(0)
    n = 200 * len(allowed)
    sample, _ = constrained_walk(rng, n, steps, 9, 9, path)
    counts = np.zeros(len(allowed))
    for w in sample.tolist():
        counts[allowed[tuple(w)]] += 1               # KeyError: not allowed
    p = stats.chisquare(counts).pvalue
    assert p > 1e-3, p


def test_config_field_reaches_generator_and_record():
    """``task.generator.path`` sets ``Config.path``; the args record and
    final.json carry it only when it is not 'walk' (so a walk run records
    exactly as before the field)."""
    from omegaconf import OmegaConf

    import run
    from tests.test_config import compose
    for path in PATHS:
        e = run.build(OmegaConf.create(OmegaConf.to_container(compose(
            'model=se2d/hybrid4', f'task.generator.path={path}'))))
        config, _ = e.task.generator.data(e.task, 0)
        assert config.path == path
        a = vars(e.legacy_args())
        assert a.get('path', 'walk') == path and ('path' in a) == (path != 'walk')
