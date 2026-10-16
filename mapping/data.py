"""Episode generation for the Z^2 mapping study.

A map is a grid of MNIST tiles sampled fresh per episode. A trajectory is a
random walk of fixed step with no immediate reversals, observed through a
square window. Queries are window positions, at any pixel offset, whose every
pixel was observed at some earlier time, so nothing is ever masked.

Everything an episode holds is small integers; pixels are rendered on demand.
Only the tiles the trajectory can touch are sampled, which is equivalent in
distribution to sampling a whole map because tiles are independent.

Coordinates are pixels in the episode's local map, y then x. A position is
the top-left corner of its window.
"""
from dataclasses import dataclass, asdict

import numpy as np

# direction k moves by MOVES[k]; the reverse of k is (k + 2) % 4
MOVES = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int64)

# odometry vocabularies, and the reserved "unknown" symbol odometry dropout
# writes: the next index after the real ones (4 on Z^2, 16 on SE(2))
VOCAB = {False: 4, True: 16}           # keyed by Config.heading
UNKNOWN = dict(VOCAB)


@dataclass(frozen=True)
class Config:
    tile: int = 28          # MNIST tile side, pixels
    obs: int = 9            # observation window side, pixels
    step: int = 9           # walk step, pixels
    steps: int = 32         # transitions T; T + 1 observations
    queries: int = 8        # Q per episode
    start_window: int = 28  # one tile: uniform phase against the tile grid
    cutoff: int = 32        # brightness below this is zeroed in the bank
    dihedral: bool = True   # each tile gets a random one of its 8 symmetries
    heading: bool = False   # SE(2) task: a heading per step, see mapping.se2
    query_times: str = 'last'   # 'last': every query is asked at the final
                                # step; 'uniform': each query at a step drawn
                                # uniformly from 0..T and answerable from the
                                # observations up to that step
    odometry_dropout: float = 0.0   # probability that each step's odometry
                                    # symbol is replaced by the reserved
                                    # "unknown" symbol (see make_batch)
    seed: int = 0

    def __post_init__(self):
        if self.start_window < self.tile:
            raise ValueError('start_window below one tile leaves the phase '
                             'non-uniform')
        if self.query_times not in ('last', 'uniform'):
            raise ValueError(f'query_times {self.query_times!r}')
        if not 0.0 <= self.odometry_dropout <= 1.0:
            raise ValueError(f'odometry_dropout {self.odometry_dropout}')
        if not 1 <= self.step:
            raise ValueError(f'step {self.step}')



@dataclass
class Episode:
    """One episode, pixels not included.

    tile_grid  (h, w) int32 indices into the bank; its (0, 0) tile covers
               local pixels [0, tile) x [0, tile)
    tile_sym   (h, w) int8 dihedral symmetry applied to each tile, 0..7:
               ``k`` quarter turns plus 4 if mirrored first; all zero when
               augmentation is off
    positions  (T + 1, 2) int32 window top-left corners, local pixels
    queries    (Q, 2) int32 query top-left corners, local pixels
    query_times (Q,) int64 the step each query is asked at; every query's
               pixels were observed by that step. All T when the config's
               ``query_times`` is 'last'.
    """
    tile_grid: np.ndarray
    tile_sym: np.ndarray
    positions: np.ndarray
    queries: np.ndarray
    query_times: np.ndarray


def apply_dihedral(tiles, sym):
    """Each tile under its own symmetry; ``tiles`` is ``(..., t, t)`` and
    ``sym`` is ``(...)`` in 0..7: ``k`` quarter turns, plus 4 to mirror
    left-right first.

    Transforms the tiles of each symmetry together, so the work is eight
    small gathers rather than eight images of every tile.
    """
    out = np.ascontiguousarray(tiles).copy()
    for s in range(1, 8):
        mask = sym == s
        if not mask.any():
            continue
        picked = tiles[mask]
        if s >= 4:
            picked = picked[..., :, ::-1]
        out[mask] = np.rot90(picked, s % 4, axes=(-2, -1))
    return out


def load_bank(path, cutoff, split='train', dihedral=False):
    """MNIST tiles with near-black residue zeroed, ``(N, 28, 28)`` uint8.

    With ``dihedral`` the bank is ``(8, N, 28, 28)``: every tile under each
    of its eight symmetries, precomputed once so that rendering stays a
    single gather. Eight copies of MNIST is 360 MiB, and forked loader
    workers share it copy-on-write.
    """
    with np.load(path) as f:
        images = f[f'{split}_images']
        labels = f[f'{split}_labels']
    images = np.where(images < cutoff, 0, images).astype(np.uint8)
    if dihedral:
        images = np.stack([apply_dihedral(images, np.full(len(images), s,
                                                          np.int8))
                           for s in range(8)])
    return images, labels


def bank_size(bank):
    return bank.shape[-3]


# ---------------------------------------------------------------- the walk
def random_walk(rng, batch, steps, step):
    """Directions ``(b, T)`` and displacements ``(b, T + 1, 2)`` from the
    origin, no immediate reversals, for a whole batch at once."""
    directions = np.empty((batch, steps), dtype=np.int64)
    directions[:, 0] = rng.integers(0, 4, size=batch)
    for t in range(1, steps):
        # the three choices other than the reverse of the last move
        reverse = (directions[:, t - 1] + 2) % 4
        directions[:, t] = (reverse + 1 + rng.integers(0, 3, size=batch)) % 4
    moves = MOVES[directions] * step                       # (b, T, 2)
    displacement = np.zeros((batch, steps + 1, 2), dtype=np.int64)
    np.cumsum(moves, axis=1, out=displacement[:, 1:])
    return directions, displacement


# ------------------------------------------------------------- one episode
def coverage(positions, obs, height, width, first=False):
    """Last (or, with ``first``, first) time each local pixel was observed,
    ``(height, width)``, -1 never. A query asked at step t needs the
    first-seen map: a pixel re-observed later is still known at t."""
    seen = np.full((height, width), -1, dtype=np.int32)
    for t, (y, x) in enumerate(positions):
        block = seen[y:y + obs, x:x + obs]
        if first:
            block[block < 0] = t
        else:
            block[...] = t
    return seen


def fully_covered_corners(seen, obs):
    """Boolean ``(h - obs + 1, w - obs + 1)``: window at that corner has
    every pixel in ``seen``.

    An integral image makes the window sum a four-corner difference.
    """
    integral = np.zeros((seen.shape[0] + 1, seen.shape[1] + 1), np.int32)
    np.cumsum(np.cumsum(seen, axis=0), axis=1, out=integral[1:, 1:])
    total = (integral[obs:, obs:] - integral[:-obs, obs:]
             - integral[obs:, :-obs] + integral[:-obs, :-obs])
    return total == obs * obs


def sample_queries(rng, last_seen, obs, count, at_time=None):
    """``count`` window corners whose every pixel was seen by ``at_time``,
    uniform over the valid corners. With ``at_time`` pass the first-seen
    map from ``coverage(..., first=True)``; the window at that step is
    then always a valid corner, so there is at least one."""
    seen = last_seen >= 0
    if at_time is not None:
        seen &= last_seen <= at_time
    valid = fully_covered_corners(seen, obs)
    corners = np.flatnonzero(valid)
    picks = rng.choice(corners, size=count, replace=True)
    return np.stack(np.unravel_index(picks, valid.shape), axis=-1).astype(
        np.int32), corners.size


def make_episode(rng, config, bank_size, displacement):
    """Build one episode around a pre-sampled walk.

    Only the tiles the footprints touch are sampled. The start phase is
    uniform over a window one tile wide, so the walk's lattice sits at a
    uniformly random offset against the tile grid.
    """
    c = config
    start = rng.integers(0, c.start_window, size=2)
    positions = displacement + start                         # (T + 1, 2)

    low = positions.min(axis=0)
    high = positions.max(axis=0) + c.obs                     # exclusive
    tile_low = np.floor_divide(low, c.tile)
    tile_high = -np.floor_divide(-high, c.tile)              # ceil
    shape = tile_high - tile_low
    tile_grid = rng.integers(0, bank_size, size=tuple(shape)).astype(np.int32)
    if c.dihedral:
        tile_sym = rng.integers(0, 8, size=tuple(shape)).astype(np.int8)
    else:
        tile_sym = np.zeros(tuple(shape), dtype=np.int8)

    positions = (positions - tile_low * c.tile).astype(np.int32)
    height, width = shape * c.tile
    last_seen = coverage(positions, c.obs, height, width)
    if c.query_times == 'last':
        queries, _ = sample_queries(rng, last_seen, c.obs, c.queries)
        times = np.full(c.queries, c.steps, dtype=np.int64)
    else:
        times = rng.integers(0, c.steps + 1, size=c.queries)
        first_seen = coverage(positions, c.obs, height, width, first=True)
        queries = np.concatenate(
            [sample_queries(rng, first_seen, c.obs, 1, at_time=t)[0]
             for t in times])
    return Episode(tile_grid, tile_sym, positions, queries, times), last_seen


def render_map(bank, tile_grid, tile, tile_sym=None):
    """``(h * tile, w * tile)`` uint8 from the tile indices."""
    h, w = tile_grid.shape
    if bank.ndim == 4:                                       # (8, N, t, t)
        tiles = bank[tile_sym, tile_grid]                    # (h, w, t, t)
    else:
        tiles = bank[tile_grid]
        if tile_sym is not None and tile_sym.any():
            tiles = apply_dihedral(tiles, tile_sym)
    return np.ascontiguousarray(
        tiles.transpose(0, 2, 1, 3).reshape(h * tile, w * tile))


def crop_windows(image, corners, obs):
    """``(n, obs, obs)`` windows of ``image`` at the given corners."""
    ys = corners[:, 0, None] + np.arange(obs)                # (n, obs)
    xs = corners[:, 1, None] + np.arange(obs)
    return image[ys[:, :, None], xs[:, None, :]]


# ------------------------------------------------------------------ batches
def make_batch(rng, config, bank, batch):
    """One training batch as numpy arrays.

    observations (b, T + 1, S, S) uint8
    directions   (b, T)           int64, index into MOVES
    positions    (b, T + 1, 2)    int32, absolute in each local map
    query_offset (b, Q, 2)        int32, relative to the window at the
                                  query's step (the last window unless
                                  ``config.query_times`` varies them)
    query_time   (b, Q)           int64, the step each query is asked at
    target       (b, Q, S, S)     uint8, every pixel observed by that step
    episodes     list of Episode, the stored form

    With ``config.heading`` (the SE(2) task, ``mapping.se2``) the batch is
    instead

    observations (b, T + 1, S, S) uint8, each rotated to face its heading
    odometry     (b, T)           int64 in 0..15, egocentric move and turn
    positions    (b, T + 1, 2)    int32, world, diagnostics only
    headings     (b, T + 1)       int64 in 0..3, world, diagnostics only
    query_pose   (b, Q, 3)        int32, body-frame (dy, dx) from the pose
                                  at the query's step and the heading change
    query_time   (b, Q)           int64, the step each query is asked at
    target       (b, Q, S, S)     uint8, rotated to face the query heading
    episodes     list of Episode

    The two layouts share no key names beyond the common ones, so a model
    written for one cannot silently consume the other. The heading draws
    come after everything else, so with ``heading`` off the batch is
    byte-identical to what the Z^2 generator always produced.

    Odometry dropout (``config.odometry_dropout`` p > 0): each step's
    symbol in ``directions`` / ``odometry`` is independently replaced, with
    probability p, by ``UNKNOWN[config.heading]`` (4 on Z^2, 16 on SE(2)),
    and the batch gains the undropped symbols as ``directions_true`` /
    ``odometry_true``, for diagnostics only (in the task modules'
    ``FORBIDDEN_KEYS``). The mask is the last draw from ``rng``, and is not
    drawn at all when p is 0, so with p = 0 the batch is exactly what it was
    before the option existed, and with p > 0 every other field is
    identical to the p = 0 batch of the same seed.
    """
    c = config
    directions, displacement = random_walk(rng, batch, c.steps, c.step)
    observations = np.empty((batch, c.steps + 1, c.obs, c.obs), np.uint8)
    targets = np.empty((batch, c.queries, c.obs, c.obs), np.uint8)
    positions = np.empty((batch, c.steps + 1, 2), np.int32)
    offsets = np.empty((batch, c.queries, 2), np.int32)
    times = np.empty((batch, c.queries), np.int64)
    episodes = []
    for i in range(batch):
        episode, _ = make_episode(rng, c, bank_size(bank), displacement[i])
        image = render_map(bank, episode.tile_grid, c.tile, episode.tile_sym)
        observations[i] = crop_windows(image, episode.positions, c.obs)
        targets[i] = crop_windows(image, episode.queries, c.obs)
        positions[i] = episode.positions
        offsets[i] = episode.queries - episode.positions[episode.query_times]
        times[i] = episode.query_times
        episodes.append(episode)
    if not c.heading:
        out = dict(observations=observations, directions=directions,
                   positions=positions, query_offset=offsets,
                   query_time=times, target=targets, episodes=episodes)
        return drop_odometry(rng, c, out, 'directions')
    from mapping import se2
    headings = rng.integers(0, 4, size=(batch, c.steps + 1))
    query_headings = rng.integers(0, 4, size=(batch, c.queries))
    queries = np.stack([e.queries for e in episodes])
    rows = np.arange(batch)[:, None]
    pose = se2.query_pose(queries, query_headings, positions[rows, times],
                          headings[rows, times])
    out = dict(observations=se2.face(observations, headings),
               odometry=se2.encode(directions, headings),
               positions=positions, headings=headings,
               query_pose=pose.astype(np.int32), query_time=times,
               target=se2.face(targets, query_headings), episodes=episodes)
    return drop_odometry(rng, c, out, 'odometry')


def drop_odometry(rng, config, batch, key):
    """Odometry dropout in place on ``batch[key]``, keeping the true
    symbols under ``key + '_true'``. Must be the last draw from ``rng``;
    draws nothing when the rate is 0, which leaves the batch untouched."""
    p = config.odometry_dropout
    if p == 0:
        return batch
    true = batch[key]
    dropped = rng.random(true.shape) < p
    batch[key] = np.where(dropped, UNKNOWN[config.heading], true)
    batch[key + '_true'] = true
    return batch


class EpisodeStream:
    """Infinite stream of batches, deterministic in the seed.

    Worker ``k`` of ``n`` takes every n-th seed, so a multi-worker loader
    produces the same multiset of batches as a single one.
    """

    def __init__(self, config, bank, batch):
        self.config, self.bank, self.batch = config, bank, batch

    def batches(self, worker=0, workers=1, start=0):
        """Batches ``start + worker, start + worker + workers, ...``; a
        resumed run passes the number of batches already consumed."""
        index = start + worker
        while True:
            rng = np.random.default_rng([self.config.seed, index])
            yield make_batch(rng, self.config, self.bank, self.batch)
            index += workers


def config_dict(config):
    return asdict(config)
