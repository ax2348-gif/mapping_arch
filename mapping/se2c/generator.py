"""Episode generator for the se2c task (CONTRACT_SE2C.md), on the GPU.

``make_batch(config, index, device)`` returns batch ``index`` of the
stream with seed ``config.seed``: a pure function of (config, index), drawn
from one ``torch.Generator`` on ``device`` seeded from both. Everything
is generated where it is used, inside the training step, with no
DataLoader.

An episode:

1. **Walk.** ``motion='decoupled'`` (the default since 2026-10-16, the
   user's decision; the arc walk proved diffusive, build_log.md): each
   step moves v ~ U[v_min, v_max] in a world direction that changes by
   delta ~ U[-3 pi/4, 3 pi/4) per step (``directions='persistent'``), and
   turns by an i.i.d. omega, uniform on [-omega_max, omega_max) or on the
   n-lattice, minus any held-out band. ``directions='lattice'`` with
   ``turn_n=4``, ``v_min = v_max = 9`` is the discrete SE(2) walk
   (``tests/test_se2c_generator.py``). ``motion='arc'``: each step
   draws a speed v ~ U[v_min, v_max] and a turn omega ~ U[-omega_max,
   omega_max) (or the lattice 2 pi k / n with ``turn_n = n``), and drives
   along the arc (``geometry.arc``). The first heading is uniform (on the
   lattice when quantised).
2. **Texture.** A fresh procedural field per episode (``texture``).
3. **Queries.** Each query gets a step t (uniform in 0..T, or T) and a
   pose uniform over the poses whose every sample point lies in the
   footprints of steps <= t (``query_coverage='causal'``, the contract) or
   of all steps (``'global'``, a documented deviation): rejection
   sampling of ``candidates`` proposals, centre uniform over the union of
   footprints, heading change uniform (lattice when quantised), tested
   against the conservative ``coverage`` raster. If none passes (mostly
   t <= 4, where the footprints so far leave no room for a view at a
   free heading), the query is the exact pose of a random observation s
   <= t turned by a random multiple of pi/2, which is covered exactly;
   ``batch['stats']['found']`` marks the queries that were not fallbacks.
4. **Rendering.** Observations and targets are the texture evaluated at
   their rotated pixel centres.

Batch (torch, on ``device``):

    observations (B, T+1, S, S) float32 in [0, 1], each facing its heading
    odometry     (B, T, 3)  float32: body (dy, dx) px and turn (rad, [-pi, pi))
    query_pose   (B, Q, 3)  float32: body (dy, dx) from the pose at the
                            query's step, and the heading change
    query_time   (B, Q)     int64 in 0..T
    target       (B, Q, S, S) float32, facing the query heading
    poses        (B, T+1, 3)  world (y, x, theta), diagnostics only
    query_world  (B, Q, 3)    world query poses, diagnostics only
    stats        dict of floats (not tensors), diagnostics only
"""
import math
from dataclasses import dataclass, field as dfield

import torch

from mapping.se2c import geometry as geo
from mapping.se2c import texture as tex
from mapping.se2c.coverage import first_seen

TWO_PI = 2 * math.pi
# centre, corners and edge midpoints of a view, in units of its half-width
RING = torch.tensor([[0, 0], [-1, -1], [-1, 1], [1, -1], [1, 1],
                     [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=torch.float32)
EVAL_SEED_TAG = 7919


@dataclass(frozen=True)
class Config:
    view: int = 15
    steps: int = 32
    queries: int = 8
    v_min: float = 4.5           # px per step (user decision 2026-10-16;
    v_max: float = 9.0           # the arc walk's first default was 3.75 .. 7.5)
    omega_max: float = math.pi
    turn_n: int = 0              # 0: continuous; n: multiples of 2 pi / n
    motion: str = 'decoupled'    # 'decoupled' (default, 2026-10-16) | 'arc'
    directions: str = 'persistent'   # decoupled: 'persistent' | 'lattice'
    holdout: tuple = ()          # () or (lo, hi): a band of |omega|, rad
    holdout_mode: str = 'exclude'    # 'exclude' | 'include' | 'only'
    query_times: str = 'uniform'     # 'uniform' | 'last'
    query_coverage: str = 'causal'   # 'causal' | 'global'
    candidates: int = 512        # stage-1 proposals per query
    survivors: int = 16          # stage-2 full tests per query
    cell: float = 0.25           # coverage raster cell, px
    texture: tex.TextureSpec = dfield(default_factory=tex.TextureSpec)
    bank: str = ''               # '' procedural; else a tile bank
                                 # (mapping.data.TEXTURES, mapping.se2c.bank)
    aa: int = 4                  # bank: supersampling per pixel side
    seed: int = 0

    def __post_init__(self):
        if self.view % 2 != 1:
            raise ValueError('view must be odd (a centre pixel)')
        if not 0 < self.v_min <= self.v_max:
            raise ValueError(f'need 0 < v_min <= v_max')
        if not 0 < self.omega_max <= math.pi:
            raise ValueError('omega_max in (0, pi]')
        if self.turn_n and self.turn_n < 2:
            raise ValueError('turn_n is 0 or >= 2')
        if self.directions == 'lattice' and not (self.motion == 'decoupled' and self.turn_n):
            raise ValueError("directions='lattice' needs motion='decoupled' and turn_n")
        for k, ok in (('motion', ('arc', 'decoupled')),
                      ('directions', ('persistent', 'lattice')),
                      ('holdout_mode', ('exclude', 'include', 'only')),
                      ('query_times', ('uniform', 'last')),
                      ('query_coverage', ('causal', 'global'))):
            if getattr(self, k) not in ok:
                raise ValueError(f'{k} {getattr(self, k)!r}')
        if self.holdout:
            lo, hi = self.holdout
            if not 0 <= lo < hi:
                raise ValueError('holdout band needs 0 <= lo < hi')

    @property
    def half(self):
        return (self.view - 1) / 2


def generator(config, index, device, eval_set=False):
    """The batch's random stream: one generator seeded from (seed, index),
    with a tag for the evaluation stream."""
    g = torch.Generator(device=device)
    tag = EVAL_SEED_TAG if eval_set else 0
    g.manual_seed((config.seed * 1_000_003 + tag) * 2_147_483_647 + index)
    return g


def _rand(g, shape, device):
    return torch.rand(shape, generator=g, device=device)


DIR_DELTA_MAX = 0.75 * math.pi   # decoupled 'persistent' directions: E cos = 0.30
                                 # (the discrete walk's 1/3); user decision 2026-10-16


def lattice(n, lo=-math.pi, hi=math.pi):
    """Multiples of 2 pi / n in [lo, hi)."""
    k = torch.arange(-n, n + 1, dtype=torch.float64) * TWO_PI / n
    return k[(k >= lo - 1e-12) & (k < hi - 1e-12)].float()


def allowed_turns(config):
    """The turn set: a lattice tensor, or None for the continuous range
    (with the band handled in ``sample_turns``)."""
    if not config.turn_n:
        return None
    w = lattice(config.turn_n, -config.omega_max, config.omega_max)
    if config.holdout and config.holdout_mode != 'include':
        lo, hi = config.holdout
        band = (w.abs() >= lo) & (w.abs() < hi)
        w = w[band] if config.holdout_mode == 'only' else w[~band]
        if w.numel() == 0:
            raise ValueError('the held-out band leaves no turns')
    return w


def sample_turns(g, config, shape, device):
    w = allowed_turns(config)
    if w is not None:
        i = torch.randint(0, w.numel(), shape, generator=g, device=device)
        return w.to(device)[i]
    m = config.omega_max
    if not config.holdout or config.holdout_mode == 'include':
        return (2 * _rand(g, shape, device) - 1) * m
    # |omega| uniform on the allowed magnitudes, sign uniform
    lo, hi = min(config.holdout[0], m), min(config.holdout[1], m)
    if config.holdout_mode == 'only':
        mag = lo + _rand(g, shape, device) * (hi - lo)
    else:
        u = _rand(g, shape, device) * (lo + (m - hi))
        mag = torch.where(u < lo, u, u - lo + hi)
    sign = torch.where(_rand(g, shape, device) < 0.5, -1.0, 1.0)
    return sign * mag


def sample_headings(g, config, shape, device):
    """Uniform headings: on the lattice when quantised."""
    if config.turn_n:
        k = torch.randint(0, config.turn_n, shape, generator=g, device=device)
        return geo.wrap(k.float() * (TWO_PI / config.turn_n))
    return (2 * _rand(g, shape, device) - 1) * math.pi


def walk(g, config, batch, device):
    """World positions ``(B, T+1, 2)`` and headings ``(B, T+1)``
    (unwrapped)."""
    c, T = config, config.steps
    v = c.v_min + (c.v_max - c.v_min) * _rand(g, (batch, T), device)
    theta0 = sample_headings(g, c, (batch,), device)
    p0 = torch.zeros(batch, 2, device=device)
    if c.motion == 'arc':
        omega = sample_turns(g, c, (batch, T), device)
        return geo.integrate(geo.arc(v, omega), omega, p0, theta0)
    # decoupled: world move directions and headings drawn independently.
    # 'lattice': directions on the n-lattice, any but the immediate reverse
    # (with n = 4 this is the discrete walk). 'persistent': the direction
    # changes by delta ~ U[-DIR_DELTA_MAX, DIR_DELTA_MAX) per step, whatever
    # n (so the walk's spread does not change along the n-curriculum).
    if c.directions == 'lattice':
        n = c.turn_n
        k = torch.empty(batch, T, dtype=torch.long, device=device)
        k[:, 0] = torch.randint(0, n, (batch,), generator=g, device=device)
        for t in range(1, T):        # any lattice direction but the reverse
            step = torch.randint(1, n, (batch,), generator=g, device=device)
            k[:, t] = (k[:, t - 1] + n // 2 + step) % n if n % 2 == 0 \
                else (k[:, t - 1] + step) % n
        phi = k.float() * (TWO_PI / n)
    else:
        phi0 = (2 * _rand(g, (batch, 1), device) - 1) * math.pi
        delta = (2 * _rand(g, (batch, T - 1), device) - 1) * DIR_DELTA_MAX
        phi = phi0 + torch.cat([torch.zeros_like(phi0), torch.cumsum(delta, 1)], 1)
    move = v[..., None] * torch.stack([-torch.cos(phi), torch.sin(phi)], -1)
    pos = torch.cat([p0[:, None], p0[:, None] + torch.cumsum(move, 1)], 1)
    # turns i.i.d. (uniform on the lattice or on [-omega_max, omega_max),
    # minus a held-out band): with omega_max = pi the headings are i.i.d.
    # uniform, as in the discrete task
    turn = sample_turns(g, c, (batch, T), device)
    theta = theta0[:, None] + torch.cat([torch.zeros_like(theta0[:, None]),
                                         torch.cumsum(turn, 1)], 1)
    return pos, theta


def sample_queries(g, config, pos, theta, device):
    """Query steps ``(B, Q)``, world positions ``(B, Q, 2)``, headings
    ``(B, Q)`` and per-batch stats."""
    c = config
    b, t1 = theta.shape
    q, C, S, half = c.queries, c.candidates, c.view, c.half
    if c.query_times == 'last':
        qt = torch.full((b, q), c.steps, dtype=torch.long, device=device)
    else:
        qt = torch.randint(0, c.steps + 1, (b, q), generator=g, device=device)
    limit = qt if c.query_coverage == 'causal' else torch.full_like(qt, c.steps)
    raster = first_seen(pos, theta, half, c.cell)
    # centre proposals: uniform over the union of footprints 0..limit.
    # draw a footprint s <= limit, a point uniform in it, and keep it with
    # probability 1 / (number of footprints <= limit containing it)
    s = torch.floor(_rand(g, (b, q, C), device) * (limit[..., None] + 1)).long()
    s = torch.minimum(s, limit[..., None])
    rows = torch.arange(b, device=device)[:, None, None]
    off = (2 * _rand(g, (b, q, C, 2), device) - 1) * half
    centre = pos[rows, s] + geo.unrotate(off, theta[rows, s])
    inside = geo.in_footprint(centre[..., None, :], pos[:, None, None],
                              theta[:, None, None], half)       # (b,q,C,t1)
    steps = torch.arange(t1, device=device)
    inside &= steps <= limit[..., None, None]
    count = inside.sum(-1).clamp(min=1).float()
    keep = _rand(g, (b, q, C), device) < 1.0 / count
    # heading: the anchor's plus a uniform change (lattice when quantised)
    anchor_theta = torch.gather(theta, 1, qt)                       # (b,q)
    heading = anchor_theta[..., None] + sample_headings(g, c, (b, q, C), device)
    limit4 = limit[..., None, None]
    # stage 1, a necessary condition on every candidate: the centre, the
    # four corners and the four edge midpoints of the view are seen
    ring = RING.to(device, non_blocking=True) * half
    rpts = centre[..., None, :] + geo.unrotate(ring, heading[..., None])
    ok1 = (raster.lookup(rpts.reshape(b, -1, 2)).view(b, q, C, 9)
           <= limit4).all(-1) & keep
    # stage 2, the full test on the first K survivors, in candidate order,
    # so the pick is the first candidate passing both: uniform over the
    # valid poses whenever one is found
    K = min(c.survivors, C)
    order = torch.arange(C, device=device).expand(b, q, C)
    rank = torch.where(ok1, order, order + C)
    cand = torch.topk(rank, K, -1, largest=False).indices         # (b,q,K)
    alive = torch.gather(ok1, 2, cand)
    cc = torch.gather(centre, 2, cand[..., None].expand(-1, -1, -1, 2))
    ch = torch.gather(heading, 2, cand)
    pts = geo.view_points(cc, ch, S)                                # (b,q,K,S,S,2)
    seen = raster.lookup(pts.reshape(b, -1, 2)).view(b, q, K, S * S)
    ok = (seen <= limit4).all(-1) & alive
    first = torch.argmax(ok.int(), -1)                              # (b,q)
    found = ok.any(-1)
    qpos = torch.gather(cc, 2, first[..., None, None].expand(-1, -1, 1, 2))[:, :, 0]
    qth = torch.gather(ch, 2, first[..., None])[..., 0]
    # fallback: the exact pose of a random observation s <= limit, turned
    # by a random multiple of pi/2 (the footprint maps onto itself, so
    # every sample point is one of observation s's): a valid query that
    # still needs the pose of s relative to the anchor
    fs = torch.minimum(torch.floor(_rand(g, (b, q), device) * (limit + 1)).long(),
                       limit)
    quarter = torch.randint(0, 4, (b, q), generator=g, device=device).float()
    fpos = torch.gather(pos, 1, fs[..., None].expand(-1, -1, 2))
    fth = torch.gather(theta, 1, fs) + quarter * (math.pi / 2)
    qpos = torch.where(found[..., None], qpos, fpos)
    qth = torch.where(found, qth, fth)
    # diagnostics as tensors (no host sync in the training step)
    stats = {'found': found, 'stage1': ok1.float().mean(),
             'raster_cells': raster.seen[0].numel()}
    return qt, qpos, qth, stats, raster


def snap(angle, n):
    """Angles near the lattice 2 pi k / n to the exact lattice value, k
    in [-n/2, n/2): wrapping in fp32 can otherwise send -pi to +pi."""
    k = torch.round(angle * (n / TWO_PI)).long()
    k = torch.remainder(k + n // 2, n) - n // 2
    return k.float() * (TWO_PI / n)


def make_batch(config, index, device, batch=64, eval_set=False,
               with_raster=False, bank=None):
    """Batch ``index`` of the stream (or of the evaluation stream).
    ``bank``: the loaded tile bank when ``config.bank`` is set
    (``mapping.se2c.bank.load``; the test split for evaluation)."""
    c = config
    g = generator(c, index, device, eval_set)
    pos, theta = walk(g, c, batch, device)
    if c.bank:
        if bank is None:
            raise ValueError(f'config.bank={c.bank!r} needs the loaded bank')
        from mapping.se2c import bank as tb
        maps, origin = tb.build_maps(g, bank, pos, c.view, device)
        params = {'maps': maps, 'origin': origin}
        draw = lambda p, th: tb.render(maps, origin, p, th, c.view, c.aa)
    else:
        params = tex.sample_params(g, batch, c.texture, device)
        draw = lambda p, th: tex.render(params, p, th, c.view, c.texture)
    qt, qpos, qth, stats, raster = sample_queries(g, c, pos, theta, device)
    obs = draw(pos, theta)
    target = draw(qpos, qth)
    body, turn = geo.odometry(pos, theta)
    anchor_pos = torch.gather(pos, 1, qt[..., None].expand(-1, -1, 2))
    anchor_th = torch.gather(theta, 1, qt)
    qpose = geo.relative_pose(qpos, qth, anchor_pos, anchor_th)
    if c.turn_n:     # exact lattice angles in [-pi, pi), free of fp32 drift
        turn = snap(turn, c.turn_n)
        qpose[..., 2] = snap(qpose[..., 2], c.turn_n)
    out = {'observations': obs,
           'odometry': torch.cat([body, turn[..., None]], -1),
           'query_pose': qpose,
           'query_time': qt, 'target': target,
           'poses': torch.cat([pos, theta[..., None]], -1),
           'query_world': torch.cat([qpos, qth[..., None]], -1),
           'texture_params': params, 'stats': stats}
    if with_raster:
        out['raster'] = raster
    return out
