"""Exact GPU rendering of the ocean world through a pinhole camera.

World: points (y, x, z) in pixels; the surface is z = h(y, x) = amp * z_h(y, x),
with z_h a unit-variance wave field (mapping.ocean.field). The surface
carries an albedo field z_a, drawn through the se2c stroke map (white lines
on the zero set of z_a).

Camera: at (y, x, altitude) with SE(2) heading theta (repo convention:
heading 0 faces -y, positive turns right), pitched down by ``pitch``, square
image of ``size`` pixels and horizontal/vertical field of view ``fov``.
Views may see the horizon (default pitch 20 deg, fov 60: the horizon sits
about a sixth of the way down the image); rays that miss the surface are sky
(``sky``), and surface colour fades into the horizon colour with distance
(``OceanSpec.fog``).

Ray march (exact up to a tolerance): along a ray p(t) = c + t d with d_z < 0,
g(t) = p_z(t) - h(p_xy(t)) > 0 above the surface. With L the Lipschitz
constant of h, |g'(t)| <= |d_z| + L |d_xy| =: D, so stepping t += g / D can
never cross the surface (sphere tracing for height fields). The bound L is
loose (about 2.5x the largest slope actually present), so the march takes
over-relaxed steps, ``relax * g / D`` (relax 1.6): a step that lands below the
surface brackets the root, which bisection then pins to float precision. A
relaxed step could in principle hop over a crest thinner than the extra
(relax - 1) * g / D. A minimum step (``min_step``, 0.5 px along the ray) stops
grazing rays that skim the back of a crest from crawling; it can only miss a
near-tangent clip of a crest shorter than that. The float64 safe-step reference
(mapping.ocean.reference) checks first-hit depths agree. The march starts where
the ray enters the slab |z| <= sup|h|.

Anti-aliasing: each pixel's ground footprint, w = t / (F |cos i|) with F the
focal length in pixels and i the incidence angle, sets a Gaussian prefilter
(std w / sqrt(12)) on the albedo waves (exact for plane waves). The stroke map
is then made mean-preserving (``shade``): far away it fades to the pattern's
average grey instead of aliasing into speckle. ``aa='none'`` turns this off; ``supersample``
renders at a multiple of the resolution and box-averages (the check).
"""
import math
from dataclasses import dataclass, field as dfield

import torch
import torch.nn.functional as F

from mapping.ocean import field as fld


@dataclass(frozen=True)
class OceanSpec:
    height: fld.WaveSpec = dfield(default_factory=lambda: fld.WaveSpec(32, 0.008, 0.025))
    amp: float = 5.0                 # height std, px (5: some occlusion, not dominant)
    albedo: fld.WaveSpec = dfield(default_factory=lambda: fld.WaveSpec(96, 0.025, 0.08))
    width: float = 0.35              # stroke half-width in units of z_a (se2c default)
    base: float = 0.03               # albedo floor under the strokes (near black)
    shading: bool = True
    ambient: float = 0.15
    # sky and haze (choices MINE, 2026-10-16): rays that miss the surface
    # within tmax get a sky value graded by elevation; surface colour fades
    # toward the horizon colour with distance, exp(-t / fog) (fog=0: off)
    sky_horizon: float = 0.72        # sky brightness at the horizon
    sky_zenith: float = 0.92         # ... straight up
    fog: float = 120.0               # px; haze length scale (meaning depends on haze)
    haze: str = 'late_elevation'     # see haze_visibility: exp | late | smooth | exp2
                                     # | elevation | late_elevation | contrast | band
                                     # (default MINE: best alias/contrast trade-off measured)
    haze_start: float = 60.0         # px; near-clear distance (late*, smooth, contrast)
    haze_end: float = 450.0          # px; full haze (smooth, contrast)
    tmax: float = 600.0              # px; rays reaching no surface by here are sky
    gravity: float = 1.0             # dispersion for h(x, y, t): px per time unit^2
                                     # (MINE: the dominant wave's period is ~20 time units)
    light: tuple = (-0.45, 0.30, 0.84)   # (y, x, z) direction toward the light


@dataclass(frozen=True)
class CameraSpec:
    size: int = 32           # pixels per side
    fov: float = 60.0        # degrees, square image
    pitch: float = 25.0      # degrees below the horizontal (horizon near the top: row 2.6 of 32)
    altitude: float = 24.0   # px above the mean surface level (about 4.8 height sd at amp 5)

    def __post_init__(self):
        if not -90 < self.pitch < 90:
            raise ValueError('pitch in (-90, 90) degrees')

    @property
    def horizon_row(self):
        """Image row of the horizon at the image centre column (may lie outside the image)."""
        return (self.size - 1) / 2 - self.focal * math.tan(math.radians(self.pitch))

    @property
    def focal(self):
        return (self.size / 2) / math.tan(math.radians(self.fov) / 2)


def sample_world(gen, batch, spec, device, albedo_gen=None):
    """Height and albedo waves. With ``albedo_gen`` the albedo comes from its
    own generator, so the surface shape (``gen``) and the pattern can be varied
    independently; without it both come from ``gen`` (height first).
    The albedo is fixed in the world (it does not move with the waves)."""
    height = fld.sample_waves(gen, batch, spec.height, device)
    albedo = fld.sample_waves(albedo_gen if albedo_gen is not None else gen, batch, spec.albedo, device)
    world = {'height': height, 'albedo': albedo}
    world['omega'] = fld.dispersion(height, spec.gravity)
    return world


def sample_world_seeds(height_seed, albedo_seed, batch, spec, device):
    """Independent seeds for the surface shape and the albedo pattern."""
    gh = torch.Generator(device=device).manual_seed(height_seed)
    ga = torch.Generator(device=device).manual_seed(albedo_seed)
    return sample_world(gh, batch, spec, device, albedo_gen=ga)


def tight_bound(world, spec, center, radius, spacing=2.0):
    """Store in ``world['hmax'] (B,)`` an exact bound on max h over the square
    of half-side ``radius`` around ``center (B, 2)``: the grid maximum plus
    L * spacing / sqrt(2) (no point is farther than that from a grid node).
    Valid only for rays that stay over that square."""
    dev = center.device
    n = int(2 * radius / spacing) + 1
    o = torch.linspace(-radius, radius, n, device=dev)
    grid = torch.stack(torch.meshgrid(o, o, indexing='ij'), -1).reshape(1, -1, 2)
    h = heights(world, center[:, None] + grid, spec)
    _, lip = fld.bounds(world['height'])
    world['hmax'] = h.amax(1) + lip * spec.amp * spacing / math.sqrt(2)
    return world


def select(world, idx):
    """The world restricted to episodes ``idx`` (a slice or index tensor)."""
    out = {}
    for k, w in world.items():
        if isinstance(w, dict):
            out[k] = {kk: vv[idx] for kk, vv in w.items()}
        elif torch.is_tensor(w) and w.dim() > 0:
            out[k] = w[idx]
        else:
            out[k] = w
    return out


def heights(world, points, spec, grad=False):
    """h (and grad h) at ``points``; at ``world['time']`` (``(B, V)`` or a
    scalar, broadcast over the rest) when present, else static."""
    time = world.get('time')
    if time is not None:
        time = torch.as_tensor(time, device=points.device, dtype=points.dtype)
        if time.dim() == 2 and points.dim() > 3:
            time = time.view(*time.shape, *([1] * (points.dim() - 3)))
    out = fld.evaluate(world['height'], points, grad=grad, time=time,
                       omega=world.get('omega'))
    if grad:
        return out[0] * spec.amp, out[1] * spec.amp
    return out * spec.amp


def camera_rays(pos, theta, cam, size=None):
    """Origins ``(B, V, 3)`` and unit directions ``(B, V, S, S, 3)`` in
    (y, x, z). ``size`` overrides cam.size (supersampling keeps the fov)."""
    s = size or cam.size
    f = (s / 2) / math.tan(math.radians(cam.fov) / 2)
    dev = pos.device
    c, sn = torch.cos(theta), torch.sin(theta)
    zero = torch.zeros_like(c)
    fwd = torch.stack([-c, sn, zero], -1)                     # (B,V,3)
    right = torch.stack([sn, c, zero], -1)
    up = torch.tensor([0.0, 0.0, 1.0], device=dev).expand_as(fwd)
    ph = math.radians(cam.pitch)
    axis = math.cos(ph) * fwd - math.sin(ph) * up
    cup = math.sin(ph) * fwd + math.cos(ph) * up
    o = torch.arange(s, device=dev, dtype=pos.dtype) - (s - 1) / 2
    oi, oj = o[:, None], o[None, :]                            # rows, cols
    d = (f * axis[..., None, None, :] + oj[..., None] * right[..., None, None, :]
         - oi[..., None] * cup[..., None, None, :])
    d = d / d.norm(dim=-1, keepdim=True)
    origin = torch.cat([pos, torch.full_like(pos[..., :1], cam.altitude)], -1)
    return origin, d


def march(world, spec, origin, d, iters=256, bisect=20, relax=1.6, min_step=0.5,
          far_step=0.0, tol=1e-4):
    """First surface hit along rays: ``t (B, ...)``, residual g(t) and ``hit``.

    Rays that point up or level (d_z >= -1e-6) never reach the surface when
    the camera is above every crest, and rays still above the surface at
    ``spec.tmax`` are declared sky: ``hit`` False, ``t`` = tmax. The minimum
    step grows with range (``far_step * t``): a crest thinner than that far
    away can be missed, which the haze hides (the reference below uses only
    exact safe steps)."""
    b = origin.shape[0]
    sup, lip = fld.bounds(world['height'])
    if 'hmax' in world:            # tighter exact bound from a grid (tight_bound)
        sup = world['hmax'] / spec.amp
    sup = (sup * spec.amp).view(b, *([1] * (d.dim() - 2)))
    lip = (lip * spec.amp).view(b, *([1] * (d.dim() - 2)))
    o = origin.view(b, origin.shape[1], *([1] * (d.dim() - 3)), 3).expand_as(d)
    dz = d[..., 2]
    dxy = d[..., :2].norm(dim=-1)
    down = dz < -1e-6
    D = dz.abs() + lip * dxy
    tmax = spec.tmax

    def g(t):
        p = o + t[..., None] * d
        return p[..., 2] - heights(world, p[..., :2], spec)

    if (g(torch.zeros_like(dz)) <= 0).any():
        raise ValueError('a camera is below the surface')
    # start where a downward ray enters the slab z <= sup; others are sky
    t = torch.where(down, ((o[..., 2] - sup) / (-dz).clamp(min=1e-9)).clamp(min=0),
                    torch.full_like(dz, tmax))
    t = t.clamp(max=tmax)
    gt = g(t)
    lo_t, hi_t = t.clone(), t.clone()
    found = torch.zeros_like(t, dtype=torch.bool)
    done = ~down | (t >= tmax)
    for it in range(iters):
        if it % 16 == 0 and bool((found | done | (gt < tol)).all()):
            break
        step = torch.maximum(relax * gt.clamp(min=0) / D, min_step + far_step * t)
        tn = (t + step).clamp(max=tmax)
        gn = g(tn)
        live = ~found & ~done
        cross = (gn < 0) & live
        lo_t = torch.where(cross, t, lo_t)
        hi_t = torch.where(cross, tn, hi_t)
        found = found | cross
        adv = live & ~cross
        t = torch.where(adv, tn, t)
        gt = torch.where(adv, gn, gt)
        done = done | (adv & (tn >= tmax))
    for _ in range(bisect):
        mid = 0.5 * (lo_t + hi_t)
        below = g(mid) < 0
        hi_t = torch.where(found & below, mid, hi_t)
        lo_t = torch.where(found & ~below, mid, lo_t)
    t = torch.where(found, 0.5 * (lo_t + hi_t), t)
    hit = found | (down & (gt.abs() < tol) & (t < tmax))
    t = torch.where(hit, t, torch.full_like(t, tmax))
    return t, torch.where(hit, g(t), torch.zeros_like(t)), hit


def sky(spec, d):
    """Sky value for ray directions ``d (..., 3)``: horizon colour at and below
    the horizon, brightening toward the zenith."""
    e = d[..., 2].clamp(min=0) ** 0.5
    return spec.sky_horizon + (spec.sky_zenith - spec.sky_horizon) * e


def shade(world, spec, cam, points, d, t, aa=True):
    """Image values at surface ``points (B, ..., 3)`` hit along ``d`` at range t."""
    zh, gh = heights(world, points[..., :2], spec, grad=True)
    n = torch.cat([-gh, torch.ones_like(zh)[..., None]], -1)
    n = n / n.norm(dim=-1, keepdim=True)
    sigma = None
    if aa:
        cos_i = (n * d).sum(-1).abs().clamp(min=0.05)
        w = t / (cam.focal * cos_i)
        sigma = w / math.sqrt(12.0)
    w = spec.width
    sv = None
    if aa:
        # Mean-preserving prefilter of the stroke map: with v the fraction of
        # z_a's variance the footprint blur keeps, draw the stroke from the
        # renormalised field z / sqrt(v) and blend toward the stroke map's
        # spatial mean E[exp(-z^2 / 2w^2)] = w / sqrt(1 + w^2) by sqrt(v).
        # v -> 1 near the camera (exact strokes); v -> 0 far away (flat mean
        # grey, not the white a blurred-to-zero z would give).
        za, v = fld.evaluate(world['albedo'], points[..., :2], sigma=sigma, return_retained=True)
        sv = v.clamp(min=1e-6).sqrt()
        mean = w / math.sqrt(1 + w * w)
        albedo = mean + sv * (torch.exp(-0.5 * (za / (sv * w)) ** 2) - mean)
    else:
        za = fld.evaluate(world['albedo'], points[..., :2])
        albedo = torch.exp(-0.5 * (za / w) ** 2)
    albedo = spec.base + (1 - spec.base) * albedo
    lam = None
    if spec.shading:
        light = torch.tensor(spec.light, device=points.device, dtype=points.dtype)
        light = light / light.norm()
        lam = (n * light).sum(-1).clamp(min=0)
        albedo = albedo * (spec.ambient + (1 - spec.ambient) * lam)
    if spec.fog and t is not None:
        if sv is None and spec.haze == 'band':
            # the haze is defined by the native footprint even when the
            # caller supersamples (cam.focal is the native focal length)
            cos_i = (n * d).sum(-1).abs().clamp(min=0.05)
            sig = t / (cam.focal * cos_i) / math.sqrt(12.0)
            _, v = fld.evaluate(world['albedo'], points[..., :2], sigma=sig, return_retained=True)
            sv = v.clamp(min=1e-6).sqrt()
        k = haze_visibility(spec, t, d, sv)
        if spec.haze == 'contrast':
            # fade the strokes' contrast around the shaded local mean (the
            # surface stays surface-dark; no blend toward the sky colour)
            mean = w / math.sqrt(1 + w * w)
            m = spec.base + (1 - spec.base) * mean
            if spec.shading:
                m = m * (spec.ambient + (1 - spec.ambient) * lam)
            albedo = m + (albedo - m) * k
        else:
            albedo = spec.sky_horizon + (albedo - spec.sky_horizon) * k
    return albedo


def smoothstep(a, b, x):
    u = ((x - a) / (b - a)).clamp(0, 1)
    return u * u * (3 - 2 * u)


def haze_visibility(spec, t, d, sv=None):
    """Fraction of the surface colour kept at range ``t`` (1 = clear), per
    ``spec.haze``:
      exp        exp(-t / fog)                      classic, starts at t = 0
      late       exp(-max(0, t - start) / fog)      clear out to ``haze_start``
      smooth     1 - smoothstep(start, end, t)      clear near, gone by ``haze_end``
      exp2       exp(-(t / fog)^2)                  flat near, steep far
      elevation  exp(-t / (fog * (0.25 + 6 |d_z|)))   thicker along rays near the horizon line
      late_elevation  the same, clear out to ``haze_start``
      contrast   1 - smoothstep(start, end, t), applied to stroke contrast only
      band       sv: the anti-aliasing filter's retained std of the pattern,
                 so the haze fades exactly where the texture is lost
    """
    h = spec.haze
    if h == 'exp':
        return torch.exp(-t / spec.fog)
    if h == 'late':
        return torch.exp(-(t - spec.haze_start).clamp(min=0) / spec.fog)
    if h in ('smooth', 'contrast'):
        return 1 - smoothstep(spec.haze_start, spec.haze_end, t)
    if h == 'exp2':
        return torch.exp(-(t / spec.fog) ** 2)
    if h == 'elevation':
        return torch.exp(-t / (spec.fog * (0.25 + 6 * d[..., 2].abs())))
    if h == 'late_elevation':
        return torch.exp(-(t - spec.haze_start).clamp(min=0) / (spec.fog * (0.25 + 6 * d[..., 2].abs())))
    if h == 'band':
        if sv is None:
            raise ValueError("haze 'band' needs anti-aliasing on")
        return sv
    raise ValueError(f'haze {h!r}')


def render(world, spec, cam, pos, theta, aa=True, supersample=1, iters=256, time=None):
    """Views ``(B, V, S, S)``; at native resolution also depth, hit points and
    the hit mask (False = sky). ``time (B, V)``: render each view of the
    evolving surface h(x, y, t) at its own time (the albedo stays fixed)."""
    if time is not None:
        if 'hmax' in world:
            raise ValueError('tight_bound is for a static surface; drop it with time')
        world = dict(world, time=time)
    s = cam.size * supersample
    origin, d = camera_rays(pos, theta, cam, size=s)
    t, resid, hit = march(world, spec, origin, d, iters=iters)
    o = origin.view(origin.shape[0], origin.shape[1], 1, 1, 3)
    pts = o + t[..., None] * d
    img = shade(world, spec, cam, pts, d, t, aa=aa and supersample == 1)
    img = torch.where(hit, img, sky(spec, d))
    if supersample > 1:
        b, v = img.shape[:2]
        img = F.avg_pool2d(img.reshape(b * v, 1, s, s), supersample).reshape(b, v, cam.size, cam.size)
        return {'image': img}
    return {'image': img, 'depth': t, 'points': pts, 'resid': resid, 'hit': hit}
