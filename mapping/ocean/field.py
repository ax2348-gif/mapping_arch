"""Band-limited random fields for the ocean world: the height field h(y, x)
and the surface albedo, both sums of plane waves as in mapping.se2c.texture.

    z(u) = sqrt(2 / K) * sum_k cos(2 pi kappa_k . u + phi_k)

has zero mean and unit variance over space. Both fields are exact
everywhere, with analytic gradients, and two exact bounds hold for every
episode (used by the ray march):

    |z(u)|      <= sqrt(2 / K) * K                    = sqrt(2 K)
    |grad z(u)| <= sqrt(2 / K) * 2 pi * sum_k |kappa_k|  (Lipschitz constant)

Coordinates are the repo's (y, x) in pixels; heights are in pixels too.
"""
import math
from dataclasses import dataclass

import torch

TWO_PI = 2 * math.pi


@dataclass(frozen=True)
class WaveSpec:
    components: int          # K plane waves
    f_lo: float              # cycles per pixel
    f_hi: float
    slope: float = 0.0       # radial density ~ f^-slope
    anisotropy: float = 0.0  # 0: directions uniform; 1: all along one random axis

    def __post_init__(self):
        if not 0 < self.f_lo < self.f_hi < 0.5:
            raise ValueError(f'need 0 < f_lo < f_hi < 0.5: {self.f_lo}, {self.f_hi}')
        if not 0 <= self.anisotropy <= 1:
            raise ValueError('anisotropy in [0, 1]')


def sample_waves(gen, batch, spec, device):
    """``freq (B, K, 2)`` cycles/px as (ky, kx) and ``phase (B, K)``."""
    k = spec.components
    u = torch.rand(batch, k, generator=gen, device=device)
    lo, hi, s = spec.f_lo, spec.f_hi, spec.slope
    if abs(s - 1) < 1e-9:
        r = lo * (hi / lo) ** u
    else:
        e = 1 - s
        r = (lo ** e + u * (hi ** e - lo ** e)) ** (1 / e)
    axis = torch.rand(batch, 1, generator=gen, device=device) * TWO_PI
    spread = torch.rand(batch, k, generator=gen, device=device) * TWO_PI
    # anisotropy a: directions uniform within +-(1 - a) pi around the axis
    # (a = 0 uniform on the circle); the sign of a wave vector is irrelevant
    direction = axis + (spread - math.pi) * (1 - spec.anisotropy)
    phase = torch.rand(batch, k, generator=gen, device=device) * TWO_PI
    freq = torch.stack([r * torch.cos(direction), r * torch.sin(direction)], -1)
    return {'freq': freq, 'phase': phase}


def bounds(waves):
    """Per-episode exact bounds ``(sup |z|, Lipschitz constant of z)``,
    each ``(B,)``."""
    k = waves['freq'].shape[1]
    norm = math.sqrt(2.0 / k)
    sup = torch.full_like(waves['phase'][:, 0], norm * k)
    lip = norm * TWO_PI * waves['freq'].norm(dim=-1).sum(-1)
    return sup, lip


def evaluate(waves, points, grad=False, sigma=None, chunk=1 << 15, return_retained=False,
             time=None, omega=None):
    """z (and grad z) at ``points (B, ..., 2)``.

    ``sigma (B, ...)``, optional: a Gaussian prefilter of std ``sigma`` px per
    point (anti-aliasing by footprint); each wave is multiplied by
    exp(-2 pi^2 |kappa|^2 sigma^2), the exact Gaussian blur of a plane wave.
    ``return_retained``: also the fraction of z's variance the prefilter keeps
    at each point, mean of the squared wave gains (1 without a prefilter).
    ``time (B, ...)`` with ``omega (B, K)``: the field at time t, each wave's
    phase advanced by -omega_k t (``time`` None or 0: the static field, exactly).
    The sup and Lipschitz bounds hold at every t.
    """
    freq, phase = waves['freq'], waves['phase']
    b, k = phase.shape
    norm = math.sqrt(2.0 / k)
    flat = points.reshape(b, -1, 2)
    sig = None if sigma is None else sigma.reshape(b, -1)
    tim = None if time is None else torch.as_tensor(time, device=flat.device, dtype=flat.dtype).expand(points.shape[:-1]).reshape(b, -1)
    zs, gs, retained = [], [], []
    for i in range(0, flat.shape[1], chunk):
        pts = flat[:, i:i + chunk]
        arg = TWO_PI * torch.einsum('bnc,bkc->bnk', pts, freq) + phase[:, None]
        if tim is not None:
            arg = arg - tim[:, i:i + chunk, None] * omega[:, None]
        amp = None
        if sig is not None:
            f2 = (freq ** 2).sum(-1)                                   # (B,K)
            amp = torch.exp(-2 * math.pi ** 2 * f2[:, None] * sig[:, i:i + chunk, None] ** 2)
        c = torch.cos(arg)
        zs.append(((c * amp) if amp is not None else c).sum(-1) * norm)
        if amp is not None:
            retained.append((amp ** 2).mean(-1))       # fraction of z's variance kept
        if grad:
            sn = torch.sin(arg)
            if amp is not None:
                sn = sn * amp
            gs.append(-norm * TWO_PI * torch.einsum('bnk,bkc->bnc', sn, freq))
    z = torch.cat(zs, 1).reshape(points.shape[:-1])
    out = [z]
    if grad:
        out.append(torch.cat(gs, 1).reshape(points.shape))
    if return_retained:
        out.append(torch.cat(retained, 1).reshape(points.shape[:-1]) if retained
                   else torch.ones_like(z))
    return out[0] if len(out) == 1 else tuple(out)


def dispersion(waves, gravity):
    """Deep-water dispersion omega_k = sqrt(g * 2 pi |kappa_k|), ``(B, K)``;
    ``gravity`` in px per time unit squared."""
    return torch.sqrt(gravity * TWO_PI * waves['freq'].norm(dim=-1))


def regular(r, f_center=0.015, components=32, broad=0.6, narrow=0.08, max_aniso=0.92):
    """The REGULARITY dial for a height spectrum, r in [0, 1]:
    r = 0: irregular sea, broad band (relative half-width ``broad``),
           directions uniform on the circle, random phases;
    r = 1: regular swell, narrow band (``narrow``), directions within
           +-(1 - max_aniso) pi of one axis (about +-14 deg).
    The band is centred on ``f_center`` (cycles/px) in log frequency. Phases
    stay random at every r (a narrow band already makes the crests regular).
    """
    if not 0 <= r <= 1:
        raise ValueError('regularity in [0, 1]')
    half = broad + (narrow - broad) * r
    lo, hi = f_center * (1 - half), f_center * (1 + half)
    return WaveSpec(components, lo, hi, slope=1.0, anisotropy=max_aniso * r)
