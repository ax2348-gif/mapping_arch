"""Procedural texture for se2c: a band-limited isotropic random field,
drawn fresh per episode, evaluated exactly wherever it is sampled.

The field is a sum of K plane waves,

    z(u) = sqrt(2 / K) * sum_k cos(2 pi kappa_k . u + phi_k),

with directions uniform on the circle, phases uniform, and radial
frequencies |kappa_k| drawn in [f_lo, f_hi] cycles/px with density
proportional to f^(-slope). slope = 1 gives equal power per octave, the
1/f amplitude spectrum of natural images; f_hi below 0.5 keeps z below
the pixel Nyquist limit in every direction, so z is exactly determined by
its samples on any unit grid, rotated or not. z has zero mean and unit
variance over space. What a model sees is a pointwise contrast map of it,
``g(z)`` in [0, 1] (``TextureSpec.nonlinearity``):

    'linear'   0.5 + gain * z, clipped to [0, 1]       (band-limited)
    'sigmoid'  sigmoid(contrast * z)                    (level-set shapes)
    'stroke'   exp(-z^2 / (2 width^2))                  (lines on the zero set)

The contrast maps are monotone ('sigmoid') or even ('stroke'), not
band-limited: they add harmonics above f_hi; how much reaches beyond
Nyquist is measured in docs/lift/build_log.md. Rendering is exact either
way: every pixel is the function at its sample point, no interpolation.

A view is rendered by a separable form of the same sum: for a view at
pose (p, theta), the wave's phase at pixel (i, j) is ``2 pi (kappa . p +
a o_i + b o_j) + phi`` with ``(a, b) = rotate(kappa, theta)`` and ``o`` the
pixel offsets, so the S x S view is one (S x K) @ (K x S) product of
cosines and sines. ``field`` evaluates the plain sum at arbitrary points
and is the check (``tests/test_se2c_texture.py``), with
``mapping.se2c.reference`` as the NumPy reference.
"""
import math
from dataclasses import dataclass

import torch

from mapping.se2c.geometry import grid_offsets, rotate

TWO_PI = 2 * math.pi


@dataclass(frozen=True)
class TextureSpec:
    # defaults: the tuned 'stroke' texture (docs/lift/build_log.md):
    # white curved lines on black, about 3 px wide, 10-40 px apart
    components: int = 96         # K plane waves per episode
    f_lo: float = 0.025          # cycles per pixel (period 40 px)
    f_hi: float = 0.08           # cycles per pixel (period 12.5 px); < 0.5
    slope: float = 0.0           # radial density ~ f^-slope; 1: 1/f spectrum
    nonlinearity: str = 'stroke'
    contrast: float = 4.0        # sigmoid steepness in units of z's std
    gain: float = 0.18           # linear: std of the output
    width: float = 0.35          # stroke: line half-width, in units of z

    def __post_init__(self):
        if not 0 < self.f_lo < self.f_hi < 0.5:
            raise ValueError(f'need 0 < f_lo < f_hi < 0.5, got '
                             f'{self.f_lo}, {self.f_hi}')
        if self.nonlinearity not in ('linear', 'sigmoid', 'stroke'):
            raise ValueError(f'nonlinearity {self.nonlinearity!r}')


def sample_radii(u, spec):
    """Inverse CDF of the radial density ~ f^-slope on [f_lo, f_hi] at
    uniforms ``u``."""
    lo, hi, s = spec.f_lo, spec.f_hi, spec.slope
    if abs(s - 1) < 1e-9:
        return lo * (hi / lo) ** u
    e = 1 - s
    return (lo ** e + u * (hi ** e - lo ** e)) ** (1 / e)


def sample_params(gen, batch, spec, device):
    """Per-episode wave parameters from the torch generator ``gen``:
    ``freq (B, K, 2)`` cycles/px as (ky, kx), ``phase (B, K)``."""
    k = spec.components
    u = torch.rand(batch, k, generator=gen, device=device)
    direction = torch.rand(batch, k, generator=gen, device=device) * TWO_PI
    phase = torch.rand(batch, k, generator=gen, device=device) * TWO_PI
    r = sample_radii(u, spec)
    freq = torch.stack([r * torch.cos(direction), r * torch.sin(direction)],
                       -1)
    return {'freq': freq, 'phase': phase}


def field(params, points, chunk=4096):
    """z at world ``points (B, ..., 2)``, direct sum, chunked over points."""
    freq, phase = params['freq'], params['phase']
    b = points.shape[0]
    flat = points.reshape(b, -1, 2)
    out = []
    norm = math.sqrt(2.0 / freq.shape[1])
    for i in range(0, flat.shape[1], chunk):
        pts = flat[:, i:i + chunk]                                 # (B,n,2)
        arg = TWO_PI * torch.einsum('bnc,bkc->bnk', pts, freq) + phase[:, None]
        out.append(torch.cos(arg).sum(-1) * norm)
    return torch.cat(out, 1).reshape(points.shape[:-1])


def contrast(z, spec):
    """``g(z)`` in [0, 1]."""
    if spec.nonlinearity == 'linear':
        return (0.5 + spec.gain * z).clamp(0, 1)
    if spec.nonlinearity == 'sigmoid':
        return torch.sigmoid(spec.contrast * z)
    return torch.exp(-0.5 * (z / spec.width) ** 2)


def view_field(params, pos, theta, size):
    """z over views at ``pos (B, V, 2)``, ``theta (B, V)``: ``(B, V, S,
    S)``, by the separable sum (module docstring)."""
    freq, phase = params['freq'], params['phase']            # (B,K,2),(B,K)
    k = freq.shape[1]
    o = grid_offsets(size, pos.device)[:, 0, 0]              # (S,) = i - c
    ab = rotate(freq[:, None], theta[..., None])             # (B,V,K,2)
    psi = (TWO_PI * torch.einsum('bvc,bkc->bvk', pos, freq)
           + phase[:, None])                                 # (B,V,K)
    rows = psi[..., None] + TWO_PI * ab[..., 0, None] * o    # (B,V,K,S)
    cols = TWO_PI * ab[..., 1, None] * o                     # (B,V,K,S)
    z = (torch.einsum('bvki,bvkj->bvij', torch.cos(rows), torch.cos(cols))
         - torch.einsum('bvki,bvkj->bvij', torch.sin(rows), torch.sin(cols)))
    return z * math.sqrt(2.0 / k)


def render(params, pos, theta, size, spec):
    """Views ``(B, V, S, S)`` in [0, 1] at poses ``pos (B, V, 2)``, ``theta
    (B, V)``."""
    return contrast(view_field(params, pos, theta, size), spec)
