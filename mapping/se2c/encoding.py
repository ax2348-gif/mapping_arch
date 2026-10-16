"""Shared feature encoding of continuous SE(2) relative poses.

One fixed, parameter-free spec used by every se2c model and by the blind
probe (``experiments/lift_blind_probe/``), so their inputs are the same
features. The interface is stable:

    encode_odometry(dxdy, dtheta) -> (..., ODOMETRY_FEATURES) float32
    encode_query(dxdy, dtheta)    -> (..., QUERY_FEATURES)    float32

Inputs (torch tensors or numpy arrays; numpy comes back as torch on CPU)

    dxdy    (..., 2)  body-frame translation in pixels, in the repository's
                      (dy, dx) order, y down: forward is -dy, right is +dx.
                      (The argument name is the one the interface was
                      specified with; the order is (dy, dx), as everywhere
                      in mapping/.) Odometry: the pose at t+1 seen from the
                      pose at t. Query: the query view's centre seen from
                      the pose at the query's step.
    dtheta  (...)     heading change in radians, positive = turn right
                      (clockwise on screen); any real value, the features
                      are 2*pi periodic, so wrapping is not required.

Features, in this order (the same layout for both functions):

    translation  2 + 4 * OCTAVES:  d / SCALE for each axis, then
                 sin(pi 2^k d / SCALE) for k = 0..OCTAVES-1 per axis,
                 then cos of the same. SCALE = 256 px, OCTAVES = 10, so the
                 periods run from 512 px down to 1 px: a query offset of up
                 to about +-256 px is resolved to sub-pixel precision and a
                 step of a few pixels is resolved by the fine octaves.
    angle        2 * len(HARMONICS): cos(m theta) for m in HARMONICS, then
                 sin(m theta); HARMONICS = (1, 2, 3, 4). m = 1 is
                 [cos theta, sin theta]; the higher harmonics let a model
                 read angle multiples (2 theta, ...) without learning them.

This is the hybrid4 recipe for offsets (scaled scalars plus octave
Fourier features) with a finer top octave, plus a periodic angle code.
It is an embedding of the raw relative pose, as CONTRACT.md allows ("MLP
on the scaled scalars, Fourier features"); it never combines the pose
with anything else.
"""
import math

import numpy as np
import torch

SCALE = 256.0                    # px; translation is divided by this
OCTAVES = 10                     # periods 2*SCALE / 2^k, k = 0..9: 512 .. 1 px
HARMONICS = (1, 2, 3, 4)         # angle harmonics

TRANSLATION_FEATURES = 2 + 4 * OCTAVES          # 42
ANGLE_FEATURES = 2 * len(HARMONICS)             # 8
ODOMETRY_FEATURES = TRANSLATION_FEATURES + ANGLE_FEATURES   # 50
QUERY_FEATURES = ODOMETRY_FEATURES              # 50


_HARMONICS = torch.tensor(HARMONICS, dtype=torch.float32)


def _tensor(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not torch.is_tensor(x):
        x = torch.as_tensor(x)
    return x.float()


def translation_features(dxdy):
    """``(..., 2)`` pixels -> ``(..., TRANSLATION_FEATURES)``."""
    d = _tensor(dxdy) / SCALE
    freqs = math.pi * 2.0 ** torch.arange(OCTAVES, device=d.device,
                                          dtype=d.dtype)
    ang = d[..., None] * freqs                         # (..., 2, OCTAVES)
    return torch.cat([d, ang.sin().flatten(-2), ang.cos().flatten(-2)], -1)


def angle_features(dtheta):
    """``(...)`` radians -> ``(..., ANGLE_FEATURES)``."""
    th = _tensor(dtheta)
    m = _HARMONICS.to(th.device, th.dtype, non_blocking=True)
    ang = th[..., None] * m
    return torch.cat([ang.cos(), ang.sin()], -1)


def _encode(dxdy, dtheta):
    dxdy, dtheta = _tensor(dxdy), _tensor(dtheta)
    if dxdy.shape[-1] != 2 or dxdy.shape[:-1] != dtheta.shape:
        raise ValueError(f'dxdy {tuple(dxdy.shape)} must be (..., 2) and '
                         f'dtheta {tuple(dtheta.shape)} its leading shape')
    return torch.cat([translation_features(dxdy), angle_features(dtheta)], -1)


def encode_odometry(dxdy, dtheta):
    """Odometry step t -> t+1 in the body frame of t: ``dxdy (..., 2)``
    (dy, dx) px, ``dtheta (...)`` rad -> ``(..., ODOMETRY_FEATURES)``."""
    return _encode(dxdy, dtheta)


def encode_query(dxdy, dtheta):
    """Query pose relative to the pose at its step: ``dxdy (..., 2)``
    (dy, dx) px, ``dtheta (...)`` rad -> ``(..., QUERY_FEATURES)``."""
    return _encode(dxdy, dtheta)
