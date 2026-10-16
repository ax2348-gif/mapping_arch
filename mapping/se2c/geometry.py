"""Continuous SE(2) conventions for the se2c task, in torch (any device,
fp32).

Everything is on ``(y, x)`` pixel vectors with y down, as in
``mapping.se2``, and agrees with it at quarter turns
(``tests/test_se2c_geometry.py``):

- A pose is a world position ``p = (y, x)``, the centre of the view, and
  a heading ``theta`` in radians. Heading 0 faces up (``-y``), pi/2 faces
  right (``+x``): the facing direction is ``(-cos theta, sin theta)``.
  Positive turns are turns to the right, clockwise on screen. At
  ``theta = h pi/2`` this is ``mapping.se2``'s heading ``h``.
- ``rotate(v, theta)`` is a world vector seen from heading ``theta``
  (body frame: forward ``-y``, right ``+x``); ``unrotate`` is its inverse.
  At quarter turns they equal ``mapping.se2.rotate`` / ``unrotate``.
- A view of side S (odd) at pose ``(p, theta)`` samples the world at
  ``p + unrotate((i - c, j - c), theta)`` for pixel ``(i, j)``, ``c = (S -
  1) / 2``: the view faces its heading, image-up is forward. At quarter
  turns and integer centres this is ``mapping.se2.face`` of the world
  crop.
- The footprint of a view is the convex hull of its sample points: a
  square of half-width ``c`` centred at ``p``, rotated by ``theta``.
- The odometry from pose t to t+1 is ``(rotate(p_{t+1} - p_t, theta_t),
  wrap(theta_{t+1} - theta_t))``: body-frame (dy, dx) and the turn, wrapped
  to [-pi, pi).
- A step of the isotropic walk drives forward at speed v while turning by
  omega, for unit time, along a circular arc (the SE(2) exponential of
  the twist (v, 0, omega)): ``arc(v, omega)`` is its body-frame
  displacement ``v * (-sin(w)/w, (1 - cos(w))/w)``.

``integrate`` composes poses in the open; the generator uses it, and a
model must never (CONTRACT_SE2C.md).
"""
import math

import torch

TWO_PI = 2 * math.pi


def wrap(theta):
    """Angles to [-pi, pi)."""
    return torch.remainder(theta + math.pi, TWO_PI) - math.pi


def rotate(v, theta):
    """World vectors ``v (..., 2)`` as seen from heading ``theta (...)``."""
    c, s = torch.cos(theta), torch.sin(theta)
    y, x = v[..., 0], v[..., 1]
    return torch.stack([c * y - s * x, s * y + c * x], -1)


def unrotate(b, theta):
    """Body-frame vectors back to the world frame; inverse of ``rotate``."""
    c, s = torch.cos(theta), torch.sin(theta)
    y, x = b[..., 0], b[..., 1]
    return torch.stack([c * y + s * x, -s * y + c * x], -1)


def arc(v, omega):
    """Body-frame displacement ``(..., 2)`` of driving at speed ``v`` while
    turning by ``omega`` for unit time. Exact; the small-angle limit is
    handled by a series."""
    small = omega.abs() < 1e-4
    w = torch.where(small, torch.ones_like(omega), omega)
    sinc = torch.where(small, 1 - omega ** 2 / 6, torch.sin(w) / w)
    cosc = torch.where(small, omega / 2 - omega ** 3 / 24,
                       (1 - torch.cos(w)) / w)
    return torch.stack([-v * sinc, v * cosc], -1)


def integrate(body, turn, p0, theta0):
    """World poses from body-frame steps: ``body (..., T, 2)`` taken from
    pose t, then ``turn (..., T)``. Returns positions ``(..., T + 1, 2)``
    and headings ``(..., T + 1)`` (unwrapped). Data generation and tests
    only."""
    theta = torch.cat([theta0[..., None],
                       theta0[..., None] + torch.cumsum(turn, -1)], -1)
    world = unrotate(body, theta[..., :-1])
    pos = torch.cat([p0[..., None, :], p0[..., None, :]
                     + torch.cumsum(world, -2)], -2)
    return pos, theta


def odometry(pos, theta):
    """``(body (..., T, 2), turn (..., T))`` between consecutive poses."""
    d = pos[..., 1:, :] - pos[..., :-1, :]
    return rotate(d, theta[..., :-1]), wrap(theta[..., 1:] - theta[..., :-1])


def relative_pose(pos, theta, anchor_pos, anchor_theta):
    """``pos (..., 2)``, ``theta (...)`` seen from the anchor pose: body
    (dy, dx) and the wrapped heading change, as ``(..., 3)``."""
    return torch.cat([rotate(pos - anchor_pos, anchor_theta),
                      wrap(theta - anchor_theta)[..., None]], -1)


def grid_offsets(size, device=None):
    """``(S, S, 2)`` body-frame offsets of a view's pixel centres from its
    centre: ``(i - c, j - c)``."""
    c = (size - 1) / 2
    r = torch.arange(size, device=device, dtype=torch.float32) - c
    return torch.stack(torch.meshgrid(r, r, indexing='ij'), -1)


def view_points(pos, theta, size):
    """World sample points ``(..., S, S, 2)`` of views at ``pos (..., 2)``,
    ``theta (...)``."""
    g = grid_offsets(size, pos.device)
    shape = theta.shape + (size, size)
    w = unrotate(g.expand(shape + (2,)), theta[..., None, None].expand(shape))
    return pos[..., None, None, :] + w


def in_footprint(points, pos, theta, half, eps=1e-4):
    """Boolean: ``points (..., 2)`` inside the rotated square of half-width
    ``half`` at ``pos (..., 2)``, ``theta (...)`` (broadcast)."""
    b = rotate(points - pos, theta)
    return (b.abs() <= half + eps).all(-1)
