"""Was a surface point seen by an earlier view? The basis for honest causal
coverage in the ocean world.

A surface point P is visible from a view at camera centre c if
  (1) P projects inside that view's image (in front of the camera, within
      the square field of view), and
  (2) the first surface hit along the ray from c toward P is P itself, not a
      crest in front of it: march (mapping.ocean.render.march) from c along
      (P - c) / |P - c| and compare the hit range with |P - c|.
(2) is an exact ray test, not a shadow-map lookup, so it has no depth-map
resolution error.
"""
import math

import torch

from mapping.ocean.render import march


def camera_frame(theta, cam):
    """Unit axis, right and camera-up vectors ``(..., 3)`` for headings."""
    c, s = torch.cos(theta), torch.sin(theta)
    zero = torch.zeros_like(c)
    fwd = torch.stack([-c, s, zero], -1)
    right = torch.stack([s, c, zero], -1)
    up = torch.zeros_like(fwd)
    up[..., 2] = 1
    ph = math.radians(cam.pitch)
    return math.cos(ph) * fwd - math.sin(ph) * up, right, math.sin(ph) * fwd + math.cos(ph) * up


def project(points, pos, theta, cam):
    """Pixel coordinates ``(i, j)`` of world ``points (B, N, 3)`` in views
    ``pos (B, 2)``, ``theta (B,)``, and whether each is inside the image."""
    axis, right, cup = camera_frame(theta, cam)
    c = torch.cat([pos, torch.full_like(pos[:, :1], cam.altitude)], -1)
    v = points - c[:, None]
    z = (v * axis[:, None]).sum(-1)
    x = (v * right[:, None]).sum(-1)
    y = (v * cup[:, None]).sum(-1)
    ctr = (cam.size - 1) / 2
    j = ctr + cam.focal * x / z.clamp(min=1e-6)
    i = ctr - cam.focal * y / z.clamp(min=1e-6)
    inside = (z > 0) & (i > -0.5) & (i < cam.size - 0.5) & (j > -0.5) & (j < cam.size - 0.5)
    return i, j, inside


def visible(world, spec, cam, points, pos, theta, tol=0.05):
    """``(B, N)`` bool: points visible (in frame and unoccluded) from views at
    ``pos (B, 2)``, ``theta (B,)``. ``tol`` in px of range."""
    _, _, inside = project(points, pos, theta, cam)
    c = torch.cat([pos, torch.full_like(pos[:, :1], cam.altitude)], -1)
    v = points - c[:, None]
    dist = v.norm(dim=-1)
    d = v / dist[..., None]
    t, _, hit = march(world, spec, c[:, None, :], d[:, None])
    return inside & ((t[:, 0] >= dist - tol) | ~hit[:, 0])


def coverage(world, spec, cam, target_points, poses, upto):
    """For each target's ``(B, S, S, 3)`` surface points, whether each pixel's
    point was visible in some view s <= upto among ``poses (B, V, 3)`` =
    (y, x, theta). Returns ``(B, S, S)`` bool. Sky pixels are not handled
    here (the caller masks them: sky depends only on the ray direction)."""
    b = target_points.shape[0]
    pts = target_points.reshape(b, -1, 3)
    seen = torch.zeros(pts.shape[:2], dtype=torch.bool, device=pts.device)
    for s in range(int(upto) + 1):
        seen |= visible(world, spec, cam, pts, poses[:, s, :2], poses[:, s, 2])
    return seen.reshape(target_points.shape[:-1])
