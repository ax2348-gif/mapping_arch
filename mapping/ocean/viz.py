"""Shared drawing helpers for the ocean contact sheets and videos."""
import math

import numpy as np
import torch
from PIL import Image, ImageDraw

from mapping.ocean import field as fld
from mapping.ocean.render import camera_rays, heights, march, shade


def topdown(world, spec, b, y0, x0, n, step=1.0, device='cuda'):
    """Top-down ``(n, n)`` arrays for episode ``b``: height and shaded albedo,
    rows = y from y0, cols = x from x0, ``step`` px per cell."""
    o = torch.arange(n, device=device) * step
    grid = torch.stack(torch.meshgrid(y0 + o, x0 + o, indexing='ij'), -1)
    sub = {k: {kk: vv[b:b + 1] for kk, vv in w.items()} for k, w in world.items() if isinstance(w, dict)}
    pts2 = grid.reshape(1, -1, 2)
    h = heights(sub, pts2, spec)
    pts = torch.cat([pts2, h[..., None]], -1)
    down = torch.tensor([0.0, 0.0, -1.0], device=device).expand_as(pts)
    img = shade(sub, spec, None, pts, down, None, aa=False)  # top-down: no haze
    return h.reshape(n, n).cpu().numpy(), img.reshape(n, n).cpu().numpy()


def footprint(world, spec, cam, pos, theta, b, per_side=12, far=90.0):
    """Ground trace of the view's frustum: hit points of its border rays (rays
    that see sky or reach past ``far`` px are cut at ``far``),
    ``(M, 2)`` (y, x) for episode ``b``, pose ``pos (2,)``, ``theta`` scalar."""
    sub = {k: {kk: vv[b:b + 1] for kk, vv in w.items()} for k, w in world.items() if isinstance(w, dict)}
    o, d = camera_rays(pos.view(1, 1, 2), theta.view(1, 1), cam)
    s = cam.size
    idx = np.linspace(0, s - 1, per_side).round().astype(int)
    border = ([(0, j) for j in idx] + [(i, s - 1) for i in idx] +
              [(s - 1, j) for j in idx[::-1]] + [(i, 0) for i in idx[::-1]])
    dd = torch.stack([d[0, 0, i, j] for i, j in border]).view(1, 1, -1, 3)
    t, _, hit = march(sub, spec, o, dd)
    t = torch.where(hit, t.clamp(max=far), torch.full_like(t, far))   # sky rays: cut at `far` px
    p = o.view(1, 1, 1, 3) + t[..., None] * dd
    return p[0, 0, :, :2].cpu().numpy()


def to_pil(a, scale=1, lo=0.0, hi=1.0):
    a = np.clip((np.asarray(a, dtype=np.float32) - lo) / (hi - lo), 0, 1)
    im = Image.fromarray((a * 255).astype(np.uint8))
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    return im


def height_rgb(h, amp):
    """Height as a blue (low) to sand (high) ramp, shared scale +-3 amp."""
    t = np.clip((h / (3 * amp) + 1) / 2, 0, 1)[..., None]
    lo, hi = np.array([20, 40, 110]), np.array([235, 215, 160])
    return Image.fromarray((lo + t * (hi - lo)).astype(np.uint8))


def draw_poly(draw, pts_yx, y0, x0, scale, fill=None, outline=(255, 80, 40), width=2):
    xy = [((x - x0) * scale, (y - y0) * scale) for y, x in pts_yx]
    draw.polygon(xy, fill=fill, outline=outline, width=width)
