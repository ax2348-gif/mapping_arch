"""Causal coverage for se2c queries, on the GPU.

The contract (CONTRACT_SE2C.md): every sample point of a query asked at
step t lies inside the union of the footprints of observations 0..t. A
footprint is the convex hull of a view's sample points, a square of
half-width ``c = (S - 1) / 2`` rotated by its heading.

``first_seen`` rasterises the footprints onto a fine grid (cell side
``cell`` px) and stores, per cell, the first step whose footprint
contains the WHOLE cell. That is conservative: a point in a cell marked
seen by step t is inside a footprint of a step <= t, exactly, whatever
its position in the cell. A point near a footprint edge can be covered
and still fail the raster test; that costs some valid queries near the
boundary of the covered region (measured in docs/lift/build_log.md) but
never admits an uncovered one. ``mapping.se2c.reference.covered`` is the
exact check the tests hold the raster to.

A cell of half-side h is inside a square of half-width c rotated by theta
iff its centre, in the square's frame, is within ``c - h (|cos| + |sin|)``
on both axes (the cell's extent along a rotated axis).
"""
import math

import torch

from mapping.se2c.geometry import rotate

NEVER = 32767            # int16 "never seen"


class Raster:
    """First-seen steps on a per-episode grid. ``origin (B, 2)`` is the
    world position of cell (0, 0)'s corner; ``seen (B, H, W)`` int16."""

    def __init__(self, origin, seen, cell):
        self.origin, self.seen, self.cell = origin, seen, cell

    def lookup(self, points):
        """First-seen step at world ``points (B, ..., 2)``; NEVER outside
        the grid."""
        b, h, w = self.seen.shape
        flat = points.reshape(b, -1, 2)
        ij = torch.floor((flat - self.origin[:, None]) / self.cell).long()
        inside = ((ij[..., 0] >= 0) & (ij[..., 0] < h)
                  & (ij[..., 1] >= 0) & (ij[..., 1] < w))
        idx = (ij[..., 0].clamp(0, h - 1) * w + ij[..., 1].clamp(0, w - 1))
        val = torch.gather(self.seen.view(b, -1), 1, idx)
        val = torch.where(inside, val, torch.full_like(val, NEVER))
        return val.view(points.shape[:-1])


def first_seen(pos, theta, half, cell=0.25):
    """Raster of first-seen steps for footprints at ``pos (B, T1, 2)``,
    ``theta (B, T1)`` of half-width ``half``. Each footprint is scattered
    into the cells of its bounding box (an ``amin`` over steps)."""
    b, t1 = theta.shape
    dev = pos.device
    reach = half * math.sqrt(2) + cell
    lo = pos.amin(1) - reach - cell                       # (B, 2)
    hi = pos.amax(1) + reach + cell
    origin = torch.floor(lo / cell) * cell
    size = torch.ceil((hi - origin) / cell).long().amax(0)    # (2,)
    H, W = int(size[0]), int(size[1])
    nb = int(math.ceil(2 * reach / cell)) + 2
    seen = torch.full((b, H * W), NEVER, dtype=torch.int16, device=dev)
    # cells of each footprint's bounding box; the body-frame coordinates
    # of a cell centre are affine in its row and column, so they are built
    # as a broadcast sum of a row part and a column part
    corner = torch.floor((pos - reach - origin[:, None]) / cell).long()  # (B,T1,2)
    r = torch.arange(nb, device=dev)
    ii = corner[..., 0, None] + r                         # (B,T1,nb)
    jj = corner[..., 1, None] + r
    yc = ii.float() * cell + cell / 2 + origin[:, None, 0, None] - pos[..., 0, None]
    xc = jj.float() * cell + cell / 2 + origin[:, None, 1, None] - pos[..., 1, None]
    cs, sn = torch.cos(theta)[..., None], torch.sin(theta)[..., None]
    # rotate((y, x), theta) = (c y - s x, s y + c x)
    margin = (half - (cell / 2) * (cs.abs() + sn.abs()) - 1e-5)[..., None]
    by = (cs * yc)[..., :, None] - (sn * xc)[..., None, :]          # (B,T1,nb,nb)
    inside = by.abs() <= margin
    bx = (sn * yc)[..., :, None] + (cs * xc)[..., None, :]
    inside &= bx.abs() <= margin
    inside &= (((ii >= 0) & (ii < H))[..., :, None]
               & ((jj >= 0) & (jj < W))[..., None, :])
    step = torch.arange(t1, device=dev, dtype=torch.int16)
    val = torch.where(inside, step[None, :, None, None], NEVER).to(torch.int16)
    idx = (ii.clamp(0, H - 1)[..., :, None] * W
           + jj.clamp(0, W - 1)[..., None, :]).reshape(b, -1)
    seen.scatter_reduce_(1, idx, val.reshape(b, -1), 'amin')
    return Raster(origin, seen.view(b, H, W), cell)
