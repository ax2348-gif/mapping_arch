"""Tile-bank textures for se2c (MNIST, Imagenette), kept as an option for
continuity with the discrete tasks: ``Config.bank`` names one of
``mapping.data.TEXTURES``.

A map is a grid of random tiles from the bank, each under a random one of
its eight dihedral symmetries, as in ``mapping.data``, built on the GPU
per episode around the walk. The map is a pixel image, so it is turned
into a function of continuous position by bilinear interpolation between
pixel centres (pixel (r, c) sits at map coordinate (r, c)), and a view
pixel is the AVERAGE of that function over an ``aa x aa`` grid of points
spread over the pixel's own (rotated) square: supersampling with a box
filter, which antialiases the resampling. With ``aa = 1`` and a view at
an integer centre and a quarter-turn heading, a view is exactly
``mapping.se2.face`` of the crop (``tests/test_se2c_texture.py``).

The start position is uniform within one tile, so the walk's phase
against the tile grid is uniform. The bank's test split is used for
evaluation, its train split for training, as in ``mapping.loader``.
"""
import torch
import torch.nn.functional as F

from mapping.data import TEXTURES, load_bank
from mapping.se2c.geometry import grid_offsets, unrotate


def load(name, split, device):
    """``(8, N, t, t)`` uint8 bank on ``device``: every tile under each of
    its dihedral symmetries (``mapping.data.load_bank``)."""
    t = TEXTURES[name]
    bank, _ = load_bank(t['path'], t['cutoff'], split, dihedral=True)
    return torch.from_numpy(bank).to(device)


def build_maps(g, bank, pos, view, device):
    """Random tile maps covering every view of the walk. Returns ``maps
    (B, H, W)`` float in [0, 1] and ``origin (B, 2)``, the world position
    of map pixel (0, 0)."""
    b = pos.shape[0]
    tile = bank.shape[-1]
    reach = (view - 1) / 2 * 1.4143 + 2
    lo = pos.amin(1) - reach
    hi = pos.amax(1) + reach
    # tile grid aligned at a uniformly random phase
    phase = torch.rand(b, 2, generator=g, device=device) * tile
    origin = torch.floor((lo - phase) / tile) * tile + phase
    n = torch.ceil((hi - origin) / tile).long().amax(0)          # (2,)
    nh, nw = int(n[0]), int(n[1])
    idx = torch.randint(0, bank.shape[1], (b, nh, nw), generator=g,
                        device=device)
    sym = torch.randint(0, 8, (b, nh, nw), generator=g, device=device)
    tiles = bank[sym, idx]                                       # (b,nh,nw,t,t)
    maps = tiles.permute(0, 1, 3, 2, 4).reshape(b, nh * tile, nw * tile)
    return maps.float() / 255, origin


def render(maps, origin, pos, theta, size, aa=4):
    """Views ``(B, V, S, S)`` at ``pos (B, V, 2)``, ``theta (B, V)`` from
    the maps, bilinear and supersampled ``aa x aa`` per pixel."""
    b, v = theta.shape
    h, w = maps.shape[-2:]
    sub = (torch.arange(aa, device=pos.device, dtype=torch.float32)
           + 0.5) / aa - 0.5
    sub = torch.stack(torch.meshgrid(sub, sub, indexing='ij'), -1).view(-1, 2)
    off = grid_offsets(size, pos.device)[:, :, None, :] + sub   # (S,S,aa2,2)
    off = off.expand(b, v, *off.shape)
    th = theta[..., None, None, None].expand(off.shape[:-1])
    pts = pos[:, :, None, None, None] + unrotate(off, th)       # (B,V,S,S,aa2,2)
    m = pts - origin[:, None, None, None, None]
    grid = torch.stack([m[..., 1] / (w - 1), m[..., 0] / (h - 1)], -1) * 2 - 1
    out = F.grid_sample(maps[:, None], grid.view(b, 1, -1, 2), mode='bilinear',
                        padding_mode='border', align_corners=True)
    return out.view(b, v, size, size, aa * aa).mean(-1)
