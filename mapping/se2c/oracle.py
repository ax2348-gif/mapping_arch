"""Pose-oracle references for se2c: how well can a query be answered by
someone who is GIVEN the true poses and only has to reconstruct the
texture from the observed pixels?

- ``analytic``: render the query from the episode's own texture
  function. Exact by construction, since every target point is covered,
  so it checks the generator's consistency (it must be 0). Unfair as a
  ceiling: the function also holds the unobserved texture.
- ``interpolate``: each target point is interpolated (bilinear or
  bicubic) in the one observation, of step <= t, in whose footprint it
  lies deepest. This is the ceiling reported: the residual the task
  imposes on a pose-perfect reconstruction that copies pixels, from the
  texture's content above the pixel Nyquist limit (the contrast
  nonlinearity's harmonics) and from interpolating between pixel
  centres. A cleverer reconstruction could do somewhat better; it is a
  reference, not a bound. (A kriging oracle with the texture's empirical
  covariance was tried and dropped: the tabulated covariance is not
  positive definite and the noise-free system is too ill-conditioned to
  give a stable answer; docs/lift/build_log.md.)
"""
import math

import torch
import torch.nn.functional as F

from mapping.se2c import geometry as geo
from mapping.se2c import texture as tex


@torch.no_grad()
def interpolate(batch, config, mode='bicubic'):
    """Each target point from the observation (<= its step) in whose
    footprint it lies deepest, interpolated in that observation's pixel
    grid (``mode`` 'bilinear' or 'bicubic'; bicubic clamps at the
    border)."""
    S, half = config.view, config.half
    pos, th = batch['poses'][..., :2], batch['poses'][..., 2]
    obs = batch['observations']
    qw, qt = batch['query_world'], batch['query_time']
    b, t1 = th.shape
    q = qt.shape[1]
    tp = geo.view_points(qw[..., :2], qw[..., 2], S).reshape(b, q * S * S, 2)
    body = geo.rotate(tp[:, :, None] - pos[:, None], th[:, None])   # (b,n,t1,2)
    depth = half - body.abs().amax(-1)
    lim = (qt if config.query_coverage == 'causal'
           else torch.full_like(qt, t1 - 1)).repeat_interleave(S * S, 1)
    depth = torch.where(torch.arange(t1, device=obs.device) <= lim[..., None],
                        depth, torch.full_like(depth, -1e9))
    s = depth.argmax(-1)                                           # (b,n)
    bb = torch.gather(body, 2, s[..., None, None].expand(-1, -1, 1, 2))[:, :, 0]
    pix = bb + half                                                # (row, col)
    img = torch.gather(obs, 1, s[..., None, None].expand(-1, -1, S, S))
    grid = torch.stack([pix[..., 1], pix[..., 0]], -1) / (S - 1) * 2 - 1
    v = F.grid_sample(img.reshape(-1, 1, S, S), grid.reshape(-1, 1, 1, 2),
                      mode=mode, align_corners=True, padding_mode='border')
    return v.view(b, q, S, S).clamp(0, 1)


@torch.no_grad()
def analytic(batch, config):
    """The query from the episode's own texture (exact; a consistency
    check)."""
    qw = batch['query_world']
    return tex.render(batch['texture_params'], qw[..., :2], qw[..., 2],
                      config.view, config.texture)
