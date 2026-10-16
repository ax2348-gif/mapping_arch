"""Ocean renderer prototype (mapping/ocean): exactness, bounds, visibility, AA."""
import math

import numpy as np
import pytest
import torch

from mapping.ocean import field as fld
from mapping.ocean import reference as ref
from mapping.ocean.render import CameraSpec, OceanSpec, camera_rays, march, render, sample_world, select
from mapping.ocean.visibility import project, visible

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def setup(seed=0, batch=2, views=3, pitch=45.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    spec, cam = OceanSpec(), CameraSpec(pitch=pitch)
    world = sample_world(g, batch, spec, DEV)
    pos = torch.rand(batch, views, 2, device=DEV, generator=g) * 80
    th = torch.rand(batch, views, device=DEV, generator=g) * 2 * math.pi
    return spec, cam, world, pos, th


def test_bounds_hold():
    spec, cam, world, pos, th = setup()
    sup, lip = fld.bounds(world['height'])
    pts = torch.rand(2, 20000, 2, device=DEV) * 400
    z, gz = fld.evaluate(world['height'], pts, grad=True)
    assert (z.abs().amax(1) <= sup + 1e-4).all()
    assert (gz.norm(dim=-1).amax(1) <= lip + 1e-4).all()


def test_gradient_matches_finite_difference():
    spec, cam, world, pos, th = setup()
    p = torch.rand(2, 50, 2, device=DEV, dtype=torch.float64) * 100
    w64 = {k: v.double() for k, v in world['height'].items()}
    _, gz = fld.evaluate(w64, p, grad=True)
    e = 1e-5
    for c in range(2):
        dp = torch.zeros_like(p); dp[..., c] = e
        fd = (fld.evaluate(w64, p + dp) - fld.evaluate(w64, p - dp)) / (2 * e)
        assert (fd - gz[..., c]).abs().max() < 1e-5


def test_march_matches_float64_safe_step_reference():
    spec, cam, world, pos, th = setup(seed=1, pitch=40.0)
    o, d = camera_rays(pos, th, cam)
    t, r, hit = march(world, spec, o, d)
    assert hit.all() and (r.abs() < 1e-3).all()
    b, v = 0, 1
    fr = world['height']['freq'][b].double().cpu().numpy()
    ph = world['height']['phase'][b].double().cpu().numpy()
    dd = d[b, v].reshape(-1, 3).double().cpu().numpy()
    oo = np.repeat(o[b, v].double().cpu().numpy()[None], dd.shape[0], 0)
    tr, _ = ref.march(fr, ph, spec.amp, oo, dd, tmax=spec.tmax)
    assert np.abs(tr - t[b, v].reshape(-1).double().cpu().numpy()).max() < 1e-3


def test_horizon_sky_and_reference_hit_mask():
    """Default camera sees the horizon: some rays are sky, the sky/hit mask
    and the depths of hits match the float64 safe-step reference."""
    spec, cam, world, pos, th = setup(seed=1, pitch=CameraSpec().pitch)
    assert 0 < cam.horizon_row < cam.size / 2
    o, d = camera_rays(pos, th, cam)
    t, r, hit = march(world, spec, o, d)
    assert (~hit).any() and hit.any()
    b, v = 1, 2
    fr = world['height']['freq'][b].double().cpu().numpy()
    ph = world['height']['phase'][b].double().cpu().numpy()
    dd = d[b, v].reshape(-1, 3).double().cpu().numpy()
    oo = np.repeat(o[b, v].double().cpu().numpy()[None], dd.shape[0], 0)
    tr, hr = ref.march(fr, ph, spec.amp, oo, dd, tmax=spec.tmax)
    hg = hit[b, v].reshape(-1).cpu().numpy()
    assert (hr == hg).all()
    tg = t[b, v].reshape(-1).double().cpu().numpy()
    assert np.abs(tr[hr] - tg[hr]).max() < 1e-2
    with pytest.raises(ValueError):
        CameraSpec(pitch=95.0)


def test_rendered_points_are_visible_from_their_own_view_and_reproject():
    spec, cam, world, pos, th = setup()
    out = render(world, spec, cam, pos[:, :1], th[:, :1])
    pts = out['points'][:, 0].reshape(2, -1, 3)
    h = out['hit'][:, 0].reshape(2, -1)
    vis = visible(world, spec, cam, pts, pos[:, 0], th[:, 0])
    assert vis[h].all()
    i, j, inside = project(pts, pos[:, 0], th[:, 0], cam)
    ii = torch.arange(cam.size, device=DEV).repeat_interleave(cam.size).float().expand_as(i)
    jj = torch.arange(cam.size, device=DEV).repeat(cam.size).float().expand_as(j)
    assert inside[h].all() and (i - ii)[h].abs().max() < 1e-2 and (j - jj)[h].abs().max() < 1e-2


def test_points_behind_a_crest_are_hidden():
    """A camera far behind at a grazing pitch cannot see everything a nearby
    view sees: some in-frame points are occluded (rough sea)."""
    spec = OceanSpec(amp=8.0)
    cam = CameraSpec(pitch=36.0, altitude=36.0)
    g = torch.Generator(device=DEV).manual_seed(3)
    world = sample_world(g, 4, spec, DEV)
    pos = torch.rand(4, 1, 2, device=DEV, generator=g) * 80
    th = torch.rand(4, 1, device=DEV, generator=g) * 2 * math.pi
    out = render(world, spec, cam, pos, th)
    pts = out['points'][:, 0].reshape(4, -1, 3)
    fwd = torch.stack([-torch.cos(th[:, 0]), torch.sin(th[:, 0])], -1)
    back = pos[:, 0] - 25.0 * fwd
    _, _, inside = project(pts, back, th[:, 0], cam)
    vis = visible(world, spec, cam, pts, back, th[:, 0])
    assert (inside & ~vis & out['hit'][:, 0].reshape(4, -1)).any()


def test_footprint_aa_is_closer_to_supersampling_than_no_aa():
    spec, cam, world, pos, th = setup(seed=4, pitch=36.0)
    a = render(world, spec, cam, pos, th, aa=False)['image']
    b = render(world, spec, cam, pos, th, aa=True)['image']
    c = render(world, spec, cam, pos, th, supersample=4)['image']
    assert (b - c).abs().mean() < (a - c).abs().mean()


def test_select_slices_episodes():
    spec, cam, world, pos, th = setup()
    w0 = select(world, slice(0, 1))
    assert w0['height']['freq'].shape[0] == 1


def test_time_zero_is_the_static_field_and_time_moves_it():
    from mapping.ocean.render import sample_world_seeds
    spec, cam = OceanSpec(), CameraSpec()
    w = sample_world_seeds(3, 7, 2, spec, DEV)
    pos = torch.zeros(2, 2, 2, device=DEV); th = torch.zeros(2, 2, device=DEV)
    a = render(w, spec, cam, pos, th)
    b = render(w, spec, cam, pos, th, time=torch.zeros(2, 2, device=DEV))
    c = render(w, spec, cam, pos, th, time=torch.full((2, 2), 7.0, device=DEV))
    assert torch.equal(a['image'], b['image'])
    assert (a['image'] - c['image']).abs().mean() > 0.01
    assert c['resid'][c['hit']].abs().max() < 1e-3           # still exact at t > 0


def test_height_and_albedo_seeds_are_independent():
    from mapping.ocean.render import sample_world_seeds
    spec = OceanSpec()
    a = sample_world_seeds(3, 7, 2, spec, DEV)
    b = sample_world_seeds(3, 8, 2, spec, DEV)
    c = sample_world_seeds(4, 7, 2, spec, DEV)
    assert torch.equal(a['height']['freq'], b['height']['freq'])
    assert not torch.equal(a['albedo']['freq'], b['albedo']['freq'])
    assert torch.equal(a['albedo']['freq'], c['albedo']['freq'])
    assert not torch.equal(a['height']['freq'], c['height']['freq'])


def test_regularity_dial_narrows_band_and_directions():
    lo = fld.regular(0.0); hi = fld.regular(1.0)
    assert (hi.f_hi - hi.f_lo) < (lo.f_hi - lo.f_lo) / 5
    assert lo.anisotropy == 0 and hi.anisotropy > 0.9
    g = torch.Generator(device=DEV).manual_seed(0)
    w = fld.sample_waves(g, 1, hi, DEV)
    ang = torch.atan2(w['freq'][0, :, 1], w['freq'][0, :, 0])
    spread = torch.remainder(ang - ang[0] + math.pi, 2 * math.pi) - math.pi
    assert spread.abs().max() < 2 * (1 - hi.anisotropy) * math.pi + 1e-4
