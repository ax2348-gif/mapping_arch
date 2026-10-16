"""se2c textures: the GPU field agrees with the NumPy reference, the
separable view renderer agrees with the direct sum, generation is
deterministic in the seed, and the tile-bank option reduces to the
discrete crop at quarter turns."""
import math

import numpy as np
import pytest
import torch

from mapping import se2d
from mapping.data import crop_windows, render_map
from mapping.se2c import bank as tb
from mapping.se2c import geometry as geo
from mapping.se2c import reference as ref
from mapping.se2c import texture as tex

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SPECS = [tex.TextureSpec(),
         tex.TextureSpec(nonlinearity='sigmoid', f_lo=1 / 32, f_hi=0.18,
                         slope=1.0),
         tex.TextureSpec(nonlinearity='linear')]


@pytest.mark.parametrize('spec', SPECS)
def test_gpu_matches_numpy_reference(spec):
    g = torch.Generator(device=DEV).manual_seed(3)
    p = tex.sample_params(g, 2, spec, DEV)
    pos = (torch.rand(2, 3, 2, generator=g, device=DEV) - 0.5) * 400
    th = (torch.rand(2, 3, generator=g, device=DEV) * 2 - 1) * math.pi
    views = tex.render(p, pos, th, 15, spec).cpu().numpy()
    z_direct = tex.field(p, geo.view_points(pos, th, 15)).cpu().numpy()
    z_sep = tex.view_field(p, pos, th, 15).cpu().numpy()
    assert np.abs(z_direct - z_sep).max() < 2e-4
    f, ph = p['freq'].cpu().numpy(), p['phase'].cpu().numpy()
    for b in range(2):
        for v in range(3):
            want = ref.render_view(f[b], ph[b], pos[b, v].cpu().numpy(),
                                   th[b, v].item(), 15, spec)
            assert np.abs(views[b, v] - want).max() < 5e-4


def test_field_statistics_and_determinism():
    spec = tex.TextureSpec()
    g1 = torch.Generator(device=DEV).manual_seed(7)
    g2 = torch.Generator(device=DEV).manual_seed(7)
    p1, p2 = (tex.sample_params(g, 32, spec, DEV) for g in (g1, g2))
    assert torch.equal(p1['freq'], p2['freq'])
    radius = p1['freq'].norm(dim=-1)
    assert radius.min() >= spec.f_lo - 1e-6 and radius.max() <= spec.f_hi + 1e-6
    pts = torch.rand(32, 4096, 2, device=DEV) * 500
    z = tex.field(p1, pts)
    assert abs(z.mean().item()) < 0.05 and abs(z.var().item() - 1) < 0.1


def test_bank_quarter_turns_equal_discrete_crop():
    bank = tb.load('mnist', 'test', DEV)
    g = torch.Generator(device=DEV).manual_seed(0)
    pos = torch.tensor([[[40., 50.], [43., 47.]]], device=DEV)
    maps, origin = tb.build_maps(g, bank, pos, 15, DEV)
    origin = torch.round(origin)      # integer pixel alignment for the check
    image = (maps[0] * 255).round().byte().cpu().numpy()
    for h in range(4):
        th = torch.full((1, 2), h * math.pi / 2, device=DEV)
        v = tb.render(maps, origin, pos, th, 15, aa=1)[0]
        for k in range(2):
            centre = (pos[0, k] - origin[0]).long().cpu().numpy()
            crop = crop_windows(image, (centre - 7)[None], 15)
            want = se2d.face(crop, [h])[0]
            got = (v[k] * 255).round().byte().cpu().numpy()
            assert (got == want).all()
    # antialiased views are close to, but smoother than, the point samples
    th = torch.full((1, 2), 0.3, device=DEV)
    a = tb.render(maps, origin, pos, th, 15, aa=4)
    b = tb.render(maps, origin, pos, th, 15, aa=1)
    assert (a - b).abs().mean() < 0.1
