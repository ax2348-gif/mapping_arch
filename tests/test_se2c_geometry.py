"""se2c geometry (``mapping.se2c.geometry``): agrees with the discrete
conventions of ``mapping.se2d`` at quarter turns, and the walk's
integration and odometry are inverse to each other."""
import math

import numpy as np
import torch

from mapping import se2d
from mapping.data import crop_windows
from mapping.se2c import geometry as geo


def test_rotate_matches_discrete_at_quarter_turns():
    v = torch.tensor([[3., -7.], [0., 2.], [-5., 4.]])
    for h in range(4):
        th = torch.full((3,), h * math.pi / 2)
        want = se2d.rotate(v.numpy().astype(np.int64), h)
        assert torch.allclose(geo.rotate(v, th), torch.tensor(want).float(),
                              atol=1e-5)
        assert torch.allclose(geo.unrotate(geo.rotate(v, th), th), v,
                              atol=1e-5)
    # facing direction of heading theta is (-cos, sin): forward = body -y
    th = torch.rand(10) * 6 - 3
    f = torch.stack([-torch.cos(th), torch.sin(th)], -1)
    assert torch.allclose(geo.rotate(f, th),
                          torch.tensor([-1., 0.]).expand(10, 2), atol=1e-5)


def test_view_points_are_discrete_face_at_quarter_turns():
    rng = np.random.default_rng(0)
    world = rng.integers(0, 255, (40, 40)).astype(np.uint8)
    for size in (9, 15):
        c = (size - 1) // 2
        for h in range(4):
            centre = np.array([17, 21])
            crop = crop_windows(world, (centre - c)[None], size)
            want = se2d.face(crop, [h])[0]
            pts = geo.view_points(torch.tensor(centre).float(),
                                  torch.tensor(h * math.pi / 2), size)
            ij = pts.round().long().numpy()
            assert np.abs(pts.numpy() - ij).max() < 1e-4
            assert (world[ij[..., 0], ij[..., 1]] == want).all()


def test_arc_matches_numerical_integration():
    v = torch.tensor([4.0, 7.5, 3.0, 5.0], dtype=torch.float64)
    w = torch.tensor([0.0, 1e-6, math.pi * 0.999, -2.0], dtype=torch.float64)
    d = geo.arc(v, w)
    n = 20000
    tau = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    ang = w[:, None] * tau
    num = torch.stack([(-torch.cos(ang)).mean(1), torch.sin(ang).mean(1)],
                      -1) * v[:, None]
    assert torch.allclose(d, num, atol=1e-6)


def test_integrate_and_odometry_roundtrip():
    g = torch.Generator().manual_seed(0)
    body = torch.randn(5, 32, 2, generator=g) * 4
    turn = (torch.rand(5, 32, generator=g) * 2 - 1) * math.pi
    p0 = torch.randn(5, 2, generator=g) * 10
    th0 = torch.rand(5, generator=g) * 6
    pos, th = geo.integrate(body, turn, p0, th0)
    b2, t2 = geo.odometry(pos, th)
    assert torch.allclose(b2, body, atol=1e-3)
    assert torch.allclose(geo.wrap(t2 - turn), torch.zeros_like(turn),
                          atol=1e-4)
    rel = geo.relative_pose(pos[:, 5], th[:, 5], pos[:, 4], th[:, 4])
    assert torch.allclose(rel[:, :2], body[:, 4], atol=1e-3)


def test_wrap_half_open():
    x = torch.tensor([-math.pi, math.pi, 3 * math.pi, 0.5])
    w = geo.wrap(x)
    assert (w >= -math.pi).all() and (w < math.pi).all()
    assert torch.allclose(w[:3], torch.full((3,), -math.pi))
