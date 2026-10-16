"""The shared se2c pose encoding (``mapping.se2c.encoding``): a stable
interface used by the models and the blind probe."""
import math

import numpy as np
import torch

from mapping.se2c import encoding as enc


def test_shapes_and_layout():
    d = torch.randn(3, 5, 2) * 50
    th = torch.rand(3, 5) * 2 * math.pi - math.pi
    o, q = enc.encode_odometry(d, th), enc.encode_query(d, th)
    assert o.shape == (3, 5, enc.ODOMETRY_FEATURES) == (3, 5, 50)
    assert q.shape == (3, 5, enc.QUERY_FEATURES)
    assert o.dtype == torch.float32 and torch.isfinite(o).all()
    assert torch.allclose(o[..., :2], d / enc.SCALE)
    a = o[..., enc.TRANSLATION_FEATURES:]
    assert torch.allclose(a[..., 0], th.cos(), atol=1e-6)
    assert torch.allclose(a[..., len(enc.HARMONICS)], th.sin(), atol=1e-6)
    assert torch.allclose(a[..., 1], (2 * th).cos(), atol=1e-5)


def test_angle_periodic_and_finest_octave_is_one_pixel():
    d = torch.zeros(2, 2)
    th = torch.tensor([-math.pi, math.pi])
    a = enc.encode_odometry(d, th)
    assert torch.allclose(a[0], a[1], atol=1e-5)
    # shifting by one pixel leaves the finest octave unchanged
    x = torch.tensor([[3.3, -7.1]])
    f0, f1 = enc.translation_features(x), enc.translation_features(x + 1)
    k = enc.OCTAVES - 1
    sin_idx = [2 + k, 2 + enc.OCTAVES + k]
    assert torch.allclose(f0[0, sin_idx], f1[0, sin_idx], atol=1e-4)


def test_numpy_and_device_agree():
    rng = np.random.default_rng(0)
    d = rng.normal(size=(7, 2)) * 30
    th = rng.uniform(-math.pi, math.pi, 7)
    a = enc.encode_query(d, th)
    b = enc.encode_query(torch.tensor(d), torch.tensor(th))
    assert torch.allclose(a, b)
    if torch.cuda.is_available():
        c = enc.encode_query(torch.tensor(d).cuda(), torch.tensor(th).cuda())
        assert torch.allclose(a, c.cpu(), atol=1e-5)
