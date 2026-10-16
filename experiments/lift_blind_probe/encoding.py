"""Odometry encoding for the blind probe: the shared spec,
``mapping.se2c.encoding.encode_odometry`` (commit d8b0da7), used as is.

The walk (walk.py) works in body-frame units with x forward, y left and
the heading counter-clockwise. The shared encoder takes pixels in the
repository's (dy, dx) order with y down (forward = -dy, right = +dx) and
dtheta positive = clockwise. ``encode`` converts, with PX pixels per walk
unit (9, the mapping task's step length, so a step is 4.5 to 9 px):

    dy_px = -x * PX,  dx_px = -y * PX,  dtheta_shared = -dtheta

A local implementation of the spec was written first while the shared
module did not exist; the shared module landed before any main run, so
every reported run uses it.
"""
import sys

import torch

sys.path.insert(0, '.')
from mapping.se2c.encoding import encode_odometry, ODOMETRY_FEATURES  # noqa: E402

SOURCE = 'shared: mapping.se2c.encoding (d8b0da7)'
PX = 9.0


def encode(dxdy, dtheta):
    """Walk-convention (x forward, y left, CCW) -> shared features."""
    px = torch.stack([-dxdy[..., 0] * PX, -dxdy[..., 1] * PX], -1)
    return encode_odometry(px, -dtheta)


def check():
    # a pure forward step: forward is -dy in the shared convention
    f = encode(torch.tensor([[1.0, 0.0]]), torch.tensor([0.5]))
    assert f.shape[-1] == ODOMETRY_FEATURES
    from mapping.se2c.encoding import angle_features, translation_features
    assert torch.allclose(f[0, :42], translation_features(torch.tensor([-PX, 0.0])))
    assert torch.allclose(f[0, 42:], angle_features(torch.tensor(-0.5)))
    print('encoding checks passed')


if __name__ == '__main__':
    check()
