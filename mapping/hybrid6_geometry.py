"""Local camera calibration for hybrid6. No trajectory integration.

Camera coordinates are (x right, y down, z forward). Matrices map sensor
coordinates into the associated time frame. Planar inputs use (dy, dx,
clockwise radians), matching mapping.se2c. The plane is a renderer convention,
never a model input. Calibration is (fx, fy, cx, cy, skew) in patch pixels.
"""
import math

import torch

PATCH = 3
SCALE = 256.0
OCTAVES = 10
POSE_FEATURES = 3 + 6 * OCTAVES + 9 + 1
SENSOR_FEATURES = 2 + 2 + 4 * OCTAVES + 1


def fourier_translation(x):
    x = x.float() / SCALE
    freq = math.pi * 2 ** torch.arange(OCTAVES, device=x.device)
    phase = x[..., None] * freq
    return torch.cat((x, phase.sin().flatten(-2), phase.cos().flatten(-2)), -1)


def pose_features(pose, known=None):
    """Raw local SE3 pose; known=False means unknown odometry, not identity."""
    if pose.shape[-2:] != (4, 4):
        raise ValueError('extrinsics must end in (4, 4)')
    features = torch.cat((fourier_translation(pose[..., :3, 3]),
                          pose[..., :3, :3].float().flatten(-2)), -1)
    if known is None:
        known = torch.ones_like(pose[..., 0, 0], dtype=torch.bool)
    flag = known.to(features.dtype)[..., None]
    return torch.cat((features * flag, flag), -1)


def sensor_features(calibration):
    """Focal lengths, principal point, skew; no depth or world pixel scale."""
    if calibration.shape[-1] != 5:
        raise ValueError('calibration must end in (fx, fy, cx, cy, skew)')
    c = calibration.float()
    if not torch.isfinite(c).all() or (c[..., :2] <= 0).any():
        raise ValueError('calibration must be finite with positive focal lengths')
    return torch.cat((c[..., :2].log(), fourier_translation(c[..., 2:4]),
                      c[..., 4:5] / c[..., :1]), -1)


def planar_pose(pose):
    """(dy, dx, clockwise radians) -> sensor-to-local SE3 matrix."""
    pose = pose.float()
    if pose.shape[-1] != 3:
        raise ValueError('planar pose must end in (dy, dx, angle)')
    out = torch.eye(4, device=pose.device).expand(*pose.shape[:-1], 4, 4).clone()
    c, s = pose[..., 2].cos(), pose[..., 2].sin()
    out[..., 0, 0], out[..., 0, 1] = c, -s
    out[..., 1, 0], out[..., 1, 1] = s, c
    out[..., 0, 3], out[..., 1, 3] = pose[..., 1], pose[..., 0]
    return out


def patch_calibration(height, width, device, focal=1.0):
    """Centered 3x3 crops of a centered image; row-major, c_i=-a_i."""
    if height <= 0 or width <= 0 or height % PATCH or width % PATCH:
        raise ValueError('image dimensions must be positive multiples of 3')
    y = torch.arange(height // PATCH, device=device) * PATCH + 1 - (height - 1) / 2
    x = torch.arange(width // PATCH, device=device) * PATCH + 1 - (width - 1) / 2
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    c = torch.zeros(height // PATCH * (width // PATCH), 5, device=device)
    c[:, :2] = focal
    c[:, 2], c[:, 3] = -xx.flatten(), -yy.flatten()
    return c


def patchify(x):
    *lead, h, w = x.shape
    if h <= 0 or w <= 0 or h % PATCH or w % PATCH:
        raise ValueError('image dimensions must be positive multiples of 3')
    return (x.reshape(*lead, h // PATCH, PATCH, w // PATCH, PATCH)
            .transpose(-3, -2).reshape(*lead, (h // PATCH) * (w // PATCH), PATCH ** 2))


def unpatchify(x, height, width):
    *lead, n, pixels = x.shape
    if n != (height // PATCH) * (width // PATCH) or pixels != PATCH ** 2:
        raise ValueError('patch count does not match output dimensions')
    return (x.reshape(*lead, height // PATCH, width // PATCH, PATCH, PATCH)
            .transpose(-3, -2).reshape(*lead, height, width))
