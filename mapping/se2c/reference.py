"""Slow, plain NumPy (float64) references for the se2c generator: the
texture sum, view rendering, and exact coverage. Tests compare the GPU
code against these; nothing in training calls them."""
import math

import numpy as np


def rotate(v, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c * v[..., 0] - s * v[..., 1],
                     s * v[..., 0] + c * v[..., 1]], -1)


def unrotate(b, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c * b[..., 0] + s * b[..., 1],
                     -s * b[..., 0] + c * b[..., 1]], -1)


def view_points(pos, theta, size):
    """``(S, S, 2)`` world points of one view."""
    c = (size - 1) / 2
    r = np.arange(size) - c
    g = np.stack(np.meshgrid(r, r, indexing='ij'), -1).astype(np.float64)
    return pos + unrotate(g, theta)


def field(freq, phase, points):
    """z at ``points (..., 2)`` for one episode's ``freq (K, 2)``, ``phase
    (K,)``: the defining sum, term by term."""
    freq = np.asarray(freq, np.float64)
    phase = np.asarray(phase, np.float64)
    pts = np.asarray(points, np.float64)
    z = np.zeros(pts.shape[:-1])
    for f, p in zip(freq, phase):
        z += np.cos(2 * math.pi * (pts[..., 0] * f[0] + pts[..., 1] * f[1]) + p)
    return z * math.sqrt(2.0 / len(freq))


def contrast(z, spec):
    if spec.nonlinearity == 'linear':
        return np.clip(0.5 + spec.gain * z, 0, 1)
    if spec.nonlinearity == 'sigmoid':
        return 1 / (1 + np.exp(-spec.contrast * z))
    return np.exp(-0.5 * (z / spec.width) ** 2)


def render_view(freq, phase, pos, theta, size, spec):
    return contrast(field(freq, phase, view_points(np.asarray(pos, np.float64),
                                                   float(theta), size)), spec)


def covered(points, pos, theta, half, eps=1e-4):
    """Exact: is each point ``(N, 2)`` inside at least one of the rotated
    squares ``pos (M, 2)``, ``theta (M,)`` of half-width ``half``? Loops
    over the squares."""
    pts = np.asarray(points, np.float64)
    hit = np.zeros(len(pts), bool)
    for p, th in zip(np.asarray(pos, np.float64), np.asarray(theta, np.float64)):
        b = rotate(pts - p, th)
        hit |= (np.abs(b) <= half + eps).all(-1)
    return hit
