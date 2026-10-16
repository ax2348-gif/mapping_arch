"""float64 NumPy reference for the ocean renderer: the same fields and rays,
but a pure safe-step march (t += g / D with the exact Lipschitz bound, no
over-relaxation, no minimum step), run to a 1e-9 px residual. It cannot step
over the surface, so its depth is the true FIRST intersection.
"""
import math

import numpy as np

TWO_PI = 2 * math.pi


def field(freq, phase, pts):
    """z and grad z at ``pts (N, 2)`` for one episode's waves."""
    k = phase.shape[0]
    norm = math.sqrt(2.0 / k)
    arg = TWO_PI * pts @ freq.T + phase
    z = np.cos(arg).sum(-1) * norm
    g = -norm * TWO_PI * np.sin(arg) @ freq
    return z, g


def march(freq, phase, amp, origin, d, tol=1e-9, max_iter=400000, tmax=600.0):
    """Depth along rays ``d (N, 3)`` from ``origin (N, 3)``, all in float64,
    and a hit mask: rays pointing up or still above the surface at ``tmax``
    are sky (depth tmax)."""
    k = phase.shape[0]
    norm = math.sqrt(2.0 / k)
    lip = amp * norm * TWO_PI * np.linalg.norm(freq, axis=-1).sum()
    sup = amp * norm * k
    dz = d[:, 2]
    D = np.abs(dz) + lip * np.linalg.norm(d[:, :2], axis=-1)
    down = dz < -1e-6
    t = np.where(down, np.clip((origin[:, 2] - sup) / np.maximum(-dz, 1e-9), 0, None), tmax)
    t = np.minimum(t, tmax)

    def g(t):
        p = origin + t[:, None] * d
        return p[:, 2] - amp * field(freq, phase, p[:, :2])[0]
    gt = g(t)
    live = down & (t < tmax)
    for _ in range(max_iter):
        live = live & (gt >= tol) & (t < tmax)
        if not live.any():
            break
        t = np.where(live, np.minimum(t + np.clip(gt, 0, None) / D, tmax), t)
        gt = np.where(live, g(t), gt)
    hit = down & (gt < tol) & (t < tmax)
    return np.where(hit, t, tmax), hit
