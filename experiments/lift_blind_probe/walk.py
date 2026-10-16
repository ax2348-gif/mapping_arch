"""Isotropic SE(2) walks for the blind probe: odometry in, relative pose out.

Each step i = 1..T applies a unit-time twist (v, 0, w) through the SE(2)
exponential, giving the relative pose u_i of pose i in the body frame of
pose i-1:

    dtheta = w,  dx = v sin(w) / w,  dy = v (1 - cos(w)) / w     (x forward, y left)

v ~ U[V_MIN, V_MAX] (v > 0), w ~ U[-pi, pi) or on the grid 2 pi k / n.
A held-out band of w can be removed from the training distribution.
Poses are integrated from pose 0 = identity; the target of query (s, t),
s < t, is pose s in the body frame of pose t: R(-theta_t)(p_s - p_t) and
the heading change theta_s - theta_t as (cos, sin).

``python experiments/lift_blind_probe/walk.py`` runs the assertions.
"""
import math

import torch

V_MIN, V_MAX = 0.5, 1.0
BAND_DEG = (50.0, 70.0)        # held-out band of the turn w, degrees (closed)


def wrap(a):
    """Angle to [-pi, pi)."""
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def in_band(w, band=BAND_DEG):
    d = torch.rad2deg(w)
    return (d >= band[0]) & (d <= band[1])


def sample_turns(shape, n, band, gen, device):
    """Turns w of the given shape. ``n`` None = continuous U[-pi, pi), else
    the grid 2 pi k / n, k = -n/2 .. n/2 - 1. With ``band`` (deg, deg),
    values inside it are never drawn (continuous: uniform on the rest)."""
    if n is None:
        lo, hi = (math.radians(band[0]), math.radians(band[1])) if band else (0.0, 0.0)
        width = 2 * math.pi - (hi - lo)
        u = torch.rand(shape, generator=gen, device=device) * width - math.pi
        return torch.where(u >= lo, u + (hi - lo), u) if band else u
    grid = 2 * math.pi * torch.arange(-n // 2, n // 2, device=device) / n
    if band:
        grid = grid[~in_band(grid, band)]
    idx = torch.randint(len(grid), shape, generator=gen, device=device)
    return grid[idx]


def step_pose(v, w):
    """SE(2) exponential of the twist (v, 0, w) for unit time: (dx, dy, dth)."""
    small = w.abs() < 1e-6
    ws = torch.where(small, torch.ones_like(w), w)
    dx = torch.where(small, v, v * torch.sin(ws) / ws)
    dy = torch.where(small, v * w / 2, v * (1 - torch.cos(ws)) / ws)
    return dx, dy, w


def integrate(dx, dy, dth):
    """Poses 0..T from relative steps (B, T): returns p (B, T+1, 2), th (B, T+1)."""
    B, T = dx.shape
    th = torch.cat([dx.new_zeros(B, 1), torch.cumsum(dth, 1)], 1)       # unwrapped
    c, s = torch.cos(th[:, :-1]), torch.sin(th[:, :-1])
    gx = c * dx - s * dy
    gy = s * dx + c * dy
    p = torch.cat([dx.new_zeros(B, 1, 2), torch.stack([gx, gy], -1).cumsum(1)], 1)
    return p, th


def relative(p, th, s, t):
    """Pose s in the frame of pose t, for index tensors s, t (B, Q):
    (dx, dy, dtheta wrapped)."""
    ps = torch.gather(p, 1, s[..., None].expand(-1, -1, 2))
    pt = torch.gather(p, 1, t[..., None].expand(-1, -1, 2))
    ths, tht = torch.gather(th, 1, s), torch.gather(th, 1, t)
    d = ps - pt
    c, sn = torch.cos(tht), torch.sin(tht)
    rx = c * d[..., 0] + sn * d[..., 1]
    ry = -sn * d[..., 0] + c * d[..., 1]
    return rx, ry, wrap(ths - tht)


def step_rms(T=32, B=20000):
    """Per-axis RMS of the relative position per sqrt(lag), measured on
    continuous isotropic walks (fixed seed). Consecutive chord directions
    are correlated (E cos(w/2) > 0), so this is not sqrt(E|u|^2 / 2)."""
    g = torch.Generator().manual_seed(12345)
    w = sample_turns((B, T), None, None, g, 'cpu').double()
    v = V_MIN + (V_MAX - V_MIN) * torch.rand(B, T, generator=g, dtype=torch.float64)
    p, th = integrate(*step_pose(v, w))
    s, t = sample_queries(B, 16, T, g, 'cpu')
    rx, ry, _ = relative(p, th, s, t)
    return math.sqrt(((rx ** 2 + ry ** 2) / (2 * (t - s))).mean().item())


def sample_queries(B, Q, T, gen, device):
    """Q query pairs per episode: lag uniform in 1..T, t uniform in lag..T,
    s = t - lag."""
    lag = torch.randint(1, T + 1, (B, Q), generator=gen, device=device)
    u = torch.rand(B, Q, generator=gen, device=device)
    t = lag + torch.floor(u * (T - lag + 1)).long()
    return t - lag, t


def batch(B, T, Q, n, band, gen, device):
    """One batch. Returns dict with odometry (dx, dy, dth) (B, T, 3), the
    turns w (B, T), queries s, t (B, Q), targets (B, Q, 4) as
    (dx / scale, dy / scale, cos, sin) with scale = STEP_RMS sqrt(lag),
    and raw relative poses (B, Q, 3)."""
    w = sample_turns((B, T), n, band, gen, device)
    v = V_MIN + (V_MAX - V_MIN) * torch.rand(B, T, generator=gen, device=device)
    dx, dy, dth = step_pose(v, w)
    p, th = integrate(dx, dy, dth)
    s, t = sample_queries(B, Q, T, gen, device)
    rx, ry, rth = relative(p, th, s, t)
    scale = STEP_RMS * (t - s).float().sqrt()
    tgt = torch.stack([rx / scale, ry / scale, torch.cos(rth), torch.sin(rth)], -1)
    return dict(odo=torch.stack([dx, dy, dth], -1), w=w, s=s, t=t, tgt=tgt,
                rel=torch.stack([rx, ry, rth], -1), scale=scale)


STEP_RMS = step_rms()


# ----------------------------------------------------------------- checks

def _se2(x, y, th):
    m = torch.zeros(*x.shape, 3, 3, dtype=x.dtype)
    m[..., 0, 0] = torch.cos(th); m[..., 0, 1] = -torch.sin(th); m[..., 0, 2] = x
    m[..., 1, 0] = torch.sin(th); m[..., 1, 1] = torch.cos(th); m[..., 1, 2] = y
    m[..., 2, 2] = 1
    return m


def check():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    B, T, Q = 64, 32, 16
    for n in (None, 4, 8, 16, 32):
        for band in (None, BAND_DEG):
            w = sample_turns((20000,), n, band, gen, 'cpu')
            assert (w >= -math.pi).all() and (w < math.pi).all()
            if band:
                assert not in_band(w, band).any(), 'band leaked into training turns'
            if n is not None:
                k = w * n / (2 * math.pi)
                assert torch.allclose(k, k.round(), atol=1e-4)
    # 1. step pose = matrix exponential of the twist
    v = (V_MIN + (V_MAX - V_MIN) * torch.rand(500, dtype=torch.float64))
    w = sample_turns((500,), None, None, gen, 'cpu').double()
    w[:3] = torch.tensor([0.0, 1e-8, -math.pi], dtype=torch.float64)
    dx, dy, dth = step_pose(v, w)
    xi = torch.zeros(500, 3, 3, dtype=torch.float64)
    xi[:, 0, 1], xi[:, 1, 0], xi[:, 0, 2] = -w, w, v
    E = torch.linalg.matrix_exp(xi)
    assert torch.allclose(E, _se2(dx, dy, dth), atol=1e-9), 'exp map mismatch'
    # 2. walk composition = integrated poses
    for n in (None, 4, 32):
        bt = batch(B, T, Q, n, BAND_DEG, gen, 'cpu')
        odo = bt['odo'].double()
        steps = _se2(odo[..., 0], odo[..., 1], odo[..., 2])          # (B, T, 3, 3)
        p, th = integrate(odo[..., 0], odo[..., 1], odo[..., 2])
        G = torch.eye(3, dtype=torch.float64).expand(B, 3, 3).clone()
        for i in range(T):                                          # pose i+1
            G = G @ steps[:, i]
            assert torch.allclose(G, _se2(p[:, i + 1, 0], p[:, i + 1, 1], th[:, i + 1]),
                                  atol=1e-9), 'integration mismatch'
        for b in range(8):
            for q in range(Q):
                s, t = int(bt['s'][b, q]), int(bt['t'][b, q])
                assert 0 <= s < t <= T
                M = torch.eye(3, dtype=torch.float64)
                for i in range(s, t):                               # pose t in frame s
                    M = M @ steps[b, i]
                R = torch.linalg.inv(M)                             # pose s in frame t
                rel = bt['rel'][b, q].double()
                assert abs(R[0, 2] - rel[0]) < 1e-4 and abs(R[1, 2] - rel[1]) < 1e-4, 'relative pos'
                ang = torch.atan2(R[1, 0], R[0, 0])
                assert abs(wrap(ang - rel[2])) < 1e-5, 'relative heading'
    # 3. target scaling gives roughly unit variance per position axis at every lag
    bt = batch(4096, T, Q, None, None, gen, 'cpu')
    var = bt['tgt'][..., :2].pow(2).mean().item()
    assert 0.9 < var < 1.1, var
    print(f'walk checks passed (STEP_RMS = {STEP_RMS:.4f}, normalised position var {var:.3f})')


if __name__ == '__main__':
    check()
