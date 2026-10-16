"""Transition probe and beta reading for blind-probe checkpoints.

    python experiments/lift_blind_probe/probe.py RUN_DIR [RUN_DIR ...]
    python experiments/lift_blind_probe/probe.py --synthetic

Adapted from tools/transition_probe.py and tools/read_beta.py. For the
first mixer (Gated DeltaNet), recompute every token's k, decay a and beta
with the layer's own ``gates`` and, per walk step i and per head, multiply
the homogeneous updates

    A_tok = a_tok (I - beta_tok k_tok k_tok^T)              (32 x 32)

over the step's span in that head's scan order: the odometry token of step
i and every token up to the next odometry token.
  left-to-right: odo_i, anchor_i, queries at i
  right-to-left: odo_i, queries at i-1, anchor_{i-1}
Products of consecutive spans compose exactly into the transition between
any two steps.

Finding the rotating eigenpair, without using the commanded turns:
  1. Plane. On a fit half of the episodes, every complex eigenvalue with
     |lambda| > 0.5 and |arg| > 5 deg contributes the real plane of its
     eigenvector (Re v, Im v); the top-2 eigenvectors of the mean projector
     give U (32 x 2). Its eigenvalue spectrum says how concentrated the
     rotations are in one plane (1, 1, 0, ... = a single shared plane).
  2. Signed eigenangle on the other half: M = U^T P U,
     phi = atan2(M10 - M01, M00 + M11), with the invariance residual
     |P U - U M| / |P U| and the pair modulus sqrt(det M). One global
     orientation bit per head (the sign of U's basis) is fixed so that phi
     correlates positively with the turn on the fit half; it is reported.
  3. Basis-free check: the eigenvalue whose eigenvector overlaps U most;
     |arg lambda| should equal |phi|.
For the complex-diagonal control the transition is already block-diagonal:
each of the 16 channel pairs rotates by the sum of its per-token angles
over the span; the pair that tracks the turn best on the fit half is
reported on the other half.

beta and a: per head, for the odometry token and the anchor token of each
step span, against the step's turn.
"""
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(__file__))
import walk                                                  # noqa: E402
from models import BlindProbe                                # noqa: E402
from encoding import encode                                  # noqa: E402

T, Q = 32, 16


# ------------------------------------------------------------ core algebra

def span_products(k, g, beta, spans):
    """k (L, d), g (L,), beta (L,) for one head in scan order; spans a list
    of index lists. Returns P (n_spans, d, d), later tokens on the left."""
    d = k.shape[-1]
    out = np.empty((len(spans), d, d))
    eye = np.eye(d)
    for n, sp in enumerate(spans):
        P = eye.copy()
        for j in sp:
            kj = k[j][:, None]
            P = (np.exp(g[j]) * (eye - beta[j] * (kj @ kj.T))) @ P
        out[n] = P
    return out


def fit_plane(Ps):
    """Mean projector onto the real planes of the significant rotating
    eigenpairs. Returns U (d, 2) and the projector's top eigenvalues."""
    d = Ps.shape[-1]
    acc, cnt = np.zeros((d, d)), 0
    ev, evec = np.linalg.eig(Ps)
    for i in range(len(Ps)):
        for j in range(d):
            lam = ev[i, j]
            if abs(lam) > 0.5 and abs(np.angle(lam)) > math.radians(5) and lam.imag > 0:
                B = np.stack([evec[i, :, j].real, evec[i, :, j].imag], 1)
                Qm, _ = np.linalg.qr(B)
                acc += Qm @ Qm.T
                cnt += 1
    if cnt == 0:
        return None, np.zeros(4), 0
    acc /= cnt
    w, V = np.linalg.eigh(acc)
    return V[:, ::-1][:, :2], w[::-1][:6], cnt


def plane_angles(Ps, U):
    """Signed angle, invariance residual and modulus of P restricted to U."""
    M = np.einsum('da,ndc,cb->nab', U, Ps, U)
    phi = np.arctan2(M[:, 1, 0] - M[:, 0, 1], M[:, 0, 0] + M[:, 1, 1])
    PU = Ps @ U
    res = np.linalg.norm(PU - U @ M, axis=(1, 2)) / np.maximum(np.linalg.norm(PU, axis=(1, 2)), 1e-12)
    mod = np.sqrt(np.abs(np.linalg.det(M)))
    return phi, res, mod


def eig_angles(Ps, U):
    """Basis-free: |arg| of the eigenvalue whose eigenvector lies most in U."""
    ev, evec = np.linalg.eig(Ps)
    ov = np.linalg.norm(np.einsum('da,ndj->naj', U, evec), axis=1) / np.linalg.norm(evec, axis=1)
    j = ov.argmax(1)
    lam = ev[np.arange(len(Ps)), j]
    return np.abs(np.angle(lam)), np.abs(lam), ov.max(1)


def orient(phi_fit, w_fit):
    return 1.0 if np.mean(np.sin(phi_fit) * np.sin(w_fit)) >= 0 else -1.0


# ------------------------------------------------------------ synthetic check

def synthetic_check(seed=0):
    """A known rotation built from two reflections (beta = 2) in a random
    plane of R^32, a decayed third token that leaves the plane invariant,
    and a contracting complement: the probe must recover the signed angle
    (up to the one global orientation bit) and its modulus."""
    rng = np.random.default_rng(seed)
    d, N = 32, 400
    Qm, _ = np.linalg.qr(rng.standard_normal((d, d)))
    e1, e2 = Qm[:, 0], Qm[:, 1]
    comp = Qm[:, 2:]
    w = rng.uniform(-math.pi, math.pi, N)
    ks, gs, bs, spans = [], [], [], []
    for i in range(N):
        k1 = e1
        k2 = math.cos(w[i] / 2) * e1 + math.sin(w[i] / 2) * e2
        k3 = comp @ rng.standard_normal(d - 2); k3 /= np.linalg.norm(k3)
        k4 = comp @ rng.standard_normal(d - 2); k4 /= np.linalg.norm(k4)
        base = len(ks)
        ks += [k1, k2, k3, k4]
        gs += [0.0, 0.0, math.log(0.9), 0.0]
        bs += [2.0, 2.0, 0.6, 1.0]
        spans.append([base, base + 1, base + 2, base + 3])
    Ps = span_products(np.array(ks), np.array(gs), np.array(bs), spans)
    # products of two reflections in the plane are a rotation by w there
    U, spec, cnt = fit_plane(Ps[: N // 2])
    assert spec[0] > 0.999 and spec[1] > 0.999 and spec[2] < 1e-6, spec
    phi, res, mod = plane_angles(Ps, U)
    sgn = orient(phi[: N // 2], w[: N // 2])
    err = np.abs(np.angle(np.exp(1j * (sgn * phi - w))))
    assert err.max() < 1e-8, err.max()
    assert res.max() < 1e-8 and np.allclose(mod, 0.9), (res.max(), mod[:3])
    ea, el, ov = eig_angles(Ps, U)
    assert np.allclose(ea, np.abs(w), atol=1e-6) and np.allclose(el, 0.9) and ov.min() > 0.999
    # the same two reflections composed over consecutive spans give the summed angle
    Pc = Ps[1] @ Ps[0]
    phic, _, _ = plane_angles(Pc[None], U)
    assert abs(np.angle(np.exp(1j * (sgn * phic[0] - w[0] - w[1])))) < 1e-8
    print(f'synthetic probe check passed: max angle error {err.max():.1e} rad over {N} steps, '
          f'plane spectrum {np.round(spec[:3], 6)}')


# ------------------------------------------------------------ model probe

def load(run, ckpt='final.pt'):
    ck = torch.load(f'{run}/{ckpt}', map_location='cuda', weights_only=False)
    cfg = ck['config']
    feat_dim = encode(torch.zeros(1, 2, device='cuda'), torch.zeros(1, device='cuda')).shape[-1]
    model = BlindProbe(cfg['kind'], feat_dim, T=T).cuda()
    model.load_state_dict(ck['model'])
    model.eval()
    return model, cfg


@torch.no_grad()
def collect(model, episodes=512, seed=777):
    """Continuous full-circle walks (band included). Returns per-head token
    quantities in scan order plus labels."""
    g = torch.Generator(device='cuda').manual_seed(seed)
    bt = walk.batch(episodes, T, Q, None, None, g, 'cuda')
    feats = encode(bt['odo'][..., :2], bt['odo'][..., 2])
    x, order = model(feats, bt['s'], bt['t'], stop_at=0)
    _, _, kind, step = BlindProbe.layout(bt['t'], T)
    kind = torch.gather(kind, 1, order).cpu().numpy()
    step = torch.gather(step, 1, order).cpu().numpy()
    return x.float(), kind, step, bt['w'].cpu().numpy().astype(np.float64)


def spans_for(kind, step, reverse):
    """Per episode: dict step i -> (token indices in scan order, kind list)."""
    kd = kind[::-1] if reverse else kind
    st = step[::-1] if reverse else step
    odo = np.where(kd == 1)[0]
    out = {}
    for a, p in enumerate(odo):
        end = odo[a + 1] if a + 1 < len(odo) else len(kd)
        out[int(st[p])] = (list(range(p, end)), kd[p:end].tolist())
    return out


def probe_gdn(model, episodes=512):
    gdn = model.blocks[0].mixer
    x, kind, step, w = collect(model, episodes)
    with torch.no_grad():
        q, k, v, g, beta = gdn.gates(x)
    k = k.float().cpu().numpy().astype(np.float64)
    g = g.float().cpu().numpy().astype(np.float64)
    beta = beta.float().cpu().numpy().astype(np.float64)
    H, half = gdn.heads, gdn.heads // 2
    out = {'w': w}
    for name, heads, rev in (('l2r', range(half), False), ('r2l', range(half, H), True)):
        for hh in heads:
            Ps, ws, ep, st, b_odo, b_anc, a_odo, a_anc = [], [], [], [], [], [], [], []
            for e in range(episodes):
                sp = spans_for(kind[e], step[e], rev)
                steps = sorted(sp)
                idx = [sp[i][0] for i in steps]
                Ps.append(span_products(k[e, :, hh], g[e, :, hh], beta[e, :, hh], idx))
                for i in steps:
                    toks, kinds = sp[i]
                    anc = toks[kinds.index(0)]
                    b_odo.append(beta[e, toks[0], hh]); b_anc.append(beta[e, anc, hh])
                    a_odo.append(math.exp(g[e, toks[0], hh])); a_anc.append(math.exp(g[e, anc, hh]))
                    ws.append(w[e, i - 1]); ep.append(e); st.append(i)
            Ps = np.concatenate(Ps)
            ws, ep = np.array(ws), np.array(ep)
            fit = ep < episodes // 2
            U, spec, cnt = fit_plane(Ps[fit & (np.abs(ws) > math.radians(20))][:4000])
            tag = f'{name}{hh - (0 if name == "l2r" else half)}'
            r = {'spec': spec, 'n_planes': cnt, 'beta_odo': np.array(b_odo), 'beta_anchor': np.array(b_anc),
                 'a_odo': np.array(a_odo), 'a_anchor': np.array(a_anc), 'w': ws, 'ep': ep, 'step': np.array(st)}
            if U is not None:
                phi, res, mod = plane_angles(Ps, U)
                sgn = orient(phi[fit], ws[fit])
                ea, el, ov = eig_angles(Ps, U)
                r.update(phi=sgn * phi, res=res, mod=mod, sign=sgn, eig_angle=ea, eig_mod=el, overlap=ov)
                # composition over spans of 1..8 consecutive steps (held-out half)
                comp = {}
                hold = np.where(~fit)[0]
                for lag in (1, 2, 4, 8):
                    errs = []
                    for e in range(episodes // 2, episodes):
                        base = np.where(ep == e)[0]                  # steps 1..T in order
                        for s0 in range(0, T - lag + 1, lag):
                            P = np.eye(Ps.shape[-1])
                            for j in range(s0, s0 + lag):
                                P = (Ps[base[j]] @ P) if name == 'l2r' else (P @ Ps[base[j]])
                            ph, _, _ = plane_angles(P[None], U)
                            errs.append(np.angle(np.exp(1j * (sgn * ph[0] - ws[base[s0:s0 + lag]].sum()))))
                    comp[lag] = np.array(errs)
                r['comp'] = comp
                del hold
            out[tag] = r
    return out


def probe_cdiag(model, episodes=512):
    cd = model.blocks[0].mixer
    x, kind, step, w = collect(model, episodes)
    with torch.no_grad():
        b, l, _ = x.shape
        h, d, half = cd.heads, cd.d, cd.heads // 2
        theta = math.pi * torch.tanh(cd.angle(x).float()).view(b, l, h, d // 2)
        th = torch.cat([theta[:, :, :half], theta[:, :, half:].flip(1)], 2).cpu().numpy()
    out = {'w': w}
    for name, heads, rev in (('l2r', range(half), False), ('r2l', range(half, h), True)):
        for hh in heads:
            ang, ws, ep = [], [], []
            for e in range(episodes):
                sp = spans_for(kind[e], step[e], rev)
                for i in sorted(sp):
                    ang.append(th[e, sp[i][0], hh].sum(0)); ws.append(w[e, i - 1]); ep.append(e)
            ang, ws, ep = np.array(ang), np.array(ws), np.array(ep)
            fit = ep < episodes // 2
            score = [abs(np.mean(np.exp(1j * (ang[fit, j] - ws[fit])))) for j in range(ang.shape[1])]
            score_neg = [abs(np.mean(np.exp(1j * (-ang[fit, j] - ws[fit])))) for j in range(ang.shape[1])]
            j = int(np.argmax(np.maximum(score, score_neg)))
            sgn = 1.0 if score[j] >= score_neg[j] else -1.0
            tag = f'{name}{hh - (0 if name == "l2r" else half)}'
            out[tag] = {'phi': np.angle(np.exp(1j * sgn * ang[:, j])), 'w': ws, 'ep': ep, 'pair': j,
                        'sign': sgn, 'pair_scores': np.maximum(score, score_neg),
                        'all_pairs': np.angle(np.exp(1j * ang))}
    return out


def calibration(phi, w, ep, episodes):
    """Slope / residual on the held-out half of episodes, out of band and in
    band. Slope fits use |w| < 160 deg to stay off the +-180 wrap."""
    hold = ep >= episodes // 2
    wd, pd = np.degrees(w[hold]), np.degrees(phi[hold])
    resid = np.degrees(np.angle(np.exp(1j * np.radians(pd - wd))))
    band = (wd >= walk.BAND_DEG[0]) & (wd <= walk.BAND_DEG[1])
    core = np.abs(wd) < 160
    out = {}
    for name, m in (('out_of_band', ~band), ('in_band', band)):
        fm = m & core
        A = np.stack([wd[fm], np.ones(fm.sum())], 1)
        coef = np.linalg.lstsq(A, wd[fm] + resid[fm], rcond=None)[0]
        out[name] = dict(n=int(m.sum()), slope=float(coef[0]), intercept=float(coef[1]),
                         resid_rms=float(np.sqrt(np.mean(resid[m] ** 2))),
                         resid_median_abs=float(np.median(np.abs(resid[m]))),
                         resid_mean=float(np.mean(resid[m])),
                         frac_within_10=float(np.mean(np.abs(resid[m]) < 10)))
    return out


def main(runs):
    for run in runs:
        model, cfg = load(run)
        if cfg['kind'] in ('gdn', 'gdn_b1'):
            res = probe_gdn(model)
        elif cfg['kind'] == 'cdiag':
            res = probe_cdiag(model)
        else:
            print(f'{run}: no recurrence to probe'); continue
        summary = {}
        flat = {}
        for tag, r in res.items():
            if tag == 'w':
                continue
            s = {}
            if 'phi' in r:
                s['calibration'] = calibration(r['phi'], r['w'], r['ep'], 512)
                s['sign'] = r['sign']
            if 'spec' in r:
                s['plane_spectrum'] = [round(float(x), 4) for x in r['spec']]
                if 'res' in r:
                    s['invariance_residual_median'] = float(np.median(r['res']))
                    s['modulus_median'] = float(np.median(r['mod']))
                    s['basis_free_vs_plane_median_deg'] = float(np.degrees(np.median(np.abs(r['eig_angle'] - np.abs(r['phi'])))))
                    s['composition_within_10deg'] = {str(k): float(np.mean(np.abs(np.degrees(v)) < 10)) for k, v in r['comp'].items()}
                bins = np.linspace(-180, 180, 13)
                wd = np.degrees(r['w'])
                s['beta_odo_by_turn'] = [float(np.median(r['beta_odo'][(wd >= lo) & (wd < hi)])) for lo, hi in zip(bins[:-1], bins[1:])]
                s['beta_anchor_by_turn'] = [float(np.median(r['beta_anchor'][(wd >= lo) & (wd < hi)])) for lo, hi in zip(bins[:-1], bins[1:])]
                s['a_odo_median'] = float(np.median(r['a_odo'])); s['a_anchor_median'] = float(np.median(r['a_anchor']))
            if 'pair' in r:
                s['pair'] = r['pair']; s['pair_scores'] = [round(float(x), 3) for x in r['pair_scores']]
            summary[tag] = s
            for key, val in r.items():
                if isinstance(val, np.ndarray):
                    flat[f'{tag}__{key}'] = val
                elif key == 'comp':
                    for lag, arr in val.items():
                        flat[f'{tag}__comp{lag}'] = arr
        np.savez_compressed(f'{run}/probe.npz', **flat)
        json.dump(summary, open(f'{run}/probe.json', 'w'), indent=1)
        print(run)
        print(json.dumps({t: {k: v for k, v in s.items() if 'by_turn' not in k} for t, s in summary.items()}, indent=1))


if __name__ == '__main__':
    if sys.argv[1:] == ['--synthetic']:
        synthetic_check()
    else:
        synthetic_check()
        main(sys.argv[1:])
