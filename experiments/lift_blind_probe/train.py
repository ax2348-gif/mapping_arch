"""Train one blind-probe model.

    python experiments/lift_blind_probe/train.py --kind gdn --n 0 --steps 20000 --out runs/gdn_n0

--n 0 is continuous turns; --n 4/8/16/32 quantises them to 2 pi / n. The
held-out band (walk.BAND_DEG) is excluded from training unless --no-band.

Recipe (hybrid4's, scaled down): AdamW (0.9, 0.95), weight decay 0.1 on
matrices, base lr 3e-4 with the muP multipliers (effective 1.2e-3 on
hidden matrices at width 64), 100 warmup steps, cosine to zero, grad clip
1.0, bf16 autocast, mse on the 4 targets. Fresh episodes every step.

Eval sets (fixed seeds), every --eval-every steps:
  train   the training distribution (granularity n, band excluded)
  cont    continuous turns on the full circle, band included
  band    every turn inside the held-out band
  ctrl    every turn inside a trained control band of the same width (20-40 deg)
Per lag bin: nmse (mse / per-bin target variance, 1 = constant
prediction), and the same for the position pair (pnmse) and the heading
pair (hnmse) alone; position RMS error (walk units); median |heading
error| (deg). Checkpoints every --save-every steps and at the end.
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(__file__))
import walk                                                  # noqa: E402
from encoding import encode, SOURCE                          # noqa: E402
from models import BlindProbe                                 # noqa: E402

T, Q = 32, 16
LAG_BINS = [(1, 1), (2, 4), (5, 8), (9, 16), (17, 32)]
CTRL_BAND = (20.0, 40.0)


def feats_of(odo):
    return encode(odo[..., :2], odo[..., 2])


def band_walk(B, lo, hi, gen, device):
    """Batch whose every turn is uniform in [lo, hi] degrees."""
    bt = walk.batch(B, T, Q, None, None, gen, device)
    w = torch.deg2rad(lo + (hi - lo) * torch.rand(B, T, generator=gen, device=device))
    v = walk.V_MIN + (walk.V_MAX - walk.V_MIN) * torch.rand(B, T, generator=gen, device=device)
    dx, dy, dth = walk.step_pose(v, w)
    p, th = walk.integrate(dx, dy, dth)
    s, t = bt['s'], bt['t']
    rx, ry, rth = walk.relative(p, th, s, t)
    scale = walk.STEP_RMS * (t - s).float().sqrt()
    return dict(odo=torch.stack([dx, dy, dth], -1), w=w, s=s, t=t, scale=scale,
                tgt=torch.stack([rx / scale, ry / scale, rth.cos(), rth.sin()], -1),
                rel=torch.stack([rx, ry, rth], -1))


def eval_sets(n, band, device, B=2048):
    out = {}
    for name, seed in (('train', 101), ('cont', 102), ('band', 103), ('ctrl', 104)):
        g = torch.Generator(device=device).manual_seed(seed)
        if name == 'train':
            out[name] = walk.batch(B, T, Q, n, band, g, device)
        elif name == 'cont':
            out[name] = walk.batch(B, T, Q, None, None, g, device)
        elif name == 'band':
            out[name] = band_walk(B, *walk.BAND_DEG, g, device)
        else:
            out[name] = band_walk(B, *CTRL_BAND, g, device)
    return out


@torch.no_grad()
def evaluate(model, sets):
    model.eval()
    res = {}
    for name, bt in sets.items():
        preds = []
        for i in range(0, bt['odo'].shape[0], 512):
            sl = slice(i, i + 512)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                preds.append(model(feats_of(bt['odo'][sl]), bt['s'][sl], bt['t'][sl]).float())
        pred = torch.cat(preds)
        tgt = bt['tgt']
        lag = bt['t'] - bt['s']
        se = (pred - tgt).pow(2)
        pos_err = ((pred[..., :2] - tgt[..., :2]) * bt['scale'][..., None]).pow(2).sum(-1)
        hd = torch.atan2(pred[..., 3], pred[..., 2]) - bt['rel'][..., 2]
        hd = torch.rad2deg(walk.wrap(hd)).abs()
        r = {'nmse': float(se.mean() / tgt.var(dim=(0, 1)).mean())}
        for lo, hi in LAG_BINS:
            m = (lag >= lo) & (lag <= hi)
            var = tgt[m].var(0).mean()
            r[f'nmse_{lo}_{hi}'] = float(se[m].mean() / var)
            r[f'pnmse_{lo}_{hi}'] = float(se[m][:, :2].mean() / tgt[m][:, :2].var(0).mean())
            r[f'hnmse_{lo}_{hi}'] = float(se[m][:, 2:].mean() / tgt[m][:, 2:].var(0).mean())
            r[f'pos_{lo}_{hi}'] = float(pos_err[m].mean().sqrt())
            r[f'head_{lo}_{hi}'] = float(hd[m].median())
        res[name] = r
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', required=True)
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--no-band', action='store_true')
    ap.add_argument('--steps', type=int, default=100000)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--save-every', type=int, default=25000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = 'cuda'
    torch.manual_seed(a.seed)
    n = a.n or None
    band = None if a.no_band else walk.BAND_DEG
    os.makedirs(a.out, exist_ok=True)
    feat_dim = feats_of(torch.zeros(1, 1, 3, device=dev)).shape[-1]
    model = BlindProbe(a.kind, feat_dim, T=T).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.1},
                             {'params': no_decay, 'weight_decay': 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    sets = eval_sets(n, band, dev)
    gen = torch.Generator(device=dev).manual_seed(1000 + a.seed)
    cfg = dict(vars(a), params=nparams, encoding=SOURCE, band=band, T=T, Q=Q,
               v=(walk.V_MIN, walk.V_MAX), step_rms=walk.STEP_RMS)
    json.dump(cfg, open(f'{a.out}/config.json', 'w'), indent=1)
    print(f'{a.kind} n={a.n} params={nparams} encoding={SOURCE}', flush=True)
    log = []
    t0 = time.time()
    for step in range(a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            r = evaluate(model, sets)
            log.append({'step': step, 'time': time.time() - t0, **r})
            print(f"step {step:6d} {time.time() - t0:6.0f}s  train nmse {r['train']['nmse']:.4f} "
                  f"(lag17-32 {r['train']['nmse_17_32']:.4f}, pos {r['train']['pnmse_17_32']:.4f})  cont {r['cont']['nmse']:.4f}  "
                  f"band {r['band']['nmse']:.4f}  ctrl {r['ctrl']['nmse']:.4f}", flush=True)
            json.dump(log, open(f'{a.out}/log.json', 'w'))
        if step and step % a.save_every == 0 and step < a.steps:
            torch.save({'model': model.state_dict(), 'config': cfg}, f'{a.out}/step{step}.pt')
        if step == a.steps:
            break
        bt = walk.batch(a.batch, T, Q, n, band, gen, dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(feats_of(bt['odo']), bt['s'], bt['t']).float()
        loss = (pred - bt['tgt']).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    torch.save({'model': model.state_dict(), 'config': cfg}, f'{a.out}/final.pt')


if __name__ == '__main__':
    main()
