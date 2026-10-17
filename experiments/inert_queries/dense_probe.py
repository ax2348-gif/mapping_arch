"""Dense-query probe of an inert-query se2d model, with a mixed control.

For held-out episodes, every window corner fully seen by time t, at every
heading, is a query (thousands per episode), all in ONE forward pass for
the inert model (its queries are independent, so one pass equals chunks),
and in chunks of 8 (its trained regime) for the mixed control. The
predictions, rotated back to the world frame, cover each world pixel many
times; their disagreement measures whether the model holds one map.

Measured per (episode, t) and pooled:
- per world pixel: the spread of the covering queries' predictions (std)
  and the consensus (their mean) error to truth. The individual squared
  error splits exactly into spread^2 + consensus error^2 (per pixel,
  population variance); ``shared`` = consensus^2 / individual is the share
  of the error every query makes alike (1: one committed belief, right or
  wrong; small: queries are independently noisy);
- by distance from the pose at t and by how long ago the pixel was last
  seen (by t);
- pairwise, over overlapping query pairs, by heading difference and
  offset: mean squared disagreement on shared pixels, and 1 - error
  correlation = msd / (e1^2 + e2^2) (1: independent errors, 0: identical);
- one-pass vs chunked deviation, on the fused kernel and, for a subset, on
  the sequential reference scan.

Parameters: ``Probe`` below, composed with Hydra's compose API (no
argparse); override on the command line as ``key=value``:

    python experiments/inert_queries/dense_probe.py episodes=4 out=results/probes/x

Writes ``<out>/results.json`` (with the code commit and the checkpoints'
run ids), per-episode mosaics and time-lapse GIFs.
"""
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

sys.path.insert(0, '.')
from mapping import data, se2d  # noqa: E402
import mapping.tasks as tasks  # noqa: E402
from mapping.checkpoint import load_checkpoint  # noqa: E402
from mapping.loader import load_banks  # noqa: E402

S = 9


@dataclass
class Probe:
    inert: str = 'results/se2d/hybrid4/s/inert_seed0_400k_qt_wsd/step100000.pt'
    mixed: str = ('/home/adrianna/code/mapping/results/se2/hybrid4/s/'
                  'seed0_400k_qt_wsd_small/step100000.pt')
    mixed_run_id: str = 'legacy-torch-se2d-hybrid4-s-seed0_400k_qt_wsd_small'
    out: str = 'results/probes/inert_dense_s100k'
    episodes: int = 16
    seed: int = 20261017          # episode draw (held-out bank)
    headings: list = field(default_factory=lambda: [0, 1, 2, 3])
    stride: int = 1               # corner grid stride in px
    mid_times: list = field(default_factory=lambda: [0, 4, 8, 12, 16, 20, 24, 28])
    mixed_chunk: int = 8          # the mixed model's trained query count
    check_chunk: int = 64         # chunk size of the one-pass check (inert)
    max_queries: int = 16384      # per forward; larger sets split by time
    reference_queries: int = 192  # one-pass vs chunked on the reference scan
    video_episodes: int = 3
    scan: str = 'exact'           # exact: the delta-rule recurrence (exact_scan);
                                  # kernel: FLA's fused chunked kernel (see results)
    device: str = 'cuda'


def compose_probe():
    from hydra import compose, initialize
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf
    ConfigStore.instance().store(name='dense_probe', node=Probe)
    with initialize(version_base=None, config_path=None):
        cfg = compose(config_name='dense_probe', overrides=sys.argv[1:])
    return OmegaConf.to_object(cfg)


# ------------------------------------------------------------ episodes, queries
def episodes(p, config, bank):
    b = data.make_batch(np.random.default_rng(p.seed), config, bank, p.episodes)
    out = []
    for i, ep in enumerate(b['episodes']):
        image = data.render_map(bank, ep.tile_grid, config.tile, ep.tile_sym)
        # the batch's own queries through this module's geometry: a check
        qp = se2d.query_pose(ep.queries, np.zeros(len(ep.queries), int),
                             ep.positions[ep.query_times], b['headings'][i][ep.query_times])
        assert np.array_equal(qp[:, :2], b['query_pose'][i][:, :2])
        out.append(dict(obs=b['observations'][i], odo=b['odometry'][i],
                        positions=ep.positions, headings=b['headings'][i],
                        image=image.astype(np.float32) / 255))
    return out


def dense_queries(ep, times, headings, stride):
    """corners (n, 2), query headings (n,), times (n,) for every corner whose
    window was fully seen by its time, at every heading."""
    H, W = ep['image'].shape
    first = data.coverage(ep['positions'], S, H, W, first=True)
    cs, hs, ts = [], [], []
    for t in times:
        valid = data.fully_covered_corners((first >= 0) & (first <= t), S)
        c = np.argwhere(valid)
        c = c[(c[:, 0] % stride == 0) & (c[:, 1] % stride == 0)]
        for h in headings:
            cs.append(c)
            hs.append(np.full(len(c), h))
            ts.append(np.full(len(c), t))
    return np.concatenate(cs), np.concatenate(hs), np.concatenate(ts)


def model_inputs(ep, corners, qh, qt, device):
    pose = se2d.query_pose(corners, qh, ep['positions'][qt], ep['headings'][qt])
    obs = torch.from_numpy(ep['obs']).float().div(255)[None].to(device)
    odo = torch.from_numpy(ep['odo']).long()[None].to(device)
    return (obs, odo, torch.from_numpy(pose).long()[None].to(device),
            torch.from_numpy(qt).long()[None].to(device))


@torch.no_grad()
def predict(net, inputs, chunk=None, order=None):
    """(n, 9, 9) predictions, query-facing; in one pass, or in chunks (of
    the queries permuted by ``order``)."""
    obs, odo, pose, qt = inputs
    if chunk is None:
        return net(obs, odo, pose, query_time=qt)[0]
    n = pose.shape[1]
    order = torch.arange(n, device=pose.device) if order is None else order
    pad = (-n) % chunk                       # pad the last chunk by repeating queries
    idx = torch.cat([order, order[:pad]]).view(-1, chunk)                # (chunks, chunk)
    out = torch.empty(n, S, S, device=pose.device)
    for i in range(0, idx.shape[0], 64):     # 64 chunks per forward, as a batch
        ix = idx[i:i + 64]
        m = ix.shape[0]
        y = net(obs.expand(m, -1, -1, -1), odo.expand(m, -1), pose[0, ix], query_time=qt[0, ix])
        out[ix.flatten()] = y.flatten(0, 1)
    return out


def to_world(pred, qh):
    """Query-facing windows back to the world frame (inverse of
    ``se2d.face``: np.rot90 by the heading)."""
    out = pred.clone()
    for h in range(4):
        m = qh == h
        if m.any():
            out[m] = torch.rot90(pred[m], -h, dims=(-2, -1))
    return out


# ------------------------------------------------------------------ statistics
def last_seen_by(ep, t):
    H, W = ep['image'].shape
    return data.coverage(ep['positions'][:t + 1], S, H, W)          # last time <= t


DIST_BINS = [0, 9, 18, 36, 1e9]              # px from the window centre at t
AGO_BINS = [0, 1, 4, 9, 17, 33]              # steps since last seen (<= t)


def pixel_stats(ep, world, corners, t, dev):
    """Per world pixel of one (episode, t): count, mean, var over covering
    queries; and bins."""
    H, W = ep['image'].shape
    r = torch.arange(S, device=dev)
    cy = torch.from_numpy(corners[:, 0]).to(dev)
    cx = torch.from_numpy(corners[:, 1]).to(dev)
    idx = ((cy[:, None, None] + r[None, :, None]) * W + cx[:, None, None] + r[None, None, :]).flatten()
    x = world.flatten()
    cnt = torch.zeros(H * W, device=dev).index_add_(0, idx, torch.ones_like(x))
    s1 = torch.zeros(H * W, device=dev).index_add_(0, idx, x)
    s2 = torch.zeros(H * W, device=dev).index_add_(0, idx, x * x)
    covered = cnt > 0
    mean = torch.where(covered, s1 / cnt.clamp(min=1), torch.nan)
    var = torch.where(covered, (s2 / cnt.clamp(min=1) - mean ** 2).clamp(min=0), torch.nan)
    truth = torch.from_numpy(ep['image']).to(dev).flatten()
    err = mean - truth
    # bins: distance from the window centre at t, steps since last seen by t
    yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing='ij')
    py, px = (ep['positions'][t] + (S - 1) / 2)
    dist = torch.sqrt((yy - py) ** 2 + (xx - px) ** 2).flatten().float()
    ago = (t - torch.from_numpy(last_seen_by(ep, t)).to(dev).flatten()).float()
    return dict(cnt=cnt, mean=mean, var=var, err=err, dist=dist, ago=ago, covered=covered,
                truth=truth, shape=(H, W))


def summarise(stats_list):
    """Pool pixel stats (weighted by covering queries, so individual mse =
    var + consensus err^2 exactly)."""
    def agg(mask_fn=None):
        tot = dict(w=0.0, var=0.0, bias=0.0, pixels=0)
        for s in stats_list:
            m = s['covered'] & (s['cnt'] >= 2)
            if mask_fn is not None:
                m = m & mask_fn(s)
            w = s['cnt'][m]
            tot['w'] += w.sum().item()
            tot['var'] += (w * s['var'][m]).sum().item()
            tot['bias'] += (w * s['err'][m] ** 2).sum().item()
            tot['pixels'] += int(m.sum())
        if tot['w'] == 0:
            return None
        var, bias = tot['var'] / tot['w'], tot['bias'] / tot['w']
        return dict(individual_mse=var + bias, disagreement_var=var, consensus_mse=bias,
                    disagreement_std=math.sqrt(var), shared=bias / max(var + bias, 1e-12),
                    pixels=tot['pixels'])
    out = dict(all=agg())
    out['by_distance_px'] = {f'{lo:g}-{hi:g}': agg(lambda s, lo=lo, hi=hi: (s['dist'] >= lo) & (s['dist'] < hi))
                             for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:])}
    out['by_steps_since_seen'] = {f'{lo}-{hi - 1}': agg(lambda s, lo=lo, hi=hi: (s['ago'] >= lo) & (s['ago'] < hi))
                                  for lo, hi in zip(AGO_BINS[:-1], AGO_BINS[1:])}
    # disagreement vs consensus error: pixels binned by |consensus error|
    edges = [0, 0.05, 0.1, 0.2, 0.4, 1.01]
    out['std_by_consensus_abs_error'] = {
        f'{lo}-{hi}': agg(lambda s, lo=lo, hi=hi: (s['err'].abs() >= lo) & (s['err'].abs() < hi))
        for lo, hi in zip(edges[:-1], edges[1:])}
    # committed wrong: consensus far off but queries agree
    wrong, committed = 0, 0
    for s in stats_list:
        m = s['covered'] & (s['cnt'] >= 2)
        bad = m & (s['err'].abs() > 0.25)
        wrong += int(bad.sum())
        committed += int((bad & (s['var'].sqrt() < 0.05)).sum())
    out['wrong_pixels(|consensus err|>0.25)'] = wrong
    out['of_which_committed(std<0.05)'] = committed
    return out


def pairwise(world, corners, qh, truth_world, acc):
    """Accumulate, over overlapping query pairs of one (episode, t), the
    squared disagreement and the pair's squared errors on shared pixels,
    by heading difference (0, 90, 180 deg) and Chebyshev offset."""
    dev = world.device
    cy, cx = corners[:, 0], corners[:, 1]
    y0, x0 = cy.min(), cx.min()
    Hc, Wc = cy.max() - y0 + 1, cx.max() - x0 + 1
    G = torch.full((4, Hc, Wc, S, S), torch.nan, device=dev)
    E = torch.full((4, Hc, Wc, S, S), torch.nan, device=dev)
    yi, xi = torch.from_numpy(cy - y0).to(dev), torch.from_numpy(cx - x0).to(dev)
    hi = torch.from_numpy(qh).to(dev)
    G[hi, yi, xi] = world
    E[hi, yi, xi] = world - truth_world
    for dy in range(-(S - 1), S):
        for dx in range(-(S - 1), S):
            ya, yb = max(0, -dy), min(Hc, Hc - dy)
            xa, xb = max(0, -dx), min(Wc, Wc - dx)
            if ya >= yb or xa >= xb:
                continue
            ra, rb, ca, cb = max(0, dy), min(S, S + dy), max(0, dx), min(S, S + dx)
            cheb = max(abs(dy), abs(dx))
            for h1 in range(4):
                A = G[h1, ya:yb, xa:xb, ra:rb, ca:cb]
                Ae = E[h1, ya:yb, xa:xb, ra:rb, ca:cb]
                for h2 in range(4):
                    if dy == 0 and dx == 0 and h1 == h2:
                        continue
                    B = G[h2, ya + dy:yb + dy, xa + dx:xb + dx, ra - dy:rb - dy, ca - dx:cb - dx]
                    Be = E[h2, ya + dy:yb + dy, xa + dx:xb + dx, ra - dy:rb - dy, ca - dx:cb - dx]
                    d = (A - B) ** 2
                    key = (min((h1 - h2) % 4, (h2 - h1) % 4) * 90, cheb)
                    a = acc.setdefault(key, [0.0, 0.0, 0.0])      # GPU tensors: no syncs
                    a[0] = a[0] + torch.nansum(d)
                    a[1] = a[1] + torch.nansum(Ae ** 2 + Be ** 2)
                    a[2] = a[2] + (~torch.isnan(d)).sum()


def pair_table(acc):
    out = {}
    acc = {k: [float(x) for x in v] for k, v in acc.items()}
    acc = {k: v for k, v in acc.items() if v[2] > 0}
    for (dh, cheb), (msd, e2, n) in sorted(acc.items()):
        out[f'dh{dh}_off{cheb}'] = dict(msd=msd / n, one_minus_err_corr=msd / max(e2, 1e-12),
                                         shared_pixels=int(n))
    for dh in (0, 90, 180):
        tot = [sum(v[i] for (h, _), v in acc.items() if h == dh) for i in range(3)]
        if tot[2]:
            out[f'dh{dh}_all'] = dict(msd=tot[0] / tot[2], one_minus_err_corr=tot[0] / tot[1],
                                      shared_pixels=int(tot[2]))
    return out


# --------------------------------------------------------------------- visuals
def canvas(stats, key):
    H, W = stats['shape']
    v = stats[key].clone()
    if key == 'var':
        v = v.sqrt()
    if key == 'err':
        v = v.abs()
    return v.view(H, W).cpu().numpy()


def mosaic(path, ep, s_inert, s_mixed, t, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    H, W = s_inert['shape']
    seen = s_inert['covered'].view(H, W).cpu().numpy()
    truth = np.where(seen, ep['image'], np.nan)
    panels = [('truth (seen by t)', truth, 'gray', 0, 1),
              ('inert, one pass (consensus)', canvas(s_inert, 'mean'), 'gray', 0, 1),
              ('inert disagreement (std)', canvas(s_inert, 'var'), 'magma', 0, 0.3),
              ('inert |consensus error|', canvas(s_inert, 'err'), 'magma', 0, 0.6),
              ('mixed, chunks of 8 (consensus)', canvas(s_mixed, 'mean'), 'gray', 0, 1),
              ('mixed disagreement (std)', canvas(s_mixed, 'var'), 'magma', 0, 0.3)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
    for ax, (name, img, cmap, lo, hi) in zip(axes, panels):
        cm = matplotlib.colormaps[cmap].copy()
        cm.set_bad('#cfe3f2')
        ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation='nearest')
        ax.set_title(name, fontsize=8)
        ax.axis('off')
        y, x = ep['positions'][t]
        ax.add_patch(matplotlib.patches.Rectangle((x - .5, y - .5), S, S, fill=False, ec='#e4572e', lw=1))
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def gif(path, ep, frames, scale=3):
    """frames: list of (t, consensus (H, W) with nan, std (H, W))."""
    from PIL import Image, ImageDraw
    import matplotlib
    magma = matplotlib.colormaps['magma']
    imgs = []
    for t, mean, std in frames:
        seen = ~np.isnan(mean)
        bg = np.array([207, 227, 242], np.float32) / 255
        def gray(a):
            rgb = np.repeat(np.nan_to_num(a)[..., None], 3, -1)
            return np.where(seen[..., None], rgb, bg)
        truth = gray(np.where(seen, ep['image'], np.nan))
        belief = gray(np.clip(mean, 0, 1))
        sd = np.where(seen[..., None], magma(np.clip(np.nan_to_num(std) / 0.3, 0, 1))[..., :3], bg)
        sep = np.ones((mean.shape[0], 2, 3))
        row = np.concatenate([truth, sep, belief, sep, sd], 1)
        im = Image.fromarray((row * 255).astype(np.uint8)).resize(
            (row.shape[1] * scale, row.shape[0] * scale), Image.NEAREST)
        d = ImageDraw.Draw(im)
        y, x = ep['positions'][t]
        W = mean.shape[1] + 2
        for k in range(3):
            d.rectangle([(x + k * W) * scale, y * scale, (x + k * W + S) * scale - 1, (y + S) * scale - 1],
                        outline=(228, 87, 46))
        d.text((4, 2), f't = {t}   truth | inert belief | disagreement', fill=(228, 87, 46))
        imgs.append(im)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=350, loop=0)


# ------------------------------------------------------------------------ main
def reference_scan(q, k, v, g, beta):
    b, l, h, dk = k.shape
    Sx = q.new_zeros(b, h, dk, v.shape[-1])
    out = []
    for t in range(l):
        Sx = Sx * g[:, t].exp().to(Sx.dtype)[..., None, None]
        kS = torch.einsum('bhk,bhkv->bhv', k[:, t], Sx)
        Sx = Sx + torch.einsum('bhk,bhv->bhkv', k[:, t] * beta[:, t, :, None].to(Sx.dtype), v[:, t] - kS)
        out.append(torch.einsum('bhk,bhkv->bhv', q[:, t], Sx))
    return torch.stack(out, 1)


def exact_scan(q, k, v, g, beta, chunk=4096):
    """The gated delta rule exactly as ``reference_scan`` (the sequential
    recurrence), but stepping only through the tokens that change the
    state. A token with beta = 0 and g = 0 (an inert query) leaves the
    state unchanged, so its output is the state after the writers before
    it, read by its q. Per (row, head) the writers are found, the recurrence
    runs over them, and every token reads its state."""
    b, l, h, dk = k.shape
    dv = v.shape[-1]
    fold = lambda x: x.transpose(1, 2).reshape(b * h, l, *x.shape[3:])   # (b*h, l, ...)
    q, k, v, g, beta = fold(q), fold(k), fold(v), fold(g), fold(beta)
    writer = (beta != 0) | (g != 0)                                       # (bh, l)
    n = writer.sum(1)
    assert (n == n[0]).all(), 'rows with different writer counts'
    n = int(n[0])
    widx = torch.argsort((~writer).to(torch.int8), dim=1, stable=True)[:, :n]  # writers, in order
    gk = lambda x: torch.gather(x, 1, widx.view(*widx.shape, *([1] * (x.dim() - 2))).expand(-1, -1, *x.shape[2:]))
    kw, vw, gw, bw = gk(k), gk(v), gk(g), gk(beta)
    Sx = q.new_zeros(b * h, dk, dv)
    states = [Sx]
    for t in range(n):
        Sx = Sx * gw[:, t].exp()[:, None, None]
        kS = torch.einsum('bk,bkv->bv', kw[:, t], Sx)
        Sx = Sx + torch.einsum('bk,bv->bkv', kw[:, t] * bw[:, t, None], vw[:, t] - kS)
        states.append(Sx)
    states = torch.stack(states, 1)                                       # (bh, n+1, dk, dv)
    j = writer.long().cumsum(1)                                           # state index per token
    out = torch.empty(b * h, l, dv, device=q.device, dtype=q.dtype)
    for i in range(0, l, chunk):
        St = torch.gather(states, 1, j[:, i:i + chunk, None, None].expand(-1, -1, dk, dv))
        out[:, i:i + chunk] = torch.einsum('blk,blkv->blv', q[:, i:i + chunk], St)
    return out.view(b, h, l, dv).transpose(1, 2)


def commit():
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True).stdout.strip()
    return dict(head=head, dirty_tracked_files=dirty.splitlines())


def kernel_vs_exact(p, nets, ep, T):
    """Max |kernel - exact| on the dense set at T (inert one pass; mixed in
    chunks of 8), per model."""
    from mapping.models.se2d.hybrid4 import GatedDeltaNet
    corners, qh, qt = dense_queries(ep, [T], p.headings, p.stride)
    inputs = model_inputs(ep, corners, qh, qt, p.device)
    out = {}
    for name, net, chunk in (('inert', nets[0], None), ('mixed', nets[1], p.mixed_chunk)):
        mods = [m for m in net.modules() if isinstance(m, GatedDeltaNet)]
        saved = [m.scan for m in mods]
        for m in mods:
            m.scan = exact_scan
        ex = predict(net, inputs, chunk)
        for m in mods:
            m.scan = GatedDeltaNet.scan.__get__(m)
        predict(net, inputs, chunk)                       # warm
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        base, t0 = torch.cuda.memory_allocated(), time.time()
        ke = predict(net, inputs, chunk)
        torch.cuda.synchronize()
        kt, kmem = time.time() - t0, (torch.cuda.max_memory_allocated() - base) / 2**20
        for m, f in zip(mods, saved):
            m.scan = f
        d = (ke - ex).abs()
        out[name] = dict(max_abs=d.max().item(), mean_abs=d.mean().item(),
                         queries_over_0_01=int((d.amax((-1, -2)) > 0.01).sum()), queries=len(qt),
                         kernel_forward_s=kt, kernel_peak_mib=kmem)
    return out


def main():
    p = compose_probe()
    os.makedirs(p.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = p.device
    inert, iargs, spec, task = load_checkpoint(p.inert, dev)
    mixed, margs, _, _ = load_checkpoint(p.mixed, dev)
    assert getattr(inert, 'query_mode', 'mixed') == 'inert' and getattr(mixed, 'query_mode', 'mixed') == 'mixed'
    config = tasks.config_from_args(task, spec, iargs)
    _, bank = load_banks(iargs.get('bank') or 'data/mnist.npz', config, iargs.get('texture', 'mnist'))
    eps = episodes(p, config, bank)
    T = config.steps
    inert_ck = torch.load(p.inert, map_location='cpu', weights_only=False)
    res = dict(config=asdict(p), code=commit(),
               checkpoints=dict(inert=dict(path=p.inert, step=inert_ck.get('step'),
                                           run_id=(inert_ck.get('provenance') or {}).get('run_id')),
                                mixed=dict(path=p.mixed, run_id=p.mixed_run_id)),
               note='fp32 eval, TF32 off; inert one pass per (episode, time set), mixed in '
                    'chunks of 8 (queries shuffled, seed 0)')
    del inert_ck
    from mapping.models.se2d.hybrid4 import GatedDeltaNet
    if p.scan == 'exact':
        for net in (inert, mixed):
            for m in net.modules():
                if isinstance(m, GatedDeltaNet):
                    m.scan = exact_scan
    # the fused kernel's deviation from the exact recurrence on these models
    res['kernel_vs_exact'] = kernel_vs_exact(p, (inert, mixed), eps[0], T)
    rng = torch.Generator(device='cpu').manual_seed(0)

    def run_set(ep, times, with_mixed):
        corners, qh, qt = dense_queries(ep, times, p.headings, p.stride)
        inputs = model_inputs(ep, corners, qh, qt, dev)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.time()
        one = predict(inert, inputs) if len(qt) <= p.max_queries else None
        torch.cuda.synchronize()
        timing = dict(queries=len(qt), tokens=int(inputs[0].shape[1] * 10 - 1 + 9 * len(qt)))
        if one is not None:
            timing.update(one_pass_s=time.time() - t0,
                          one_pass_peak_mib=(torch.cuda.max_memory_allocated() - base) / 2**20)
        t0 = time.time()
        chunked = predict(inert, inputs, p.check_chunk)
        torch.cuda.synchronize()
        timing['chunked_s'] = time.time() - t0
        if one is None:
            one = chunked
        else:
            timing['one_pass_vs_chunked_max_abs'] = (one - chunked).abs().max().item()
        qh_t = torch.from_numpy(qh).to(dev)
        out = dict(corners=corners, qh=qh, qt=qt, timing=timing, inert=to_world(one, qh_t))
        if with_mixed:
            order = torch.randperm(len(qt), generator=rng).to(dev)
            t0 = time.time()
            out['mixed'] = to_world(predict(mixed, inputs, p.mixed_chunk, order), qh_t)
            timing['mixed_chunked_s'] = time.time() - t0
        return out

    def analyse(ep, r, t_values, accs, stats, pairs=True):
        img = torch.from_numpy(ep['image']).to(dev)
        r_ = torch.arange(S, device=dev)
        for t in t_values:
            m = r['qt'] == t
            c, h = r['corners'][m], r['qh'][m]
            cy, cx = torch.from_numpy(c[:, 0]).to(dev), torch.from_numpy(c[:, 1]).to(dev)
            truth_w = img[cy[:, None, None] + r_[None, :, None], cx[:, None, None] + r_[None, None, :]]
            mt = torch.from_numpy(m).to(dev)
            for which in ('inert', 'mixed'):
                if which not in r:
                    continue
                w = r[which][mt]
                stats[which].setdefault(t, []).append(pixel_stats(ep, w, c, t, dev))
                if pairs:
                    pairwise(w, c, h, truth_w, accs[which].setdefault(t, {}))

    # warm-up: compile the flex kernels outside the timings
    w = run_set_inputs = model_inputs(eps[0], *dense_queries(eps[0], [T], p.headings, p.stride), dev)
    predict(inert, w); predict(inert, w, p.check_chunk); predict(mixed, w, p.mixed_chunk)
    torch.cuda.synchronize()
    del run_set_inputs

    # ---- 1, 2, 4: t = T, one pass per episode; mixed in chunks of 8
    stats = dict(inert={}, mixed={})
    accs = dict(inert={}, mixed={})
    timings = []
    for i, ep in enumerate(eps):
        r = run_set(ep, [T], True)
        timings.append(r['timing'])
        analyse(ep, r, [T], accs, stats)
        mosaic(os.path.join(p.out, f'mosaic_T_ep{i:02d}.png'), ep, stats['inert'][T][-1],
               stats['mixed'][T][-1], T,
               f'episode {i}, t = T = {T}: {len(r["qt"])} queries (inert in one pass)')
        print(f'ep {i}: {r["timing"]}', flush=True)
    res['scale_T'] = timings
    # 3: intermediate times, mixed in one pass per episode
    timings_mid = []
    for i, ep in enumerate(eps):
        r = run_set(ep, p.mid_times, True)
        timings_mid.append(r['timing'])
        analyse(ep, r, p.mid_times, accs, stats)
        for t in (p.mid_times[len(p.mid_times) // 2],):
            mosaic(os.path.join(p.out, f'mosaic_t{t}_ep{i:02d}.png'), ep, stats['inert'][t][-1],
                   stats['mixed'][t][-1], t, f'episode {i}, t = {t} (one pass mixing {len(p.mid_times)} times)')
        print(f'ep {i} mid: {r["timing"]}', flush=True)
    res['scale_mid_times'] = timings_mid
    res['consistency'] = {which: {f't{t}': dict(pixels=summarise(stats[which][t]),
                                                 pairs=pair_table(accs[which][t]))
                                  for t in sorted(stats[which])}
                          for which in ('inert', 'mixed')}

    # ---- one pass vs chunked on the sequential reference scan (exactness)
    saved = {}
    for m in inert.modules():
        if isinstance(m, GatedDeltaNet):
            saved[m] = m.scan
            m.scan = reference_scan
    ep = eps[0]
    corners, qh, qt = dense_queries(ep, [T], p.headings, p.stride)
    pick = np.random.default_rng(0).choice(len(qt), p.reference_queries, replace=False)
    inputs = model_inputs(ep, corners[pick], qh[pick], qt[pick], dev)
    a, b = predict(inert, inputs), predict(inert, inputs, 8)
    res['reference_scan_one_pass_vs_chunks_of_8_max_abs'] = (a - b).abs().max().item()
    for m, f in saved.items():
        m.scan = f

    # ---- 5: time-lapse of the inert belief, every t
    for i, ep in enumerate(eps[:p.video_episodes]):
        frames = []
        all_t = list(range(T + 1))
        groups, cur, n = [], [], 0
        for t in all_t:                       # split into passes of <= max_queries
            k = len(dense_queries(ep, [t], p.headings, p.stride)[0])
            if cur and n + k > p.max_queries:
                groups.append(cur); cur, n = [], 0
            cur.append(t); n += k
        groups.append(cur)
        for g in groups:
            r = run_set(ep, g, False)
            st = {}
            analyse(ep, r, g, None, dict(inert=st, mixed={}), pairs=False)
            for t in g:
                s = st[t][0]
                frames.append((t, canvas(s, 'mean'), canvas(s, 'var')))
        gif(os.path.join(p.out, f'timelapse_ep{i:02d}.gif'), ep, frames)
        res.setdefault('timelapse_passes', []).append([len(g) for g in groups])

    with open(os.path.join(p.out, 'results.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    print('wrote', p.out)


if __name__ == '__main__':
    main()
