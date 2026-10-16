"""Contact sheets for the ocean prototype, to LOOK at (docs/ocean/sheets/).

    python tools/ocean_sheets.py [--out docs/ocean/sheets]

world.png      top-down height and shaded albedo of two episodes, with the
               walk and the camera's ground footprint at a few steps
episode.png    one walk's 33 views in order, each facing its heading
queries.png    query targets at offset poses, with the pixels seen by an
               earlier view (causal coverage) overlaid
occlusion.png  views at shallow pitch: image, depth, and the pixels that a
               neighbouring view could not see (occluded or out of frame)
aa.png         no anti-aliasing / footprint prefilter / 4x4 supersampling
params.png     parameter settings side by side, to pick by eye
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '.')
from mapping.ocean import field as fld  # noqa: E402
from mapping.ocean.render import CameraSpec, OceanSpec, render, sample_world, sample_world_seeds, select  # noqa: E402
from mapping.ocean.visibility import coverage, visible  # noqa: E402
from mapping.ocean import viz  # noqa: E402
from mapping.se2c.generator import Config as WalkConfig, walk  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def font(n=13):
    for f in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',):
        if os.path.exists(f):
            return ImageFont.truetype(f, n)
    return ImageFont.load_default()


def label(im, text, size=13, pad=18):
    out = Image.new('RGB', (im.width, im.height + pad), (34, 34, 38))
    out.paste(im.convert('RGB'), (0, pad))
    ImageDraw.Draw(out).text((3, 2), text, fill=(230, 230, 230), font=font(size))
    return out


def grid(ims, cols, gap=6, bg=(34, 34, 38)):
    w = max(i.width for i in ims); h = max(i.height for i in ims)
    rows = math.ceil(len(ims) / cols)
    out = Image.new('RGB', (cols * (w + gap) + gap, rows * (h + gap) + gap), bg)
    for k, im in enumerate(ims):
        out.paste(im.convert('RGB'), (gap + (k % cols) * (w + gap), gap + (k // cols) * (h + gap)))
    return out


def title(im, text):
    out = Image.new('RGB', (im.width, im.height + 26), (24, 24, 28))
    out.paste(im, (0, 26))
    ImageDraw.Draw(out).text((8, 5), text, fill=(240, 240, 240), font=font(15))
    return out


def episode(seed, batch=2):
    g = torch.Generator(device=DEV).manual_seed(seed)
    pos, theta = walk(g, WalkConfig(), batch, DEV)
    return g, pos, theta


def world_sheet(spec, cam, out):
    g, pos, theta = episode(0)
    world = sample_world(g, 2, spec, DEV)
    panels = []
    for b in range(2):
        c = pos[b].mean(0).cpu().numpy()
        n = 220
        y0, x0 = c[0] - n / 2, c[1] - n / 2
        h, alb = viz.topdown(world, spec, b, y0, x0, n, device=DEV)
        hi = viz.height_rgb(h, spec.amp).resize((n * 2, n * 2), Image.NEAREST)
        ai = viz.to_pil(alb, 2).convert('RGB')
        for im in (hi, ai):
            dr = ImageDraw.Draw(im)
            path = [((x - x0) * 2, (y - y0) * 2) for y, x in pos[b].cpu().numpy()]
            dr.line(path, fill=(255, 40, 40), width=2)
            for s in (0, 16, 32):
                fp = viz.footprint(world, spec, cam, pos[b, s], theta[b, s], b)
                viz.draw_poly(dr, fp, y0, x0, 2)
        panels += [label(hi, f'episode {b}: height (blue low, sand high; +-3 sd = {3*spec.amp:.0f} px)'),
                   label(ai, f'episode {b}: shaded albedo from above; walk red, footprints at steps 0/16/32')]
    title(grid(panels, 2), 'Ocean world from above (220 x 220 px around the walk)').save(out)


def episode_sheet(spec, cam, out):
    g, pos, theta = episode(0)
    world = sample_world(g, 2, spec, DEV)
    img = render(world, spec, cam, pos, theta)['image'][0].cpu().numpy()
    ims = [label(viz.to_pil(img[s], 4), f'step {s}  {math.degrees(theta[0, s].item()) % 360:.0f} deg') for s in range(33)]
    title(grid(ims, 11), f'One walk, 33 views (camera {cam.size}px, fov {cam.fov:.0f}, pitch {cam.pitch:.0f}, altitude {cam.altitude:.0f} px)').save(out)


def query_poses(g, pos, theta, t, offset=(6.0, 18.0)):
    """Prototype query sampler (MINE, not a decided design): the anchor pose
    at step t moved by a random offset in [6, 18] px in a random direction,
    with a uniformly random heading."""
    b = pos.shape[0]
    r = offset[0] + torch.rand(b, generator=g, device=DEV) * (offset[1] - offset[0])
    a = torch.rand(b, generator=g, device=DEV) * 2 * math.pi
    qpos = pos[torch.arange(b), t] + torch.stack([r * torch.cos(a), r * torch.sin(a)], -1)
    qth = torch.rand(b, generator=g, device=DEV) * 2 * math.pi
    return qpos, qth


def queries_sheet(spec, cam, out, stats_episodes=16):
    g, pos, theta = episode(1, batch=stats_episodes)
    world = sample_world(g, stats_episodes, spec, DEV)
    poses = torch.cat([pos, theta[..., None]], -1)
    rows, fracs = [], {}
    for t in (4, 12, 24, 32):
        tt = torch.full((stats_episodes,), t, device=DEV, dtype=torch.long)
        qpos, qth = query_poses(g, pos, theta, tt)
        r = render(world, spec, cam, qpos[:, None], qth[:, None])
        cov = coverage(world, spec, cam, r['points'][:, 0], poses, t)
        hit = r['hit'][:, 0]
        cov = cov | ~hit                      # sky: determined by the ray direction alone
        fracs[t] = ((cov & hit).float().sum((1, 2)) / hit.float().sum((1, 2)).clamp(min=1)).cpu().numpy()
        for b in range(2):
            im = r['image'][b, 0].cpu().numpy()
            rgb = np.repeat(im[..., None], 3, -1)
            m = ~cov[b].cpu().numpy()
            rgb[m] = rgb[m] * 0.45 + np.array([0.85, 0.15, 0.1]) * 0.55
            rows.append(label(Image.fromarray((rgb * 255).astype(np.uint8)).resize((128, 128), Image.NEAREST),
                              f't={t}: {100*fracs[t][b]:.0f}% of surface seen'))
    sheet = title(grid(rows, 8), 'Query targets (anchor step t, offset 6-18 px, random heading); red = surface not seen by any view <= t')
    sheet.save(out)
    return fracs


def occlusion_sheet(spec, cam, out):
    """For views at shallow pitch, a companion view at the same heading moved
    20 px BEHIND: which of this view's pixels can it see? Blue = out of its
    frame, red = in its frame but hidden behind a crest."""
    from mapping.ocean.visibility import project
    cam2 = cam
    g, pos, theta = episode(2)
    world = sample_world(g, 2, spec, DEV)
    w0 = select(world, slice(0, 1))
    r = render(world, spec, cam2, pos, theta)
    ims, fr = [], []
    for s in (6, 14, 22, 30):
        img = r['image'][0, s].cpu().numpy()
        dep = r['depth'][0, s].cpu().numpy()
        pts = r['points'][0:1, s].reshape(1, -1, 3)
        th = theta[0:1, s]
        fwd = torch.stack([-torch.cos(th), torch.sin(th)], -1)
        cpos = pos[0:1, s] - 20.0 * fwd
        _, _, inside = project(pts, cpos, th, cam2)
        vis = visible(w0, spec, cam2, pts, cpos, th)
        inside = inside[0].reshape(cam.size, cam.size).cpu().numpy()
        vis = vis[0].reshape(cam.size, cam.size).cpu().numpy()
        hit = r['hit'][0, s].cpu().numpy()
        occ = inside & ~vis & hit
        fr.append(occ.sum() / max((inside & hit).sum(), 1))
        rgb = np.repeat(img[..., None], 3, -1)
        rgb[~inside] = rgb[~inside] * 0.5 + np.array([0.15, 0.35, 0.9]) * 0.5
        rgb[occ] = rgb[occ] * 0.4 + np.array([0.95, 0.15, 0.1]) * 0.6
        ims += [label(viz.to_pil(img, 4), f'step {s}: view'),
                label(viz.to_pil(np.log(dep), 4, np.log(dep).min(), np.log(spec.tmax)), 'log depth (sky white)'),
                label(Image.fromarray((rgb * 255).astype(np.uint8)).resize((128, 128), Image.NEAREST),
                      f'hidden: {100 * fr[-1]:.0f}% of surface')]
    title(grid(ims, 6), 'Occlusion at the default camera: seen from 20 px further back (same heading). Blue out of frame, red hidden by a crest').save(out)
    return fr


def aa_sheet(spec, cam, out):
    from dataclasses import replace
    g, pos, theta = episode(3)
    world = sample_world(g, 2, spec, DEV)
    nofog = replace(spec, fog=0.0)
    ims = []
    for s in (3, 11, 19):
        p, t = pos[:, s:s + 1], theta[:, s:s + 1]
        truth = render(world, spec, cam, p, t, supersample=4)['image'][0, 0].cpu().numpy()
        truth_nf = render(world, nofog, cam, p, t, supersample=4)['image'][0, 0].cpu().numpy()
        a = render(world, nofog, cam, p, t, aa=False)['image'][0, 0].cpu().numpy()
        b = render(world, nofog, cam, p, t, aa=True)['image'][0, 0].cpu().numpy()
        c = render(world, spec, cam, p, t, aa=True)['image'][0, 0].cpu().numpy()
        ims += [label(viz.to_pil(a, 4), f'none: {np.abs(a - truth_nf).mean():.3f}'),
                label(viz.to_pil(b, 4), f'footprint: {np.abs(b - truth_nf).mean():.3f}'),
                label(viz.to_pil(c, 4), f'footprint+haze: {np.abs(c - truth).mean():.3f}'),
                label(viz.to_pil(truth, 4), '4x4 supersample+haze')]
    title(grid(ims, 4), 'Anti-aliasing with the horizon in view; number = mean |x - 4x4 supersampled| (same haze setting)').save(out)


def horizon_sheet(spec, cam, out):
    """Pitch / fov alternatives, each with and without haze."""
    from dataclasses import replace
    g, pos, theta = episode(7)
    world = sample_world(g, 2, spec, DEV)
    rows = []
    for pitch, fov, alt in ((24.0, 60.0, 24.0), (25.0, 60.0, 24.0), (26.0, 60.0, 24.0),
                            (25.0, 60.0, 18.0), (20.0, 60.0, 24.0)):
        c = replace(cam, pitch=pitch, fov=fov, altitude=alt)
        for fog in (spec.fog, 0.0):
            sp = replace(spec, fog=fog)
            img = render(world, sp, c, pos[:, ::6], theta[:, ::6])['image'][0].cpu().numpy()
            strip = grid([viz.to_pil(img[k], 3) for k in range(img.shape[0])], img.shape[0], gap=3)
            rows.append(label(strip, f'pitch {pitch:.0f}, altitude {alt:.0f} (horizon row {c.horizon_row:.1f} of {c.size}), '
                                     + (f'haze {spec.haze}' if fog else 'no haze'), size=13, pad=19))
    title(grid(rows, 2), 'Camera: pitch 24/25/26 at altitude 24, pitch 25 at altitude 18 (low), old pitch 20; with haze (left) and without (right)').save(out)


SETTINGS = [
    ('A calm: amp 2', dict(amp=2.0), dict()),
    ('B default: amp 5, pitch 25, altitude 24, late-elevation haze', dict(), dict()),
    ('C rough: amp 8, altitude 36', dict(amp=8.0), dict(altitude=36.0)),
    ('D low camera: altitude 18', dict(), dict(altitude=18.0)),
    ('E high horizon: pitch 20', dict(), dict(pitch=20.0)),
    ('F default, no shading, no haze', dict(shading=False, fog=0.0), dict()),
    ('G short waves (f 0.015-0.04)', dict(height=fld.WaveSpec(32, 0.015, 0.04)), dict()),
    ('H swell (anisotropy 0.8)', dict(height=fld.WaveSpec(32, 0.008, 0.025, anisotropy=0.8)), dict()),
]


def setting(name):
    for n, o, c in SETTINGS:
        if n == name:
            return OceanSpec(**o), CameraSpec(**c)
    raise KeyError(name)


def params_sheet(out):
    ims = []
    for name, o, c in SETTINGS:
        spec, cam = OceanSpec(**o), CameraSpec(**c)
        g, pos, theta = episode(4)
        world = sample_world(g, 2, spec, DEV)
        img = render(world, spec, cam, pos[:, ::4], theta[:, ::4])['image'][0].cpu().numpy()
        strip = grid([viz.to_pil(img[s], 3) for s in range(img.shape[0])], img.shape[0], gap=3)
        ims.append(label(strip, name, size=14, pad=20))
    title(grid(ims, 1), 'Settings side by side: every 4th view of the same walk and seed').save(out)


REGULARITY = (0.0, 0.33, 0.66, 1.0)

HAZES = [
    ('exp (old default): exp(-t/320)', dict(haze='exp', fog=320.0)),
    ('late exp: clear to 100, then exp(-(t-100)/220)', dict(haze='late', haze_start=100.0, fog=220.0)),
    ('smoothstep 100 -> 500', dict(haze='smooth', haze_start=100.0, haze_end=500.0)),
    ('exp-squared: exp(-(t/380)^2)', dict(haze='exp2', fog=380.0)),
    ('elevation: thicker near the horizon line', dict(haze='elevation', fog=200.0)),
    ('RECOMMENDED late elevation: clear to 60, then elevation-weighted, 120', dict(haze='late_elevation', fog=120.0, haze_start=60.0)),
    ('contrast only: strokes fade to local mean, 100 -> 450', dict(haze='contrast', haze_start=100.0, haze_end=450.0)),
    ('band-matched: fades as the AA filter loses the pattern', dict(haze='band')),
    ('none (aliasing reference)', dict(fog=0.0)),
]


def haze_curve(spec, cam, w=150, h=96):
    """Kept fraction vs range along the centre ray (0..600 px)."""
    from mapping.ocean.render import haze_visibility
    t = torch.linspace(0, 600, w, device=DEV)
    d = torch.zeros(w, 3, device=DEV); d[:, 2] = -math.sin(math.radians(cam.pitch))
    if not spec.fog:
        k = torch.ones(w, device=DEV)
    elif spec.haze == 'band':
        k = None
    else:
        k = haze_visibility(spec, t, d)
    im = Image.new('RGB', (w, h), (18, 18, 22)); dr = ImageDraw.Draw(im)
    for x in (100, 200, 300, 400, 500):
        dr.line([(x / 600 * w, 0), (x / 600 * w, h)], fill=(45, 45, 52))
    if k is None:
        dr.text((6, h / 2 - 7), 'set per pixel by', fill=(200, 200, 200), font=font(11))
        dr.text((6, h / 2 + 6), 'the AA footprint', fill=(200, 200, 200), font=font(11))
    else:
        pts = [(i, (1 - k[i].item()) * (h - 4) + 2) for i in range(w)]
        dr.line(pts, fill=(120, 200, 255), width=2)
    dr.text((3, h - 13), '0', fill=(150, 150, 150), font=font(9)); dr.text((w - 22, h - 13), '600px', fill=(150, 150, 150), font=font(9))
    return im


def haze_sheet(cam, out):
    from dataclasses import replace
    g, pos, theta = episode(9)
    rows = []
    for name, kw in HAZES:
        spec = replace(OceanSpec(), **kw)
        world = sample_world(torch.Generator(device=DEV).manual_seed(21), 2, spec, DEV)
        img = render(world, spec, cam, pos[:, ::5], theta[:, ::5])['image'][0].cpu().numpy()
        views = grid([viz.to_pil(img[k], 3) for k in range(img.shape[0])], img.shape[0], gap=3)
        curve = haze_curve(spec, cam, h=views.height)
        panel = Image.new('RGB', (curve.width + 8 + views.width, views.height), (34, 34, 38))
        panel.paste(curve, (0, 0)); panel.paste(views, (curve.width + 8, 0))
        rows.append(label(panel, name, size=13, pad=19))
    title(grid(rows, 1), 'Haze techniques (pitch 25, altitude 24): kept fraction vs distance (left), every 5th view (right)').save(out)





def regularity_sheet(cam, out, height_seed=11, albedo_seed=12):
    """The regularity dial, same height and albedo seeds throughout."""
    from dataclasses import replace
    g, pos, theta = episode(8)
    rows = []
    for r in REGULARITY:
        spec = replace(OceanSpec(), height=fld.regular(r))
        world = sample_world_seeds(height_seed, albedo_seed, 2, spec, DEV)
        n = 200
        c = pos[0].mean(0).cpu().numpy()
        h, _ = viz.topdown(world, spec, 0, c[0] - n / 2, c[1] - n / 2, n, device=DEV)
        top = viz.height_rgb(h, spec.amp).resize((150, 150), Image.NEAREST)
        img = render(world, spec, cam, pos[:, ::8], theta[:, ::8])['image'][0].cpu().numpy()
        views = grid([viz.to_pil(img[k], 4) for k in range(img.shape[0])], img.shape[0], gap=3)
        panel = Image.new('RGB', (150 + 8 + views.width, max(150, views.height)), (34, 34, 38))
        panel.paste(top, (0, 0)); panel.paste(views, (158, 0))
        rows.append(label(panel, f'regularity {r:.2f}: band {spec.height.f_lo:.4f}-{spec.height.f_hi:.4f} cyc/px, '
                                 f'directions within +-{180 * (1 - spec.height.anisotropy):.0f} deg   '
                                 f'(left: height 200x200 px; right: every 8th view)', size=13, pad=19))
    title(grid(rows, 1), f'Regularity dial: height seed {height_seed}, albedo seed {albedo_seed} fixed; only the height spectrum changes').save(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='docs/ocean/sheets')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spec, cam = OceanSpec(), CameraSpec()
    world_sheet(spec, cam, f'{a.out}/world.png')
    episode_sheet(spec, cam, f'{a.out}/episode.png')
    fr = queries_sheet(spec, cam, f'{a.out}/queries.png')
    for t, f in fr.items():
        print(f'coverage at t={t}: mean {f.mean():.2f}, min {f.min():.2f}, max {f.max():.2f}')
    print('occluded fraction of in-frame surface (default camera, companion 20 px behind):', [round(float(x), 3) for x in occlusion_sheet(spec, cam, f'{a.out}/occlusion.png')])
    aa_sheet(spec, cam, f'{a.out}/aa.png')
    params_sheet(f'{a.out}/params.png')
    horizon_sheet(spec, cam, f'{a.out}/horizon.png')
    regularity_sheet(cam, f'{a.out}/regularity.png')
    haze_sheet(cam, f'{a.out}/haze.png')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
