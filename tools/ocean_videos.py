"""Videos of the ocean prototype, to judge the 2.5D structure in motion
(docs/ocean/videos/): parallax, occlusion, how the relief reads.

    python tools/ocean_videos.py [--out docs/ocean/videos]

walk_default    the ego's view along one walk at the default settings, with
                true in-between poses (each step: translate, then turn, as
                the decoupled walk does), beside the top-down map with the
                path and the camera's ground footprint
rotate_default  a pure rotation in place (no parallax: rotation only)
translate_default  a pure straight translation (parallax only)
setting_<X>     one short walk clip per parameter setting of params.png
regularity      2x2: regularity 0 / 0.33 / 0.66 / 1, same seeds, slow translation
evolving        the surface h(x, y, t) evolving, albedo fixed in the world;
                camera still, then a slow drift

Each clip is an MP4 (H.264) and a GIF fallback. Frames are rendered at the
true poses (no frame blending) and upscaled nearest-neighbour.
"""
import argparse
import math
import os
import sys

import imageio.v2 as imageio
from dataclasses import replace
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '.')
sys.path.insert(0, 'tools')
from mapping.ocean.render import CameraSpec, OceanSpec, render, sample_world, sample_world_seeds, select
from mapping.ocean import field as fld  # noqa: E402
from mapping.ocean import viz  # noqa: E402
from mapping.se2c.generator import Config as WalkConfig, walk  # noqa: E402
import ocean_sheets as sheets  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
VIEW_PX = 256          # view panel side (32 px x 8)
MAP_PX = 256


def font(n):
    f = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    return ImageFont.truetype(f, n) if os.path.exists(f) else ImageFont.load_default()


def ease(u):
    """Smoothstep easing: zero velocity at both ends."""
    return u * u * (3 - 2 * u)


def interp_poses(pos, theta, move_frames=12, deg_per_frame=3.0, min_turn_frames=10):
    """True in-between poses, eased: within each step translate (eased over
    ``move_frames``), then turn by the shortest angle at an average
    ``deg_per_frame`` (eased). Each step ends exactly on the walk's pose."""
    P, T, step_of = [pos[0]], [theta[0]], [0]
    for s in range(len(pos) - 1):
        for k in range(1, move_frames + 1):
            u = ease(k / move_frames)
            P.append(pos[s] + (pos[s + 1] - pos[s]) * u); T.append(theta[s]); step_of.append(s)
        dth = (theta[s + 1] - theta[s] + math.pi) % (2 * math.pi) - math.pi
        n = max(min_turn_frames, int(round(abs(math.degrees(float(dth))) / deg_per_frame)))
        for k in range(1, n + 1):
            P.append(pos[s + 1]); T.append(theta[s] + dth * ease(k / n)); step_of.append(s + 1)
    return torch.stack(P), torch.stack(T), step_of


def render_views(world, spec, cam, P, T, chunk=128):
    out = []
    for i in range(0, len(P), chunk):
        out.append(render(world, spec, cam, P[None, i:i + chunk], T[None, i:i + chunk])['image'][0])
    return torch.cat(out).cpu().numpy()


def view_panel(img):
    return viz.to_pil(img, VIEW_PX // img.shape[0]).convert('RGB')


def write(frames, path, fps):
    """MP4 (H.264) and GIF fallback; returns the sizes in MB."""
    h, w = frames[0].shape[:2]
    pad_h, pad_w = (-h) % 16, (-w) % 16
    if pad_h or pad_w:
        frames = [np.pad(f, ((0, pad_h), (0, pad_w), (0, 0))) for f in frames]
    imageio.mimsave(path + '.mp4', frames, fps=fps, codec='libx264', quality=7,
                    macro_block_size=16, ffmpeg_params=['-pix_fmt', 'yuv420p'])
    small = [np.asarray(Image.fromarray(f).resize((f.shape[1] // 2, f.shape[0] // 2), Image.NEAREST)) for f in frames[::2]]
    imageio.mimsave(path + '.gif', small, duration=2 / fps, loop=0)
    return os.path.getsize(path + '.mp4') / 1e6, os.path.getsize(path + '.gif') / 1e6


def walk_clip(spec, cam, out, seed=0, steps=16, fps=30, caption='default'):
    g = torch.Generator(device=DEV).manual_seed(seed)
    pos, theta = walk(g, WalkConfig(), 1, DEV)
    world = sample_world(g, 1, spec, DEV)
    pos, theta = pos[0, :steps + 1], theta[0, :steps + 1]
    P, T, step_of = interp_poses(pos, theta)
    views = render_views(world, spec, cam, P, T)
    c = pos.mean(0).cpu().numpy()
    n = 200
    y0, x0 = c[0] - n / 2, c[1] - n / 2
    h, alb = viz.topdown(world, spec, 0, y0, x0, n, device=DEV)
    base = viz.to_pil(alb).convert('RGB').resize((MAP_PX, MAP_PX), Image.NEAREST)
    sc = MAP_PX / n
    frames = []
    for f in range(len(P)):
        canvas = Image.new('RGB', (VIEW_PX + MAP_PX + 24, max(VIEW_PX, MAP_PX) + 30), (24, 24, 28))
        canvas.paste(view_panel(views[f]), (8, 26))
        m = base.copy(); dr = ImageDraw.Draw(m)
        path = [((x - x0) * sc, (y - y0) * sc) for y, x in P[:f + 1].cpu().numpy()]
        if len(path) > 1:
            dr.line(path, fill=(255, 60, 40), width=2)
        fp = viz.footprint(world, spec, cam, P[f], T[f], 0)
        viz.draw_poly(dr, fp, y0, x0, sc, outline=(60, 200, 255), width=2)
        canvas.paste(m, (VIEW_PX + 16, 26))
        step = step_of[f]
        ImageDraw.Draw(canvas).text((8, 5), f'{caption}  step {step}/{steps}  heading {math.degrees(T[f].item()) % 360:.0f} deg',
                                    fill=(235, 235, 235), font=font(14))
        frames.append(np.asarray(canvas))
    return write(frames, out, fps)


def rotate_clip(spec, cam, out, seed=5, frames=360, fps=30):
    """One full turn in place, eased in and out (about 1 deg per frame)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    world = sample_world(g, 1, spec, DEV)
    P = torch.zeros(frames, 2, device=DEV)
    T = 2 * math.pi * ease(torch.linspace(0, 1, frames, device=DEV))
    views = render_views(world, spec, cam, P, T)
    fr = []
    for f in range(frames):
        canvas = Image.new('RGB', (VIEW_PX + 16, VIEW_PX + 30), (24, 24, 28))
        canvas.paste(view_panel(views[f]), (8, 26))
        ImageDraw.Draw(canvas).text((8, 5), f'pure rotation  {math.degrees(T[f].item()):.0f} deg', fill=(235, 235, 235), font=font(14))
        fr.append(np.asarray(canvas))
    return write(fr, out, fps)


def translate_clip(spec, cam, out, seed=6, frames=240, dist=120.0, fps=30):
    """Straight ahead at constant heading, eased in and out (pure parallax)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    world = sample_world(g, 1, spec, DEV)
    T = torch.zeros(frames, device=DEV)                     # heading 0 faces -y
    u = ease(torch.linspace(0, 1, frames, device=DEV))
    P = torch.stack([-dist * u, torch.zeros(frames, device=DEV)], -1)
    views = render_views(world, spec, cam, P, T)
    fr = []
    for f in range(frames):
        canvas = Image.new('RGB', (VIEW_PX + 16, VIEW_PX + 30), (24, 24, 28))
        canvas.paste(view_panel(views[f]), (8, 26))
        ImageDraw.Draw(canvas).text((8, 5), f'pure translation  {dist * u[f].item():.0f} px forward', fill=(235, 235, 235), font=font(14))
        fr.append(np.asarray(canvas))
    return write(fr, out, fps)


def regularity_clip(cam, out, frames=240, dist=90.0, fps=30, height_seed=11, albedo_seed=12):
    """2x2 panels, regularity 0 / 0.33 / 0.66 / 1, same seeds, same slow
    eased forward translation."""
    from dataclasses import replace
    u = ease(torch.linspace(0, 1, frames, device=DEV))
    P = torch.stack([-dist * u, torch.zeros(frames, device=DEV)], -1)
    T = torch.full((frames,), 0.6, device=DEV)
    panels = []
    for r in sheets.REGULARITY:
        spec = replace(OceanSpec(), height=fld.regular(r))
        world = sample_world_seeds(height_seed, albedo_seed, 1, spec, DEV)
        panels.append((r, render_views(world, spec, cam, P, T)))
    half = VIEW_PX // 2 + 64
    fr = []
    for f in range(frames):
        canvas = Image.new('RGB', (2 * half + 24, 2 * (half + 22) + 8), (24, 24, 28))
        for k, (r, v) in enumerate(panels):
            x, y = 8 + (k % 2) * (half + 8), 4 + (k // 2) * (half + 22)
            canvas.paste(viz.to_pil(v[f], half // v.shape[-1]).convert('RGB'), (x, y + 18))
            ImageDraw.Draw(canvas).text((x, y), f'regularity {r:.2f}', fill=(235, 235, 235), font=font(13))
        fr.append(np.asarray(canvas))
    return write(fr, out, fps)


def evolving_clip(spec, cam, out, frames=300, fps=30, t_end=60.0, seed=13):
    """The surface h(x, y, t) evolving (deep-water dispersion), albedo fixed
    in the world. First half: camera still; second half: slow eased drift
    forward. t runs 0 -> t_end time units (dominant period ~20)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    world = sample_world(g, 1, spec, DEV)
    half = frames // 2
    u = torch.cat([torch.zeros(half, device=DEV), ease(torch.linspace(0, 1, frames - half, device=DEV))])
    P = torch.stack([-30.0 * u, torch.zeros(frames, device=DEV)], -1)
    T = torch.zeros(frames, device=DEV)
    tim = torch.linspace(0, t_end, frames, device=DEV)
    views = []
    for i in range(0, frames, 100):
        views.append(render(world, spec, cam, P[None, i:i + 100], T[None, i:i + 100],
                            time=tim[None, i:i + 100])['image'][0])
    views = torch.cat(views).cpu().numpy()
    fr = []
    for f in range(frames):
        canvas = Image.new('RGB', (VIEW_PX + 16, VIEW_PX + 30), (24, 24, 28))
        canvas.paste(view_panel(views[f]), (8, 26))
        txt = f'evolving sea, t = {tim[f].item():.1f}  ' + ('camera still' if f < half else 'slow drift')
        ImageDraw.Draw(canvas).text((8, 5), txt, fill=(235, 235, 235), font=font(14))
        fr.append(np.asarray(canvas))
    return write(fr, out, fps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='docs/ocean/videos')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spec, cam = OceanSpec(), CameraSpec()
    sizes = {}
    only = os.environ.get('OCEAN_ONLY')
    if only:   # render just these clips (comma list): regularity, evolving
        for k in only.split(','):
            low = CameraSpec(altitude=18.0)
            c_spec, c_cam = sheets.setting('C rough: amp 8, altitude 36')
            sizes[k] = {'regularity': lambda: regularity_clip(cam, f'{a.out}/regularity'),
                        'evolving': lambda: evolving_clip(spec, cam, f'{a.out}/evolving'),
                        'walk_default': lambda: walk_clip(spec, cam, f'{a.out}/walk_default'),
                        'translate_default': lambda: translate_clip(spec, cam, f'{a.out}/translate_default'),
                        'setting_C': lambda: walk_clip(c_spec, c_cam, f'{a.out}/setting_C', seed=4, steps=6,
                                                       caption='C rough: amp 8, altitude 36'),
                        'walk_low': lambda: walk_clip(spec, low, f'{a.out}/walk_low', caption='low camera: altitude 18'),
                        'translate_low': lambda: translate_clip(spec, low, f'{a.out}/translate_low'),
                        'haze_late_elevation': lambda: walk_clip(spec, cam, f'{a.out}/haze_late_elevation', seed=4, steps=6,
                                                                 caption='haze: late elevation (recommended)'),
                        'haze_band': lambda: walk_clip(replace(spec, haze='band'), cam, f'{a.out}/haze_band', seed=4, steps=6,
                                                       caption='haze: band-matched'),
                        'haze_contrast': lambda: walk_clip(replace(spec, haze='contrast', haze_start=100.0, haze_end=450.0),
                                                           cam, f'{a.out}/haze_contrast', seed=4, steps=6,
                                                           caption='haze: contrast only'),
                        'haze_exp_old': lambda: walk_clip(replace(spec, haze='exp', fog=320.0), cam, f'{a.out}/haze_exp_old',
                                                          seed=4, steps=6, caption='haze: old exp(-t/320)')}[k]()
        for k, (m, gsz) in sizes.items():
            print(f'{k:22s} mp4 {m:5.2f} MB   gif {gsz:5.2f} MB')
        return
    sizes['walk_default'] = walk_clip(spec, cam, f'{a.out}/walk_default')
    sizes['rotate_default'] = rotate_clip(spec, cam, f'{a.out}/rotate_default')
    sizes['translate_default'] = translate_clip(spec, cam, f'{a.out}/translate_default')
    for name, o, c in sheets.SETTINGS:
        key = 'setting_' + name.split(':')[0].split(' ')[0]
        sizes[key] = walk_clip(OceanSpec(**o), CameraSpec(**c), f'{a.out}/{key}', seed=4, steps=6, caption=name)
    sizes['regularity'] = regularity_clip(cam, f'{a.out}/regularity')
    sizes['evolving'] = evolving_clip(spec, cam, f'{a.out}/evolving')
    for k, (m, gsz) in sizes.items():
        print(f'{k:22s} mp4 {m:5.2f} MB   gif {gsz:5.2f} MB')


if __name__ == '__main__':
    main()
