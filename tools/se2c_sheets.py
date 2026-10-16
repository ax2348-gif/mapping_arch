"""Contact sheets for the se2c texture and generator, to LOOK at.

    python tools/se2c_sheets.py [--out docs/lift/sheets]

Writes
  texture.png   world patches of several episodes, and 15x15 views of
                one place at several rotations (all exact renders)
  mockup.png    targets beside imitation predictions: small and large pose
                errors, blurs (what a regressing model draws) and the
                constant floor, with their MSE, so success and failure
                can be told apart at a glance
  episode.png   one episode on its world texture: observation footprints
                (yellow, the walk in red), query footprints (green: found
                by the sampler, magenta: fallback), and the observations
                and targets as the model receives them
  bank.png      the MNIST tile-bank option, antialiased views
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, '.')
from mapping.se2c import geometry as geo  # noqa: E402
from mapping.se2c import texture as tex  # noqa: E402
from mapping.se2c.generator import Config, make_batch  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
BG = 50


def pil(a, scale, mode='L'):
    a = (np.clip(np.asarray(a, np.float32), 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(a)
    return im.resize((a.shape[1] * scale, a.shape[0] * scale), Image.NEAREST)


def world(params, spec, y0, x0, n, b=0):
    r = torch.arange(n, device=DEV).float()
    pts = torch.stack(torch.meshgrid(r + y0, r + x0, indexing='ij'), -1)[None]
    p = {k: v[b:b + 1] for k, v in params.items()}
    return tex.contrast(tex.field(p, pts), spec)[0].cpu().numpy()


def texture_sheet(spec, out):
    g = torch.Generator(device=DEV).manual_seed(1)
    p = tex.sample_params(g, 4, spec, DEV)
    angles = [0, 20, 45, 90, 135, 200]
    can = Image.new('L', (4 * 290 + 10, 300 + 4 * 140 + 30), BG)
    d = ImageDraw.Draw(can)
    d.text((6, 4), 'world patches 128x128 px (x2), four episodes', fill=255)
    for b in range(4):
        can.paste(pil(world(p, spec, 0, 0, 128, b), 2), (6 + b * 290, 20))
    y = 300
    d.text((6, y - 16), f'15x15 views at one place, headings {angles} deg (x8)',
           fill=255)
    for b in range(4):
        pos = torch.full((1, len(angles), 2), 64.0, device=DEV)
        th = torch.tensor([[math.radians(a) for a in angles]], device=DEV)
        v = tex.render({k: x[b:b + 1] for k, x in p.items()}, pos, th, 15, spec)
        for j in range(len(angles)):
            can.paste(pil(v[0, j].cpu(), 8), (6 + j * 130, y + b * 140))
    can.save(out)


def blur(v, sig):
    k = torch.arange(-6, 7, device=v.device).float()
    k = torch.exp(-k ** 2 / 2 / sig ** 2)
    k = k / k.sum()
    x = v[None, None]
    x = F.conv2d(F.pad(x, (6, 6, 0, 0), mode='replicate'), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, 6, 6), mode='replicate'), k.view(1, 1, -1, 1))
    return x[0, 0]


def mockup_sheet(spec, out, rows=6):
    cols = ['target', 'err 0.5px 3deg', 'err 1px 5deg', 'err 2px 10deg',
            'err 4px 30deg', 'blur s1', 'blur s2', 'blur s3', 'floor']
    errs = [(0, 0, 0), (0.35, 0.35, 3), (0.7, 0.7, 5), (1.4, 1.4, 10),
            (2.8, 2.8, 30)]
    cell = 15 * 7 + 8
    can = Image.new('L', (len(cols) * cell + 10, 40 + rows * (cell + 18)), BG)
    d = ImageDraw.Draw(can)
    for j, c in enumerate(cols):
        d.text((6 + j * cell, 6), c, fill=255)
    g = torch.Generator(device=DEV).manual_seed(100)
    p = tex.sample_params(g, rows, spec, DEV)
    for r in range(rows):
        th0 = torch.rand(1, generator=g, device=DEV).item() * 6.28
        pos = torch.tensor([[[60 + a, 60 + b] for a, b, _ in errs]], device=DEV)
        th = torch.tensor([[th0 + math.radians(c) for *_, c in errs]], device=DEV)
        v = tex.render({k: x[r:r + 1] for k, x in p.items()}, pos, th, 15, spec)[0]
        ims = list(v) + [blur(v[0], s) for s in (1, 2, 3)]
        ims.append(torch.full_like(v[0], 0.33))
        y = 24 + r * (cell + 18)
        for j, im in enumerate(ims):
            can.paste(pil(im.cpu(), 7), (6 + j * cell, y))
            d.text((6 + j * cell, y + 15 * 7 + 2),
                   f'mse {((im - v[0]) ** 2).mean().item():.3f}', fill=220)
    can.save(out)


def episode_sheet(config, out, scale=6):
    b = make_batch(config, 0, DEV, batch=2, eval_set=True)
    pos, th = b['poses'][0, :, :2], b['poses'][0, :, 2]
    lo = pos.amin(0) - 14
    hi = pos.amax(0) + 14
    n = int((hi - lo).max().item()) + 1
    w = world(b['texture_params'], config.texture, lo[0].item(), lo[1].item(), n)
    im = pil(w * 0.7, scale).convert('RGB')
    d = ImageDraw.Draw(im)
    half = config.half
    corners = torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]],
                           device=DEV).float() * half

    def poly(p, t, colour, width=1):
        c = p + geo.unrotate(corners, t.expand(4))
        xy = [((x - lo[1].item()) * scale, (y - lo[0].item()) * scale)
              for y, x in c.tolist()]
        d.polygon(xy, outline=colour, width=width)
        f = p + geo.unrotate(torch.tensor([-half, 0.], device=DEV), t)
        d.line([((p[1] - lo[1]).item() * scale, (p[0] - lo[0]).item() * scale),
                ((f[1] - lo[1]).item() * scale, (f[0] - lo[0]).item() * scale)],
               fill=colour, width=width)
    for t in range(pos.shape[0]):
        poly(pos[t], th[t], (230, 200, 40))
    path = [((x - lo[1].item()) * scale, (y - lo[0].item()) * scale)
            for y, x in pos.tolist()]
    d.line(path, fill=(230, 40, 40), width=2)
    qw = b['query_world'][0]
    found = b['stats']['found'][0]
    for q in range(qw.shape[0]):
        poly(qw[q, :2], qw[q, 2], (40, 220, 80) if found[q] else (230, 60, 230), 2)
    S7 = 15 * 4 + 4
    obs = b['observations'][0]
    strip = Image.new('RGB', (max(im.width, 17 * S7), 2 * S7 + 60), (BG,) * 3)
    sd = ImageDraw.Draw(strip)
    sd.text((4, 2), 'observations 0..16 (each facing its heading)', fill=(255,) * 3)
    for t in range(17):
        strip.paste(pil(obs[t].cpu(), 4).convert('RGB'), (4 + t * S7, 16))
    sd.text((4, S7 + 20), 'targets: query t, |dtheta| deg', fill=(255,) * 3)
    for q in range(qw.shape[0]):
        x0 = 4 + q * 2 * S7
        strip.paste(pil(b['target'][0, q].cpu(), 4).convert('RGB'), (x0, S7 + 34))
        dq = abs(math.degrees(b['query_pose'][0, q, 2].item()))
        sd.text((x0 + S7, S7 + 40), f't{int(b["query_time"][0, q])}\n{dq:.0f}',
                fill=(255,) * 3)
    can = Image.new('RGB', (max(im.width, strip.width), im.height + strip.height),
                    (BG,) * 3)
    can.paste(im, (0, 0))
    can.paste(strip, (0, im.height))
    can.save(out)


def bank_sheet(out):
    from mapping.se2c import bank as tb
    c = Config(bank='mnist')
    b = make_batch(c, 0, DEV, batch=2, eval_set=True,
                   bank=tb.load('mnist', 'test', DEV))
    can = Image.new('L', (17 * 64 + 8, 2 * 64 + 24), BG)
    ImageDraw.Draw(can).text((4, 2), 'MNIST bank, aa=4: observations 0..16 '
                             'and 8 targets', fill=255)
    for t in range(17):
        can.paste(pil(b['observations'][0, t].cpu(), 4), (4 + t * 64, 16))
    for q in range(8):
        can.paste(pil(b['target'][0, q].cpu(), 4), (4 + q * 64, 84))
    can.save(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='docs/lift/sheets')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    c = Config()
    texture_sheet(c.texture, os.path.join(a.out, 'texture.png'))
    mockup_sheet(c.texture, os.path.join(a.out, 'mockup.png'))
    episode_sheet(c, os.path.join(a.out, 'episode.png'))
    bank_sheet(os.path.join(a.out, 'bank.png'))
    print(f'wrote {a.out}/texture.png mockup.png episode.png bank.png')


if __name__ == '__main__':
    main()
