"""Draw sampled predictions beside their targets.

One row per sampled held-out query: the context's last observation, the
target, the prediction, the prediction when the episode's directions are
permuted in time, and the absolute error. The shuffled column is the
visual form of the harness diagnostic: if it matches the prediction, the
model is not using the odometry. Row labels give the query's offset from
the last window and the per-query MSE.

``train.py`` writes one of these as ``grid.png`` in every run directory;
``render.py`` draws one from any checkpoint.
"""
import numpy as np
import torch
from PIL import Image, ImageDraw

from mapping.task import prepare

SCALE = 8          # output pixels per cell
GAP = 6
LABEL_W = 150
COLUMNS = ('last obs', 'target', 'prediction', 'dirs shuffled', '|error|')


def _to_image(x):
    a = (np.clip(np.asarray(x, dtype=np.float32), 0, 1) * 255).astype(
        np.uint8)
    return Image.fromarray(a).resize((a.shape[1] * SCALE,
                                      a.shape[0] * SCALE), Image.NEAREST)


@torch.no_grad()
def render_grid(model, batches, view, out, rows=12, seed=0, device=None,
                autocast=None):
    """Write a grid of ``rows`` sampled queries from ``batches`` to ``out``.

    model    the module, in eval mode on ``device``
    batches  list of loader batches (CPU tensors), e.g. the held-out set
    view     window side in pixels
    """
    device = device or next(model.parameters()).device
    autocast = autocast or torch.autocast('cuda', enabled=False)
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(0)
    samples = []
    for b in batches:
        obs, directions, offset, target = prepare(b, device)
        with autocast:
            pred = model(obs, directions, offset).float()
            perm = torch.argsort(torch.rand(directions.shape, generator=gen),
                                 dim=1).to(device)
            shuffled = model(obs, torch.gather(directions, 1, perm),
                             offset).float()
        for i in range(obs.shape[0]):
            for q in range(offset.shape[1]):
                samples.append((obs[i, -1].cpu(), target[i, q].cpu(),
                                pred[i, q].cpu(), shuffled[i, q].cpu(),
                                offset[i, q].cpu()))
    if was_training:
        model.train()

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(samples), size=min(rows, len(samples)),
                       replace=False)
    cell = view * SCALE
    width = LABEL_W + len(COLUMNS) * (cell + GAP)
    height = 20 + len(picks) * (cell + GAP)
    canvas = Image.new('L', (width, height), 40)
    draw = ImageDraw.Draw(canvas)
    for c, name in enumerate(COLUMNS):
        draw.text((LABEL_W + c * (cell + GAP), 4), name, fill=220)
    for r, k in enumerate(picks):
        last, tgt, prd, shf, off = samples[k]
        y = 20 + r * (cell + GAP)
        mse = (prd - tgt).pow(2).mean().item()
        draw.text((4, y + 4), f'off ({int(off[0]):+d},{int(off[1]):+d})',
                  fill=220)
        draw.text((4, y + 18), f'mse {mse:.3f}', fill=220)
        for c, img in enumerate((last, tgt, prd, shf, (prd - tgt).abs())):
            canvas.paste(_to_image(img), (LABEL_W + c * (cell + GAP), y))
    canvas.save(out)
    return out
