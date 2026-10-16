"""High-pass the Imagenette texture bank once: data/imagenette64.npz ->
data/imagenette64_hp.npz (texture ``imagenette_hp``).

    python -I scripts/get_imagenette_hp.py

Why: natural images carry most of their variance at low spatial
frequency, so on the raw bank predicting each episode's mean brightness,
with no geometry at all, already takes the SE(2) visual-odometry pixel
MSE from the constant floor (about 0.070) to about 0.045. Filtering the
bank once, here, removes that shortcut for observations and targets
alike, so the model learns no extra operation.

Input: data/imagenette64.npz (``scripts/get_imagenette.py``), opened read
only and checked against a pinned SHA-256 of its array contents (key,
dtype, shape, bytes; the zip container itself is not byte-reproducible).
That bank is left untouched.

The filter, per 64 x 64 tile, in float64 on 0..255 gray levels:

1. ``v = x - G_sigma * x``, ``G_sigma`` an isotropic Gaussian with
   ``SIGMA`` = 2 px, kernel radius ``round(TRUNCATE * SIGMA)`` = 8,
   normalised to sum 1, applied separably with symmetric (half-sample)
   reflection at the tile border (scipy.ndimage's ``mode='reflect'``).
   Isotropic kernel and a border rule that commutes with flips and
   quarter turns: the filter commutes with the 8 dihedral symmetries the
   generator applies afterwards, so augmentation still removes
   orientation cues.
2. ``y = clip(rint(OFFSET + SCALE * v), 0, 255)`` as uint8, one fixed
   affine map for both splits: ``SCALE = 127.5 / q``, ``q`` the
   ``PERCENTILE`` (99.8th) percentile of ``|v|`` over the TRAIN split, so
   about 0.2% of pixels saturate; no per-image stretching.

Output keys are those of data/imagenette64.npz (same images in the same
order, same labels), plus the filter's identity and parameters:
``texture`` ('imagenette_hp', checked at load time against the texture a
run names, ``mapping.data.check_texture``), ``filter`` (a JSON string of
everything above, including the input content hash) and ``filter_scale``.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'data', 'imagenette64.npz')
OUT = os.path.join(ROOT, 'data', 'imagenette64_hp.npz')
SRC_CONTENT_SHA256 = ('c6d54bf729f1261b1e9fae42d95b0dd3'
                      '989bc9005cd1c58ba115ab85a85b009a')
TEXTURE = 'imagenette_hp'
SIGMA = 2.0
TRUNCATE = 4.0
PERCENTILE = 99.8
OFFSET = 128.0
KEYS = ('test_images', 'test_labels', 'train_images', 'train_labels')


def content_sha256(arrays):
    """SHA-256 over sorted keys, each key's dtype, shape and bytes."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def gaussian_kernel(sigma, truncate):
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur(images, sigma=SIGMA, truncate=TRUNCATE):
    """Separable isotropic Gaussian over the last two axes of ``(N, t, t)``
    float64, symmetric border (``d c b a | a b c d``)."""
    k = gaussian_kernel(sigma, truncate)
    r = len(k) // 2
    out = images
    for axis in (1, 2):
        pad = [(0, 0)] * 3
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode='symmetric')
        n = out.shape[axis]
        acc = np.zeros_like(out)
        for i, w in enumerate(k):
            acc += w * np.take(p, np.arange(i, i + n), axis=axis)
        out = acc
    return out


def highpass(images):
    x = images.astype(np.float64)
    return x - blur(x)


def quantise(v, scale):
    return np.clip(np.rint(OFFSET + scale * v), 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default=SRC)
    p.add_argument('--out', default=OUT)
    a = p.parse_args()
    if os.path.exists(a.out):
        print(f'{a.out} exists')
        return
    with np.load(a.src, allow_pickle=False) as f:
        src = {k: f[k] for k in f.files}
    if tuple(sorted(src)) != KEYS:
        raise SystemExit(f'{a.src}: keys {sorted(src)}, expected {KEYS}')
    digest = content_sha256(src)
    if digest != SRC_CONTENT_SHA256:
        raise SystemExit(f'{a.src}: content sha256 {digest}, expected '
                         f'{SRC_CONTENT_SHA256} (scripts/get_imagenette.py)')
    v_train = highpass(src['train_images'])
    q = float(np.percentile(np.abs(v_train), PERCENTILE))
    scale = 127.5 / q
    out = {'train_images': quantise(v_train, scale),
           'test_images': quantise(highpass(src['test_images']), scale),
           'train_labels': src['train_labels'],
           'test_labels': src['test_labels']}
    params = dict(texture=TEXTURE, source='data/imagenette64.npz',
                  source_content_sha256=digest,
                  filter='x - gaussian(x)', sigma_px=SIGMA,
                  truncate=TRUNCATE, kernel_radius=len(gaussian_kernel(
                      SIGMA, TRUNCATE)) // 2, border='symmetric reflection',
                  quantise='clip(rint(offset + scale * v), 0, 255)',
                  offset=OFFSET, scale=scale, scale_from=(
                      f'127.5 / percentile(|v|, {PERCENTILE}) on train'),
                  percentile=PERCENTILE, percentile_value=q)
    for split in ('train', 'test'):
        im = out[f'{split}_images']
        sat = float(np.mean((im == 0) | (im == 255)))
        print(f'{split}: {im.shape}, mean {im.mean():.2f}, std '
              f'{im.std():.2f}, saturated {sat:.4f}', flush=True)
    print(json.dumps(params, indent=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **out, texture=np.array(TEXTURE),
                        filter=np.array(json.dumps(params, sort_keys=True)),
                        filter_scale=np.array(scale))
    print(f'wrote {a.out}')


if __name__ == '__main__':
    sys.exit(main())
