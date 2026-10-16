"""Fetch Imagenette (160 px) once and store it as data/imagenette64.npz,
a natural-image texture bank with about MNIST's pixel count.

    python -I scripts/get_imagenette.py [--raw DIR]

Imagenette is fast.ai's open 10-class subset of ImageNet
(https://github.com/fastai/imagenette). The archive is untrusted data: it
is downloaded into ``--raw`` (its own directory, default
``data/imagenette_raw/``, git-ignored), checked against a pinned SHA-256,
and extracted with ``tarfile``'s ``data`` filter (no absolute paths, no
``..``, no links out of the tree, no device files). Run with ``python -I``
so nothing in the working directory can shadow a module.

Every image is converted the same way:

1. grayscale with ITU-R 601-2 luma (PIL mode ``L``: L = 299 R/1000 +
   587 G/1000 + 114 B/1000),
2. centre-cropped to a square on the shorter side,
3. resized to 64 x 64 with PIL's Lanczos filter (antialiased: its support
   widens with the downscale factor),
4. stored as uint8.

Splits: Imagenette ``train`` -> ``train_*`` (9,469 images), ``val`` ->
``test_*`` (3,925). Labels are class indices 0..9 in sorted synset order.
The keys are those of data/mnist.npz. No brightness cutoff is applied
here or at load time (``mapping.data.BANKS``).

Licensing: the fastai/imagenette repository is Apache-2.0, but the images
are ImageNet's (Flickr and other web sources) and keep ImageNet's terms of
access, non-commercial research and education. Do not redistribute the
npz.
"""
import argparse
import hashlib
import os
import sys
import tarfile
import urllib.request

import numpy as np

URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'
SHA256 = '64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'imagenette64.npz')
SIZE = 64
SPLITS = {'train': 'train', 'val': 'test'}     # Imagenette -> bank key
EXPECTED = {'train': 9469, 'val': 3925}
SUFFIXES = ('.jpeg', '.jpg', '.png')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(raw):
    os.makedirs(raw, exist_ok=True)
    archive = os.path.join(raw, os.path.basename(URL))
    if not os.path.exists(archive):
        print(f'downloading {URL}', flush=True)
        urllib.request.urlretrieve(URL, archive + '.part')
        os.replace(archive + '.part', archive)
    digest = sha256(archive)
    if digest != SHA256:
        raise SystemExit(f'{archive}: sha256 {digest}, expected {SHA256}')
    tree = os.path.join(raw, 'imagenette2-160')
    if not os.path.isdir(tree):
        with tarfile.open(archive) as tar:
            tar.extractall(raw, filter='data')
    return tree


def convert(path):
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert('L')                      # ITU-R 601-2 luma
        w, h = im.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        im = im.crop((left, top, left + s, top + s))
        im = im.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', default=os.path.join(ROOT, 'data',
                                                 'imagenette_raw'),
                   help='directory for the archive and its extraction')
    p.add_argument('--out', default=OUT)
    a = p.parse_args()
    if os.path.exists(a.out):
        print(f'{a.out} exists')
        return
    tree = fetch(a.raw)
    synsets = sorted(d for d in os.listdir(os.path.join(tree, 'train'))
                     if d.startswith('n'))
    if len(synsets) != 10:
        raise SystemExit(f'expected 10 classes, found {synsets}')
    out = {}
    for split, key in SPLITS.items():
        images, labels = [], []
        for label, syn in enumerate(synsets):
            folder = os.path.join(tree, split, syn)
            for name in sorted(os.listdir(folder)):
                if not name.lower().endswith(SUFFIXES):
                    continue
                images.append(convert(os.path.join(folder, name)))
                labels.append(label)
        if len(images) != EXPECTED[split]:
            raise SystemExit(f'{split}: {len(images)} images, expected '
                             f'{EXPECTED[split]}')
        out[f'{key}_images'] = np.stack(images)
        out[f'{key}_labels'] = np.asarray(labels, dtype=np.int64)
        print(f'{split} -> {key}: {out[f"{key}_images"].shape}', flush=True)
    n = sum(len(out[f'{k}_images']) for k in SPLITS.values())
    print(f'{n} images x {SIZE * SIZE} = {n * SIZE * SIZE:,} pixels '
          f'(MNIST: 70,000 x 784 = {70000 * 784:,})')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    sys.exit(main())
