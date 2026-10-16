"""Are two checkpoints' model tensors identical? Prints one JSON object.

    python tools/compare_checkpoints.py a/final.pt b/final.pt

``bitwise`` is true when both hold the same parameter names and every
tensor is equal element for element (``torch.equal``); ``max_abs_diff`` is
the largest difference otherwise. ``xp reproduce`` calls this as a
subprocess, so the xp porcelain itself never imports torch.
"""
import json
import sys

import torch


def compare(a, b):
    x = torch.load(a, map_location='cpu', weights_only=False)['model']
    y = torch.load(b, map_location='cpu', weights_only=False)['model']
    same_keys = x.keys() == y.keys()
    worst, equal = 0.0, same_keys
    for k in x:
        if k not in y or x[k].shape != y[k].shape:
            equal = False
            continue
        if not torch.equal(x[k], y[k]):
            equal = False
            worst = max(worst, (x[k].double() - y[k].double()).abs().max().item())
    return {'bitwise': bool(equal), 'same_keys': same_keys, 'max_abs_diff': worst,
            'tensors': len(x)}


if __name__ == '__main__':
    print(json.dumps(compare(sys.argv[1], sys.argv[2])))
