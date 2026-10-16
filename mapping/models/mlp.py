"""Reference floor model: an MLP with no sequence structure at all.

Flattens every observation and direction into one vector, encodes it, and
decodes each query from that code and the query offset. It cannot cheat,
because it is given nothing but the task inputs, and it is not expected
to do well: it exists to test the harness and to mark where "no useful
inductive bias" lands on the scaling plot.
"""
import sys

import torch
import torch.nn as nn

from mapping.models import register

from mapping.models import ladder  # noqa: E402
SIZES = ladder('mlp')   # conf/ladders/mlp.yaml

SCALING_RULE = ('Width doubles per rung; depth grows by one every other '
                'rung. Parameters scale roughly with width squared.')


class FlatMLP(nn.Module):
    def __init__(self, spec, width, depth):
        super().__init__()
        s, t = spec.view, spec.steps
        self.spec = spec
        context = (t + 1) * s * s + t * spec.directions
        layers = [nn.Linear(context, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        self.encoder = nn.Sequential(*layers)
        self.query = nn.Sequential(nn.Linear(width + 2, width), nn.GELU(),
                                   nn.Linear(width, s * s))

    def forward(self, obs, directions, query_offset):
        b, q = query_offset.shape[:2]
        s = self.spec.view
        onehot = nn.functional.one_hot(directions, self.spec.directions)
        context = torch.cat([obs.flatten(1), onehot.flatten(1).float()], 1)
        code = self.encoder(context)                           # (B, W)
        offset = query_offset.float() / self.spec.offset_max  # (B, Q, 2)
        joined = torch.cat([code[:, None].expand(-1, q, -1), offset], -1)
        return self.query(joined).view(b, q, s, s)


def build(size, spec):
    return FlatMLP(spec, **SIZES[size])


register('mlp')(sys.modules[__name__])
