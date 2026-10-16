"""vanilla at fixed width 256, depth only: 4, 8, 16, 32 layers.

A one-axis ladder for the question the width ladder cannot answer: is
the vanilla transformer depth-starved on this task? Prefix-sum-then-match
over the direction tokens is a computation that grows in sequential depth,
not width. Same module, same tokens, same positions, same audit as
``vanilla``; only ``depth`` is pinned. Residual output projections scale
as 0.02 / sqrt(2 depth) as in the parent, so the deeper rungs are not
disadvantaged at init.

Not a parameter ladder for the power-law fit (parameters grow linearly
with depth); read it as a curve of eval mse against depth.
"""
import sys

from mapping.models import register
from mapping.models.vanilla import VanillaTransformer

WIDTH = 256
from mapping.models import ladder  # noqa: E402
SIZES = ladder('depth')   # conf/ladders/depth.yaml
SCALING_RULE = ('Depth-only family at width 256 (4 heads, hidden 704): '
                'depth 4, 8, 16, 32. Learning rate: harness default 3e-4 at '
                'every rung, as the parent uses at width 256.')


def build(size, spec):
    return VanillaTransformer(spec, **SIZES[size])


register('vanilla_depth')(sys.modules[__name__])
