"""hybrid4_time_full: ``hybrid4_time`` with its LAST layer hybrid4's full
bidirectional Gated DeltaNet over ALL tokens (hybrid4's token order)
instead of attention. Same layer count: the ablation of whether a full
linear-attention layer is needed before readout.
"""
import sys

from mapping.models.se2d import register
from mapping.models.se2d.hybrid4 import LR, SCALING_RULE, SIZES  # noqa: F401
from mapping.models.se2d.hybrid4_time import Hybrid4Time

ARCH = {'last': 'gdn'}   # what this model's config file binds


def build(size, spec):
    return Hybrid4Time(spec, **SIZES[size], **ARCH)


register('hybrid4_time_full')(sys.modules[__name__])
