"""hybrid4c_time_full: ``hybrid4c_time`` with its LAST layer a full
bidirectional Gated DeltaNet over ALL tokens in hybrid4c's token order,
exactly hybrid4c's layer, instead of attention. Same layer count.
"""
import sys

from mapping.models.se2c import register
from mapping.models.se2c.hybrid4c_time import (LR, SCALING_RULE,  # noqa: F401
                                               SIZES, Hybrid4cTime)

ARCH = {'last': 'gdn'}   # what this model's config file binds


def build(size, spec):
    return Hybrid4cTime(spec, **SIZES[size], **ARCH)


register('hybrid4c_time_full')(sys.modules[__name__])
