"""Continuous SE2 adapter for hybrid6's shared local-camera implementation.

Audit: raw continuous local odometry/query poses replace the discrete symbol
decoder; every camera, calibration, temporal, and inert-query rule is inherited
from se2d.hybrid6. No changes to the continuous task generator or renderer.
"""
import sys

from mapping.models.se2c import register
from mapping.models.se2d.hybrid6 import ARCH, Hybrid6, LR, SCALING_RULE, SIZES


class Hybrid6c(Hybrid6):
    QUERY_MODES = ('mixed', 'inert')
    SCAN_PRECISIONS = ('tf32', 'ieee')
    continuous = True


def build(size, spec):
    return Hybrid6c(spec, **SIZES[size], **ARCH)


register('hybrid6c')(sys.modules[__name__])
