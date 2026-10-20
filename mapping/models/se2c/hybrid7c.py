"""Continuous local-pose adapter; all Hybrid7 weights and geometry are shared."""
import sys
from mapping.models.se2c import register
from mapping.models.se2d.hybrid7 import Hybrid7, LR, SIZES, SCALING_RULE, ARCH

class Hybrid7c(Hybrid7):
    QUERY_MODES = ('inert',)
    SCAN_PRECISIONS = ('tf32', 'ieee')
    continuous = True

def build(size, spec):
    return Hybrid7c(spec, **SIZES[size], **ARCH, query_mode='inert')

register('hybrid7c')(sys.modules[__name__])
