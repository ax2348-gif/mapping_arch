"""Causal continuous hybrid6 registration. Same local-camera audit as hybrid6."""
import sys
from mapping.models.se2c import register
from mapping.models.se2c.hybrid6c import Hybrid6c, LR, SCALING_RULE, SIZES

class Hybrid6cCausal(Hybrid6c):
    QUERY_MODES = ('mixed', 'inert')
    SCAN_PRECISIONS = ('tf32', 'ieee')

    def __init__(self, spec, width, query_mode='mixed', scan_precision='tf32'):
        super().__init__(spec, width, query_mode, causal=True, scan_precision=scan_precision)


def build(size, spec):
    return Hybrid6cCausal(spec, **SIZES[size])


register('hybrid6c_causal')(sys.modules[__name__])
