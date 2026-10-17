"""vanilla4c_causal: vanilla4c with causal attention over the sorted token
order (H1e), as ``mapping/models/se2d/vanilla4_causal.py`` does for vanilla4.
Same modules and initial values as vanilla4c at the same seed. Odometry and
masking as in ``hybrid4c_causal``.
"""
import sys

from mapping.models.se2c import register
from mapping.models.se2c.vanilla4c import (LR, SCALING_RULE, SIZES,  # noqa: F401
                                           Vanilla4c)
from mapping.models.se2d.vanilla4_causal import CausalAttention


class Vanilla4cCausal(Vanilla4c):
    def __init__(self, spec, width, patch=3):
        super().__init__(spec, width, patch=patch)
        for blk in self.blocks:
            blk.mixer.__class__ = CausalAttention


def build(size, spec):
    return Vanilla4cCausal(spec, **SIZES[size])


register('vanilla4c_causal')(sys.modules[__name__])
