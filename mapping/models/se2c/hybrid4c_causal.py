"""hybrid4c_causal: hybrid4c with every operator causal in token order
(H1e, HYPOTHESIS.md): every Gated DeltaNet head scans left to right and
attention is masked causally, exactly as ``mapping/models/se2d/hybrid4_causal.py``
does for hybrid4. Same modules and initial values as hybrid4c at the same
seed; only the operators' forward changes (``CausalGatedDeltaNet``,
``CausalAttention`` from hybrid4_causal).

Odometry is the INPUT form hybrid4c already uses: the relative pose of
step t+1 from t, one token, placed after observation t and the queries
asked at t and before observation t+1 (``token_order``: observation <
query < odometry per step), i.e. "before the next observation". Under the
causal mask a query at t therefore sees observations 0..t and odometry
0..t-1 (the moves into t) and nothing later (tests/test_se2c_causal.py).
"""
import sys

from mapping.models.se2c import register
from mapping.models.se2c.hybrid4c import (LR, SCALING_RULE, SIZES,  # noqa: F401
                                          Hybrid4c)
from mapping.models.se2d.hybrid4 import GatedDeltaNet
from mapping.models.se2d.hybrid4_causal import (CausalAttention,
                                                CausalGatedDeltaNet)


class Hybrid4cCausal(Hybrid4c):
    QUERY_MODES = ('mixed', 'inert')    # mapping/models/query_mode.py
    SCAN_PRECISIONS = ('tf32', 'ieee')  # mapping/models/scan_precision.py

    def __init__(self, spec, width, patch=3, query_mode='mixed', scan_precision='tf32'):
        super().__init__(spec, width, patch=patch, query_mode=query_mode,
                         scan_precision=scan_precision)
        for blk in self.blocks:          # same modules and initial values
            blk.mixer.__class__ = (CausalGatedDeltaNet
                                   if isinstance(blk.mixer, GatedDeltaNet)
                                   else CausalAttention)


def build(size, spec):
    return Hybrid4cCausal(spec, **SIZES[size])


register('hybrid4c_causal')(sys.modules[__name__])
