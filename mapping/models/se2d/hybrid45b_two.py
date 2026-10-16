"""hybrid45b_two: hybrid4.5b with coprocessor placement ('first', 'middle') (trunk blocks
run before each coprocessor: first = 0, middle = n/2, last = n; see
hybrid45b.py for the design, the muP rule and the mechanism trace).

Two independent coprocessors, first and middle: the second re-expresses
the frame after half the trunk has retrieved content (lift notes section 04,
multi-hop re-anchoring).
"""
import sys

from mapping.models.se2d import register
from mapping.models.se2d import hybrid45b as base

SIZES = base.SIZES
LR = base.LR
SCALING_RULE = base.SCALING_RULE
PLACEMENT = ('first', 'middle')
ARCH = {'placement': PLACEMENT}   # what this model's config file binds


def build(size, spec):
    return base.build(size, spec, placement=PLACEMENT)


register('hybrid45b_two')(sys.modules[__name__])
