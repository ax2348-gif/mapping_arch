"""Hybrid7 with registers AND outgoing odometry as recurrent writers."""
import sys
from mapping.models.se2d import register
from mapping.models.se2d.hybrid7 import Hybrid7, LR, SIZES, ARCH as BASE_ARCH
ARCH = {**BASE_ARCH, 'odometry_writes': True}
SCALING_RULE = 'Hybrid7 AGAGA width ladder, register and odometry writes/convolution.'
def build(size, spec):
    return Hybrid7(spec, **SIZES[size], **ARCH, query_mode='inert')
register('hybrid7_odom')(sys.modules[__name__])
