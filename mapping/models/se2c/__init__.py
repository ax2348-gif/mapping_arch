"""Registry of continuous-SE(2) models (``CONTRACT_SE2C.md``), separate
from the Z^2 and SE(2) registries so a model name means one task.

A module here provides ``SIZES``, ``SCALING_RULE``, optional ``LR`` and
``NO_DECAY``, and ``build(size, spec)`` with ``spec`` a
``mapping.task_se2c.TaskSpec``; its forward is

    forward(obs, odometry, query_pose, query_time=None) -> pred

with the float pose inputs of ``mapping.task_se2c``, encoded by
``mapping.se2c.encoding``.
"""
import importlib
import pkgutil

from mapping.models import count_params  # noqa: F401

REGISTRY = {}


def register(name):
    def wrap(module):
        REGISTRY[name] = module
        return module
    return wrap


def get(name):
    if name not in REGISTRY:
        raise KeyError(f'unknown se2c model {name!r}; have {sorted(REGISTRY)}')
    return REGISTRY[name]


for _info in pkgutil.iter_modules(__path__):
    importlib.import_module(f'{__name__}.{_info.name}')
