"""Registry of SE(2)-task models (``CONTRACT_SE2.md``), separate from the
Z^2 registry in ``mapping.models`` so a model name means one task.

A module here provides ``SIZES``, ``SCALING_RULE``, optional ``LR`` and
``NO_DECAY``, and ``build(size, spec)`` with ``spec`` a
``mapping.task_se2.TaskSpec``; its forward is

    forward(obs, odometry, query_pose) -> pred

A port of a Z^2 model keeps the Z^2 file's name and changes the
interface and nothing else where possible, stating what changed.
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
        raise KeyError(f'unknown se2 model {name!r}; have {sorted(REGISTRY)}')
    return REGISTRY[name]


for _info in pkgutil.iter_modules(__path__):
    importlib.import_module(f'{__name__}.{_info.name}')
