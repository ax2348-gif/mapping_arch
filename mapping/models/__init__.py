"""Model registry.

A model module registers a name and provides

    SIZES: dict[str, dict]        an ordered ladder, smallest first; each
                                  value is the kwargs that build that size
                                  (``ladder(name)``, conf/ladders/<name>.yaml)
    ARCH: dict (optional)         fixed constructor kwargs beyond the size,
                                  e.g. hybrid45b's placement; the model's
                                  config file must bind the same
    SCALING_RULE: str             one paragraph: which dimensions grow along
                                  the ladder and in what ratio, so a point
                                  on the ladder is reproducible from its
                                  name alone
    build(size, spec) -> nn.Module

and the module's forward is

    forward(obs, directions, query_offset) -> pred      see mapping.task

Register with the decorator; importing ``mapping.models`` imports every
module in this package so the registry is full.
"""
import importlib
import pkgutil

REGISTRY = {}


def register(name):
    def wrap(module_or_fn):
        REGISTRY[name] = module_or_fn
        return module_or_fn
    return wrap


def ladder(name):
    """A size ladder, ``conf/ladders/<name>.yaml``, as the ordered ``SIZES``
    dict a model module exposes: rung name -> builder kwargs, in file
    order (smallest first). The ladder files are the single source of the
    ladders: a model module reads its own with ``SIZES = ladder(...)``, and
    its config file names the same ladder (``model.ladder``), which
    ``mapping.config.Experiment`` checks."""
    import os
    import yaml
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'conf', 'ladders', name + '.yaml')
    if not os.path.exists(path):
        raise KeyError(f'no ladder {name!r} ({path})')
    with open(path) as f:
        rungs = yaml.safe_load(f)
    return {str(size): dict(kwargs) for size, kwargs in rungs.items()}


def get(name):
    if name not in REGISTRY:
        raise KeyError(f'unknown model {name!r}; have {sorted(REGISTRY)}')
    return REGISTRY[name]


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# import every sibling module so its register() runs
for _info in pkgutil.iter_modules(__path__):
    if not _info.ispkg:                     # subpackages are other tasks' registries
        importlib.import_module(f'{__name__}.{_info.name}')
