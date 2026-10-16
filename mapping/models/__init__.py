"""Model registry.

A model module registers a name and provides

    SIZES: dict[str, dict]        an ordered ladder, smallest first; each
                                  value is the kwargs that build that size
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


def ladder(name=''):
    """A size ladder from the config tree, ``conf/model/size/<name>/``
    (``''``: the shared width ladder at ``conf/model/size/``), as the
    ordered ``SIZES`` dict a model module exposes: size name -> builder
    kwargs, smallest first. The config files are the single source of the
    ladders; ``tests/test_config.py`` checks them against the tables the
    modules held before the move (``tests/sizes_snapshot.json``)."""
    import os
    import yaml
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'conf', 'model', 'size', name)
    sizes = {}
    for f in os.listdir(d):
        if f.endswith('.yaml'):
            with open(os.path.join(d, f)) as fh:
                c = yaml.safe_load(fh)
            sizes[c['size']] = dict(c['arch'])
    return dict(sorted(sizes.items(), key=lambda kv: tuple(kv[1].values())))


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
