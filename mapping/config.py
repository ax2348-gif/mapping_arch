"""The run configuration as objects (Hydra config tree in ``conf/``).

Every node of a composed config is instantiated into one of the classes
below, and the trainer (``train.main``) reads its settings from these
objects and nothing else: the model is ``Model.build``, the learning rate
is ``schedule.lr``, the batches come from the generator's ``data`` (bank)
or ``config`` (se2c). Their signatures are strict (dataclasses, and the
model classes' own constructors), so a key no object consumes is an error
at ``instantiate`` time. Variants (which generator a task uses, which
schedule) are chosen by the defaults list, so only the fields the chosen
variant uses are ever composed. ``tests/test_config.py`` checks that the
trainer reads every field and that changing any field changes what it
trains on, or is refused.

``Experiment.legacy_args()`` is an output only: the flat ``args`` record
runs write into checkpoints, ``final.json`` and ``log.jsonl`` (the same
``argparse.Namespace`` ``train.py --flags`` gives), so every existing tool
keeps reading new runs. Every task, generator and schedule field is
copied into it under its own name (the se2c texture as ``se2c_texture``),
so a new field reaches the record without code here. Settings a run's
variants do not compose (the se2c generator on a z2 run, say) appear there
at train.py's defaults.
"""
import functools
import inspect
import math
import os
from dataclasses import dataclass, fields, is_dataclass
from typing import Optional

from omegaconf import OmegaConf

import mapping.models as models
import mapping.tasks as tasks

DEFAULT_LR = 3e-4


def model_lr(task, name, size):
    """A model's default learning rate: its module's ``LR[size]`` if it
    defines one, else 3e-4 (as train.py). Resolver for ``optim.lr``."""
    module = tasks.get(task)['models'].get(name)
    return getattr(module, 'LR', {}).get(size, DEFAULT_LR)


def ladder_rung(name, size):
    """A rung's constructor arguments, ``conf/ladders/<name>.yaml``.
    Resolver for ``model.size_kwargs``."""
    rungs = models.ladder(name)
    if size not in rungs:
        raise KeyError(f'model.size={size!r} is not a rung of the {name!r} ladder '
                       f'(conf/ladders/{name}.yaml): have {list(rungs)}')
    return dict(rungs[size])


def run_name(name, seed):
    """A run's directory name: ``run.name``, or ``seed<seed>`` when it is
    null (as the trainer always named it). Resolver for ``run.dir``."""
    return name if name is not None else f'seed{seed}'


if not OmegaConf.has_resolver('run_name'):
    OmegaConf.register_new_resolver('run_name', run_name)
if not OmegaConf.has_resolver('model_lr'):
    OmegaConf.register_new_resolver('model_lr', model_lr)
if not OmegaConf.has_resolver('ladder'):
    OmegaConf.register_new_resolver('ladder', ladder_rung)


class Node:
    """A config node: a frozen dataclass whose values are coerced to their
    annotated types, as argparse coerced flags (``v_min=9`` is 9.0,
    ``holdout=[0.8, 1.2]`` a tuple)."""

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if f.type in (float, Optional[float]) and isinstance(v, int) \
                    and not isinstance(v, bool):
                object.__setattr__(self, f.name, float(v))
            elif f.type in (tuple, Optional[tuple]) and isinstance(v, list):
                object.__setattr__(self, f.name, tuple(v))


@dataclass(frozen=True)
class Provenance(Node):
    """The code snapshot every run makes (mapping/provenance.py)."""
    max_untracked_mb: int        # an untracked file larger than this is refused


@dataclass(frozen=True)
class Run(Node):
    """Where the run goes, how long, and on what."""
    name: Optional[str]          # run directory name; None: seed<seed>
    out: Optional[str]           # explicit run directory; wins over dir
    root: str                    # where run directories live
    dir: str                     # the run directory: ${root}/<task>/<model>/<size>/<name>
    note: Optional[str]          # required at launch (run.py refuses none); ledger only
    tag: tuple                   # campaign labels; ledger only
    provenance: Provenance
    steps: int
    seed: int
    save_every: int              # write step<k>.pt every k steps; 0: never
    branch: Optional[str]        # a step<k>.pt to continue from
    device: Optional[str]        # None: cuda if available, else cpu
    bf16: bool                   # bf16 autocast (CUDA only; ignored on CPU)
    compile: bool
    overwrite: bool              # write into a non-empty run directory

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.tag, str):
            object.__setattr__(self, 'tag', (self.tag,))
        if isinstance(self.note, str):
            object.__setattr__(self, 'note', self.note.strip())
        if not isinstance(self.provenance, Provenance):
            raise TypeError(f'run.provenance must instantiate a Provenance, got '
                            f'{type(self.provenance).__name__}')

    def location(self):
        """Where this run writes: ``{'dir', 'root', 'rel'}``. ``dir`` is
        ``out`` when set, else the template ``dir``; ``rel`` is its path
        under ``root`` (None when it lies outside, as an explicit ``out``
        may). Tools look runs up by this, not by re-deriving paths."""
        d = self.out or self.dir
        rel = os.path.relpath(os.path.abspath(d), os.path.abspath(self.root))
        return {'dir': d, 'root': self.root,
                'rel': None if rel == os.pardir or rel.startswith(os.pardir + os.sep) else rel}


@dataclass(frozen=True)
class Eval(Node):
    batches: int                 # held-out batches
    every: int                   # evaluate every k steps
    log_every: int


@dataclass(frozen=True)
class AdamW(Node):
    """AdamW (betas 0.9, 0.95; 1-D parameters never decay) with gradient
    clipping, on batches of ``batch`` episodes."""
    lr: float
    weight_decay: float
    clip: float
    batch: int


class Schedule(Node):
    """A learning-rate schedule: ``lr(step, steps, peak)``, the rate at
    0-based ``step`` of a run of ``steps`` steps peaking at ``peak``."""
    name = None

    def lr(self, step, steps, peak):
        raise NotImplementedError


@dataclass(frozen=True)
class Cosine(Schedule):
    """Linear warmup, then cosine decay to zero at the last step."""
    warmup: int
    name = 'cosine'

    def lr(self, step, steps, peak):
        if step < self.warmup:
            return peak * (step + 1) / self.warmup
        progress = (step - self.warmup) / max(1, steps - self.warmup)
        return peak * 0.5 * (1 + math.cos(math.pi * progress))


@dataclass(frozen=True)
class WSD(Schedule):
    """Warmup-stable-decay: linear warmup, hold, then linear decay to zero
    over the last ``cooldown`` fraction of the steps."""
    warmup: int
    cooldown: float
    name = 'wsd'

    def lr(self, step, steps, peak):
        if step < self.warmup:
            return peak * (step + 1) / self.warmup
        decay_start = int(round(steps * (1 - self.cooldown)))
        if step < decay_start:
            return peak
        return peak * max(0.0, (steps - step) / max(1, steps - decay_start))


class Generator(Node):
    """A task's batch generator. ``record`` maps a field to its name in
    the legacy ``args`` record where the two differ."""
    record = {}

    def __post_init__(self):
        super().__post_init__()
        p = getattr(self, 'odometry_dropout', 0.0)
        if not 0.0 <= p <= 1.0:
            raise ValueError(
                f'task.generator.odometry_dropout={p!r} is not a probability: it must be '
                f'in [0, 1] (0: every step keeps its odometry; 1: all hidden, visual '
                f'odometry)')


@dataclass(frozen=True)
class BankGenerator(Generator):
    """The tile-bank generator of the z2 and se2d tasks (mapping.data).
    ``data(task, seed)`` is the generator config and model spec."""
    texture: str                 # mapping.data.TEXTURES
    bank: Optional[str]          # None: the texture's bank file
    step_px: int
    odometry_dropout: float
    workers: int                 # loader processes
    path: str                    # mapping.data.PATHS: walk | saw | no_revisit_view

    def bank_path(self):
        from mapping.data import TEXTURES
        return self.bank if self.bank is not None else TEXTURES[self.texture]['path']

    def data(self, task, seed):
        """``(mapping.data.Config, TaskSpec)`` for ``task`` (a ``Task``)."""
        return tasks.config_and_spec(task.name, task.episode_steps, task.queries,
                                     seed, task.query_times, self.step_px,
                                     self.odometry_dropout, self.texture, self.path)


@dataclass(frozen=True)
class Se2cGenerator(Generator):
    """The continuous-SE(2) GPU generator (mapping.se2c.generator,
    CONTRACT_SE2C.md). ``config(task, seed)`` is its
    ``mapping.se2c.generator.Config``: every field here is the field of
    that name there, except ``texture`` (its ``bank``) and ``holdout``
    (None is ``()``). A new generator setting is a field in both."""
    texture: str
    turn_n: int
    motion: str
    directions: str
    v_min: float
    v_max: float
    omega_max: float
    holdout: Optional[tuple]     # (lo, hi) band of |turn| in rad
    holdout_mode: str
    query_coverage: str
    odometry_dropout: float      # P(a step's odometry -> zero vector); 1.0: visual odometry
    record = {'texture': 'se2c_texture'}

    def config(self, task, seed):
        from mapping.se2c.generator import Config
        from mapping.se2c.texture import TextureSpec
        same = {f.name: getattr(self, f.name) for f in fields(self)
                if f.name not in ('texture', 'holdout')}
        return Config(steps=task.episode_steps, queries=task.queries,
                      query_times=task.query_times, seed=seed,
                      holdout=tuple(self.holdout or ()),
                      bank='' if self.texture == 'procedural' else self.texture,
                      texture=TextureSpec(), **same)


@dataclass(frozen=True)
class Task(Node):
    """A task contract with its episode settings and its generator."""
    name: str
    episode_steps: int
    queries: int
    query_times: str             # last | uniform
    generator: Generator

    def __post_init__(self):
        super().__post_init__()
        if self.name not in tasks.ALL:
            raise ValueError(f'unknown task {self.name!r}; have {sorted(tasks.ALL)}')
        want = Se2cGenerator if tasks.get(self.name).get('gpu_data') else BankGenerator
        if not isinstance(self.generator, want):
            raise TypeError(f'task {self.name} needs a {want.__name__}, '
                            f'got {type(self.generator).__name__}')


def _plain(v):
    """Lists (as yaml gives them) as tuples, for comparison and binding."""
    return tuple(_plain(x) for x in v) if isinstance(v, (list, tuple)) else v


class Model:
    """A registered model at one rung of its size ladder.

    ``ladder`` names ``conf/ladders/<ladder>.yaml``; ``size`` is a rung of
    it and ``size_kwargs`` that rung's constructor arguments (resolved
    from the ladder file and saved with the run). ``arch`` is the model
    class with any fixed arguments bound (Hydra ``_partial_``), such as
    hybrid45b's placement. ``build(spec)`` makes the module the trainer
    trains.

    Checked here: the size is a rung of the ladder, ``size_kwargs`` is
    that rung exactly, ``arch`` does not set a ladder argument, and every
    argument binds to the class's signature. ``Experiment`` checks that
    this is the registered model (the same ladder and fixed arguments as
    its module). So a model's architecture is always its name at a rung:
    an override that would train something else is an error, never a run
    whose saved config differs from what trained.

    ``query_mode`` (``mixed`` | ``inert``, mapping/models/query_mode.py)
    is passed to the class only when it is not ``mixed``, so a mixed run's
    constructor arguments, checkpoint ``arch`` and args record are what
    they were before the field existed; ``inert`` on a class that does not
    implement it is refused.

    ``scan_precision`` (``tf32`` | ``ieee``, mapping/models/scan_precision.py)
    likewise: passed only when not ``tf32`` (every run before the field
    trained tf32), and ``ieee`` is refused by a class that does not declare
    it (one without a Gated DeltaNet)."""

    def __init__(self, name, ladder, size, size_kwargs, arch, query_mode='mixed',
                 scan_precision='tf32'):
        from mapping.models.query_mode import QUERY_MODES
        from mapping.models.scan_precision import SCAN_PRECISIONS
        self.name, self.ladder, self.size = name, ladder, str(size)
        if query_mode not in QUERY_MODES:
            raise ValueError(f'model.query_mode={query_mode!r}: must be one of {QUERY_MODES}')
        if scan_precision not in SCAN_PRECISIONS:
            raise ValueError(f'model.scan_precision={scan_precision!r}: must be one of '
                             f'{SCAN_PRECISIONS}')
        self.query_mode = query_mode
        self.scan_precision = scan_precision
        rung = ladder_rung(ladder, self.size)
        size_kwargs = {k: _plain(v) for k, v in dict(size_kwargs).items()}
        if size_kwargs != rung:
            raise ValueError(
                f'model.size_kwargs {size_kwargs} is not rung {self.size!r} of the '
                f'{ladder!r} ladder ({rung}). A size is a rung: pick one with '
                f'model.size, or add one to conf/ladders/{ladder}.yaml')
        fixed = {k: _plain(v) for k, v in arch.keywords.items()}
        clash = sorted(set(fixed) & set(rung))
        if clash:
            raise ValueError(
                f'model.arch.{clash[0]} is set by the size ({ladder} ladder, rung '
                f'{self.size!r}: {rung}). Pick another rung with model.size, or add '
                f'one to conf/ladders/{ladder}.yaml')
        self.size_kwargs, self.fixed = rung, fixed
        self.kwargs = {**rung, **fixed}
        if query_mode != 'mixed':
            if query_mode not in getattr(arch.func, '__dict__', {}).get('QUERY_MODES', ()):
                raise ValueError(f'model.query_mode={query_mode!r}: {name} '
                                 f'({arch.func.__qualname__}) does not implement it '
                                 f'(mapping/models/query_mode.py)')
            self.kwargs['query_mode'] = query_mode
        if scan_precision != 'tf32':
            if scan_precision not in getattr(arch.func, '__dict__', {}).get('SCAN_PRECISIONS', ()):
                raise ValueError(f'model.scan_precision={scan_precision!r}: {name} '
                                 f'({arch.func.__qualname__}) does not implement it '
                                 f'(mapping/models/scan_precision.py)')
            self.kwargs['scan_precision'] = scan_precision
        self.cls = arch.func
        self.arch = functools.partial(arch.func, *arch.args, **self.kwargs)
        inspect.signature(arch.func).bind(None, *arch.args, **self.kwargs)

    def module(self, task):
        return tasks.get(task)['models'].get(self.name)

    def check_registered(self, task):
        """This is the model ``name`` registers for ``task``: its module
        reads the same ladder and binds the same fixed arguments."""
        module = self.module(task)
        if dict(module.SIZES) != models.ladder(self.ladder):
            raise ValueError(f'model {self.name}: its config names ladder '
                             f'{self.ladder!r} but its module builds from another '
                             f'(SIZES in {module.__name__})')
        want = {k: _plain(v) for k, v in getattr(module, 'ARCH', {}).items()}
        if self.fixed != want:
            diff = sorted(k for k in set(want) | set(self.fixed)
                          if want.get(k) != self.fixed.get(k))
            raise ValueError(
                f'model.arch.{diff[0]}={self.fixed.get(diff[0])!r} is not what '
                f'{self.name} registers ({want.get(diff[0])!r}, ARCH in '
                f'{module.__name__}). A different architecture is a different '
                f'model: register one (docs/config.md, "Add a model")')

    def build(self, spec):
        return self.arch(spec)

    def record(self):
        """What checkpoints save to rebuild this model without the config
        or the registry (``mapping.checkpoint.load_checkpoint``)."""
        return {'class': f'{self.cls.__module__}.{self.cls.__qualname__}',
                'kwargs': dict(self.kwargs)}


@dataclass(frozen=True)
class Experiment:
    """One training run: every object the trainer uses."""
    task: Task
    model: Model
    schedule: Schedule
    optim: AdamW
    run: Run
    eval: Eval

    def __post_init__(self):
        for name, cls in (('task', Task), ('model', Model), ('schedule', Schedule),
                          ('optim', AdamW), ('run', Run), ('eval', Eval)):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f'{name} must instantiate a {cls.__name__}, got '
                                f'{type(getattr(self, name)).__name__}')
        self.model.check_registered(self.task.name)

    def legacy_args(self):
        """``argparse.Namespace`` exactly as ``train.parse`` gives it for
        the equivalent ``train.py --flags``: train.py's defaults, then
        every field of these objects. An output only (see the module
        docstring)."""
        import train
        a = train.parse(['--model', self.model.name, '--size', self.model.size])
        r, o, e, t, s = self.run, self.optim, self.eval, self.task, self.schedule
        a.task, a.run, a.out, a.steps, a.seed = t.name, r.name, r.out, r.steps, r.seed
        a.save_every, a.branch, a.compile, a.no_bf16 = r.save_every, r.branch, r.compile, not r.bf16
        if r.device is not None:
            a.device = r.device
        a.lr, a.weight_decay, a.clip, a.batch = o.lr, o.weight_decay, o.clip, o.batch
        a.schedule = s.name
        if self.model.query_mode != 'mixed':   # absent for mixed: records as before it
            a.query_mode = self.model.query_mode
        if self.model.scan_precision != 'tf32':   # absent for tf32: records as before it
            a.scan_precision = self.model.scan_precision
        for f in fields(s):                  # warmup, cooldown, ...
            setattr(a, f.name, getattr(s, f.name))
        a.eval_batches, a.eval_every, a.log_every = e.batches, e.every, e.log_every
        for f in fields(t):                  # episode_steps, queries, query_times, ...
            if f.name not in ('name', 'generator'):
                setattr(a, f.name, getattr(t, f.name))
        g = t.generator
        for f in fields(g):
            v = getattr(g, f.name)
            if f.name == 'path' and v == 'walk':   # absent for walk: records as before it
                continue
            setattr(a, g.record.get(f.name, f.name), list(v) if isinstance(v, tuple) else v)
        # argparse's types, as if every value had come in as a flag (an
        # override weight_decay=0 is then 0.0, as --weight-decay 0 gives)
        for action in train.build_parser()._actions:
            v = getattr(a, action.dest, None)
            if callable(action.type) and v is not None:
                setattr(a, action.dest, [action.type(x) for x in v]
                        if isinstance(v, (list, tuple)) else action.type(v))
        return a


# the classes a config node may instantiate (every Node dataclass defined
# here), for the consumed-field tests
NODES = tuple(c for c in list(globals().values())
              if isinstance(c, type) and issubclass(c, Node) and is_dataclass(c))


def field_names(cls):
    return [f.name for f in fields(cls)]
