"""The run configuration as objects (Hydra config tree in ``conf/``).

Every node of a composed config is instantiated into one of the classes
below. Their signatures are strict (dataclasses, and the model classes'
own constructors), so a key no object consumes is an error at
``instantiate`` time, never a silently ignored setting. Variants (which
generator a task uses, which schedule, which size) are chosen by the
defaults list, so only the fields the chosen variant uses are ever
composed. ``tests/test_config.py`` composes every task, model, schedule
and experiment preset and checks that changing any field changes the run.

``Experiment.legacy_args()`` turns the objects into the flat settings
``train.main`` takes (the same ``argparse.Namespace`` ``train.py --flags``
gives). It is also what runs write into their checkpoints and
``final.json`` as ``args``, unchanged, so every existing tool keeps
working. Settings a run's variants do not compose (the se2c generator on
a z2 run, say) take train.py's defaults there: they are part of that
legacy record only, never of the config.
"""
import functools
import inspect
from dataclasses import dataclass, fields
from typing import Optional, Union

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


if not OmegaConf.has_resolver('model_lr'):
    OmegaConf.register_new_resolver('model_lr', model_lr)
if not OmegaConf.has_resolver('ladder'):
    OmegaConf.register_new_resolver('ladder', ladder_rung)


@dataclass(frozen=True)
class Run:
    """Where the run goes, how long, and on what."""
    name: Optional[str]          # run directory name; None: seed<seed>
    out: Optional[str]           # explicit run directory, overriding name
    steps: int
    seed: int
    save_every: int              # write step<k>.pt every k steps; 0: never
    branch: Optional[str]        # a step<k>.pt to continue from
    device: Optional[str]        # None: cuda if available, else cpu
    bf16: bool
    compile: bool


@dataclass(frozen=True)
class Eval:
    batches: int                 # held-out batches
    every: int                   # evaluate every k steps
    log_every: int


@dataclass(frozen=True)
class AdamW:
    """AdamW (betas 0.9, 0.95; 1-D parameters never decay) with gradient
    clipping, on batches of ``batch`` episodes."""
    lr: float
    weight_decay: float
    clip: float
    batch: int


@dataclass(frozen=True)
class Cosine:
    """Linear warmup, then cosine decay to zero at the last step."""
    warmup: int
    name = 'cosine'


@dataclass(frozen=True)
class WSD:
    """Warmup-stable-decay: linear warmup, hold, then linear decay to zero
    over the last ``cooldown`` fraction of the steps."""
    warmup: int
    cooldown: float
    name = 'wsd'


@dataclass(frozen=True)
class BankGenerator:
    """The tile-bank generator of the z2 and se2d tasks (mapping.data)."""
    texture: str                 # mapping.data.TEXTURES
    bank: Optional[str]          # None: the texture's bank file
    step_px: int
    odometry_dropout: float
    workers: int                 # loader processes


@dataclass(frozen=True)
class Se2cGenerator:
    """The continuous-SE(2) GPU generator (mapping.se2c.generator,
    CONTRACT_SE2C.md)."""
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


@dataclass(frozen=True)
class Task:
    """A task contract with its episode settings and its generator."""
    name: str
    episode_steps: int
    queries: int
    query_times: str             # last | uniform
    generator: Union[BankGenerator, Se2cGenerator]

    def __post_init__(self):
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
    whose saved config differs from what trained."""

    def __init__(self, name, ladder, size, size_kwargs, arch):
        self.name, self.ladder, self.size = name, ladder, str(size)
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


@dataclass(frozen=True)
class Experiment:
    """One training run: every object the trainer uses."""
    task: Task
    model: Model
    schedule: Union[Cosine, WSD]
    optim: AdamW
    run: Run
    eval: Eval

    def __post_init__(self):
        self.model.check_registered(self.task.name)

    def legacy_args(self):
        """``argparse.Namespace`` exactly as ``train.parse`` gives it for
        the equivalent ``train.py --flags``: train.py's defaults, then
        every field of these objects."""
        import train
        a = train.parse(['--model', self.model.name, '--size', self.model.size])
        r, o, e, t, s = self.run, self.optim, self.eval, self.task, self.schedule
        a.task, a.run, a.out, a.steps, a.seed = t.name, r.name, r.out, r.steps, r.seed
        a.save_every, a.branch, a.compile, a.no_bf16 = r.save_every, r.branch, r.compile, not r.bf16
        if r.device is not None:
            a.device = r.device
        a.lr, a.weight_decay, a.clip, a.batch = o.lr, o.weight_decay, o.clip, o.batch
        a.schedule, a.warmup = s.name, s.warmup
        if isinstance(s, WSD):
            a.cooldown = s.cooldown
        a.eval_batches, a.eval_every, a.log_every = e.batches, e.every, e.log_every
        a.episode_steps, a.queries, a.query_times = t.episode_steps, t.queries, t.query_times
        g = t.generator
        if isinstance(g, BankGenerator):
            a.texture, a.bank, a.step_px = g.texture, g.bank, g.step_px
            a.odometry_dropout, a.workers = g.odometry_dropout, g.workers
        else:
            a.se2c_texture, a.turn_n, a.motion, a.directions = (
                g.texture, g.turn_n, g.motion, g.directions)
            a.v_min, a.v_max, a.omega_max = g.v_min, g.v_max, g.omega_max
            a.holdout = list(g.holdout) if g.holdout is not None else None
            a.holdout_mode, a.query_coverage = g.holdout_mode, g.query_coverage
        # argparse's types, as if every value had come in as a flag (an
        # override weight_decay=0 is then 0.0, as --weight-decay 0 gives)
        for action in train.build_parser()._actions:
            v = getattr(a, action.dest, None)
            if callable(action.type) and v is not None:
                setattr(a, action.dest, [action.type(x) for x in v]
                        if isinstance(v, (list, tuple)) else action.type(v))
        return a


# the classes a config node may instantiate, for the consumed-field tests
NODES = (Run, Eval, AdamW, Cosine, WSD, BankGenerator, Se2cGenerator, Task,
         Experiment)


def field_names(cls):
    return [f.name for f in fields(cls)]
