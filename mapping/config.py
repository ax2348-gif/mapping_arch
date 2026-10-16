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

import mapping.tasks as tasks

DEFAULT_LR = 3e-4


def model_lr(task, name, size):
    """A model's default learning rate: its module's ``LR[size]`` if it
    defines one, else 3e-4 (as train.py). Resolver for ``optim.lr``."""
    module = tasks.get(task)['models'].get(name)
    return getattr(module, 'LR', {}).get(size, DEFAULT_LR)


if not OmegaConf.has_resolver('model_lr'):
    OmegaConf.register_new_resolver('model_lr', model_lr)


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


class Model:
    """A registered model at one size. ``arch`` is the model class with
    its architecture arguments bound (Hydra ``_partial_``); ``build(spec)``
    makes the module. Checked at construction: the name is registered for
    the task, and the architecture arguments bind to the class's own
    signature, so a stray key is an error here, not at training time."""

    def __init__(self, name, size, arch):
        self.name, self.size = name, size
        kwargs = {k: tuple(v) if isinstance(v, list) else v
                  for k, v in arch.keywords.items()}
        self.arch = functools.partial(arch.func, *arch.args, **kwargs)
        inspect.signature(arch.func).bind(None, *arch.args, **kwargs)

    def module(self, task):
        return tasks.get(task)['models'].get(self.name)

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
        self.model.module(self.task.name)       # registered for this task

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
