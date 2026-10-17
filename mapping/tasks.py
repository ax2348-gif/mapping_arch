"""The tasks the harness can train: one row per contract.

    models   the model registry for the task
    task     the module with TaskSpec, prepare, loss_fn, metrics, floor
    heading  the generator switch (mapping.data.Config)
    odometry the TaskSpec field that sizes a model's odometry table

Two generator settings apply to either task without making a new one:
``step_px`` (walk step; ``offset_max`` follows as ``steps * step_px``) and
``odometry_dropout`` (each odometry symbol replaced by a reserved unknown
symbol with that probability; when it is above 0 the spec's vocabulary
grows by one so tables have a row for it). A third, ``texture``
(``mapping.data.TEXTURES``: 'mnist', 'imagenette' or 'imagenette_hp'),
picks the tile bank and with it the tile side, brightness cutoff and start
window; it does not touch the spec, so models are sized exactly as before.

``se2c`` (continuous SE(2)) has its own generator settings
(``mapping.task_se2c.add_args``) and generates batches on the GPU.

``z2`` is the default everywhere, so code written before ``se2d`` existed
behaves as it did. Checkpoints record their task; ones without it are z2.

Where runs live is not a property of the task: a run's directory is the
config's ``run.dir``, ``${run.root}/<task>/<model>/<size>/<run>``
(docs/config.md, "Run directories"), the same template for every task.
``results_dir`` gives a task's directory under a root, for the tools that
read runs back.
"""
import os

import mapping.models
import mapping.models.se2d
import mapping.models.se2c
import mapping.task
import mapping.task_se2d
import mapping.task_se2c
from mapping.data import VOCAB, Config, texture_settings

TASKS = {
    'z2': dict(models=mapping.models, task=mapping.task,
               heading=False, odometry='directions'),
    'se2d': dict(models=mapping.models.se2d, task=mapping.task_se2d,
                heading=True, odometry='odometry'),
}

# Tasks with their own GPU generator (batches made inside the training
# step by the task module's ``data``), kept apart from TASKS so code and
# tests written for the tile-bank tasks never meet them by iteration.
# se2c: continuous SE(2), CONTRACT_SE2C.md.
GPU_TASKS = {
    'se2c': dict(models=mapping.models.se2c, task=mapping.task_se2c,
                 gpu_data=True),
}
ALL = {**TASKS, **GPU_TASKS}


# Read-only aliases: names old checkpoints, final.json files and results
# directories carry. ``se2`` is the discrete SE(2) task, renamed ``se2d``
# (2026-10-16). Its old runs stay under results/se2/, which is the run
# directory template applied to the name they were recorded with, so they
# are not moved (docs/config.md, "se2 is now se2d"). New runs cannot use an
# alias (train.py offers ``ALL`` only).
ALIASES = {'se2': 'se2d'}

RESULTS = 'results'          # the default run.root (conf/config.yaml)


def canonical(name):
    """The current name of a task, resolving read-only aliases."""
    return ALIASES.get(name, name)


def get(name='z2'):
    """A task row; an alias gives the current task's."""
    name = canonical(name)
    if name not in ALL:
        raise KeyError(f'unknown task {name!r}; have {sorted(ALL)}')
    return ALL[name]


def results_dir(name, root=RESULTS):
    """Where a task's runs live under ``root``: ``<root>/<name>``, the
    first segment of the ``run.dir`` template. An alias keeps its own name
    (``se2`` -> ``results/se2``, its old runs)."""
    if name not in ALL and name not in ALIASES:
        raise KeyError(f'unknown task {name!r}; have {sorted(ALL) + sorted(ALIASES)}')
    return os.path.join(root, name)


def config_and_spec(name, steps, queries, seed, query_times='last',
                    step_px=9, odometry_dropout=0.0, texture='mnist',
                    path='walk'):
    """The generator config and the model-facing spec for a task. With the
    defaults both are exactly what they were before ``step_px``,
    ``odometry_dropout`` and ``texture`` existed. The texture sets the
    config's tile, cutoff and start window; the spec does not depend on
    it."""
    t = get(name)
    if t.get('gpu_data'):          # se2c: the generator config and its spec
        config = t['task'].config_from_args(
            {'episode_steps': steps, 'queries': queries, 'seed': seed,
             'query_times': query_times})
        return config, t['task'].TaskSpec(view=config.view, steps=steps,
                                          queries=queries)
    config = Config(steps=steps, queries=queries, seed=seed,
                    heading=t['heading'], query_times=query_times,
                    step=step_px, odometry_dropout=odometry_dropout, path=path,
                    **texture_settings(texture))
    extra = {}
    if odometry_dropout > 0:     # a row for the reserved "unknown" symbol
        extra[t['odometry']] = VOCAB[t['heading']] + 1
    spec = t['task'].TaskSpec(view=config.obs, steps=config.steps,
                              step_px=config.step, queries=config.queries,
                              **extra)
    return config, spec


def config_from_args(task, spec, args):
    """Rebuild a run's generator config from its saved arguments (render.py,
    tools). Checkpoints older than a setting get its default (texture
    'mnist')."""
    config, _ = config_and_spec(
        task, spec.steps, spec.queries, args['seed'],
        args.get('query_times', 'last'),
        args.get('step_px', spec.step_px),
        args.get('odometry_dropout', 0.0),
        args.get('texture', 'mnist'),
        args.get('path', 'walk'))
    return config
