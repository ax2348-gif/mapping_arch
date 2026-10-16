"""The tasks the harness can train: one row per contract.

    models   the model registry for the task
    task     the module with TaskSpec, prepare, loss_fn, metrics, floor
    root     where its runs live
    heading  the generator switch (mapping.data.Config)
    odometry the TaskSpec field that sizes a model's odometry table

Two generator settings apply to either task without making a new one:
``step_px`` (walk step; ``offset_max`` follows as ``steps * step_px``) and
``odometry_dropout`` (each odometry symbol replaced by a reserved unknown
symbol with that probability; when it is above 0 the spec's vocabulary
grows by one so tables have a row for it). A third, ``texture``
(``mapping.data.TEXTURES``: 'mnist' or 'imagenette'), picks the tile bank
and with it the tile side, brightness cutoff and start window; it does not
touch the spec, so models are sized exactly as before.

``z2`` is the default everywhere, so code written before ``se2`` existed
behaves as it did. Checkpoints record their task; ones without it are z2.
"""
import mapping.models
import mapping.models.se2
import mapping.task
import mapping.task_se2
from mapping.data import VOCAB, Config, texture_settings

TASKS = {
    'z2': dict(models=mapping.models, task=mapping.task, root='results',
               heading=False, odometry='directions'),
    'se2': dict(models=mapping.models.se2, task=mapping.task_se2,
                root='results/se2', heading=True, odometry='odometry'),
}


def get(name='z2'):
    if name not in TASKS:
        raise KeyError(f'unknown task {name!r}; have {sorted(TASKS)}')
    return TASKS[name]


def config_and_spec(name, steps, queries, seed, query_times='last',
                    step_px=9, odometry_dropout=0.0, texture='mnist'):
    """The generator config and the model-facing spec for a task. With the
    defaults both are exactly what they were before ``step_px``,
    ``odometry_dropout`` and ``texture`` existed. The texture sets the
    config's tile, cutoff and start window; the spec does not depend on
    it."""
    t = get(name)
    config = Config(steps=steps, queries=queries, seed=seed,
                    heading=t['heading'], query_times=query_times,
                    step=step_px, odometry_dropout=odometry_dropout,
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
        args.get('texture', 'mnist'))
    return config
