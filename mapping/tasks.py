"""The tasks the harness can train: one row per contract.

    models   the model registry for the task
    task     the module with TaskSpec, prepare, loss_fn, metrics, floor
    root     where its runs live
    heading  the generator switch (mapping.data.Config)

``z2`` is the default everywhere, so code written before ``se2`` existed
behaves as it did. Checkpoints record their task; ones without it are z2.
"""
import mapping.models
import mapping.models.se2
import mapping.task
import mapping.task_se2
from mapping.data import Config

TASKS = {
    'z2': dict(models=mapping.models, task=mapping.task, root='results',
               heading=False),
    'se2': dict(models=mapping.models.se2, task=mapping.task_se2,
                root='results/se2', heading=True),
}


def get(name='z2'):
    if name not in TASKS:
        raise KeyError(f'unknown task {name!r}; have {sorted(TASKS)}')
    return TASKS[name]


def config_and_spec(name, steps, queries, seed):
    """The generator config and the model-facing spec for a task."""
    t = get(name)
    config = Config(steps=steps, queries=queries, seed=seed,
                    heading=t['heading'])
    spec = t['task'].TaskSpec(view=config.obs, steps=config.steps,
                              step_px=config.step, queries=config.queries)
    return config, spec
