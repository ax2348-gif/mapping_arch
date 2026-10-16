"""Checkpoints: what a run saves and how to get a model back from it.

A run directory ``results/<task>/<model>/<size>/<run>/`` (``run.dir``) holds

    final.pt      model weights plus the arguments, the task spec and
                  the architecture (class and constructor arguments)
                  that built it, enough to rebuild the model with no
                  other information
    final.json    the summary a scaling fit reads: parameters, steps,
                  final and best evaluation, the shuffled-direction
                  evaluation, the constant floor, timing, arguments
    log.jsonl     one record per logging interval
    grid.png      sampled predictions beside their targets

``load_checkpoint`` rebuilds the model from ``final.pt`` alone.
"""
import torch

import mapping.tasks as tasks


def save_checkpoint(path, model, args, spec, task='z2', arch=None):
    """``arch``: ``mapping.config.Model.record()``, the class and
    constructor arguments the model was built with."""
    state = {'model': model.state_dict(), 'args': dict(args),
             'spec': spec.__dict__, 'task': task}
    if arch is not None:
        state['arch'] = arch
    torch.save(state, path)


def build_from_arch(arch, spec):
    """The module a checkpoint's ``arch`` record describes."""
    import importlib
    module, _, name = arch['class'].rpartition('.')
    return getattr(importlib.import_module(module), name)(spec, **arch['kwargs'])


def load_checkpoint(path, device='cpu'):
    """``(model in eval mode, args dict, TaskSpec, task name)`` from
    ``final.pt`` or a mid-run ``step<k>.pt``. The task name is the
    current one (``tasks.canonical``: an old checkpoint's ``se2`` is
    ``se2d``). The model is built from the
    checkpoint's ``arch`` record; checkpoints from before it existed use
    the registered module's ``build(size, spec)``, and ones from before
    tasks existed are z2."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    args = state['args']
    if 'spec' in state:
        task = state.get('task', 'z2')
        spec = tasks.get(task)['task'].TaskSpec(**state['spec'])
    else:   # a mid-run step<k>.pt: rebuild the spec from the run's arguments
        task = args['task']
        _, spec = tasks.config_and_spec(
            task, args['episode_steps'], args['queries'], args['seed'],
            args.get('query_times', 'last'), args.get('step_px', 9),
            args.get('odometry_dropout', 0.0), args.get('texture', 'mnist'))
    if 'arch' in state:
        model = build_from_arch(state['arch'], spec)
    else:
        model = tasks.get(task)['models'].get(args['model']).build(args['size'], spec)
    model.load_state_dict(state['model'])
    return model.to(device).eval(), args, spec, tasks.canonical(task)
