"""Checkpoints: what a run saves and how to get a model back from it.

A run directory ``results/<model>/<size>/<run>/`` holds

    final.pt      model weights plus the arguments and task spec that
                  built it, enough to rebuild the model with no other
                  information
    final.json    the summary a scaling fit reads: parameters, steps,
                  final and best evaluation, the shuffled-direction
                  evaluation, the constant floor, timing, arguments
    log.jsonl     one record per logging interval
    grid.png      sampled predictions beside their targets

``load_checkpoint`` rebuilds the model from ``final.pt`` alone.
"""
import torch

import mapping.tasks as tasks


def save_checkpoint(path, model, args, spec, task='z2'):
    torch.save({'model': model.state_dict(), 'args': dict(args),
                'spec': spec.__dict__, 'task': task}, path)


def load_checkpoint(path, device='cpu'):
    """``(model in eval mode, args dict, TaskSpec, task name)`` from
    ``final.pt`` or a mid-run ``step<k>.pt``. Checkpoints written before
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
            args['query_times'], args.get('step_px', 9),
            args.get('odometry_dropout', 0.0), args.get('texture', 'mnist'))
    t = tasks.get(task)
    model = t['models'].get(args['model']).build(args['size'], spec)
    model.load_state_dict(state['model'])
    return model.to(device).eval(), args, spec, task
