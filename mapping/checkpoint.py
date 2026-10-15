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

import mapping.models as models
from mapping.task import TaskSpec


def save_checkpoint(path, model, args, spec):
    torch.save({'model': model.state_dict(), 'args': dict(args),
                'spec': spec.__dict__}, path)


def load_checkpoint(path, device='cpu'):
    """``(model in eval mode, args dict, TaskSpec)`` from ``final.pt``."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    args, spec = state['args'], TaskSpec(**state['spec'])
    model = models.get(args['model']).build(args['size'], spec)
    model.load_state_dict(state['model'])
    return model.to(device).eval(), args, spec
