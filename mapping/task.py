"""The task as a model sees it: tensors, loss, metrics, floors.

Scope: the single offline setting with relative odometry. A model receives

    obs           (B, T + 1, S, S) float in [0, 1]   the observations
    directions    (B, T)           long in 0..3      step direction t -> t+1
    query_offset  (B, Q, 2)        long              query corner minus the
                                                     corner of the window
                                                     at the query's step,
                                                     in pixels (y, x)
    query_time    (B, Q)           long in 0..T      the step each query is
                                                     asked at; all T unless
                                                     the run varies them
and returns

    pred          (B, Q, S, S) float                 the view at each query

Nothing else reaches the model. Absolute positions exist in the batch for
diagnostics only; a model that reads them is cheating. Integrating the
directions into positions is the model's job, inside its parameters.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TaskSpec:
    """What a model needs to know to size itself."""
    view: int = 9            # S
    steps: int = 32          # T; T + 1 observations
    directions: int = 4      # 5 under odometry dropout: 4 is "unknown"
    step_px: int = 9
    queries: int = 8

    @property
    def offset_max(self):
        """Query offsets lie in [-offset_max, offset_max] on each axis: the
        farthest a window can be from the last one is T steps back."""
        return self.steps * self.step_px


# directions_true: the undropped symbols under odometry dropout, diagnostics
FORBIDDEN_KEYS = ('positions', 'directions_true')


def prepare(batch, device):
    """Batch of tensors from the loader to model inputs on ``device``."""
    obs = batch['observations'].to(device, non_blocking=True).float() / 255
    directions = batch['directions'].to(device, non_blocking=True)
    offset = batch['query_offset'].to(device, non_blocking=True).long()
    target = batch['target'].to(device, non_blocking=True).float() / 255
    time = batch['query_time'].to(device, non_blocking=True)
    return obs, directions, offset, target, time


def loss_fn(pred, target):
    return F.mse_loss(pred.float(), target)


@torch.no_grad()
def metrics(pred, target):
    """Per-pixel errors, plus the same on stroke pixels only.

    Background is most of a target, so a model that draws nothing scores
    well on the whole-view numbers; ``fg_*`` is where the task lives.
    """
    pred = pred.float()
    err = pred - target
    fg = target > 0
    return {
        'mse': err.pow(2).mean().item(),
        'mae': err.abs().mean().item(),
        'within_0.1': (err.abs() < 0.1).float().mean().item(),
        'fg_mae': err.abs()[fg].mean().item() if fg.any() else float('nan'),
        'fg_within_0.1': ((err.abs() < 0.1)[fg].float().mean().item()
                          if fg.any() else float('nan')),
        'fg_fraction': fg.float().mean().item(),
    }


@torch.no_grad()
def metrics_by_time(pred, target, time, steps):
    """``mse`` of the queries asked at the last step, and per prefix-length
    bin (quarters of the episode), as a dict; nan where a bin is empty."""
    pred, err = pred.float(), (pred.float() - target).pow(2).mean((-1, -2))
    out = {'mse_last': err[time == steps].mean().item()
           if (time == steps).any() else float('nan')}
    edges = [0, steps // 4, steps // 2, 3 * steps // 4, steps]
    for lo, hi in zip(edges, edges[1:]):
        m = (time >= lo) & (time < hi) if hi < steps else (time >= lo)
        out[f'mse_t{lo}_{hi}'] = err[m].mean().item() if m.any() else float('nan')
    return out


@torch.no_grad()
def constant_floor(targets):
    """Metrics of predicting the per-pixel mean target: the floor a model
    with no information at all reaches. ``targets`` is a list of
    ``(B, Q, S, S)`` tensors."""
    stacked = torch.cat([t.flatten(0, 1) for t in targets])
    mean = stacked.mean(dim=0, keepdim=True)
    return metrics(mean.expand_as(stacked), stacked)
