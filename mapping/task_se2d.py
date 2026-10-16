"""The SE(2) task as a model sees it (``CONTRACT_SE2D.md``). A model receives

    obs           (B, T + 1, S, S) float in [0, 1]   observations, each
                                                     facing its own heading
    odometry      (B, T)           long in 0..15     relative pose t -> t+1
                                                     in the body frame of t,
                                                     4 * turn + move
    query_pose    (B, Q, 3)        long              body-frame (dy, dx) of
                                                     the query from the pose
                                                     at the query's step,
                                                     and the heading change
    query_time    (B, Q)           long in 0..T      the step each query is
                                                     asked at; all T unless
                                                     the run varies them
and returns

    pred          (B, Q, S, S) float                 the view at each query,
                                                     facing the query heading

``positions`` and ``headings`` are in the batch for diagnostics only, as is
``odometry_true`` under odometry dropout.
Loss, metrics and the constant floor are those of the Z^2 task.
"""
from dataclasses import dataclass

from mapping.task import (constant_floor, loss_fn, metrics,  # noqa: F401
                          metrics_by_time)


@dataclass(frozen=True)
class TaskSpec:
    """What a model needs to know to size itself."""
    view: int = 9            # S
    steps: int = 32          # T; T + 1 observations
    odometry: int = 16       # symbols: 4 turns x 4 moves; 17 under
                             # odometry dropout, 16 being "unknown"
    turns: int = 4
    step_px: int = 9
    queries: int = 8

    @property
    def offset_max(self):
        return self.steps * self.step_px


# odometry_true: the undropped symbols under odometry dropout, diagnostics
FORBIDDEN_KEYS = ('positions', 'headings', 'odometry_true')


def prepare(batch, device):
    """Batch of tensors from the loader to model inputs on ``device``."""
    obs = batch['observations'].to(device, non_blocking=True).float() / 255
    odometry = batch['odometry'].to(device, non_blocking=True)
    pose = batch['query_pose'].to(device, non_blocking=True).long()
    target = batch['target'].to(device, non_blocking=True).float() / 255
    time = batch['query_time'].to(device, non_blocking=True)
    return obs, odometry, pose, target, time
