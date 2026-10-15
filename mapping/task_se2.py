"""The SE(2) task as a model sees it (``CONTRACT_SE2.md``). A model receives

    obs           (B, T + 1, S, S) float in [0, 1]   observations, each
                                                     facing its own heading
    odometry      (B, T)           long in 0..15     relative pose t -> t+1
                                                     in the body frame of t,
                                                     4 * turn + move
    query_pose    (B, Q, 3)        long              body-frame (dy, dx) of
                                                     the query from the
                                                     last pose, and the
                                                     heading change 0..3
and returns

    pred          (B, Q, S, S) float                 the view at each query,
                                                     facing the query heading

``positions`` and ``headings`` are in the batch for diagnostics only.
Loss, metrics and the constant floor are those of the Z^2 task.
"""
from dataclasses import dataclass

from mapping.task import constant_floor, loss_fn, metrics  # noqa: F401


@dataclass(frozen=True)
class TaskSpec:
    """What a model needs to know to size itself."""
    view: int = 9            # S
    steps: int = 32          # T; T + 1 observations
    odometry: int = 16       # symbols: 4 turns x 4 moves
    turns: int = 4
    step_px: int = 9
    queries: int = 8

    @property
    def offset_max(self):
        return self.steps * self.step_px


FORBIDDEN_KEYS = ('positions', 'headings')


def prepare(batch, device):
    """Batch of tensors from the loader to model inputs on ``device``."""
    obs = batch['observations'].to(device, non_blocking=True).float() / 255
    odometry = batch['odometry'].to(device, non_blocking=True)
    pose = batch['query_pose'].to(device, non_blocking=True).long()
    target = batch['target'].to(device, non_blocking=True).float() / 255
    return obs, odometry, pose, target
