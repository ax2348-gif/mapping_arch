"""The continuous SE(2) task as a model sees it (``CONTRACT_SE2C.md``).

A model receives

    obs          (B, T + 1, S, S) float in [0, 1]  observations, S = 15,
                                                   each facing its heading
    odometry     (B, T, 3)  float   body-frame (dy, dx) px of pose t+1 seen
                                    from pose t, and the turn in [-pi, pi)
    query_pose   (B, Q, 3)  float   body-frame (dy, dx) px of the query view's
                                    centre seen from the pose at the query's
                                    step, and the heading change
    query_time   (B, Q)     long    the step each query is asked at
and returns

    pred         (B, Q, S, S) float the view at each query, facing its heading

Models encode the two pose inputs with ``mapping.se2c.encoding`` (the
shared spec). Batches are generated on the GPU inside the training step
(``data``); there is no DataLoader. ``poses``, ``query_world``,
``texture_params``, ``stats`` and ``raster`` are diagnostics only.

Metrics are those of the other tasks, with "foreground" meaning ink:
target pixels above 0.5 (on the stroke texture no pixel is exactly 0),
plus ``metrics_by_query``: error per |heading change| bin.
"""
import math
from dataclasses import dataclass, fields, replace

import torch

from mapping.task import centered_sq_err, loss_fn, metrics_by_time  # noqa: F401
from mapping.se2c import encoding
from mapping.se2c.generator import Config, make_batch
from mapping.se2c.texture import TextureSpec

INK = 0.5


@dataclass(frozen=True)
class TaskSpec:
    """What a model needs to know to size itself."""
    view: int = 15                   # S
    steps: int = 32                  # T
    queries: int = 8
    odometry_features: int = encoding.ODOMETRY_FEATURES
    query_features: int = encoding.QUERY_FEATURES


FORBIDDEN_KEYS = ('poses', 'query_world', 'texture_params', 'stats', 'raster')


def prepare(batch, device):
    """Batch (already on the GPU) to model inputs."""
    return (batch['observations'].to(device), batch['odometry'].to(device),
            batch['query_pose'].to(device), batch['target'].to(device),
            batch['query_time'].to(device))


@torch.no_grad()
def metrics(pred, target):
    """As ``mapping.task.metrics``, foreground = ink (target > 0.5)."""
    pred = pred.float()
    err = pred - target
    fg = target > INK
    return {
        'mse': err.pow(2).mean().item(),
        'mse_centered': centered_sq_err(pred, target).mean().item(),
        'mae': err.abs().mean().item(),
        'within_0.1': (err.abs() < 0.1).float().mean().item(),
        'fg_mae': err.abs()[fg].mean().item() if fg.any() else float('nan'),
        'fg_within_0.1': ((err.abs() < 0.1)[fg].float().mean().item()
                          if fg.any() else float('nan')),
        'fg_fraction': fg.float().mean().item(),
    }


# query kinds (diagnostic split, added 2026-10-16 at the user's request):
# fallback = an earlier observation turned by a quarter turn (the sampler
# found no valid pose); sampled queries by their centre's distance from the
# anchor, in px: inside half a window (a re-render of the anchor view), up to
# a window, beyond a window (needs the map)
QUERY_KINDS = ('fallback', 'near', 'mid', 'far')
NEAR_PX, FAR_PX = 7.5, 15.0


def query_kind(query_pose, found):
    """``(...)`` long index into ``QUERY_KINDS``."""
    d = query_pose[..., :2].norm(dim=-1)
    kind = torch.where(d < NEAR_PX, 1, torch.where(d < FAR_PX, 2, 3))
    return torch.where(found.bool(), kind, torch.zeros_like(kind))


def query_kinds(batch):
    """Per-query kind names for a batch, for the grid."""
    k = query_kind(batch['query_pose'], batch['stats']['found'])
    return [[QUERY_KINDS[int(x)] for x in row] for row in k.cpu()]


@torch.no_grad()
def metrics_by_query(pred, target, query_pose, found=None):
    """``mse`` per |heading change| quarter: ``mse_turn0_45``, ...,
    ``mse_turn135_180``; with ``found``, also per query kind:
    ``mse_fallback``, ``mse_sampled``, ``mse_near`` (sampled, < 7.5 px from
    the anchor), ``mse_mid`` (7.5-15 px), ``mse_far`` (>= 15 px), and each
    kind's share ``frac_<kind>``."""
    err = (pred.float() - target).pow(2).mean((-1, -2))
    deg = query_pose[..., 2].abs() * 180 / math.pi
    out = {}
    for lo in (0, 45, 90, 135):
        m = (deg >= lo) & (deg < lo + 45) if lo < 135 else deg >= lo
        out[f'mse_turn{lo}_{lo + 45}'] = (err[m].mean().item() if m.any()
                                          else float('nan'))
    if found is not None:
        kind = query_kind(query_pose, found)
        masks = {name: kind == i for i, name in enumerate(QUERY_KINDS)}
        masks['sampled'] = kind > 0
        for name, m in masks.items():
            out[f'mse_{name}'] = err[m].mean().item() if m.any() else float('nan')
            out[f'frac_{name}'] = m.float().mean().item()
    return out


@torch.no_grad()
def constant_floor(targets):
    """Metrics of predicting the per-pixel mean target."""
    stacked = torch.cat([t.flatten(0, 1) for t in targets])
    mean = stacked.mean(dim=0, keepdim=True)
    return metrics(mean.expand_as(stacked), stacked)


# ------------------------------------------------------------- the harness
def add_args(p):
    """se2c generator settings for train.py (ignored by other tasks)."""
    g = p.add_argument_group('se2c generator settings (CONTRACT_SE2C.md)')
    g.add_argument('--se2c-texture', default='procedural',
                   choices=('procedural', 'mnist', 'imagenette',
                            'imagenette_hp'))
    g.add_argument('--turn-n', type=int, default=0,
                   help='0: continuous turns; n: multiples of 2 pi / n')
    g.add_argument('--motion', default=Config.motion, choices=('arc', 'decoupled'))
    g.add_argument('--directions', default=Config.directions,
                   choices=('persistent', 'lattice'),
                   help="decoupled: 'lattice' = the discrete walk's no-reversal rule on the n-lattice")
    g.add_argument('--v-min', type=float, default=Config.v_min)
    g.add_argument('--v-max', type=float, default=Config.v_max)
    g.add_argument('--omega-max', type=float, default=math.pi)
    g.add_argument('--holdout', type=float, nargs=2, default=None,
                   metavar=('LO', 'HI'), help='band of |turn| in rad')
    g.add_argument('--holdout-mode', default='exclude',
                   choices=('exclude', 'include', 'only'))
    g.add_argument('--query-coverage', default='causal',
                   choices=('causal', 'global'))


# Config fields a run's args record does not carry under the same name
# (the episode settings are task fields; the texture is ``se2c_texture``)
_NOT_IN_ARGS = ('view', 'steps', 'queries', 'seed', 'texture', 'bank', 'holdout',
                'holdout_mode', 'query_times', 'motion', 'directions')


def config_from_args(args, eval_holdout=None):
    """Generator config from a run's ``args`` record (render.py, tools);
    ``eval_holdout`` overrides the held-out mode (evaluation sees the
    band). Records older than a setting get its value before the setting
    existed. Any other Config field the record carries is passed through,
    so a new generator setting reaches the tools with no code here."""
    a = dict(args)
    bank = a.get('se2c_texture', 'procedural')
    same = {f.name: a[f.name] for f in fields(Config)
            if f.name not in _NOT_IN_ARGS and f.name in a}
    same.setdefault('v_min', Config.v_min)
    same.setdefault('v_max', Config.v_max)
    same.setdefault('omega_max', math.pi)
    same.setdefault('turn_n', 0)
    same.setdefault('query_coverage', 'causal')
    same.setdefault('odometry_dropout', 0.0)  # records before the se2c knob
    return Config(steps=a.get('episode_steps', 32), queries=a.get('queries', 8),
                  motion=a.get('motion', 'arc'),
                  directions=a.get('directions', 'lattice' if a.get('motion') == 'decoupled'
                                   and a.get('turn_n') else 'persistent'),
                  holdout=tuple(a.get('holdout') or ()),
                  holdout_mode=eval_holdout or a.get('holdout_mode', 'exclude'),
                  query_times=a.get('query_times', 'uniform'),
                  bank='' if bank == 'procedural' else bank,
                  texture=TextureSpec(), seed=a.get('seed', 0), **same)


def data(args, device, start=0, eval_batches=None, batch=None):
    """``from_config`` for a run's ``args`` record (render.py, tools)."""
    a = dict(args)
    return from_config(config_from_args(a), device, start,
                       eval_batches if eval_batches is not None else a.get('eval_batches', 16),
                       batch or a.get('batch', 64))


def from_config(config, device, start=0, eval_batches=16, batch=64):
    """Everything train.py needs: ``config``, ``spec``, ``stream`` (an
    iterator of GPU batches starting at batch ``start``), ``held_out``
    (fixed evaluation batches from the evaluation stream: the band
    included when one is held out, the test split of a tile bank) and
    ``info`` (settings recorded with the run)."""
    ev_config = replace(config, holdout_mode='include') if config.holdout else config
    train_bank = eval_bank = None
    if config.bank:
        from mapping.se2c import bank as tb
        train_bank = tb.load(config.bank, 'train', device)
        eval_bank = tb.load(config.bank, 'test', device)
    spec = TaskSpec(view=config.view, steps=config.steps,
                    queries=config.queries)

    def stream():
        i = start
        while True:
            yield make_batch(config, i, device, batch, bank=train_bank)
            i += 1
    held_out = [make_batch(ev_config, i, device, batch, eval_set=True,
                           bank=eval_bank) for i in range(eval_batches)]
    info = {'texture': config.bank or 'procedural',
            'texture_spec': config.texture.__dict__,
            'generator': {k: v for k, v in config.__dict__.items()
                          if k != 'texture'}}
    return {'config': config, 'spec': spec, 'stream': stream(),
            'held_out': held_out, 'info': info}
