"""``mse_centered`` (``mapping.task.metrics``): reported beside the
existing metrics, blind to a per-query brightness offset, and leaving the
loss and every existing metric value as they were."""
import torch
import torch.nn.functional as F

from mapping import task, task_se2


def sample(seed=0):
    g = torch.Generator().manual_seed(seed)
    target = torch.rand(4, 8, 9, 9, generator=g)
    pred = torch.rand(4, 8, 9, 9, generator=g)
    return pred, target


def test_constant_offset_scores_zero():
    _, target = sample()
    shift = torch.randn(4, 8, 1, 1)
    m = task.metrics(target + shift, target)
    assert m['mse_centered'] < 1e-12
    assert m['mse'] > 0.1                        # plain MSE does see it
    time = torch.randint(0, 33, (4, 8))
    time[0] = 32
    by_t = task.metrics_by_time(target + shift, target, time, 32)
    assert by_t['mse_centered_last'] < 1e-12


def test_definition():
    pred, target = sample(1)
    pc = pred - pred.mean((-1, -2), keepdim=True)
    tc = target - target.mean((-1, -2), keepdim=True)
    want = (pc - tc).pow(2).mean((-1, -2)).mean().item()
    assert abs(task.metrics(pred, target)['mse_centered'] - want) < 1e-7
    assert task_se2.metrics is task.metrics


def test_existing_values_unchanged():
    pred, target = sample(2)
    err = pred - target
    m = task.metrics(pred, target)
    assert m['mse'] == err.pow(2).mean().item()
    assert m['mae'] == err.abs().mean().item()
    assert task.loss_fn(pred, target).item() == F.mse_loss(pred, target).item()
    time = torch.randint(0, 33, (4, 8), generator=torch.Generator(
        ).manual_seed(0))
    time[0] = 32                                 # mse_last is not empty
    by_t = task.metrics_by_time(pred, target, time, 32)
    per_q = err.pow(2).mean((-1, -2))
    assert by_t['mse_last'] == per_q[time == 32].mean().item()
    floor = task.constant_floor([target])
    assert 'mse_centered' in floor and floor['mse'] > 0
