"""The se2c harness path: data(), metrics, the shuffled-odometry
permutation, and the task registry entry."""
import math

import torch

import mapping.tasks as tasks
from mapping import task_se2c
from train import permute_steps

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_data_and_floor():
    args = {'seed': 0, 'batch': 8, 'query_times': 'uniform', 'turn_n': 8,
            'holdout': [0.5, 1.0]}
    d = task_se2c.data(args, DEV, eval_batches=2)
    assert d['config'].holdout_mode == 'exclude'
    tr = next(d['stream'])
    a = tr['odometry'][..., 2].abs()
    assert not ((a > 0.5) & (a < 1.0)).any()           # 2 pi / 8 = 0.785
    ev = torch.cat([b['odometry'][..., 2].abs() for b in d['held_out']])
    assert ((ev > 0.5) & (ev < 1.0)).any()             # evaluation sees it
    floor = task_se2c.constant_floor([b['target'] for b in d['held_out']])
    assert 0.1 < floor['mse'] < 0.16
    assert next(d['stream'])['observations'].shape == (8, 33, 15, 15)


def test_metrics_by_query_bins():
    pred = torch.zeros(2, 4, 15, 15)
    target = torch.ones(2, 4, 15, 15)
    qp = torch.zeros(2, 4, 3)
    qp[..., 2] = torch.tensor([0.1, -1.0, 2.0, -3.1])
    m = task_se2c.metrics_by_query(pred, target, qp)
    assert set(m) == {'mse_turn0_45', 'mse_turn45_90', 'mse_turn90_135',
                      'mse_turn135_180'}
    assert all(v == 1.0 for v in m.values())


def test_permute_steps_matches_gather_for_symbols():
    x = torch.randint(0, 16, (4, 32))
    perm = torch.argsort(torch.rand(4, 32), 1)
    assert torch.equal(permute_steps(x, perm), torch.gather(x, 1, perm))
    v = torch.randn(4, 32, 3)
    p = permute_steps(v, perm)
    assert torch.equal(p[..., 1], torch.gather(v[..., 1], 1, perm))


def test_registry_entry():
    t = tasks.get('se2c')
    assert tasks.results_dir('se2c') == 'results/se2c' and t['gpu_data']
    config, spec = tasks.config_and_spec('se2c', 32, 8, 0)
    assert spec == task_se2c.TaskSpec() and config.view == 15
