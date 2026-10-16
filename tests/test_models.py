"""Every registered model, on both tasks, at its smallest rung:
the forward has the contract's shape, one episode cannot influence
another, and a model that takes query times gives the same output with
none as with every query at the last step."""
import inspect

import pytest
import torch

import mapping.tasks as tasks

CUDA = torch.cuda.is_available()


def cases():
    for task in sorted(tasks.TASKS):
        for name in sorted(tasks.get(task)['models'].REGISTRY):
            yield task, name


def inputs(task, b=4, q=8, device='cpu'):
    obs = torch.rand(b, 33, 9, 9, device=device)
    if task == 'z2':
        odo = torch.randint(0, 4, (b, 32), device=device)
        query = torch.randint(-288, 289, (b, q, 2), device=device)
    else:
        odo = torch.randint(0, 16, (b, 32), device=device)
        query = torch.cat([torch.randint(-288, 289, (b, q, 2)),
                           torch.randint(0, 4, (b, q, 1))], -1).to(device)
    return obs, odo, query


@pytest.mark.parametrize('task,name', list(cases()))
def test_contract_forward(task, name):
    t = tasks.get(task)
    module = t['models'].get(name)
    size = next(iter(module.SIZES))
    try:
        net = module.build(size, t['task'].TaskSpec())
    except Exception as e:                       # e.g. a kernel needing CUDA
        pytest.skip(f'cannot build on this machine: {e}')
    device = 'cuda' if CUDA else 'cpu'
    net = net.to(device).eval()
    obs, odo, query = inputs(task, device=device)
    takes_time = 'query_time' in inspect.signature(net.forward).parameters
    kw = ({'query_time': torch.randint(0, 33, query.shape[:2], device=device)}
          if takes_time else {})
    try:
        with torch.no_grad():
            out = net(obs, odo, query, **kw).float()
    except Exception as e:
        if not CUDA:
            pytest.skip(f'needs CUDA: {e}')
        raise
    assert out.shape == (4, 8, 9, 9)
    obs2, odo2 = obs.clone(), odo.clone()
    obs2[0] = torch.rand_like(obs2[0])
    odo2[0] = (odo2[0] + 1) % (4 if task == 'z2' else 16)
    with torch.no_grad():
        out2 = net(obs2, odo2, query, **kw).float()
    assert (out[1:] - out2[1:]).abs().max().item() < 1e-4, 'episodes leak'
    if takes_time:
        last = torch.full(query.shape[:2], 32, device=device)
        with torch.no_grad():
            a = net(obs, odo, query).float()
            b = net(obs, odo, query, query_time=last).float()
        assert (a - b).abs().max().item() < 1e-5
