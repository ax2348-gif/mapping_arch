"""Small actual-kernel check before the approved hybrid6 Torch training job.

Synthetic inputs test execution and checkpoint reconstruction, not learning.
Run as python -m experiments.hybrid6.gpu_preflight inside the allocation.
"""
import io
import json

import torch

import run
from mapping.checkpoint import build_from_arch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU preflight requires a CUDA allocation')
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    exp = run.build(run.resolved(run.load(['+experiment=hybrid6_initial'])))
    _, spec = exp.task.generator.data(exp.task, exp.run.seed)
    net = exp.model.build(spec).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=exp.optim.lr,
                           weight_decay=exp.optim.weight_decay)
    # Full sequence and query lengths; small batch keeps validation cheap.
    obs = torch.rand(2, 33, 9, 9, device='cuda')
    odo = torch.randint(16, (2, 32), device='cuda')
    pose = torch.cat((torch.randint(-100, 101, (2, 8, 2), device='cuda'),
                      torch.randint(4, (2, 8, 1), device='cuda')), -1)
    time = torch.randint(33, (2, 8), device='cuda')
    target = torch.rand(2, 8, 9, 9, device='cuda')
    losses = []
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = net(obs, odo, pose, time)
            loss = (pred.float() - target).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite GPU loss')
        loss.backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError('missing or nonfinite GPU gradients')
        torch.nn.utils.clip_grad_norm_(net.parameters(), exp.optim.clip)
        opt.step()
        losses.append(loss.item())
    for name in ('roles', 'pose_embed', 'sensor_embed', 'patch_embed'):
        if getattr(net, name).weight.grad.abs().sum() == 0:
            raise RuntimeError(f'no gradient reached {name}')
    # Exercise the same architecture record used by production checkpoints.
    buffer = io.BytesIO()
    torch.save({'arch': exp.model.record(), 'model': net.state_dict(),
                'optimizer': opt.state_dict()}, buffer)
    buffer.seek(0)
    state = torch.load(buffer, map_location='cpu', weights_only=True)
    restored = build_from_arch(state['arch'], spec).cuda()
    restored.load_state_dict(state['model'])
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=exp.optim.lr)
    restored_opt.load_state_dict(state['optimizer'])
    if len(restored_opt.state) != len(opt.state):
        raise RuntimeError('optimizer checkpoint state not restored')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        net.eval()
        restored.eval()
        expected = net(obs, odo, pose, time)
        actual = restored(obs, odo, pose, time)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.cuda.synchronize()
    print(json.dumps({'status': 'passed', 'device': torch.cuda.get_device_name(),
                      'losses': losses, 'parameters': sum(p.numel() for p in net.parameters()),
                      'peak_bytes': torch.cuda.max_memory_allocated(),
                      'checkpoint_exact': True}), flush=True)


if __name__ == '__main__':
    main()
