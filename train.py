"""Train one model at one size on the offline mapping task.

    python train.py --model vanilla --size xs --steps 500

Streams training episodes from the MNIST train bank, evaluates on a fixed
set from the test bank, and writes the run directory
``results/<model>/<size>/<run>/`` (see ``mapping/checkpoint.py`` for its
contents). Every number a scaling fit needs is in ``final.json``, and
``grid.png`` shows sampled predictions beside their targets.

Evaluation is reported twice at every interval: on the held-out set, and
on the same set with each episode's directions permuted in time. A model
whose error does not rise under the permutation is not using the
odometry, whatever its loss says.
"""
import argparse
import json
import math
import os
import time

import torch

import mapping.tasks as tasks
from mapping.checkpoint import save_checkpoint
from mapping.loader import eval_set, load_banks, train_loader
from mapping.plot import plot_run
from mapping.render import render_grid, run_title

DEFAULT_LR = 3e-4


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='z2', choices=sorted(tasks.TASKS),
                   help='which contract: z2 (CONTRACT.md) or se2 '
                        '(CONTRACT_SE2.md); picks the model registry, the '
                        'batch layout and the results root')
    p.add_argument('--model', required=True)
    p.add_argument('--size', required=True)
    p.add_argument('--run', default=None,
                   help='run directory name; default seed<seed>')
    p.add_argument('--out', default=None,
                   help='explicit run directory, overriding --run')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=None,
                   help="default: the model module's LR[size] if it "
                        f'defines one, else {DEFAULT_LR}')
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--eval-batches', type=int, default=16)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--bank', default='data/mnist.npz')
    p.add_argument('--no-bf16', action='store_true')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    # data knobs that change the task; the defaults are the design
    p.add_argument('--episode-steps', type=int, default=32)
    p.add_argument('--queries', type=int, default=8)
    return p.parse_args()


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, batches, device, autocast, prepare, metrics,
             shuffle_directions=False):
    """``shuffle_directions`` permutes each episode's odometry tokens in
    time (directions on z2, symbols on se2)."""
    model.eval()
    preds, targets = [], []
    gen = torch.Generator().manual_seed(0)
    for b in batches:
        obs, directions, offset, target = prepare(b, device)
        if shuffle_directions:
            perm = torch.argsort(torch.rand(directions.shape, generator=gen),
                                 dim=1).to(device)
            directions = torch.gather(directions, 1, perm)
        with autocast:
            pred = model(obs, directions, offset)
        preds.append(pred.float())
        targets.append(target)
    model.train()
    return metrics(torch.cat(preds), torch.cat(targets))


def main():
    args = parse()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    task = tasks.get(args.task)
    models, prepare = task['models'], task['task'].prepare
    loss_fn, metrics = task['task'].loss_fn, task['task'].metrics
    constant_floor = task['task'].constant_floor
    run = args.run or f'seed{args.seed}'
    out = args.out or os.path.join(task['root'], args.model, args.size, run)
    os.makedirs(out, exist_ok=True)

    config, spec = tasks.config_and_spec(args.task, args.episode_steps,
                                         args.queries, args.seed)
    train_bank, eval_bank = load_banks(args.bank, config)
    loader = train_loader(config, train_bank, args.batch, args.workers)
    held_out = eval_set(config, eval_bank, args.batch, args.eval_batches,
                        args.seed)
    floor = constant_floor([b['target'].float() / 255 for b in held_out])

    module = models.get(args.model)
    if args.lr is None:
        args.lr = getattr(module, 'LR', {}).get(args.size, DEFAULT_LR)
    model = module.build(args.size, spec).to(device)
    params = models.count_params(model)
    raw_model = model
    if args.compile:
        model = torch.compile(model)
    use_bf16 = device.type == 'cuda' and not args.no_bf16
    autocast = torch.autocast('cuda', dtype=torch.bfloat16,
                              enabled=use_bf16)

    # 1-D tensors (norms, biases) never decay. A model module may exclude
    # more by name with NO_DECAY(name) -> bool, e.g. recurrent matrices.
    exclude = getattr(module, 'NO_DECAY', lambda name: False)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.dim() >= 2 and not exclude(name)
         else no_decay).append(p)
    opt = torch.optim.AdamW([{'params': decay,
                              'weight_decay': args.weight_decay},
                             {'params': no_decay, 'weight_decay': 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))

    print(f'{args.task} {args.model}/{args.size} -> {out}: {params:,} params, '
          f'lr {args.lr:.2g}; floor mse {floor["mse"]:.4f}, '
          f'floor fg_mae {floor["fg_mae"]:.4f}', flush=True)
    log = open(os.path.join(out, 'log.jsonl'), 'w')
    best, best_step = None, -1
    t_start = time.perf_counter()
    running, since = 0.0, 0
    it = iter(loader)
    rec = None
    for step in range(args.steps):
        for g in opt.param_groups:
            g['lr'] = lr_at(step, args)
        obs, directions, offset, target = prepare(next(it), device)
        with autocast:
            pred = model(obs, directions, offset)
        loss = loss_fn(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        running += loss.item()
        since += 1

        last = step == args.steps - 1
        if (step + 1) % args.log_every == 0 or last:
            elapsed = time.perf_counter() - t_start
            rec = {'step': step + 1, 'train_mse': running / since,
                   'lr': lr_at(step, args), 'elapsed_s': elapsed,
                   'steps_per_s': (step + 1) / elapsed}
            running, since = 0.0, 0
            if (step + 1) % args.eval_every == 0 or last:
                ev = evaluate(model, held_out, device, autocast, prepare,
                              metrics)
                rec['eval'] = ev
                rec['eval_shuffled'] = evaluate(model, held_out, device,
                                                autocast, prepare, metrics,
                                                True)
                if best is None or ev['mse'] < best['mse']:
                    best, best_step = ev, step + 1
                print(f'step {step + 1:6d} train {rec["train_mse"]:.4f} '
                      f'eval mse {ev["mse"]:.4f} fg_mae {ev["fg_mae"]:.4f} '
                      f'fg<0.1 {ev["fg_within_0.1"]:.3f} '
                      f'| dirs shuffled mse '
                      f'{rec["eval_shuffled"]["mse"]:.4f} '
                      f'({rec["steps_per_s"]:.1f} it/s)', flush=True)
            log.write(json.dumps(rec) + '\n')
            log.flush()
    log.close()

    final = {'task': args.task, 'model': args.model, 'size': args.size,
             'run': run,
             'params': params, 'steps': args.steps, 'batch': args.batch,
             'episodes_seen': args.steps * args.batch,
             'final_eval': rec['eval'],
             'final_eval_shuffled': rec['eval_shuffled'],
             'best_eval': best, 'best_step': best_step, 'floor': floor,
             'scaling_rule': getattr(module, 'SCALING_RULE', ''),
             'size_kwargs': module.SIZES[args.size],
             'elapsed_s': time.perf_counter() - t_start,
             'args': vars(args)}
    with open(os.path.join(out, 'final.json'), 'w') as f:
        json.dump(final, f, indent=1)
    save_checkpoint(os.path.join(out, 'final.pt'), raw_model, vars(args),
                    spec, args.task)
    render_grid(raw_model, held_out[:2], spec.view,
                os.path.join(out, 'grid.png'), device=device,
                autocast=autocast, title=run_title(final), prepare=prepare)
    plot_run(out, final)
    print(f'done: eval mse {final["final_eval"]["mse"]:.4f} '
          f'(shuffled {final["final_eval_shuffled"]["mse"]:.4f}, '
          f'floor {floor["mse"]:.4f}); wrote {out}')


if __name__ == '__main__':
    main()
