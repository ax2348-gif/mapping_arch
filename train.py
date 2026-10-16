"""Train one model at one size on the offline mapping task.

    python train.py --model vanilla --size xs --steps 500

Streams training episodes from the train split of the tile bank (MNIST
by default; ``--texture imagenette`` for natural images, see
``mapping.data.TEXTURES``), evaluates on a fixed set from its test split,
and writes the run directory
``results/<model>/<size>/<run>/`` (see ``mapping/checkpoint.py`` for its
contents). Every number a scaling fit needs is in ``final.json``, and
``grid.png`` shows sampled predictions beside their targets.

Evaluation is reported twice at every interval: on the held-out set, and
on the same set with each episode's directions permuted in time. A model
whose error does not rise under the permutation is not using the
odometry, whatever its loss says.

Schedules. ``cosine`` (the original) warms up then decays to zero at
``--steps``. ``wsd`` (warmup-stable-decay) holds the peak rate and decays
linearly to zero over the last ``--cooldown`` fraction of ``--steps``.
With ``--save-every N`` the run writes ``step<k>.pt`` (model, optimizer,
step) during training, and ``--branch step<k>.pt --steps S`` starts a new
run from that state, continuing the data stream where it left off, with
the schedule computed for total length S. One long stable run plus one
short branch per budget gives a compute curve under one recipe.
"""
import argparse
import inspect
import json
import math
import os
import time

import torch

import mapping.tasks as tasks
from mapping.data import TEXTURES, check_texture
from mapping.checkpoint import save_checkpoint
from mapping.loader import eval_set, load_banks, train_loader
from mapping.plot import plot_run
from mapping.render import render_grid, run_title
from mapping.task import metrics_by_time
from mapping.task_se2c import add_args as add_se2c_args

DEFAULT_LR = 3e-4


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='z2', choices=sorted(tasks.ALL),
                   help='which contract: z2 (CONTRACT.md), se2 '
                        '(CONTRACT_SE2.md) or se2c (CONTRACT_SE2C.md); picks the model registry, the '
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
    p.add_argument('--schedule', default='cosine', choices=('cosine', 'wsd'))
    p.add_argument('--cooldown', type=float, default=0.2,
                   help='wsd: fraction of --steps spent decaying to zero')
    p.add_argument('--save-every', type=int, default=0,
                   help='write step<k>.pt (with optimizer) every k steps')
    p.add_argument('--branch', default=None,
                   help='a step<k>.pt to continue from, into a new run')
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--eval-batches', type=int, default=16)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--texture', default='mnist', choices=sorted(TEXTURES),
                   help='tile bank (generator setting, not a new task): '
                        'sets the tile side, brightness cutoff and start '
                        'window together (mapping.data.TEXTURES); the bank '
                        'file must have that tile side and identity tag '
                        '(mapping.data.check_texture) or the run refuses')
    p.add_argument('--bank', default=None,
                   help="bank file; default the texture's "
                        '(data/mnist.npz, data/imagenette64.npz, '
                        'data/imagenette64_hp.npz)')
    p.add_argument('--no-bf16', action='store_true')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    # data knobs that change the task; the defaults are the design
    p.add_argument('--episode-steps', type=int, default=32)
    p.add_argument('--queries', type=int, default=8)
    p.add_argument('--query-times', default='last', choices=('last', 'uniform'),
                   help="'last': queries at the final step (the original "
                        "tasks); 'uniform': each query at a step drawn "
                        "uniformly from 0..T, answerable from the "
                        "observations up to it")
    p.add_argument('--step-px', type=int, default=9,
                   help='walk step in pixels (generator setting, not a new '
                        'task); below the 9-pixel window consecutive '
                        'observations overlap. offset_max = steps * step_px')
    p.add_argument('--odometry-dropout', type=float, default=0.0,
                   help='probability that each step\'s odometry symbol is '
                        'replaced by a reserved "unknown" symbol (generator '
                        'setting, not a new task); 1.0 hides all odometry')
    add_se2c_args(p)
    return p.parse_args()


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    if args.schedule == 'wsd':
        decay_start = int(round(args.steps * (1 - args.cooldown)))
        if step < decay_start:
            return args.lr
        return args.lr * max(0.0, (args.steps - step)
                             / max(1, args.steps - decay_start))
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * 0.5 * (1 + math.cos(math.pi * progress))


def permute_steps(x, perm):
    """``x (B, T, ...)`` with each row's steps permuted by ``perm (B, T)``
    (symbols on z2/se2, pose vectors on se2c)."""
    idx = perm.view(*perm.shape, *([1] * (x.dim() - 2))).expand_as(x)
    return torch.gather(x, 1, idx)


def call(model, obs, directions, offset, qtime, takes_time):
    """Forward with the query times when the model accepts them."""
    if takes_time:
        return model(obs, directions, offset, query_time=qtime)
    return model(obs, directions, offset)


@torch.no_grad()
def evaluate(model, batches, device, autocast, prepare, metrics,
             takes_time, steps, shuffle_directions=False, by_query=None):
    """``shuffle_directions`` permutes each episode's odometry tokens in
    time (directions on z2, symbols on se2). Returns the whole-set metrics
    plus the per-prefix-length breakdown. Under odometry dropout the
    permutation moves the unknown symbols too, so it tests only the known
    ones; with every symbol dropped it is a no-op (see
    ``shuffle_informative``). ``by_query(pred, target, query)`` adds a
    task's per-query breakdown (se2c: by heading change)."""
    model.eval()
    preds, targets, times, queries, found = [], [], [], [], []
    gen = torch.Generator().manual_seed(0)
    for b in batches:
        obs, directions, offset, target, qtime = prepare(b, device)
        if shuffle_directions:
            perm = torch.argsort(torch.rand(directions.shape[:2], generator=gen),
                                 dim=1).to(device)
            directions = permute_steps(directions, perm)
        with autocast:
            pred = call(model, obs, directions, offset, qtime, takes_time)
        preds.append(pred.float())
        targets.append(target)
        times.append(qtime)
        queries.append(offset)
        if isinstance(b, dict) and 'found' in (b.get('stats') or {}):
            found.append(b['stats']['found'])
    model.train()
    pred, target, qtime = torch.cat(preds), torch.cat(targets), torch.cat(times)
    out = metrics(pred, target)
    out.update(metrics_by_time(pred, target, qtime, steps))
    if by_query is not None:
        kw = {'found': torch.cat(found)} if len(found) == len(preds) else {}
        out.update(by_query(pred, target, torch.cat(queries), **kw))
    return out


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

    gpu_data = task.get('gpu_data', False)   # se2c: generated on the GPU
    by_query = getattr(task['task'], 'metrics_by_query', None)
    # the shuffled-odometry diagnostic permutes the model-facing symbols;
    # with all of them replaced by "unknown" it cannot change anything
    shuffle_informative = args.odometry_dropout < 1.0
    if not gpu_data:
        if args.bank is None:
            args.bank = TEXTURES[args.texture]['path']
        config, spec = tasks.config_and_spec(args.task, args.episode_steps,
                                             args.queries, args.seed,
                                             args.query_times, args.step_px,
                                             args.odometry_dropout, args.texture)
        # everything the bank fixes, recorded beside the run's numbers
        texture = {'texture': args.texture, 'bank': args.bank,
                   'tile': config.tile, 'cutoff': config.cutoff,
                   'start_window': config.start_window,
                   'dihedral': config.dihedral}
        try:
            train_bank, eval_bank = load_banks(args.bank, config, args.texture)
            bank_filter = check_texture(args.bank, args.texture)
        except ValueError as e:
            raise SystemExit(f'--texture {args.texture} --bank {args.bank}: {e}')
        if bank_filter is not None:     # a derived bank's exact parameters
            texture['bank_filter'] = bank_filter
    start = 0
    branch = None
    if args.branch:
        branch = torch.load(args.branch, map_location='cpu',
                            weights_only=False)
        start = branch['step']
        defaults = {'step_px': 9, 'odometry_dropout': 0.0,
                    'query_times': 'last', 'texture': 'mnist'}
        for key in ('task', 'model', 'size', 'batch', 'seed', 'step_px',
                    'odometry_dropout', 'query_times', 'texture'):
            if branch['args'].get(key, defaults.get(key)) != vars(args)[key]:
                raise SystemExit(f'--branch {key} mismatch: checkpoint '
                                 f'{branch["args"].get(key, defaults.get(key))!r}, '
                                 f'run '
                                 f'{vars(args)[key]!r}')
        if start >= args.steps:
            raise SystemExit(f'checkpoint step {start} >= --steps {args.steps}')
    if gpu_data:
        d = task['task'].data(vars(args), device, start)
        config, spec, held_out = d['config'], d['spec'], d['held_out']
        texture, source = d['info'], d['stream']
        floor = constant_floor([b['target'] for b in held_out])
    else:
        loader = train_loader(config, train_bank, args.batch, args.workers,
                              start)
        held_out = eval_set(config, eval_bank, args.batch, args.eval_batches,
                            args.seed)
        floor = constant_floor([b['target'].float() / 255 for b in held_out])

    module = models.get(args.model)
    if args.lr is None:
        args.lr = getattr(module, 'LR', {}).get(args.size, DEFAULT_LR)
    if branch is not None and branch['args']['lr'] != args.lr:
        raise SystemExit(f'--branch lr mismatch: checkpoint '
                         f'{branch["args"]["lr"]}, run {args.lr}')
    model = module.build(args.size, spec).to(device)
    params = models.count_params(model)
    raw_model = model
    takes_time = 'query_time' in inspect.signature(model.forward).parameters
    if args.query_times != 'last' and not takes_time:
        raise SystemExit(f'{args.model} does not accept query_time; it cannot '
                         f'be trained with --query-times {args.query_times}')
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
    if branch is not None:
        raw_model.load_state_dict(branch['model'])
        opt.load_state_dict(branch['opt'])
        print(f'branched from {args.branch} at step {start}', flush=True)

    print(f'{args.task} {args.model}/{args.size} -> {out}: {params:,} params, '
          f'lr {args.lr:.2g}; floor mse {floor["mse"]:.4f}, '
          f'floor fg_mae {floor["fg_mae"]:.4f}, floor c-mse '
          f'{floor["mse_centered"]:.4f}; step_px {args.step_px}, '
          f'odometry_dropout {args.odometry_dropout}; texture '
          + (f'{texture["texture"]} (se2c generator {texture["generator"]})'
             if gpu_data else
             f'{args.texture} ({args.bank}, tile {config.tile}, cutoff '
             f'{config.cutoff})'), flush=True)
    if not shuffle_informative:
        print('all odometry is dropped: the "dirs shuffled" number permutes '
              'unknown symbols only and equals eval mse by construction; it '
              'says nothing about odometry use', flush=True)
    log = open(os.path.join(out, 'log.jsonl'), 'w')
    log.write(json.dumps({'step': 0, 'task': args.task, 'model': args.model,
                          'size': args.size, 'run': run, 'params': params,
                          'lr': args.lr, 'floor': floor,
                          'schedule': args.schedule, 'branch': args.branch,
                          'start': start, 'step_px': args.step_px,
                          'odometry_dropout': args.odometry_dropout,
                          'offset_max': getattr(spec, 'offset_max', None),
                          'shuffle_informative': shuffle_informative,
                          **texture,
                          'args': vars(args)}) + '\n')
    log.flush()
    best, best_step = None, -1
    t_start = time.perf_counter()
    running, since = 0.0, 0
    it = source if gpu_data else iter(loader)
    rec = None
    for step in range(start, args.steps):
        for g in opt.param_groups:
            g['lr'] = lr_at(step, args)
        obs, directions, offset, target, qtime = prepare(next(it), device)
        with autocast:
            pred = call(model, obs, directions, offset, qtime, takes_time)
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
                              metrics, takes_time, spec.steps,
                              by_query=by_query)
                rec['eval'] = ev
                rec['eval_shuffled'] = evaluate(model, held_out, device,
                                                autocast, prepare, metrics,
                                                takes_time, spec.steps, True)
                if best is None or ev['mse'] < best['mse']:
                    best, best_step = ev, step + 1
                by_t = ('' if args.query_times == 'last' else
                        f' last-step {ev["mse_last"]:.4f} ')
                print(f'step {step + 1:6d} train {rec["train_mse"]:.4f} '
                      f'eval mse {ev["mse"]:.4f}{by_t} '
                      f'c-mse {ev["mse_centered"]:.4f} '
                      f'fg_mae {ev["fg_mae"]:.4f} '
                      f'fg<0.1 {ev["fg_within_0.1"]:.3f} '
                      f'| dirs shuffled mse '
                      f'{rec["eval_shuffled"]["mse"]:.4f}'
                      f'{"" if shuffle_informative else " (n/a: all dropped)"} '
                      f'({rec["steps_per_s"]:.1f} it/s)', flush=True)
            log.write(json.dumps(rec) + '\n')
            log.flush()
        if args.save_every and (step + 1) % args.save_every == 0 and not last:
            torch.save({'model': raw_model.state_dict(),
                        'opt': opt.state_dict(), 'step': step + 1,
                        'args': vars(args)},
                       os.path.join(out, f'step{step + 1}.pt'))
    log.close()

    final = {'task': args.task, 'model': args.model, 'size': args.size,
             'run': run, 'query_times': args.query_times,
             'step_px': args.step_px,
             'odometry_dropout': args.odometry_dropout,
             'shuffle_informative': shuffle_informative,
             **texture,
             'schedule': args.schedule, 'branch': args.branch,
             'start': start,
             'params': params, 'steps': args.steps, 'batch': args.batch,
             'episodes_seen': args.steps * args.batch,   # including the parent's
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
                autocast=autocast, title=run_title(final), prepare=prepare,
                takes_time=takes_time,
                kinds=getattr(task['task'], 'query_kinds', None))
    plot_run(out, final)
    print(f'done: eval mse {final["final_eval"]["mse"]:.4f} '
          f'(shuffled {final["final_eval_shuffled"]["mse"]:.4f}'
          f'{"" if shuffle_informative else " n/a"}, '
          f'floor {floor["mse"]:.4f}); wrote {out}')


if __name__ == '__main__':
    main()
