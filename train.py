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

The trainer (``main``) takes a ``mapping.config.Experiment`` and reads
every setting from its objects (docs/config.md). These flags are
translated into the same config tree (``main_from_flags``), and the flat
``args`` they parse into is still what runs record, for the tools.
"""
import argparse
import inspect
import json
import os
import time

import torch

import mapping.tasks as tasks
from mapping.data import TEXTURES, check_texture
from mapping.checkpoint import save_checkpoint
from mapping.loader import eval_set, load_banks, train_loader
from mapping.models import cuda_only
from mapping.plot import plot_run
from mapping.render import render_grid, run_title
from mapping.task import metrics_by_time
from mapping.task_se2c import add_args as add_se2c_args

DEFAULT_LR = 3e-4


def build_parser():
    """train.py's flags: every setting, its type, choices and default.
    ``mapping.config.Experiment.legacy_args`` fills the same namespace from
    the Hydra config tree."""
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='z2', choices=sorted(tasks.ALL),
                   help='which contract: z2 (CONTRACT.md), se2d '
                        '(CONTRACT_SE2D.md) or se2c (CONTRACT_SE2C.md); picks the model registry, the '
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
    return p


def parse(argv=None):
    return build_parser().parse_args(argv)


def flags(args):
    """The train.py flags that differ from the defaults, for ``args``."""
    out = []
    for action in build_parser()._actions:
        if not action.option_strings or action.dest == 'help':
            continue
        v = getattr(args, action.dest)
        if v == action.default:
            continue
        flag = action.option_strings[0]
        if action.nargs == 0:
            out.append(flag)
        else:
            out += [flag] + [str(x) for x in (v if isinstance(v, list) else [v])]
    return out


# train.py flag -> (generator it belongs to, or None for every task; the
# config key). The flags path composes the same config tree as run.py.
FLAG_KEYS = {
    'run': (None, 'run.name'), 'out': (None, 'run.out'),
    'steps': (None, 'run.steps'), 'seed': (None, 'run.seed'),
    'save_every': (None, 'run.save_every'), 'branch': (None, 'run.branch'),
    'device': (None, 'run.device'), 'compile': (None, 'run.compile'),
    'no_bf16': (None, 'run.bf16'),
    'batch': (None, 'optim.batch'), 'lr': (None, 'optim.lr'),
    'weight_decay': (None, 'optim.weight_decay'), 'clip': (None, 'optim.clip'),
    'warmup': (None, 'schedule.warmup'), 'cooldown': ('wsd', 'schedule.cooldown'),
    'eval_batches': (None, 'eval.batches'), 'eval_every': (None, 'eval.every'),
    'log_every': (None, 'eval.log_every'),
    'episode_steps': (None, 'task.episode_steps'), 'queries': (None, 'task.queries'),
    'query_times': (None, 'task.query_times'),
    'texture': ('bank', 'task.generator.texture'), 'bank': ('bank', 'task.generator.bank'),
    'step_px': ('bank', 'task.generator.step_px'),
    'odometry_dropout': ('bank', 'task.generator.odometry_dropout'),
    'workers': ('bank', 'task.generator.workers'),
    'se2c_texture': ('se2c', 'task.generator.texture'),
    'turn_n': ('se2c', 'task.generator.turn_n'), 'motion': ('se2c', 'task.generator.motion'),
    'directions': ('se2c', 'task.generator.directions'),
    'v_min': ('se2c', 'task.generator.v_min'), 'v_max': ('se2c', 'task.generator.v_max'),
    'omega_max': ('se2c', 'task.generator.omega_max'),
    'holdout': ('se2c', 'task.generator.holdout'),
    'holdout_mode': ('se2c', 'task.generator.holdout_mode'),
    'query_coverage': ('se2c', 'task.generator.query_coverage'),
}


def _value(v):
    """A Python value as a Hydra override value."""
    if v is None:
        return 'null'
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return '[' + ','.join(_value(x) for x in v) + ']'
    if isinstance(v, str):
        return "'" + v.replace('\\', '\\\\').replace("'", "\\'") + "'"
    return str(v)


def overrides_from_flags(args):
    """Hydra overrides equivalent to parsed train.py flags: the model
    (which selects the task), its rung on the model's ladder, and every
    flag that differs from its default. A non-default flag for a generator
    or schedule the run does not use is an error (train.py used to ignore
    it silently)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf',
                        'model', args.task, args.model + '.yaml')
    if not os.path.exists(path):
        raise SystemExit(f'no config for model {args.model!r} on task {args.task!r} ({path})')
    generator = 'se2c' if tasks.get(args.task).get('gpu_data') else 'bank'
    out = [f'model={args.task}/{args.model}', f'model.size={_value(args.size)}']
    if args.schedule != 'cosine':
        out.append(f'schedule={args.schedule}')
    defaults = {a.dest: a.default for a in build_parser()._actions}
    for dest, (needs, key) in FLAG_KEYS.items():
        v = getattr(args, dest)
        if v == defaults[dest]:
            continue
        if needs not in (None, generator, args.schedule):
            raise SystemExit(f'--{dest.replace("_", "-")} does not apply to this run '
                             f'(task {args.task}, schedule {args.schedule})')
        if dest == 'no_bf16':
            v = not v
        out.append(f'{key}={_value(v)}')
    return out


def permute_steps(x, perm):
    """``x (B, T, ...)`` with each row's steps permuted by ``perm (B, T)``
    (symbols on z2/se2d, pose vectors on se2c)."""
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
    time (directions on z2, symbols on se2d). Returns the whole-set metrics
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


def main(experiment, config=None, sources=None):
    """Train one run. ``experiment``: the ``mapping.config.Experiment``
    (``run.build`` of a composed config); every setting is read from its
    objects. ``config``: the resolved config tree, written to the run
    directory as ``config.yaml``. ``sources``: what it was composed from,
    ``{'overrides': [...], 'experiment': file or None}``, written as
    ``overrides.yaml`` (and the file copied to ``experiment.yaml``) so
    ``python run.py --from <run>/config.yaml`` can recompose it. The
    legacy ``args`` record (``experiment.legacy_args()``) is written into
    log.jsonl, final.json and the checkpoints for the tools; nothing here
    reads a setting from it.

    A run directory that is not empty is refused unless
    ``run.overwrite``; nothing is written until the data, the model and
    the device have been checked."""
    e = experiment
    r, t, g, o, ev, sched = e.run, e.task, e.task.generator, e.optim, e.eval, e.schedule
    args = e.legacy_args()                  # the record: output only
    torch.manual_seed(r.seed)
    device = torch.device(r.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    task = tasks.get(t.name)
    prepare = task['task'].prepare
    loss_fn, metrics = task['task'].loss_fn, task['task'].metrics
    constant_floor = task['task'].constant_floor
    run = r.name or f'seed{r.seed}'
    out = r.out or os.path.join(task['root'], e.model.name, e.model.size, run)
    if not r.overwrite and os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f'{out} is not empty (an earlier run). Write elsewhere: '
                         f'run.name=<new name> (or run.out=<dir> when the run sets '
                         f'run.out), or replace it with run.overwrite=true')

    gpu_data = task.get('gpu_data', False)   # se2c: generated on the GPU
    by_query = getattr(task['task'], 'metrics_by_query', None)
    # the shuffled-odometry diagnostic permutes the model-facing symbols;
    # with all of them replaced by "unknown" it cannot change anything
    shuffle_informative = gpu_data or g.odometry_dropout < 1.0
    if not gpu_data:
        bank = args.bank = g.bank_path()
        if not os.path.exists(bank):
            raise SystemExit(f'{bank}: no such bank file (paths are relative to the '
                             f'working directory, {os.getcwd()}). Run from the repository '
                             f'root (a git worktree needs data/ linked in), or fetch it '
                             f'(README.md, Setup)')
        config_, spec = g.data(t, r.seed)
        # everything the bank fixes, recorded beside the run's numbers
        texture = {'texture': g.texture, 'bank': bank,
                   'tile': config_.tile, 'cutoff': config_.cutoff,
                   'start_window': config_.start_window,
                   'dihedral': config_.dihedral}
        try:
            train_bank, eval_bank = load_banks(bank, config_, g.texture)
            bank_filter = check_texture(bank, g.texture)
        except ValueError as err:
            raise SystemExit(f'task.generator.texture={g.texture} '
                             f'task.generator.bank={bank}: {err}')
        if bank_filter is not None:     # a derived bank's exact parameters
            texture['bank_filter'] = bank_filter
    start = 0
    branch = None
    if r.branch:
        branch = torch.load(r.branch, map_location='cpu', weights_only=False)
        start = branch['step']
        defaults = {'step_px': 9, 'odometry_dropout': 0.0,
                    'query_times': 'last', 'texture': 'mnist'}
        for key in ('task', 'model', 'size', 'batch', 'seed', 'step_px',
                    'odometry_dropout', 'query_times', 'texture'):
            saved = branch['args'].get(key, defaults.get(key))
            if key == 'task':               # old checkpoints say 'se2'
                saved = tasks.canonical(saved)
            if saved != vars(args)[key]:
                raise SystemExit(f'--branch {key} mismatch: checkpoint '
                                 f'{branch["args"].get(key, defaults.get(key))!r}, '
                                 f'run '
                                 f'{vars(args)[key]!r}')
        parent = os.path.join(os.path.dirname(os.path.abspath(r.branch)), 'config.yaml')
        if config is not None and os.path.exists(parent):
            changed = branch_changes(parent, config)
            if changed:
                raise SystemExit(f'--branch: the run differs from its parent ({parent}) in '
                                 f'{", ".join(changed)}. A branch continues the same run '
                                 f'(task, model, lr, batch, seed); only run.*, eval.*, '
                                 f'schedule.* and optim.weight_decay/clip may change')
        if start >= r.steps:
            raise SystemExit(f'checkpoint step {start} >= run.steps {r.steps}')
    if gpu_data:
        d = task['task'].from_config(g.config(t, r.seed), device, start,
                                     ev.batches, o.batch)
        config_, spec, held_out = d['config'], d['spec'], d['held_out']
        texture, source = d['info'], d['stream']
        floor = constant_floor([b['target'] for b in held_out])
    else:
        loader = train_loader(config_, train_bank, o.batch, g.workers, start)
        held_out = eval_set(config_, eval_bank, o.batch, ev.batches, r.seed)
        floor = constant_floor([b['target'].float() / 255 for b in held_out])

    module = e.model.module(t.name)
    lr = o.lr
    if branch is not None and branch['args']['lr'] != lr:
        raise SystemExit(f'--branch lr mismatch: checkpoint '
                         f'{branch["args"]["lr"]}, run {lr}')
    model = e.model.build(spec).to(device)
    kernels = cuda_only(model)
    if device.type != 'cuda' and kernels:
        raise SystemExit(f'{e.model.name} runs {", ".join(kernels)} on a CUDA-only '
                         f'(Triton) kernel; it cannot train on {device}. Use a GPU '
                         f'(run.device=cuda), or a model without it for a CPU smoke test')
    params = tasks.get(t.name)['models'].count_params(model)
    raw_model = model
    takes_time = 'query_time' in inspect.signature(model.forward).parameters
    if t.query_times != 'last' and not takes_time:
        raise SystemExit(f'{e.model.name} does not accept query_time; it cannot '
                         f'be trained with task.query_times={t.query_times}')
    if r.compile:
        model = torch.compile(model)
    use_bf16 = r.bf16 and device.type == 'cuda'     # bf16 autocast: CUDA only
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
                              'weight_decay': o.weight_decay},
                             {'params': no_decay, 'weight_decay': 0.0}],
                            lr=lr, betas=(0.9, 0.95))
    if branch is not None:
        raw_model.load_state_dict(branch['model'])
        opt.load_state_dict(branch['opt'])
        print(f'branched from {r.branch} at step {start}', flush=True)

    # everything is checked: only now does the run directory get written
    os.makedirs(out, exist_ok=True)
    write_sources(out, config, sources)

    print(f'{t.name} {e.model.name}/{e.model.size} -> {out}: {params:,} params, '
          f'lr {lr:.2g}; floor mse {floor["mse"]:.4f}, '
          f'floor fg_mae {floor["fg_mae"]:.4f}, floor c-mse '
          f'{floor["mse_centered"]:.4f}; step_px {args.step_px}, '
          f'odometry_dropout {args.odometry_dropout}; texture '
          + (f'{texture["texture"]} (se2c generator {texture["generator"]})'
             if gpu_data else
             f'{g.texture} ({bank}, tile {config_.tile}, cutoff '
             f'{config_.cutoff})'), flush=True)
    if not shuffle_informative:
        print('all odometry is dropped: the "dirs shuffled" number permutes '
              'unknown symbols only and equals eval mse by construction; it '
              'says nothing about odometry use', flush=True)
    log = open(os.path.join(out, 'log.jsonl'), 'w')
    log.write(json.dumps({'step': 0, 'task': t.name, 'model': e.model.name,
                          'size': e.model.size, 'run': run, 'params': params,
                          'lr': lr, 'floor': floor,
                          'schedule': sched.name, 'branch': r.branch,
                          'start': start, 'step_px': args.step_px,
                          'odometry_dropout': args.odometry_dropout,
                          'offset_max': getattr(spec, 'offset_max', None),
                          'shuffle_informative': shuffle_informative,
                          **texture,
                          'args': vars(args)}) + '\n')
    log.flush()
    arch = e.model.record()
    best, best_step = None, -1
    t_start = time.perf_counter()
    running, since = 0.0, 0
    it = source if gpu_data else iter(loader)
    rec = None
    for step in range(start, r.steps):
        for group in opt.param_groups:
            group['lr'] = sched.lr(step, r.steps, lr)
        obs, directions, offset, target, qtime = prepare(next(it), device)
        with autocast:
            pred = call(model, obs, directions, offset, qtime, takes_time)
        loss = loss_fn(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if o.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), o.clip)
        opt.step()
        running += loss.item()
        since += 1

        last = step == r.steps - 1
        if (step + 1) % ev.log_every == 0 or last:
            elapsed = time.perf_counter() - t_start
            rec = {'step': step + 1, 'train_mse': running / since,
                   'lr': sched.lr(step, r.steps, lr), 'elapsed_s': elapsed,
                   'steps_per_s': (step + 1) / elapsed}
            running, since = 0.0, 0
            if (step + 1) % ev.every == 0 or last:
                res = evaluate(model, held_out, device, autocast, prepare,
                               metrics, takes_time, spec.steps,
                               by_query=by_query)
                rec['eval'] = res
                rec['eval_shuffled'] = evaluate(model, held_out, device,
                                                autocast, prepare, metrics,
                                                takes_time, spec.steps, True,
                                                by_query=by_query)
                if best is None or res['mse'] < best['mse']:
                    best, best_step = res, step + 1
                by_t = ('' if t.query_times == 'last' else
                        f' last-step {res["mse_last"]:.4f} ')
                print(f'step {step + 1:6d} train {rec["train_mse"]:.4f} '
                      f'eval mse {res["mse"]:.4f}{by_t} '
                      f'c-mse {res["mse_centered"]:.4f} '
                      f'fg_mae {res["fg_mae"]:.4f} '
                      f'fg<0.1 {res["fg_within_0.1"]:.3f} '
                      f'| dirs shuffled mse '
                      f'{rec["eval_shuffled"]["mse"]:.4f}'
                      f'{"" if shuffle_informative else " (n/a: all dropped)"} '
                      f'({rec["steps_per_s"]:.1f} it/s)', flush=True)
                if 'mse_far' in res:     # se2c: by query kind, true | shuffled
                    sh = rec['eval_shuffled']
                    print('             by kind: ' + '  '.join(
                        f'{k} {res["mse_" + k]:.4f}|{sh["mse_" + k]:.4f}'
                        for k in ('fallback', 'near', 'mid', 'far')), flush=True)
            log.write(json.dumps(rec) + '\n')
            log.flush()
        if r.save_every and (step + 1) % r.save_every == 0 and not last:
            torch.save({'model': raw_model.state_dict(),
                        'opt': opt.state_dict(), 'step': step + 1,
                        'args': vars(args), 'arch': arch},
                       os.path.join(out, f'step{step + 1}.pt'))
    log.close()

    final = {'task': t.name, 'model': e.model.name, 'size': e.model.size,
             'run': run, 'query_times': t.query_times,
             'step_px': args.step_px,
             'odometry_dropout': args.odometry_dropout,
             'shuffle_informative': shuffle_informative,
             **texture,
             'schedule': sched.name, 'branch': r.branch,
             'start': start,
             'params': params, 'steps': r.steps, 'batch': o.batch,
             'episodes_seen': r.steps * o.batch,   # including the parent's
             'final_eval': rec['eval'],
             'final_eval_shuffled': rec['eval_shuffled'],
             'best_eval': best, 'best_step': best_step, 'floor': floor,
             'scaling_rule': getattr(module, 'SCALING_RULE', ''),
             'size_kwargs': dict(e.model.size_kwargs),
             'elapsed_s': time.perf_counter() - t_start,
             'args': vars(args)}
    with open(os.path.join(out, 'final.json'), 'w') as f:
        json.dump(final, f, indent=1)
    save_checkpoint(os.path.join(out, 'final.pt'), raw_model, vars(args),
                    spec, t.name, arch=arch)
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


# what a branch may change from its parent: how long and where it runs,
# how it is evaluated, its schedule, and (as before, unchecked) weight decay
# and clipping. Everything else (task, generator, model, lr, batch, seed)
# must be the parent's. task.generator.workers is a loader setting.
BRANCH_MAY_CHANGE = ('run.', 'eval.', 'schedule.', 'optim.weight_decay', 'optim.clip',
                     'task.generator.workers')


def branch_changes(parent, config):
    """Dotted keys in which ``config`` differs from the parent run's saved
    config (a path), other than those a branch may change."""
    import run as entry
    from omegaconf import OmegaConf
    a = OmegaConf.to_container(OmegaConf.load(parent))
    b = OmegaConf.to_container(config) if OmegaConf.is_config(config) else dict(config)
    return [k for k in entry.differences(a, b)
            if not k.startswith(BRANCH_MAY_CHANGE) and k != 'schedule']


def write_sources(out, config, sources):
    """``config.yaml`` (the resolved config), ``overrides.yaml`` (what it
    was composed from) and ``experiment.yaml`` (a copy of the experiment
    file, if any) in the run directory."""
    import shutil
    from omegaconf import OmegaConf
    if config is not None:
        OmegaConf.save(config, os.path.join(out, 'config.yaml'))
    if sources is not None:
        exp = sources.get('experiment')
        if exp is not None and os.path.abspath(exp) != os.path.abspath(
                os.path.join(out, 'experiment.yaml')):
            shutil.copy(exp, os.path.join(out, 'experiment.yaml'))
        overrides = [o for o in sources['overrides']
                     if o.split('=', 1)[0].lstrip('+~') != 'run.overwrite']
        OmegaConf.save(OmegaConf.create(
            {'overrides': overrides,
             'experiment': 'experiment.yaml' if exp is not None else None}),
            os.path.join(out, 'overrides.yaml'))


def main_from_flags(argv=None):
    """``train.py --flags``: the same run, through the config tree."""
    import sys
    import run as entry
    argv = list(sys.argv[1:] if argv is None else argv)
    overwrite = '--overwrite' in argv      # not a setting: not in the args record
    argv = [a for a in argv if a != '--overwrite']
    parsed = parse(argv)
    overrides = overrides_from_flags(parsed) + (['run.overwrite=true'] if overwrite else [])
    cfg = entry.resolved(entry.load(overrides))
    experiment = entry.build(cfg)
    args = experiment.legacy_args()
    if parsed.lr is None:      # main() would take the model's default; the config already did
        from mapping.config import model_lr
        parsed.lr = model_lr(parsed.task, parsed.model, parsed.size)
    # the translation is exact, or no run (keys the flags cannot express
    # take their config defaults and are not compared)
    diff = {k: (v, vars(args).get(k, '<missing>')) for k, v in vars(parsed).items()
            if v != vars(args).get(k, '<missing>')}
    if diff:
        raise SystemExit(f'config translation differs from the flags '
                         f'(flag value, config value): {diff}')
    main(experiment, config=cfg, sources={'overrides': overrides, 'experiment': None})


if __name__ == '__main__':
    main_from_flags()
