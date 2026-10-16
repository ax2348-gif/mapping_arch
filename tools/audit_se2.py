"""Audit a trained SE(2) checkpoint for flukes: batch isolation, a fresh
4x larger test set, the trivial-copy rate, and targeted corruption of the
odometry (turns, moves, query heading change).

    python tools/audit_se2.py results/se2/<model>/<size>/<run>/final.pt
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
import mapping.tasks as tasks
from mapping.checkpoint import load_checkpoint
from mapping.loader import eval_set, load_banks
from mapping.task import metrics, metrics_by_time
from mapping import se2

ckpt = sys.argv[1]
dev = 'cuda'
torch.manual_seed(0)
model, args, spec, task = load_checkpoint(ckpt, dev)
t = tasks.get(task)
prepare = t['task'].prepare
config, _ = tasks.config_and_spec(task, spec.steps, spec.queries, args['seed'],
                                  args['query_times'])
_, test_bank = load_banks(args['bank'], config)
print(f'checkpoint {ckpt}\n  task {task}, query_times {args["query_times"]}, '
      f'dihedral {config.dihedral}, heading {config.heading}')

# ---- 1. batch isolation
b = eval_set(config, test_bank, 8, 1, 123)[0]
obs, odo, pose, tgt, qt = prepare(b, dev)
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    base = model(obs, odo, pose, query_time=qt).float()
    obs2, odo2 = obs.clone(), odo.clone()
    obs2[0] = torch.rand_like(obs2[0]); odo2[0] = torch.randint_like(odo2[0], 16)
    pert = model(obs2, odo2, pose, query_time=qt).float()
iso = (base[1:] - pert[1:]).abs().max().item()
print(f'1. batch isolation: changing episode 0 moves other episodes by {iso:.2e}'
      f' (episode 0 itself by {(base[0]-pert[0]).abs().max().item():.2e})')
assert iso < 1e-3

# ---- 2. fresh, larger test set, different seed
fresh = eval_set(config, test_bank, 64, 64, 4242)    # 4096 episodes, 32768 queries


@torch.no_grad()
def run(batches, corrupt=None):
    preds, tgts, times = [], [], []
    g = torch.Generator(device=dev).manual_seed(0)
    for bb in batches:
        obs, odo, pose, tgt, qt = prepare(bb, dev)
        if corrupt:
            odo, pose = corrupt(odo, pose, g)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p = model(obs, odo, pose, query_time=qt)
        preds.append(p.float()); tgts.append(tgt); times.append(qt)
    p, y, tq = torch.cat(preds), torch.cat(tgts), torch.cat(times)
    out = metrics(p, y); out.update(metrics_by_time(p, y, tq, spec.steps))
    return out


def fmt(m):
    return (f"all {m['mse']:.4f} | t0-8 {m['mse_t0_8']:.4f} t8-16 {m['mse_t8_16']:.4f} "
            f"t16-24 {m['mse_t16_24']:.4f} t24-32 {m['mse_t24_32']:.4f} last {m['mse_last']:.4f}")


clean = run(fresh)
print(f'2. fresh test set (4096 episodes, seed 4242): {fmt(clean)}')

# ---- 3. trivial-copy rate: target equal to some faced observation or to one
#         rotated by any multiple of 90 degrees
copies = exact = zero_turn_on_lattice = total = 0
for bb in fresh[:8]:
    o = bb['observations'].numpy(); y = bb['target'].numpy()
    pose = bb['query_pose'].numpy()
    for i in range(o.shape[0]):
        rots = np.stack([np.rot90(o[i], k, axes=(-2, -1)) for k in range(4)])
        for q in range(y.shape[1]):
            total += 1
            eq = (rots == y[i, q]).all(axis=(-1, -2))
            nonblank = y[i, q].any()
            if eq.any() and nonblank:
                copies += 1
            dy, dx, turn = pose[i, q]
            if dy % 9 == 0 and dx % 9 == 0:
                zero_turn_on_lattice += 1
print(f'3. of {total} test queries: {copies} ({100*copies/total:.1f}%) have a non-blank target '
      f'identical to some observation up to rotation; {zero_turn_on_lattice} '
      f'({100*zero_turn_on_lattice/total:.1f}%) have an offset on the 9-pixel lattice')


# ---- 4. targeted corruption
def rand_turns(odo, pose, g):
    move, turn = odo % 4, odo // 4
    return 4 * torch.randint(0, 4, turn.shape, device=dev, generator=g) + move, pose


def rand_moves(odo, pose, g):
    move, turn = odo % 4, odo // 4
    return 4 * turn + torch.randint(0, 4, move.shape, device=dev, generator=g), pose


def rand_query_turn(odo, pose, g):
    pose = pose.clone()
    pose[..., 2] = torch.randint(0, 4, pose[..., 2].shape, device=dev, generator=g)
    return odo, pose


def zero_turns(odo, pose, g):
    return odo % 4, pose      # every turn set to "none", moves kept


for name, fn in (('turns randomised, moves kept', rand_turns),
                 ('turns all set to none, moves kept', zero_turns),
                 ('moves randomised, turns kept', rand_moves),
                 ('query heading change randomised', rand_query_turn)):
    print(f'4. {name:36s}: {fmt(run(fresh, fn))}')

