# mapping

Can a sequence model assemble a global representation from local
observations and relative odometry? A Z² probe: a map of MNIST tiles, a
random walk observed through 9×9 windows, 32 steps of 9 pixels, and a
query that asks for the window at an arbitrary offset from the last one.
Every pixel of the answer was observed at some earlier step; the model's
whole job is to work out where. `CONTRACT.md` has the task, the rules
that keep architectures honest, and the deliverables for a new model.

## Setup

```bash
pip install torch numpy pillow datasets     # torch with CUDA for training
pip install --no-deps fla-core==0.5.2 einops # kernels (hybrid4 and later);
                                             # --no-deps keeps torch untouched
python scripts/get_mnist.py                 # writes data/mnist.npz once
python profile_loader.py                    # optional: data path sanity
```

Nothing else is stored. Training batches are generated on the fly from a
seed; the only data file is the MNIST bank.

## Layout

```
mapping/data.py        episode generation (walk, tiles, queries)
mapping/loader.py      torch stream for training; fixed held-out set
mapping/task.py        model inputs and outputs, loss, metrics, floor
mapping/models/        one file per architecture, auto-registered
mapping/checkpoint.py  what a run saves; load a model back from it
mapping/render.py      prediction grids
mapping/results.py     reading runs back, power-law fit
train.py               train one model at one size -> a run directory
scaling.py             ladder, sweep (local or SLURM), fit
summarize.py           every run as a table; fits per model
render.py              redraw a grid from any checkpoint
docs/models/           one page per architecture
results/               run directories (not in git)
shelved/               variants set aside, with a README on resuming
```

A run directory is `results/<model>/<size>/<run>/` and always holds
`final.json` (every number a fit needs), `final.pt` (rebuilds the model
by itself), `log.jsonl`, and `grid.png` (sampled predictions beside their
targets). Runs are named `seed<seed>` by default, `seed<seed>_lr<lr>`
from a learning-rate grid; name one anything with `--run`.

## Adding an architecture

1. **Read `CONTRACT.md`**, in full. It says what the model receives, what
   counts as cheating, and what a finished contribution is.
2. **Copy the pattern.** `mapping/models/mlp.py` is the smallest complete
   example; `vanilla.py` is a full one. A model file defines `SIZES` (an
   ordered ladder of at least five rungs over two orders of magnitude),
   `SCALING_RULE` (one paragraph: which dimensions grow and how), an
   optional `LR = {size: lr}`, and `build(size, spec) -> nn.Module` whose
   `forward(obs, directions, query_offset)` returns the query views. End
   the docstring with the audit: every place positional information
   enters, and why it is within the rules. Register at the bottom:
   `register('<name>')(sys.modules[__name__])`. Dropping the file into
   `mapping/models/` is the whole registration.
3. **Check the ladder.** `python scaling.py --model <name> --dry` prints
   parameters and learning rate per rung. Confirm the largest rung fits a
   batch of 64 in bf16 on the local GPU.
4. **Smoke test.** `python train.py --model <name> --size xs --steps 500`.
   Read three things from the output: eval mse against the constant floor
   printed at the start; `dirs shuffled mse`, which must eventually rise
   above eval mse or the model is not using the odometry; and
   `results/<name>/xs/seed0/grid.png`. Around this task, 500 steps shows
   only that it runs; 8000 steps at xs is where models have been seen to
   leave the plateau.
5. **Write `docs/models/<name>.md`**: the architecture, which inductive
   bias it adds or removes relative to the vanilla transformer, the
   scaling rule, expected failure modes, the `--dry` output, the smoke
   numbers.

## Training and evaluating

```bash
python train.py --model vanilla --size m --steps 20000          # one run
python train.py --model vanilla --size m --steps 20000 --lr 1e-3 --run seed0_lr1e-3
python render.py results/vanilla/m/seed0/final.pt --rows 24     # redraw
# generator settings (same tasks, CONTRACT.md "Generator settings"):
python train.py --task se2 --model hybrid4 --size xs --query-times uniform \
    --step-px 3 --odometry-dropout 1.0 --run seed0_vo  # visual odometry
python tools/registrability.py --step-px 1 2 3     # ceiling for that run
python summarize.py                                             # all runs
python summarize.py --fit                                       # best per rung + fits
```

`train.py` evaluates on a fixed set from the MNIST test split, so no
evaluation digit instance was seen in training. It reports the error with
true directions and with each episode's directions permuted in time; the
gap between them is the odometry the model is using. `summarize.py` reads
every `final.json` under `results/`.

## Scaling sweeps

```bash
python scaling.py --model vanilla --dry                          # ladder
python scaling.py --model vanilla --local --steps 2000 --sizes xs s
python scaling.py --model vanilla --slurm --steps 20000 --lr-grid 1e-3 3e-4 1e-4
scripts/sync.sh push
ssh torch "cd /scratch/ax2348/mapping && sbatch jobs/vanilla.sbatch"
scripts/sync.sh pull && python scaling.py --model vanilla --fit
```

The sweep is parameter scaling at fixed steps and batch, so every rung
sees the same episodes, with the best learning rate taken per rung. The
local GPU is for smoke tests; sweeps run on the NYU Torch cluster as a
SLURM array (`scripts/remote_setup.sh` builds the environment there once;
run it as a CPU job, the login node kills pip). `--max-concurrent` sets
how many array jobs run at once.

## What results exist

`python summarize.py` is the authoritative list. `docs/models/*.md` hold
each architecture's smoke numbers at the time it was written.
