# Model contract for the mapping study

Read this whole file before writing a model. It is the interface every
variant shares, the rules that keep the comparison honest, and what a
finished contribution consists of.

## The task

A map is a grid of MNIST tiles. An agent takes 32 steps of 9 pixels in
cardinal directions, observing a 9x9 window at each of the 33 positions.
Windows never overlap. The model is then asked to render the 9x9 window at
a query position, given only:

| input | shape | meaning |
| --- | --- | --- |
| `obs` | `(B, 33, 9, 9)` float in [0, 1] | the observations, in order |
| `directions` | `(B, 32)` long in 0..3 | step direction from t to t+1; `mapping.data.MOVES[k]` is the (dy, dx) unit move |
| `query_offset` | `(B, 8, 2)` long | query corner minus the last window's corner, pixels (y, x), within ±288 |

Output: `pred (B, 8, 9, 9)` float, the window at each query. Loss is MSE.
Every query pixel was observed at some earlier step, so the task is fully
determined by the inputs: there is nothing to guess, only to assemble.
Query offsets fall at arbitrary pixel positions, so a target straddles up
to four observed windows and is never a single stored one.

This is the **single offline setting with relative odometry**: full
attention over the whole episode is allowed, queries are answered at the
end, and the only pose information is the step directions. No absolute
positions, no online or causal variant, no other odometry condition.

The hypothesis under test is that transformers struggle to integrate the
directions into positions and aggregate the fragments into a map. The
architecture's job is to make that integration learnable **inside the
parameters**.

## Why this study exists

The prevalent assumption is that general architectures scaling with data
and compute are superior to anything with a built-in inductive bias, and
that what we have as of 2026 is already very general. This task is a probe
of that assumption: it is fully determined by its inputs, it needs one
abelian group operation (summing moves) plus a gather, and every pixel of
the answer is in the context. If the standard general-purpose transformer,
scaled along its ladder, solves it with a healthy exponent, the assumption
survives here. If it plateaus, or scales worse than a variant carrying an
inductive bias for integration, that is evidence that "general" is doing
less work than claimed.

So the **vanilla transformer is the arm under interrogation**, and must be
the strongest faithful version of the 2026 default, not a strawman. Every
other variant is a hypothesis about a specific inductive bias the general
architecture lacks, and must say which one. The comparison is made on the
scaling plot: offset, exponent, and where each curve bends, with the
recurrent baseline as the reference for "an old architecture with the
right bias".

## Build as a machine learning architect, not a roboticist

Approach this as an expert in state-of-the-art scalable architectures
would approach a sequence task they want to scale: tokenisation,
attention and recurrence variants, memory tokens, depth and width rules,
muP-style learning-rate transfer, modern normalisation and
initialisation. Do not approach it as a mapping problem: no SLAM-inspired
modules, no canvas, no "map representation" designed by hand, no module
whose purpose is stated in terms of poses or world coordinates. The task
vocabulary (map, odometry, fragments) describes the data, not the model.
If an architectural idea comes from the LLM or vision scaling literature,
it is in scope; if it comes from the robotics literature, it is almost
certainly the kind of external geometric processing the next section
excludes.

## What counts as cheating

The point is to find out whether the network can do the geometry. Anything
that does the geometry for it is excluded:

- **No use of absolute positions.** The batch contains `positions` for
  diagnostics; a model never receives it and must not look for it.
- **No external integration of the directions.** No cumulative sum of the
  moves outside learned layers, no coordinates computed from `directions`
  and then fed in, embedded, or used to index anything. A recurrent or
  linear-attention layer that *learns* to sum is the whole point and is
  fine; a hand-written prefix sum is not.
- **No positional encoding keyed on world geometry.** RoPE, relative
  attention biases, ALiBi and the like may be keyed on the **sequence
  index** (time step, patch index within an observation), never on
  anything derived from the directions or the query offset. The marlenv
  trick of RoPE over dead-reckoned world coordinates is exactly what is
  excluded.
- **No geometric scatter or gather.** No pasting observations onto a
  canvas at computed coordinates, no sampling a canvas at the query, no
  spatial transformer networks over coordinates the model did not produce
  by learning.
- **Query offset is a raw input.** Embed it however you like (learned
  table over the offset range, MLP on the scaled scalars, Fourier
  features); do not combine it with anything else by arithmetic the model
  did not learn.

Allowed and encouraged: any attention variant, recurrence, state space or
linear-attention layers, memory or register tokens, looped or weight-tied
depth, mixture of experts, gated MLPs, learned per-patch and per-step
embeddings, normalisation and initialisation of your choice. Pure PyTorch,
no custom CUDA kernels, no external architecture packages.

Every model file ends with an **audit section** in its docstring listing
every place positional or geometric information enters the network and why
each is within the rules. A reviewer who disagrees with one line should be
able to find it.

## Files and interface

- Write one module `mapping/models/<name>.py`. It must define
  `SIZES`, `SCALING_RULE`, `build(size, spec)`, and register itself with
  `register('<name>')(sys.modules[__name__])` at the bottom. See
  `mapping/models/mlp.py` for the pattern and `mapping/models/__init__.py`
  for the registry.
- `build` returns an `nn.Module` whose `forward(obs, directions,
  query_offset)` returns `pred`. `spec` is a `mapping.task.TaskSpec` with
  `view`, `steps`, `directions`, `step_px`, `queries`, `offset_max`.
- `SIZES` is an ordered ladder of at least five rungs, smallest first,
  spanning at least two orders of magnitude in parameters. The smallest
  should train in minutes on the local GPU; the largest should fit a
  batch of 64 in bf16 on 23 GiB.
- `SCALING_RULE` is one paragraph saying which dimensions grow along the
  ladder and in what ratio, so each rung is reproducible from its name.
  Width, depth, heads and any extras (slots, loops, state size) must scale
  by a stated rule, not by hand-picking per rung.
- Do not modify `mapping/data.py`, `mapping/loader.py`, `mapping/task.py`,
  `train.py`, `scaling.py`, or another model's file. If the harness needs
  a change, say so in your report and work around it.

## Scaling procedure

`scaling.py --model <name> --dry` prints the ladder with parameter counts.
`scaling.py --model <name> --local --steps N --sizes ...` runs rungs here.
`scaling.py --model <name> --slurm --steps N` writes a SLURM array for the
Torch cluster, which is where the real sweeps run. `--fit` fits a power
law in parameters to the final evaluation loss once results exist.

The sweep is parameter scaling at fixed steps and batch, so every rung
sees the same episodes. Your model file must make that sweep meaningful:
the ladder must be a clean one-parameter family, and the learning rate
must either be reasonable across the ladder or be set per rung by a stated
rule (for instance muP-style scaling with width). State which in
`SCALING_RULE`. A per-rung rule is expressed as an optional module-level
`LR = {size: lr}`; `train.py` uses it whenever `--lr` is not given, so a
sweep needs no extra flags.

## What a finished contribution is

1. `mapping/models/<name>.py`, with the audit section.
2. `docs/models/<name>.md`, at most a page: the architecture, why it might
   integrate odometry where a vanilla transformer would not, the scaling
   rule and expected failure modes, and the smoke results.
3. A smoke run on the local GPU: `python train.py --model <name> --size xs
   --steps 500` must run, and its final `eval mse` must beat the constant
   floor printed at the start. Report the numbers. If 500 steps is not
   enough to beat the floor, run longer and say how long it took. The
   trainer also reports `dirs shuffled mse`: the same evaluation with each
   episode's directions permuted in time. If it equals the unshuffled
   number, the model is not using the odometry yet; report both. The run
   directory `results/<name>/xs/seed0/` also gets `grid.png`, sampled
   predictions beside targets; look at it. See README.md for the full
   workflow and `python summarize.py` for every run.
4. `python scaling.py --model <name> --dry` output in the doc.
5. A forward-pass check that the largest rung fits a batch of 64 in bf16
   on the local GPU, with the peak memory.

Keep to the local GPU for smoke runs only; it is shared with other agents
working in parallel, so do not run anything longer than a few minutes
without need. Do not submit cluster jobs.

## Reference numbers

The harness prints the constant floor per run; on the evaluation set it
is mse 0.097 and fg_mae 0.63, and a model that draws nothing sits there.
Stroke pixels are 17% of a target, so watch `fg_mae` and `fg_within_0.1`,
not only `mse`. The reference `mlp` at size xs (402k params) reaches mse
0.0958 after 300 steps at 145 it/s: barely off the floor, which is the
point of it. Any variant with a real inductive bias should leave it far
behind.
