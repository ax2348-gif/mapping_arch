# Direct odometry-write experiment (prepared, not launched)

Hypothesis: providing a direct motion update avoids the learned routing
bottleneck between odometry and register writes. Preserve observation-patch
permutation symmetry by keeping observations and queries passive.

Variant `se2d/hybrid7_odom`: S A,G,A,G,A, width128, five learned registers,
six32-dimensional GDN heads,1,155,105 parameters. Writers and short-convolution
inputs are now registers AND outgoing odometry, in temporal order. There is
no outgoing odometry after the final observation. Attention masks, RoPE and
query placement are unchanged: query_t cannot see outgoing odometry_t→t+1.
At T32 the convolution receives165 registers+32 odometry tokens, versus165
registers in the current run. Kernel size remains4. Thus this experiment
changes both direct writes and convolution context, not writes alone.

## Fresh run

From the repository root, using the environment's Python:

```bash
python run.py +experiment=hybrid7_odometry \
  'run.note=Fresh Hybrid7 S with direct odometry writes and register convolution'
```

100k total steps, LR3e-4, WSD decay80–100k, batch64, fresh MNIST SAW T32,
8 uniform-time queries, IEEE scans, eval1k/save10k. Separate result directory
under `results/se2d/hybrid7_odom/s/`; no current outputs are overwritten.

## Continue current weights AND optimizer

```bash
python run.py +experiment=hybrid7_odometry \
  run.branch=results/se2d/hybrid7/s/hybrid7_causal_inert_saw_s_agaga_100k_ieee_seed0/step30000.pt \
  run.branch_transition=hybrid7_odometry_writes \
  run.name=hybrid7_odom_from_agaga30k_to100k_ieee_seed0 \
  'run.note=Continue AGAGA 30k with direct odometry writes retaining optimizer and schedule'
```

Choose another completed step checkpoint by changing the path and run name.
The narrowly validated transition accepts only SE2D Hybrid7 AGAGA register-only
→ Hybrid7 odometry-writer with otherwise identical architecture. Full model
weights load strictly; parameter names/shapes/order are identical, so Adam
moments are retained too. Training resumes at the saved step and uses the
existing deterministic data-loader continuation. This is an explicit change
in computation, not an exact uninterrupted continuation of the old experiment.
The checkpoint is read only; the intervention writes a new run with parent
provenance, explicit transition config and target architecture record.

`run.steps=100000` means the final absolute step, not100k additional steps:
from30k this trains70k more, with decay still80–100k. Keep LR/batch/seed/task
settings fixed. Ordinary branch and warm-start safeguards remain in force;
no generic architecture-mismatch bypass is added. The source must contain
optimizer state and step. A final weights-only checkpoint is insufficient.

CPU preflight (does not launch training):

```bash
python experiments/hybrid7/check_odometry_resume.py \
  results/se2d/hybrid7/s/hybrid7_causal_inert_saw_s_agaga_100k_ieee_seed0/step30000.pt
```

Verified on the30k checkpoint: strict full weight loading, all96 optimizer
parameter states restored with matching moment shapes, and only model name
and odometry-write flag differ in protected config fields. Output is
`results/hybrid7_odometry_preflight/results.json`.

Compare against current run at equal total steps and elapsed compute; report
intact/shuffled and last-step errors. A resumed branch tests whether the new
path helps an existing representation adapt; fresh training tests whether it
helps discovery. Do not conflate these. Observe immediate validation change
at the intervention before attributing later differences to learning.
