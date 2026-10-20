# Hybrid7: five-register recurrent state with block-causal attention

Implemented2026-10-18 following the user's design. This is an architecture and
kernel validation, not evidence that the model learns the mapping task.

## Computation graph

Each timestep contains five learned register identities, an observation block,
and (except the last timestep) one outgoing odometry token:

`registers_t → observations_t → odometry_(t→t+1) → registers_(t+1)`

Attention is unrestricted within each block and causal between blocks. Registers
read previous observations and incoming odometry but not current observations.
Observation patches see all current patches symmetrically and preceding blocks.
Odometry sees everything earlier, including the current observations. Queries
at t see history through observations_t and all patches of their own query, never
outgoing odometry, other queries, or future frames. History never reads queries.
RoPE uses timestep t for every phase/token at t, with no within-frame advance.

Mixer blocks alternate **attention then GDN**, each followed by the existing
SwiGLU MLP. Regular-attention heads remain width/32; GDN now defaults to six heads,
each with a32×32 state and a192-dimensional internal projection independent
of residual width. S has four mixer blocks A,G,A,G; XS has two A,G. This
puts attention before the first GDN so register tokens can gather evidence
before their recurrent updates.

Only the five registers write or decay the left-to-right GDN state. Its causal
depthwise kernel-size4 short convolution gathers ONLY registers and carries
across timesteps without reset. The five identities have a deliberate fixed
scan order; observation patch serialization does not define a recurrent update
order. Observation, odometry, and query tokens read state passively with beta=0
and log-decay=0 and bypass the short convolution. Their per-token MLPs remain.

Implementation detail: the existing gated-delta scan kernel still evaluates
passive read positions, but only registers have nonzero writes/decay. Thus the
logical recurrent update count is5 per timestep independent of patch count;
this does not claim the runtime cost of those passive reads is eliminated.
IEEE scans are selected by the initial preset to minimize numerical dependence
on kernel chunk boundaries. Permutation equivariance is a mathematical property
of the graph, checked with floating-point tolerances on the GPU.

The first timestep's registers begin from learned vectors and have no earlier
observation. Its current image can be read directly by queries. Observation_t
enters recurrent memory via registers_(t+1), as intended in the phase graph.

## Interfaces and configuration

- `mapping/models/se2d/hybrid7.py`: shared camera implementation and SE2D adapter.
- `mapping/models/se2c/hybrid7c.py`: continuous adapter, identical learned shapes.
- Both inherit hybrid6's local SE3 pose and per-patch calibration features,
  fixed3x3 grayscale patch codec, and runtime-sized observation/query grids.
- Five registers are fixed in this initial design. Mixed query mode is rejected.
- `conf/experiment/hybrid7_initial.yaml` prepares the previous fast validation
  recipe (XS,T32,9px SAW MNIST,100k,WSD20% decay,IEEE), with held-out
  evaluation every1k steps. The six-head default is recorded explicitly in
  both model configs and checkpoint architecture records.
- Use the preset, or explicitly pass `model.query_mode=inert` and
  `model.scan_precision=ieee` with `model=se2d/hybrid7` or `model=se2c/hybrid7c`.
  The global harness defaults otherwise select mixed mode and are rejected.

Full state dictionaries transfer strictly between the discrete and continuous
Hybrid7 adapters. This does not mean Hybrid6 weights load directly into Hybrid7:
the block ordering and register parameters have changed. No partial transfer
or warm start was performed.

## Verification

`tests/test_hybrid7.py` checks nonzero-readout forward/backward, runtime image
sizes, exact mask semantics, timestep-only RoPE, observation permutation
invariance, query-patch equivariance, future/outgoing-odometry exclusion, query
independence, zero-query operation, register-only convolution/write/decay, shared
SE2D/SE2C weights and equivalent physical inputs, config registration and
checkpoint reconstruction. Existing hybrid6 tests are also run for regression.

`experiments/hybrid7/gpu_smoke.py` uses actual FlexAttention and IEEE scan kernels
for a few synthetic forward/backward optimizer steps, checks finite gradients
reaching all shared encoders and registers, reconstructs the model from the
checkpoint architecture record, and checks history permutation in FP32 and BF16
and15x15 runtime inputs. Results and source hashes are saved under
`results/hybrid7_smoke/results.json`. These synthetic steps do not train or save
a task checkpoint and are not a task-learning validation.

Reproduce:

```bash
python -m pytest tests/test_hybrid7.py tests/test_hybrid6.py -q
python experiments/hybrid7/gpu_smoke.py
```

The next scientific test is fresh-data SE2D learning, followed by the exact
permutation and9→15 resolution probes that exposed hybrid6's sensitivity.
Permutation invariance does not imply invariance to extra observations: adding
patches changes available evidence and softmax weighting. It also does not
establish that five register updates provide enough computational capacity.

### Original four-head S validation result

2026-10-18: **51 CPU tests passed** (13 hybrid7 cases plus38 hybrid6 regression
cases). Actual GPU smoke passed on RTX PRO4000 Blackwell. Hybrid7 S has867,545
parameters (640 more than hybrid6 S: five128-dimensional register vectors).
Permutation prediction MSE after the synthetic updates was2.16e-16 in FP32 and
1.62e-8 in BF16; maximum absolute differences4.47e-8 and9.77e-4 respectively.
Checkpoint reconstruction was bit-exact. Each GDN convolution saw165 register
positions plus3 padding positions at T32, independent of observation patches.
The smoke also completed15x15 forward inference. No task-training run launched.

### Six-head XS learning validation

Requested2026-10-18: six GDN heads are now the default for both task adapters.
XS keeps residual width64, two attention heads, and two mixer blocks A,G,
with five registers and the existing SwiGLU MLPs. GDN internal width is192;
its six32×32 state matrices are not obtained by splitting residual width64.
The previous four-head S smoke above is historical, not the current default.

Launch preset: `+experiment=hybrid7_initial`. Fresh MNIST SAW SE2D, step9,
T32, eight uniform-time queries, batch64, LR3e-4, IEEE scan, seed0,
100k steps with100-step warmup and final20k decay. Eval every1k (16 batches),
checkpoints every10k. Earlier probes/decoder remain paused.
CPU regression:51 passed. GPU smoke writes `results/hybrid7_xs_6head_smoke`.
Training output: `results/se2d/hybrid7/xs/hybrid7_causal_inert_saw_xs_100k_ieee_seed0`.

GPU XS six-head smoke passed:166,709 parameters, exact checkpoint roundtrip,
permutation MSE2.15e-16 FP32 /2.50e-8 BF16,15x15 inference successful.
Training launched in tmux `hybrid7-xs`, run ID `20261019T001247Z-erid-2-50c1e9`.
Launcher log: `results/hybrid7_launch/launcher.log`. An initial note-string
parsing failure happened before model creation; its log was retained separately.


### Revised default: final attention, S learning run

Following review, the default swaps the final two mixer blocks: S now uses
**A,G,G,A**, retaining width128, four regular-attention heads, six GDN heads
per recurrent layer, five registers, and SwiGLU after every mixer. Larger
sizes likewise swap the last pair of the alternating sequence. The architecture
record explicitly saves `final_attention: true`. This default requires S or
larger to retain both initial and final attention; XS is rejected. The prior
A,G XS experiment remains archived in its original code snapshot.

XS was stopped at user request after its27k evaluation (MSE0.071576,
shuffled0.074887);10k/20k checkpoints survive. It is not evidence against the
register-only design because it lacked attention after recurrent state reads.
The preset now selects S: fresh training,100k steps, same LR3e-4, WSD final20k,
batch64, T32, eight queries, IEEE, seed0. No XS weights are transferred.
S launcher: `results/hybrid7_s_launch/launcher.log`, tmux `hybrid7-s`.
GPU validation: `results/hybrid7_s_final_attention_smoke/results.json`.
Earlier sections describe historical architecture/validation stages.

Revised S GPU smoke passed:952,545 parameters, exact checkpoint roundtrip,
permutation MSE1.77e-16 FP32 /2.13e-8 BF16. Fresh S run launched as
`20261019T002500Z-erid-2-7ba5cc` with code snapshot `6a9c3c15fdf9`.

### Added inter-GDN attention experiment

User requested stopping A,G,G,A and trying **A,G,A,G,A**. The previous run
exited intentionally after the15k evaluation (intact MSE0.067406, shuffled
0.069231); its10k checkpoint remains. A small odometry gap had begun to emerge,
so this is an architectural comparison, not a demonstrated permanent failure.

The default registered config now sets `inter_gdn_attention: true`, adding a
full existing Attention+SwiGLU block between the two GDNs at S. Attention uses
the same phase-block-causal mask, timestep RoPE and inert query isolation.
Width128, four attention heads, six GDN heads, five registers, and training
hyperparameters are unchanged. Existing block initialization scale is retained
to avoid an additional global initialization-scale change; the added block
increases total parameters/compute. Both task adapters record this option.
Omitted option defaults to false in the constructor so the prior four-block
checkpoint architecture record still reconstructs correctly.

Fresh run name: `hybrid7_causal_inert_saw_s_agaga_100k_ieee_seed0`.
Tmux `hybrid7-s-agaga`; launcher `results/hybrid7_s_agaga_launch/launcher.log`.
Budget100k, LR3e-4, final20k WSD decay, eval1k/checkpoint10k, no weight transfer.
53 CPU tests passed, including the old four-block reconstruction check.
GPU smoke outputs: `results/hybrid7_s_agaga_smoke/results.json`.

AGAGA GPU smoke passed (1,155,105 parameters, exact checkpoint reconstruction,
FP32 permutation MSE1.97e-16 / BF16 2.21e-8). Training launched as
`20261019T003709Z-erid-2-fcdb75`, snapshot `f3323b6a7987`.

### 30k routing diagnostic while training continues

`diagnose_routing.py` ran on CPU with FP32 reference delta scan and dense
attention,16 fresh SAW episodes, seed1821. It leaves the live GPU training
untouched. Results: `results/hybrid7_routing/step30000/results.json`.
This is a small descriptive/intervention screen, not validated GPU activation
replay or a population-level causal circuit identification.

Baseline MSE0.060031. Zeroing the entire GDN mixer output (preserving residual
and MLP) gives0.062864 for block1 and0.062585 for block3: both contribute,
but neither individually has a large effect at this checkpoint.

For registers at t>0, mean attention mass on incoming odometry_(t-1→t):
first attention heads0.60%,0.37%,0.44%,0.20%; middle attention heads12.73%,
0.014%,1.07%,4.84%. These average over timesteps, register slots and episodes;
they do not report peak or per-slot specialization, and attention mass alone
is not causal attribution. Erasing precisely these incoming-odometry edges
for register queries and renormalizing attention gives MSE0.060036 in the
first attention (prediction-change MSE3.44e-8), versus0.061153 in the middle
attention (prediction-change MSE8.20e-4). Other motion paths remain available.

Interpretation: learning to route the local motion into recurrent writes is
a plausible bottleneck. First-layer register queries start from repeated
learned slot vectors and must select the appropriate motion from a growing
history; odometry has no direct recurrent write. The added middle attention
provides an empirically used route. This does not show the first GDN lacks
all motion information or establish that routing is the sole training limit.

Beta is not uniformly small: first GDN head0 mean1.980 with96.97% >1.9;
second GDN head4 mean1.953 and head5 mean1.931. Reflection-capable gates are
available, but high beta does not prove useful rotations: k directions,
value writes and decay still matter. Five register updates are opportunities
for computation, not guaranteed distinct learned operations.

Next discriminating checks, without changing the active run: repeat at later
checkpoints; separate register slots/time bins and inspect motion sensitivity
of update keys/values; compare restricted local-motion routing or direct
odometry writes in a future controlled training run. Do not conflate adding
heads (parallel state capacity) with reliable delivery of motion or more
sequential communication stages.

Prepared next experiment: [direct odometry writes](odometry_experiment.md),
with fresh-training and explicit weights+optimizer continuation commands.
Current training is unchanged; this variant has not been launched.

Completed100k evaluation: [15x15 transfer results](view15_results.md).
History expansion partially survives: center MSE0.01505→0.04185, shuffled
0.10671, unlike Hybrid6's near-total collapse. Full15 output remains poor
(MSE0.08542). No retraining or odometry-write experiment was launched.

User-authorized [full-weight SE2C transfer](se2c_transfer.md): same register-only
AGAGA S, fresh optimizer, LR1e-4,100k continuous-task steps. Source final100k
checkpoint preserved. See transfer note for zero-shot metrics and run settings.
