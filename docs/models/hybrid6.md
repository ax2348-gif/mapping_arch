# Hybrid6 calibrated camera patches

Implementation candidate on `exp/hybrid6`, based on the fixed SAW revision
`4b5556f`. All model changes are new files; existing hybrid4 models, generators,
kernels, and training code are unchanged. The design is specified in
[the proposal](../patch_geometry_proposal.md).

Three role vectors distinguish observations, renders, and transitions. Images
use shared sensor-to-local extrinsics and patch-specific crop intrinsics.
Observation extrinsics are identity in the standard single-camera adapter.
The matched planar task has f=1, principal point minus the patch-center offset,
and 3x3 grayscale samples. No scene depth enters the model.

The pose encoder shares a projection of raw xyz translation Fourier features,
the nine rotation-matrix entries, and a known-measurement flag. Translation uses
a fixed 256-unit normalization and ten Fourier octaves; it never depends on
sequence length. Unknown discrete odometry masks all pose features, distinct
from identity. Sensor features are log fx/fy, Fourier principal-point features,
and normalized skew. The same pose and sensor encoders serve writes and reads.
Discrete symbols become single-step local transforms, never integrated poses.
This changes both positional and odometry parameterization relative to hybrid4;
it is not an isolated patch-position-table ablation.

The hybrid4 mixer stack, initialization, width ladder, temporal RoPE, and
inert/causal behavior are reused. Registrations are `se2d/hybrid6`,
`se2d/hybrid6_causal`, `se2c/hybrid6c`, and `se2c/hybrid6c_causal`. TF32 and IEEE
scan options are passed to the existing layers. The normal forward accepts
runtime rectangular input grids and an optional independent `query_shape`.
`forward_cameras` accepts explicit local SE3 extrinsics and intrinsics for
dense batches of 3x3 patches, including multiview slots. It does not support
ragged/padded slots, variable per-token resolution, or a streaming state API.
Having an SE3-shaped interface does not establish learned SE3 competence.

| Rung | Hybrid6 parameters | Hybrid4 parameters |
| --- | ---: | ---: |
| xs | 122861 | 119661 |
| s | 866905 | 860505 |
| m | 2823501 | 2813901 |
| l | 6781129 | 6768329 |
| xl | 22325337 | 22306137 |
| xxl | 53067273 | 53041673 |

Validation: 56 targeted CPU tests passed, including geometry against the
existing continuous renderer's coordinate convention, independent runtime grid
counts, backward gradients, unknown odometry, inert history/query invariance,
causality with a nonzero-output control, full-pose/calibration gradients, model
registry/config composition, and the proposed schedule's equality to the
parent's first 100k steps. CPU backward substitutes the existing reference
delta scan and SDPA using the actual attention-mask predicate; a separate
forward comparison checks that attention against CPU FlexAttention.

Local GPU preflight subsequently passed on RTX PRO 4000 Blackwell: three
synthetic optimizer steps at batch 2, full T=32/Q=8, finite gradients, exact
checkpoint reconstruction, and 364530176 bytes peak tensor allocation. A
two-step real SAW-data run through the production trainer also completed,
including evaluation and checkpoint writing (`/tmp/hybrid6-trainer-smoke`).
These are execution checks, not learning results. H200 execution, sustained
throughput, learning quality, and largest-rung memory remain untested. The job
argument parsing is also covered to catch Hydra quoting failures before queueing.

## Initial Torch experiment for review

The preset [hybrid6_initial.yaml](../../conf/experiment/hybrid6_initial.yaml)
was approved for Torch. Use one H200 GPU, with the established
Torch environment and a separate deployment directory. Verify GPU forward,
backward, finite loss, and checkpoint restore before the main training run.

- SE2d, causal + inert, size s (width 128), from scratch, seed 0.
- MNIST, SAW paths (no repeated lattice locations), 32 transitions and 33
  observations, step size 9, no odometry dropout. This is `saw`, not the stricter
  `no_revisit_view` condition. Eight queries with uniform anchor times.
- Fixed 9x9 images, 3x3 patches, f=1. No scale augmentation or longer sequences.
- Batch 64, AdamW lr 0.0003, weight decay 0.1, gradient clip 1.0; bf16 autocast,
  TF32 scan to match the parent, no torch.compile on the model.
- 100k steps: 100-step warmup, then constant LR, no cooldown. This exactly
  matches the first 100k steps of the parent's 400k WSD schedule, whose cooldown
  starts at 320k. Continuation or cooldown is a subsequent decision.
- Evaluation and checkpoints every 10k, 16 evaluation batches, logs every 2k,
  six data-loader workers. Use the same MNIST bank and evaluation seed/batches
  as the parent; compare overall, late-anchor, foreground, and shuffled-odometry
  metrics at matched steps, not differently scheduled endpoints.

Reference: SAW causal-inert hybrid4 s, run `563beb`, directory
`results/se2d/hybrid4_causal/s/inert_seed0_400k_qt_wsd_saw` in the SAW worktree.
Its reported 100k MSE is 0.0054 and late-query MSE 0.0104. These are reference
observations, not acceptance thresholds or new measurements. The first seed is
a functionality and learning check; a claim of improvement needs replication.

The main risks are optimization changes from the shared numeric pose encoding,
untrained scale/SE3 dimensions, and existing memory/token-count limitations.
The interface does not solve cross-frame integration or cross-scale information
loss. The [job](../../jobs/hybrid6/initial.sbatch) gates training on a small
actual-kernel GPU preflight and exact model-checkpoint reconstruction.
