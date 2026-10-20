# Frozen hybrid6 transfer to 15x15 images with 9px moves

2026-10-18. User-requested zero-shot test, no training. Checkpoint:
`results/se2d/hybrid6_causal/s/local_causal_inert_saw_s_100k_ieee_seed0/final.pt`.

256 fresh held-out MNIST SAW episodes, T32, unchanged9px translations and
quarter-turn headings. Eight uniformly timed queries per episode, plus four
source-aligned queries (sources8/16/24/31) at the final time. Original9x9
observations and targets are the exact central crops of their15x15 counterparts,
asserted byte-for-byte. Maps gain random tiles only in a halo beyond the original
map boundary; query centers, trajectories, headings, and core textures stay fixed.

Fully covered9x9 query windows remain covered when observations and targets are
both expanded to15x15: expansion by3px preserves inclusion in the union of
observation footprints. In the9-input/15-output condition, only the central9x9
is scored, since the outer target may be unseen. No unknown-area targets are
silently included in that condition's metrics.

## Matched central9x9 MSE

| Observation size | Query output size | Uniform-time | Shuffled odometry | Final-time aligned |
|---|---|---:|---:|---:|
| 9 | 9 | .002020 | .151305 | .002791 |
| 15 | 15 | .146498 | .144612 | .145819 |
| 15 | 9 | .146356 | .144479 | .145953 |
| 9 | 15 | .002149 | .151173 | .002882 |

Full15x15 MSE with15x15 history is .145117 (uniform) and .138993 (aligned).
Outer-border MSE is .144340 (uniform). This is a substantial failure, not just
poor extrapolation of the newly requested border pixels. Native9x9 evaluation
reproduces good reconstruction on this independent sample.

The key control is that expanding the QUERY grid alone preserves central
retrieval, whereas expanding the OBSERVATION grid destroys it even with the
original9x9 query output. The failure is therefore associated with processing
the expanded history. It does not demonstrate a shape/API incompatibility or
inability to train at15x15. History expansion simultaneously introduces new
patch calibration offsets, more observation tokens/updates per time step
(25 rather than9), additional image content and greater overlap. This screen
cannot distinguish those causes. A next intervention could separate token
count/order from added calibration/content; no such follow-up or fine-tuning
was launched here.

## Artifacts and reproduction

- Script: `experiments/hybrid6/view15_eval.py`.
- Complete factorial controls: `results/hybrid6_view15_controls/results.json`.
  Includes checkpoint SHA256, per-batch errors, episode standard errors, and
  final-time subsets. Seed1808; halo seed1809.
- Rendering: `results/hybrid6_view15_controls/renders.png`. First episode,
  fixed query indices, equal display scale6px per source pixel. Nine-pixel
  outputs are centered in the15px frame rather than stretched.
- The initial three-condition screen is retained in `results/hybrid6_view15/`.

Run the script from the repository root using the CUDA environment and existing
dataset link. It refuses to overwrite its output directory. The checkpoint and
all model parameters remain unchanged.
