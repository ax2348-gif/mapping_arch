# Hybrid6 history patch permutation probe

2026-10-18. Frozen100k hybrid6 causal inert S checkpoint.256 fresh held-out SAW MNIST episodes, T32,9px moves,9x9 images, seed1810; permutation seed1811. Eight uniformly timed queries plus four final-time queries for source observations8/16/24/31. No training or parameter changes.

Each permutation moves the nine observation patches together with their per-patch extrinsics and intrinsics. All timestamps, frame boundaries, odometry tokens, query poses, query calibration, and query-token ordering remain fixed. Physical observations and token count are unchanged. Inverse-gather checks exactly restore pixels and calibration. The identity `forward_cameras` path matches ordinary model forward bitwise.

## Results

| History patch order | Uniform-time MSE | Final-time aligned MSE |
|---|---:|---:|
| identity | 0.001867 | 0.002691 |
| adjacent_swap | 0.127169 | 0.149543 |
| column_major | 0.141562 | 0.150322 |
| reverse | 0.135475 | 0.146906 |
| fixed_random | 0.128553 | 0.141871 |
| independent_frame_random | 0.131581 | 0.141187 |
| reverse_last_frame | 0.006227 | 0.159044 |
| shuffled_odometry | 0.143431 | 0.172677 |

`adjacent_swap` exchanges slots0 and1 in every frame. `column_major` transposes the3x3 patch-grid traversal. `reverse` reverses all nine slots in every frame. `fixed_random` uses one permutation shared by every frame/episode; `independent_frame_random` draws a permutation per frame and episode. `reverse_last_frame` changes only observation32. Shuffled odometry rolls motion inputs across episodes with ordinary patch order.

Changing ONLY the final observation’s order leaves every query asked before time32 bitwise unchanged (maximum prediction-change MSE0). Yet it destroys final-time retrieval of source8/16/24/31, all observed earlier. Its small uniform-time average is diluted by unaffected earlier queries; the final-time result is the relevant measure.

## Interpretation

The trained checkpoint has severe dependence on arbitrary history serialization. Count, geometry, content and available information are held fixed here, so those cannot explain the permutation result. This strengthens the case for separating observation-set processing from a fixed per-timestep recurrent computation budget.

This is not a localization to GDN alone. Permutation changes the GDN short-convolution neighborhoods and scan order, and the within-frame causal attention visibility/order. It also changes downstream history activations. No frozen-cache intervention was used. A future mixer-specific intervention would be needed to attribute the effect to one component. Similarly, inference failure under an unseen permutation does not prove that training with randomized order could not teach robustness.

Reversing the last frame perturbs retrieval of older images without changing any earlier observation or odometry. This is consistent with a fragile state/address computation, but can also involve the last frame’s attention features; it does not by itself prove a particular pose-state circuit.

The15x15 transfer failure therefore has a concrete related weakness: even a count-preserving reordering of native9-patch frames breaks the model. These tests do not establish that ordering is the sole cause of resolution failure, nor validate the proposed register architecture.

## Artifacts

- Script: `experiments/hybrid6/permutation_eval.py`.
- Raw paired per-query metrics and checkpoint SHA256: `results/hybrid6_permutation/results.json`.
- Render sheet: `results/hybrid6_permutation/renders.png`, first two episodes and fixed source queries, with no quality-based selection.

Run the script from the repository root with the existing dataset and CUDA environment. It refuses an existing output directory.
