# hybrid3.5: hybrid3 with a hand-written Gated DeltaNet

One change from `hybrid3`: the linear-attention mixer is a bidirectional
Gated DeltaNet with negative eigenvalues, its scan a plain sequential
PyTorch loop (`delta_scan`). Registered as `hybrid3_5`.

Why: hybrid3's scan can only shrink its state (positive scalar decay plus
additive writes). SE(2) needs the stored relative positions rotated at
every turn. The delta-rule transition `a (I - beta k k^T)` with
`beta = 2 sigmoid(.)` in (0, 2) reaches a reflection, and two reflections
compose into a rotation (Grazzi et al. 2024). A depthwise short
convolution (width 4, causal in each head's scan direction) on q, k, v lets
the tokens after an odometry token supply the second reflection. q and k
are L2-normalised, as the rule requires.

Checks: the scan equals the matrix definition to 7e-7; two beta = 2 steps
give an orthogonal map with determinant 1; left-to-right heads ignore
later tokens and right-to-left heads earlier ones; forward with no query
times equals the all-last-step forward exactly; smoke runs pass on both
tasks. Parameters at xs: 118,637 (hybrid3 117,739).

Speed: the loop runs 401 sequential steps per layer, about 175 ms per
forward-backward at xs and batch 64 on the local GPU, so it is a reference,
not a sweep model. `hybrid4` is the same layer on the fused
flash-linear-attention kernel, checked against this one.
