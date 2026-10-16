# Blind probe: can the operator learn continuous SE(2) composition?

Lift notes, prediction **L3**: "twists in, relative pose out, no pixels.
The homomorphism (held-out-angle eigenangle calibration, slope 1) is
learnable cheaply... Falsified by: staircase calibration (lookup, not
chart) or failure on held-out angles." Also measured: breakaway step against
angle granularity (L1's cliff detector), and the learned beta by turn angle.

This section was written before the main runs. Results are at the end.

## Files

| file | what |
| --- | --- |
| `walk.py` | isotropic SE(2) walks, quantisation, held-out band, targets; `python walk.py` runs the assertions |
| `encoding.py` | adapter onto the shared `mapping.se2c.encoding.encode_odometry` (convention conversion); `python encoding.py` checks it |
| `models.py` | the four models; `python models.py` checks our GDN against hybrid4's layer |
| `train.py` | one training run |
| `probe.py` | transition probe, eigenangle calibration, beta reading; `--synthetic` runs the known-rotation check |
| `figures.py` | figures from the runs |
| `run_all.sh` | the main runs, one at a time |

## Task

- **Walk.** T = 32 steps. Each step applies the twist (v, 0, w) for unit time
  through the SE(2) exponential, with v ~ U[0.5, 1] and w ~ U[-pi, pi).
  The odometry token for step i is pose i in the body frame of pose i-1:
  (dx, dy, dtheta) = (v sin w / w, v (1 - cos w) / w, w).
- **Granularity.** w is continuous (n = 0 below), or on the grid 2 pi k / n
  for n in {4, 8, 16, 32}.
- **Held-out band.** w in [50, 70] degrees is never drawn in training, for
  every granularity. Grid points inside it: none for n = 4 or 8, 67.5 for
  n = 16, and 56.25 and 67.5 for n = 32.
- **Encoding.** The shared spec, `mapping/se2c/encoding.py` (commit d8b0da7):
  translation as scaled values plus 10 octave Fourier features, and the
  angle as cos and sin of m theta for m = 1..4. I wrote a local version
  first, but the shared module landed before any main run, so every run
  uses it. `encoding.py` converts conventions (the walk's x forward, y left,
  counter-clockwise, to the repo's (dy, dx), y down, clockwise) at 9 px per
  walk unit, the mapping task's step length.
- **Queries and targets.** 16 queries (s, t) per episode, with s < t. The
  lag t - s is uniform in 1..32, then t is uniform in lag..32. The target
  is pose s in the frame of pose t: (dx, dy) / (0.63 sqrt(lag)), plus the
  cos and sin of the heading change. The position scale is chosen so each
  position component has unit variance at every lag. Loss: mse on the 4
  numbers.

## Models (width 64, ~0.12M parameters, 2 mixer blocks + SwiGLU)

Sequence: `anchor_0, odo_1, anchor_1, [queries at 1], ..., odo_32,
anchor_32, [queries at 32]`, with RoPE position = step.

| name | mixers | what it tests |
| --- | --- | --- |
| `gdn` | bidirectional Gated DeltaNet (beta in (0, 2), short conv 4) then attention | the hybrid4 mechanism |
| `gdn_b1` | same, beta = sigmoid in (0, 1) | no reflections (P2's control) |
| `attn` | attention, attention | vanilla4-style, no recurrence |
| `cdiag` | hybrid4_cdiag's complex-diagonal recurrence, then attention | commuting rotations, angle = pi tanh(.) |

The attention, MLP, RoPE, muP and init code is imported unchanged from
`mapping/models/se2/hybrid4.py`. The GDN is a copy of hybrid4's layer with
the beta range as a parameter, and is asserted equal to hybrid4's layer
when they share weights.

**The anchor token.** It is my design choice, standing in for the
observation patches. The odometry token can supply one Householder update
per step, and a rotation needs two (det = +1). In hybrid4 the second came
from the observation patches after the odometry token, via the short conv.
The anchor is a learned constant token per step that plays that role. It
gives the same two-update budget per step that the lift notes cite.

**Readout.** Each query is one token, placed after anchor_t (as the mapping
task places queries at their step), carrying the lag as Fourier features.
Its final residual goes through LayerNorm and one linear map to the 4
targets: the mapping task's readout with 4 outputs instead of 9 pixels.
There is no separate pose head. The question asks for a readout that reads
the model's own state. This is that, in the same sense as the mapping
task. It is not a readout of the delta-rule state itself.

Recipe: hybrid4's. AdamW (0.9, 0.95), weight decay 0.1, base lr 3e-4 with
muP multipliers, 100 warmup steps, cosine schedule, clip 1.0, bf16,
batch 256 fresh episodes per step, **100k steps**, seed 0.

## Choices I made that were not in the brief (flagged)

- **v range** [0.5, 1], T = 32, Q = 16, batch 256, 100k steps.
- **Band.** [50, 70] degrees, one-sided: [-70, -50] is trained.
  A model could in principle reach +60 as the inverse of -60. A two-sided
  band would be stricter. This is open.
- **The anchor token and query-token readout**, as above.
- **Queries are scanned too.** Query tokens sit in the scan and write into
  the recurrent state, as in hybrid4.
- **The 100k-step budget came from a pilot.** One 20k-step pilot of `gdn`
  (n = 0) informed it; it is not in the results. At 20k, heading was
  learned (median error 4.7 degrees at lags 17-32), long-lag position was
  not (RMS 1.2 units against a ~3 unit typical displacement), and the
  delta-rule heads had beta ~0.2-0.3 with no rotating eigenpair.
- **Breakaway keys on position because of that pilot.** The heading part
  converges early, so breakaway is defined on the position part, which is
  where the non-commuting composition lives.

## A priori: the chart cannot be exactly continuous on the whole circle

I derived this while designing, before any main result. Write the step's
transition as a product of reflections, I - 2 k k^T, built from continuous
unit vectors k(w). The k are normalised outputs of continuous maps of
circle-continuous features. Such a product is a continuous loop in
Pin(32), and it covers the loop of rotations R(w) in SO(32), w from -pi to
pi. A rotation of one plane by a full turn is the non-trivial element of
pi_1(SO(n)), n >= 3. It does not lift to a closed loop in Spin.

So an exact, everywhere-continuous chart w -> two reflections is
impossible. The concrete version: the second key must sit at angle w/2,
and k(w/2) ends at -k(0) after a full turn. Somewhere on the circle there
must be a **seam**: a short interval where the transition stops being a
clean rotation. There the pre-normalised key passes near zero, beta
leaves 2, or the key leaves the plane.

The complex-diagonal control has the analogous obstruction, in a
different form. Its angle pi tanh(f(w)) is a real-valued continuous
function on the circle, so it has winding number 0 and cannot equal w
all the way round.

Consequences for the calibration figure:
- slope 1 everywhere except a narrow seam is the best possible chart
- the seam's location is free
- a model could put it inside the held-out band, where it costs nothing
  in training

If a seam lands in the band, that is a failure on held-out angles with a
known cause, not a lookup.

## Measurements and how they are read (fixed before the main runs)

1. **Pose error by composition length and n.**
   - Eval sets: the training distribution; continuous turns on the full
     circle; walks whose every turn is in the held-out band; and walks
     whose every turn is in a trained control band of the same width
     ([20, 40] degrees).
   - Errors are reported per lag bin {1, 2-4, 5-8, 9-16, 17-32}, as RMS
     position error (walk units) and median heading error (degrees).
   - The constant-turn sets are off-distribution (the agent circles), so
     the band set is read against the control set, not against the
     training set.
2. **Eigenangle calibration.**
   - **Probe walks.** 512 continuous full-circle walks, so held-out-band
     turns are included.
   - **Per-step transition.** Per head, the product over each step's span
     in that head's scan order: the odometry token and every token up to
     the next odometry token.
   - **Plane and angle.** The rotating plane U is found on half the
     episodes without using the turns (mean projector of the significant
     complex eigenpairs). The signed eigenangle of U^T P U is measured on
     the other half.
   - **Fits.** Slope and intercept are fit for |w| < 160 degrees, out of
     band and in band separately. The circular residual (measured minus
     commanded) is reported as RMS, median |.| and the fraction within 10
     degrees.
   - **Reading.** A chart has slope ~1, small residual, and the same in
     band. A staircase has residual structure periodic in the grid,
     slope ~0 inside any 20-degree window between grid points, and wrong
     in-band values.
   - **Quantised-trained models** are probed on continuous turns too. This
     asks whether a model trained only on the grid interpolates between
     grid points.
   - **Synthetic check.** `probe.py --synthetic` builds steps from known
     reflection pairs in a random plane of R^32, with contracting
     distractors, and asserts the pipeline recovers the signed angle to
     1e-8 rad.
3. **beta by turn.** Medians of beta and decay a for the odometry and the
   anchor token of each step, per head, in 30-degree bins of w.
4. **Breakaway.**
   - **Quantity.** Position nmse at lags 17-32 on the training
     distribution: mse on the position pair over its variance, so 1 =
     constant prediction.
   - **Definition.** Breakaway at threshold x = the first eval step (every
     1k) at which it is <= x, for x in {0.5, 0.1, 0.02}.
   - **Censoring.** A run that never crosses is reported as > 100k.

## Results

Pending: the main runs follow this commit.
