# Potential pitfall: endpoint odometry versus twist-driven motion

Proposed2026-10-18. This is a task-design concern and an experimental hypothesis,
not a demonstrated explanation of current learning or transfer failures. No
training inputs, generator, or active run are changed by this note.

## Central hypothesis: internal computation and readout

The primary concern is that non-twist input formats may make the useful
internal representation substantially harder to learn, manipulate, or read
out. A model can solve the rendering task while adopting an internal convention
that makes explicit pose decoding difficult. Endpoint translation plus turn
and twist coordinates contain equivalent local geometric information, but
that equivalence does not imply equal learning difficulty for a finite model,
equal computational cost, or equally accessible learned states.

Twist inputs might better align the learned updates with a common algebraic
representation of motion, making composition, action on coordinates, or pose
readout easier. This is a hypothesis, not an established advantage. Twists do
not make finite noncommuting motions additive: composition still requires the
appropriate group operations (or their learned equivalent). Endpoint
coordinates also have a straightforward composition law, so there is no
mathematical guarantee that twist inputs are easier for this architecture.

The relevant empirical question is whether equivalent twist inputs lead to
more readily usable and decodable internal state at comparable rendering
quality. This is distinct from accidental Exp/Log misuse or transfer to a new
motion distribution. Those are secondary risks and controls below, not the
main explanation being proposed for our observations.

## What the current tasks supply

SE2D supplies a categorical local move and independent quarter turn. SE2C
supplies body-frame endpoint displacement `(dy, dx)` and relative heading.
Its default decoupled generator moves and then turns, with broad independent
turns. These are coordinates of a relative **group element**, not the linear
and angular components of a constant body twist.

Evidence: `CONTRACT_SE2D.md`, `CONTRACT_SE2C.md`,
`mapping/se2c/geometry.py::odometry`, and
`mapping/models/se2d/hybrid6.py::_task_poses`. Hybrid6/7's
`mapping/hybrid6_geometry.py::planar_pose` constructs the endpoint transform
from these coordinates; it does not interpret them as a twist exponential.
The SE2C contract already recognizes twist inputs as a separate task variant.
This audit establishes the issue for these task families, not every historical
run. Pure translation in Z² is the commuting special case; SE2C also retains
an arc-motion generator, although its reported odometry is still an endpoint
relative pose. Ocean/SE3 conventions need their own audit before generalizing.

## The distinction that must not be lost

Using conventional planar matrix coordinates for this equation, write
`T = [R(theta), t; 0, 1]`. For a twist `(v, theta)`,

`Exp(hat(v, theta)) = [R(theta), V(theta) v; 0, 1]`,

where `V(theta) = (sin(theta)/theta) I + ((1-cos(theta))/theta) J`
and J generates the chosen rotation convention. Thus endpoint displacement
`t` is generally not twist translation `v`. Repository `(y,x)` and heading
sign conventions must be respected in any implementation.

For example, in standard `(x,y)` coordinates, exponentiating twist
`(v=(9,0), theta=pi/2)` gives displacement `(18/pi,18/pi)`, not `(9,0)`.
Using the old observations and targets with that replacement corrupts the
odometry labels. The discrepancy can be large at the turns used in training.

This does **not** mean these planar relative transforms lack a logarithm:
with a suitable branch, compute `v = inverse(V(theta)) t` and the exponential
recovers the same endpoints. Half-turn branch choices require care; small
angles require numerically stable formulas. A finite endpoint transform also
does not identify the actual intermediate motion or rotation winding.

Two different interventions must therefore be separated:

1. **Equivalent encoding:** hold poses, observations, query times and targets
   fixed; replace each relative transform by its correctly computed logarithm.
   This tests input representation. If decoded back to the identical matrix
   before the same encoder, it should be a numerical-equivalence control.
2. **Different motion distribution:** sample bounded body twists and integrate
   their exponentials to generate new endpoints and observations. This changes
   the task distribution, translation/rotation coupling and possibly coverage.
   It is not an encoding-only comparison.

## Why this might have dangerous effects

- A silent endpoint/twist substitution can give internally inconsistent
  supervision while looking like an architectural or optimization failure.
- Independent large moves and turns can reward a specialized discrete or
  move-then-turn computation. Success need not imply efficient processing of
  coupled motion, finer temporal sampling, or the intended downstream inputs.
  This is a transfer hypothesis, not proof of a shortcut: solving the existing
  task still requires its correct relative transformations.
- Componentwise scaling of endpoint `(t,theta)` is generally not motion along
  a fixed twist: `T(alpha*t, alpha*theta)` need not equal
  `Exp(alpha*Log(T))`. Interpolation, step subdivision and policy actions can
  therefore test a different geometry than their labels suggest.
- Changing motion also changes overlap, revisit rates and target coverage.
  Apparent improvements can come from texture-retrieval shortcuts rather than
  better composition. This interacts with the
  [partial-coverage proposal](partial_coverage_proposal.md).
- Query poses are endpoint transforms too. Converting odometry while silently
  interpreting query coordinates as twists introduces another mismatch.

## Proposed controlled checks

First establish Exp/Log round trips, inverse/composition agreement, small-angle
and half-turn behavior, and agreement with existing discrete quarter turns.
Apply conversions per step only; do not feed externally integrated poses.
Preserve unknown-odometry flags rather than exponentiating them as identity.

Compare endpoint and correctly converted twist encodings on identical fresh
trajectories, with matched model capacity, training budgets and query semantics.
At matched rendering quality as well as matched training budgets, compare
pose readout from frozen states using identical probe capacity, fresh-data
budgets, and pose targets in a common frame. Report translation and rotation
errors separately, decoder learning curves, and rendering performance; keep
readout targets fixed when changing input encoding. Use causal composition
interventions to test whether any decoding improvement corresponds to more
accessible task-relevant state. Better readout alone would not establish the
precise internal algorithm.

Then separately compare motion distributions, tracking displacement, rotation,
overlap, coverage and query difficulty. Include far queries, shuffled odometry,
and texture OOD controls; do not infer integration from rendering alone.

For temporal subdivision, compare a step T with k increments
`Exp(Log(T)/k)` whose product is T. Intermediate observations/register updates
change the computation, so test odometry composition separately from rendering
with extra observations and changed visibility. Equal endpoint motion alone
does not make those full input sequences equivalent.

The concern warrants explicit experiments before interpreting cross-task
failures as insufficient capacity, hard-to-decode state, or a GDN defect.
