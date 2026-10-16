# Literature review: ego-motion composition, state tracking and group structure

Written 2026-10-15. It places the mapping study's findings and frozen
predictions (`HYPOTHESIS.md`, tag `hypothesis-v1`, amendments to commit
19a52d4) against prior work. Every paper cited was located and its title,
authors, year and venue checked against arXiv or the official proceedings
page. A claim attributed to a paper was checked against its abstract or
full text. Where this review reasons beyond what a paper states, it says
so ("our reading").

## 1. Bottom line

The study's central mechanism is already known in theory. Grazzi et al.
(2025) prove that linear RNNs whose transition matrices have eigenvalues
only in [0, 1] cannot be reflections or rotations and cannot solve parity
at arbitrary length, and that extending DeltaNet's β to (0, 2) fixes
this. DeltaProduct (Siems et al. 2025) draws the exact picture hybrid4
relies on: two β = 2 reflections compose into a 2-D rotation. Liu et al.
(2023) show that transformers can represent and learn shallow
"shortcuts" for solvable groups, including gridworld path integration and
dihedral groups. Z4 ⋉ Z² is solvable, so the circuit-complexity
separations (Merrill et al. 2024) do not apply to it, and at a fixed
length of 32 no known theorem forbids a deep positive-decay recurrence or
a transformer from solving the task. What the literature does not
contain, as far as we found, is three things this study has: a
perceptual, pixel-output mapping task in which the non-commuting pose
group is the only difficulty; a controlled lineage in which the
rotation-capable transition is the change that turns failure into
success at about 120k parameters; and a direct measurement that the
trained delta-rule head implements the faithful 2-D representation of Z4
(a conjugate eigenvalue pair at exactly 90 degrees on quarter turns, two
eigenvalues at 180 degrees on turn-arounds, identity otherwise) with no
geometric prior. The user's framing is that general-purpose
architectures need their hypothesis class expanded with geometric
principles, such as state transitions that can act as group elements, to
learn efficiently, or at all, on tasks whose geometric structure has been
underappreciated. Current evidence supports it in its "efficient" form.
The delta rule adds capacity, not an equivariance constraint, which is
what the framing calls for, and the trained model used that capacity as
the motion group's representation. The closest prior versions of the
argument are GTA (group representations in attention, poses given),
TEM-t (action-dependent transitions in a transformer), Gao et al.
(a group representation is necessary for path integration) and PaTH
(data-dependent Householder positions, framed for language). Two caveats
hold. The operator came from state-tracking work, which frames it in
group-theoretic but not geometric terms. And at bounded length theory
does not rule out transformers or positive-decay cascades, so the
evidence supports "much more efficient", not "required". The claim concerns
geometry as an internal operation of an end-to-end memory supervised only
by the downstream objective; evidence from systems that take poses as
input or as supervised output (LVSM, GTA, DROID-SLAM) does not bear on
it. That internal side has prior occupants (TEM-t, with commuting
actions; Banino et al., with pose labels; recent predictive-learning
models), but none we found composes non-commuting ego-motion in context
with render-only supervision. The most decisive pending tests are P2 (β capped
at 1), P3 at depth, P1 at scale, and length generalisation, which the
literature predicts will separate recurrent rotation from transformer
shortcuts.

## 2. Where the task sits in the theory

Three facts from the literature frame every verdict below.

- **Z4 ⋉ Z² is solvable.** Liu et al. (2023) prove that solvable
  semiautomata have constant-depth transformer shortcuts (via
  Krohn-Rhodes), and that non-solvable ones do not unless TC⁰ = NC¹. The
  SE(2) task, with offsets bounded by ±288 px, is a finite solvable group
  action. So it is on the "shortcut exists" side. Liu et al. also give
  even shorter shortcuts for gridworld path integration (their Theorem
  3), and train transformers successfully on C4 and D8 automata. The
  "illusion of state" separation (Merrill et al. 2024) concerns
  NC¹-complete problems such as S5 composition, so it does not bear on
  this task.
- **The negative results for positive recurrences are about arbitrary
  length and a single layer.** Sarrof et al. (2024) prove nonnegative,
  time-varying diagonal SSMs cannot recognise parity at arbitrary length.
  Grazzi et al. (2025) extend this to non-diagonal LRNNs with positive
  eigenvalues, show that products of generalised Householders with
  eigenvalues in [0, 1] are either the identity or not orthogonal, and
  note that cyclic groups can be represented with rotation matrices. They
  also show that two layers using only reflections can do addition modulo
  m. DeltaProduct proves that with two Householder factors, complex
  eigenvalues appear only if both β exceed a threshold above 1. Our
  reading: at T = 32, a two-layer cascade of positive-decay recurrences
  can in principle solve SE(2). Layer 1 sums turns (decay 1); an MLP
  reduces mod 4 and rotates each body-frame move into a fixed frame;
  layer 2 sums those moves. So "fails" at xs (one recurrent block) is
  consistent with theory, but not implied by it at larger depth.
- **Diagonal transitions commute.** Terzić et al. (2025) find that
  complex diagonal selective SSMs learn the commutative C2 × C30 automaton
  but fail in-domain on the solvable non-commutative D30 unless an
  extra input term is added. Our reading: SE(2) with rotations only
  (Z4 ⋉ Z²) is a subgroup of the affine maps z ↦ e^{iθ}z + b of the
  complex plane. A linear recurrence S ← A S + B is affine. So a
  selective recurrence with input-dependent complex phase (a 1-D complex
  diagonal transition plus an additive write) can represent this group's
  action in one layer, while D30's reflections cannot be written that
  way. This makes a complex-diagonal recurrence the sharpest missing
  ablation (section 6).

## 3. Verdicts on findings and predictions

Verdict scale: **known**; **reproduces known result in a new setting**;
**plausibly new**; **at risk / contradicted**.

| # | Our finding or prediction | Closest prior work | Verdict | Why |
| --- | --- | --- | --- | --- |
| F1 | On Z², a plain transformer eventually solves; slower in steps, params and compute than a gated linear-recurrence hybrid | Liu et al. 2023 (gridworld path-integration shortcuts; RNNs better under indirect supervision); Sarrof et al. 2024 (SSMs give exact solutions where transformers struggle) | Reproduces known result in a new setting | Abelian path integration is learnable by shallow transformers. A recurrence computes the sum directly. Our numbers show the gap is speed, not capability, which matches. |
| F2 | On SE(2), positive scalar-decay (GLA-style) hybrids fail, even with SwiGLU, step-RoPE and per-step queries | Grazzi et al. 2025 (Thm 1, Prop 1); Sarrof et al. 2024 (Thm 3.2); Terzić et al. 2025 (D30) | Reproduces known result in a new setting, with a caveat | Theory forbids rotation in a single positive-eigenvalue layer. It does not forbid a deeper cascade at T = 32. Our failures are at xs, one recurrent block; P3 tests depth. |
| F3 | hybrid4 (bidirectional Gated DeltaNet, β ∈ (0, 2), short conv, + RoPE attention, SwiGLU) solves SE(2) at 120k params, 2 of 2 seeds | Grazzi et al. 2025; Yang et al. 2025 (Gated DeltaNet); Siems et al. 2025 (DeltaProduct) | Reproduces known result in a new setting | "Negative eigenvalues unlock state tracking" is established on formal languages. Applying it to egocentric mapping with pixel output, and the clean before/after lineage, is new as a demonstration. |
| F4 | Without SwiGLU, learning is slow; without the delta rule, it fails | Shazeer 2020 (GLU variants); Grazzi et al. and DeltaProduct constructions use MLP lookups between layers | Plausibly new, minor | No paper we found isolates gated MLPs in a state-tracking setting. One seed each, so weak. |
| F5 | Audit: corrupting turns or moves pushes error above floor at every prefix length | Standard causal-ablation practice | Not a literature claim | A control that rules out shortcuts; it strengthens F3 but is not itself a finding to place. |
| F6 | Trained R→L head: β = 2.00 and decay 1.0000 on every odometry token; L→R head grades β by turn; observation tokens decay | Grazzi et al. 2025 (β = 2 gives reflections); DeltaProduct Fig. 3 (two reflections make a rotation) | Plausibly new | The construction is known. We found no paper that reads learned β and decay per token type in a trained model and finds them pinned at the reflection and no-forget values. |
| F7 | Transition probe (both seeds): per-step product in R→L head has one conjugate pair at 90.0 to 90.3 deg on quarter turns (100% of steps, abs 0.998), two eigenvalues at 180 deg on turn-arounds, all others at 0 deg; L→R head contracts (no eigenvalue above 0.5) | Chughtai et al. 2023 (networks learn group composition via irreducible representations); Nanda et al. 2023 (modular addition as rotation about a circle); Gao et al. 2021 (path integration requires a group representation; linear case is a rotation Lie group); Cueva et al. 2020 (head-direction circuits emerge in RNNs) | Plausibly new | Prior learned-representation results either supervise the group operation directly (Chughtai, Nanda) or build the matrix form into the model (Gao). Here a generic delta-rule head, trained only on pixel rendering, learned the faithful 2-D real irrep of Z4 in its transition matrices. Limits: shows use, not necessity; not a linear probe of relative pose. |
| F8 | With all odometry removed and a 3 px step on high-passed natural texture, hybrid4 does visual odometry | DeepVO (2017); DROID-SLAM (2021); DPVO; MapNet (2018); UpSRT (2022); Gornet and Thomson 2024 | Capability known; plausibly new in this form | Prior systems establish that registration and relative-pose estimation are learnable, mostly with pose supervision (DeepVO, DROID-SLAM, DPVO, MapNet) or explicit geometric modules (DROID-SLAM's bundle adjustment, MapNet's rotation stack). UpSRT shows pose-free registration under render-only loss, but over an unordered set. None of these composes a motion history in a latent memory with render-only supervision; see the junction in section 4. |
| P1 | Vanilla transformers fail SE(2) up to about 100M params, 400k steps | Liu et al. 2023 (constant-depth shortcuts for solvable groups, learned in practice for D8); Merrill and Sabharwal 2023 (TC⁰ bound does not exclude solvable groups); Sajjadi et al. 2022 and Jin et al. 2024 (geometry-light transformers do novel view synthesis); our own Z² amendment | At risk | Theory and practice both say a transformer can represent this. Vanilla already solved Z² once given 300k steps. If P1 holds, it would be plausibly new; the literature leans against it. |
| P2 | hybrid4 with β capped at 1 fails SE(2) but solves Z² | Grazzi et al. 2025 (Prop. 1: [0, 1] products are not orthogonal); DeltaProduct (β ≤ 1 gives real eigenvalues for two factors) | Known in theory; testing would reproduce it in a new setting | The single-layer prediction is a theorem at arbitrary length. At T = 32 with attention and MLPs around it, a cascade route exists, so failure is likely but not guaranteed. |
| P3 | hybrid3 (scalar decay, SwiGLU) fails SE(2) at every rung to xxl | Liu et al. 2023 (Krohn-Rhodes cascades); Grazzi et al. 2025 App. D (two reflection layers do mod-m addition); DeltaProduct (S3 with n_h = 1 needs 3 layers) | At risk | A depth-2 cascade of positive-decay sums can in principle solve bounded-length SE(2) (section 2). Whether training finds it is open. |
| P4 | Seed robustness, 2 of 3 seeds | none | Met (2 of 2) | Methodological. |
| P5 | L→R delta-rule state at a query linearly decodes relative pose (R² ≥ 0.8) | Banino et al. 2018 (metric quantities decodable from path-integrating RNN); Li et al. 2023 (linear probes of world state); Gao et al. 2021 | At risk, from our own data | The transition probe shows the L→R head contracts every step and the R→L head carries the rotation. The prediction names the wrong head. Pose may still be decodable from the R→L state or the residual stream. |
| P6 | Among vanilla forks, PaTH narrows the SE(2) gap most | Yang et al. 2025 (PaTH: data-dependent Householder products, beyond TC⁰, better on flip-flop and A5); Golovneva et al. 2024 (CoPE); Lin et al. 2025 (FoX) | Plausibly new | PaTH applies the same Householder mechanism inside attention, so the prediction follows from F7. No one has tested PaTH on a spatial task. |

## 4. The geometric-principles framing

**The claim, as the user states it.** The hypothesis class of so-called
general-purpose architectures needs to be expanded with geometric
principles, for example state transitions that can act as group elements
such as rotations. Then learning becomes more efficient, or possible at
all, on a class of tasks whose geometric structure has been
underappreciated. Whether "efficient" or "possible" holds depends on P1
and P3. The claim is not that models should be constrained to be
equivariant. An earlier draft of this review misread it that way.

### What the evidence supports

- **Added capacity is what the claim asks for.** The delta rule with
  β ∈ (0, 2) imposes no symmetry constraint. It widens the set of state
  transitions from positive scalings to products of reflections, which
  include rotations. That is an expansion of the hypothesis class by a
  geometric principle, transitions that can be group elements. In the
  lineage it is the change that turned failure (F2) into success (F3) at
  about 120k parameters. So the fact that it "adds capacity, not an
  equivariance constraint" supports the claim rather than undercutting
  it.
- **The trained model uses the expansion as the motion group.** F7: the
  right-to-left head's per-step transition is the faithful 2-D real
  representation of Z4, exact to under half a degree, with |λ| 0.998.
  The new capacity was used as a rotation of the plane, not as some
  other state-tracking device.
- **Our reading: a linear recurrence is an affine group.** S ← A S + B
  composes as (A, B) pairs, a semidirect product of the transition group
  with additive writes. SE(2) = SO(2) ⋉ R² has exactly this form. Once A
  can rotate, the layer's algebra matches the motion group's. With A a
  positive scalar it is scaling ⋉ translation, which cannot rotate.

### Prior papers closest to the hypothesis-class argument, and how ours differs

- **Miyato et al. (2024), GTA.** Closest in spirit. They argue that
  position encodings designed for language are suboptimal for 3D vision,
  and put group representations ρ(g) of each token's pose (SE(3),
  SO(2) × SO(2)) into attention, improving learning efficiency for novel
  view synthesis. Difference: GTA is given each token's pose. The group
  element is computed outside the network. Here only relative odometry
  is given, and the network must compose the group elements itself; the
  contract forbids external composition. A GTA model with true poses is
  therefore a ceiling for this task, not a competitor.
- **Whittington et al. (2022), TEM-t, and Whittington et al. (2020),
  TEM.** A transformer whose position codes are generated recurrently by
  learnable action-dependent matrices W_a (with a nonlinearity)
  reproduces place and grid cells. This is a general architecture with
  an action-driven, path-integrating transition, on a closely related
  task (predict observations along a walk). Difference: their aim is a
  model of the hippocampal formation, the path-integration module is a
  dedicated component, and they do not test whether standard sequence
  operators fail without it.
- **Gao et al. (2019, 2021).** They argue that a group-representation
  condition on the self-motion transformation is necessary for path
  integration, and that the linear case is a matrix Lie group of
  rotations. Difference: a normative theory of grid cells, with the
  vector-matrix form built into the model. It argues necessity of the
  representation, not of an architectural capacity in general-purpose
  models.
- **Yang et al. (2025), PaTH; also LieRE, STRING, PRoPE.** PaTH argues
  that RoPE's input-independent transforms limit transformers, and
  replaces them with accumulated data-dependent Householder products.
  It is the closest operator to hybrid4's. Difference: its motivation is
  state tracking and expressivity in language (flip-flop, A5 word
  problems, beyond TC⁰), not geometry. LieRE, STRING and PRoPE extend
  position encodings to Lie groups for vision and robotics, with poses or
  coordinates given.
- **Grazzi et al. (2025) and Siems et al. (2025).** They expand
  linear-RNN transitions with Householder reflections and products of
  them. DeltaProduct notes that such products can form any orthogonal
  matrix (Cartan-Dieudonné) and proves results for finite subgroups of
  O(n) and SO(n). The framing is explicitly group-theoretic, but the
  groups are those of formal-language word problems, not of physical
  space.
- **How ours differs overall.** (i) The task is physical ego-motion with
  egocentric observations; nothing but relative odometry is given.
  (ii) The target is the general-purpose architectures used for robot
  policies and video world models, where this structure is the stated
  pain point. (iii) There is mechanistic evidence that the added
  capacity is used as the motion group's representation. None of the
  papers above combines these.

### Where the geometry lives: the junction

Prior systems that "learn geometry" fall on two sides of a junction, and
only one side bears on this study.

- **(a) Geometry as an output or an input.** Pose or depth is the
  supervised target, or poses are given in a common frame, and the map,
  renderer or pose graph downstream is an explicit system.
- **(b) Geometry as an internal operation.** A latent memory must
  compose the motion itself, supervised only by the downstream objective
  (rendered or predicted observations), with only relative motion, or
  nothing, as input.

This study is (b) in miniature: only rendered pixels supervise, odometry
is relative (or absent in the visual-odometry setting), there are no pose
labels and no external renderer, and each episode is a new map, so the
composition must happen in activations, not weights.

| System | Geometry enters as | Supervised by | Who composes the motion |
| --- | --- | --- | --- |
| DeepVO (Wang et al. 2017) | Output | MSE to ground-truth pose | Not composed in memory; pose is the target |
| DROID-SLAM (Teed and Deng 2021); DPVO (Teed et al.) | Output, via an explicit dense bundle-adjustment layer (DROID-SLAM) | Ground-truth pose and flow losses (DROID-SLAM); ground-truth trajectories (DPVO) | Explicit optimisation |
| MapNet (Henriques and Vedaldi 2018) | Internal but hand-built: rotation stack, allocentric grid | Ground-truth position and orientation | Explicit module |
| Banino et al. 2018; Cueva and Wei 2018 | Internal: an RNN integrates velocities (Banino: translational and angular, so non-commuting) | Place- and head-direction-cell targets, i.e. pose labels | The network, under pose supervision |
| GQN (Eslami et al. 2018); SRT (Sajjadi et al. 2022); LVSM | Input: each image comes with its viewpoint or camera pose | Rendered pixels only | Not needed: poses are given. GQN sums per-view codes without order, so (our reading) viewpoints share one frame |
| UpSRT (Sajjadi et al. 2022) | No input poses; target views still specified by a ray pose (frame not checked) | Rendered pixels only | Registration over an unordered set, not composition of a motion history |
| GTA; PRoPE; WorldMem, Wu et al., Context as Memory | Input: poses, or an explicit 3D or pose-indexed memory | Pixels (world models) or task loss | Outside the network |
| TEM (2020); TEM-t (Whittington et al. 2022) | Internal: learned action-indexed transitions | Next observation only | The network, in context, in new environments; but actions are allocentric (North, East, ...), so the group commutes |
| Gornet and Thomson 2024 | Internal: images only, self-attention over up to about ten past frames | Next image only | The network; we found no test on unseen environments, so the map may live in weights |
| Levenstein et al. 2024 (preprint) | Internal: egocentric 7 x 7 view plus action | Next egocentric observation | The network; head-direction input matters. We did not verify the action set or whether maps are per episode |
| This study | Internal: relative odometry (Z4 ⋉ Z²) or none | Rendered pixels only | The network, in context, new map each episode |

**What this changes.** Evidence from side (a), such as LVSM's
state-of-the-art view synthesis with minimal 3D bias, shows that
networks can consume geometry that is supplied or supervised. It does not
show they can compose it internally, which is what the hypothesis-class
claim is about. When poses are given, the composition was done outside
the network, as with marlenv's dead-reckoned RoPE. Side (b) has prior
occupants, mostly in neuro-AI predictive learning, and Banino et al.
compose non-commuting ego-motion internally but with pose labels. As far
as we checked, none combines in-context composition of non-commuting
ego-motion over a fresh map each episode, render-only supervision, and a
controlled comparison of sequence-operator transition classes. How
practice at scale sits on this junction is the subject of the separate
applications review, `docs/applications_review.md` (in progress), and is
not repeated here.

### Caveats

- **(a) Provenance.** The negative-eigenvalue delta rule came from
  state-tracking and formal-language work. DeltaNet itself came from
  associative recall framed as online regression. Grazzi et al. motivate
  β ∈ (0, 2) by parity, modular counting and code evaluation. DeltaProduct
  is motivated by permutation-group word problems. These papers frame the
  operator as state-tracking expressivity, in group-theoretic but not
  geometric terms. Provenance does not decide whether the principle is
  geometric: a rotation of the state is a rotation, and F7 shows the
  model uses it as one. But "underappreciated" is best stated as
  "underappreciated for spatial and embodied tasks", since the
  state-tracking literature already identifies the principle.
- **(b) Bounded length.** The impossibility results are for arbitrary
  length. At T = 32, theory does not rule out a transformer shortcut
  (Liu et al.) or a two-layer positive-decay cascade (section 2). So
  current evidence supports "much more efficient", not "required".
  P1 and P3 decide between "efficient" and "possible".
- **(c) Scale evidence is mixed, and mostly from side (a).** Brehmer et
  al. find built-in symmetry helps at every compute budget tested on
  rigid-body interactions, though augmentation closes the data-efficiency
  gap. Gruver et al. (2023) find larger vision models become more
  equivariant regardless of architecture. LVSM reaches state-of-the-art
  view synthesis with minimal 3D bias, but with poses given, so it is
  side (a) evidence and does not test internal composition.
- **(d) Discrete group, exact symbols.** Z4 with 16 odometry symbols
  allows an exact rotation per symbol. Continuous headings and noisy
  odometry need a continuous, approximately orthogonal map. Nothing yet
  shows the mechanism survives that.

### Assessment

On current evidence the framing holds in its "efficient" form: expanding
the transition class to include rotations made the task learnable at
120k parameters and 100k steps, where the positive-decay lineage failed,
and the model used the expansion as the motion group's representation.
The "possible at all" form is open until P1 and P3 report. The closest
prior versions of the argument are GTA, for attention with given poses,
and TEM-t, for action-driven transitions in a transformer.

## 5. Open questions the literature leaves that this setup can answer

1. **Necessity at bounded length.** Theory separates positive and
   rotation-capable transitions only at arbitrary length. Does the
   separation hold empirically at T = 32 when depth and attention are
   available? (P2, P3 with depth.)
2. **Do index-RoPE transformers learn non-commuting ego-motion in a
   perceptual task?** Liu et al. show learnability for small automata
   with dense state supervision. Nobody has tested it with pixel targets
   and indirect supervision through rendering. (P1.)
3. **Length generalisation.** Liu et al. find transformer shortcuts
   brittle off-distribution and at unseen lengths; DeltaProduct reports
   better extrapolation with more Householder factors. Train at 32
   steps, test at 64 and 128: a true rotation should extrapolate, a
   shortcut should not.
4. **Which representation, and is it universal?** Chughtai et al. find
   that networks learn irreducible representations but which ones is
   arbitrary across seeds. Our probe found the faithful 2-D irrep in both
   seeds. Does that hold across sizes, and does any head use the sign
   (1-D) irrep for turn-arounds?
5. **Rotation in attention versus recurrence.** PaTH puts the Householder
   product inside attention. Does it learn the same Z4 representation in
   its accumulated transforms?
6. **Continuous SE(2) with noise.** Does a delta-rule model learn an
   approximate SO(2) representation from continuous odometry, and how
   does error accumulate compared with Gao et al.'s conformal-isometry
   analysis?

## 6. Experiments and baselines the literature implies are missing

- **Complex-diagonal selective recurrence** (input-dependent phase, as in
  a selective LRU-style layer). It can rotate but its transitions
  commute. By our reading of section 2 it can still represent Z4 ⋉ Z²
  through the additive write. If it solves SE(2), H should be restated
  from "non-diagonal" to "rotation-capable". If it fails, commutation
  matters. This separates the two halves of H.
- **Mamba or GLA with eigenvalues in [−1, 1]** (Grazzi et al.'s diagonal
  variant). It can represent turn-around (−1) but not quarter turns.
  Prediction: fails on quarter turns, partly succeeds on others.
- **DeltaProduct with n_h = 2.** Removes the need for the short
  convolution to supply the second reflection. Pairs with the pending
  no-conv ablation.
- **TEM-t style baseline** (Whittington et al. 2022): softmax attention
  whose position code is produced by a recurrent, action-indexed learned
  matrix. Contract-compliant if the matrix is learned per odometry
  symbol.
- **GTA as an oracle ceiling** (Miyato et al. 2024): attention with ρ(g)
  from true poses. This breaks the contract's no-external-geometry rule,
  so it is a ceiling, like the marlenv dead-reckoned RoPE, not a
  competitor.
- **Explicit SE(2)-equivariant baseline:** a MapNet-style allocentric map
  with registration over a C4 rotation stack (Henriques and Vedaldi
  2018), or a C4-steerable observation encoder (Weiler and Cesa 2019).
  The encoder does not use odometry, so it is contract-compliant; it
  removes the fragment-alignment half of the task and isolates
  composition.
- **Length-generalisation evaluation** at 64 and 128 steps for every
  solving model, as above.
- **P5 as a probe of the R→L state**, plus a Chughtai-style check of
  which irreps appear across seeds and rungs.

## 7. Bibliography

All entries were checked on 2026-10-15 against the arXiv abstract page or
the official proceedings page given.

**State tracking and expressivity**

- Liu, Ash, Goel, Krishnamurthy, Zhang. Transformers Learn Shortcuts to
  Automata. ICLR 2023. https://arxiv.org/abs/2210.10749
- Merrill, Sabharwal. The Parallelism Tradeoff: Limitations of
  Log-Precision Transformers. TACL 2023.
  https://arxiv.org/abs/2207.00729
- Merrill, Sabharwal. The Expressive Power of Transformers with Chain of
  Thought. ICLR 2024. https://arxiv.org/abs/2310.07923
- Merrill, Petty, Sabharwal. The Illusion of State in State-Space Models.
  ICML 2024. https://arxiv.org/abs/2404.08819
- Sarrof, Veitsman, Hahn. The Expressive Capacity of State Space Models:
  A Formal Language Perspective. NeurIPS 2024.
  https://arxiv.org/abs/2405.17394
- Grazzi, Siems, Zela, Franke, Hutter, Pontil. Unlocking State-Tracking in
  Linear RNNs Through Negative Eigenvalues. ICLR 2025.
  https://arxiv.org/abs/2411.12537
- Yang, Wang, Zhang, Shen, Kim. Parallelizing Linear Transformers with
  the Delta Rule over Sequence Length. NeurIPS 2024.
  https://arxiv.org/abs/2406.06484
- Yang, Kautz, Hatamizadeh. Gated Delta Networks: Improving Mamba2 with
  Delta Rule. ICLR 2025. https://arxiv.org/abs/2412.06464
- Siems, Carstensen, Zela, Hutter, Pontil, Grazzi. DeltaProduct:
  Improving State-Tracking in Linear RNNs via Householder Products.
  2025. https://arxiv.org/abs/2502.10297
- Peng et al. RWKV-7 "Goose" with Expressive Dynamic State Evolution.
  2025. https://arxiv.org/abs/2503.14456
- Terzić, Hersche, Camposampiero, Hofmann, Sebastian, Rahimi. On the
  Expressiveness and Length Generalization of Selective State-Space
  Models on Regular Languages. AAAI 2025.
  https://arxiv.org/abs/2412.19350
- Movahedi, Sarnthein, Cirone, Orvieto. Fixed-Point RNNs: Interpolating
  from Diagonal to Dense. NeurIPS 2025.
  https://arxiv.org/abs/2503.10799
- Shazeer. GLU Variants Improve Transformer. 2020.
  https://arxiv.org/abs/2002.05202

**Position encodings and geometric attention**

- Su, Lu, Pan, Murtadha, Wen, Liu. RoFormer: Enhanced Transformer with
  Rotary Position Embedding. 2021. https://arxiv.org/abs/2104.09864
- Miyato, Jaeger, Welling, Geiger. GTA: A Geometry-Aware Attention
  Mechanism for Multi-View Transformers. ICLR 2024.
  https://arxiv.org/abs/2310.10375
- Ostmeier et al. LieRE: Lie Rotational Positional Encodings. ICML 2025.
  https://arxiv.org/abs/2406.10322
- Schenck et al. Learning the RoPEs: Better 2D and 3D Position Encodings
  with STRING. 2025. https://arxiv.org/abs/2502.02562
- Li, Yi, Liu, Gao, Ma, Kanazawa. Cameras as Relative Positional
  Encoding. 2025. https://arxiv.org/abs/2507.10496
- Yang, Shen, Wen, Tan, Mishra, Ren, Panda, Kim. PaTH Attention: Position
  Encoding via Accumulating Householder Transformations. NeurIPS 2025.
  https://arxiv.org/abs/2505.16381
- Golovneva, Wang, Weston, Sukhbaatar. Contextual Position Encoding:
  Learning to Count What's Important. 2024.
  https://arxiv.org/abs/2405.18719
- Lin, Nikishin, He, Courville. Forgetting Transformer: Softmax Attention
  with a Forget Gate. ICLR 2025. https://arxiv.org/abs/2503.02130

**Geometric deep learning and equivariance**

- Bronstein, Bruna, Cohen, Veličković. Geometric Deep Learning: Grids,
  Groups, Graphs, Geodesics, and Gauges. 2021.
  https://arxiv.org/abs/2104.13478
- Cohen, Welling. Group Equivariant Convolutional Networks. ICML 2016.
  https://arxiv.org/abs/1602.07576
- Weiler, Cesa. General E(2)-Equivariant Steerable CNNs. NeurIPS 2019.
  https://arxiv.org/abs/1911.08251
- Wang, Walters, Platt. SO(2)-Equivariant Reinforcement Learning. ICLR
  2022. https://arxiv.org/abs/2203.04439
- Huang, Wang, Walters, Platt. Equivariant Transporter Network. RSS 2022.
  https://www.roboticsproceedings.org/rss18/p007.html (arXiv
  https://arxiv.org/abs/2202.09400)
- Wang et al. Equivariant Diffusion Policy. CoRL 2024.
  https://arxiv.org/abs/2407.01812
- Gruver, Finzi, Goldblum, Wilson. The Lie Derivative for Measuring
  Learned Equivariance. ICLR 2023. https://arxiv.org/abs/2210.02984
- Brehmer, Behrends, de Haan, Cohen. Does equivariance matter at scale?
  TMLR. https://arxiv.org/abs/2410.23179
- Sajjadi et al. Scene Representation Transformer: Geometry-Free Novel
  View Synthesis Through Set-Latent Scene Representations. CVPR 2022.
  https://arxiv.org/abs/2111.13152
- Jin et al. LVSM: A Large View Synthesis Model with Minimal 3D Inductive
  Bias. 2024 (ICLR 2025). https://arxiv.org/abs/2410.17242

**Path integration, grid cells and learned group representations**

- Eslami et al. Neural scene representation and rendering. Science
  360(6394), 1204-1210, 2018. https://doi.org/10.1126/science.aar6170
- Gornet, Thomson. Automated construction of cognitive maps with visual
  predictive coding. Nature Machine Intelligence 6, 820-833, 2024.
  https://doi.org/10.1038/s42256-024-00863-1 (preprint:
  https://arxiv.org/abs/2308.10913)
- Levenstein, Efremov, Henha Eyono, Peyrache, Richards. Sequential
  predictive learning is a unifying theory for hippocampal
  representation and replay. bioRxiv preprint, 2024.
  https://doi.org/10.1101/2024.04.28.591528

- Banino, Barry et al. Vector-based navigation using grid-like
  representations in artificial agents. Nature 557, 429-433, 2018.
  https://doi.org/10.1038/s41586-018-0102-6
- Cueva, Wei. Emergence of grid-like representations by training
  recurrent neural networks to perform spatial localization. ICLR 2018.
  https://arxiv.org/abs/1803.07770
- Cueva, Wang, Chin, Wei. Emergence of functional and structural
  properties of the head direction system by optimization of recurrent
  neural networks. ICLR 2020. https://arxiv.org/abs/1912.10189
- Gao, Xie, Zhu, Wu. Learning Grid Cells as Vector Representation of
  Self-Position Coupled with Matrix Representation of Self-Motion. ICLR
  2019. https://arxiv.org/abs/1810.05597
- Gao, Xie, Wei, Zhu, Wu. On Path Integration of Grid Cells: Group
  Representation and Isotropic Scaling. NeurIPS 2021.
  https://proceedings.neurips.cc/paper/2021/hash/f076073b2082f8741a9cd07b789c77a0-Abstract.html
- Sorscher, Mel, Ganguli, Ocko. A unified theory for the origin of grid
  cells through the lens of pattern formation. NeurIPS 2019.
  https://proceedings.neurips.cc/paper/2019/hash/6e7d5d259be7bf56ed79029c4e621f44-Abstract.html
- Whittington et al. The Tolman-Eichenbaum Machine: Unifying Space and
  Relational Memory through Generalization in the Hippocampal Formation.
  Cell 183(5), 2020. https://doi.org/10.1016/j.cell.2020.10.024
- Whittington, Warren, Behrens. Relating transformers to models and
  neural representations of the hippocampal formation. ICLR 2022.
  https://arxiv.org/abs/2112.04035
- Quessard, Barrett, Clements. Learning Group Structure and Disentangled
  Representations of Dynamical Environments. NeurIPS 2020.
  https://arxiv.org/abs/2002.06991
- Keurti, Pan, Besserve, Grewe, Schölkopf. Homomorphism Autoencoder:
  Learning Group Structured Representations from Observed Transitions.
  ICML 2023. https://arxiv.org/abs/2207.12067

**Learned mapping, world models and visual odometry**

- Gupta et al. Cognitive Mapping and Planning for Visual Navigation. CVPR
  2017. https://arxiv.org/abs/1702.03920
- Parisotto, Salakhutdinov. Neural Map: Structured Memory for Deep
  Reinforcement Learning. 2017 (ICLR 2018).
  https://arxiv.org/abs/1702.08360
- Henriques, Vedaldi. MapNet: An Allocentric Spatial Memory for Mapping
  Environments. CVPR 2018.
  https://openaccess.thecvf.com/content_cvpr_2018/papers/Henriques_MapNet_An_Allocentric_CVPR_2018_paper.pdf
- Fang, Toshev, Fei-Fei, Savarese. Scene Memory Transformer for Embodied
  Agents in Long-Horizon Tasks. CVPR 2019.
  https://arxiv.org/abs/1903.03878
- Chaplot, Gandhi, Gupta, Gupta, Salakhutdinov. Learning to Explore using
  Active Neural SLAM. ICLR 2020. https://arxiv.org/abs/2004.05155
- Xiao et al. WorldMem: Long-term Consistent World Simulation with
  Memory. 2025. https://arxiv.org/abs/2504.12369
- Wu et al. Video World Models with Long-term Spatial Memory. 2025.
  https://arxiv.org/abs/2506.05284
- Yu et al. Context as Memory: Scene-Consistent Interactive Long Video
  Generation with Memory Retrieval. SIGGRAPH Asia 2025.
  https://arxiv.org/abs/2506.03141
- Wang, Clark, Wen, Trigoni. DeepVO: Towards End-to-End Visual Odometry
  with Deep Recurrent Convolutional Neural Networks. ICRA 2017.
  https://arxiv.org/abs/1709.08429
- Teed, Deng. DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and
  RGB-D Cameras. NeurIPS 2021. https://arxiv.org/abs/2108.10869
- Teed, Lipson, Deng. Deep Patch Visual Odometry. NeurIPS 2023.
  https://arxiv.org/abs/2208.04726

**Mechanistic interpretability of group operations**

- Chughtai, Chan, Nanda. A Toy Model of Universality: Reverse Engineering
  How Networks Learn Group Operations. ICML 2023.
  https://proceedings.mlr.press/v202/chughtai23a.html
- Nanda, Chan, Lieberum, Smith, Steinhardt. Progress measures for
  grokking via mechanistic interpretability. ICLR 2023.
  https://arxiv.org/abs/2301.05217
- Li, Hopkins, Bau, Viégas, Pfister, Wattenberg. Emergent World
  Representations: Exploring a Sequence Model Trained on a Synthetic
  Task. ICLR 2023. https://arxiv.org/abs/2210.13382

### Notes on verification

- Venues not printed on the arXiv page (ICLR 2018 for Cueva and Wei,
  ICLR 2019 for Gao et al. 2019, ICLR 2018 for Neural Map, CVPR 2017 for
  Gupta et al., NeurIPS 2021 and 2023 for DROID-SLAM and DPVO, ICLR 2025
  for LVSM, ICLR 2023 for Nanda et al., ICLR 2022 for
  Whittington et al. 2022, and the year of LieRE's ICML version) are from memory and were not
  separately checked. Titles, authors and arXiv ids were checked.
- The relevance of the world-model papers (WorldMem, Wu et al., Context
  as Memory) rests on their abstracts: each reports that video world
  models struggle with long-term spatial consistency and fixes it with a
  pose-indexed or geometry-grounded memory, which is external geometry
  in this study's terms.
- Left out because not verified: claims about AlphaFold 3 dropping
  equivariance, Genie 3, the later Neuron (2023) version of Sorscher
  et al., and a 2020 bioRxiv preprint on egocentric-to-allocentric
  predictive learning (doi 10.1101/2020.11.11.378141) whose author list
  we could not confirm.
- Gornet and Thomson's journal volume and pages, and Levenstein et al.'s
  action set, come from search summaries; bioRxiv blocked direct
  retrieval. Titles, authors and dois were checked.
