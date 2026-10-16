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
geometric prior. The "geometric deep learning smuggled in" framing is
half right. The trained model does learn a group representation, and
several prior papers (TEM-t, GTA, Gao et al. on grid cells, PaTH) argue
the same thesis. But the delta rule adds capacity, not a symmetry
constraint, and it came from associative memory and formal-language state
tracking, not geometry. The most decisive pending tests are P2 (β capped
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
| F8 | With all odometry removed and a 3 px step on high-passed natural texture, hybrid4 does visual odometry | Wang et al. 2017 (DeepVO); Teed and Deng 2021 (DROID-SLAM); Teed, Lipson and Deng (DPVO); Henriques and Vedaldi 2018 (MapNet registers frames by correlation over a rotation stack) | Reproduces known result in a new setting | Learned frame-to-frame registration is well established. Doing registration and composition in one small in-context model, under the contract's no-external-geometry rule, is a new combination but not a new capability. |
| P1 | Vanilla transformers fail SE(2) up to about 100M params, 400k steps | Liu et al. 2023 (constant-depth shortcuts for solvable groups, learned in practice for D8); Merrill and Sabharwal 2023 (TC⁰ bound does not exclude solvable groups); Sajjadi et al. 2022 and Jin et al. 2024 (geometry-light transformers do novel view synthesis); our own Z² amendment | At risk | Theory and practice both say a transformer can represent this. Vanilla already solved Z² once given 300k steps. If P1 holds, it would be plausibly new; the literature leans against it. |
| P2 | hybrid4 with β capped at 1 fails SE(2) but solves Z² | Grazzi et al. 2025 (Prop. 1: [0, 1] products are not orthogonal); DeltaProduct (β ≤ 1 gives real eigenvalues for two factors) | Known in theory; testing would reproduce it in a new setting | The single-layer prediction is a theorem at arbitrary length. At T = 32 with attention and MLPs around it, a cascade route exists, so failure is likely but not guaranteed. |
| P3 | hybrid3 (scalar decay, SwiGLU) fails SE(2) at every rung to xxl | Liu et al. 2023 (Krohn-Rhodes cascades); Grazzi et al. 2025 App. D (two reflection layers do mod-m addition); DeltaProduct (S3 with n_h = 1 needs 3 layers) | At risk | A depth-2 cascade of positive-decay sums can in principle solve bounded-length SE(2) (section 2). Whether training finds it is open. |
| P4 | Seed robustness, 2 of 3 seeds | none | Met (2 of 2) | Methodological. |
| P5 | L→R delta-rule state at a query linearly decodes relative pose (R² ≥ 0.8) | Banino et al. 2018 (metric quantities decodable from path-integrating RNN); Li et al. 2023 (linear probes of world state); Gao et al. 2021 | At risk, from our own data | The transition probe shows the L→R head contracts every step and the R→L head carries the rotation. The prediction names the wrong head. Pose may still be decodable from the R→L state or the residual stream. |
| P6 | Among vanilla forks, PaTH narrows the SE(2) gap most | Yang et al. 2025 (PaTH: data-dependent Householder products, beyond TC⁰, better on flip-flop and A5); Golovneva et al. 2024 (CoPE); Lin et al. 2025 (FoX) | Plausibly new | PaTH applies the same Householder mechanism inside attention, so the prediction follows from F7. No one has tested PaTH on a spatial task. |

## 4. The "geometric deep learning smuggled in" framing

The claim: delta rules with negative eigenvalues, Householder products and
data-dependent orthogonal position encodings are group-structured
inductive biases in disguise, so geometric deep learning returns inside
general architectures by widening their hypothesis class.

### Where it holds

- **The trained model does learn a group representation.** F7 is the
  strongest evidence available: the per-step transition of the R→L head
  is the faithful 2-D real representation of Z4, exact to under half a
  degree, with |λ| 0.998. This is representation theory realised in
  weights, which is the language of the GDL blueprint (Bronstein et al.
  2021).
- **The mechanism is the affine group of a linear recurrence.** Our
  reading: a recurrence S ← A S + B composes as (A, B) pairs, which is a
  semidirect product of the transition group with additive writes.
  SE(2) = SO(2) ⋉ R² has exactly this form. Once A can rotate, the
  layer's own algebra matches the motion group's. With A a positive
  scalar, the algebra is scaling ⋉ translation, which cannot rotate.
- **Prior papers already argue the thesis**, in different words:
  - Whittington et al. (2022), TEM-t: a transformer whose position
    encodings are generated recurrently by learnable action-dependent
    matrices W_a reproduces place and grid cells. This is a general
    architecture with a path-integrating, action-indexed linear
    transition. It is the closest prior architecture to hybrid4 on a
    closely related task (predicting observations along a walk).
  - Gao et al. (2021): a group-representation condition is necessary for
    path integration, and the linear case is a matrix Lie group of
    rotations.
  - Miyato et al. (2024), GTA: attention should transform queries, keys
    and values by group representations ρ(g) of each token's pose,
    including SE(3) and SO(2) × SO(2).
  - Su et al. (2021), Ostmeier et al. (2025), Schenck et al. (2025):
    RoPE, LieRE and STRING frame positional encoding as Lie-group
    rotations. Li et al. (2025), PRoPE, extends this to full camera
    frustums.
  - Keurti et al. (2023) and Quessard et al. (2020): agents learn group
    representations of their actions from observed transitions.
  - Yang et al. (2025), PaTH, and Grazzi et al. / Siems et al.: state
    tracking via products of Householder transforms, i.e. learned
    orthogonal group elements.

### Where it overreaches

- **Capacity is not a constraint.** GDL's core move is equivariance:
  restricting the function class so that a known symmetry holds by
  construction. The delta rule restricts nothing. It enlarges the set of
  transitions to include orthogonal ones, and the network must discover
  which group to represent. Nothing guarantees the learned map is
  equivariant off the training distribution. "Group-capable" is accurate;
  "group-structured inductive bias" is not, unless a bias toward
  norm-preserving transitions is meant.
- **The history is different.** The delta rule came from fast-weight
  associative memory and online regression; β up to 2 came from formal
  language state tracking (parity, modular counting, permutation
  groups). "Smuggled" implies geometric intent that the authors did not
  have. A fairer phrasing: state-tracking research and geometric deep
  learning converge on the same object, a learned group representation
  in the state transition.
- **The theory gap at bounded length.** The impossibility results are for
  arbitrary length and finite precision. At T = 32 a two-layer
  positive-decay cascade or a transformer shortcut can represent the
  task. So the evidence so far supports "rotation-capable transitions
  make this learnable at small scale", not "general architectures cannot
  do it". P1 and P3 at depth decide which.
- **Evidence on scale cuts both ways.** Brehmer et al. (TMLR) find
  equivariant transformers beat non-equivariant ones at every compute
  budget tested on rigid-body interactions, though augmentation closes
  the data-efficiency gap. Gruver et al. (2023) find that larger, more
  accurate vision models become more equivariant regardless of
  architecture. Jin et al. (2024), LVSM, reach state-of-the-art novel
  view synthesis with minimal 3D inductive bias, though with camera poses
  given as input. None of these test egocentric composition from
  relative odometry.
- **Discrete group, exact symbols.** Z4 with 16 odometry symbols lets the
  network learn an exact rotation per symbol. Continuous headings and
  noisy odometry require a continuous, approximately orthogonal map from
  inputs. Nothing yet shows the mechanism survives that.

### A more defensible statement

Sequence operators whose state transitions can form a matrix group let a
network learn the task's group representation from data. In this study
the trained model did exactly that. This is GDL's representation theory
appearing as a learned object, not GDL's equivariance appearing as a
constraint. Whether it is needed, rather than merely faster, is open
until P1, P2 and P3 report.

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
  equivariance, Genie 3, and the later Neuron (2023) version of Sorscher
  et al.
