# Group composition over SO(3) / SE(3): literature check

Date: 2026-10-16. Purpose: position the planned toy experiment (pure pose
composition over octahedral S4, icosahedral A5 and Haar-random SO(3)/SE(3);
hybrid4 vs vanilla4; causal vs bidirectional; sweeps over T and depth) before
pre-registration.

Method. Every claim below comes from a paper page that was opened (arXiv
abstract or HTML full text; see Sources). Section, theorem, table and figure
numbers are as given in those pages. Several key results live only in figure
images (heatmaps, layer-count plots) that the HTML text does not carry. Where
that happened, the text says **not verified**. Papers from Sept 2026 or later
are recent preprints. They are labelled as such and should be weighted
accordingly.

---

## 1. Summary

- **Already done, symbolic groups.** These results exist for finite groups
  given as tokens:
  - Transformers need depth that grows with length on the A5 word problem
    (Merrill, Petty, Sabharwal 2024, Fig. 3).
  - The growth is about logarithmic: d ≈ 4.8·log₂n − 15.8 (Merrill &
    Sabharwal 2025, §7, Fig. 2).
  - Width would have to grow exponentially instead (same figure).
  - Recurrent models (RNN, IDS4) stay at one layer.
- **Already done, and against our predicted contrast.** Transformers' depth
  *also* grows with length for the solvable A4×Z5 (Merrill et al. 2024, §6,
  result 2). The "solvable stays flat" half of our hypothesis was tested
  there on symbolic groups, and it did not hold at trainable scales.
- **Already done for DeltaNet-family models: S4 and A5 as rotation groups.**
  - DeltaProduct (§5.2) explains its S4 and A5 results by their embedding
    in SO(3), the cube and dodecahedron groups. One layer with two
    Householders per token (n_h = 2) suffices for both.
  - Complex KDA (2026, Thm. 3, Table 1) proves that one layer tracks every
    finite subgroup of SO(3). Its Table 1 lists plain **Gated DeltaNet**
    as needing about 4 layers for S4/A5.
- **A design problem we have to fix: one Gated DeltaNet token per step
  cannot apply a 3D rotation.**
  - A single update α(I − βkkᵀ) with α ∈ (0,1) and β ∈ (0,2) is, at best,
    a scaled reflection (determinant −1).
  - A rotation needs two reflections. Our own hybrid4 doc says the same
    (docs/models/hybrid4.md §3.2–3.3).
  - On SE(2), hybrid4 got its second reflection from the 9 patch tokens
    after each odometry token, through the short convolution.
  - A pure-composition stream with one increment token per step has no
    such neighbour tokens. "The hybrid stays flat because the delta rule
    rotates its state" is then not supported by theory unless the design
    changes (see §4).
- **Not found, so plausibly new:**
  1. Continuous Haar-random SO(3) or SE(3) composition used as a word
     problem for transformers or delta-rule hybrids, with depth or width
     sweeps over T.
  2. Finite rotation groups given *geometrically* (as matrices or
     quaternions) rather than as symbols, in the transformer-depth setting.
  3. Any causal-versus-bidirectional comparison on state tracking.

---

## 2. Prior results

"Sym" = group elements given as discrete tokens. "Geo" = given as
rotation matrices, quaternions or angles.

| Work | Models | Groups / tasks | Input | Lengths | Metric | Main result | Open |
|---|---|---|---|---|---|---|---|
| Liu, Ash, Goel, Krishnamurthy, Zhang 2023, *Transformers learn shortcuts to automata* (arXiv 2210.10749) | GPT-2-style transformer, L = 1–16; LSTM | 19 semiautomata incl. parity, cyclic, dihedral, A4, S4, A5, S5, Dyck, gridworld | Sym | T = 100 for the depth sweep (App. B.1.1); larger T was "prohibitively unstable" | accuracy; max and median over 20 runs (Figs. 8, 9) | Theory: depth O(log T) for any automaton (Thm. 1); for solvable ones, depth O(\|Q\|² log\|Q\|), independent of T, but MLP width grows with T (Thm. 2); no constant depth for non-solvable unless TC⁰ = NC¹ (Thm. 4). Experiments: all groups > 99% in distribution at T = 100; "S5 requiring the largest depth" (App. B.1.1). Shortcuts fail to length-generalize (§5.2, Fig. 7); scratchpad plus recency bias recovers recurrent solutions | No sweep of depth against T. Per-group depth thresholds are only in heatmap images (**not verified**) |
| Merrill, Petty, Sabharwal 2024, *The illusion of state in state-space models* (arXiv 2404.08819) | Transformer, S4, Mamba, RNN, IDS4 | A5 (non-solvable), A4×Z5 (solvable), Z60 (abelian), all of order 60; prefix-product tagging | Sym | "successively larger values of n" (§6); exact range **not verified** | minimum layers for > 90% accuracy (Fig. 3) | S4 and Mamba are in TC⁰ (Thms. 4.2, 4.4, 4.6) and so cannot solve the S5 word problem (Cor. 4.7). Transformer, S4 and Mamba need "depths monotonically increasing in sequence length"; RNN and IDS4 need one layer (Fig. 3). Transformers "require greater depth even for A4×Z5" (§6, result 2) | Positional encoding, masking, seeds and the maximum depth are not given in the text I could read |
| Merrill & Sabharwal 2025, *A little depth goes a long way* (arXiv 2503.03961; NeurIPS 2025 per arXiv comment) | Non-looped transformers | A5 state tracking only | Sym | up to 1024, curriculum from length 2 | n* = largest index with ≥ 95% validation accuracy, best seed | Depth ⌈log₂ n⌉ suffices for regular languages (Thm. 1). Width scaling (Thm. 3) and O(log n) chain of thought (Thm. 4) stay in TC⁰. Empirical fits (Fig. 2): d ≈ 4.8·log₂n − 15.8 (r² = 0.93) at width 512; n ≈ 7.2·log₂w − 41.7 at depth 6 | No solvable or abelian control. Positional encoding and masking not stated |
| Merrill & Sabharwal 2023, *The parallelism tradeoff* (arXiv 2207.00729; TACL) | Log-precision transformers | — | — | — | — | Log-precision transformers are simulated by constant-depth, log-space-uniform threshold circuits (i.e. TC⁰) | Theory only |
| Sarrof, Veitsman, Hahn 2024, *Expressive capacity of SSMs* (arXiv 2405.17394; NeurIPS 2024) | Mamba | 27 Bhattamishra languages, flip-flop, bounded Dyck | Sym | train 1–50, test 51–100 | per-string, all-positions correct | Nonnegative-gate SSMs cannot do parity at finite precision (Thm. 3.2), and model a regular language at all lengths iff it is star-free (Thm. 3.4, via Krohn–Rhodes). Mamba length-generalizes on star-free languages and fails on the rest | No groups beyond counting mod K |
| Grazzi et al. 2024/25, *Unlocking state-tracking in linear RNNs through negative eigenvalues* (arXiv 2411.12537) | Mamba and DeltaNet with eigenvalues in [0,1] vs [−1,1]; mLSTM, sLSTM, transformer | parity, mod arithmetic, S5 (§5.2), Z60 (App. E.2.2) | Sym | Chomsky tasks: train 3–40, test 40–256. S5: train 32 (one panel 64) | scaled accuracy; sequence accuracy | Eigenvalues must leave [0,∞) for parity (Thm. 1) and become complex for mod-m counting (Thm. 2). One layer of k−1 reflections *per token* implements a group automaton (Thm. 3). DeltaNet[−1,1] solves parity (1.000 vs 0.017). On S5, DeltaNet[−1,1] needs 5 layers and **4 tokens per group element** to extrapolate to about 200 (Fig. 4) | Multi-layer DeltaNet[−1,1] expressivity "still unclear" (Conclusion) |
| Siems et al. 2025, *DeltaProduct* (arXiv 2502.10297) | DeltaProduct with n_h Householders per token; one layer, 12 heads × 32 (App. C.1) | S3, S4, A5, S5 | Sym (encoding not stated in C.1) | train 128, test 512 | accuracy | S_n needs n_h = n−1 at one layer (Thm. 1). Empirically S4 and A5 extrapolate with n_h = 2: "isomorphism to subgroups of SO(3)… only require n_h=2 and keys of size 3" (§5.2, Fig. 6). With n_h = 1, S3 needs 3 layers, **S4 6** and **A5 3**; S5 is unsolved at 10 layers. [0,1] eigenvalues fail even in training (Fig. 12). Probed β ≈ 2, so two reflections give a rotation (Fig. 7) | Inputs are symbols; no continuous SO(3); no transformer depth comparison |
| Siems, Grazzi et al. 2026, *Complex KDA* (arXiv 2609.24797; recent preprint) | CKDA, KDA, Gated DeltaNet, Gated DeltaProduct, causal transformer | S3, S4, A5; audio continuation (2D rotations) | Sym | train ≤ 32 | scaled accuracy | One CKDA layer tracks every finite subgroup of SO(3) (Thm. 3). A5 is impossible in dimension 3 for signed-Householder transitions (Thm. 17) but possible in dimension 4 with a quaternion construction (Prop. 18). Table 1, as I parsed it from garbled HTML: GDN α(I−βkkᵀ), α ∈ [0,1], β ∈ [0,2], needs 2 layers for cyclic/dihedral groups and **4 for S4/A5**; Gated DeltaProduct with k ≥ 2 needs 1. Plain training on A5 fails; initialising near the construction works | Table 1 reading **partly unverified**. Experiment details (App. G) not read |
| Peng et al. 2025, *RWKV-7* (arXiv 2503.14456) | RWKV-7 vs transformer, Mamba, S4, RNN | A5, A4×Z5, Z60, following Merrill et al. | Sym | **not verified** | min layers for > 95% (Fig. 8) | Claims S5 with one layer and all regular languages with constant depth (§1, App. D). Better than transformers, Mamba and S4 in Fig. 8 | Layer counts only in the figure image |
| Li, Guo, Andreas 2025, *(How) do language models track state?* (arXiv 2503.02854) | Pythia, GPT-2 (pretrained and from scratch) | S3, S5 | Sym | train 100 | probes, patching, length cutoff at 98% | Models learn an associative scan, where layer l covers about 2^l tokens (§3), or a parity-then-scan variant (§4). Which one depends on architecture and initialisation (§5.2) | Depth tied to log length, mechanistically |
| Zhang & Li 2026, *Tracking states or tracking cosets?* (arXiv 2609.29951; recent preprint) | Transformers (4L/256, 3L/512), LSTM, GRU, Mamba-2 | 17 finite groups incl. S4, A5, S5 | Sym | train 100 | exact, quotient and conditional accuracy | Transformers plateau at about 1/(class size) of the abelianization: right coset, uniform within it (§3.2–3.4). S5 transformers stop at the A5 stage (§5.2) | Pitfall for S4, which has the S3 and Z2 quotients |
| Jia 2026, *Beyond diagonal SSMs* (arXiv 2610.00329; single-author recent preprint) | Diagonal SSM baselines vs Lie-group SSMs (SU(2), SO(3), Cayley SO(N), SE(3)) | D4, Q8, S3, A5, S5; gyro integration; rigid body | Sym for groups; Geo (noisy gyro) for IMU | groups at L = 16/32 (train 32 → test 128); IMU T = 100 | accuracy; geodesic degrees | SO(3)-SSM gets 100% on A5 at L = 32 vs 2–9% for diagonal SSMs (Table 1/2). Noisy-gyro attitude: 3.42° vs 3.80° for exact dead reckoning (§3.3) | **No transformer, DeltaNet or Mamba baselines; no length or depth sweep; no bidirectional comparison.** Some internal number inconsistencies |
| Yang, Kautz, Hatamizadeh 2024, *Gated DeltaNet* (arXiv 2412.06464) | — | — | — | — | — | S_t = S_{t−1} α_t(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ (Eq. 10). α_t ∈ (0,1) is a scalar; β ∈ (0,1), with (0,2) noted in footnote 2; L2-normalised q and k (§3.4) | Defines our recurrent layer |
| Ewer et al. 2024, *ENTP* (arXiv 2410.01600) | Encoder-only (recomputed per prefix) vs decoder-only | Count3, addition, ICL, LM | Sym | — | — | The two express different function classes (Thms. 4.1, 4.2). The encoder learns Count3 and decoders fail (§6) | **No state-tracking or group tasks** |
| Continuous IMU / pose: EqNIO (arXiv 2408.06321); CTIN (AAAI 2022, 2112.02143, abstract seen via search only) | ResNet / TLIO / RoNIN backbones; transformer (CTIN) | Real IMU odometry | Geo, noisy | windowed | ATE / displacement | Networks regress windowed displacement. Orientation integration and composition are done by a filter or canonicalisation, not learned over long sequences | Nothing tests learned composition over growing T |
| Representation papers: Zhou et al. (1812.07035); Levinson et al. (2006.14616) | — | — | — | — | — | Every rotation representation in ≤ 4D real space is discontinuous; 5D and 6D continuous ones exist (Zhou). SVD orthogonalisation of a 3×3 output works well (Levinson) | Output-head design |

---

## 3. What is new in our design, and what duplicates prior work

### Duplicates (pre-register as replication or sanity checks, not as findings)

1. **Vanilla depth grows with T on A5.** Merrill et al. 2024 (Fig. 3) and
   Merrill & Sabharwal 2025 (Fig. 2, with a log fit) already show this on
   symbolic A5. Our icosahedral condition replicates it with geometric
   inputs.
2. **Delta-rule models with β up to 2 handle S4 and A5 because they embed
   in SO(3).** This is DeltaProduct §5.2 and CKDA Thm. 3, including the
   probe showing β ≈ 2 and two reflections composing into a rotation.
3. **Solvable groups do not escape the depth growth in practice.** Merrill
   et al. 2024 (§6, result 2) found transformer depth grows with n for
   A4×Z5. DeltaProduct found S4 needs *more* layers (6) than A5 (3) at
   n_h = 1. If our S4 condition shows depth growing with T, that
   reproduces prior work rather than refuting it. A "flat for solvable"
   result would be the surprising outcome.
4. **Recurrent baselines at one layer.** RNN and LSTM solving every group
   at constant depth is established (Liu §5; Merrill et al. Fig. 3).

### Not found in the literature (candidate contributions)

1. **Continuous Haar-random SO(3) and SE(3) composition as a depth-vs-T
   question for transformers and delta-rule hybrids.** The only
   continuous-SO(3) learned-composition results I found:
   - Jia 2026 uses a hand-built SO(3)-SSM, no transformer, and no sweep.
   - CKDA uses 2D rotations for audio.
   - Complexity theory covers finite groups only. SO(3) contains A5, so an
     exact solver on all inputs would also solve A5 (my inference, not
     stated in any opened paper). But Haar walks never contain icosahedral
     words: the Haar task is average-case and approximate, and no opened
     paper predicts its depth scaling.
2. **Finite groups given geometrically rather than symbolically.** Every
   prior S4/A5 experiment I could check feeds symbols, and the SO(3)
   structure appears only as an explanation of the internal mechanism.
   Feeding rotation matrices removes the lookup-table route and makes the
   finite and continuous conditions share one input interface. That
   shared interface is the novel comparison.
3. **SE(3) semidirect product with translations, in these model families.**
   Not found apart from Jia's SE(3)-SSM, which runs on symbolic A5 and
   Dyck tasks.
4. **Causal vs bidirectional.** I found no paper comparing masked and
   unmasked (or two-direction scan) models on group word problems.
   - ENTP compares encoder-only and decoder-only models on other tasks.
   - Theory note (my reading, not stated explicitly in the opened pages):
     the TC⁰ simulation does not depend on the mask. A bidirectional
     transformer is therefore still TC⁰, and the prediction for vanilla is
     "no change in the depth scaling class".
   - For the delta-rule layer, direction does matter in our own data. The
     right-to-left head carried the SE(2) rotation (HYPOTHESIS.md,
     2026-10-15 amendment).

---

## 4. Recommendations

### 4.1 Make the hybrid's mechanism actually available (most important)

- **One increment token per step gives Gated DeltaNet one reflection per
  step.**
  - The orthogonal part of α(I − βkkᵀ) with α ∈ (0,1) is a reflection
    (β = 2) or a contraction. It is never a rotation with determinant +1
    (DeltaProduct §4 and Fig. 3; CKDA §3; hybrid4.md §3.3).
  - CKDA Table 1 puts GDN at about 4 layers for S4/A5, and DeltaProduct
    measured 3–6 layers at n_h = 1.
  - So with one token per step, hybrid4 at s size (2 GDN layers) is not
    *expected* to be flat in T.
- **Pick one of these explicitly** (this is an open design decision, not
  filled in here):
  - (a) Emit ≥ 2 tokens per step. For example, a rotation token followed
    by a filler or translation token. This mirrors Grazzi's "4 tokens per
    element" for S5 and hybrid4's SE(2) setup.
  - (b) Add a Gated DeltaProduct n_h = 2 arm. fla-core 0.5.2, which is
    installed, ships `fla/ops/gated_delta_product`.
  - (c) Keep one token per step and *pre-register* that hybrid4 is
    expected to need multiple GDN layers.
- **Report the mechanism check.** Reuse `tools/transition_probe.py`:
  per-step transition eigen-angles should match the step's rotation
  angle.

### 4.2 Groups and step sampling

- **Order-matched controls inside SO(3).** O (24) vs I (60) confounds
  group size with solvability. The finite subgroups of SO(3) are cyclic,
  dihedral, T ≅ A4, O ≅ S4 and I ≅ A5. Two order-60 controls exist, which
  recreates the Merrill triad (Z60 / A4×Z5 / A5) geometrically:
  - **C60**: rotations about one axis by multiples of 6°, abelian;
  - **D30**: dihedral of order 60, solvable.
  - Keep O and T too.
- **Step sampling.** Prior work samples uniform group elements per step
  (Merrill et al.; Liu). Do the same for the finite groups and Haar for
  SO(3). State whether the identity is allowed.
- **Fixed frame vs random conjugation per episode.** A fixed frame lets a
  model memorise 24 or 60 input matrices as symbols. Randomly conjugating
  the whole group per episode forces geometric composition, but changes
  the task. CKDA Thm. 17 shows the axis orientation matters for some
  delta-rule variants. This is an open design decision; flag it in the
  pre-registration.

### 4.3 Encoding

- **Inputs.** Prefer the 3×3 rotation matrix, as 9 numbers. It is
  continuous and sign-unambiguous.
- **If quaternions are used:**
  - Randomise the input sign per token, so the model cannot exploit a
    canonical hemisphere. Canonicalising w ≥ 0 is discontinuous at
    w = 0 (Zhou et al.: every representation in ≤ 4D is discontinuous).
  - Unit-quaternion left-multiplication is a 4×4 orthogonal *linear* map.
    That makes the 4D linear-RNN construction trivial (CKDA Prop. 18 uses
    the same idea).
  - The encoding therefore changes which architecture is favoured. Keep it
    fixed across arms, or make it an explicit factor.
- **Outputs.**
  - Predict a 9D matrix projected by SVD (Levinson) or a 6D representation
    (Zhou).
  - If the output is a quaternion, use a sign-invariant loss such as
    1 − ⟨q, q̂⟩², because the target's sign is arbitrary.
- **SE(3) translations.** The composed displacement grows like √T for
  random steps and like T for biased walks. Normalise:
  - report error relative to the true displacement norm, or
  - use unit-norm steps and bucket by span.
  Otherwise the loss scale changes with T and confounds the sweep.

### 4.4 Lengths and depth budget

- T ∈ {16, 32, 64, 128, 256} is in range of prior work: T = 100 (Liu),
  train 128 / test 512 (DeltaProduct), up to 1024 (Merrill & Sabharwal).
- Use a length curriculum, as Merrill & Sabharwal did.
- **Depth budget.** The fit d ≈ 4.8·log₂T − 15.8 gives about 3 layers at
  T = 16 and about 23 at T = 256 for A5, at width 512. A vanilla depth
  sweep that stops near 8 layers will show only "fails" at large T.
  Either extend depth to about 24, or report n* (largest T solved) per
  depth, as they did.
- Report the slope of depth against log₂T, so the result can be compared
  directly with their 4.8.
- Liu et al. found T > 100 "prohibitively unstable". Budget extra seeds
  for the large-T runs.

### 4.5 Metrics

- **Per query:** geodesic angle error in degrees, bucketed by span t − s.
  For SE(3), also translation error, normalised as above.
- **Threshold accuracy:** share of queries under a fixed angle, the same
  across conditions.
  - The smallest non-identity rotation is 90° in O and 72° in I, so snap
    radii are 45° and 36°.
  - Use a common threshold well below both, for example 10°, so the Haar
    condition is scored the same way.
- **Finite groups:**
  - exact accuracy after snapping to the nearest group element;
  - **quotient accuracy**, to detect coset plateaus (Zhang & Li 2026):
    for O ≅ S4, the S3 quotient (mod the Klein four-group of face
    half-turns) and the Z2 sign; for D30, its Z2×Z2 abelianization. A5 has no
    proper quotient.
- **Depth criterion:** minimum depth reaching the threshold, best of
  seeds (Merrill et al. use > 90%, Merrill & Sabharwal ≥ 95%). Report the
  median too; Liu reports both max and median over 20 runs.
- **Evaluate in fp32.** hybrid4's kernel has 7e-3 relative error in bf16
  (hybrid4.md §3.5), and composition errors accumulate over 256 steps.

### 4.6 Baselines to include

- **LSTM or GRU at one layer.** Positive control; it solves all groups in
  Liu and Merrill et al.
- **hybrid4 with β ∈ (0,1)** (`hybrid4_beta1` exists). Negative control:
  without reflections there is no rotation (Grazzi; DeltaProduct
  Fig. 12).
- **Gated DeltaProduct n_h = 2.** Positive control for "one layer suffices
  in SO(3)" (DeltaProduct §5.2).
- **Abelian control (C60 or SO(2)).** Liu and Merrill et al. both find
  abelian groups cheap for transformers.

### 4.7 Causal vs bidirectional

- **Causal hybrid.** Define it as left-to-right GDN heads plus masked
  attention. In SE(2) the right-to-left head carried the rotation, so the
  causal arm may need its own mechanism check.
- **Query placement.** For a query relative to step s, answered at t, the
  causal model needs the query token after step t.
  - In a right-to-left scan, the span product comes from suffix products
    and needs an inverse: P(s,t) = Suf(s)·Suf(t)⁻¹.
  - Orthogonal states make the inverse a transpose, but the readout has
    to learn it.
- **Pre-register the prediction for vanilla.** Theory predicts no change
  in the scaling class from bidirectionality, though empirical constants
  may differ. No prior result exists to anchor this.

---

## 5. What I could not verify

- Per-group depth and accuracy values in figure images:
  - Liu Figs. 8 and 9;
  - Merrill et al. Fig. 3;
  - Merrill & Sabharwal Fig. 2 (only the caption fit was read);
  - RWKV-7 Fig. 8;
  - DeltaProduct Fig. 5;
  - CKDA Figs. 6 and 7.
- Merrill et al. 2024: the exact length range, maximum depth, positional
  encoding and masking.
- Merrill & Sabharwal 2025: positional encoding and masking.
- CKDA Table 1: the HTML table was garbled, so the GDN row ("4 layers for
  S4/A5") is my parse. It is consistent with DeltaProduct Thm. 1, which
  gives 4 layers at n_h = 1 for S_n.
- DeltaProduct: how group elements are tokenised (not stated in App. C.1).
- Venues: Liu et al. (ICLR 2023), Merrill et al. (ICML 2024), Grazzi et al.
  (ICLR 2025) and DeltaProduct were not confirmed on the opened pages.
- CTIN: seen only through a search result, not opened.
- GTA (Miyato et al., ICLR 2024; arXiv 2310.10375): only the abstract was
  read. It encodes known poses in attention, which is a different problem
  from composing them.
- Ben-Or & Cleve's result that products of 3×3 matrices capture formula
  evaluation would directly support continuous-SO(3) hardness, but I did
  not open it. Treat it as a lead, not a citation.
- Absence claims: there is no prior causal-vs-bidirectional comparison on
  group word problems, and no prior transformer or delta-rule depth-vs-T
  study on continuous SO(3)/SE(3). Both rest on targeted searches that
  found nothing. They are not proofs of absence.

---

## Sources (pages actually opened)

- https://arxiv.org/abs/2210.10749 and https://arxiv.org/html/2210.10749v2 (Liu et al.)
- https://arxiv.org/html/2404.08819 (Merrill, Petty, Sabharwal)
- https://arxiv.org/html/2411.12537 (Grazzi et al.)
- https://arxiv.org/html/2502.10297 (DeltaProduct)
- https://arxiv.org/abs/2405.17394 and https://arxiv.org/html/2405.17394 (Sarrof, Veitsman, Hahn)
- https://arxiv.org/abs/2207.00729 (Merrill & Sabharwal, parallelism tradeoff)
- https://arxiv.org/abs/2503.03961 and https://arxiv.org/html/2503.03961 (Merrill & Sabharwal, log depth)
- https://arxiv.org/html/2503.02854 (Li, Guo, Andreas)
- https://arxiv.org/html/2609.24797 (Complex KDA)
- https://arxiv.org/html/2610.00329 (Jia, Lie-group SSMs)
- https://arxiv.org/html/2609.29951v1 (Zhang & Li, cosets)
- https://arxiv.org/html/2503.14456 (RWKV-7)
- https://arxiv.org/html/2412.06464 (Gated DeltaNet)
- https://arxiv.org/html/2410.01600 (ENTP)
- https://arxiv.org/html/2602.18417 (Nunley, U(d) subgroups; LM only, no state tracking)
- https://arxiv.org/html/2608.03425 (complex-valued SSM; parity, mod 3 and Dyck only)
- https://arxiv.org/abs/2408.06321 (EqNIO)
- https://arxiv.org/abs/2310.10375 (GTA)
- https://arxiv.org/abs/1812.07035 (Zhou et al., rotation continuity)
- https://arxiv.org/abs/2006.14616 (Levinson et al., SVD)
- Local: docs/models/hybrid4.md (§3.2–3.5, §6) and HYPOTHESIS.md (2026-10-15 amendments)
