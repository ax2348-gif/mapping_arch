# Belief in activations, causal vs bidirectional: literature review

**Revision 2, 2026-10-16.** This revision re-judges the literature against the
program's actual criterion (section 0) and adds a second, narrower search
round (section 6).
- The round-1 classification tables (columns (a)–(e)) are kept as Appendix A.
- The round-1 verdicts and gap statement are kept, marked superseded, as
  Appendix B.
- Shared evidence on causal against bidirectional processing is in section 5.
- Published report: https://claude.ai/artifact/G3Ud86sAZtJw3wneRKg2BH ("Belief, Causality and Free Queries").

**What prompted it.** The finding that a fully causal hybrid4 fails at xs
where the bidirectional hybrid4 succeeds, including on last-step queries
where both see identical history (STATUS.md 2026-10-16; the page "Causality
and egocentric memory (provisional)"). It builds on
docs/lvsm_memory_review.md (LVSM internals, pose contracts, latent memory)
and does not repeat it.

**How claims are sourced.**
- Every claim cites a section, table, figure or code file that was opened
  (see Sources).
- † marks numbers re-opened and checked a second time.
- "Our reading" marks inference beyond what a paper states.
- Metric direction: lower is better for Acc, Comp, ATE, AbsRel and FVD;
  higher is better for PSNR, NC, SR and SPL.

## 0. The program's criterion

From "Spatial POMDP Program Notes" (https://claude.ai/artifact/9cVaS2gpjzbzVx8ch8a6gu), section 01, Thesis:

> Every form of decision-making under partial observability — control,
> planning, RL, search, world modeling — factors through a recursively
> maintained sufficient statistic of history (the belief). … The mapping
> tasks are belief-sufficiency probes; hybrid4's rotation-capable
> recurrence is the candidate operator.

and section 07, Scope language:

> Bidirectional formulations apply as-is to the **re-encoding regime**
> (frame-stacked policies, common at VLA scale); the cached/streaming
> regime needs the causal variant.

**The program.** An end-to-end system for embodied decision making.
- **Input:** the egocentric observation sequence plus the action or
  odometry history.
- **Output:** an action or plan, or observation prediction for model-based
  RL or search.
- **In between:** a learnable belief held in model activations and
  differentiable end to end.

**What this means for judging related work.**
1. **The relevant distinction is belief vs explicit geometry.** The program
   wants a belief in activations, readable by free pose queries through the
   network. It does not want explicit geometry outputs (pointmaps,
   Gaussians, voxels, point clouds) that live outside the activations and
   need a renderer or geometric pipeline before a decision can use them.
   VGGT-style models are therefore a poor conceptual fit, however good
   their geometry. The right family is LVSM, SRT, LaCT, CUT3R's state with
   its virtual-camera query, and tttLRM's fast weights.
2. **Three regimes.**
   - **Causal:** cached or streaming.
   - **Prefix-bidirectional (PB):** the history up to step t is re-encoded at
     every decision. It is deployable as is.
   - **Fully bidirectional:** offline.

   A belief is a causal *function* of the history. It need not be computed
   by a causally masked network, and PB satisfies that. Masked Trajectory
   Models (2305.02968) is the one paper found that states this separation
   between a causal policy and a causal attention mask.
3. **The downstream end.** For each model, ask whether the belief feeds
   actions, plans, or prediction for model-based RL or search, and whether
   gradients flow through it.

## 1. Bottom line (revision 2)

- **Nobody has the whole program.** We found no system that combines all of:
  - a belief in activations;
  - built from egocentric observations plus motion that the network
    composes internally;
  - readable at free poses through the network;
  - feeding a decision head end to end.

  Each piece exists separately:
  - **CUT3R:** a belief state in activations, with a read-only virtual-camera
    query that returns RGB. But it is causal, pose-free and has no decision
    head.
  - **NeuWorld and WorldCrafter:** LVSM-style, pose-queried latent scene
    memories inside video world models. No agent uses them.
  - **Gregor et al. 2019:** a belief trained end to end with RL and
    action-conditioned frame prediction, but no free-pose query.
  - **VNT-PA:** a bidirectional, pose-RoPE memory read by a query token at
    the robot's pose, feeding actions. But it uses ground-truth poses and a
    pre-collected keyframe set, and it does no rendering.
- **Evidence on direction.**
  - Same-weights tests in 3D reconstruction favour non-causal processing.
  - The world-model field openly trains bidirectional teachers and distils
    them into causal students. WorldPlay states the teacher is better "due
    to the bidirectional nature" (Table 8†: PSNR 19.31 vs 16.27).
  - No paper does our controlled comparison: matched size, trained from
    scratch, equal information, decision-relevant queries.
- **Policies.** Long-history PB encoders were tried and worked:
  - Scene Memory Transformer, up to 500 steps with RL;
  - HAMT and DUET, which were state of the art in vision-language
    navigation (VLN) with padding-only masks.

  The field then moved to causal-by-default designs for stated
  engineering reasons, without a controlled test against PB. The "causal
  bias" claim survives only in this narrow form.

## 2. Verdicts at a glance

| Hypothesis | Revision 2 verdict | Round-1 verdict (superseded) | Why it changed |
| --- | --- | --- | --- |
| H1. Recent learning-heavy approaches come very close; only data or domain limits | **Mixed, leaning contradicted on "only".** The belief-in-activations family exists, but differs structurally: no internal composition of motion, and no decision coupling | Mixed | Round 1 counted VGGT-style models as near neighbours; they now fall outside the criterion. The remaining family's gaps are structural, not data |
| H2. The real distinction is queries that break frame boundaries vs per-frame outputs | **Supported as the distinction that sorts the literature.** It is untested as a cause of decision quality | Mixed | Round 1 let "explicit pointmaps in first-frame coordinates are already a global map" weaken it. Under the criterion an explicit-geometry map is outside the activations |
| H3a. Bidirectionality is key to good global representations | **Supported, with caveats** (now including WorldPlay Table 8†) | Supported with caveats | Unchanged in direction. The relevant comparison for decisions is causal vs PB, which our last-step check measures |
| H3b. It has not been adopted in end-to-end policies because of causal bias | **Mixed.** PB long-history encoders existed and worked (SMT, HAMT, DUET, OMT). Causal-by-default replaced them for stated reasons, never on a controlled comparison | Mixed, leaning contradicted | Round 1 rebutted the claim with short-context (2–6 frame) PB policies. Round 2 found long-horizon PB was used and displaced, not refuted |

## 3. Closest existing work under the program's criterion

| Work | Belief in activations | Free-pose query through the network | Motion input | Regime, history | Feeds a decision head end to end | What is missing for the program |
| --- | --- | --- | --- | --- | --- | --- |
| **CUT3R** (2501.12387†) | Yes: 768 state tokens (paper; code default 324 plus a local memory), read and written jointly per frame (Sec. 3.1, 4) | **Yes**: a raymap virtual camera reads the state without updating it and returns a pointmap plus RGB (Sec. 3.2) | None; input frames are pose-free (G-CUT3R 2508.11379 adds poses as ray images) | Causal; trained on 4–64 frames | No. PROSPECT and VLM-3R read *frozen* decoder tokens, not the state | Internal motion composition; a decision head; a quantitative unseen-view evaluation (only Fig. 6, qualitative); drift (Sec. 5) |
| **WorldCrafter** (2609.24984†) | Yes: a memory encoder initialised from LagerNVS writes history into tokens (Sec. 3.3) | Yes, for poses on the upcoming trajectory: M = Readout(Φ(z^s, C^s), C^q) (Eq. 7); arbitrary poses not tested | Given SE(3) camera trajectory | History re-encoded every chunk (Sec. 5); generator chunk-causal (Sec. 3.2) | No (user-specified camera paths) | Decisions; internal motion composition. Ablation (Fig. 9): raw retrieved context 13.91, pose-free readout 16.49, full 18.02 PSNR |
| **NeuWorld** (2606.30045†) | Yes: 1024×64 implicit scene tokens (Sec. 4.1.1) | Yes: LVSM-style decoder, target rays attend to the tokens (Sec. 3.2) | Given SE(3) poses | Re-formed every step, not recurrent; set encoder (mask not stated); 8 memory frames | No: "richer action spaces" is future work (Sec. 5) | Persistence; decisions; internal composition |
| **Gregor et al. 2019**, Shaping Belief States (1906.09237†) | Yes: belief LSTM shared with the agent | No: frames are predicted forward under future actions; pose and map are read only by probes (Sec. 4.1) | Actions | Causal recurrent; whole episode | **Yes**: "trained jointly with the agent, sharing the belief network" (Sec. 4) | A free-pose query; a non-recurrent operator; Sec. 5 notes joint training makes the map harder to decode |
| **VNT-PA** (2609.21212†) | Yes: bidirectional self-attention over a keyframe set with pose RoPE (Sec. III-D) | Partly: a goal token at the robot's pose queries the set for actions, with no rendering | Ground-truth poses (x, y, ψ) | PB over a pre-collected set, about 67 keyframes on average, up to 838 | Yes (imitation) | Rendering or prediction; its own episode's history; odometry instead of ground-truth pose |
| **Scene Memory Transformer** (1903.03878) | Yes: full self-attention over the episode memory | Partly: the current observation queries the memory | Pose relative to the current pose, re-embedded every step (Sec. 3.4) | PB, 50–500 steps (Fig. 5a) | Yes (DQN); embedding networks pretrained and frozen (Sec. 3.3) | Rendering or free-pose query; internal composition (poses are given) |
| **Neural Volumetric Memory** (2304.01201) | Partly: per-frame feature volumes warped by a learned SE(3) encoder | Within-window frames only | Learned from images | Causal, 5 frames | Yes (behaviour cloning, end to end) | Long history; free queries; equivariance is built in by construction |
| **PROSPECT** (2603.03739†) | Partly: frozen CUT3R decoder features, with the CUT3R state carried causally | No | None | Causal, 8 short-term plus 8 keyframes | Yes (VLN). R2R SR 45.5 SigLIP only, 46.7 with CUT3R, 48.7 with latent prediction (Table II) | The state itself is never read; CUT3R is frozen; no pose query |
| **Ours** (hybrid4 / hybrid4c) | Yes | **Yes**: render at any pose relative to any step | **Odometry tokens composed in the network** | Bidirectional; the last-step evaluation measures PB; 32–64-step histories | **Not yet** (decision arm on hold) | A decision head; real images; scale |

Our reading: CUT3R is the single closest system. It differs from us on two
axes, motion input and causality, and shares our missing piece, a decision
head. The world-model family (WorldCrafter, NeuWorld) shows LVSM-style
latent memories are now being built into generators. Their own ablations
favour a pose-conditioned latent readout over raw retrieved frames.

## 4. Classification under the program's criterion

Columns:
- **Belief**: belief in activations, queryable end to end.
  - **yes**: a state or latents in activations, read by a free-pose query
    through the network.
  - **partial**: in activations, but readable only per step, by the
    decision head, by retrieval, or through frozen parts.
  - **no**: explicit geometry outputs or explicit memory, or no history
    state.
- **Regime**: C (causal), BC (block-causal), PB (prefix-bidirectional,
  re-encoded per decision), B (fully bidirectional).
- **History**: trained or typical history length.
- **Head**: whether it feeds a decision or prediction head: A (actions or
  plan), P (observation prediction for planning, RL or search), NVS (view
  synthesis only), G (geometry only), V (video generation only).

Round-1 columns (a)–(e), with full citations, are in Appendix A.

### 4.1 Novel-view synthesis and scene-latent models

| Model | Belief | Regime | History | Head | Note |
| --- | --- | --- | --- | --- | --- |
| LVSM decoder-only | partial (no separate state; input tokens re-attended per query) | B | 2–4 views trained, >10 at test | NVS | |
| LVSM encoder-decoder | yes (3072 latents) | B | 2–4; degrades beyond 8 | NVS | Decoder also refines latents (lvsm_memory_review.md) |
| SRT / UpSRT / RUST | yes (set latent, per-ray decoder) | B | 5 views | NVS | |
| RayZer | yes | B | 5–16 | NVS | Pose-free |
| LaCT (NVS) | yes (fast weights) | B (one chunk) | Up to 128 views | NVS | |
| tttLRM | partial (fast-weight state, explicit Gaussian readout) | B full; C for AR | 16–128 | NVS | AR trails full by 0.8–1.1 dB† |
| NSTM | yes | C | Up to 60 frames | NVS | |
| CausNVS | partial (KV memory, diffusion) | C | Trained F=8; 4-view window | NVS | Matched non-causal comparison† |
| NeuWorld | yes | Re-formed per step (set) | 8 memory frames | V | No agent |
| WorldCrafter | yes | Re-encoded per chunk; generator chunk-causal | Trajectory history | V | No agent |
| Long-LRM, LVT, StreamSplat | no (Gaussians) | B / B / C | 32 / 52–66 / ≤1024 | NVS | |

### 4.2 Feed-forward 3D reconstruction

| Model | Belief | Regime | History | Head | Note |
| --- | --- | --- | --- | --- | --- |
| **CUT3R** | **yes** (state plus raymap query) | C | 4–64 | G, plus RGB at the query | Closest existing belief state |
| G-CUT3R | partial (pose input, no query evaluation) | C | — | G | 7-Scenes Acc 0.098 → 0.061 with pose (Table 1) |
| TTT3R | partial (CUT3R state, gated test-time-training update) | C | Up to thousands of frames | G | |
| ZipMap | partial (TTT state, queryable at novel poses) | B; streaming variant C | — | G | |
| VGGT, π³, Fast3R | no (pointmaps) | B | To 200 / 24 / 1500 | G | Poor fit to the criterion |
| MUSt3R | no | C online; B "rendering" | ~50 keyframes | G | Same-weights evidence† |
| Spann3R, Point3R, StreamVGGT, STream3R, WinT3R | no (Point3R: explicit pointer memory) | C (WinT3R BC) | 5 to ~700 | G | Same-weights evidence (STream3R†) |

### 4.3 World models

| Model | Belief | Regime | History | Head | Note |
| --- | --- | --- | --- | --- | --- |
| NWM | no persistent belief (context frames as cross-attention keys and values) | Per-step; autoregressive rollout | 4 frames | P (CEM planning, ranking NoMaD proposals) | Relative actions (u, φ) |
| Genie / Genie 2 / Genie 3 | partial (KV, no pose query) | BC / C / C | 16 frames / ~1 min / ~1 min | V (SIMA acts in Genie 3; no RL results) | |
| Dreamer 4 | partial | BC | 192 frames | **P + A** (agents trained in imagination) | "short memory" is its main shortfall |
| DreamerV3 | partial (RSSM) | C | Recurrent | **P + A** | Program notes predict rotation-correlated failure |
| DIAMOND | no (4-frame stack) | — | 4 | **P + A** | |
| Long-context SSM world model (2505.20171) | partial (SSM state) | C (bidirectional within a frame) | Up to 800 | V | State 30.8 vs full causal context 32.6 PSNR (Table 2) |
| WorldPlay | partial (FOV-retrieved chunks) | Student BC, teacher B | Long | V | **Teacher B 19.31 vs student C 16.27 PSNR (Table 8†)** |
| RELIC, Matrix-Game 2 | partial (compressed or rolling KV) | BC, distilled from B | — | V | |
| WorldMem, Context-as-Memory, VMem, SPMem, Memory Forcing, Voyager, DeepVerse | no (explicit frames, surfels, point clouds) | C / B / — | 8–600 | V | Pose used for retrieval, not as a belief query |
| 3D-Persistent world model (2505.05495) | no (explicit DINO voxel grid) | — | — | **P + A** (MPC, ranking NoMaD) | The only spatial-memory world model in this set that feeds planning |

### 4.4 Policies and navigation

| Model | Belief | Regime | History | Head | Note |
| --- | --- | --- | --- | --- | --- |
| Decision Transformer, RT-1, Octo | partial | C | 5–50 / 6 / 2 | A | |
| ACT, π0, π0.5, RT-2 | no (no history) | B over the current frame and chunk | 1 | A | |
| Diffusion Policy | partial | Observation encoder unmasked; actions causal | 2–4 | A | |
| GNM, ViNT, NoMaD | partial | PB (ViNT: no mask in code†) | 5–6 | A | |
| **Scene Memory Transformer** | partial | **PB** | **50–500** | A (DQN) | Re-embeds memory poses every step |
| **HAMT** | partial | **PB** (padding-only masks; re-run every step, per code) | Tens of steps | A (imitation + RL) | |
| **DUET** | partial (explicit topological graph plus encoder) | **PB** | Episode | A | |
| Object Memory Transformer | partial | PB | 4–32 | A (A3C) | |
| **VNT-PA** | partial (pose query to actions) | **PB** | ~67, up to 838 keyframes | A | Causal 90.3 vs B-temporal 87.1 vs B-pose 93.3 SR† |
| E.T. (Episodic Transformer) | partial | C | ~50 steps | A | Causal for one-pass dense supervision (Sec. 4.2) |
| GTrXL, AMAGO | partial | C | Up to 10k steps | A | AMAGO: non-causal "would not make sense" |
| StreamVLN, NaVid, Uni-NaVid, MemER, CronusVLA, HAMLET | partial | C (LLM or VLM decoders) | 8+8 to long | A | |
| PROSPECT, VLM-3R | partial (frozen CUT3R tokens) | C (CUT3R) | 8+8 / clip | A (VLN) / QA | |
| Evo-0, GeoAware-VLA, Spatial Forcing | partial (frozen VGGT features, or alignment to them) | B within a step | 1 | A | |
| Gregor et al. 2019 | partial-to-yes (belief shaped by a generative model) | C | Episode | **A + P, end to end** | |
| NeRF-dy (2107.04004) | partial (latent decodable at any view; encoder frozen) | Markov | 1 | P (MPPI) | |
| Neural Volumetric Memory | partial | C | 5 | A, end to end | |
| GNFactor, NeRF-RL, SNeRL, SPA, MV-MWM | no or partial (voxels; two-stage latents) | — | 1–2 | A | GNFactor trained jointly with explicit voxels |
| Uni[MASK], MaskDP | partial | B vs C (tested) | 5–10 / 64 | A | Short horizon; favoured causal |
| **Ours** | **yes** | **B; PB measured by last-step queries** | **32–64** | **NVS only, so far** | |

## 5. Evidence on causal against bidirectional (rounds 1 and 2)

### 5.1 Same weights or same architecture (the cleanest)

| Source | Comparison | Result | Information confound? |
| --- | --- | --- | --- |
| STream3R Table 3†, App. B† | Causally trained STream3Rβ, mask removed at test time ("we directly use the causally trained model … and remove the causal mask") | 7-Scenes Acc 0.122 → 0.091, Comp 0.110 → 0.075, NC 0.746 → 0.769 (VGGT 0.087 / 0.091 / 0.787) | **Yes**: early frames now see later frames |
| MUSt3R Table 2† | Same weights and memory; causal MUSt3R-C vs "rendering" (re-predict every frame against full memory) | TUM RGB-D ATE mean 20.8 → 14.4 cm, median 11.4 → 7.4 cm | **Yes**, by construction |
| CUT3R Table 5† | Online vs "revisit" (freeze final state, re-query all frames) | 7-Scenes Acc 0.126 → 0.113, Comp 0.154 → 0.107 | **Yes**: the final state has seen everything |
| WinT3R Tables 5–6 | Frame-by-frame causal vs bidirectional windows of 4 | NRGBD Overall 0.404 → 0.258; pose AUC@30 7.39 → 15.01; removing overlap (no revision) 0.285 | Partly |
| tttLRM Table 2† | AR (causal across mini-batches) vs full; AR **fine-tuned from full** (App. B.1) with ordered sequences | DL3DV 32 views 24.31 vs 25.07 PSNR; 64 views 24.81 vs 25.95; T&T 64 views 19.80 vs 20.31 | **No**: targets rendered after all inputs, so equal information. Confounded by fine-tuning and ordered training. **Closest literature analogue to our last-step check.** |
| CausNVS Table 2† | Same architecture and training config, "trained identically albeit removing causality" | Non-causal better at or beyond training length (F=32: N=1 19.27 vs 17.44; N=4 25.12 vs 24.55; N=16 28.56 vs 28.13). Non-causal much worse at short lengths (F=2, N=1: 9.84 vs 16.56) and when run autoregressively (13.62 vs 20.40) | Mixed. **Counter-evidence on robustness** |
| CausVid Table 4† | Bidirectional DiT vs the same fine-tuned to causal (many-step); bidirectional vs causal teacher | Temporal/Frame/Text 94.6/62.7/29.6 vs 92.4/60.1/28.5; bidirectional teacher → better causal student (94.7 vs 91.9 temporal). "the many-step causal model performs substantially worse" (Sec. 5.2) | Generation, not memory |
| VNT-PA Table I† | Causal transformer (temporal RoPE, query last) vs bidirectional (temporal RoPE, VNT-TA) vs bidirectional + pose RoPE (VNT-PA) | SR 90.3 vs 87.1 vs 93.3; ≥15 m: 80.4 vs 67.7 vs 85.4 | **Counter-evidence**: bidirectional alone, without spatial position, did not beat causal. Baselines differ in more than the mask |
| Incantation App. A.8 Table 5 | Bidirectional vs causal history attention, same fine-tune | FVD 201.9 vs 245.1 | Confounded by a bidirectional pretrained prior |

### 5.2 Cross-model, and distillation as evidence

- **StreamVGGT is VGGT made causal, plus distillation from VGGT** (Sec.
  3.3). Without the teacher, 7-Scenes Acc is 0.202; with it 0.129; VGGT
  0.088 (Table 9). Gaps concentrate on multi-frame quantities: ATE TUM
  0.026 vs 0.012 (Table 6), KITTI video depth 0.173 vs 0.061 (Table 5),
  while single-frame depth is comparable (Table 4).
- **Causal models are bootstrapped from bidirectional ones** across
  domains: StreamVGGT (distilled), STream3Rβ (initialised from VGGT),
  tttLRM-AR (fine-tuned from full), ZipMap streaming (fine-tuned), CausVid
  and Self-Forcing (bidirectional teachers). Naive causal VGGT without
  fine-tuning collapses (NRGBD Acc 0.073 → 0.219, STream3R Table 8). Our
  reading: causal models seem harder to train from scratch, which matches a
  tabula-rasa failure at xs. No paper tests this directly.
- **Offline-vs-online gaps are acknowledged by streaming authors.** TTT3R:
  "a gap persists between offline and online methods" (Sec. 5); StreamSplat:
  offline methods "remain more accurate than causal processing at the
  context lengths where they still fit in memory" (App. E).
- **The gap is not universal.** Fine-tuned STream3Rβ beats VGGT on NRGBD Acc
  (0.057 vs 0.073, Table 8). Fast3R (full attention) loses to causal CUT3R
  on 7-Scenes (0.164 vs 0.126). Training data and recipe can dominate the
  mask.
- **Outside vision-for-geometry:** LLM2Vec says causal attention "prevents
  the representations from capturing information across the entire input
  sequence" (Sec. 1); after MNTP adaptation, bidirectional beats causal on
  chunking (LLaMA-2-7B 91.61 vs 88.23, Table 4), but switching the mask
  without training hurts (Sec. 2.1, 3.2). Vision Mamba: bidirectional SSM +
  conv 73.9 top-1 / 35.9 mIoU vs unidirectional 73.2 / 32.3 (Table 4, Sec.
  4.4), with the larger gain on dense prediction.

### 5.3 Why authors choose causal

Stated reasons are always about inference or rollout, never representation
quality:
- linear cost in frames (Genie Sec. 2; NWM's cross-attention design, Sec.
  3.2);
- interactive frame-by-frame generation and KV caching (Dreamer 4 Sec. 3.1,
  3.4; CausVid Sec. 1, 9.4 vs 0.6 FPS in Table 3; Self-Forcing Sec. 1, 3.4;
  MAGI-1 Sec. 1, 2.4.4);
- streaming, unbounded length (STream3R, StreamVGGT, MAGI-1);
- autoregressive generation borrowed from LLMs (Decision Transformer Sec.
  2.2), VLM-pretraining consistency and caching (π0 App. B), modularity
  (Octo Sec. III-A), or no reason (Diffusion Policy copies minGPT).


### 5.4 Round-2 additions

| Source | Comparison | Result | Reading |
| --- | --- | --- | --- |
| WorldPlay, App. C.3 Table 8† | Memory-augmented bidirectional teacher vs causal AR student vs distilled student, long-term test | PSNR 19.31 vs 16.27 vs 18.94; R_dist 0.209 vs 0.611 vs 0.332. "The teacher model exhibits better control capability and visual quality due to the bidirectional nature" | Supports H3a. Different training (teacher vs student), so not a mask-only test |
| Long-context SSM world model, Table 2 | Recurrent block-SSM state vs full-context causal attention vs 192-frame window, Memory Maze retrieval | 30.8 vs 32.6 vs 26.4 PSNR | A compressed causal state trails full causal context; no bidirectional arm |
| CUT3R, Table 5† | Online state vs revisit | 7-Scenes Comp 0.154 → 0.107 | Re-reading with the full state helps; confounded by information |
| VLN-BERT / RecBERT (2011.13922), App. C.2 | Re-encoding the instruction every step vs caching | With batch size equalised, re-encoding "even outperforms" caching, at about 2.5× training time | Re-encoding costs compute, not quality |
| Uni[MASK], Table 1, Sec. 5.3; MaskDP, App. B.3 | BERT-style vs GPT-style policies | At context 10, BERT "performs poorly across the board"; GPT more stable on 60–120-step rollouts | Against H3a, at short horizon |
| VLM-3R, Table 6 | Implicit CUT3R tokens vs fused explicit points, VSI-Bench | 60.90 vs 57.87 | Supports the criterion (activations over explicit geometry), for question answering |
| WorldCrafter, Fig. 9† | Pose-conditioned latent readout vs pose-free readout vs raw retrieved frames | 18.02 vs 16.49 vs 13.91 PSNR | Supports a pose-queried latent memory over explicit retrieval |

### 5.5 Why policy authors choose causal (round 2)

1. **Dense supervision in one forward pass.**
   - E.T.: "needs only one forward pass for a gradient descent update on a
     full episode" (Sec. 4.2).
   - AMAGO's code hardcodes `causal=True`.
   - DTQN finds that training only the last position loses badly.
   - A PB policy supervised at every step needs T passes, or a trick.
2. **Inference cost.** KV and Transformer-XL caching (POPGym, Memory Gym,
   StreamVLN: "eliminate over 99% of prefilling time"; SMT factorises
   attention because of its O(|M|²) cost).
3. **Treating a causal policy as needing a causal mask.** VPT ("policies
   must be causal"); the AMAGO docstring ("would not make sense … to be
   bi-directional or non-causal"). Masked Trajectory Models is the
   exception that separates the two.
4. **Compatibility with pretrained causal LLMs.** CronusVLA cites
   "misalignment with the VLM's pretrained causal structure"; StreamVLN.
5. **Speculation that BERT-style models handle long sequences poorly**
   (Uni[MASK]).

Item 1 is a real cost, and our own design must pay it (section 8, Q2).
Items 3 and 4 are conventions, not evidence.

## 6. Second-round findings (narrow search on the right family)

**B1. CUT3R in depth.**
- **State.** The paper describes 768 tokens of width 768 (Sec. 4). The
  code default is `state_size=324` plus `local_mem_size=256`; the released
  checkpoint's size is unverified. Image and state tokens cross-attend in
  every block, so read and write happen in one pass. The code write is
  `state = new*update_mask + old*(1-update_mask)` (src/dust3r/model.py).
- **Query.** A 6-channel raymap goes through its own encoder and reads the
  state read-only. It outputs a pointmap and RGB.
- **Query training.** The query is trained only on metric datasets: with
  probability 0.2 a frame is replaced by its raymap (App. B). Ground-truth
  intrinsics and pose are used.
- **Query evaluation.** No quantitative evaluation of unseen views exists
  (Fig. 6 is qualitative).
- **Pose or motion input.**
  - The paper uses pose-free images.
  - The code has an image-plus-raymap path, which is effectively a known
    pose. Whether it was ever trained is unverified.
  - G-CUT3R (2508.11379) adds poses explicitly through ZeroConv
    injection (Sec. 3.2). It has no raymap evaluation.
  - No variant takes per-step odometry or actions.
- **Who reads it for decisions.**
  - PROSPECT (VLN) and VLM-3R (QA) read frozen decoder or pose tokens, not
    the state.
  - Point3R and TTT3R modify it for geometry only.
  - No work trains CUT3R jointly with a decision head.
- **Causal vs re-encoding.**
  - Only the revisit (Table 5) and TTT3R's "VGGT is the offline upper
    bound" (VGGT runs out of memory beyond about 150 frames).
  - PROSPECT Table III: VGGT runs out of memory on R2R; InfiniteVGGT 43.2
    vs CUT3R 48.7 SR. Streaming wins there for cost and fit, not by a
    masked comparison.

**B2. Policies or world models reading NVS-trained latents, beyond GNFactor
and NeRF-RL.**
- **3D Neural Scene Representations for Visuomotor Control** (2107.04004):
  a latent decodable at any view by a NeRF, with a frozen encoder and MPPI
  planning. Markov, with no history.
- **SNeRL** (2301.11520): a NeRF-trained encoder feeds SAC or Dreamer; the
  encoder only.
- **MV-MWM:** a multi-view MAE feeds an RSSM. Not novel-view queries.
- **Neural Volumetric Memory:** end to end with behaviour cloning, over a
  5-frame causal window.
- **SPA, 3D-MVP:** static pretraining, no history.
- **Gregor et al. 2019:** the cleanest "belief shaped by a generative model,
  trained jointly with RL". Prediction runs forward under actions, not at
  free poses.
- **Not found:** any policy reading LVSM or SRT latents; the bounded search
  gives absence of evidence only.

**B3. World models with a persistent spatial memory.**
- **Explicit memory, retrieved by pose:** WorldMem (causal), Context-as-
  Memory (effectively bidirectional, by our inference), VMem (CUT3R
  surfels), SPMem, Memory Forcing, Voyager, DeepVerse.
- **Memory in activations, no pose query:** the long-context SSM world
  model, Mixture of Contexts, Matrix-Game 2, RELIC, Yume.
- **Memory in activations with a pose query:** only WorldCrafter
  (re-encoded per chunk) and NeuWorld (re-formed per step).
- **Feeding planning:** only the 3D-Persistent world model (explicit
  voxels, ranking NoMaD trajectories, MPC).
- **The dominant recipe** is to train a bidirectional teacher and distil it
  into a block-causal student (WorldPlay, RELIC, Matrix-Game 2,
  Self-Forcing, CausVid). BiWM argues that cached history "can never be
  revised".

**B4. Long-history PB encoders in policies.**
- **Tried and working:**
  - SMT: up to 500 steps; removing memory self-attention costs 23.7 on
    search.
  - HAMT: the code shows padding-only masks and the cross-modal
    transformer re-run every step over the cached per-step embeddings.
    Hierarchical history encoder 54.4 vs recurrent 52.2 SR (Table 2).
  - DUET, the Object Memory Transformer (to 32 steps), and VNT-PA.
- **Never done:** a matched causal vs PB mask ablation at 32 or more steps
  with a decision head. VNT-PA is the nearest, confounded by positional
  encoding and query handling.
- **Displaced, not refuted:** VLN moved to causal VLM decoders with KV
  caches, and memory RL (GTrXL, AMAGO, POPGym) was causal from the start.
- **A cheap, untried niche:** DTQN and TrMRL already re-encode their window
  every step and read only the last token. A bidirectional mask would add
  no inference cost; neither paper tested it.

## 7. The gap (revision 2)

What we found no published work doing:
1. **A belief in activations built from egocentric observations plus
   odometry that the network composes internally**, readable at free poses
   relative to any step through the network. CUT3R, WorldCrafter and
   NeuWorld are queryable but take pose-free frames or given poses. SMT and
   VNT-PA take given poses and do no rendering.
2. **A controlled causal vs PB vs bidirectional comparison of such a
   belief:** matched size, trained from scratch, at equal information
   (last-step queries). The literature's comparisons each change something
   else as well:
   - information: STream3R, MUSt3R, CUT3R;
   - training (fine-tuned or distilled): tttLRM, WorldPlay;
   - positional scheme: VNT-PA;
   - or no information control at all: CausNVS.
3. **A mechanism reading:** where the rotation forms and in which scan
   direction.

What we do **not** yet have, and must not claim:
- a decision head or belief gap (the decision arm is on hold);
- real images;
- scale beyond s or m;
- more than one seed on the causal result;
- robustness to history length (CausNVS's threat).

## 8. Open questions

1. **Can causal match bidirectional at all?**
   - Does the causal gap survive at size s?
   - Does distilling the bidirectional model into the causal one close it?
     This is the field's standard bridge, and would separate "harder to
     optimise" from "cannot represent".
2. **How should a PB model be trained?** Dense per-step supervision costs T
   passes. Is last-step-only supervision enough (DTQN suggests not), or do
   chunked PB or a causal trunk with a bidirectional top recover it? This is
   the k-sweep planned on the causality page.
3. **Does the bidirectional advantage hold at history lengths not seen in
   training?** CausNVS's non-causal model collapses there.
4. **Direction or position?** VNT-PA found the positional scheme mattered
   more than direction. Our odometry-driven rotations are the analogue of
   pose RoPE. Does a causal model given pose-relative positions close the
   gap?
5. **A CUT3R-style baseline under our contract.** A recurrent state-token
   model with odometry tokens and a read-only raymap query, trained on our
   tasks. It is the literature's closest belief state, so it is the natural
   external baseline.
6. **Does belief sufficiency carry over to decisions?** This is the belief
   gap in the program notes, section 04. A belief is a causal function of
   history; a PB network satisfies that without a causal mask.

## 9. Experiments the literature suggests

1. **Length generalisation** (CausNVS threat): evaluate both models at
   history lengths not seen in training.
2. **Distil bidirectional into causal** (StreamVGGT, CausVid, WorldPlay,
   RELIC): does a bidirectional teacher let causal hybrid4 learn the map at
   xs?
3. **Revisit or re-query** (CUT3R, MUSt3R): a causal pass followed by one
   bidirectional pass over cached states. This is the k-sweep already
   planned (causality page, plan B).
4. **Chunk-bidirectional** (WinT3R, CausVid, Dreamer 4 block-causal): the
   cheapest bridge in the literature.
5. **A CUT3R-style recurrent state-token baseline** with odometry tokens and
   raymap queries (Q5).
6. **A PB mask on a re-encoding policy** at 32 or more steps, against a
   causal mask at matched cost. This is the untested DTQN/TrMRL niche, and
   the first decision-arm experiment.

## Appendix A. Round-1 classification, columns (a)–(e)

Kept as written in round 1. Column (b) there is "arbitrary-pose or global queries vs per-frame outputs"; under the program criterion, explicit-geometry models are no longer counted as holding a global map (section 0).

Columns: (a) causal (C), bidirectional (B), block-causal (BC: bidirectional
within a frame or chunk, causal across), or prefix-bidirectional (PB:
bidirectional over the context, recomputed per decision, hence deployable);
(b) arbitrary-pose or global queries (Q) vs per-frame or per-step outputs
(F); (c) pose input; (d) memory; (e) reported causal-vs-bidirectional
comparison.

### A.1 End-to-end novel-view synthesis

| Model | (a) | (b) | (c) | (d) | (e) | Citation |
| --- | --- | --- | --- | --- | --- | --- |
| LVSM decoder-only | B: "dense full self-attention … without introducing special attention masks" | Q: Plücker rays of any target | Calibrated | 2–4 views trained, >10 zero-shot; quadratic | None | 2410.17242 Sec. 3.2, 4.5 |
| LVSM encoder-decoder | B (encoder and decoder) | Q | Calibrated | 3072 latents; degrades beyond 8 views | None | 2410.17242 Sec. 3.2, 4.5, Fig. 7 |
| SRT | B over the set of patch tokens; random reference view | Q: per-ray decoder | Calibrated, relative to view 0 | 5 views; latent set grows with inputs | None | 2111.13152 Sec. 3, 4.3 |
| UpSRT | B | Q (rays in view-0 frame) | Pose-free inputs | 5 views | None | 2111.13152 Sec. 4.4 |
| RUST | B | Q via latent pose from half the target image | Pose-free | 5 views | None (Table 1 ablates pose estimator) | 2211.14306 Sec. 2.1, 4.2 |
| RayZer | B; image-index embeddings for video order | Q, but tested only on predicted or interpolated poses | Pose-free (self-predicted) | 5–16 inputs | Ordering only: ordered 24.36 vs shuffled 20.56 PSNR | 2505.00702 Sec. 4.2, Table 4 |
| Long-LRM | B: bidirectional Mamba2 scans, global attention; order shuffled/reversed in training | Q (Gaussians) | Calibrated | 32 views, ~250K tokens | None; one-direction scan called "suboptimal for images" (qualitative) | 2410.12781 Sec. 3.2, Table 4 |
| LaCT (NVS) | B: one TTT chunk over all input tokens; per-image bidirectional window attention | Q | Calibrated | Up to 128 views, ~1M tokens | None for NVS | 2505.23884 Sec. 4.1, App. B.1 |
| tttLRM full | B (single-chunk LaCT) | Q (Gaussians) | Calibrated | 16–64 (128 tested) views; fixed fast weights | **AR vs full, Table 2** (below) | 2602.20160 Sec. 3.3, 3.5, Table 2 |
| tttLRM AR | C across mini-batches of views; ordered training sequences | Q | Calibrated | Fixed-size; "earlier inputs may be gradually forgotten" | AR fine-tuned from full | 2602.20160 Sec. 3.3, App. A, B.1 |
| Efficient-LVSM | Per-view encoding (order-free), KV cache lets views be appended | Q | Calibrated | To 64 views | None | 2602.06478 Sec. 2.3, 2.7 |
| LVT | Local symmetric attention over camera-distance neighbours | Q | Relative poses in attention | ~52–66 views, linear | None (relative pose 22.02 vs Plücker 20.59 PSNR) | 2509.25001 Table 3 |
| NSTM | C (online TTT memory) | Q | Calibrated | To 60 frames | Online vs other baselines only | 2607.15271 Sec. 3, Table 1 |
| StreamSplat | C (chunks of 4) | Q (Gaussians) | Calibrated | To 1024 views (voxel cache) | Against offline models (not matched); authors concede offline is more accurate | 2608.01659 Tables 1–3, App. E |
| CausNVS (diffusion) | C over views (KV cache) | Q, arbitrary query orderings | Calibrated, relative (CaPE) | Trained F=8; 4-view pose-aware window | **Matched causal vs non-causal, Table 2** (below) | 2509.06579 Sec. 3, 4.2, Table 2† |

### A.2 Feed-forward 3D reconstruction, bidirectional

| Model | (a) | (b) | (c) | (d) | (e) | Citation |
| --- | --- | --- | --- | --- | --- | --- |
| VGGT | B: alternating frame and global attention, no mask | F (camera, depth, pointmap per frame, in frame-1 coords); NVS fine-tune exists | Pose-free | 200 frames at 40.6 GB | None causal; Table 5: global-only 0.827 vs alternating 0.709 (ETH3D) | 2503.11651 Sec. 3.2, 4.6, Tables 5, 7, 9 |
| π³ | B, permutation-equivariant (no reference frame, no index embedding) | F | Pose-free | 2–24 views trained | Reference-frame dependence: std 0.000 vs VGGT 0.049 (ETH3D Acc) | 2507.13347 Sec. 3.1, Tables 6–7 |
| Fast3R | B: all-to-all over N views | F (local and global pointmaps) | Pose-free | 1000–1500 views | None; loses to causal CUT3R on 7-Scenes (0.164 vs 0.126 Acc, STream3R Table 3†) | 2501.13928 Sec. 3, Tables 2–4 |
| MUSt3R | Online: C ("every new image attends to previously seen images, but these are not updated"). Rendering: re-predicts every frame against full memory, "breaks the causality" | F | Pose-free | Keyframe memory to ~50 views | **Causal vs rendering, same weights** (below) | 2503.01661 Sec. 3.3, 5.1, Table 2† |
| ZipMap | B (TTT over all views); queryable at novel poses | Q | Pose-free | — | Streaming variant fine-tuned from bidirectional; cross-table only | 2603.04385 Sec. 3, App. D.5 |

### A.3 Feed-forward 3D reconstruction, causal or streaming

| Model | (a) | (b) | (c) | (d) | (e) | Citation |
| --- | --- | --- | --- | --- | --- | --- |
| CUT3R | C: 768-token recurrent state, read and updated per frame | **Q: raymap query of a virtual camera reads the state without updating it, outputs pointmap and RGB for unseen views** | Pose-free stream; query camera known | Constant state; trained 4–64 views; "may eventually drift over very long sequences" | **Revisit, Table 5** (below) | 2501.12387 Sec. 3.2, 4.4, 5, Table 5† |
| Spann3R | C: KV spatial memory (working + long-term) | F | Pose-free | ~4000 memory tokens; "our model may drift" | vs DUSt3R-GA (optimisation), not matched | 2408.16061 Sec. 4.2, Table 1 |
| StreamVGGT | C: "temporal causal attention", KV cache | F | Pose-free | Linear cache growth | **Distilled from VGGT**; vs VGGT (below) | 2507.11539 Sec. 3.1–3.3, Tables 1, 6, 9 |
| Point3R | C with explicit 3D pointer memory | F | Pose-free | Grows with explored area; OOM >700 frames (per TTT3R) | Not against bidirectional | 2507.02863 Table 1 |
| STream3R | C decoder-only, KV cache; initialised from VGGT | F | Pose-free | Trained 4–10 frames; window variant | **Mask removed at test time, same weights, Table 3** (below) | 2508.10893 Eq. 5, Table 3†, App. B†, Sec. 6 |
| TTT3R | C (CUT3R with TTT-style gated update, training-free) | F | Pose-free | Thousands of frames at 6 GB | Gap to offline VGGT persists (Table 6) | 2509.26645 Sec. 1, 5, Tables 6–7 |
| WinT3R | BC: bidirectional inside sliding windows of 4 (stride 2), causal across | F | Pose-free | Camera-token pool, +1 per frame | **Window ablation, Tables 5–6** (below) | 2509.05296 Tables 5–6 |

### A.4 Embodied world models and video generation

| Model | (a) | (b) | (c) | (d) | (e) | Citation |
| --- | --- | --- | --- | --- | --- | --- |
| NWM | Neither: context frames are only cross-attention keys/values; target self-attends; rolled out autoregressively | F: one future frame at time shift k given actions | **Relative actions (u, φ) from odometry**, sin-cos + MLP | 4 context frames; degrades after ~8 s, "loses context" OOD | None; Table 1 varies context count only | 2412.03572 Sec. 3.1–3.2, 4.1, 4.3, 5 |
| Genie | BC: ST-transformer, "temporal layer … with a causal mask" | F next frame | Latent actions | "limited to 16 frames of memory" | None; reason: linear cost in frames | 2402.15391 Sec. 2, 5 |
| Genie 2 / Genie 3 | C ("causal mask similar to … large language models"; "auto-regressively") | F | Keyboard/mouse; navigation | ~1 minute visual memory | None (blog posts, no technical report) | DeepMind blog posts |
| Dreamer 4 | BC: "masked to be causal in time, so that all tokens within a time step can attend to each other and to the past" | F | Raw keyboard/mouse | 192 frames (9.6 s); "short memory" named as main shortfall | None; reason: interactive generation, KV cache | 2509.24527 Sec. 3.1, 3.4, Tables 1–2, Conclusion |
| DreamerV3 | C (RSSM recurrence) | F | Actions | Recurrent state | None | 2301.04104 |
| DIAMOND | Frame stacking (4 frames into a U-Net) | F | Actions | "limited memory": invents new map areas after looking away | None | 2405.12399 Sec. 3.1, 6 |
| WorldMem | BC with pose-addressed memory retrieval (FOV overlap); Eq. 6 mask causal | F (sequential), memory retrieved by pose | Actions + Plücker poses (GT) | 600-frame bank | "Full Seq." baseline worse, but mask and objective differ: not a clean control | 2504.12369 Eq. 6, Tables 1, 7 |
| Context-as-Memory | Context latents attend jointly with target (mask not stated) | F, FOV-overlap retrieval | Camera [R,t] | Up to 30 context frames; PSNR rises 15.72 → 20.31 | None | 2506.03141 Sec. 3.3, Tables 1–2 |
| Po et al., long-context SSM world model | C (block SSM + local causal attention) | F | Actions | 400-frame Memory Maze retrieval | Full causal attention 32.6 vs theirs 30.8 PSNR; vs bidirectional DFoT 17.1 vs 17.8 (Minecraft), unmatched | 2505.20171 Tables 2, 4 |
| CausVid | BC student (bidirectional in chunk, causal across), distilled from bidirectional teacher | F | Text | — | **Table 4** (below) | 2412.07772 Sec. 4.1–4.2, 5.2, Table 4† |
| Self-Forcing | BC, distilled from bidirectional Wan2.1 | F | Text | Rolling KV cache | Matches Wan2.1 VBench only via bidirectional teacher | 2506.08009 Table 1, Sec. 3.4, 5 |
| DFoT | Deliberately non-causal; causality optional via noise, not mask | F | Relative poses (rays) | 862-frame rollout | No causal-vs-non-causal numbers | 2502.06764 Sec. 2, 4, App. A.6 |
| MAGI-1 | BC | F | Text/video | Linear in duration | Claims causality helps physics; no ablation | 2505.13211 Sec. 2.2.2, Table 10 |
| Incantation | Bidirectional over history; causal boundary to target | F | Game actions | — | **Bidirectional 201.9 vs causal 245.1 FVD**; confounded by a bidirectional pretrained prior | 2605.18601 Sec. 3.1, App. A.8 Table 5 |
| BiWM | Bidirectional within chunks; argues KV-cached history "can never be revised" | F | Camera/actions | — | Qualitative only (Fig. 3); quantitative study "ongoing" | 2606.10135 Sec. 1, 3.1, 5† |
| Vid2World | Causalises a bidirectional video model | F | Actions | — | Non-AR 25.8 vs AR 24.6 PSNR (RT-1); mask of non-AR variant unverified | 2505.14357 Table 1 |

### A.5 Transformer policies and navigation models

| Model | (a) | (b) | (c) | (d) | (e) | Citation |
| --- | --- | --- | --- | --- | --- | --- |
| Decision Transformer | C (GPT causal mask, "to enable autoregressive generation") | F, one action per step | None (RTG, state, action) | K = 5–50 steps | None | 2106.01345 Sec. 2.2, Table 5 |
| RT-1 | C over 6-frame history ("look ahead mask ensures causality") | F action tokens | None | 6 images × 8 tokens | None; concern is <100 ms inference | 2212.06817 Sec. 5.1; robotics_transformer/transformer_network.py |
| RT-2 | Single image; PaLI-X encoder is B over the prefix | F | None | None | None | 2307.15818 |
| ACT | B over the current observation and within the action chunk (no mask described); no history, by design to avoid "causal confusion" | F chunk (k=100) | Joint positions | None | None | 2304.13705 Sec. IV-A, IV-C |
| Diffusion Policy (transformer) | C over action tokens ("causal attention"), memory mask over obs; no reason given beyond minGPT | F chunk | Proprioception | 2–4 obs steps; longer obs horizon hurts vision DP | None | 2303.04137 Fig. 2, App. B.1, Table 8; transformer_for_diffusion.py |
| π0 | Block-causal: 3 blocks, bidirectional within; "action expert uses a full bidirectional attention mask" | F chunk (H=50) | Joint angles | Single timestep | None; reasons: VLM-pretraining consistency, KV caching | 2410.24164 Sec. IV, App. A–B |
| π0.5 | Bidirectional over image, prompt and action tokens | F chunk | State as text | Current frame | None | 2504.16054 Sec. IV-A |
| Octo | C over time ("attend causally to tokens from the same or earlier time steps"); reason: modularity | F chunk | None at pretraining | 2 frames; diminishing gains beyond 1 extra | None | 2405.12213 Sec. III-A, App. E |
| GNM | CNN over stacked 5 past frames (non-causal) | F waypoints + distance | None (odometry for labels only) | 5 frames | Context vs none: 0.92 vs 0.79 (Table V) | 2210.03370 Sec. IV-C, Table V |
| ViNT | **PB**: `nn.TransformerEncoder` called with no mask† | F waypoints + distance, goal image | None | 5 past frames + current | None | 2306.14846; vint/self_attention.py† |
| NoMaD | **PB**: TransformerEncoder with only a goal padding mask, mean-pooled | F diffusion chunk | None | context_size 3 (config) | None | 2310.07896; nomad/nomad_vint.py |
| Scene Memory Transformer | **PB**: full self-attention over the whole episode memory | Current observation queries memory | Pose relative to current frame | Whole episode | Removing memory self-attention −23.7 on search; beats LSTM | 1903.03878 Sec. 3.2.2, 3.4, 4.3 |
| VNT-PA | **PB** over unordered keyframe set, pose RoPE | Goal token at robot pose | GT pose (x, y, ψ) | ~67 keyframes avg, to 838 | **Causal vs bidirectional, Table I** (below)† | 2609.21212 Sec. III-D, IV-B, Table I† |
| Uni[MASK] | B (BERT-style) vs C (GPT) | F | None | Context 5–10 | GPT ≥ BERT for BC at longer context (Maze2D) | 2211.10869 Table 1, Sec. 5.3 |
| MaskDP | B (MAE-style); switched to causal for offline RL | Goal reaching | None | — | Beats causal Goal-GPT on goal reaching (Table 3) | 2211.12740 Sec. 3.2, Table 3 |
| NavFoM | LLM backbone (mask not stated; default causal, unverified) | F waypoints | None | Token-budgeted history | None | 2509.12129 |
| **Ours (hybrid4 / hybrid4c)** | **B** (bidirectional GDN β∈(0,2) + attention); causal variant fails at xs | **Q: render at any pose relative to any step** | **Relative odometry tokens, composed in network** | Full history recomputed in parallel | **Matched, from scratch, with last-step information control** | STATUS.md 2026-10-16 |


## Appendix B. Round-1 verdicts and gap statement (SUPERSEDED by sections 2 and 7)

Kept for the record. One-line reasons for each change are in the section 2 table. Section references inside this appendix (sections 2 and 3) point to Appendix A and section 5.

### B.1 Round-1 verdicts (superseded)

#### H1. Recent learning-heavy approaches already come very close to our mapping models, with only data-coverage or domain limitations — **mixed**

- **For.**
  - **CUT3R** is the closest single model. A learned state built from a
    pose-free image stream is queried by a virtual camera, without updating
    the state, and returns pointmap and RGB for unseen views (Sec. 3.2). That
    is our "query the map at an arbitrary pose" contract in a different
    domain (static real scenes, metric data), at far larger scale.
  - **tttLRM / LaCT / LVSM** render at arbitrary calibrated poses from long
    histories (to 128 views), parallel and bidirectional, as hybrid4 does.
  - **NWM** takes exactly our kind of action, relative (u, φ) from odometry
    (Sec. 4.1).
  - **SMT and VNT-PA** encode an episode-long, pose-indexed memory
    bidirectionally for navigation.
- **Against "only data or domain".**
  - **The pose contract differs.** LVSM-family models need calibrated poses
    in a shared frame; CUT3R and RayZer infer motion from pixels; NWM embeds
    actions as features for one-step prediction with no map query; SMT and
    VNT-PA take ground-truth poses. **None composes per-step odometry inside
    the network into a state that is then queried at an arbitrary relative
    pose** (lvsm_memory_review.md Sec. 4 found the same).
  - **CUT3R is causal and reports drift** (Sec. 5), and its query camera is
    in the state's frame. It is not a test of composing motion.
  - The state-update algebra (exact rotations from β∈(0,2)) has no analogue
    in these systems.
- So the systems are close in **capability**, but different in **where
  composition happens**, which is the variable we study. The gap is not
  only data.

#### H2. The real distinction is free queries that break frame boundaries vs per-frame outputs — **mixed**

- **For.** It is a real and clean taxonomic axis. Per-frame models (VGGT,
  π³, StreamVGGT, Spann3R, Point3R, STream3R, all policies and world models)
  output one prediction per input step. Free-query models (LVSM, SRT, LaCT,
  tttLRM, CUT3R raymaps, CausNVS) must answer about a pose no input frame
  holds. CUT3R's raymap query and VGGT's NVS fine-tune (Sec. 4.6, Table 7)
  show the free query can be added to a per-frame model.
- **Against "the real distinction".**
  - **Per-frame outputs can already force global structure.** VGGT,
    StreamVGGT and Fast3R predict pointmaps in the first frame's
    coordinates, which is a global map read out frame by frame.
  - **The causality effect appears in per-frame models too.** The
    same-weights gains in STream3R, MUSt3R and WinT3R are on per-frame
    outputs, so causality matters independently of query type.
  - **No paper varies query type as an experimental variable.** We found
    nothing comparing, at matched model, a free-query objective against
    per-frame outputs for the quality of the global representation.
- Our reading: free queries are a stronger and less shortcuttable
  supervision of the map than per-frame outputs (a query at a pose no frame
  holds cannot be answered by copying one frame). That is a hypothesis we
  could test, not a literature result.

#### H3. Bidirectionality is key to good global representations, and end-to-end policies avoid it from a bias toward causal models — **part 1 supported with caveats; part 2 mixed, leaning contradicted**

- **Part 1, bidirectionality helps global representations: supported.**
  - Every same-weights comparison in 3D reconstruction favours non-causal
    processing (STream3R, MUSt3R, CUT3R revisit, WinT3R).
  - tttLRM's AR model trails its full model by 0.8–1.1 dB at equal
    information.
  - CausVid's causal fine-tune trails its bidirectional parent.
  - Streaming authors concede the offline gap (TTT3R, StreamSplat).
  - Strong causal models are bootstrapped from bidirectional ones.
  - Outside geometry: LLM2Vec and Vision Mamba point the same way.
- **Caveats on part 1.**
  1. Most same-weights wins are **confounded by information**: earlier
     frames re-predicted with later context. Our last-step control removes
     this; tttLRM is the only literature result that does, and it is
     confounded by fine-tuning.
  2. **CausNVS** shows a matched non-causal model can be brittle off its
     training length and useless for rollout.
  3. **VNT-PA** shows bidirectional attention without spatial positions
     loses to causal attention; positional scheme mattered more than
     direction.
  4. Uni[MASK] finds GPT ≥ BERT for behaviour cloning at longer context.
  5. No matched from-scratch study exists.
- **Part 2, policies avoid it from bias: mixed, leaning contradicted.**
  - Navigation models are already prefix-bidirectional: ViNT (code, no
    mask†), NoMaD, GNM, SMT, VNT-PA. π0 and π0.5 are bidirectional over
    their prefix and action chunk, ACT within its chunk.
  - Where causal masks are used, authors state engineering reasons:
    autoregressive generation, KV cache, streaming, modularity (Sec. 3.3).
    That is a deliberate trade, not a bias.
  - What does hold: **those bidirectional policies use 2–6 frames of
    history**, so the long-history regime where global structure matters
    is rarely tested. Long-history policies built on LLM backbones
    (NavFoM, likely causal; unverified) inherit the causal mask. No policy
    paper justifies causality by representation quality.
  - World models are causal for a real reason (rollout), and the open
    problem is how to get bidirectional quality there. BiWM argues exactly
    this, but qualitatively; distillation from bidirectional teachers
    (CausVid, Self-Forcing) is the field's current bridge.

### B.2 Round-1 gap statement (superseded)

- **A matched, from-scratch causal-vs-bidirectional comparison with an
  information control.** Same shapes, same recipe, a test that no output
  depends on a later token, and a last-step evaluation where both see
  identical history. We found no published analogue: STream3R, MUSt3R and
  CUT3R change information along with the mask, tttLRM fine-tunes AR from
  full, and CausNVS does not separate information from memory quality.
- **In the free-query regime, with internally composed relative motion.**
  The query is at an arbitrary pose relative to any step, and the network
  must compose odometry to answer it. No surveyed system combines these.
- **With a mechanism reading.** The exact rotation formed in the
  right-to-left delta-rule head of the bidirectional model. The pending
  forward-head probe can say whether a causal model forms it at all.
- **A deployable framing the literature already half-uses.**
  Prefix-bidirectional recompute is how ViNT, NoMaD and SMT run. Our
  last-step result is a direct measure of causal vs prefix-bidirectional,
  which is the comparison that matters for a policy.
- **Limits, stated plainly.** One size (xs), one seed, discrete SE(2), a
  synthetic textured world, 100k steps. The bidirectional vanilla4 xs does
  not solve either, and the vanilla4_causal control is pending. CausNVS's
  length brittleness and VNT-PA's result are live threats: the bidirectional
  advantage must survive held-out history lengths, and the comparison
  should hold position encoding fixed. Until the s run is in, "causal
  cannot" may be "causal costs capacity or data".


## Sources (pages opened)

### Round 1

arXiv HTML unless noted. † = re-opened and checked a second time.

**End-to-end NVS**
- Jin et al. LVSM. 2410.17242. Sec. 3.2, 4.4, 4.5, App. A.1–A.2.
- Sajjadi et al. SRT / UpSRT. 2111.13152v3. Sec. 3, 4.3–4.4, App. A.2–A.3.
- Sajjadi et al. RUST. 2211.14306v2. Sec. 2.1, 4.1.1, 4.2, Table 1.
- Jiang et al. RayZer. 2505.00702. Sec. 4.1–4.2, 5.1–5.2, Tables 4, 8.
- Ziwen et al. Long-LRM. 2410.12781. Sec. 3.1–3.3, Tables 4, 5, 11, App. E.
- Zhang et al. LaCT. 2505.23884. Sec. 4.1, 5.1, 5.3, Table 2, App. B.1, C.1, D.
- Wang et al. tttLRM. 2602.20160†. Sec. 3.1–3.5, Tables 2, 4, 6, App. A, B.1.
- Efficient-LVSM. 2602.06478. Sec. 2.3–2.7, Tables 6–8.
- Imtiaz et al. LVT. 2509.25001. Sec. 3.2–3.3, Tables 2, 3, 7.
- SVSM, Scaling View Synthesis Transformers. 2602.21341v1. Sec. 1–3.
- FSM, Fast Spatial Memory. 2604.07350v1. Sec. 2.2–2.3, 4.2, Table 1.
- NSTM. 2607.15271v1. Sec. 3, 4.1, 4.4, Table 1.
- StreamSplat. 2608.01659v1. Sec. 3.1, 4.1, Tables 1–3, App. E.
- CausNVS. 2509.06579†. Sec. 3, 4.2, Table 2.

**3D reconstruction**
- Wang et al. VGGT. 2503.11651. Sec. 3.2, 4.6, Tables 5, 7, 9.
- π³. 2507.13347. Sec. 3.1, Tables 1, 2, 6, 7.
- Yang et al. Fast3R. 2501.13928. Sec. 3, Tables 2–4.
- Cabon et al. MUSt3R. 2503.01661†. Sec. 3.3, 5.1, Tables 1, 2, 4, 5, App. A.3.
- Wang et al. CUT3R. 2501.12387†. Sec. 3.2, 4.4, 5, Tables 3–5.
- Wang & Agapito. Spann3R. 2408.16061. Sec. 4.2, Tables 1–2.
- StreamVGGT. 2507.11539. Sec. 3.1–3.3, Tables 1, 2, 4–9, 14.
- Point3R. 2507.02863. Tables 1, 4, 7.
- Lan et al. STream3R. 2508.10893†. Eq. 5, Tables 2, 3, 8, App. B, Sec. 6.
- TTT3R. 2509.26645. Sec. 1, 5, Tables 6–7, App. C.3.
- WinT3R. 2509.05296. Tables 5–6.
- ZipMap. 2603.04385. Sec. 3.2–3.5, App. B.3, D.5, Tables 9–15.

**World models and video**
- Bar et al. NWM. 2412.03572. Sec. 3.1–3.3, 4.1, 4.3, 5, 6, Table 1.
- Bruce et al. Genie. 2402.15391. Sec. 2, 5, Table 3.
- Genie 2 blog: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
- Genie 3 blog: https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/
- Hafner et al. Dreamer 4. 2509.24527 (HTML and PDF). Sec. 3.1, 3.4, 4.1, Tables 1–2.
- Hafner et al. DreamerV3. 2301.04104.
- Alonso et al. DIAMOND. 2405.12399. Sec. 3.1, 6, 8, App. M.
- WorldMem. 2504.12369. Eq. 6, Tables 1, 7.
- Context-as-Memory. 2506.03141. Sec. 3.3, Tables 1–2.
- Video World Models with Long-term Spatial Memory. 2506.05284. Table 1.
- Po et al. Long-Context State-Space Video World Models. 2505.20171. Sec. 4.1, Tables 2–4.
- Yin et al. CausVid. 2412.07772†. Sec. 1, 4.1–4.2, 5.2, Tables 3–4.
- Huang et al. Self-Forcing. 2506.08009. Sec. 1, 3.4, 5, Tables 1–2.
- Chen et al. Diffusion Forcing. 2407.01392. Sec. 3.2, 4.1, 5, App. B.1.
- Song et al. DFoT / History-Guided Video Diffusion. 2502.06764. Sec. 2, 4, 6.5, App. A.6, C.2.
- MAGI-1. 2505.13211. Sec. 1, 2.2.2, 2.4.4, Table 10.
- Incantation. 2605.18601. Sec. 3.1, App. A.8 Table 5.
- BiWM. 2606.10135†. Sec. 1, 2, 3.1, 5, Fig. 3, 7.
- Vid2World. 2505.14357. Table 1.

**Policies and navigation**
- Chen et al. Decision Transformer. 2106.01345. Sec. 2.2, Table 5.
- Brohan et al. RT-1. 2212.06817 (abs, v2 HTML); github.com/google-research/robotics_transformer transformer_network.py.
- Brohan et al. RT-2. 2307.15818.
- Zhao et al. ACT. 2304.13705. Sec. IV-A, IV-C, Fig. 7.
- Chi et al. Diffusion Policy. 2303.04137. Fig. 2, App. B.1, Table 8; github.com/real-stanford/diffusion_policy transformer_for_diffusion.py and train_diffusion_transformer_hybrid_workspace.yaml.
- Black et al. π0. 2410.24164. Sec. IV, App. A–B.
- π0.5. 2504.16054. Sec. IV-A, VI.
- Octo. 2405.12213. Sec. III-A, App. E.
- Shah et al. GNM. 2210.03370. Sec. IV-C, Table V.
- Shah et al. ViNT. 2306.14846; github.com/robodhruv/visualnav-transformer vint/vint.py, vint/self_attention.py†.
- Sridhar et al. NoMaD. 2310.07896; nomad/nomad_vint.py, config/nomad.yaml.
- Fang et al. Scene Memory Transformer. 1903.03878. Sec. 3.2.2, 3.4, 4.3, Tables 1–3.
- VNT-PA. 2609.21212† (abs and HTML). Sec. III-D, IV-B, Table I.
- Torne et al. Past-Token Prediction. 2505.09561v2. Sec. 3, App. B.3.1, Table 4.
- de Haan et al. Causal Confusion in Imitation Learning. 1905.11979 (abs).
- Uni[MASK]. 2211.10869 (abs and HTML). Sec. 3.2, 5.3, Table 1.
- Liu et al. MaskDP. 2211.12740. Sec. 3.2, Table 3.
- MemoryVLA. 2508.19236.
- NavFoM. 2509.12129v1.
- ReMEmbR. 2409.13682.
- ScaleDP. 2409.14411. Table IV.
- SmolVLA. 2506.01844. Table 7.

**General representation evidence**
- BehnamGhader et al. LLM2Vec. 2404.05961. Sec. 1, 2.1, 3.1–3.2, Tables 4–5.
- Zhu et al. Vision Mamba. 2401.09417. Sec. 1, 4.4, Table 4.


### Round 2 (program-criterion search)

- Program Notes: https://claude.ai/artifact/9cVaS2gpjzbzVx8ch8a6gu (sections 01, 04, 07).
- Causality page: https://claude.ai/artifact/WP6anaPetTzFXYfMefbkeG (Claude Docs).

**CUT3R family and 3D-feature readers**
- CUT3R 2501.12387† (HTML; github.com/CUT3R/CUT3R src/dust3r/model.py). Sec. 3.1–3.4, 4, 4.4, 5, App. B, Table 5.
- G-CUT3R 2508.11379v2. Sec. 3.2, Table 1.
- TTT3R 2509.26645. Eqs. 7–8, Sec. 4.1, Figs. 7–9 (first 100k characters).
- Point3R 2507.02863v2. Eqs. 8–10, Table 5.
- VLM-3R 2505.20279. Sec. 4.1, Table 6.
- PROSPECT 2603.03739†. Sec. III-C, IV-A, V-B, Tables II–III.
- Spatial-MLLM 2505.23747.
- Spatial Forcing 2510.12276v2. Tables 1–2.
- Evo-0 2507.00416v3.
- GeoAware-VLA 2509.14117v3. Table I.
- StreamVLN 2507.05240v1. Fig. 5, Table 4.
- MindJourney 2507.12508.

**NVS-trained latents for control**
- Li et al., 3D Neural Scene Representations for Visuomotor Control 2107.04004v2. Sec. 4.2.
- SNeRL 2301.11520v3. Sec. 5.3.
- MV-MWM 2302.02408. Table 1.
- SPA 2410.08208.
- 3D-MVP 2406.18158.
- Neural Volumetric Memory 2304.01201. Sec. 3.3.
- Gregor et al., Shaping Belief States 1906.09237v2†. Sec. 4, 4.1, 5, App. B, D, Table 1.
- NWM 2412.03572v2. Table 2.
- Long-context SSM world model 2505.20171. Fig. 3, Tables 1, 2, 4.
- NeuWorld, Walking in the Implicit 2606.30045†. Sec. 3.1–3.4, 4.1, 5.

**Spatial-memory world models**
- WorldMem 2504.12369. Sec. 3.2–3.3, Alg. 1, Eq. 6.
- Context-as-Memory 2506.03141 (HTML, PDF). Sec. 3.1–3.3.
- VMem 2506.18903. Table 4.
- SPMem 2506.05284.
- Memory Forcing 2510.03198.
- Geometry Forcing 2507.07982. Sec. 4.3.
- 3D-Persistent Embodied World Models 2505.05495. Sec. 4.3, Tables 1–2.
- Mixture of Contexts 2508.21058 (PDF). Fig. 2.
- WorldPlay 2512.14614†. Sec. 3.3–3.4, Table 2, App. C.3 Table 8.
- Voyager 2506.04225.
- Matrix-Game 2.0 2508.13009.
- Yume 2507.17744.
- RELIC 2512.04040.
- WorldCrafter 2609.24984†. Sec. 3.2–3.3, Eq. 7, 4.5 (Fig. 9), 5.
- RAYNOVA 2602.20685. Table 5.
- DeepVerse 2506.01103.
- PE-Field 2510.20385.
- Aether 2503.18945.
- Genie 3 blog.

**Long-history policies**
- Scene Memory Transformer 1903.03878 (PDF). Sec. 3.3, 3.4, 4.3, Fig. 5a.
- E.T. 2105.06453 (abs, PDF). Sec. 3.2, 4.2, Table 2.
- HAMT 2110.13309 (abs, PDF); github.com/cshizhe/VLN-HAMT finetune_src/models/vilmodel_cmt.py, r2r/agent_cmt.py. Table 2.
- DUET 2202.11742; VLN-DUET map_nav_src/models/vilmodel.py, ops.py.
- Object Memory Transformer 2203.14708.
- Masked Trajectory Models 2305.02968.
- VLN-BERT / RecBERT 2011.13922 (PDF). Table 4, App. C.2.
- GTrXL 1910.06764.
- AMAGO (code: transformer.py, traj_encoders.py).
- DTQN (code: dtqn.py).
- POPGym 2303.01859.
- Memory Gym 2309.17207.
- Ni et al. 2307.03864.
- Cherepanov 2412.06531.
- CoBERL 2107.05431.
- Uni[MASK] 2211.10869. Table 1, Sec. 5.3.
- MaskDP 2211.12740. App. B.3.
- Past-Token Prediction 2505.09561v2.
- Diffusion Policy transformer_for_diffusion.py.
- VNT-PA 2609.21212† (PDF). Sec. III-D, IV-B, Table I.
- NaVid 2402.15852.
- Further VLN and VLA PDFs opened by the round-2 agent: 2412.06224, 2305.16986, 2502.14254, 2510.20328, 2501.18564, 2510.00695, 2506.19816, 2508.19958, 2206.06614, 2411.11188, 2306.09459, 2106.02039, 2206.11795, 2206.01078, 2101.05181, 2402.19161, 2402.06402, 2603.24581, 2310.09971.

## Not verified

- **Hydra** (2407.09941, bidirectional quasiseparable SSMs): the HTML and
  PDF could not be extracted; not used.
- **Wen et al., Fighting Copycat Agents** (2010.14876), FlexiBiT,
  Mem2Ego, π0-FAST: search results only; not used as evidence.
- **Masks not confirmed:** NavFoM (assumed causal from its LLM backbone);
  the Past-Token-Prediction policy; Context-as-Memory; WorldMem's "Full
  Seq." baseline; Vid2World's non-AR variant; whether tttLRM's views inside
  one AR mini-batch attend bidirectionally; Genie's spatial layer (implied
  bidirectional).
- **Inconsistent numbers:**
  - MUSt3R Table 1 came out as 7.1 vs 5.5 cm and as 11.1 vs 8.4 cm in two
    extractions, so only Table 2 is cited.
  - Inconsistent within the papers themselves: StreamVGGT ETH3D (Table 2
    vs Table 14) and Point3R (Table 1 vs Table 7).
- **Plot-only results, numbers not extracted:** LaCT's autoregressive video
  results (Fig. 6); TTT3R ATE vs sequence length (Figs. 7–9); NWM Table 1
  (garbled extraction; cited only as "varies context count").
- **Extraction method:** numbers came through a summarising fetch tool.
  Those marked † were re-opened and matched; the rest should be
  spot-checked against the PDFs before publication.
- **Absence is from a bounded search:** "no matched from-scratch
  causal-vs-bidirectional study with an information control" means none
  was found, not that none exists. BiWM's promised quantitative study, and
  RAYNOVA, VMem and WorldCrafter (seen only in search results), were not
  read.

**Round 2:**
- **CUT3R:** which state size (768 in the paper, 324 in the code default)
  the released checkpoint uses, and whether the image-plus-raymap path was
  ever trained.
- **Masks not stated or not checked in code:** VMem/SEVA, SPMem, Memory
  Forcing, Geometry Forcing, 3D-Persistent, Voyager, DeepVerse, Aether,
  WorldCrafter's encoder and readout, NeuWorld's set encoder, SMT (paper
  only), PTP, NaVid, StreamVLN, Uni[MASK], MaskDP, MTM, POPGym, Memory Gym.
- **Abstract or search only:** DNAct, ManiGaussian, Video Prediction Policy,
  SpawnNet, GeoVLA, SpatialVLA, Ross3D, LagerNVS, and the 2026
  robot-memory papers (2610.14406, 2610.13262, 2509.20297, 2610.08978).
- **Inconsistent within the paper:** VLM-3R's text (50.09) vs its Table 6
  (59.09) for the no-view-tokens row.
- **HAMT:** maximum history length not verified.
- **VNT-PA:** whether its context is encoded once per episode is not
  verified.
- **Absence claims** ("no system has all of …", "no controlled long-horizon
  PB-vs-causal mask ablation") rest on bounded searches.
