# Causal vs bidirectional context, and frame-bound vs free queries: literature review

Written 2026-10-16, after the provisional finding that a fully causal hybrid4
fails at xs where the bidirectional hybrid4 succeeds, including on last-step
queries where both see identical history (STATUS.md, 2026-10-16;
"Causality and egocentric memory (provisional)" page). Builds on
docs/lvsm_memory_review.md (LVSM internals, pose contracts, latent memory),
which is not repeated here.

Every claim cites a section, table, figure or code file that was opened (see
Sources). Numbers marked † were re-opened and checked a second time. "Our
reading" marks inference beyond what a paper states. Lower is better for
Acc, Comp, ATE, AbsRel, FVD; higher for PSNR, NC, SR, SPL.

## 1. Bottom line

- **Every ingredient of our setup exists somewhere; the combination does not.**
  - Rendering at arbitrary query poses from a multi-view history: LVSM, SRT,
    LaCT, tttLRM, CausNVS, NSTM. All take calibrated poses.
  - A learned global state queried at an arbitrary camera, rendering RGB for
    unseen views, from pose-free input: CUT3R (causal). **The closest
    existing model to ours.**
  - Relative actions or odometry as input: NWM, WorldMem, VNT-PA, SMT. None
    composes odometry inside the network into a queryable map.
- **On causality the 3D-reconstruction and video literature points the same
  way as our finding, with an important weakness.** Every same-weights
  comparison favours non-causal processing (STream3R, MUSt3R, CUT3R revisit,
  WinT3R). But in all of them the gain comes, at least partly, from earlier
  frames seeing later frames: the information confound our last-step check
  removes. Only tttLRM (AR vs full, evaluated after all inputs) and
  StreamSplat's authors' statement speak to a memory-quality gap at equal
  information, and both are confounded by training (fine-tuning, different
  models).
- **No paper we found trains the same architecture from scratch, causal and
  bidirectional, at matched size, and controls for information.** CausNVS
  comes closest (same architecture and training, mask removed), and it is
  also the strongest counter-evidence: its non-causal model wins near the
  training length but collapses at other lengths and under autoregressive
  use.
- **Policies are not uniformly causal.** ViNT, NoMaD, GNM, SMT, VNT-PA and
  π0's prefix are bidirectional over their context. Causal masks appear where
  there is an autoregressive or KV-cache reason (Decision Transformer, RT-1,
  Octo, world models). So "bias" is the wrong word for most cases; the
  stated reasons are inference cost, streaming and rollout. But **no paper
  justifies a causal mask by representation quality**, and only one
  (VNT-PA) ablates it over a long history.

## 2. Classification table

Columns: (a) causal (C), bidirectional (B), block-causal (BC: bidirectional
within a frame or chunk, causal across), or prefix-bidirectional (PB:
bidirectional over the context, recomputed per decision, hence deployable);
(b) arbitrary-pose or global queries (Q) vs per-frame or per-step outputs
(F); (c) pose input; (d) memory; (e) reported causal-vs-bidirectional
comparison.

### 2.1 End-to-end novel-view synthesis

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

### 2.2 Feed-forward 3D reconstruction, bidirectional

| Model | (a) | (b) | (c) | (d) | (e) | Citation |
| --- | --- | --- | --- | --- | --- | --- |
| VGGT | B: alternating frame and global attention, no mask | F (camera, depth, pointmap per frame, in frame-1 coords); NVS fine-tune exists | Pose-free | 200 frames at 40.6 GB | None causal; Table 5: global-only 0.827 vs alternating 0.709 (ETH3D) | 2503.11651 Sec. 3.2, 4.6, Tables 5, 7, 9 |
| π³ | B, permutation-equivariant (no reference frame, no index embedding) | F | Pose-free | 2–24 views trained | Reference-frame dependence: std 0.000 vs VGGT 0.049 (ETH3D Acc) | 2507.13347 Sec. 3.1, Tables 6–7 |
| Fast3R | B: all-to-all over N views | F (local and global pointmaps) | Pose-free | 1000–1500 views | None; loses to causal CUT3R on 7-Scenes (0.164 vs 0.126 Acc, STream3R Table 3†) | 2501.13928 Sec. 3, Tables 2–4 |
| MUSt3R | Online: C ("every new image attends to previously seen images, but these are not updated"). Rendering: re-predicts every frame against full memory, "breaks the causality" | F | Pose-free | Keyframe memory to ~50 views | **Causal vs rendering, same weights** (below) | 2503.01661 Sec. 3.3, 5.1, Table 2† |
| ZipMap | B (TTT over all views); queryable at novel poses | Q | Pose-free | — | Streaming variant fine-tuned from bidirectional; cross-table only | 2603.04385 Sec. 3, App. D.5 |

### 2.3 Feed-forward 3D reconstruction, causal or streaming

| Model | (a) | (b) | (c) | (d) | (e) | Citation |
| --- | --- | --- | --- | --- | --- | --- |
| CUT3R | C: 768-token recurrent state, read and updated per frame | **Q: raymap query of a virtual camera reads the state without updating it, outputs pointmap and RGB for unseen views** | Pose-free stream; query camera known | Constant state; trained 4–64 views; "may eventually drift over very long sequences" | **Revisit, Table 5** (below) | 2501.12387 Sec. 3.2, 4.4, 5, Table 5† |
| Spann3R | C: KV spatial memory (working + long-term) | F | Pose-free | ~4000 memory tokens; "our model may drift" | vs DUSt3R-GA (optimisation), not matched | 2408.16061 Sec. 4.2, Table 1 |
| StreamVGGT | C: "temporal causal attention", KV cache | F | Pose-free | Linear cache growth | **Distilled from VGGT**; vs VGGT (below) | 2507.11539 Sec. 3.1–3.3, Tables 1, 6, 9 |
| Point3R | C with explicit 3D pointer memory | F | Pose-free | Grows with explored area; OOM >700 frames (per TTT3R) | Not against bidirectional | 2507.02863 Table 1 |
| STream3R | C decoder-only, KV cache; initialised from VGGT | F | Pose-free | Trained 4–10 frames; window variant | **Mask removed at test time, same weights, Table 3** (below) | 2508.10893 Eq. 5, Table 3†, App. B†, Sec. 6 |
| TTT3R | C (CUT3R with TTT-style gated update, training-free) | F | Pose-free | Thousands of frames at 6 GB | Gap to offline VGGT persists (Table 6) | 2509.26645 Sec. 1, 5, Tables 6–7 |
| WinT3R | BC: bidirectional inside sliding windows of 4 (stride 2), causal across | F | Pose-free | Camera-token pool, +1 per frame | **Window ablation, Tables 5–6** (below) | 2509.05296 Tables 5–6 |

### 2.4 Embodied world models and video generation

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

### 2.5 Transformer policies and navigation models

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

## 3. Evidence on causal against bidirectional

### 3.1 Same weights or same architecture (the cleanest)

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

### 3.2 Cross-model, and distillation as evidence

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

### 3.3 Why authors choose causal

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

## 4. Verdicts on the three hypotheses

### H1. Recent learning-heavy approaches already come very close to our mapping models, with only data-coverage or domain limitations — **mixed**

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

### H2. The real distinction is free queries that break frame boundaries vs per-frame outputs — **mixed**

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

### H3. Bidirectionality is key to good global representations, and end-to-end policies avoid it from a bias toward causal models — **part 1 supported with caveats; part 2 mixed, leaning contradicted**

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

## 5. What our work adds, without overclaiming

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

## 6. Experiments the literature suggests

1. **Length generalisation** (CausNVS threat): evaluate both models at
   history lengths not seen in training.
2. **Distil bidirectional into causal** (StreamVGGT, CausVid): does a
   bidirectional teacher let causal hybrid4 learn the map at xs? This
   separates "hard to optimise" from "cannot represent".
3. **Revisit / re-query** (CUT3R, MUSt3R): a causal pass followed by one
   bidirectional pass over cached states. This is the k-sweep already
   planned (causality page, plan B).
4. **Chunk-bidirectional** (WinT3R, CausVid, Dreamer 4 block-causal): the
   cheapest bridge in the literature.

## Sources (pages opened)

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
