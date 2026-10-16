# LVSM-style latent scene tokens as memory for an egocentric, odometry-conditioned NVS model

Written 2026-10-16, in answer to: "LVSM does almost look like it could
support end to end policy or world model memory though, especially the
encoder decoder version with latent register tokens to represent the whole
scene."

**Framing (after the user's correction).** The niche is egocentric,
odometry- or action-conditioned, end-to-end NVS as a memory objective,
recomputed in parallel from the whole history every forward pass. hybrid4
already works this way: it scans the whole history in both directions and
recomputes it each pass. Streaming, incremental updates and constant-size
state are not requirements and appear here only in passing.

Every claim cites a section, figure or table that was opened (see Sources).
"Our reading" marks inference beyond what a paper states.

## 1. Bottom line

- **The architecture supports it.**
  - LVSM's encoder-decoder is a differentiable map:
    (posed views) → a fixed set of latent tokens z → (rays) → pixels.
  - Recomputing z in parallel from the whole history is exactly how LVSM
    runs: it is offline and set-based.
  - A policy or world-model head can read z and train end to end. Nothing
    in the design prevents this.
  - We found no paper in which a policy reads LVSM latents. The nearest
    evidence is from SRT, NeRF-RL, GNFactor and VNT-PA (section 5).
- **What the bottleneck buys:**
  - a fixed-size, decoder-agnostic interface;
  - rendering cost independent of history length;
  - a place where "the map" must live, so the decoder (or a policy) cannot
    route around it.
- **What it costs:**
  - quality: about 1.1 dB on RealEstate10K and 2.2–3.0 dB on objects
    against the decoder-only variant;
  - worse behaviour as the number of views grows, which is our regime.
- **Poses are the real gap.**
  - LVSM needs calibrated per-image poses in a common frame.
  - Feeding it odometry means dead-reckoning poses outside the network.
    That is the C1 or "oracle" side of our junction (literature review,
    section 4), not internal composition.
  - Prior pose-free work learns poses from images (UpSRT, RUST, RayZer).
    None takes odometry or actions as relative motion tokens and composes
    them inside the network.
- **Strongest design.** An LVSM-style encoder-decoder whose encoder mixing
  is a hybrid:
  - per-frame attention within each image;
  - bidirectional β∈(0,2) delta-rule layers across frames, keyed by
    odometry tokens, which can apply the frame-to-frame rotation;
  - interleaved global attention;
  - latent tokens read the result.

  The vanilla baseline is the same model with attention only. The oracle
  ceiling is GTA or PRoPE on dead-reckoned poses.

## 2. LVSM precisely (Jin et al., arXiv 2410.17242v2, ICLR 2025)

**Input tokens** (Sec. 3.1, Eq. 1–2):
- Each input image is concatenated channel-wise with its per-pixel Plücker
  ray map, computed from known intrinsics and extrinsics.
- The result is patchified (p = 8) and linearly projected to d = 768.
- Target views are represented by their Plücker rays alone, tokenized with
  a separate linear layer.
- Output patches go through linear + sigmoid to RGB (Eq. 4). The loss is
  MSE + λ·perceptual (Eq. 5).

**Decoder-only** (Sec. 3.2, Eq. 8):
- One stack of full bidirectional self-attention over the concatenated
  [input tokens; target tokens].
- The target outputs are read out; there is no intermediate
  representation.

**Encoder-decoder** (Sec. 3.2, Eq. 6–7):
- Learnable latent tokens e₁..e_l are concatenated with the input tokens.
  The encoder (self-attention only) updates both. The latent outputs z are
  kept and the input outputs are discarded.
- The decoder runs self-attention over [z; target ray tokens] and reads out
  the target tokens.
- **Latent count and size** (App. A.2): l = 3072 latent tokens; 12 encoder
  plus 12 decoder layers; d = 768.
- **The decoder also updates the latents.** Sec. 4.4 says "the parameters
  of the decoder are a part of the scene representation". Removing this
  ("w/o latents' updating") costs 1.46 dB on GSO, and a pure
  cross-attention decoder in the SRT style costs 3.47 dB (Sec. 4.4
  architecture ablation table; Fig. 11 attention masks).
- **Our reading.** z as stored is not the whole representation. A policy
  head should be a set of query tokens inside the decoder, not an MLP
  on z.
- **How strong the compression is.** At 256² with p = 8 an image is 1024
  tokens. The scene setting uses 2 input views, so 2048 input tokens are
  mapped to 3072 latents: no compression in count. Objects use 4 views,
  4096 → 3072 (our arithmetic from A.2 and Sec. 4.1). The latents are a
  fixed-size interface rather than a strong bottleneck at the trained
  setting.

**Training data and scale:**
- Objects: Objaverse renders, 32 views per object, tested on GSO and ABO.
  Scenes: RealEstate10K, with the pixelSplat split (Sec. 4.1).
- 64 A100s, batch 8 per GPU, 80k iterations (objects) or 100k (scenes),
  then 512-resolution fine-tuning; 3–7 days (Sec. 1; App. A.2).
- QK-Norm was needed for stability, and steps with gradient norm above 5
  were skipped (Sec. 4.2).
- Scene poses are normalized relative to the mean input camera and scaled
  into [-1,1]³. This is GS-LRM's procedure, which LVSM follows (GS-LRM
  App. A.5; LVSM App. A.2).

**Small-compute results** (Sec. 4.3):
- The 6-layer decoder-only model on 1 A100 for 7 days reaches 27.66 dB on
  RealEstate10K.
- The 12-layer model on 2 GPUs reaches 28.56 dB, above GS-LRM trained on
  64 GPUs.

**Quality and speed trade-off between the variants:**

| | RE10K PSNR | GSO-512 PSNR | ABO-512 PSNR | Source |
| --- | --- | --- | --- | --- |
| Encoder-decoder | 28.58 | 29.32 | 29.81 | Table 1 |
| Decoder-only | 29.67 | 32.36 | 32.10 | Table 1 |

- **Depth split:** with 24 layers fixed, 6+18 beats 12+12, which beats
  18+6. A deeper encoder hurts; the authors attribute this to harder
  compression (Sec. 4.4, model-size ablation).
- **Number of views** (Sec. 4.5, Fig. 7): trained with 4 views, the
  encoder-decoder drops beyond 8 input views at test, like GS-LRM. The
  decoder-only model keeps improving.
- **Speed** (Sec. 4.5, Fig. 8): encoder-decoder rendering speed is constant
  in the number of inputs. Decoder-only cost grows quadratically. Fig. 8
  gives FPS only as a plot; we did not extract numbers.
- **Limitations** (App. A.7):
  - The models are deterministic, so unseen regions come out as noisy,
    flickering "fixed pattern" artifacts.
  - They are fragile to aspect ratios not seen in training.

## 3. Latent tokens as a parallel-recomputed scene memory

**What the architecture gives:**
- z is a fixed-size set computed by one parallel pass over all posed
  inputs. This matches "recompute from the full history, in parallel".
- Encoder self-attention costs O((N·hw + l)²) in the number of history
  frames N. LVSM's own fixed-latent advantage is at decoding.
- A history of hundreds of frames needs sub-quadratic mixing. Precedents
  in NVS encoders:
  - **Long-LRM** (Table 4): a hybrid of 7 Mamba2 blocks to 1 transformer
    block, repeated 3 times, beats pure attention at 4 views (21.58 vs
    21.13 dB). Pure Mamba2 is far worse (19.82; 24.28 vs 26.82 dB at
    32 views). App. D.3: training is more stable with the attention
    blocks spread evenly.
  - **LaCT** (Sec. 4.1, 5.1; Table 2; Fig. 4):
    - Per-image window attention, plus a large-chunk test-time-training
      fast-weight layer updated once with all input tokens and applied to
      input and target tokens. It is parallel, not streaming.
    - It matches full attention with about 11× faster prefill (1.4 s vs
      16.1 s on 48 images at 512²).
    - It "clearly outperforms" a Perceiver-style baseline with 4096
      registers, which is close to LVSM's encoder-decoder.

**Directly relevant warning** (LaCT Sec. 5.4, Fig. 8b):
- At equal d² state per block, a bidirectional per-token Mamba-2 (one
  layer each direction, with per-image window attention) does
  significantly worse NVS than LaCT's linear large-chunk recurrence.
- Linear fast weights (f(x) = Wx with a dot-product loss, i.e. linear
  attention) also lose to SwiGLU fast weights (Fig. 8a).
- Neither ablation is a delta rule, and neither uses β ∈ (0, 2).
- **Our reading.** On image tokens, recurrence should step at frame
  granularity, with patches mixed within the frame by attention, not
  token by token across patches.

**Bypass.** Two papers show that a downstream module routes around memory
when it can:
- NSTM (Sec. 3.2): "the model tends to over-rely on attention and bypasses
  historical memory". The fix is readout tokens masked to see only the
  memory.
- World Tokens (abstract): action tokens see only the world tokens
  ("exclusive routing prevents the policy from bypassing that
  representation").
- The LVSM encoder-decoder is bypass-free by construction: the decoder
  sees only z and the target rays. This is the main thing the bottleneck
  buys for a memory study.

**Streaming relatives, in passing:**
- CUT3R: 768 learnable state tokens (d = 768) updated per frame and read
  by a raymap query with an RGB head (Sec. 3, implementation details).
- tttLRM: LaCT fast weights as an implicit 3D memory, with an online
  variant (abstract).
- NSTM: periodic TTT memory writes and per-frame reads (Sec. 3.1).

## 4. Pose input: calibrated poses vs relative motion

| Work | Input-view pose | Query (target) pose | Who composes motion |
| --- | --- | --- | --- |
| LVSM | Plücker rays, relative to the mean input camera (GS-LRM A.5) | Plücker rays, same frame | Outside (SfM/COLMAP) |
| SRT | Rays in the canonical camera c₀'s frame (Sec. 3, Eq. 4) | Rays in c₀'s frame | Outside |
| UpSRT | None, only c₀ is identity (Sec. 4.4) | Rays in c₀'s frame | Network registers an unordered set |
| RUST | None (Sec. 2) | 8-D latent pose from half the target image (Sec. 2.1) | Network; pose space follows the data distribution (Sec. 4.2; Limitations: out-of-distribution poses may not be reachable) |
| RayZer | Predicted SE(3) cameras relative to a canonical view, turned into Plücker rays (Sec. 4.2, Eq. 6) | Predicted from the target image at evaluation (Sec. 5.1) | Network; the learned pose space is warped from real SE(3) (Sec. 5.3). An image-index encoding gives video order (Sec. 4.2; Table 4) |
| GTA / PRoPE | Per-token SE(3) (PRoPE: full projective frustum) as relative transforms in attention (GTA Eq. 4–6; PRoPE Eq. 16–20) | Same | Outside; invariant to the global frame |
| NWM | None; past frames as context | Relative egocentric action (u, φ) with sin-cos + MLP embedding (Sec. 3) | Network, as a feature embedding with no group structure; generative |
| **Ours** | **Relative odometry or action tokens per step** | **Body-frame offset relative to the pose at step t** | **Network, in context** |

- **Nobody we found** feeds per-step relative motion into an LVSM-style
  model and composes it internally.
- **The closest are RayZer and RUST.** They infer motion from pixels, which
  is our visual-odometry setting (F8), but:
  - neither can render at an externally specified pose without the target
    image or a search in latent-pose space (RayZer Sec. 5.1; RUST
    Sec. 4.2);
  - an odometry- or action-specified query is what a policy or world model
    needs ("what will I see after a?"). That is an advantage of our
    contract.
- **Dead reckoning plus Plücker rays or PRoPE turns our task into LVSM's.**
  It is the right oracle ceiling, but it answers a different question: the
  composition happens outside the network (literature review, section 4,
  junction (a)).
- **Evidence that relative encodings beat Plücker rays inside LVSM:**
  - PRoPE Table 1 (RE10K, about 25M parameters): Plücker 20.48, GTA 22.51,
    PRoPE 22.80 dB.
  - PRoPE Table 6, at 100× compute: Plücker 25.64 vs PRoPE 26.56 dB.
  - Camera-frame-only raymaps ("CamRay") beat absolute Plücker rays in the
    hybrid experiments (Sec. 4.4).
  - Which LVSM variant PRoPE used is not stated explicitly. We assume the
    decoder-only variant (6 blocks, App. A.1.1). Not verified.

## 5. Neighbours: policies and world models reading NVS-trained latents

| Work | Representation | How the policy or world model uses it | End to end? | Against our niche |
| --- | --- | --- | --- | --- |
| SRT (Sec. 3.1, 4.5) | Set latent, grows with the inputs | Frozen or fine-tuned encoder plus a new decoder for semantic segmentation | Optional | Posed set; not a policy |
| OSRT (Sec. 2.2) | Slot Attention → N user-chosen slots (fixed size) | Object decomposition | n/a | Posed set |
| NeRF-RL, Driess et al. 2022 (Fig. 1; Sec. 4.1) | Latent from multi-view encoder plus NeRF decoder | RL on the frozen latent; beats CNN-decoder, contrastive and keypoint states | No, two stages | Fixed calibrated cameras, not egocentric |
| Driess et al. 2022, compositional NeRF dynamics (abstract) | Per-object latents trained through a NeRF decoder | GNN latent dynamics for planning (world model) | Two stages | Fixed cameras |
| GNFactor (Sec. 3.3; Table 4) | Voxel features trained by NVS (RGB + distilled features) | Perceiver behaviour cloning, **jointly trained**: 36.8% vs 24.2% success without the rendering objective | **Yes** | Explicit 3D voxels; RGB-D; calibrated |
| MV-MWM (Sec. 3) | Multi-view masked video autoencoder: reconstructs masked viewpoints | Dreamer-style world model and policy on its representations | Two stages | Viewpoints are slots, not poses |
| VNT-PA (abstract; Sec. III–IV) | Set of keyframe tokens indexed by pose, with pose RoPE in attention | Imitation planner; pose RoPE beats temporal RoPE, by up to 17.1 SPL points on long episodes | Yes | Ground-truth poses in a world frame; no rendering objective |
| World Tokens (abstract) | Fixed set of world tokens trained by future-video denoising | Sole context of the action expert | Yes | Not NVS; no geometry |

- **No paper found** uses LVSM latents, or latent tokens trained by
  LVSM-style NVS, as a policy or world-model state.
  - Searches: LVSM + policy, world model, robot, navigation.
  - This is absence of evidence from a bounded search, not proof.
- **The pattern that is supported:** NVS supervision of the shared
  representation helps the policy. GNFactor shows it with joint training;
  NeRF-RL shows it two-stage.

## 6. Assessment

### 6.1 As the camera-rung model, or its vanilla baseline

- **Vanilla baseline (recommended).** An LVSM encoder-decoder trained on
  our egocentric stream under our contract. Per input token:
  - the image patch;
  - camera-frame-only rays (fixed intrinsics; no world pose).
  Per step:
  - one odometry or action token;
  - RoPE on the time step, as now.
  - Learned latent tokens.
  - Query: Plücker or camera-frame rays of the query camera, plus the
    body-frame offset relative to step t.
  - Attention only. This is "LVSM-style decoding over an egocentric
    stream" (plan, Horizon section) made honest: the network must compose
    the odometry.
- **Oracle ceiling.** The same model with dead-reckoned poses as Plücker
  rays, or better, PRoPE or GTA. It measures what external composition
  buys.
- **Our model.** The same encoder-decoder with the encoder's cross-frame
  mixing replaced by hybrid layers:
  - per-image (window) attention within a frame;
  - bidirectional Gated DeltaNet with β ∈ (0, 2) across frames, where
    odometry tokens drive the reflections;
  - global attention every k layers.

  The latents read the result, by attention or by sitting in the scanned
  sequence. The decoder stays LVSM's: self-attention over latents and
  query rays.
- **Policy or world-model head.** Extra query tokens in the decoder,
  alongside the target-ray tokens. Train jointly with the NVS loss, as in
  GNFactor. Do not use an MLP on z, because LVSM's latents are refined in
  the decoder (section 2).

### 6.2 Where the geometric bias plugs in

- **In the update (the encoder's cross-frame mixing).** This is where the
  hypothesis-class claim lives: exact state rotations from odometry,
  composed in context. It keeps the contract (internal composition).
  - Apply the transition at frame boundaries rather than per patch token.
    LaCT Fig. 8b and Long-LRM Table 4 both show per-token linear recurrence
    over image tokens underperforming. hybrid4's 9 patches per step will
    become hundreds.
  - This is a design point, not a decided choice. **Open question for the
    user:** do odometry rotations act on a frame-level state (one update
    per step, LaCT-like chunking), or on a per-token scan with patches
    ordered within each frame?
- **In the positional encoding (rays and latents).**
  - GTA and PRoPE give the largest controlled gains over Plücker rays
    inside LVSM (section 4), but they need per-token poses.
  - Under our contract, two forms are allowed:
    - (a) within-image RoPE plus camera-frame rays;
    - (b) a relative encoding between the query and step t, which the task
      gives.
  - Odometry composed into the encoding is either external (GTA on
    dead-reckoned poses: the oracle) or internal but in the encoding (a
    PaTH-style cumulative Householder product: the planned hybrid5). That
    makes "both" a testable ladder:
    1. update only;
    2. update plus PaTH;
    3. oracle GTA or PRoPE.
- **Open question for the user.** Learned latent tokens carry no geometric
  attribute, so GTA and PRoPE are undefined on them. One option is to give
  them the query or reference frame; another is to leave them unencoded.
  This needs a decision.

### 6.3 Main risks

1. **The bottleneck caps quality, and the cap tightens with history
   length.**
   - LVSM's encoder-decoder trails decoder-only and degrades beyond 8 views
     (Sec. 4.5).
   - LaCT's 4096-register Perceiver loses clearly (Fig. 4).
   - A long egocentric history is exactly the many-view regime. The latent
     count l is a real hyperparameter for us, not LVSM's 3072 by default.
2. **Per-token bidirectional recurrence on image tokens underperforms**
   (LaCT Fig. 8b; Long-LRM pure Mamba2). Our mechanism must survive at
   patch counts far above 9 per step. The hybrid ratio and the chunking
   matter (Long-LRM App. D.3).
3. **Unseen regions.** Deterministic MSE NVS produces blur or fixed-pattern
   flicker where the history never looked (LVSM App. A.7). An egocentric
   agent queries such regions often. We need a visibility mask, or a
   decision that hallucination is out of scope.
4. **The oracle may dominate.** With poses dead-reckoned outside the
   network, GTA or PRoPE will likely set a ceiling the internal model must
   approach. The claim to test is efficiency relative to the matched
   vanilla, not beating the oracle.
5. **Odometry noise and drift.**
   - SRT and GTA degrade gracefully under pose noise (SRT Sec. 4.4,
     Fig. 6; GTA App. Table 11).
   - Internal composition of noisy odometry is untested in the
     literature. This matches caveat (d) of the literature review.
6. **Interface subtleties.**
   - LVSM latents are refined in the decoder, so the policy readout must
     sit in the decoder.
   - Training needs QK-Norm (Sec. 4.2).
   - The [-1,1]³ scale normalization uses all input poses. That is fine
     under parallel recompute, but it has no analogue for odometry without
     a chosen scale.
7. **Scale.** Published LVSM numbers use 64 A100s. The 1–2 GPU results
   (Sec. 4.3) and PRoPE's 6-block, 2-GPU setting (App. A.1.1) show that
   small-scale comparisons are meaningful. The camera rung can stay small.

## Sources (pages opened, arXiv HTML unless noted)

- Jin et al. LVSM. arXiv 2410.17242v2. Sec. 1, 3.1–3.2, 4.1–4.5,
  Table 1, ablation tables, App. A.1–A.7.
- Zhang et al. GS-LRM. arXiv 2404.19702. App. A.5 (camera pose
  normalization).
- Sajjadi et al. SRT. arXiv 2111.13152. Sec. 3, 3.1, 4.4, Fig. 6.
- Sajjadi et al. OSRT. arXiv 2206.06922. Sec. 2.2–2.3.
- Sajjadi et al. RUST. arXiv 2211.14306. Sec. 2, 2.1–2.2, 4.2,
  Limitations.
- Jiang et al. RayZer. arXiv 2505.00702. Abstract, Sec. 3–4.2, 5.1,
  5.3–5.4, App. (interpolation, curriculum).
- Miyato et al. GTA. arXiv 2310.10375. Sec. 3.1–3.2, Sec. 4 (RePAST
  comparison), App. (noise, timing).
- Li et al. Cameras as Relative Positional Encoding (PRoPE). arXiv
  2507.10496. Sec. 3.4–3.5, 4.1–4.2, 4.4, 4.7 (Table 6), App. A.1.1.
- Zhang et al. Test-Time Training Done Right (LaCT). arXiv 2505.23884.
  Sec. 4.1, 5.1 (Table 2, Fig. 4), 5.4 (Fig. 8), App. A, B.1.
- Ziwen et al. Long-LRM. arXiv 2410.12781. Sec. 3.1–3.2, 5.1 (Table 4),
  App. D.3, E.
- Elmieh et al. Online Neural Space Time Memory (NSTM). arXiv 2607.15271.
  Sec. 3.1–3.2.
- Wang et al. CUT3R. arXiv 2501.12387. Sec. 3 (state, raymap query, RGB
  loss), implementation details.
- Wang et al. tttLRM. arXiv 2602.20160. Abstract only.
- Jia et al. Efficient-LVSM. arXiv 2602.06478. Abstract only.
- Imtiaz et al. LVT. arXiv 2509.25001. Title and Fig. 1 caption only.
- VNT-PA, Visual Navigation Transformer with Pose Attention. arXiv
  2609.21212. Abstract, Sec. I–IV.
- Driess et al. Reinforcement Learning with Neural Radiance Fields. arXiv
  2206.01634. Abstract, Fig. 1, Sec. 4.
- Driess et al. Learning Multi-Object Dynamics with Compositional NeRFs.
  arXiv 2202.11855. Abstract.
- Ze et al. GNFactor. arXiv 2308.16891. Abstract, Fig. 3, Sec. 3.3,
  Table 4.
- Seo et al. Multi-View Masked World Models. arXiv 2302.02408. Sec. 3.
- World Tokens. arXiv 2608.09730. Abstract only.
- Seitzer et al. DyST. arXiv 2310.06020. Abstract only.
- Bar et al. Navigation World Models. arXiv 2412.03572. Sec. 3, 4.

**Not verified:**
- LVSM Fig. 8 FPS values;
- which LVSM variant PRoPE used;
- venues beyond what the arXiv pages print;
- full texts of tttLRM, Efficient-LVSM, LVT, DyST and World Tokens.
