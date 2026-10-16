# Learned ego-motion at scale: applications review

Written 2026-10-15. This is the second review for the mapping study. The
first, [docs/literature_review.md](literature_review.md) (published at
https://claude.ai/artifact/DJ4WFQ5V7ArJrDV31EQ8bN), covers the theory:
linear recurrences whose transitions can have negative eigenvalues (Grazzi
et al. 2025; Gated DeltaNet; DeltaProduct) can represent rotations and
track state. This review asks what has reached practice. Every source was
opened on its primary page (arXiv abstract or full text, proceedings,
official code, model card or company page). Company statements are
labelled **claim**. "Shows" means the source demonstrates it. "Suggests"
means it argues or implies it. Unverified items are listed in section 9.

## 1. Bottom line

The theory has reached language models, not embodied systems.
Qwen3-Next, Qwen3.5 and Kimi Linear ship delta-rule layers with β in
(0, 1), which cannot rotate the state. Olmo Hybrid 7B (Ai2, 2026) is the
one open model we found with negative eigenvalues on, and it cites
Grazzi et al. as the reason. No robot policy, world model, driving
stack or 3D model we found uses a rotation-capable recurrence. The
group-structured position encodings used there (GTA, PRoPE, STRING) all
take poses computed outside the network. The blocker is real but
narrower than the working belief. Attention learns registration
(relative pose from pixels) at scale, under 3D supervision. Short-horizon
control is learned end to end with no geometric module. What nobody
learns end to end is the composition of ego-motion over a long history.
We audited 63 systems for an unbroken gradient path from observations
and actions to the decision. Those that are both end to end and
reliable keep a few seconds of memory or are given pose (GPS+compass,
ego history, engine poses). Every long-memory end-to-end attempt
reports drift, forgetting or lost consistency; DD-PPO without
GPS+compass reaches 0.15 SPL after 2.5B steps. The goal is not rare.
DeepMind Lab and Habitat agents have pursued it since 2016, and
reliability was bought with pose inputs or pose supervision. Whether a
motion-group-capable operator clears the blocker is untested. Studies
that vary only the memory operator in egocentric tasks exist, and the
operator matters: in ObjectNav, DeltaNet and Gated DeltaNet beat a
transformer and a GRU with everything else fixed (LANav, 2026). But
none of them includes an operator that can apply an input-dependent
rotation; those delta rules use β in (0, 1). The failures usually
blamed on credit assignment, the RL algorithm or reward were shown
either on tasks with no rotation or with memories that cannot rotate,
so for ego-motion tasks they are confounded with memory capability.
Recency bias in long-context embodied models fits the same reading.
Two risks remain independent of the operator: imitation shortcuts
(copycat, causal confusion) and reward sparsity early in training. The
cleanest first tests are world-model memory on revisits, and a
one-flag change (β to (0, 2)) in the LANav ObjectNav setup.

## 2. Map of geometric junctions

### 2.1 Four ways a junction is handled

| Code | How geometry is handled | Typical example |
| --- | --- | --- |
| C1 | Poses in a common frame come from an external estimator (localisation, CAN bus, SfM, game engine, GPS+compass) and enter as input or drive a transform | LVSM with Plücker rays from SfM; WorldMem pose-indexed retrieval |
| C2 | An explicit 2D or 3D state with a hand-built warp, projection or renderer | BEV grid warped by ego motion; point-cloud cache in GEN3C |
| C3 | Geometry is a supervised output that an explicit system consumes | VGGT poses and depth; DROID-SLAM flow into bundle adjustment |
| C4 | Fully end-to-end latent memory: only observations and actions in, no pose labels, no external renderer | GameNGen; π0.5; PoliFormer |

The toy task in this study is C4 in miniature. Only rendered pixels
supervise, odometry is relative (or absent), and there are no pose
labels and no renderer. The question for each junction is whether a C4
solution exists at scale, and if not, what replaced it. Section 4
audits whether each system listed as C4 is end to end in the gradient
sense.

### 2.2 Junction table

| Domain | Junction | How it is handled now | C4 at scale? | Evidence |
| --- | --- | --- | --- | --- |
| Driving perception | Temporal fusion of past BEV features or object queries | C1 + C2. Past BEV is warped by the CAN or localisation ego pose; query and anchor positions are transformed by ego pose | No | BEVFormer, BEVDet4D, SOLOFusion, StreamPETR, Sparse4D v2 / SparseDrive code, DriveTransformer |
| Driving planning | Ego history for the planner | C1. Past ego waypoints as text (EMMA), egomotion tokens (Alpamayo-R1), CAN-bus MLP (BEVFormer, UniAD, DriveTransformer). Image memory 2 to 4 s | No. Ego status alone matches UniAD and VAD on nuScenes open-loop | EMMA, Alpamayo-R1, DriveVLM; Li et al. 2024; Zhai et al. 2023 |
| Driving, industry | "End-to-end" production stacks | Tesla and Wayve: undetermined from primary sources. Waymo keeps structured outputs (C3). comma runtime policy: C4 over 2 s, trained with C1/C2 machinery | Claimed, not shown | Tesla release note (claim); Wayve site (claim); Waymo blog Dec 2025 (claim); Goff et al. 2025 |
| Driving world models | Ego and camera conditioning | C1. Speed and curvature as action; explicit camera intrinsics and extrinsics | No | GAIA-1, GAIA-2 |
| Navigation | Odometry and path integration | C1 (GPS+compass given by benchmarks) or C3 (learned visual odometry, composed by an explicit SE(3) or goal-vector update) | No. With pose removed, the winning recipe is learned VO plus explicit composition | DD-PPO; Wijmans et al. 2023; Datta et al. 2020; Partsey et al. 2022 |
| Navigation | Long-range memory and localisation | C1 + C2 (posed back-projection into metric maps; SfM + PnP) or a topological graph with learned distances | Partly: topological graphs avoid metric pose but are explicit structures | VLMaps, ConceptGraphs, OK-Robot, Mobility VLA, 3D-Mem, MapNav; ViNT, NoMaD |
| Navigation | Short-horizon policy memory | C4 as frame or token history (5 frames to a full episode) | Yes, at short horizon. Degrades at long horizon | GNM, ViNT, SPOC, PoliFormer, NaVid, Uni-NaVid, NavFoM; StreamVLN prunes tokens with a pose-based voxel grid (C2) |
| Manipulation VLAs | Observation history | C4 with almost no memory: 1 frame (OpenVLA, π0, π0.5, GR00T N1), 2 frames (Octo, RDT-1B), 6 (RT-1) | Only short. Naive multi-frame input often hurts | Octo; Diffusion Policy; HAMLET; CronusVLA; Torne et al. 2025 |
| Manipulation VLAs | Long-horizon memory | C4 add-ons: language summaries (MEM), keyframe retrieval (MemER), fast-weight TTT (RoboTTT), gated delta rule (MemBodied, Dual-Stream) | Emerging, 2026, research scale | MEM, MemER, RoboTTT, MemBodied, Dual-Stream |
| Mobile manipulation | Base pose and memory over base motion | π0.5: C4, base velocity in state, no memory. MoManipVLA: C1 + C2 (tracking camera pose, ESDF). BUMBLE: topological plus occupancy map | No memory over base motion is shown end to end | π0.5; MoManipVLA; BUMBLE |
| 3D-aware VLAs | Spatial input | C1 + C2 (calibrated point clouds: BridgeVLA, SAM2Act, GeoVLA), C2 with estimated depth (SpatialVLA), or frozen VGGT features (C3 as a feature: Evo-0, GeoAware-VLA, Spatial Forcing) | Geometry enters as an input, not as memory | BridgeVLA, SAM2Act, GeoVLA, SpatialVLA, Evo-0, Spatial Forcing |
| Video world models | Consistency on revisits | C4 fails beyond its window. Fixes are C1 (pose-indexed retrieval, PRoPE) or C2 (point clouds, surfels, 3D caches) | Claimed (Genie 3) or research scale (Po et al., Infinite-World) | GameNGen, DIAMOND, Oasis, Matrix-Game 2.0, LoopNav; WorldMem, Context-as-Memory, RELIC, WorldPlay; GEN3C, Wu et al., VMem, Memory Forcing |
| Video world models | Camera control | C1. Plücker ray maps or relative camera encodings from given poses | No | CameraCtrl, Cosmos, PRoPE, WorldPlay |
| SLAM | Pose and map estimation | C3. Learned correspondences or pointmaps; explicit Gauss-Newton bundle adjustment or pose-graph solve | No | DROID-SLAM, DPVO, MASt3R-SLAM, VGGT-SLAM |
| 3D reconstruction | Multi-view geometry | C3. Feed-forward poses and pointmaps under 3D supervision; global alignment (DUSt3R) or optional BA (VGGT) | Registration yes; persistence no | DUSt3R, MASt3R, VGGT, π³, MapAnything |
| 3D reconstruction | Streaming state | C3 with a recurrent latent state. Drift and forgetting on long sequences | Partly | CUT3R, TTT3R, MeMix, StreamVGGT, Point3R |
| Novel view synthesis | Camera pose | C1 (LVSM, GTA, PRoPE). C4 at modest scale (RUST, RayZer, XFactor) | Modest scale only | LVSM, GTA, PRoPE, RUST, RayZer, XFactor |

### 2.3 Details by domain

**Driving.** Every camera BEV or query stack we checked aligns memory
using ego pose. BEVFormer "align[s] B_{t-1} to Q according to
ego-motion"; its official code rotates the previous BEV by the CAN-bus
yaw change, and also adds the CAN-bus vector to every BEV query, which
the paper does not mention. BEVDet4D warps the previous BEV by the ego
transform. StreamPETR transforms the 3D centres of memory queries by the
relative ego pose (its Eq. 9) and conditions them on the pose through a
motion-aware LayerNorm. SparseDrive's paper does not mention a warp, but
its code projects cached anchors with `T_temp2cur`, inherited from
Sparse4D v2. Planners then add ego status as input. Li et al. (CVPR 2024)
show this is a shortcut: an MLP on velocity, acceleration, yaw and
command matches UniAD and VAD on nuScenes L2 and collision, and VAD with
ego status loses little when the cameras are blanked (L2 0.37 to 0.46 m).
NAVSIM gives agents at most 1.5 s of history and filters easy scenes to
blunt the shortcut. No benchmark we found tests long-term spatial memory
in driving.

**Navigation.** Odometry is the most consistently explicit junction.
Habitat benchmarks hand the agent GPS+compass. When it is removed, Datta
et al. (2020) and Partsey et al. (2022) train a separate visual odometry
network and compose its output with an explicit transform; Partsey et
al. reach 94% success under actuation and sensor noise. Wijmans et al.
(ICLR 2023) show that a blind LSTM agent forms a decodable map in its
memory, but its per-step input includes GPS and compass, so the agent
learns mapping, not path integration. Long-range systems use metric maps
built by posed back-projection (VLMaps, ConceptGraphs, OK-Robot) or SfM
poses plus PnP (Mobility VLA, where Gemini outputting waypoints directly
scored 0% against 90% with the graph). Short-horizon policies are C4 and
work: PoliFormer reaches 85.5% on CHORES-S ObjectNav with RGB only, but
"the agent's recollection of the rooms it has explored deteriorates"
after more than four rooms.

**Manipulation and VLAs.** Leading VLAs use the current timestep only or
a two-frame window (table above). Octo reports "significantly diminishing
gains beyond the first additional frame". Diffusion Policy's image
variant gets worse with longer observation horizons. HAMLET finds naive
multi-frame GR00T N1.5 drops LIBERO from 95.6 to 86.8 and costs up to
2.4 times the latency, though it helps on real history-dependent tasks.
The reported causes split. Copycat behaviour and causal confusion (de
Haan et al. 2019; Wen et al. 2020; cited again by Physical Intelligence's
MEM) are learning problems. Torne et al. (2025) find diffusion policies
under-use the past instead, and fix it with an auxiliary past-token loss.
Compute and pre-training mismatch are the other stated causes (MemoryVLA,
HAMLET). π0.5 says its "relatively modest context" limits "navigating
between different rooms or remembering where objects are stored".

**World models.** Action-only models forget once a place leaves the
window. GameNGen has "a little over 3 seconds of history". DIAMOND says
losing visibility "may cause the model to forget". Matrix-Game 2.0 lacks
"explicit memory mechanisms". LoopNav measures this: on the return leg of
a Minecraft loop, the models it tests hold for about 20 frames and then
lose consistency. The 2025-2026 fixes are C1 or C2. WorldMem retrieves by
field-of-view overlap of stored poses, which come from the game engine;
with poses predicted from actions, PSNR falls from 25.32 to 23.13.
Context-as-Memory uses Unreal Engine poses. WorldPlay converts discrete
actions to poses by fixed translations and rotations and feeds them
through PRoPE. GEN3C, Wu et al. and Memory Forcing keep point clouds.
Infinite-World argues the other way: estimated poses on real video are
noisy, so it uses a pose-free compressed memory. WorldTrace (2026) shows
that KV-cache retrieval fails past the training horizon because temporal
RoPE offsets leave the trained range.

## 3. Who has tried to build the end-to-end agent

The goal here is an agent that takes egocentric video plus its action
history and acts as if it had a unified map. This is not a rare goal.
Deep RL in partially observable 3D worlds has pursued it since 2016.
What is rare is reaching it without pose: once reliability matters, the
field adds pose or velocity as an input, supervises geometry with its
own labels, or builds an explicit map.

### 3.1 Memory agents in partially observable 3D environments

| System | History in | Memory | Supervision | Horizon | Pose or velocity in? | Reported failure or bottleneck |
| --- | --- | --- | --- | --- | --- | --- |
| UNREAL (Jaderberg et al. 2016) | frame, previous action, reward | LSTM | A3C plus pixel-control, value-replay and reward-prediction auxiliaries | 20-step unrolls | No | Input reconstruction hurts; in fixed mazes, layout is stored in the weights |
| Mirowski et al. 2017 | frame, velocity, action, reward | stacked LSTM | A3C plus depth prediction (Z-buffer labels) and loop-closure labels from integrated velocity | 3,600 to 10,800 steps | Velocity | LSTM capacity "would be stretched"; about 59% of human score |
| Mirowski et al. 2018 (cities) | RGB crop, landmark-distance goal, action, reward | LSTM | IMPALA; compass heading as an auxiliary target only | 1,000-step episodes | No (compass only as a label) | Needs reward shaping and a curriculum; 11 to 38% failure on held-out goals |
| MERLIN (Wayne et al. 2018) | image, velocity, action, reward, text | external memory matrix | Predictor trained only on a variational prediction loss; policy by policy gradient, with a gradient stop between them | 1.3 s truncation on 90 s to 6 min tasks | Velocity | RL-only LSTM and memory agents "can fail to train" when much must be recalled; credit assignment |
| Neural Map (Parisotto and Salakhutdinov 2017) | frame | 2D memory written at the agent's position | A2C | 100 to 500 steps | Oracle (x, y) for writes | The egocentric variant was proposed but not run |
| CMP (Gupta et al. 2017) | frame | egocentric map warped by odometry, differentiable value iteration | imitation (DAgger) | 200 steps | Perfect egomotion | Authors call perfect odometry "a central limitation" |
| Scene Memory Transformer (Fang et al. 2019) | image, action, pose | growing memory, transformer | deep Q-learning on frozen pretrained embeddings | 500 steps | Pose embedded in each memory entry | Reward falls as pose noise grows |
| EgoMap (Beeching et al. 2020) | frame, depth | projected egocentric map | A2C, end to end | ViZDoom episodes, 1.2B steps | Simulator ego-motion and depth | Falls to baseline above about 10% ego-motion noise |
| Active Neural SLAM (Chaplot et al. 2020) | RGB, sensor pose | explicit map | mapper and pose estimator supervised with ground truth; PPO global policy; Fast Marching planner | episode | Sensor pose | Drift without the pose estimator |
| DD-PPO (Wijmans et al. 2020) | RGB-D, action | LSTM | PPO | episode | GPS+compass | Without GPS+compass: 0 SPL at 100M steps, 0.15 SPL at 2.5B, "an open frontier" |
| Wijmans et al. 2023 (blind) | GPS+compass, action | 3-layer LSTM | PPO, about 2B steps | up to 1,000+ steps | GPS+compass | Forgets excursions; maps not conclusively decodable from sighted agents |
| Partsey et al. 2022 | RGB-D | LSTM, no map | PPO with true pose, then a separately supervised VO dropped in | episode | VO estimate | "Visual odometry is the limiting factor" |
| GTrXL (Parisotto et al. 2020) | frame | gated transformer, up to 6,144-step receptive field | V-MPO | DMLab-30, Memory Maze | No | Ungated transformers fail to optimise (DMLab-30 score 5.0 vs 117.6) |
| DreamerV1 to V3 (Hafner et al.) | frame, action | RSSM (GRU) | world-model prediction losses; actor in imagination with the world model fixed | imagination 15 to 16 steps | No | "Reward prediction alone was not sufficient" |
| R2I (Samsami et al. 2024) | frame, action | state-space model in place of the GRU | DreamerV3 losses | Memory Maze up to 4K steps | No | Below human on 15×15 Memory Maze; non-stationary features |

**Benchmarks for spatial memory under partial observability.** Memory
Maze (Pasukonis et al. 2022): 3D mazes from 9×9 to 15×15 with offline
probes of maps and positions; 15×15 is "out of reach" for current RL.
Memory Gym (Pleines et al. 2023): finite and endless memory tasks.
POPGym (Morad et al. 2023): 13 memory models under one PPO setup.
MIKASA-Robo (Cherepanov et al. 2025): memory tasks for manipulation.
LoopNav and MIND test revisits in world models (section 2.3). None of
these benchmarks tests memory of a 3D space under SE(3) ego-motion
without pose, except the RGB-only Habitat setting and Memory Maze.

### 3.2 Does anyone vary the algebra of the state update?

Several studies vary only the memory operator in an egocentric task,
with the encoder, objective, data and algorithm held fixed. None of them
includes an operator whose transition can apply an input-dependent
rotation to stored state.

| Study | Task | Operators compared (held fixed) | Result | Input-dependent rotation in the set? |
| --- | --- | --- | --- | --- |
| LANav (Zhang et al. 2026) | HM3D-OVON ObjectNav, egocentric RGB | GRU, transformer (context 500), linear transformer, DeltaNet, Gated DeltaNet, WSLA (frozen SigLIP, DAgger then PPO, 4 layers, width 512) | Average SR 26.7, 30.1, 31.3, 33.9, 34.6, 36.4. "Transformer policies do not benefit from larger training contexts"; linear-attention backbones "improve consistently" | No. DeltaNet and Gated DeltaNet use β = sigmoid, in (0, 1) |
| GTrXL (Parisotto et al. 2020) | DMLab-30, first person | LSTM, TrXL, TrXL-I, GTrXL (V-MPO) | 99.3, 5.0, 107.0, 117.6 | No |
| Po et al. 2025 | Memory Maze world model | Mamba2, Mamba2 plus local attention, windowed and full causal attention, block-SSM (same objective, about 200M parameters) | Retrieval PSNR 20.4, 19.3, 26.4, 32.6 (full), 30.8 (block-SSM) | No. Mamba2 decays are real and positive |
| SPOC; PoliFormer | ProcTHOR navigation | transformer decoder against GRU | Transformer ahead (49.9 vs 43.6; 80.4 vs 73.6) | No |
| SRU (Yang et al. 2025) | mapless navigation, egocentric depth, PPO | GRU, LSTM, SRU variants | SR 61.0, 63.5, up to 78.9 | No. Real element-wise gates |
| TransDreamer (Chen et al. 2022) | 3D Hidden Order Discovery, first person | transformer vs GRU state-space model in Dreamer | transformer ahead; not strictly one change | No |
| AGaLiTe (Pramanik et al. 2024) | Memory Maze | GTrXL vs gated linear transformer | parity; AGaLiTe faster | No, as far as we can tell |
| RoboTTT (2026) | manipulation, egocentric and fixed cameras | TTT fast weights vs Gated DeltaNet | 79% vs 56% | β range not stated |

Three non-embodied RL works come closer to the algebra itself, and they
show how rarely rotation is input-dependent:

- **FFM** (Morad et al. 2023) uses a complex decay,
  γ = exp(−α)exp(−iω). α and ω are trained parameters, not functions
  of the input, so the rotation is fixed per channel.
- **SHM** (Le et al. 2025) writes the update as
  M_t = M_{t−1} ⊙ C_t + U_t with an input-dependent C_t in (0, 2):
  real and positive, so no rotation.
- **Complex-valued memories** (Bhethanabhotla et al. 2026) test unitary
  RNNs. Most are fixed unitary maps; one variant, IC-uRNN, makes the
  Householder vectors and phases functions of the input, which is an
  input-dependent rotation. It was not tested on navigation.

The complex-eigenvalue models used in RL (S4D and LMU in POPGym, S5 in
Lu et al., the SSM in R2I, FFM, fixed uRNNs) all rotate by angles that do
not depend on the input. Mamba, mLSTM, SHM, DNC and fast-weight models
cannot rotate at all. A one-change test of an input-dependent rotation
on an egocentric pose-free task is therefore open. LANav is the natural
base for it: its DeltaNet and Gated DeltaNet already beat the
transformer, and switching β to (0, 2) is one flag in the
flash-linear-attention library.

### 3.3 Risks to the thesis, separated from memory capability

The earlier version of this section listed credit assignment, the RL
algorithm and reward sparsity as failure causes unrelated to the
operator. That needs a correction. A failure can only be attributed to a
non-architectural cause if some memory in the comparison could have
represented what the task needs. We checked which memories each source
tested. None tested a memory with input-dependent rotation, and most
tested tasks that do not need ego-motion composition.

| Source | Memories tested | Task needs rotation composition? | Failure cause claimed | Separable from memory capability? |
| --- | --- | --- | --- | --- |
| Ni et al. 2023 | LSTM, GPT-2 | No (T-mazes, key-to-door) | Transformers do not improve long-term credit assignment | Yes for those tasks; says nothing about ego-motion tasks |
| Ni et al. 2022 | LSTM, GRU | No | Shared actor-critic encoder, RL algorithm | Yes for those tasks |
| POPGym | 13 models incl. S4D, LMU (fixed rotation) | No. Labyrinth tasks use absolute moves, no heading | All models converge to similar reward; PPO limits | Yes for those tasks |
| MIKASA-Robo | MLP, LSTM, transformers, Mamba, VLAs | No (tabletop) | Sparse reward | Yes. Dense reward lets PPO-LSTM solve the easy task |
| Memory Gym | GRU, Transformer-XL | No for the verified task (absolute moves) | Advantage normalisation, weak signal | Yes for that task |
| Wen et al. 2020; de Haan et al. 2019; Torne et al. 2025 | MLP, RNN, transformer | No | Copycat, causal confusion, under-use of history in imitation | Yes. These are properties of the imitation objective |
| MERLIN | LSTM, LSTM plus content memory, MERLIN | Partly (DMLab navigation; velocity input) | Credit assignment; RL alone cannot train memory | Partly. The predictive loss helped, but all three memories cannot rotate |
| GTrXL | LSTM and transformer variants | Partly (DMLab navigation levels) | Transformer optimisation | The TrXL fix is real; residual navigation failures are confounded |
| R2I | GRU, LTI complex SSM | Yes (Memory Maze turns) | Vanishing gradients | Confounded |
| Mirowski et al. 2018 | LSTM | Partly (Street View turns, landmarks) | Sparse reward, needed shaping | Partly confounded |
| DD-PPO; Partsey et al. 2022 | LSTM | Yes (PointNav without GPS) | None stated (DD-PPO); "visual odometry" (Partsey) | Confounded. Partsey's fix composes ego-motion outside the network |

**Risks that stay independent of the operator.**

- **Imitation-learning shortcuts.** Copycat behaviour and causal
  confusion come from the objective and appear with stateless and
  recurrent models alike. A better memory may make them worse, by
  making past actions easier to read.
- **Reward sparsity early in training.** With no reward signal, no
  memory gets a gradient. MIKASA-Robo shows dense reward rescues the
  same LSTM. This is a property of the task and objective.
- **Long-term credit assignment.** Ni et al. show it on tasks that need
  no rotation, so it is a genuine separate limit.
- **Optimisation of specific architectures.** TrXL's layer-norm
  placement and Memory Gym's advantage normalisation are fixable
  engineering issues, unrelated to rotation.
- **Positional range.** KV-cache retrieval fails past the trained
  RoPE range (WorldTrace). This is a property of index-based
  positions, and a recurrence avoids it by design.

**Risks confounded with memory capability.**

- **Pose-free navigation failures.** DD-PPO without GPS+compass and
  R2I on the largest Memory Maze fail with memories that cannot apply
  an input-dependent rotation. The cause cannot be assigned to the
  algorithm until a rotation-capable memory is tried.
- **Recency bias in long-context models.** History ablations often find
  returns flatten after a few frames:
  - NaVILA is flat from 8 to 64 frames (SR 49.7 to 50.1).
  - NaVIDA's 8 and 16 frames are equal.
  - StreamVLN does worse with all frames as memory than with 8 (SR 40.0
    vs 45.5).
  - GameNGen's PSNR saturates at about 4 context frames.
  - Octo sees "diminishing gains beyond the first additional frame".
  - MemoryVLA peaks at 16 entries (71.9) and falls at 64 (67.7).
  - LANav's transformer does not improve with longer training context,
    while its delta-rule backbones do.
- **No attention analysis, several explanations.** We found no
  published attention analysis showing that these models
  attend only to recent frames. The explanations given vary: attention
  dilution (NaVIDA), training bias from long sequences (StreamVLN),
  spurious correlations (Torne et al., MemER, RoboTTT) and positional
  range (WorldTrace, Po et al.). Our reading: when a memory cannot
  re-express old observations in the current frame, old frames are
  hard to use, and a model that falls back on recent frames would
  produce exactly this pattern. So recency bias is consistent
  with a capability limit, not evidence against one.
- **Counter-cases where long memory is used.** Wijmans et al. (LSTM,
  given pose: memory matters up to about 1,000 steps), SPOC (context 10
  to 100 raises average SR from 27.8 to 49.9), Uni-NaVid (VLN SR 9.6
  with the current frame only, 48.7 with full memory), and Torne et
  al. with past-token prediction. In Wijmans et al. pose was given. The
  others compared no rotation-capable memory, so they show that long
  memory can help, not which operator it needs.
- **Unpublished evidence from the study's author.** The author reports that in
  their Habitat ObjectNav experiments with multi-turn RL and full
  context in a KV cache, ablations showed the model used only the most
  recent observations. They report the same trend when ablating the
  memory of LightNav-0 (Light Origins, arXiv 2608.30935, 2026). We
  confirmed that LightNav-0 exists, with paper, code and weights. It
  keeps an RGB history in context, compressed on a forgetting curve,
  with no pose input. Its paper contains no history or memory ablation,
  so the LightNav-0 result is the author's own and is not published.

**What follows for the test design.** Any test of the thesis must
remove the operator-independent risks before comparing operators. Use
dense or self-supervised objectives so reward sparsity does not decide
the outcome. Avoid feeding past actions to imitation policies, or add
past-token prediction. Run a history ablation (truncate to the last k
frames) for every operator, since a capability gain should show up as
the model using older frames, not only as a higher score.

## 4. Gradient audit of the "end-to-end" systems

The definition used here: the input end is the egocentric observation
sequence plus the action history; the output end is the next decision,
plan or (for world models) predicted frame. Each system is classed by
its method section:

- **G1.** Fully differentiable from the final objective to perception
  and memory. For a model-free RL agent, the policy-gradient loss counts
  as the objective, and action sampling is not treated as a gap.
- **G2.** An internal non-differentiable step crossed only by a
  policy-gradient or score-function signal.
- **G3.** A hard break with no learning signal across it (modular).
- **G4.** Intermediate geometry supervised by its own labels (pose,
  depth, map, boxes), even if differentiable.

Frozen generic pretrained encoders (DINOv2, EVA-CLIP, SigLIP) are noted
as staged training. They do not demote a system on their own, but they
mean the task objective does not shape perception.

### 4.1 Audit table

| System | Class | Where the break or label supervision is | Staged or frozen | Pose or ego input | Memory |
| --- | --- | --- | --- | --- | --- |
| GNM, ViNT, NoMaD | Core G1; deployed G3 | Topological graph with Dijkstra or A*-style planner; action labels from odometry | ViNT diffusion subgoal model trained separately | No | 5 past frames |
| PoliFormer | G1 (RL) | none | DINOv2 frozen ("important to keep the vision transformer frozen") | No | Full episode, KV cache |
| SPOC | G1 (IL) | Variants add detector boxes as input | SigLIP; frozen or not unstated | No | 100 steps in training |
| NaVid | G1 | Regex parser on text actions | EVA-CLIP frozen (our reading of the text) | No | All frames, 4 tokens each |
| Uni-NaVid | G1 | Token merge by a fixed cosine threshold of 0.95 | Two stages | No | Merged tokens |
| NaVILA | VLA G1; system G3 | Language command parsed into a separately trained locomotion policy | VILA initialisation | No (locomotion uses LiDAR) | 8 to 64 frames |
| StreamVLN | G1 without pruning; G3 with | Voxel pruning by depth and odometry back-projection | Two stages | Depth and odometry at inference for pruning | 8-turn window plus memory tokens |
| NavFoM | G1 | Fixed BATS token sampler | Freezing unstated | Camera azimuth only | 1,024 to 2,048 history tokens |
| LANav | G1 (DAgger then PPO) | none | SigLIP frozen | No | Context 500 (attention) or recurrent state |
| Wijmans et al. 2023 | G1 (RL) | none | none | GPS+compass | LSTM, 1,000+ steps |
| Partsey et al. 2022 | G3 + G4 | VO supervised on pose; composed by a fixed update; no gradient between VO and policy | Staged, VO frozen | VO estimate | LSTM |
| UNREAL, GTrXL, DD-PPO (RGB only), R2I | G1 | none | R2I world model and actor trained by separate losses | No | LSTM, transformer or SSM |
| Mirowski et al. 2017 | G1 + G4 | Depth and loop-closure labels | none | Velocity | LSTM |
| MERLIN | G3 by design | Gradient stop between policy and memory; memory trained by prediction | none | Velocity | Memory matrix |
| Neural Map, CMP, EgoMap, SMT | G1 or staged, with oracle geometry | Writes or warps use given position or ego-motion | SMT embeddings frozen | Yes | Map or memory |
| Active Neural SLAM | G3 + G4 | Map and pose supervised; Fast Marching planner | Modular | Sensor pose | Explicit map |
| BEV-Planner (base) | G1 | none (trajectory L1 only) | none stated | No | 4 BEV frames, no alignment |
| UniAD, VAD | G4 (+ G3 for UniAD) | Detection, map, motion, occupancy labels; UniAD has a collision optimiser at inference | UniAD two stages | Ego status via BEVFormer | BEVFormer |
| EMMA | G1 for planning; G4 for the generalist model | Min-L2 selection among sampled trajectories | Gemini fine-tuned | Ego history as text | Up to 4 frames |
| Alpamayo-R1 | G1 + G2 | Reasoning trained by GRPO (G2); action expert stop-gradients the VLM cache | Three stages | Egomotion history | 2 s |
| comma policy | Policy G1; system G3 | Classical ACC for longitudinal control; world model conditioned on Kalman-filter poses | FastViT frozen; teacher to student | Previous actions; world model uses poses | 2 s |
| GameNGen | G1 | none | Data from a separate PPO agent; decoder fine-tuned separately | No | 64 frames |
| DIAMOND | G1 | none (pixel space, no tokenizer) | Agent trained separately by REINFORCE in imagination | No | 4 frames |
| Matrix-Game 2.0 | G1 | Fixed 6-frame KV window | Distillation stages | No | 6 latent frames |
| Dreamer 4 | World model G1; agent heads only | Policy signal reaches neither tokenizer nor dynamics | Tokenizer trained first and frozen | No | 192 frames |
| Po et al. 2025 | G1 | Fixed local KV window | VAE separate | Maze actions include positions | SSM state plus 10 frames |
| Infinite-World | G1 | Action labels from VGGT poses, discretised by thresholds | Two stages | No pose in memory | Compressed, 1,000 frames |
| WorldMem | G3 | Greedy top-k retrieval by field-of-view overlap | Built on Oasis | Engine pose | Memory bank |
| Context-as-Memory | G3 | Rule-based field-of-view retrieval | n/a | Engine pose | 20 retrieved frames |
| Genie 3 | undetermined | Blog only | undetermined | none mentioned | about 1 minute (claim) |
| RT-1, OpenVLA, Octo, π0, Diffusion Policy | G1 | none | Octo and RT-1 frozen language encoders only | Proprioception (π0, Diffusion Policy) | 0 to 6 frames |
| π0.5 | G1 + G4 (extended) | Subtask text and boxes supervised by labels; no gradient through sampled text | Two stages | Proprioception incl. base velocity | none |
| GR00T N1 | G1 + G4 | Auxiliary detection loss on OWL-v2 boxes | Language model frozen | Proprioception | none |
| Gemini Robotics 1.5 | G3 | Orchestrator sends text to the VLA | Two models | not disclosed | thinking traces |
| MEM | Mixed G1 and G3 | Language memory written by a high-level policy; action loss blocked from the backbone | Pre- and post-training | Proprioception | 6 to 18 frames plus text |
| MemER | G3 | Keyframe selection by clustering; low-level π0.5 trained separately | Vision encoder frozen | Proprioception (low level) | up to 8 keyframes |
| RoboTTT | G1 | none (truncated BPTT) | TTT layers only in pretraining; all in post-training | Proprioception | up to 8K steps |
| MemBodied | G1 | Memory value path stop-gradient at patch tokens | LoRA or full fine-tuning | Proprioception | Delta-rule state, 6 to 18 calls |
| Dual-Stream Memory | G3 + G4 (extended) | Compressor pretrained on progress labels then frozen; threshold retriever | Frozen compressor and encoders | Proprioception | One token plus retrieved frames |
| MemoryVLA, HAMLET, CronusVLA | G1, with caveats | MemoryVLA argmax consolidation; CronusVLA stop-gradient on history; HAMLET G1 only in its CogACT variant | HAMLET freezes the VLM on GR00T | Some proprioception | 4 to 256 entries |
| SAM2Act, SpatialVLA, MoManipVLA | G3 | Point-cloud rendering and a motion planner (SAM2Act); frozen ZoeDepth path (SpatialVLA); ESDF and optimiser (MoManipVLA) | Frozen modules | Depth, intrinsics or full poses | FIFO or none |
| Evo-0, Spatial Forcing | G1 + G4 via VGGT | Frozen VGGT features (Evo-0); alignment loss to VGGT features (Spatial Forcing) | VGGT frozen or distilled | Proprioception | none |
| CUT3R | G4 | Ground-truth pointmaps and poses | Four stages | No | Recurrent state, up to 64 views |
| RayZer, RUST | G1 | none; photometric loss only | none | No | Few views, not action-conditioned |

The audit relied on method sections read through arXiv HTML. Some long
papers were read only in part (ViNT, PoliFormer, NaVid, Uni-NaVid,
NavFoM, Wijmans, EMMA, Alpamayo-R1, UniAD, DIAMOND, RUST, Gemini
Robotics 1.5, π0.5). Freezing of pretrained encoders is often unstated.

### 4.2 What survives

- **Of 63 systems audited, about 42 have a G1 core** (several with
  caveats about frozen encoders or unstated gradient flow). Twenty are
  G3 or G4, and Genie 3 is undetermined. Alpamayo-R1's GRPO stage is the
  only G2 case; no robot policy in the audit uses G2.
- **The survivors fall into three groups.** First, short-memory
  policies and world models (VLAs with 0 to 6 frames, GameNGen,
  DIAMOND, Matrix-Game, GNM and ViNT cores). They need no ego-motion
  composition because they keep almost no history. Second, survivors
  given pose or ego-motion as input (Wijmans, EMMA, Alpamayo-R1, CMP,
  EgoMap, Neural Map, Po et al. on Memory Maze). They hand composition
  to the input. Third, a small set with long memory and no pose input:
  PoliFormer, DD-PPO RGB-only, R2I and GTrXL on Memory Maze, Infinite-World,
  RoboTTT and MemBodied. Of these, DD-PPO RGB-only reaches 0.15 SPL,
  R2I stays below human on the largest maze, PoliFormer degrades after
  four rooms, Infinite-World is research scale with pose-derived action
  labels, and RoboTTT and MemBodied are manipulation with a mostly fixed
  base.
- **Deployed systems drop a grade.** GNM, ViNT and NoMaD rely on a
  topological graph and a planner; NaVILA on a parsed command and a
  separate locomotion policy; StreamVLN on depth-and-odometry pruning;
  comma on classical longitudinal control.
- **The junction the survivors still hand off is ego-motion
  composition.** Every system that is both strictly end to end and
  reliable either keeps short memory or is given pose. None we found
  composes ego-motion from pixels and actions over a long horizon at
  scale.

## 5. Counter-evidence: where geometry is learned end to end

The evidence below is real and it narrows the claim. It does not show a
C4 system holding a spatial frame over a long history at scale.

- **Registration is learnable by attention at scale (C3).** DUSt3R,
  MASt3R, VGGT, π³ and MapAnything regress poses and pointmaps from
  images in one pass. VGGT predicts cameras, depth and tracks for up to
  hundreds of frames. This shows the visual-odometry step does not need
  a hand-built module. It does not show memory: these models are trained
  with ground-truth poses, depth or pointmaps, and their outputs feed
  explicit global alignment (DUSt3R), optional bundle adjustment (VGGT)
  or SLAM back ends (MASt3R-SLAM, VGGT-SLAM).
- **A recurrent 3D state exists, but drifts (C3 with a latent state).**
  CUT3R keeps a persistent set of state tokens, updated by cross-attention,
  and outputs world-frame pointmaps and poses. It is the closest 3D
  analogue of our setting. It is supervised with ground-truth geometry,
  trained on at most 64 views, and the authors say it "may eventually
  drift over very long sequences". TTT3R recasts its update as a
  delta-rule test-time-training step with a sigmoid rate and reports it
  "mitigates but does not resolve state forgetting".
- **Pose learned from pixels alone, at modest scale (C4).** RUST learns a
  latent pose "trained on RGB images alone". RayZer predicts SE(3) cameras
  with only a photometric loss and beats COLMAP-posed LVSM on DL3DV
  (24.36 against 23.69 PSNR) and RealEstate10K. XFactor learns
  transferable latent poses with no SE(3) parameterisation. Limits:
  RayZer's pose probe gets 47.6% within 10° on large baselines, and
  shuffling frames drops PSNR from 24.36 to 20.56, so it leans on video
  order. These are the strongest results against the blocker, and they
  are view synthesis over a few frames, not memory over a trajectory.
- **Maps emerge in recurrent memory.** Wijmans et al. decode occupancy
  maps from a blind agent's LSTM and transplant its memory to take
  shortcuts. Pose was an input, so composition was not learned.
- **Short-horizon end-to-end works.** PoliFormer, SPOC, π0.5 (mobile
  base, no odometry input) and comma's 2 s vision-only lateral policy,
  deployed to 500 users, run with no geometric module. The blocker
  bites only when the horizon is long enough that ego-motion must be
  composed.
- **Explicit warps are not always crucial.** BEVFormer loses only 0.007
  NDS without ego alignment. StreamPETR reports that extra explicit motion
  compensation does not help while a pose-conditioned LayerNorm does.
  The pose is still an input in both.
- **Emergent consistency claims.** Genie 3's blog calls consistency over
  "several minutes", with about a minute of visual memory, "an emergent
  capability", in contrast to methods with "an explicit 3D
  representation". This is a **claim**. There is no technical report.
  Infinite-World reports loop closure over 1000 frames with a pose-free
  memory, at research scale.

**Industry "end-to-end" claims, by category.**

| Organisation | Primary source | What it reveals | Category |
| --- | --- | --- | --- |
| Tesla | FSD v12 release-note text (mirrored; tesla.com returned 403); NeurIPS 2025 expo listing | "single end-to-end neural network", "pixels-to-actuation" | Undetermined. Nothing on pose, odometry or memory |
| Wayve | Technology page; LINGO-2 blog; GAIA-1/2 papers | "single neural network", no HD maps; LINGO-2 takes consecutive frames plus speed | Undetermined for the driver; GAIA world models are C1 |
| Waymo | EMMA paper; blog Dec 2025 | EMMA takes past ego waypoints as text, up to 4 frames. Production keeps "materialized structured representations" | EMMA: C1. Production: C3 at least |
| NVIDIA | Alpamayo-R1 paper | Egomotion history as input tokens, 2 s history | C1 |
| comma.ai | Goff et al. 2025 | Vision-only policy over 2 s at runtime; poses from a Kalman filter drive the simulator and labels | Runtime C4 at short horizon; training C1 + C2 |
| Li Auto (with Tsinghua) | DriveVLM paper | Ego pose in the prompt; 3D detector boxes matched into the VLM; conventional planner refines | C1 + C3 |

**Verdict on the counter-evidence.** Registration and short-horizon
control are learnable end to end, so "cannot dismantle" is too strong for
those steps. Long-horizon composition and persistent spatial memory are
still handled by C1 or C2 in every deployed or benchmarked system we
found, and every C4 attempt at long horizon reports drift, forgetting or
loss of consistency.

## 6. Has the group-structured recurrence theory reached practice?

| Where | System | Recurrence or encoding | Rotation-capable? | Source |
| --- | --- | --- | --- | --- |
| LLM | Qwen3-Next 80B-A3B | Gated DeltaNet : gated attention, 3 : 1 | No. `beta = b.sigmoid()` | HF model card and modelling code |
| LLM | Qwen3.5 397B-A17B (multimodal) | Gated DeltaNet : gated attention, 3 : 1 | No. `beta = b.sigmoid()` | HF model card and modelling code |
| LLM | Kimi Linear 48B-A3B | Kimi Delta Attention : MLA, 3 : 1 | No. β is a sigmoid | arXiv 2510.26692; HF code |
| LLM | Olmo Hybrid 7B | Gated DeltaNet : attention, 3 : 1 | **Yes.** `linear_allow_neg_eigval: true`; β × 2 | arXiv 2604.03444; HF config |
| LLM | RWKV-7 | Generalised delta rule | Partly. Eigenvalues in [−1, 1] in principle; released models allow "only part of the range" | arXiv 2503.14456 |
| LLM | MiniMax-M2 | Went back to full attention after lightning attention in MiniMax-01 | n/a | MiniMax blog, Oct 2025 (claim) |
| Robot policy | MemBodied (on π0 / π0.5) | Gated delta rule memory | No. Sigmoid gates | arXiv 2609.28256 |
| Robot policy | Dual-Stream Memory | Gated DeltaNet compresses history to one token | Not stated | arXiv 2610.13262 |
| Robot policy | RoboTTT (GR00T N1.7) | TTT fast weights; Gated DeltaNet as baseline | Not stated for the baseline | arXiv 2607.15275 |
| Robot policy | LRAM | xLSTM, sLSTM non-diagonal recurrence | Not rotation-specific | arXiv 2410.22391 |
| Navigation policy | LANav (HM3D-OVON) | DeltaNet and Gated DeltaNet backbones, beat transformer and GRU | No. β = sigmoid | arXiv 2607.18794 |
| RL policy (not embodied) | FFM | Complex diagonal decay | Yes, per channel | arXiv 2310.04128 |
| RL policy (not embodied) | Complex-valued memories | Unitary RNNs from Householder reflections and 2×2 rotations | Yes | arXiv 2609.38598 |
| World model | Po et al. 2025 | Mamba block-wise scan plus local attention | No (Mamba decays are positive; not discussed in the paper) | arXiv 2505.20171 |
| 3D | TTT3R | Delta-rule view of CUT3R's state update, sigmoid rate | No | arXiv 2509.26645 |
| Position encoding | GTA, PRoPE | Pose group element inside attention | Rotates, but poses are given | arXiv 2310.10375; 2507.10496 |
| Position encoding | STRING | Learned Lie-group RoPE on 3D coordinates; tested on a real KUKA arm | Rotates, coordinates from depth | arXiv 2502.02562 |
| Position encoding | PaTH | Data-dependent Householder products | Yes, but only tested on language and synthetic tasks | arXiv 2505.16381 |

Points that matter for this study:

- **The default in the main library is off.** flash-linear-attention's
  Gated DeltaNet layer has `allow_neg_eigval: bool = False`. Anyone who
  uses it as a drop-in, as RoboTTT does, gets the
  non-rotating version unless they flip it. RoboTTT's paper does not
  say which it used.
- **In navigation, the delta rule already wins without rotation.** LANav
  holds everything fixed and finds Gated DeltaNet at 34.6 average SR
  against 30.1 for a transformer and 26.7 for a GRU on HM3D-OVON, with β
  in (0, 1). Whether β in (0, 2) adds to this is the obvious next test.
- **The one direct manipulation comparison is negative for Gated DeltaNet.**
  RoboTTT replaces its TTT layers with Gated DeltaNet at matched size:
  56% against 79% average, and "GDN does not benefit from longer
  context". The authors suggest the linear associative update "struggles
  to extract structure from dense, repetitive robot streams". Their
  tasks are manipulation, not ego-motion.
- **In language, negative eigenvalues buy state tracking but little
  perplexity.** Olmo Hybrid reports that removing them hurts synthetic
  state tracking but changes bits per byte by about 0.01 at 60M to 1B
  scale. The benefit is task-specific, which is consistent with a
  spatial benefit that language benchmarks would not show.
- **Vision-language backbones now ship with delta-rule layers.** Qwen3.5
  is natively multimodal and 3:1 Gated DeltaNet. VLAs built on such a
  backbone would inherit the layer, with β in (0, 1). Our reading: this
  makes switching β to (0, 2) cheap to test, not yet a result.
- **Nothing we found uses a rotation-capable recurrence, or PaTH, in an
  embodied, 3D or world-model system.** Group-structured encodings in
  those systems all consume external poses or coordinates.

## 7. Opportunities, ranked

Ranked by practical impact times feasibility for a small team. Each
replaces a C1 or C2 component with a C4 operator that can represent the
motion group.

| Rank | Junction to replace | Current fix | Why here | Feasibility |
| --- | --- | --- | --- | --- |
| 1 | World-model memory on revisits | Pose-indexed retrieval (C1) or 3D cache (C2) | Failure is documented and benchmarked; fix needs engine poses; predicted poses cost 2.2 dB; real-video poses are noisy | High. Public benchmarks (Memory Maze, LoopNav), about 200M-parameter models, synthetic data |
| 2 | Navigation memory, then path integration without GPS+compass | Learned VO plus explicit SE(3) composition (C3) | An operator-only ObjectNav comparison already exists (LANav) and delta rules win there; closest to our toy | High in simulation |
| 3 | VLA memory over base motion in mobile manipulation | None (π0.5), language memory (MEM), or tracking camera plus ESDF (MoManipVLA) | Stated limitation of π0.5; delta-rule memories already exist with β ≤ 1 | Medium. Needs robot data or a new simulated task |
| 4 | Streaming 3D reconstruction state | CUT3R / TTT3R recurrent state with drift | Drift and forgetting are measured; update is already delta-rule-like | Medium. Still C3 supervision |
| 5 | Driving BEV temporal fusion | Ego-pose warp (C1 + C2) | Warp is cheap and accurate given localisation; ablations show small NDS effect | High, but low impact |

**1. World-model memory.** Take the Po et al. setup (Memory Maze and
Minecraft, about 200M parameters, block-wise scan plus local attention)
but replace its Memory Maze inputs, which include the agent's
position, with relative actions only. Then change only the recurrence: Mamba2, Gated DeltaNet with β in (0, 1],
Gated DeltaNet with β in (0, 2), DeltaProduct with two factors. Add
full-context attention as an upper bound and a WorldMem-style model with
engine poses as a C1 ceiling. Train on loops; test on LoopNav ABA and
ABCA loops and on held-out loop lengths beyond the training horizon.
Report return-leg PSNR and LPIPS against loop length, a linear probe of
pose from the state, and the transition eigenvalue probe used in this
study. Infinite-World's pose-free compressed memory is the strongest
existing pose-free baseline. A positive result is: the β-to-2 model
matches the pose-given ceiling on revisits, the β-to-1 model does not,
and the gap grows with loop length.

**2. Navigation: flip β in an existing operator comparison, then remove
GPS+compass.** The fastest test reuses LANav's HM3D-OVON setup, where
everything but the backbone is already fixed and Gated DeltaNet already
beats the transformer. Add Gated DeltaNet with β in (0, 2) and
DeltaProduct, and report a history ablation (last k frames) for each.
Then move to Habitat PointNav in the 2021
challenge setting (RGB-D, actuation and sensor noise, no GPS+compass),
where the best published approach composes learned VO explicitly (94%
success, Partsey et al.). Train a single recurrent policy with no VO
module and no pose input, swapping the recurrence: LSTM, Gated DeltaNet
β ≤ 1, β ≤ 2, and a transformer over the episode. Run a blind variant
fed only the previous action, the Wijmans setting minus GPS and compass.
Measure success, SPL, a decoded-map probe as in Wijmans et al., and a
pose probe. Then multi-room ObjectNav, where PoliFormer degrades after
four rooms. Because RL adds credit-assignment and reward confounds
(section 3.3), pair it with a self-supervised auxiliary that predicts
the next observation or the start-frame view, and report the operator
comparison with and without it. DD-PPO's 0.15 SPL without GPS+compass
at 2.5B steps is the reference for the pose-free RL baseline.

**3. Mobile-manipulation memory.** Add a β-to-2 delta-rule memory
(MemBodied-style) to π0 or π0.5 and test tasks where the target leaves
view during base rotation and must be found again. The memory benchmarks
in the papers we checked (MemoryBench, MIKASA-Robo, RMBench, RoboMME) do
not test memory over base ego-motion, so this needs a new task set,
built in simulation first. Compare against MoManipVLA-style explicit
base pose as a ceiling.

**4. Streaming 3D state.** Replace CUT3R's cross-attention state update,
or TTT3R's sigmoid-rate update, with a β-to-2 delta rule and measure
pose drift on sequences much longer than the 64-view training length.
This keeps 3D supervision, so it tests memory, not the full C4 claim.

**5. Driving.** Remove the BEV warp, feed CAN-bus deltas as tokens and
let a rotation-capable recurrence carry the frame. Evaluate closed-loop
on Bench2Drive, where temporal cross-attention is worth about 4 driving
score points for DriveTransformer. Low priority: the warp works and the
pose it needs is already available.

### What a convincing scaled demonstration must show

1. **Continuous motion with noise.** Continuous SE(2), then SE(3), with
   noisy relative odometry, and a vision-only variant. The toy uses an
   exact Z4 heading group.
2. **Long horizons with revisits.** Hundreds to thousands of steps, with
   loops, scored on the return leg.
3. **A pose-given ceiling and a pose-free baseline.** Match a C1 model
   given true poses, and beat full attention and positive-decay
   recurrences at equal compute.
4. **A one-change ablation of the β range** across seeds and at least
   three model sizes, so the effect is attributed to the rotation, not
   to the delta rule in general. RoboTTT shows the delta rule in general
   can fail.
5. **Length generalisation** past the training horizon, where RoPE-based
   caches are known to fail (WorldTrace).
6. **Mechanistic evidence** that the state rotates with the agent, as the
   transition probe showed in the toy.
7. **Controls for non-operator causes.** Hold the objective, RL
   algorithm and data fixed across operators, and show the gain survives
   a dense self-supervised objective, so it is not a credit-assignment
   or reward effect (section 3.3).
8. **A closed-loop task gain**, not only reconstruction metrics, and at
   least one real-data result, since estimated poses on real video are
   the stated weakness of C1 fixes.

## 8. Bibliography

All entries were opened on 2026-10-15. Where the arXiv page did not
state the venue, the venue is given as "(venue not confirmed)".

**Driving**

- Li, Zhiqi et al. BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers. ECCV 2022. https://arxiv.org/abs/2203.17270 ; code https://github.com/fundamentalvision/BEVFormer
- Huang, Junjie; Huang, Guan. BEVDet4D: Exploit Temporal Cues in Multi-camera 3D Object Detection. arXiv 2203.17054, 2022. https://arxiv.org/abs/2203.17054
- Park, Jinhyung et al. Time Will Tell: New Outlooks and A Baseline for Temporal Multi-View 3D Object Detection. ICLR 2023. https://arxiv.org/abs/2210.02443
- Wang, Shihao et al. Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection (StreamPETR). ICCV 2023. https://arxiv.org/abs/2303.11926
- Hu, Yihan et al. Planning-oriented Autonomous Driving (UniAD). CVPR 2023. https://arxiv.org/abs/2212.10156
- Jiang, Bo et al. VAD: Vectorized Scene Representation for Efficient Autonomous Driving. ICCV 2023. https://arxiv.org/abs/2303.12077
- Jiang, Bo et al. VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning. arXiv 2402.13243. https://arxiv.org/abs/2402.13243
- Li, Zhiqi et al. Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving? CVPR 2024. https://arxiv.org/abs/2312.03031
- Zhai, Jiang-Tian et al. Rethinking the Open-Loop Evaluation of End-to-End Autonomous Driving in nuScenes. arXiv 2305.10430, 2023. https://arxiv.org/abs/2305.10430
- Sun, Wenchao et al. SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation. arXiv 2405.19620 (venue not confirmed). https://arxiv.org/abs/2405.19620 ; code https://github.com/swc-17/SparseDrive
- Lin, Xuewu et al. Sparse4D v2: Recurrent Temporal Fusion with Sparse Model. arXiv 2305.14018. https://arxiv.org/abs/2305.14018
- Jia, Xiaosong et al. DriveTransformer: Unified Transformer for Scalable End-to-End Autonomous Driving. ICLR 2025. https://arxiv.org/abs/2503.07656
- Liao, Bencheng et al. DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving. CVPR 2025. https://arxiv.org/abs/2411.15139
- Guo, Mingzhe et al. FlowAD: Ego-Scene Interactive Modeling for Autonomous Driving. arXiv 2603.13399, 2026 (cited for its UniAD temporal-fusion ablation). https://arxiv.org/abs/2603.13399
- Li, Yingyan et al. Enhancing End-to-End Autonomous Driving with Latent World Model (LAW). arXiv 2406.08481. https://arxiv.org/abs/2406.08481
- Hwang, Jyh-Jing et al. (Waymo). EMMA: End-to-End Multimodal Model for Autonomous Driving. arXiv 2410.23262. https://arxiv.org/abs/2410.23262
- Waymo. Demonstrably Safe AI For Autonomous Driving. Blog, Dec 2025 (claim). https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving/
- Tesla. FSD Beta v12 release note, update 2023.44.30 (claim). Text read via third-party mirrors (notateslaapp.com, driveteslacanada.ca); tesla.com returned 403, so no primary URL.
- Kurek, Daniel (Tesla). Building Foundational Models for Robotics at Tesla. NeurIPS 2025 expo listing (claim). https://nips.cc/virtual/2025/128654
- Hu, Anthony et al. (Wayve). GAIA-1: A Generative World Model for Autonomous Driving. arXiv 2309.17080. https://arxiv.org/abs/2309.17080
- Russell, Lloyd et al. (Wayve). GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving. arXiv 2503.20523. https://arxiv.org/abs/2503.20523
- Wayve. LINGO-2: Driving with Natural Language (claim). https://wayve.ai/thinking/lingo-2-driving-with-language/ ; Technology page (claim) https://wayve.ai/technology/
- Wang, Yan et al. (NVIDIA). Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail. arXiv 2511.00088. https://arxiv.org/abs/2511.00088
- Li, Zhenxin et al. Hydra-MDP: End-to-end Multimodal Planning with Multi-target Hydra-Distillation. arXiv 2406.06978. https://arxiv.org/abs/2406.06978
- Goff, Mitchell et al. (comma.ai). Learning to Drive from a World Model. arXiv 2504.19077. https://arxiv.org/abs/2504.19077
- Tian, Xiaoyu et al. (Tsinghua, Li Auto). DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models. arXiv 2402.12289. https://arxiv.org/abs/2402.12289
- Dauner, Daniel et al. NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking. NeurIPS 2024 D&B. https://arxiv.org/abs/2406.15349
- Jia, Xiaosong et al. Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving. NeurIPS 2024 D&B. https://arxiv.org/abs/2406.03877

**Navigation and SLAM**

- Shah, Dhruv et al. GNM: A General Navigation Model to Drive Any Robot. ICRA 2023. https://arxiv.org/abs/2210.03370
- Shah, Dhruv et al. ViNT: A Foundation Model for Visual Navigation. CoRL 2023. https://arxiv.org/abs/2306.14846
- Sridhar, Ajay et al. NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration. arXiv 2310.07896 (venue not confirmed). https://arxiv.org/abs/2310.07896
- Wijmans, Erik et al. DD-PPO: Learning Near-Perfect PointGoal Navigators from 2.5 Billion Frames. arXiv 1911.00357 (venue not confirmed). https://arxiv.org/abs/1911.00357
- Wijmans, Erik et al. Emergence of Maps in the Memories of Blind Navigation Agents. ICLR 2023. https://arxiv.org/abs/2301.13261
- Datta, Samyak et al. Integrating Egocentric Localization for More Realistic Point-Goal Navigation Agents. arXiv 2009.03231 (venue not confirmed). https://arxiv.org/abs/2009.03231
- Partsey, Ruslan et al. Is Mapping Necessary for Realistic PointGoal Navigation? arXiv 2206.00997 (venue not confirmed). https://arxiv.org/abs/2206.00997
- Zeng, Kuo-Hao et al. PoliFormer: Scaling On-Policy RL with Transformers Results in Masterful Navigators. CoRL 2024. https://arxiv.org/abs/2406.20083
- Ehsani, Kiana et al. SPOC: Imitating Shortest Paths in Simulation Enables Effective Navigation and Manipulation in the Real World. CVPR 2024. https://arxiv.org/abs/2312.02976
- Zhang, Jiazhao et al. NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation. arXiv 2402.15852 (venue not confirmed). https://arxiv.org/abs/2402.15852
- Zhang, Jiazhao et al. Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks. arXiv 2412.06224 (venue not confirmed). https://arxiv.org/abs/2412.06224
- Cheng, An-Chieh et al. NaVILA: Legged Robot Vision-Language-Action Model for Navigation. arXiv 2412.04453 (venue not confirmed). https://arxiv.org/abs/2412.04453
- Wei, Meng et al. StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling. arXiv 2507.05240. https://arxiv.org/abs/2507.05240
- Zhang, Jiazhao et al. Embodied Navigation Foundation Model (NavFoM). arXiv 2509.12129. https://arxiv.org/abs/2509.12129
- Chiang, Hao-Tien Lewis et al. (Google DeepMind). Mobility VLA: Multimodal Instruction Navigation with Long-Context VLMs and Topological Graphs. arXiv 2407.07775. https://arxiv.org/abs/2407.07775
- Huang, Chenguang et al. Visual Language Maps for Robot Navigation (VLMaps). ICRA 2023. https://arxiv.org/abs/2210.05714
- Gu, Qiao et al. ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning. arXiv 2309.16650. https://arxiv.org/abs/2309.16650
- Liu, Peiqi et al. OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robotics. arXiv 2401.12202. https://arxiv.org/abs/2401.12202
- Yang, Yuncong et al. 3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning. arXiv 2411.17735. https://arxiv.org/abs/2411.17735
- Zhang, Lingfeng et al. MapNav: A Novel Memory Representation via Annotated Semantic Maps for Vision-and-Language Navigation. arXiv 2502.13451. https://arxiv.org/abs/2502.13451
- Zhu, Weiye et al. NaVIDA: Vision-Language Navigation with Inverse Dynamics Augmentation. arXiv 2601.18188, 2026. https://arxiv.org/abs/2601.18188
- Chaplot, Devendra Singh et al. Learning to Explore using Active Neural SLAM. arXiv 2004.05155. https://arxiv.org/abs/2004.05155
- Teed, Zachary; Deng, Jia. DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras. arXiv 2108.10869. https://arxiv.org/abs/2108.10869
- Teed, Zachary; Lipson, Lahav; Deng, Jia. Deep Patch Visual Odometry. arXiv 2208.04726. https://arxiv.org/abs/2208.04726
- Murai, Riku et al. MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors. CVPR 2025. https://arxiv.org/abs/2412.12392
- Maggio, Dominic et al. VGGT-SLAM: Dense RGB SLAM Optimized on the SL(4) Manifold. arXiv 2505.12549. https://arxiv.org/abs/2505.12549

**Manipulation policies and VLAs**

- Brohan, Anthony et al. RT-1: Robotics Transformer for Real-World Control at Scale. arXiv 2212.06817. https://arxiv.org/abs/2212.06817
- Kim, Moo Jin et al. OpenVLA: An Open-Source Vision-Language-Action Model. arXiv 2406.09246. https://arxiv.org/abs/2406.09246
- Octo Model Team. Octo: An Open-Source Generalist Robot Policy. arXiv 2405.12213. https://arxiv.org/abs/2405.12213
- Black, Kevin et al. (Physical Intelligence). π0: A Vision-Language-Action Flow Model for General Robot Control. arXiv 2410.24164. https://arxiv.org/abs/2410.24164
- Pertsch, Karl et al. FAST: Efficient Action Tokenization for Vision-Language-Action Models. arXiv 2501.09747. https://arxiv.org/abs/2501.09747
- Physical Intelligence. π0.5: a Vision-Language-Action Model with Open-World Generalization. arXiv 2504.16054. https://arxiv.org/abs/2504.16054
- Gemini Robotics Team (Google DeepMind). Gemini Robotics: Bringing AI into the Physical World. arXiv 2503.20020. https://arxiv.org/abs/2503.20020
- Google DeepMind. Gemini Robotics 1.5. arXiv 2510.03342. https://arxiv.org/abs/2510.03342
- NVIDIA. GR00T N1: An Open Foundation Model for Generalist Humanoid Robots. arXiv 2503.14734. https://arxiv.org/abs/2503.14734
- Liu, Songming et al. RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation. arXiv 2410.07864. https://arxiv.org/abs/2410.07864
- Wang, Lirui et al. Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers (HPT). arXiv 2409.20537. https://arxiv.org/abs/2409.20537
- Chi, Cheng et al. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. RSS 2023. https://arxiv.org/abs/2303.04137
- de Haan, Pim; Jayaraman, Dinesh; Levine, Sergey. Causal Confusion in Imitation Learning. NeurIPS 2019. https://arxiv.org/abs/1905.11979
- Wen, Chuan et al. Fighting Copycat Agents in Behavioral Cloning from Observation Histories. NeurIPS 2020. https://arxiv.org/abs/2010.14876
- Codevilla, Felipe et al. Exploring the Limitations of Behavior Cloning for Autonomous Driving. ICCV 2019. https://arxiv.org/abs/1904.08980
- Torne, Marcel et al. Learning Long-Context Diffusion Policies via Past-Token Prediction. arXiv 2505.09561. https://arxiv.org/abs/2505.09561
- Koo, Myungkyu et al. HAMLET: Switch your Vision-Language-Action Model into a History-Aware Policy. ICLR 2026. https://arxiv.org/abs/2510.00695
- Li, Hao et al. CronusVLA: Towards Efficient and Robust Manipulation via Multi-Frame Vision-Language-Action Modeling. AAAI 2026. https://arxiv.org/abs/2506.19816
- Shi, Hao et al. MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation. ICLR 2026. https://arxiv.org/abs/2508.19236
- Fang, Haoquan et al. SAM2Act: Integrating Visual Foundation Model with A Memory Architecture for Robotic Manipulation. ICML 2025. https://arxiv.org/abs/2501.18564
- Sridhar, Ajay et al. MemER: Scaling Up Memory for Robot Control via Experience Retrieval. ICLR 2026. https://arxiv.org/abs/2510.20328
- Torne, Marcel et al. (Physical Intelligence et al.). MEM: Multi-Scale Embodied Memory for VLA Models. arXiv 2603.03596, 2026. https://arxiv.org/abs/2603.03596 ; blog https://www.pi.website/research/memory
- Jiang, Yunfan et al. (NVIDIA, Stanford, UT Austin). RoboTTT: Context Scaling for Robot Policies. arXiv 2607.15275, 2026. https://arxiv.org/abs/2607.15275
- Pala, Tej Deep et al. MemBodied: Recurrent Associative Memory for Vision-Language-Action Models. arXiv 2609.28256, 2026. https://arxiv.org/abs/2609.28256
- Wang, Yang et al. Dual-Stream Memory for Vision-Language-Action Policies. arXiv 2610.13262, 2026. https://arxiv.org/abs/2610.13262
- Schmied, Thomas et al. A Large Recurrent Action Model: xLSTM enables Fast Inference for Robotics Tasks. arXiv 2410.22391. https://arxiv.org/abs/2410.22391
- Jia, Xiaogang et al. MaIL: Improving Imitation Learning with Mamba. arXiv 2406.08234. https://arxiv.org/abs/2406.08234
- Qu, Delin et al. SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model. RSS 2025. https://arxiv.org/abs/2501.15830
- Sun, Lin et al. GeoVLA: Empowering 3D Representations in Vision-Language-Action Models. arXiv 2508.09071. https://arxiv.org/abs/2508.09071
- Li, Peiyan et al. BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models. NeurIPS 2025. https://arxiv.org/abs/2506.07961
- Wu, Zhenyu et al. MoManipVLA: Transferring Vision-language-action Models for General Mobile Manipulation. CVPR 2025. https://arxiv.org/abs/2503.13446
- Shah, Rutav et al. BUMBLE: Unifying Reasoning and Acting with Vision-Language Models for Building-wide Mobile Manipulation. arXiv 2410.06237. https://arxiv.org/abs/2410.06237
- Lin, Tao et al. Evo-0: Vision-Language-Action Model with Implicit Spatial Understanding. arXiv 2507.00416. https://arxiv.org/abs/2507.00416
- Li, Fuhao et al. Spatial Forcing: Implicit Spatial Representation Alignment for Vision-language-action Model. arXiv 2510.12276. https://arxiv.org/abs/2510.12276
- Abouzeid, Ali et al. GeoAware-VLA: Implicit Geometry Aware Vision-Language-Action Model. arXiv 2509.14117. https://arxiv.org/abs/2509.14117

**World models**

- Valevski, Dani et al. Diffusion Models Are Real-Time Game Engines (GameNGen). ICLR 2025. https://arxiv.org/abs/2408.14837
- Alonso, Eloi et al. Diffusion for World Modeling: Visual Details Matter in Atari (DIAMOND). NeurIPS 2024. https://arxiv.org/abs/2405.12399
- Decart and Etched. Oasis (claim). 2024. https://oasis-model.github.io/
- Google DeepMind. Genie 2: A large-scale foundation world model. Blog, Dec 2024 (claim). https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
- Google DeepMind. Genie 3: A new frontier for world models. Blog, Aug 2025 (claim). https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/
- NVIDIA. Cosmos World Foundation Model Platform for Physical AI. arXiv 2501.03575. https://arxiv.org/abs/2501.03575
- Xiao, Zeqi et al. WorldMem: Long-term Consistent World Simulation with Memory. arXiv 2504.12369. https://arxiv.org/abs/2504.12369
- Yu, Jiwen et al. Context as Memory: Scene-Consistent Interactive Long Video Generation with Memory Retrieval. SIGGRAPH Asia 2025. https://arxiv.org/abs/2506.03141
- Wu, Tong et al. Video World Models with Long-term Spatial Memory. arXiv 2506.05284. https://arxiv.org/abs/2506.05284
- Li, Runjia et al. VMem: Consistent Interactive Video Scene Generation with Surfel-Indexed View Memory. ICCV 2025. https://arxiv.org/abs/2506.18903
- Ren, Xuanchi et al. (NVIDIA). GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control. CVPR 2025. https://arxiv.org/abs/2503.03751
- Po, Ryan et al. Long-Context State-Space Video World Models. arXiv 2505.20171. https://arxiv.org/abs/2505.20171
- Dalal, Karan et al. One-Minute Video Generation with Test-Time Training. CVPR 2025. https://arxiv.org/abs/2504.05298
- Zhang, Tianyuan et al. Test-Time Training Done Right (LaCT). arXiv 2505.23884. https://arxiv.org/abs/2505.23884
- He, Hao et al. CameraCtrl: Enabling Camera Control for Text-to-Video Generation. arXiv 2404.02101. https://arxiv.org/abs/2404.02101
- He, Xianglong et al. Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model. arXiv 2508.13009. https://arxiv.org/abs/2508.13009
- Hafner, Danijar; Yan, Wilson; Lillicrap, Timothy. Training Agents Inside of Scalable World Models (Dreamer 4). arXiv 2509.24527. https://arxiv.org/abs/2509.24527
- Huang, Junchao et al. Memory Forcing: Spatio-Temporal Memory for Consistent Scene Generation on Minecraft. arXiv 2510.03198. https://arxiv.org/abs/2510.03198
- Hong, Yicong et al. RELIC: Interactive Video World Model with Long-Horizon Memory. arXiv 2512.04040. https://arxiv.org/abs/2512.04040
- Sun, Wenqiang et al. (Tencent Hunyuan). WorldPlay: Towards Long-Term Geometric Consistency for Real-Time Interactive World Modeling. ICML 2026. https://arxiv.org/abs/2512.14614
- Wu, Ruiqi et al. Infinite-World: Scaling Interactive World Models to 1000-Frame Horizons via Pose-Free Hierarchical Memory. arXiv 2602.02393, 2026. https://arxiv.org/abs/2602.02393
- Wu, Xindi et al. Addressable Memory for Video World Models (WorldTrace). arXiv 2608.07408, 2026. https://arxiv.org/abs/2608.07408
- Assran, Mido et al. (Meta). V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. arXiv 2506.09985. https://arxiv.org/abs/2506.09985
- Pasukonis, Jurgis; Lillicrap, Timothy; Hafner, Danijar. Evaluating Long-Term Memory in 3D Mazes (Memory Maze). arXiv 2210.13383. https://arxiv.org/abs/2210.13383
- Lian, Kewei et al. LoopNav: Benchmarking Spatial Consistency in World Models. arXiv 2505.22976. https://arxiv.org/abs/2505.22976
- Ye, Yixuan et al. MIND: Benchmarking Memory Consistency and Action Control in World Models. arXiv 2602.08025, 2026. https://arxiv.org/abs/2602.08025

**Memory agents and memory benchmarks in reinforcement learning**

- Zhang, Jiahong et al. Beyond Transformers: Linear Attention Policy for Open-Vocabulary Object Goal Navigation (LANav). arXiv 2607.18794, 2026. https://arxiv.org/abs/2607.18794
- Yang, Fan et al. Spatially-Enhanced Recurrent Memory for Long-Range Mapless Navigation via End-to-End Reinforcement Learning (SRU). arXiv 2506.05997. https://arxiv.org/abs/2506.05997
- Chen, Chang et al. TransDreamer: Reinforcement Learning with Transformer World Models. arXiv 2202.09481. https://arxiv.org/abs/2202.09481
- Pramanik, Subhojeet et al. AGaLiTe: Approximate Gated Linear Transformers for Online Reinforcement Learning. TMLR 2024. https://arxiv.org/abs/2310.15719
- Wang, Shaoan et al. (Light Origins). LightNav-0: Eliciting VLM Spatial Intelligence for Generalist Embodied Navigation. arXiv 2608.30935, 2026. https://arxiv.org/abs/2608.30935 ; code https://github.com/lightorigins/LightNav-0

- Jaderberg, Max et al. Reinforcement Learning with Unsupervised Auxiliary Tasks (UNREAL). arXiv 1611.05397, 2016. https://arxiv.org/abs/1611.05397
- Mirowski, Piotr et al. Learning to Navigate in Complex Environments. arXiv 1611.03673, 2016. https://arxiv.org/abs/1611.03673
- Mirowski, Piotr et al. Learning to Navigate in Cities Without a Map. arXiv 1804.00168, 2018. https://arxiv.org/abs/1804.00168
- Wayne, Greg et al. Unsupervised Predictive Memory in a Goal-Directed Agent (MERLIN). arXiv 1803.10760, 2018. https://arxiv.org/abs/1803.10760
- Parisotto, Emilio; Salakhutdinov, Ruslan. Neural Map: Structured Memory for Deep Reinforcement Learning. arXiv 1702.08360. https://arxiv.org/abs/1702.08360
- Gupta, Saurabh et al. Cognitive Mapping and Planning for Visual Navigation. CVPR 2017. https://arxiv.org/abs/1702.03920
- Fang, Kuan et al. Scene Memory Transformer for Embodied Agents in Long-Horizon Tasks. CVPR 2019. https://arxiv.org/abs/1903.03878
- Beeching, Edward et al. EgoMap: Projective mapping and structured egocentric memory for Deep RL. arXiv 2002.02286. https://arxiv.org/abs/2002.02286
- Parisotto, Emilio et al. Stabilizing Transformers for Reinforcement Learning (GTrXL). ICML 2020. https://arxiv.org/abs/1910.06764
- Hafner, Danijar et al. Dream to Control: Learning Behaviors by Latent Imagination (DreamerV1). arXiv 1912.01603. https://arxiv.org/abs/1912.01603
- Hafner, Danijar et al. Mastering Diverse Domains through World Models (DreamerV3). arXiv 2301.04104. https://arxiv.org/abs/2301.04104
- Samsami, Mohammad Reza et al. Mastering Memory Tasks with World Models (R2I). ICLR 2024. https://arxiv.org/abs/2403.04253
- Pleines, Marco et al. Memory Gym: Towards Endless Tasks to Benchmark Memory Capabilities of Agents. arXiv 2309.17207. https://arxiv.org/abs/2309.17207
- Morad, Steven et al. POPGym: Benchmarking Partially Observable Reinforcement Learning. ICLR 2023. https://arxiv.org/abs/2303.01859
- Cherepanov, Egor et al. Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning (MIKASA). arXiv 2502.10550. https://arxiv.org/abs/2502.10550
- Lu, Chris et al. Structured State Space Models for In-Context Reinforcement Learning. NeurIPS 2023. https://arxiv.org/abs/2303.03982
- Morad, Steven et al. Reinforcement Learning with Fast and Forgetful Memory (FFM). NeurIPS 2023. https://arxiv.org/abs/2310.04128
- Ni, Tianwei et al. Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs. ICML 2022. https://arxiv.org/abs/2110.05038
- Ni, Tianwei et al. When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment. NeurIPS 2023. https://arxiv.org/abs/2307.03864
- Le, Hung et al. Stable Hadamard Memory: Revitalizing Memory-Augmented Agents for Reinforcement Learning (SHM). ICLR 2025. https://arxiv.org/abs/2410.10132
- Bhethanabhotla, Sathya Kamesh; Gavves, Efstratios; Biedenkapp, André. Reinforcement Learning with Complex (valued) Memories. arXiv 2609.38598, 2026. https://arxiv.org/abs/2609.38598

**3D foundation models and view synthesis**

- Wang, Shuzhe et al. DUSt3R: Geometric 3D Vision Made Easy. arXiv 2312.14132. https://arxiv.org/abs/2312.14132
- Leroy, Vincent et al. Grounding Image Matching in 3D with MASt3R. arXiv 2406.09756. https://arxiv.org/abs/2406.09756
- Wang, Jianyuan et al. VGGT: Visual Geometry Grounded Transformer. CVPR 2025. https://arxiv.org/abs/2503.11651
- Wang, Qianqian et al. Continuous 3D Perception Model with Persistent State (CUT3R). CVPR 2025. https://arxiv.org/abs/2501.12387
- Chen, Xingyu et al. TTT3R: 3D Reconstruction as Test-Time Training. ICLR 2026. https://arxiv.org/abs/2509.26645
- Wang, Yifan et al. π³: Permutation-Equivariant Visual Geometry Learning. arXiv 2507.13347. https://arxiv.org/abs/2507.13347
- Keetha, Nikhil et al. MapAnything: Universal Feed-Forward Metric 3D Reconstruction. 3DV 2026. https://arxiv.org/abs/2509.13414
- Zhuo, Dong et al. Streaming 4D Visual Geometry Transformer (StreamVGGT). arXiv 2507.11539. https://arxiv.org/abs/2507.11539
- Wu, Yuqi et al. Point3R: Streaming 3D Reconstruction with Explicit Spatial Pointer Memory. arXiv 2507.02863. https://arxiv.org/abs/2507.02863
- Dong, Jiacheng et al. MeMix: Writing Less, Remembering More for Streaming 3D Reconstruction. arXiv 2603.15330. https://arxiv.org/abs/2603.15330
- Jin, Haian et al. LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias. arXiv 2410.17242. https://arxiv.org/abs/2410.17242
- Sajjadi, Mehdi S. M. et al. RUST: Latent Neural Scene Representations from Unposed Imagery. CVPR 2023. https://arxiv.org/abs/2211.14306
- Jiang, Hanwen et al. RayZer: A Self-supervised Large View Synthesis Model. ICCV 2025. https://arxiv.org/abs/2505.00702
- Mitchel, Thomas W.; Ryu, Hyunwoo; Sitzmann, Vincent. True Self-Supervised Novel View Synthesis is Transferable (XFactor). ICLR 2026. https://arxiv.org/abs/2510.13063
- Ye, Botao et al. No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images (NoPoSplat). arXiv 2410.24207. https://arxiv.org/abs/2410.24207
- Smart, Brandon et al. Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs. arXiv 2408.13912. https://arxiv.org/abs/2408.13912

**Linear recurrences in production and position encodings**

- Qwen Team (Alibaba). Qwen3-Next-80B-A3B-Instruct. Model card and code, Sep 2025. https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- Qwen Team (Alibaba). Qwen3.5-397B-A17B. Model card and code, Feb 2026. https://huggingface.co/Qwen/Qwen3.5-397B-A17B
- Kimi Team (Moonshot AI). Kimi Linear: An Expressive, Efficient Attention Architecture. arXiv 2510.26692. https://arxiv.org/abs/2510.26692
- Merrill, William et al. (Ai2). Olmo Hybrid: From Theory to Practice and Back. arXiv 2604.03444, 2026. https://arxiv.org/abs/2604.03444 ; config https://huggingface.co/allenai/Olmo-Hybrid-7B
- Peng, Bo et al. RWKV-7 "Goose" with Expressive Dynamic State Evolution. arXiv 2503.14456. https://arxiv.org/abs/2503.14456
- MiniMax. MiniMax-01: Scaling Foundation Models with Lightning Attention. arXiv 2501.08313. https://arxiv.org/abs/2501.08313
- Sun, Haohai (MiniMax). Why Did M2 End Up as a Full Attention Model? Blog, Oct 2025 (claim). https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model
- fla-org. flash-linear-attention, `fla/layers/gated_deltanet.py` (`allow_neg_eigval` default False). https://github.com/fla-org/flash-linear-attention
- Yang, Songlin et al. PaTH Attention: Position Encoding via Accumulating Householder Transformations. NeurIPS 2025. https://arxiv.org/abs/2505.16381
- Miyato, Takeru et al. GTA: A Geometry-Aware Attention Mechanism for Multi-View Transformers. ICLR 2024. https://arxiv.org/abs/2310.10375
- Li, Ruilong et al. Cameras as Relative Positional Encoding (PRoPE). NeurIPS 2025. https://arxiv.org/abs/2507.10496
- Ostmeier, Sophie et al. LieRE: Lie Rotational Positional Encodings. ICML 2025. https://arxiv.org/abs/2406.10322
- Schenck, Connor et al. Learning the RoPEs: Better 2D and 3D Position Encodings with STRING. ICML 2025. https://arxiv.org/abs/2502.02562

For the theory papers (Grazzi et al., Gated DeltaNet, DeltaProduct and
others), see the bibliography of [the theory review](literature_review.md).

## 9. Not verified

These were looked for and not confirmed on a primary source. They are
not used as evidence above.

- Tesla: the Q4 2023 shareholder-deck wording and tesla.com pages (403);
  Ashok Elluswamy's ICCV 2025 talk; any technical detail on pose or
  memory.
- XPeng VLA 2.0 and Li Auto MindVLA technical details (press only).
- Genie 3 architecture (no technical report).
- How VADv2 aligns temporal features; how Alpamayo-R1 encodes egomotion
  history; whether Hydra-MDP's multi-frame LiDAR splat is
  ego-compensated.
- Whether RoboTTT's Gated DeltaNet baseline used negative eigenvalues.
- Whether Po et al. use Mamba or Mamba2 internally, and whether Memory
  Maze inputs include full camera pose.
- RT-2's and Gemini Robotics' exact history lengths.
- HOV-SG's input requirements; FLaRe's context length; X-Nav's inputs;
  official Habitat 2023 ObjectNav winners.
- Lightning-to-softmax ratio of MiniMax-01; Nemotron-H, Falcon-H1 and
  Granite 4.0 internals.
- WorldModelBench and MBench; Yume and Genie Envisioner were not checked.
- MapNet (Henriques and Vedaldi 2018) method details; how Banino et
  al. (2018) integrate the grid network into the agent; DreamerV2, and
  whether DreamerV3 actor gradients reach the world model; the Dreamer 4
  method beyond what section 4 reports; AGaLiTe and RLBenchNet methods;
  whether Active Neural SLAM blocks gradients between modules; whether
  AGaLiTe's oscillatory terms depend on the input.
- The study author's ObjectNav KV-cache ablation and LightNav-0 memory
  ablation are unpublished and reported here as personal communication.
- In the gradient audit: whether pretrained encoders are frozen in SPOC,
  NaVid, Uni-NaVid, NavFoM, GameNGen, Matrix-Game 2.0 and Po et al.;
  whether StreamVLN prunes during training; MemoryVLA's gradient path
  through memory.
- Venues marked "(venue not confirmed)" in section 8.
