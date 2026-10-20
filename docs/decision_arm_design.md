# Design note: memory-based decision making in discrete SE(2)

Status: ON HOLD (2026-10-15). DRAFT for sign-off. Nothing here is frozen or
built yet. Open questions are marked **[decide]**.

## Question

When an agent must act on something it saw long ago, under egocentric
motion with turns, does its policy use old context? Or does it fall back
on the last few observations, as reported for full-context ObjectNav
agents and navigation foundation models? And is the difference set by
the memory operator, given a dense auxiliary that teaches the memory to
hold a map?

The mapping study answered "which operator can learn to compose
ego-motion" through rendering. This arm asks whether that capability
reaches decisions.

## Approach

**The agent is the mapper plus a decision head.** It shares one sequence
model, the operator under test, between two heads:

- **Auxiliary head (the map).** Per-step relative-pose rendering, exactly
  as in the mapping study: render the window at relative pose p as of
  step t. This is spatial grounding extended across time, supervised by
  pixels. It is dense, and it is task-agnostic: it never asks about the
  goal.
- **Decision head.** Outputs the next action toward a goal the agent
  observed earlier in the episode.

Gradients from both heads reach the shared memory. The path from the
observation and action history to the decision is unbroken, so the agent
is end to end in the user's sense. The auxiliary's relative-pose labels
are used only at training time.

## Task (stage 1: supervised decision readout)

The generator is unchanged: a tile map, an egocentric 9x9 view rotated to
the agent's heading, and moves and turns in the agent's own frame.

1. **Goal.** At a controlled early step the agent sees a goal marker
   **[decide: a distinctive tile class, a marker drawn into the map, or a
   specific digit identity]**.
2. **Walk.** The agent then walks for a controlled number of steps K,
   with a controlled number of turns, so the goal is out of view.
3. **Decision.** At the decision step the target is the first action of
   a shortest path to the goal **[decide: action set; the 16 odometry
   symbols as now, or MiniGrid's left / right / forward]**. Several
   decision points per episode are possible, as with per-step queries.

The design makes recent frames insufficient: K and the number of turns
since the goal was last seen are the controlled variables.

## Stage 2: closed loop

The same agents in an environment loop with RL, plus the same auxiliary
**[decide: our own gridworld environment, or an existing benchmark such
as MiniGrid memory tasks or Memory Gym; check that their egocentric
views and turn actions match ours]**. Stage 2 is for external validity.
Credit assignment returns here as a confound, so stage 1 is the clean
test.

## Factors

- **Operator:**
  - vanilla4, all attention
  - hybrid3, positive-decay recurrence
  - hybrid4 with beta capped at 1
  - hybrid4
  - optionally, hybrid4 complex-diagonal
- **Objective:**
  - decision only
  - decision + auxiliary
  - auxiliary only, with a decision head trained on top of the frozen
    memory, as a readout probe
- **Fixed:** data, steps, size (xs first), learning rate rule, seed set.

## Metrics

1. **Decision accuracy,** as a function of K and of the number of turns
   since the goal was seen.
2. **Context-truncation ablation (primary).** At test time, drop the
   history older than the last m steps, and measure the accuracy loss as
   a function of m. A policy that uses old context loses accuracy when
   the goal sighting is truncated away. One that does not use it is
   unaffected. This is the user's ObjectNav ablation, made controlled.
3. **Auxiliary render error by prefix length,** as in the mapping study.
4. **Transition probe on the trained memory:** is the rotation present,
   and is it shared with the decision path?

## Draft predictions (to be frozen at sign-off)

- **D1.** With decision + auxiliary, hybrid4's accuracy drops sharply
  when the goal sighting is truncated away. It uses old context. vanilla4
  and hybrid4 with beta capped at 1 show a small drop at the same budget,
  so their decisions rely on recent context.
- **D2.** With decision only, every operator shows the recency pattern of
  D1's weak models: the decision signal alone does not build a map within
  the budget.
- **D3.** The auxiliary improves decision accuracy for every operator,
  most for hybrid4.
- **D4.** The frozen-memory readout probe from a hybrid4 trained on the
  auxiliary alone matches decision + auxiliary to within a small margin.
  The map is sufficient; the decision is a readout.

**Falsification criteria [decide]:** numeric thresholds for "sharp" and
"small" drops (for example relative accuracy loss of at least 50% against
at most 10%), and the budget.

## Risks and how the design meets them

- **The auxiliary leaks the answer.** It is task-agnostic by
  construction. A goal-specific auxiliary is a later variant, not the
  default.
- **The decision head ignores the memory and uses the auxiliary's
  features shortcut-style.** The truncation ablation measures what the
  decision depends on, whatever the route.
- **Interference between objectives.** Report both heads' metrics, and
  sweep the auxiliary weight **[decide: fixed weight, or a small sweep]**.
- **Scale.** Start at xs, where hybrid4 already solves the mapper. Larger
  rungs on the cluster once the kernel issue is fixed.

## External test at scale: LANav

LANav (arXiv 2607.18794, 2026) swaps only the memory backbone of an
ObjectNav agent on HM3D-OVON: SigLIP encoder, DAgger then PPO, depth and
width fixed. Average success rate:

| backbone | avg SR |
| --- | --- |
| GRU | 26.7 |
| Transformer | 30.1 |
| Linear transformer | 31.3 |
| DeltaNet | 33.9 |
| Gated DeltaNet | 34.6 |
| WSLA | 36.4 |

- **Delta rules beat attention at scale.** The transformer did not
  benefit from longer training context; the linear-attention backbones
  did.
- **Its delta rules use beta = sigmoid, in (0, 1).** That is the setting
  our ablation shows cannot rotate, so it is the hybrid4_beta1 condition
  at scale.
- **The test is cheap.** Flipping beta to (0, 2) in that setup, with
  everything else fixed, is a one-flag change on a standard benchmark.

It is the natural scaled follow-up to this arm, ideally after the
controlled stage 1. Source and verification: docs/applications_review.md.
The user's own unpublished evidence is reported as personal
communication there: full-context ObjectNav agents and LightNav-0 using
only recent frames under memory ablation.

## Engineering estimate

- **Stage 1:**
  - a goal-placement option and a decision target in the generator,
    measured in tens of lines
  - a decision head and a two-head loss in the trainer
  - the truncation evaluation

  It reuses every model. Days, not weeks.
- **Stage 2:** an environment loop and an RL learner, or adapters to an
  existing benchmark. This is larger; decide after stage 1.

## 2026-10-18 extension: feature queries, relative anchors, and policy readout

User-proposed future direction. The original decision arm above remains a draft; this extension broadens its question and does not launch work or settle the earlier open choices.

### Necessary information versus sufficient capability

Successful mapping readout is not sufficient evidence for useful downstream decision making or planning. Our current rendering interface supplies a spatial address and asks for local appearance. A model can learn to answer many such local requests without exposing a representation from which a small, efficient head can locate a feature, compare destinations, or choose a route. Even when a requested location costs only a small fixed query-token budget, searching the entire map by issuing a query for every location is a different and potentially expensive computation. The learned address/readout mechanism may retain relevant information while making other operations difficult to learn.

The motivating distinction is between necessary spatial information and a sufficient, usable representation. Saying that a task requiring a map requires mapping is tautological if “mapping” is defined as whatever spatial knowledge the task needs. More precisely, some tasks require preserving and relating particular spatial facts; this does not imply that reconstructing a full pixel map is necessary. A feature-relative pose or a goal-directed belief can sometimes suffice. Conversely, strong pixel reconstruction does not establish efficient access to those facts for a downstream head, nor the ability to perform planning over them. We should test these implications rather than use rendering success as their proxy.

Distinguish three empirical questions: is the relevant information retained; can a bounded decoder recover it; and can a policy use it successfully? A failed decoder does not prove information is absent, and a successful large decoder or exhaustive search does not establish cheap downstream access.

### Proposed query families

1. **Pattern detection and localization.** Supply a feature identity, template, or pattern specification rather than its spatial address. Return existence and the anchor pose of matching observations/features relative to the agent's current frame (or another explicitly specified anchor frame). Define the feature anchor consistently, e.g. its center plus an identifiable orientation. Test one match first, then repeated patterns and set-valued outputs. Handle rotationally symmetric patterns with symmetry-aware labels rather than arbitrary unique orientations. This tests content-to-location retrieval, complementing location-to-content rendering.
2. **Goal-conditioned policy heads.** Supply a feature descriptor and predict an action toward a previously observed instance, with the feature out of the current view. Start with supervised action or relative-waypoint targets to isolate representation/readout learning; follow with closed-loop navigation. Direct motion toward a target in an empty plane tests localization and control, not general route planning. Add obstacles, detours, alternatives, and partial observability before claiming planning capability.
3. **Queries about observation support.** Combine the above with the [partial-coverage proposal](partial_coverage_proposal.md). Distinguish a feature not observed from evidence that it is absent in an inspected region. A policy may need to report insufficient information or seek another observation. Unknown regions must not silently become free space or negative detection labels.

### Experiments that separate representation from adaptation

Use the same histories and held-out scenes to compare a frozen mapping-trained backbone with trained feature/policy heads, mapping-pretrained backbones fine-tuned on downstream tasks, jointly trained mapping/downstream heads, and downstream-only training. Include an oracle spatial-information baseline with the same downstream decision problem; its privileged information must be explicit. Evaluate reconstruction after adaptation to measure retention or tradeoffs.

Control head capacity, query-token count, training examples, and optimization budget. Sweep decoder capacity or training budget where a frozen head fails. Report exhaustive rendering-plus-search as an explicit baseline with its total query/compute cost, not as evidence that a single feature query works. Initially keep new query tokens inert so changing the query family does not silently introduce a memory-writing shortcut. Ensure descriptor inputs identify what to find without encoding the answer's location.

Use long observation-to-decision delays, varying turns and displacement, held-out layouts and feature combinations, multiple candidate matches, and unseen goals. Make the last frame insufficient where the experiment claims memory use. Include current-view-only, history truncation, shuffled motion, and target-observation removal controls. Keep invalid or unanswerable counterfactuals labeled explicitly rather than scoring them against an answer no longer supported by the history.

Report detection accuracy, false positives and missed observations, relative translation error in pixels/world units, symmetry-aware orientation error, and performance versus source age and distance. For policies report supervised action accuracy separately from closed-loop success, path efficiency, collisions, and behavior under unknown coverage. Compare these with rendering quality on the same cases; disagreement between the metrics is a finding rather than something to collapse into one score.

The intended conclusion is narrower and more useful than “the model can map”: determine which spatial operations its memory makes accessible, at what readout/training cost, and whether those operations support decisions. These experiments should become a near-term complement to reconstruction benchmarks, not depend on assuming reconstruction alone proves downstream readiness.
