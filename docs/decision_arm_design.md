# Design note: memory-based decision making in discrete SE(2)

Status: DRAFT for sign-off. Nothing here is frozen or built yet. Open
questions are marked **[decide]**.

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

## Engineering estimate

- **Stage 1:**
  - a goal-placement option and a decision target in the generator,
    measured in tens of lines
  - a decision head and a two-head loss in the trainer
  - the truncation evaluation

  It reuses every model. Days, not weeks.
- **Stage 2:** an environment loop and an RL learner, or adapters to an
  existing benchmark. This is larger; decide after stage 1.
