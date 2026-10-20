# Preliminary multi agent spatial memory design

Superseded candidate: the subsequent design replaces pairwise relation tokens with shared-prefix state routing and forward-only agent/query scans. Its implementation and current proposal are in the sibling `mapping-multiagent` worktree on branch `exp/hybrid4-multiagent`, at `docs/models/hybrid4_multiagent.md` and `docs/multi_agent_proposal.md`. The text below records the earlier relation-token design.

Extend the [hybrid6 camera interface](patch_geometry_proposal.md) to multiple agents without learned agent identities or a privileged global frame. Begin with a static scene and one observation time: test whether a render anchored to any agent can reconstruct content observed by other agents. Temporal recurrence is a later extension.

Regular attention carries information between agents through explicit routing. Linear attention stays within each agent, and each render request has a separate, temporary linear-attention state. This is a proposed extension, not part of the current single-agent hybrid6 training run.

## Frames and token roles

Agent i defines a local frame F_i. Observation and render patches retain hybrid6's shared sensor language: camera extrinsics relative to the chosen frame, crop-specific intrinsics, and pixel sampling geometry. All patches from one camera share its extrinsic pose. An observation camera defining F_i has identity extrinsics.

Each agent has observation tokens O_i, one state token H_i that also serves as its reference token, and relation tokens R_{i|j}. H_i is an ordinary learned token representation; it is distinct from the matrix-valued recurrent state of a linear-attention scan.

A supplied relation R_{i|j} specifies the pose of agent i in agent j's frame. For notation only, if G_i maps F_i coordinates into a world frame, its transform is:

```text
T_{j<-i} = inverse(G_j) * G_i
```

This maps coordinates from F_i to F_j. It belongs to i's receiving pool but reads j's reference. Its direction must remain explicit in data generation and embeddings. No G_i is supplied to the model.

Agent indices identify routing endpoints, not learned embeddings. Shared weights, shared role embeddings, and masks that permute with the agents preserve agent permutation equivariance. That property does not itself guarantee learned SE3 equivariance or generalization to larger agent counts.

## Regular attention routing

The following table specifies context access. Token residual connections remain present.

| Receiving token | Allowed context sources |
| --- | --- |
| O_i or H_i | All O_i, H_i, and R_{i\|j} in agent i's pool |
| R_{i\|j} | Only the designated reference H_j |
| Render patch anchored to i | All agents' observation and relation tokens, plus exactly H_i among the state/reference tokens |

Relations do not attend back into their receiving pool. Direct O_i/H_i access to another agent's O_j/H_j is excluded: the relation token provides the association between source information and the supplied transform. These are directed attention masks, not symmetric pool membership.

Updates are synchronous. Information travels H_j -> R_{i|j} -> O_i/H_i in two regular-attention layers, with every edge processed in parallel. This creates a depth requirement for communication hops, not one layer per agent. A dense graph provides every direct connection at O(N^2) relation-token cost; avoid implementing unrestricted attention over all relation tokens, which would introduce O(N^4) attention pairs.

H_i may contain information acquired from other agents. Its reference meaning comes from its structural association with F_i, not from keeping its contents purely local. Repeated communication can refine representations; it does not require serial agent updates or stationary frame stacking.

With only one permitted source, a relation's attention weight is trivially one. Conditioning on the supplied transform therefore relies on its residual geometry embedding and subsequent learned processing, rather than competition among source keys.

## Supplied relations and visual registration

Supplied relative poses make the first experiment a test of using spatial relations for collaborative rendering. They do not test recovering those relations from visual evidence. Keep this distinction explicit in results and subsequent experiments.

Without a supplied pose or other pair-specific input, identically initialized R_{i|j} and R_{k|j} reading only H_j remain identical under deterministic shared processing. They can instead be represented by one multicast reference message from j. Visual registration then happens where this message meets the receiving agent's observations, or in a separate pair readout. The restricted relation token cannot independently infer a pair relation without access to information about both endpoints.

For embodied-agent visibility, one source message can be routed to every agent that sees that source. This preserves separate source messages with O(N) message tokens, although recipient-message attention pairs can still be O(N^2). A supplied visibility mask is additional task information if visibility or association is otherwise meant to be inferred. Aggregating many sources into one message before delivery creates a different, potentially restrictive information bottleneck.

## Agent local linear attention

Each agent has an independent linear-attention stream with shared parameters. There is no scan across agents and no attempt to merge their recurrent states. Gather agents into a batch dimension for scans; keep convolution histories and recurrent states separate as well.

Bidirectional scans within each static agent pool are a candidate augmentation. They preserve independence from agent ordering, but remain sensitive to the order of tokens within each pool. In particular, ordering relations by arbitrary agent index could break agent permutation equivariance if those relations participate in an order-sensitive scan.

The regular-attention mask cannot simply be assumed to carry over to GDN. A relation token must not read a local recurrent summary containing O_i/H_i when its only allowed source is H_j. A conservative implementation is to scan local observation/state tokens after regular attention has delivered relation information, leaving relation-token updates to their designated regular-attention path. If relation tokens also write into a local scan, their read permissions and order dependence need an explicit design. This choice remains open.

## Inert rendering with private query computation

Each requested image defines a group of patch queries with one chosen reference H_i. Every regular-attention layer may read all observation and relation tokens plus H_i; access does not have to wait until relation processing is complete. Reference and relation roles distinguish the chosen anchor from other contextual information. Patches within a render request may communicate, while different requests remain isolated.

Each render request has its own bidirectional linear-attention stream, initialized with a fresh state. It never imports or merges the agents' recurrent states. Regular attention gathers scene information, and subsequent query-local linear attention can coordinate the requested image's patches. Alternating layers repeats these operations.

This is inert querying: observation, state, and relation tokens cannot read query tokens; queries never write agent memory; unrelated render requests cannot affect each other. The private query state is discarded after rendering. Inert does not mean that queries lack internal computation.

The scan introduces an output-patch order even when bidirectional. Test its effects on variable image dimensions and patch ordering. Queries for the same physical camera expressed relative to different anchors should agree after changing their specified extrinsics consistently; this is a useful geometric consistency test.

## Relation readout and supervision

An optional measurement of agent i relative to j can read exactly two external tokens: a reference message that reads only H_j, and the target state H_i. Distinct roles specify the ordered pair without agent identity embeddings. The reference message can be reused by all measurements anchored to j and need only be visible to those measurements.

Uniformly sampled pair losses can estimate the full pairwise objective without reading out every pair on every example. Parameter sharing removes dependence on arbitrary agent labels, but sparse supervision does not guarantee coverage of unfamiliar geometry or larger populations.

No global canonical frame is required. A global-looking render still specifies one reference frame and a camera pose in it. In SE3, metric relative translations require a metric cue or a declared scale convention; stationary monocular images alone do not generally establish metric scale. The existing planar task fixes that ambiguity through its scene and camera convention.

## Initial experiment and validation

Start with a static planar scene, multiple agents, calibrated observations, supplied relative poses, and render requests anchored to randomly selected agents. Choose targets containing content unavailable in the anchor's own observation but visible to another agent. Include an anchor-only baseline so success demonstrates use of remote observations. Hold out scenes and evaluate both reconstruction and sensitivity to incorrect relation inputs.

Before interpreting learning results, check agent permutations with masks permuted consistently, independent-request invariance, and absence of query writes into context. Test transform direction with known geometry, anchor changes for the same physical render, varying agent counts, and within-pool ordering. Compare regular attention alone with the proposed local bidirectional scans to establish their contribution.

Later temporal experiments retain separate recurrent states per agent and block-causal regular attention: every agent at time t is processed symmetrically, and no token reads future observations. Bidirectionality may operate within an available time block, but must not expose future time steps. Visual-only registration and incomplete relation graphs are subsequent task variants, with their assumptions reported separately from the supplied-pose experiment.
