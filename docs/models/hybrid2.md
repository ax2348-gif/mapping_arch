# hybrid2: the hybrid with time carried by RoPE

Fork of `hybrid` at the positional interface, as `vanilla2` is of
`vanilla`. The GLA/attention stack, the ladder, the muP learning-rate rule
and the initialisation are the parent's. Three changes:

1. **RoPE on the step index** in every softmax-attention layer: queries and
   keys are rotated by the token's step (head dim 32, 16 rotary pairs, base
   10000), so attention scores depend on the step difference between two
   tokens from the first update. The parent's attention had no positional
   information except a learned 33-entry table that starts as noise.
2. **No per-step tables.** The learned time embedding and the token-type
   embedding are gone. Observation tokens use nine learned patch embeddings
   shared across steps, query tokens nine learned query-patch embeddings
   shared across queries, direction tokens the direction embedding alone.
   Kind is carried by the table a token uses; step by RoPE and by order.
3. **Queries sit at their step** (per-step queries, `CONTRACT.md`): a query
   asked at step t has its nine tokens directly after observation t, so the
   recurrence state it reads is the prefix up to its anchor. With every
   query at the last step this is the parent's layout.

Why: with per-step queries the parent's attention could only tell steps
apart through the learned table, so early training had no notion of
"nearby step" for the short compositions that bootstrap the long ones.
RoPE gives that structure for free, and once time is in RoPE the per-step
tables are redundant, so the embeddings are tied across steps.

Ladder (`python scaling.py --model hybrid2 --dry`): 109,291 / 837,201 /
2,798,907 / 6,609,577 / 22,238,481 / 52,645,257 parameters for widths 64
to 512, about 1.5% below the parent at each rung (the 33-entry tables).

Checks: output with `query_time` absent equals output with every query at
the last step exactly; the attention score between two tokens is invariant
to a common shift of their steps; the placement is the identity for
last-step queries and keeps each query's nine tokens together. Smoke runs
pass on both query modes. First results: see `python summarize.py --model
hybrid2`.
