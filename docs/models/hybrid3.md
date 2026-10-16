# hybrid3: hybrid2 with SwiGLU

One change from `hybrid2`: every block's MLP, after both the
gated-linear-attention and the softmax-attention mixers, is SwiGLU,
`down(silu(gate(x)) * up(x))`, with hidden width 8/3 w rounded up to a
multiple of 32. The 4x GELU MLP it replaces can only approximate products
of input features; SwiGLU computes a bilinear function of the block input
directly. Everything else is hybrid2: RoPE on the step index, tied patch
and query embeddings, per-step query placement, ladder, muP rule, init.

Parameters versus hybrid2: +7.7% at xs (the rounding of 170.7 to 192),
+0.5% at xxl. Exists for both tasks (`mapping/models/hybrid3.py`,
`mapping/models/se2/hybrid3.py`). Forward with no query times equals the
all-last-step forward exactly; smoke runs pass on both tasks.
