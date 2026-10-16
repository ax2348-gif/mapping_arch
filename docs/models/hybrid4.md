# hybrid4: hybrid3.5 on the fused Gated DeltaNet kernel

Same layer as `hybrid3_5`, with `GatedDeltaNet.scan` calling
`chunk_gated_delta_rule` from flash-linear-attention 0.5.2 (Triton)
instead of the sequential loop. Inputs unchanged; beta in (0, 2) is
passed directly and the kernel applies it as given.

Checks against the hand-written scan (`delta_scan`): output relative
error 2e-3 (fp32), 7e-3 (bf16); gradients 2-4e-3; the whole model with
hybrid3.5's weights agrees to 3e-4. Forward-backward at xs, batch 64:
6 ms against 172 ms for hybrid3.5. Ladder: 118,637 to 53,033,481
parameters. Smoke runs pass on both tasks.

Requires `fla-core==0.5.2` and `einops` (installed locally with
`--no-deps` so torch is untouched); the cluster environment needs the
same before hybrid4 runs there.
