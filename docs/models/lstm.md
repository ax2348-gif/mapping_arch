# lstm: residual bidirectional LSTM encoder, state-bridged LSTM decoder

**Role.** The recurrent reference point: an older sequence architecture
whose native operation is carrying state across steps. It is pure
recurrence end to end; nothing attends back over the sequence.

**Architecture.** The episode is tokenised as 65 interleaved tokens
`obs_0, dir_0, obs_1, ..., dir_31, obs_32`: observation tokens are a
two-layer MLP of the 81 raw pixels, direction tokens a 4-entry embedding,
plus a token-type embedding. No sequence-index encoding is used. The
encoder is `depth` pre-norm residual blocks `x + LSTM(LN(x))`, each a
cuDNN bidirectional LSTM with `width/2` units per direction. The readout is
a seq2seq decoder: a unidirectional residual LSTM stack of the same depth
and width, whose layer-l initial `(h, c)` is a linear bridge from encoder
layer l's final `(h, c)`. Each query offset is embedded once (scaled
scalars plus fixed sin/cos features at 16 log-spaced periods from 3 px to
4x the offset range, then an MLP), presented to the decoder for 4 steps,
and the last step's output is normalised and decoded to 81 pixels. The 8
queries run as a batch of `B*8` independent decoder sequences from copies
of the same encoder state. Init: Xavier input matrices, orthogonal
recurrent matrices per gate, forget bias 1; the harness clips at 1.0.
Ablation variants exist in the file but are off the ladder: `readout='mlp'`
(pure, MLP on final states) and `readout='attn'` (**hybrid**: one
multi-head attention layer from the query over the encoder outputs).

**Why recurrence might integrate where a vanilla transformer does not.**
A gated recurrence is an accumulator by construction: with a forget gate
near 1, the cell state is a running sum of gated inputs, so summing the
direction embeddings over time is the identity dynamics of the layer, and
the same state can be gated by the current direction to address what it
stores. A transformer has to synthesise that prefix sum from softmax
attention over positions, which is not an operation it expresses cheaply.
Bidirectionality is deliberate for this offline task: the queries are
expressed relative to the last observation, and the backward direction
accumulates from exactly that end, so what it integrates is in the frame
the decoder is asked in. The decoder is recurrent rather than an MLP so
that retrieval can be a few gated updates of a state rather than one
feed-forward read.

**Scaling rule.** Rung r = 0..6: `width = 128 * 2^(r/2)` rounded to a
multiple of 64, `depth = 2 + r // 2`; decoder matches encoder; think steps,
Fourier features and head width constant. Params ~ `14 * width^2 * depth`,
x143 across the ladder. Learning rate constant at the harness default
3e-4: on `xl` (17.7M) 3e-4 beat 1.5e-4 at 4000 steps (eval mse 0.0706 vs
0.0726) and was stable, so no per-rung `LR` table is set.

```
$ python scaling.py --model lstm --dry
size         params  kwargs
xs          610,257  {'width': 128, 'depth': 2}
s         1,345,425  {'width': 192, 'depth': 2}
m         3,421,521  {'width': 256, 'depth': 3}
l         7,638,993  {'width': 384, 'depth': 3}
xl       17,733,201  {'width': 512, 'depth': 4}
2xl      39,772,497  {'width': 768, 'depth': 4}
3xl      87,393,361  {'width': 1024, 'depth': 5}
```

**Expected failure modes.** (1) State capacity: everything the decoder can
read is the final `(h, c)` of each layer, `2 * depth * width` numbers, and
it must hold 33 windows of 81 pixels together with where each sits; past
the point where that no longer fits, more width buys storage rather than
better integration, so the curve should bend. (2) Length: accumulation
error and forgetting grow with the 65-step scan, and the decoder has to
address content written up to 64 steps earlier; longer episodes should
hurt super-linearly. (3) Addressing precision: queries fall at arbitrary
pixel offsets straddling up to four windows, and a fixed-size state has no
natural slot per window, so sub-step alignment is where it should fail
first (watch `fg_within_0.1`).

**Smoke results (local GPU, batch 64, bf16 autocast, seed 0).**
`python train.py --model lstm --size xs --steps 500`: floor mse 0.0969 /
fg_mae 0.6298; final eval mse **0.0959**, fg_mae 0.6239, fg_within_0.1
0.076, dirs-shuffled mse 0.0958, 105 it/s on the shared GPU, 4.9 s total.
That beats the floor but barely; 500 steps is before the model leaves the
mean. The same rung for 8000 steps (~40 s, ~220 it/s) reaches eval mse
**0.0741**, fg_mae 0.489, fg_within_0.1 0.100, with dirs-shuffled mse
0.1025: once it breaks away (around step 2500) its predictions depend on
the directions, and scrambling them makes it worse than the floor. `xl`
for 4000 steps: eval mse 0.0706, fg_mae 0.467, 55 it/s.

**Largest rung.** `3xl` (87.4M) at batch 64 under bf16 autocast, forward +
backward + clip + AdamW step: peak allocated 2.54 GiB, peak reserved 2.79
GiB, 18 it/s on synthetic batches (loader excluded).

**Harness note.** Under bf16 autocast PyTorch runs cuDNN LSTM in fp16
(no bf16 RNN kernel), without loss scaling; the per-pixel MSE gradients
(~1e-5) are fp16 subnormals. The model therefore runs its recurrent layers
in fp32 (TF32 matmuls) inside a local autocast-disabled block; in a
2000-step xs check the fp16 path reached 0.0952 against 0.0939 for fp32.
