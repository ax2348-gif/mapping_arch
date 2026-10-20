# Hybrid6 causal inert S mechanistic screen — 2026-10-18

Checkpoint: `results/se2d/hybrid6_causal/s/local_causal_inert_saw_s_100k_ieee_seed0/final.pt` (100k steps, T32, SAW MNIST, IEEE scans). No training or weights were changed.

Adapted the four-block Hybrid4 transfer suite from the mapping-poseprobe worktree. Hybrid6 query construction now uses its shared pose/calibration encoders and render role; query-input reconstruction is asserted bitwise equal to captured native input. State capture uses the native IEEE FP32/chunk32 arithmetic and returns the native output dtype. Replay reads states in FP32 with native BF16 query normalization/scaling. All-history rewind controls supplement final-cache-only controls.

## Results

128 fresh held-out SAW episodes, batch8, seed1806, source-aligned queries at times8/16/24 asked at32; grid screen uses16 additional worlds. These metrics differ from the training evaluation's uniform query-time distribution.

- Native aligned MSE .002959. Maximum replay/native MSE 4.80e-6 and relative final-Q discrepancy .00498; every batch passes the inherited replay thresholds (MSE<1e-4, Q-relative-error<.03). Capture/native equality and per-batch errors are in raw JSON. Small effects near this numerical scale are inconclusive.
- Keeping final attention history plus both GDN states raises MSE to .05255. Early attention history is important; the original Hybrid4 compact final-cache sufficiency result does not transfer unchanged.
- Zeroing query reads from GDN0 barely changes error (.003007); zeroing GDN2 raises it to .12539. GDN2 head0 is most consequential (.12028), followed by head2 (.08273) and head3 (.02888). This localizes query-read dependence, not all computations performed by GDN0 during history formation.
- Final attention head1 selects the source patch 99.94% of the time on these exactly aligned queries. Swapping source final values transfers .959 of the donor texture displacement; keys transfer .0026 and GDN states .0022. Value-head1 alone transfers .788, head2 .150. This supports texture retrieval through final attention values, despite early attention also contributing to address computation/readout.
- Reversing one odometry move: on26 collision-free counterfactuals, earlier-source MSE .00339 → .18501; compensating the query restores .00298, while wrong-sign compensation gives .17088. Later-source error remains .00203. This is strong evidence of a spatial displacement computation; it does not identify its internal coordinates.
- Rewinding GDN2 with the earlier-relative query partially recovers retrieval with all attention history retained: .16107 without rewind → .03672, versus intact .00339. Rewinding both is essentially identical. Final-only rewind is not cleanly interpretable because its intact positive control already fails (.05947).
- Grid same-cell donor keys largely preserve addressing: dominant-head patch accuracy .9221 → .9026; pixel MSE .01531 → .01963. Unlike the earlier severe key-swap failures, this model shows appreciable route transfer in this test. Grid rendering is weaker than SAW, especially the row route (MSE .01611 vs column .00424), so this is not proof of route-independent coordinates.
- Final-Q cosine has a 27px recurrence on both grid routes. This is descriptive and does not prove a uniquely decodable pose representation or establish that the network learned the recurrence independently of its Fourier input features.

Full numeric tables: `results/hybrid6_mechanistic_final/report.md`; raw per-batch measurements, checkpoint hash and suite source hashes: `results/hybrid6_mechanistic_final/results.json`.

## Basic beta check and causal clipping

Separate128 fresh held-out SAW episodes, seed1807. Distributions are split by GDN layer/head and history observation versus odometry tokens. Saved statistics include quantiles, fractions above1/1.9/1.99, decay multipliers, and odometry beta means conditioned on turn. Actual effective query beta is exactly0 throughout, as required by inert reads.

GDN2 head0 is particularly striking: 95.9% of observation beta values exceed1, 62.2% exceed1.9; 74.8% of odometry beta values exceed1.9. Mean decay multiplier rounds to1.0. Thus this head often uses nearly reflective state updates with very little decay. Other GDN2 heads also use beta>1 extensively. The near-zero/near-two split is not simply a universal “zero turn versus nonzero turn” rule; the saved turn-conditioned means show a more involved dependence.

Native-history clipping interventions (not frozen-history read ablations):

| Condition | Aligned MSE |
|---|---:|
| Native | .003088 |
| Clip GDN0 beta to at most1 | .003104 |
| Clip GDN2 beta to at most1 | .149653 |
| Clip both | .149809 |

The trained model depends strongly on beta>1 in GDN2. This intervention changes history formation and downstream activations; it does not isolate translation versus rotation, nor prove that a model trained with beta<=1 could not learn another solution. For normalized keys, the homogeneous update has eigenvalue1-beta in the key direction, explaining why beta near2 is reflection-like; complete state updates also include decay and value injection.

Raw beta results: `results/hybrid6_mechanistic_beta/results.json`; histogram: `results/hybrid6_mechanistic_beta/beta.png`.

## Reproduce

From repository root, with the existing dataset and CUDA environment:

```bash
python experiments/hybrid6/mechanistic/run.py --checkpoint hybrid6=results/se2d/hybrid6_causal/s/local_causal_inert_saw_s_100k_ieee_seed0/final.pt --steps 32 --episodes 128 --batch 8 --grid-worlds 16 --seed 1806 --out results/hybrid6_mechanistic_final
python experiments/hybrid6/mechanistic/beta_check.py
```

The main suite refuses an existing output directory; choose a fresh one to rerun. The beta script regenerates its diagnostic outputs. Both are inference-only. Source image donors are deliberate nonphysical interventions, and the translation subset is small. No explicit pose circuit or downstream policy capability has been established.

## Follow-up: checkpoint stage is a material confound

The user's proposed [earlier-checkpoint comparison](checkpoint_comparison.md)
shows that Hybrid4's clean final-cache sufficiency emerges late: it is still
substantially impaired at175k total updates despite good intact T32 rendering,
and becomes clean by200k across an interval including final LR decay. Therefore
the final-cache difference above cannot yet be attributed to architecture.
The beta and GDN0-dependence differences remain present across the tested
Hybrid4 stages. Further Hybrid6 training is needed to test their persistence.
