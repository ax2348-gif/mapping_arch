# Frozen Hybrid7 AGAGA S: 15x15 views with 9px moves

Evaluated final100k checkpoint using `experiments/hybrid7/view15_eval.py`,
identical to the earlier Hybrid6 protocol:256 fresh matched MNIST SAW T32
episodes, seed1808/halo1809. Extend the same maps, assert exact center9 equality,
keep all poses/motion fixed. No training. Results and preselected render panel:
`results/hybrid7_view15_controls/{results.json,renders.png}`.

Uniform queries, central9x9 MSE:

| Observations → query | Intact | Shuffled odometry |
| --- | ---: | ---: |
| 9 → 9 |0.015051|0.114425|
| 15 → 15 |0.041849|0.106712|
| 15 → 9 |0.042017|0.106241|
| 9 → 15 |0.015644|0.115869|

Full15x15 MSE for15→15 is0.085420 (shuffled0.113255), much worse than
center performance. 9→15 scores ONLY supported central9 pixels; outer targets
can be unseen in that condition and are not scored. For15→15, full targets
are covered by expanded history under the same quarter-turn geometry.

Aligned final-time retrieval (sources8,16,24,31): center error0.022992 native,
0.046810 at15→15,0.047022 at15→9,0.023628 at9→15.

Interpretation: partial survival, not full resolution generalization. History
expansion raises center error2.78x but retains a substantial odometry advantage.
Query expansion alone changes center performance little. Earlier Hybrid6 on
the same examples collapsed from0.002020 to0.146498 center MSE at15→15,
approximately its shuffled error0.144612. Hybrid7 is markedly more robust to
history expansion despite worse native accuracy, but outer-patch rendering
remains poor. The saved examples show recognizable central fragments alongside
blur/ghosting and erroneous border content. This is not a clean causal
attribution to register-only writes: block masks, depth and training quality
also differ. Observation permutation invariance does not imply invariance to
extra evidence, novel patch calibration, overlap or changed softmax competition.
