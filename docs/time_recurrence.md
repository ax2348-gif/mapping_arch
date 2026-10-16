# Time-only recurrence on se2d (xs): results, 2026-10-16

Models: `hybrid4_time` and `hybrid4_time_full` (mapping/models/se2d/hybrid4_time.py).
Preset: `conf/experiment/se2d_time_recurrence.yaml`. Recipe: the reference's (100k steps, cosine, uniform query times, MNIST, seed 0).

| step | hybrid4 (reference) | A: hybrid4_time [time, attn] | B: hybrid4_time_full [time, full GDN] |
| --- | --- | --- | --- |
| 5k | 0.0757 | 0.0858 | 0.0810 |
| 10k | 0.0673 | 0.0755 | 0.0770 |
| 20k | 0.0566 | 0.0681 | 0.0641 |
| 40k | 0.0417 | 0.0601 | 0.0558 |
| 100k | **0.0180** | **0.0526** | **0.0487** |
| shuffled odometry, 100k | 0.1403 | 0.0916 | 0.1071 |
| params | 119,661 | 127,343 | 132,563 |
| steps/s, local | 117 | 121–155 | 98–122 |

Floor 0.0986. Both variants FAIL. They land with the earlier failed ablations (0.042–0.056).

**Mechanism.** Probes: tools/time_scan_probe.py on the time scan; tools/transition_probe.py on B's full layer.
- **A: the time scan learns the exact rotation.**
  - At 100k, both heads have an eigenpair within 10° of the turn on 100% of quarter- and half-turn steps.
  - Median beta on turn steps 1.69; det > 0 on 100%.
  - It formed late: 24% at 40k, none at 20k.
  - Yet rendering fails, so composing poses is not the bottleneck. With pose-only state, the single attention layer at xs cannot fetch and assemble the content.
- **B: the time scan never rotates** (beta ≈ 0.93; 0% hits). The rotation moves into the full GDN layer: its left-to-right head has an eigenpair at 90.2° (10–90%: 89.2–91.5) on 99.9% of quarter-turn steps. But at xs, B has no attention layer, so nothing reads the rotated memory into views.

**Grids.**
- A: blurry blobs at the right place for near-anchor queries; far queries dark or wrong.
- B: blotchy texture without strokes everywhere.

**Read.** The recurrence must hold CONTENT, not just pose, for xs to render. In hybrid4 the delta-rule state is an associative memory of patches transported by the rotation. Restricting it to the time axis keeps the rotation (A shows it is learnable from odometry alone, via DeltaProduct) but removes the memory.

**Caveat.** At xs the switch "last layer = full GDN" removes the only attention layer, because the layer count is matched. So B tests [time, GDN] and not [time, attn, ..., GDN]. At s (4 layers) the layouts differ: A [time, attn, time, attn], B [time, attn, time, full GDN].
