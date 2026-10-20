# Does clean memory organization emerge late in training?

2026-10-18. User-proposed checkpoint test. No new training. Same128 held-out SAW episodes per model and horizon, seed1806; source-aligned queries at quarter/half/three-quarter trajectory, asked at the end. All six available Hybrid4 checkpoints and Hybrid6 final were tested. All native replay gates pass (MSE<1e-4, relative-Q error<.03); tiny effects remain limited by replay roundoff.

Hybrid4 lineage: 50k and100k checkpoints from T32 training; then25k/50k/75k/100k more updates at T64, for125k/150k/175k/200k total. Offset scale stayed pinned at288. Final decay spans180k–200k total. Hybrid6 has100k T32 updates and decay80k–100k. Scan arithmetic differs (Hybrid4 TF32, Hybrid6 IEEE); this is a longitudinal test within Hybrid4, not a fully controlled architecture comparison.

## Evaluation horizon T32

| Model / total steps | Native MSE | Final attention history + both GDN states | Zero GDN0 query read | Zero GDN2 query read |
|---|---:|---:|---:|---:|
| h4_50k | 0.041742 | 0.057685 | 0.076600 | 0.097747 |
| h4_100k | 0.016815 | 0.025460 | 0.066498 | 0.101353 |
| h4_125k | 0.008430 | 0.019526 | 0.067346 | 0.107100 |
| h4_150k | 0.007427 | 0.017662 | 0.082398 | 0.114379 |
| h4_175k | 0.001604 | 0.019405 | 0.096818 | 0.110495 |
| h4_200k | 0.000202 | 0.000468 | 0.114817 | 0.107947 |
| h6_100k | 0.002959 | 0.052546 | 0.003007 | 0.125390 |
## Evaluation horizon T64

| Model / total steps | Native MSE | Final attention history + both GDN states | Zero GDN0 query read | Zero GDN2 query read |
|---|---:|---:|---:|---:|
| h4_50k | 0.092032 | 0.093906 | 0.094294 | 0.104294 |
| h4_100k | 0.071611 | 0.081950 | 0.092133 | 0.110404 |
| h4_125k | 0.038409 | 0.050113 | 0.078103 | 0.107237 |
| h4_150k | 0.028658 | 0.054000 | 0.097999 | 0.111235 |
| h4_175k | 0.014077 | 0.034725 | 0.105058 | 0.111006 |
| h4_200k | 0.000296 | 0.000772 | 0.116461 | 0.113098 |
| h6_100k | 0.033777 | 0.088743 | 0.039324 | 0.131415 |

## Interpretation

The user’s training-stage concern is supported. At T32, even Hybrid4 at175k has good intact rendering (.001604) but loses much of it when earlier attention history is removed (.019405). At200k, intact .000202 and final-only .000468 are both low. T64 shows the same late transition (.034725 final-only at175k → .000772 at200k). The compact final-cache result should not be treated as an intrinsic architectural difference between Hybrid4 and Hybrid6 based on their final checkpoints alone.

This does not show that more steps alone will clean up Hybrid6. The last Hybrid4 interval includes both more T64 training and learning-rate decay; checkpoint spacing cannot separate their effects. Hybrid6 already received a decay. Moreover, Hybrid4 GDN0 query-read dependence is present at earlier stages, whereas Hybrid6 GDN0 query-read ablation is nearly harmless. That circuit difference remains, but its persistence under further training is untested. T64 is also out of the training horizon for Hybrid6 and the first two Hybrid4 checkpoints.

The next discriminating training experiment would continue Hybrid6 through the same T64 curriculum and final decay, retaining intermediate checkpoints and repeating this suite. This is a proposal, not a launched continuation. Poor-rendering early checkpoints cannot establish meaningful rewind/addressing failures; consult their intact controls in the full reports.

## Beta clipping across stages (T32)

Independent128 held-out episodes at seed1807. Clipping is in the native history computation, not a query-read intervention. Each stage reports its own intact baseline.

| Checkpoint | Native | Clip GDN0 beta<=1 | Clip GDN2 beta<=1 | Clip both |
|---|---:|---:|---:|---:|
| h4_50k | 0.045236 | 0.099273 | 0.070326 | 0.102253 |
| h4_100k | 0.015846 | 0.119026 | 0.103993 | 0.115522 |
| h4_125k | 0.009555 | 0.123838 | 0.109410 | 0.115880 |
| h4_150k | 0.008212 | 0.133805 | 0.122036 | 0.116124 |
| h4_175k | 0.001972 | 0.128103 | 0.110662 | 0.111018 |
| h4_200k | 0.000242 | 0.138942 | 0.115671 | 0.111688 |
| h6_100k | 0.003088 | 0.003104 | 0.149653 | 0.149809 |

Full suites, including texture swaps, per-head ablations, rewind controls, odometry compensation, grid-key transfer and query-lag scans: `results/mechanistic_checkpoint_sweep_t32/report.md` and `results/mechanistic_checkpoint_sweep_t64/report.md`. Raw JSON includes checkpoint hashes and source hashes. Beta distributions and turn-conditioned statistics are in `results/mechanistic_checkpoint_beta/`. Plot: `results/mechanistic_checkpoint_sweep.png`. Reproduce the suite with `experiments/hybrid6/mechanistic/checkpoint_sweep.sh`; beta checks accept `--checkpoint` and `--out`.
