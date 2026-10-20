#!/usr/bin/env bash
set -eu
cd /home/adrianna/code/mapping-hybrid6
export OMP_NUM_THREADS=4
parent=/home/adrianna/code/mapping-saw/results/se2d/hybrid4_causal/s/inert_seed0_400k_qt_wsd_saw
continued=/home/adrianna/code/mapping-saw/results/se2d/hybrid4_causal/s/inert_init_saw100k_t64_saw_100k_wsd
for horizon in 32 64; do
 /home/adrianna/miniconda3/bin/python experiments/hybrid6/mechanistic/run.py \
  --checkpoint h4_50k=$parent/step50000.pt \
  --checkpoint h4_100k=$parent/step100000.pt \
  --checkpoint h4_125k=$continued/step25000.pt \
  --checkpoint h4_150k=$continued/step50000.pt \
  --checkpoint h4_175k=$continued/step75000.pt \
  --checkpoint h4_200k=$continued/final.pt \
  --checkpoint h6_100k=results/se2d/hybrid6_causal/s/local_causal_inert_saw_s_100k_ieee_seed0/final.pt \
  --steps "$horizon" --episodes 128 --batch 8 --grid-worlds 16 --seed 1806 \
  --out "results/mechanistic_checkpoint_sweep_t${horizon}"
done
