#!/bin/bash
# Main runs, one at a time (shared GPU). From the repository root:
#   bash experiments/lift_blind_probe/run_all.sh
set -e
D=experiments/lift_blind_probe/runs
mkdir -p $D
run() {  # kind n [extra args]
  name=$1_n$2${4:-}
  [ -f $D/$name/final.pt ] && return
  python experiments/lift_blind_probe/train.py --kind $1 --n $2 $3 --out $D/$name > $D/$name.txt 2>&1
}
run gdn 0
run attn 0
run gdn_b1 0
run cdiag 0
run gdn 4
run gdn 8
run gdn 16
run gdn 32
