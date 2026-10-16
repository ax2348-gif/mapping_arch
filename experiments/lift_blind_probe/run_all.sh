#!/bin/bash
# Main runs, one at a time. The local GPU is shared by several agents and
# training jobs must run sequentially across all of them: every training
# command takes the shared lock. From the repository root:
#   bash experiments/lift_blind_probe/run_all.sh
set -e
D=experiments/lift_blind_probe/runs
LOCK=/tmp/claude-1000/-home-adrianna/b0ebca23-53ce-4dc1-a5c5-072191a4dabe/scratchpad/gpu_train.lock
mkdir -p $D
run() {  # kind n [extra args]
  name=$1_n$2${4:-}
  [ -f $D/$name/final.pt ] && return
  flock $LOCK python experiments/lift_blind_probe/train.py --kind $1 --n $2 $3 --out $D/$name > $D/$name.txt 2>&1
}
run gdn 0
run attn 0
run gdn_b1 0
run cdiag 0
run gdn 4
run gdn 8
run gdn 16
run gdn 32
