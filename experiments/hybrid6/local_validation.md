# Local causal inert SE2D validation (2026-10-18)

User requested interrupting the local SE2C run and validating hybrid6 in a fast setting, preferably S. The SE2C L process was interrupted with SIGINT around 135k steps. Its most recent durable checkpoint is `mapping-poseprobe/results/se2c/hybrid4c_causal/l/inert_continuous_l_200k_ieee_seed0/step125000.pt`; intervening updates were not saved. Its queue status records intentional interruption. No automatic restart or monitoring automation was enabled.

The new run uses preset `hybrid6_initial` with these overrides:

- Model: hybrid6_causal S, inert queries, IEEE scan arithmetic.
- Data: SE2D MNIST, self-avoiding walks, 9px views/steps, T32, eight uniformly timed queries, fresh generated training episodes.
- Training: 100k steps, batch64, LR3e-4, 100-step warmup, WSD cooldown over steps80k–100k, AdamW weight decay .1, clip1.
- Validation: held-out evaluation and shuffled odometry every5k, 16 batches; logs every500, checkpoints every10k.
- Run directory: `results/se2d/hybrid6_causal/s/local_causal_inert_saw_s_100k_ieee_seed0`.
- Durable tmux session: `hybrid6-local-validation`.
- Launcher, stdout and exit code: `results/hybrid6_local_validation/`.

This tests whether the architecture learns the established discrete task before adding 15px geometry or continuous examples. It is not yet the proposed SE2D/SE2C curriculum. Success requires held-out learning and a substantial shuffled-odometry gap, including late query times; finite training loss alone is insufficient.

Prelaunch CPU geometry, runtime-grid, causal/inert wiring tests: 38 passed (`tests/test_hybrid6.py`). Two launch configuration failures (note-string punctuation and a missing data link) occurred before training began and were retained in separate logs. The worktree now links the existing local dataset directory.
