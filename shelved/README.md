# Shelved variants

Shelved on 2026-10-15 to keep the first study to vanilla, hybrid and the
recurrent baseline. Nothing here is imported: `mapping/models` only picks
up modules inside its own package, so these stay out of the registry.

| variant | state when shelved |
| --- | --- |
| `slots` (latent memory tokens) | `models/slots.py` written, smoke run started, no doc |
| `looped` (weight-tied iterated depth) | `models/looped.py` written, xl rung measured at 7.0 GiB peak for fwd+bwd at batch 64 bf16, smoke run started, no doc |

To resume one: move `models/<name>.py` back to `mapping/models/`, move
`results/<name>` back to `results/`, re-read CONTRACT.md (it changed after
these were started: the architect-not-roboticist section), and finish the
deliverables listed there. The in-session agent transcripts are not
preserved beyond the session; the files are the state.
