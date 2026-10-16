"""xp: porcelain over run.py, git and the run ledger (docs/provenance_design.md §9).

It wraps and does not hide: ``--explain`` prints every underlying command
before running it, ``--dry-run`` prints the ones that would change anything
without running them. Query commands take ``--json``. The data stays plain
text (ledger JSON, refs, registry YAML); losing xp loses convenience, never
data. run.py stays the only training entry: xp starts it as a subprocess
and never imports the trainer or torch (tests/test_xp.py checks).
"""
