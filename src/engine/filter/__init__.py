"""Stage 3 — filter: a rule-based biallelic shortlist with a recorded reason per row.

Pure and network-free: ``rules`` holds the functions, ``run`` streams the stage-2
table through them and writes ``03_filter/``, ``cli`` exposes ``engine filter``.
See CONTRACTS.md for the column, config and output contracts.
"""
