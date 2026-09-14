"""Report — one self-contained HTML document over a run directory.

``views`` reads whatever stages exist in a run directory and returns plain JSON
views (the same ones the UI serves); ``render`` turns those views into a clinical
document with inline CSS and JS; ``cli`` holds the ``report`` command, which
``engine.cli`` registers as ``engine report`` once its ``STAGE_PACKAGES`` table names
this package (see ``cli.py``). Every figure on the page comes from a file in the run
directory — nothing is recomputed here — and every evidence id is a link to the
record's own URL, so the document can be checked line by line against the sources it
cites. See CONTRACTS.md, "Report".
"""
