"""Reproducible rare-disease variant engine.

Stages, in order: ingest → retrieve → filter → rank → reason → medicine → bench → submit.
Each stage is a sub-command of the ``engine`` CLI, reads the previous stage's run
directory, and writes a manifest describing exactly what it did.
"""

__version__ = "0.1.0"
