"""The live mode of the engine — a local web app over the run directories.

``engine ui --work <dir>`` serves a single-user page on ``127.0.0.1`` that browses
runs, reads every stage's output through :mod:`engine.report.views` (the same facts
the report renders, never a second reading of the files) and launches stages as
subprocess jobs whose output streams into the page as it is written.

Three modules: :mod:`engine.ui.jobs` runs ``uv run engine <stage> …`` with a
per-stage argument whitelist and turns its output into events; :mod:`engine.ui.server`
is the standard-library HTTP server with the JSON API, the Server-Sent Events stream
and the static files; :mod:`engine.ui.cli` is the click command. The page itself is
``static/`` — vanilla HTML, CSS and JavaScript, no build step.
"""
