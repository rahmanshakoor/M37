"""The ``report`` command, exposed through ``commands`` the way every stage package
exposes its CLI.

``engine.cli`` registers a package's ``commands`` only when its ``STAGE_PACKAGES``
table names the package; until that table has a ``report`` entry, ``engine report``
is not a sub-command of ``engine`` and the command is reachable as this click object
only (``tests/test_report.py`` checks the registration and says so when it is
missing). ``report_command`` is the same command under a second name, for a caller
that registers it by hand.

Prints the output path and the stages it found only: no variant, gene or genotype
reaches the terminal, so a pasted shell transcript carries no patient data.
"""

from __future__ import annotations

from pathlib import Path

import click


@click.command(name="report")
@click.option("--run", "run_dir", required=True, type=click.Path(file_okay=False, exists=True, path_type=Path),
              help="Run directory; every stage present is rendered, missing ones are said to be missing.")
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path),
              help="Where to write the HTML. Default: <run>/report.html.")
@click.option("--top", "top_n", type=click.IntRange(1), default=None,
              help="Ranked genes shown in the blind-ranking section, plus every shortlist gene (default: 20).")
def report(run_dir: Path, out_path: Path | None, top_n: int | None) -> None:
    """Report: one self-contained HTML document over whatever stages the run holds."""
    from engine.report.render import render_run
    from engine.report.views import DEFAULT_TOP_N, run_summary

    out_path = out_path or run_dir / "report.html"
    html = render_run(run_dir, top_n=top_n or DEFAULT_TOP_N)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    present = run_summary(run_dir)["stages_present"]
    click.echo(f"report · {run_dir} · stages: {', '.join(present) or 'none'}")
    click.echo(f"  {len(html.encode('utf-8')):,} bytes → {out_path}")


report_command = report

commands: list[click.Command] = [report]
