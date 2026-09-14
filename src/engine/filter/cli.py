"""Stage 3 command — registered by ``engine.cli`` from ``commands``.

Prints counts and the manifest path only: no variant, gene or genotype reaches the
terminal at normal verbosity, so a pasted shell transcript carries no patient data.
"""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.command(name="filter")
@click.option("--run", "run_dir", required=True, type=click.Path(file_okay=False, exists=True, path_type=Path),
              help="Run directory that already holds 02_retrieve/variants.annotated.tsv.gz.")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Thresholds (every one is echoed into the manifest). Default: configs/filter.yaml in the source tree.")
def filter_cmd(run_dir: Path, config_path: Path | None) -> None:
    """Stage 3: rule-based biallelic shortlist with a recorded reason for every variant."""
    from engine.filter.config import DEFAULT_CONFIG
    from engine.filter.run import run_filter

    config_path = config_path or DEFAULT_CONFIG
    click.echo(f"filter · {run_dir} · config={config_path}")
    manifest = run_filter(run_dir, config_path)
    c = json.loads(manifest.read_text())["counts"]
    by_model = ", ".join(f"{k} {v}" for k, v in c["candidates_by_model"].items()) or "none"
    click.echo(f"  rows in: {c['rows_in']:,} · kept: {c['kept']:,} · dropped: {c['dropped']:,} · "
               f"candidates: {c['candidates']:,} ({by_model})")
    click.echo(f"  manifest: {manifest}")


commands: list[click.Command] = [filter_cmd]
