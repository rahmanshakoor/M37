"""``engine mosaic`` — chromosome-level mosaic-aneuploidy scan from the VCF."""

from __future__ import annotations

import json
from pathlib import Path

import click

from engine.config import CaseConfig


@click.command(name="mosaic")
@click.option("--case", "case_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="case.yaml (the VCF it names is scanned in full).")
@click.option("--out", "run_dir", required=True, type=click.Path(file_okay=False, path_type=Path),
              help="Run directory; output goes to <out>/mosaic/.")
@click.option("--min-dp", default=20, show_default=True, help="Minimum AD depth for a heterozygote to count.")
@click.option("--skew", default=0.15, show_default=True, help="A heterozygote is skewed outside 0.5 ± this.")
@click.option("--z-flag", default=3.0, show_default=True, help="Flag an autosome this many SD from the others.")
def mosaic(case_path: Path, run_dir: Path, min_dp: int, skew: float, z_flag: float) -> None:
    """Side analysis: per-chromosome allele balance and depth of the proband's own
    heterozygous calls, z-scored against the other autosomes — a screen for a
    clonal or high-fraction mosaic aneuploidy without a BAM or a karyotype."""
    from engine.mosaic.run import run_mosaic

    case = CaseConfig.load(case_path)
    manifest = run_mosaic(case, run_dir, min_dp=min_dp, skew=skew, z_flag=z_flag)
    m = json.loads(manifest.read_text())
    c = m["counts"]
    click.echo(f"mosaic · {case.vcf.name} · {c['hets_total']:,} het SNVs over {c['contigs']} contigs · "
               f"genome median depth {c['genome_median_dp']}")
    for n in m["notes"]:
        click.echo(f"  {n}")
    click.echo(f"  manifest: {manifest}")


commands = [mosaic]
