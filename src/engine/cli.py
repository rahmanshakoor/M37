"""``engine`` — one sub-command per stage, in pipeline order."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click

from engine import __version__

class PipelineOrder(click.Group):
    """List commands in the order the stages run, not alphabetically."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(self.commands)


@click.group(cls=PipelineOrder, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="engine")
def main() -> None:
    """Reproducible rare-disease variant engine.

    Stages run in order and each writes a manifest into the run directory:
    ingest → retrieve → filter → rank → reason → medicine → bench → submit.

    ENGINE_LOG=INFO (or DEBUG) prints the stages' per-batch progress to stderr.
    """
    level = os.environ.get("ENGINE_LOG", "").upper()
    if level:
        logging.basicConfig(level=getattr(logging, level, logging.INFO), stream=sys.stderr,
                            format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


@main.command()
@click.option("--case", "case_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="case.yaml describing the proband's inputs.")
@click.option("--out", "run_dir", required=True, type=click.Path(file_okay=False, path_type=Path),
              help="Run directory; stage output goes to <out>/01_ingest/.")
@click.option("--regions", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="BED of regions to restrict to (overrides the case file). Omit for whole genome.")
@click.option("--regions-overlap", type=click.IntRange(0, 2), default=1, show_default=True,
              help="With --regions: 0 = POS inside region, 1 = record overlaps, 2 = variant overlaps.")
@click.option("--threads", type=int, default=2, show_default=True)
def ingest(case_path: Path, run_dir: Path, regions: Path | None, regions_overlap: int, threads: int) -> None:
    """Stage 1: normalize the VCF into a flat, checksummed variant table."""
    from engine.config import CaseConfig
    from engine.ingest import run_ingest

    case = CaseConfig.load(case_path)
    click.echo(f"ingest · {case.vcf.name} · sample={case.sample or 'auto'} · "
               f"{'regions=' + str(regions or case.regions) if (regions or case.regions) else 'whole genome'}")
    manifest = run_ingest(case, run_dir, regions=regions, regions_overlap=regions_overlap, threads=threads)
    c = json.loads(manifest.read_text())["counts"]
    click.echo(f"  sites in: {c['sites_in']:,}   rows out: {c['rows_out']:,}   "
               f"flagged: {c['flagged_rows']:,}   split: {c['multiallelic_sites_split']:,}   "
               f"dropped non-primary: {c['dropped_non_primary']:,}")
    click.echo(f"  manifest: {manifest}")


@main.command()
@click.option("--panel", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="YAML mapping group → gene symbols.")
@click.option("--coords", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="JSON of symbol → [chrom, start, end] (1-based, GRCh38).")
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--pad", type=int, default=200, show_default=True, help="Flank added to each gene, in bp.")
@click.option("--chrom-style", type=click.Choice(["ensembl", "ucsc"]), default="ensembl", show_default=True,
              help="Spell chromosomes to match the VCF you will ingest.")
def regions(panel: Path, coords: Path, out_path: Path, pad: int, chrom_style: str) -> None:
    """Write a merged, padded BED for a gene panel."""
    from engine.regions import load_coords, load_panel, panel_genes, write_bed

    genes = panel_genes(load_panel(panel))
    n, missing = write_bed(genes, load_coords(coords), out_path, pad=pad, chrom_style=chrom_style)
    click.echo(f"regions · {len(genes)} genes → {n} merged intervals → {out_path}")
    if missing:
        click.echo(f"  no coordinates for: {', '.join(missing)}", err=True)
        sys.exit(1)


@main.command()
@click.option("--run", "run_dir", required=True, type=click.Path(file_okay=False, exists=True, path_type=Path),
              help="Run directory that already holds 01_ingest/.")
@click.option("--cache", "cache_root", type=click.Path(file_okay=False, path_type=Path),
              help="HTTP cache directory (outside the repo). Default: $ENGINE_CACHE or ../cache.")
@click.option("--funnel", "funnel_bed", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="BED restricting which variants are sent to the sources (see `engine funnel`). Omit for every ingested variant.")
@click.option("--clinvar-vcf", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Release-pinned ClinVar GRCh38 VCF (see `engine clinvar-download`).")
@click.option("--sources", default="vep,clinvar,gnomad", show_default=True,
              help="Comma-separated subset of vep,clinvar,gnomad.")
@click.option("--offline", is_flag=True, help="Serve everything from cache; fail on any cache miss.")
@click.option("--gnomad-max-af", type=float, default=0.05, show_default=True,
              help="Skip the gnomAD API for variants VEP already shows at or above this frequency.")
@click.option("--max-variants", type=int, help="Smoke-run cap on variants sent to the sources (recorded in the manifest).")
@click.option("--vep-workers", type=click.IntRange(1, 12), default=4, show_default=True, help="Concurrent VEP batches.")
def retrieve(run_dir: Path, cache_root: Path | None, funnel_bed: Path | None, clinvar_vcf: Path | None,
             sources: str, offline: bool, gnomad_max_af: float, max_variants: int | None, vep_workers: int) -> None:
    """Stage 2: fetch citable evidence records for every variant in scope."""
    from engine.retrieve.http import default_cache_root
    from engine.retrieve.run import RetrieveOptions, run_retrieve

    srcs = tuple(s.strip() for s in sources.split(",") if s.strip())
    bad = [s for s in srcs if s not in ("vep", "clinvar", "gnomad")]
    if bad:
        raise click.BadParameter(f"unknown sources: {bad}", param_hint="--sources")
    if "clinvar" in srcs and clinvar_vcf is None:
        raise click.BadParameter("--clinvar-vcf is required when clinvar is among --sources", param_hint="--clinvar-vcf")
    opts = RetrieveOptions(
        cache_root=cache_root or default_cache_root(), funnel_bed=funnel_bed, clinvar_vcf=clinvar_vcf,
        sources=srcs, offline=offline, gnomad_prefilter_max_af=gnomad_max_af, max_variants=max_variants,
        vep_workers=vep_workers, progress=lambda msg: click.echo("  " + msg),
    )
    click.echo(f"retrieve · {run_dir} · sources={','.join(srcs)} · "
               f"{'funnel=' + str(funnel_bed) if funnel_bed else 'all ingested variants'}"
               f"{' · OFFLINE' if offline else ''}")
    manifest = run_retrieve(run_dir, opts)
    c = json.loads(manifest.read_text())["counts"]
    click.echo(f"  in scope: {c['unique_variants']:,} variants · records: {c['evidence_records']:,} · "
               f"live requests: {c['http']['live_requests']:,} · cache hits: {c['http']['hits']:,}")
    click.echo(f"  manifest: {manifest}")


@main.command(name="clinvar-download")
@click.option("--config", "config_path", default="configs/clinvar.yaml", show_default=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Pinned release (date, url, md5, size).")
@click.option("--ref-dir", required=True, type=click.Path(file_okay=False, path_type=Path),
              help="Where to keep reference releases — outside the repository (e.g. ../ref).")
def clinvar_download(config_path: Path, ref_dir: Path) -> None:
    """Fetch the pinned ClinVar GRCh38 VCF release and verify its md5."""
    from engine.retrieve.clinvar import ClinvarRelease

    rel = ClinvarRelease.load(config_path)
    click.echo(f"clinvar-download · release {rel.date} · {rel.bytes:,} bytes · md5 {rel.md5}")
    vcf = rel.download(ref_dir)
    click.echo(f"  verified: {vcf}")


@main.command()
@click.option("--config", "config_path", default="configs/funnel.yaml", show_default=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Pinned GTF release, biotypes, padding.")
@click.option("--ref-dir", required=True, type=click.Path(file_okay=False, path_type=Path),
              help="Where the GTF is kept — outside the repository (e.g. ../ref).")
@click.option("--out", "out_bed", type=click.Path(dir_okay=False, path_type=Path),
              help="Output BED. Default: <ref-dir>/funnel.grch38.e<release>.pad<pad>.bed")
@click.option("--cache", "cache_root", type=click.Path(file_okay=False, path_type=Path),
              help="HTTP cache directory for the CHECKSUMS observation. Default: $ENGINE_CACHE or ../cache.")
@click.option("--offline", is_flag=True, help="Refuse any network access; the GTF must already be present.")
def funnel(config_path: Path, ref_dir: Path, out_bed: Path | None, cache_root: Path | None, offline: bool) -> None:
    """Tier A: build the exon/splice-proximal region BED from the release-pinned Ensembl GTF."""
    from engine.retrieve.funnel import build_funnel_bed, download_gtf, load_config
    from engine.retrieve.http import Http, HttpCache, default_cache_root

    cfg = load_config(config_path)
    http = Http(HttpCache(cache_root or default_cache_root()), offline=offline)
    click.echo(f"funnel · Ensembl release {cfg.release} · {cfg.filename} · pad {cfg.pad_bp} bp")
    gtf = download_gtf(cfg, ref_dir, http, progress=lambda msg: click.echo("  " + msg))
    out_bed = out_bed or (ref_dir / f"funnel.grch38.e{cfg.release}.pad{cfg.pad_bp}.bed")
    stats = build_funnel_bed(gtf, out_bed, cfg, progress=lambda msg: click.echo("  " + msg))
    click.echo(f"  {stats.n_genes:,} genes · {stats.n_transcripts:,} transcripts · {stats.n_intervals:,} intervals · "
               f"{stats.total_bp / 1e6:.1f} Mb ({stats.genome_fraction * 100:.2f}% of genome)")
    click.echo(f"  bed: {out_bed}")


# Stage packages register their own commands from ``engine.<package>.cli`` — a module
# that exposes ``commands: list[click.Command]``. Keeping each stage's CLI beside its
# code means adding a stage never edits this file. A stage that is not built yet falls
# back to a stub that says which phase delivers it.
STAGE_PACKAGES = {
    "filter": ("filter", "stage 3", "P3"),
    "rank": ("rank", "stage 4", "P4"),
    "reason": ("reason", "stage 5", "P5"),
    "medicine": ("medicine", "stage 6", "P6"),
    "report": ("report", "report", "P7"),
    "ui": ("ui", "live mode", "P7"),
    "bench": ("bench", "stage 7", "P7"),
    "submit": ("submit", "submission", "P8"),
    "mosaic": ("mosaic", "side analysis", "P8"),
}


def _planned(name: str, stage: str, phase: str) -> None:
    @main.command(name=name, help=f"({stage}) Planned for build phase {phase}; not implemented yet.")
    def _cmd() -> None:
        click.echo(f"engine {name}: {stage} is scheduled for phase {phase} and is not implemented yet.", err=True)
        sys.exit(2)


def _register_stage_commands() -> None:
    import importlib

    for name, (package, stage, phase) in STAGE_PACKAGES.items():
        try:
            module = importlib.import_module(f"engine.{package}.cli")
        except ModuleNotFoundError as e:
            if e.name != f"engine.{package}.cli" and e.name != f"engine.{package}":
                raise  # a real import error inside the stage, not a missing stage
            _planned(name, stage, phase)
            continue
        for cmd in getattr(module, "commands", []):
            main.add_command(cmd)


_register_stage_commands()


if __name__ == "__main__":
    main()
