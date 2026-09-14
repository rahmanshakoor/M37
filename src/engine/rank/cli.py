"""Stage 4 commands — registered by ``engine.cli`` from ``commands``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click


@click.command()
@click.option("--run", "run_dir", required=True, type=click.Path(file_okay=False, exists=True, path_type=Path),
              help="Run directory; output goes to <run>/04_rank/. The join reads <run>/03_filter/candidates.json when present.")
@click.option("--case", "case_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="case.yaml — the VCF path and the HPO terms come from here and nowhere else.")
@click.option("--exomiser-data", "data_dir", required=True, type=click.Path(file_okay=False, exists=True, path_type=Path),
              help="Directory holding 2406_hg38/ and 2406_phenotype/ (see `engine exomiser-download`).")
@click.option("--image", help="Docker image reference to run instead of the pinned repo@digest (recorded in the manifest).")
@click.option("--heap", help="JVM -Xmx (e.g. 6g). Default: the daemon's memory minus what its running containers "
                             "use (`docker stats`) minus 1 GiB. Required when another exomiser-cli container is running.")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Exomiser pin + analysis template. Default: configs/exomiser.yaml in the source tree.")
@click.option("--formats", default=None,
              help="Override outputFormats, e.g. JSON,TSV_GENE if the 14.0.0 variants-TSV writer crashes (issue #565).")
@click.option("--join-only", is_flag=True,
              help="Skip Docker: rebuild ranking.tsv and joined.json from the raw outputs already in 04_rank/exomiser/.")
@click.option("--regions", default=None, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="BED (e.g. the funnel BED): rank only the VCF records overlapping its intervals. The subset is "
                   "written to 04_rank/input.regions.vcf.gz (+ .tbi; M renamed to MT in it) and is what Exomiser sees, "
                   "so a whole genome need not fit the Docker daemon. Still gene-blind, no longer genome-wide — "
                   "the BED's sha256, the record counts before and after, and that restriction go in the manifest.")
def rank(run_dir: Path, case_path: Path, data_dir: Path, image: str | None, heap: str | None,
         config_path: Path | None, formats: str | None, join_only: bool, regions: Path | None) -> None:
    """Stage 4: blind phenotype ranking with Exomiser (HPO terms only, no gene list), joined onto the shortlist."""
    from engine.rank.exomiser import DEFAULT_CONFIG
    from engine.rank.run import ExomiserFailed, run_rank

    fmts = tuple(f.strip().upper() for f in formats.split(",") if f.strip()) if formats else None
    click.echo(f"rank · {run_dir} · exomiser-data={data_dir}{f' · regions={regions}' if regions else ''}"
               f"{' · JOIN ONLY' if join_only else ''}")
    try:
        manifest = run_rank(run_dir, case_path, data_dir, image=image, heap=heap, config_path=config_path or DEFAULT_CONFIG,
                            output_formats=fmts, join_only=join_only, regions=regions,
                            progress=lambda msg: click.echo("  " + msg))
    except (ExomiserFailed, FileNotFoundError, ValueError, RuntimeError) as e:  # RuntimeError: docker unreachable / pull failed
        click.echo(f"  FAILED: {e}", err=True)
        sys.exit(1)
    m = json.loads(manifest.read_text(encoding="utf-8"))
    c = m["counts"]
    click.echo(f"  genes ranked: {c['genes_ranked']:,} · candidates: {c['candidates']:,} "
               f"({c['candidates_ranked']:,} ranked by Exomiser) · exomiser-only shown: {c['exomiser_only']:,}"
               f" · wall {m['params'].get('wall_time_s', '?')}s")
    r = m["params"].get("regions")
    if r:
        click.echo(f"  restricted to regions: {r['records_after']:,} of {r['records_before']:,} records "
                   f"({r['bed_intervals']:,} intervals of {r['bed']}) — gene-blind, not genome-wide")
    click.echo(f"  manifest: {manifest}")


@click.command(name="exomiser-download")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Pinned data release (urls, sizes, hashes). Default: configs/exomiser.yaml in the source tree.")
@click.option("--ref-dir", required=True, type=click.Path(file_okay=False, path_type=Path),
              help="Where the bundles are kept and extracted — outside the repository (e.g. ../ref/exomiser). "
                   "This is the directory to pass as `engine rank --exomiser-data`.")
@click.option("--bundle", "names", multiple=True, type=click.Choice(["hg38", "phenotype"]),
              help="Only these bundles (default: all).")
@click.option("--extract/--no-extract", default=True, show_default=True, help="Unpack each verified zip.")
@click.option("--delete-zip", is_flag=True,
              help="Remove each zip after a verified extraction (the .verified.json sidecar keeps its hashes). "
                   "Needed on a disk that cannot hold ~70 GB of zips plus extracted data at once; extraction itself "
                   "needs free space for the members (hg38: ~32 GB) plus a 2 GiB margin, checked before it starts.")
@click.option("--verify-extracted", is_flag=True,
              help="Hash an already-extracted bundle against the pinned member sha256s and the checksum list it ships "
                   "(reads the whole 30 GB store once) and record the result in the .verified.json sidecar.")
def exomiser_download(config_path: Path | None, ref_dir: Path, names: tuple[str, ...], extract: bool, delete_zip: bool,
                      verify_extracted: bool) -> None:
    """Fetch (resumably), verify and extract the pinned Exomiser data release."""
    from engine.rank.download import PinMismatch, fetch_all
    from engine.rank.exomiser import DEFAULT_CONFIG, ExomiserConfig

    cfg = ExomiserConfig.load(config_path or DEFAULT_CONFIG)
    total = sum(b.bytes for n, b in cfg.bundles.items() if not names or n in names)
    click.echo(f"exomiser-download · data release {cfg.data_version} · {total / (1 << 30):.1f} GiB of zips → {ref_dir}")
    try:
        report = fetch_all(cfg, ref_dir, names=names or None, do_extract=extract, delete_zip=delete_zip,
                           verify_members=verify_extracted, progress=lambda msg: click.echo("  " + msg, err=False))
    except (PinMismatch, RuntimeError) as e:
        click.echo(f"  FAILED: {e}", err=True)
        sys.exit(1)
    for name, facts in report.items():
        members = facts.get("members") or {}
        verified = sum(1 for v in members.values() if v.get("matches_pin") or v.get("matches_checksum_list"))
        click.echo(f"  {name}: {facts.get('status')} · sha256 {facts.get('sha256', '?')} · "
                   f"members hashed {len(members)} (verified against a pin or list: {verified})")


commands: list[click.Command] = [rank, exomiser_download]
