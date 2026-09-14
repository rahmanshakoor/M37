"""The ``ui`` command — registered by ``engine.cli`` from ``commands``.

``engine ui`` with no option serves the page on ``127.0.0.1`` (the only address it
ever binds; there is no option to change it) over the *project directory* — the
directory above the checkout, where the case file lives — and finds everything
else there (:mod:`engine.ui.setup`): runs under ``<project>/work``, the HTTP cache
under ``<project>/cache``, and under ``<project>/ref`` the newest pinned ClinVar
VCF, the newest funnel BED and the Exomiser data bundle when it is verified. Every
one of those can still be given: an option first, then the environment variables
``scripts/run_public_case.sh`` uses (``ENGINE_CACHE``, ``CLINVAR_VCF``,
``EXOMISER_DATA``), then discovery. What was found and what is missing is printed
as paths and states — never a key, never anything from a case file — and the
page's Setup panel shows the same and can create what is missing.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import click


@click.command(name="ui")
@click.option("--project", "project_dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Directory holding case.yaml, ref/, cache/ and work/. Default: the directory above the checkout.")
@click.option("--work", "work_dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Directory whose sub-directories are the runs (created if missing). Default: <project>/work.")
@click.option("--port", type=click.IntRange(0, 65535), default=8765, show_default=True,
              help="TCP port on 127.0.0.1 (0 picks a free one).")
@click.option("--repo", "repo_dir", type=click.Path(file_okay=False, exists=True, path_type=Path), default=None,
              help="Checkout `uv run engine` executes in. Default: the one this command was installed from.")
@click.option("--ref", "ref_dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Where reference releases are kept and looked for. Default: <project>/ref.")
@click.option("--cache", "cache_root", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="HTTP cache passed to retrieve/reason/medicine as --cache. Default: $ENGINE_CACHE, else <project>/cache.")
@click.option("--clinvar-vcf", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Release-pinned ClinVar VCF for `retrieve`. Default: $CLINVAR_VCF, else the newest clinvar_*.vcf.gz in --ref.")
@click.option("--exomiser-data", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Extracted Exomiser data bundle for `rank`. Default: $EXOMISER_DATA, else <ref>/exomiser when verified.")
@click.option("--funnel", "funnel_bed", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Funnel BED for `retrieve --funnel` and `rank --regions`. Default: the newest funnel*.bed in --ref.")
def ui(project_dir: Path | None, work_dir: Path | None, port: int, repo_dir: Path | None, ref_dir: Path | None,
       cache_root: Path | None, clinvar_vcf: Path | None, exomiser_data: Path | None, funnel_bed: Path | None) -> None:
    """Live mode: browse runs, read every stage, run a case end to end and watch it."""
    from engine.ui.jobs import repo_root
    from engine.ui.server import serve
    from engine.ui.setup import discover, project_root

    repo = (repo_dir or repo_root()).resolve()
    project = (project_dir or project_root(repo)).resolve()
    sources: dict[str, str] = {}
    cache = _pick(sources, "cache", cache_root, "ENGINE_CACHE")
    clinvar = _pick(sources, "clinvar_vcf", clinvar_vcf, "CLINVAR_VCF")
    exomiser = _pick(sources, "exomiser_data", exomiser_data, "EXOMISER_DATA")
    if funnel_bed is not None:
        sources["funnel_bed"] = "option"
    if work_dir is not None:
        sources["work"] = "option"
    discovery = functools.partial(discover, project, repo, work=work_dir, ref=ref_dir, cache=cache, clinvar_vcf=clinvar,
                                  funnel_bed=funnel_bed, exomiser_data=exomiser, sources=sources)
    found = discovery()
    config = found.runner_config()
    click.echo(f"ui · project={project} · work={found.work}" + (f" · repo={repo}" if repo_dir else ""))
    for item in found.items:
        click.echo(f"  {item.name}: {item.path or 'not configured'} · {item.state}"
                   + (f" ({item.source})" if item.source not in ("discovered", "default") else "")
                   + (f" — {item.command}" if item.state != "ready" and item.command else ""))
    serve(found.work, port, on_ready=lambda url: click.echo(f"  serving on {url} (Ctrl-C stops it)"),
          repo=repo, config=config, discovery=discovery, project=project)


def _pick(sources: dict[str, str], name: str, option: Path | None, env: str) -> Path | None:
    if option is not None:
        sources[name] = "option"
        return option
    value = os.environ.get(env)
    if value:
        sources[name] = "env"
        return Path(value)
    return None


ui_command = ui

commands: list[click.Command] = [ui]
