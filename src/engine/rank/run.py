"""Stage 4 orchestrator — ``engine rank``.

Exomiser sees the case VCF and the HPO terms, nothing else: no gene panel, no
shortlist, no prior. That makes its ranking an independent witness — where it agrees
with stage 3 the agreement means something, and where it ranks a gene the shortlist
does not contain, the disagreement is written down (``exomiser_only``) rather than
lost. The stage therefore never feeds stage-3 results *into* Exomiser; it only joins
them *onto* the ranking afterwards.

What is written, and why in this form:

* ``exomiser/`` — the raw run: the generated ``config/analysis.yml`` (HPO terms and the
  container path of the VCF; its sha256 is in the manifest), ``config/application.properties``,
  the container log (ending in the exit code), ``run.json`` (the facts of the container
  run: exit code, image digest observed, heap, command, wall time, the sha256 of the VCF
  and of the config it ran with — what ``--join-only`` must not lose or invent), and
  Exomiser's own ``exomiser.genes.tsv`` / ``exomiser.variants.tsv`` / ``exomiser.json``.
  A judge reads the tool's output, not our reading of it.
* ``ranking.tsv`` — one row per gene (best MOI), in Exomiser's order.
* ``joined.json`` — every stage-3 candidate with its Exomiser rank and scores (``null``
  when Exomiser did not rank the gene), plus the top-N Exomiser genes that are not in
  the shortlist. Downstream stages read this, not the raw files.
* ``evidence/`` — one ``exomiser:<gene>`` record per gene that appears in ``joined.json``,
  in the stage-2 record format, so a later claim about the ranking can cite it. The
  payload carries the gene's raw ``genes.tsv`` rows (phenotype-match evidence, per MOI)
  and its contributing ``variants.tsv`` rows, so the record explains *why* the gene
  ranked where it did, not just where. ``retrieved_at`` is the moment the container
  finished (``run.json``), because that is when the evidence came into being — so any
  re-join of the same raw directory, anywhere, reproduces the record bytes.

A run starts by removing every derived file of the previous run in the directory
(``ranking.tsv``, ``joined.json``, ``evidence/``, ``manifest.json``) as well as the raw
outputs: a failed container must leave behind its log and ``run.json`` and nothing
that looks like a result, so ``04_rank/`` never presents a stale success next to a
failed run's ``analysis.yml``.

``--regions <bed>`` restricts what Exomiser sees, not what it knows: the container gets
``04_rank/input.regions.vcf.gz`` — the case VCF's records overlapping the BED (``bcftools
view -R``), with ``M`` → ``MT`` renamed in that subset when the header spells it so, indexed
(``.tbi``) — and the manifest records the BED's sha256, the records before and after, the
rename and the bcftools lines (``params.regions``, also in ``run.json``), lists the subset
as an output, and notes that the ranking is gene-blind (HPO terms only, no gene list) but
restricted to the BED's regions: a gene with no variant inside them cannot rank. The
subset is kept: it is the file Exomiser saw, small by construction, and ``--join-only``
checks it still hashes to what ``run.json`` recorded. The whole genome is never copied.

Determinism: the derived files carry no join-time timestamp, sort their keys and omit
Exomiser's sampled ``P-VALUE`` (which differs between identical runs — it stays in the
raw files), so a rerun over the same raw outputs is byte-identical (``--join-only``
proves it). The wall time, the image digest the daemon resolved and the banner version
go in the manifest and ``run.json`` only.

``--join-only`` rebuilds the derived files from the raw outputs *as they were produced*,
and refuses a raw directory it cannot vouch for: ``run.json`` must exist and record exit
0; the on-disk ``analysis.yml`` must still hash to the sha256 ``run.json`` recorded; the
case VCF must still hash to the sha256 ``run.json`` recorded. The HPO terms, the analysis
template, the output formats, the data version and the config identity in the manifest
then all describe the run as it was — never the current ``case.yaml`` or pin, which may
have changed since (a difference is noted). Only ``03_filter/candidates.json`` is read
as it is now: joining a later stage-3 result onto the same ranking is what the flag is
for.

The join is by gene symbol, then by Ensembl gene id (from the JSON) when the symbols
differ — the 2406 release carries Ensembl 111 symbols and stage 2 annotates with a
newer VEP. Variant keys are compared exactly (``chrom:pos:ref:alt``): Exomiser trims
alleles to their minimal form with one shared anchor base (svart's left-shifting
trimmer), which is what a normalising caller and ``bcftools norm`` emit too, so a
stage-1 row and its Exomiser counterpart normally share a key; a mismatch shows up as
``candidates_with_variant_match`` falling short, never as a guessed match.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from engine.config import CaseConfig
from engine.contigs import naming_style
from engine.manifest import Manifest, sha256_file
from engine.rank.download import members_status, read_verified
from engine.rank.exomiser import (
    ANALYSIS_FILENAME, CASE_KEYS, DEFAULT_CONFIG, PROPERTIES_FILENAME, REGIONS_OVERLAP, REGIONS_VCF_FILENAME,
    ExomiserConfig, GeneRanking, GeneRow, VariantRow, bed_interval_count, build_ranking, container_vcf_name,
    contig_renames, contributing_from_json, contributing_from_tsv, count_records, docker_command,
    docker_info_memory_bytes, docker_load, gene_identifiers_from_json, heap_available_bytes, heap_bytes, heap_for,
    image_digest, load_json_genes, mito_contig, parse_genes_tsv, parse_variants_tsv, rename_contigs, render_analysis,
    render_properties, sha256_text, subset_regions, vcf_header_contigs, vcf_record_count, version_from_log,
)
from engine.retrieve.store import EvidenceRecord, EvidenceStore

STAGE_DIR = "04_rank"
RAW_DIR = "exomiser"
CONFIG_SUBDIR = "config"
INPUT_SUBDIR = "input"
"""Where a contig-renamed copy of the VCF lives for the container's benefit (removed after the run)."""
LOG_FILENAME = "exomiser.log"
RUN_FILENAME = "run.json"
FILTER_DIR = "03_filter"
INGEST_DIR = "01_ingest"

RANKING_COLUMNS = ("rank", "gene_symbol", "exomiser_score", "phenotype_score", "variant_score", "moi", "n_variants", "variants")

RUN_FACT_KEYS = ("exomiser_exit_code", "image_requested", "image_observed", "image_matches_pin", "image_digest_used",
                 "heap", "docker_command", "wall_time_s", "docker_memory_bytes", "docker_memory_in_use_bytes",
                 "docker_containers_running", "docker_exomiser_containers_running", "vcf_container_path",
                 "vcf_container_sha256", "vcf_container_bytes", "vcf_contig_renames", "regions")
"""Manifest params that describe the container run; ``run.json`` keeps them for ``--join-only``."""

DERIVED_FILES = ("ranking.tsv", "joined.json", "manifest.json")
DERIVED_DIRS = ("evidence",)
"""What a run derives from the raw outputs — removed before the container starts."""

OMITTED_COLUMNS = {"P-VALUE": "sampled by Exomiser and not reproducible between runs; read it from the raw TSV"}
"""Raw TSV columns left out of the evidence payload, with the reason."""

Progress = Callable[[str], None]
Runner = Callable[..., subprocess.CompletedProcess]


class ExomiserFailed(RuntimeError):
    """The container exited non-zero or wrote no results, or the stage refused to start
    or to join; the log path (or the reason) is in the message."""


# ---------------------------------------------------------------- the run

def run_rank(
    run_dir: Path,
    case_path: Path,
    exomiser_data_dir: Path,
    image: str | None = None,
    *,
    heap: str | None = None,
    config_path: Path = DEFAULT_CONFIG,
    output_formats: tuple[str, ...] | None = None,
    join_only: bool = False,
    regions: Path | None = None,
    progress: Progress = lambda s: None,
    run: Runner = subprocess.run,
) -> Path:
    """Run stage 4 for the case into ``run_dir/04_rank``. Returns the manifest path.

    ``run`` is the subprocess runner (injected by tests). ``join_only`` skips Docker and
    rebuilds the derived files from the raw outputs already in ``04_rank/exomiser/``.
    ``regions`` (a BED) restricts the VCF the container sees to the records overlapping
    its intervals — see the module docstring; with ``join_only`` it is only compared with
    the BED the raw run used."""
    run_dir = Path(run_dir)
    regions = Path(regions) if regions is not None else None
    out_dir = run_dir / STAGE_DIR
    raw_dir = out_dir / RAW_DIR
    config_dir = raw_dir / CONFIG_SUBDIR
    config_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(exomiser_data_dir)

    case = CaseConfig.load(case_path)
    cfg = ExomiserConfig.load(config_path)
    formats = tuple(f.upper() for f in output_formats) if output_formats else cfg.output_formats
    top_n = int(cfg.join.get("top_n_not_in_shortlist", 20))

    m = Manifest(stage="rank")
    m.add_input("case", Path(case_path))
    m.add_input("vcf", case.vcf)
    for ext in (".tbi", ".csi"):
        idx = case.vcf.with_name(case.vcf.name + ext)
        if idx.exists():
            m.add_input("vcf_index", idx, checksum=False)
            break
    m.add_input("config", cfg.path)
    if regions is not None and not join_only:
        m.add_input("regions", regions)
    m.tools["python"] = sys.version.split()[0]
    m.add_tool("docker")

    candidates_path = run_dir / FILTER_DIR / "candidates.json"
    candidates: dict[str, Any] | None = None
    if candidates_path.exists():
        m.add_input("candidates", candidates_path)
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    else:
        m.note(f"stage 3 has not run ({candidates_path} missing): joined.json has no candidates; "
               f"rerun with --join-only once it has.")

    contigs, first_chrom = vcf_header_contigs(case.vcf)
    style = _naming_style(contigs, first_chrom)
    mito = mito_contig(contigs)
    renames = contig_renames(contigs)

    vcf_mount = str(cfg.docker.get("vcf_mount", "/vcf"))
    results_mount = str(cfg.docker.get("results_mount", "/results"))
    vcf_name = container_vcf_name(case.vcf)
    analysis_text = render_analysis(cfg, case.hpo, f"{vcf_mount}/{vcf_name}", results_mount, formats)
    analysis_sha = sha256_text(analysis_text)
    analysis_path = config_dir / ANALYSIS_FILENAME
    properties_path = config_dir / PROPERTIES_FILENAME
    log_path = raw_dir / LOG_FILENAME
    run_path = raw_dir / RUN_FILENAME
    paths = cfg.output_paths(raw_dir)
    bundle_facts = _bundle_facts(cfg, data_dir)

    m.params.update({
        "config_path": str(cfg.path),
        "config_sha256": cfg.sha256,
        "hpo": list(case.hpo),
        "n_hpo": len(case.hpo),
        "analysis_yaml": str(analysis_path),
        "analysis_yaml_sha256": analysis_sha,
        "analysis": cfg.analysis,
        "output_formats": list(formats),
        "output_options": {k: v for k, v in cfg.output_options.items() if k != "outputFormats"},
        "image_requested": image or cfg.image_ref,
        "image_pinned": {"repository": cfg.image_repository, "tag": cfg.image_tag, "digest": cfg.image_digest,
                         "version": cfg.image_version, "platform_digests": cfg.platform_digests},
        "data_version": cfg.data_version,
        "data_dir": str(data_dir),
        "data_citation": cfg.data_citation,
        "data_bundles": bundle_facts,
        "vcf_contig_style": style,
        "vcf_contigs_declared": len(contigs),
        "vcf_mito_contig": mito,
        "vcf_container_path": f"{vcf_mount}/{vcf_name}",
        "vcf_contig_renames": renames,
        "join_top_n_not_in_shortlist": top_n,
        "join_only": join_only,
        "regions": None,
        "omitted_columns": OMITTED_COLUMNS,
    })
    for name, facts in bundle_facts.items():
        ms = facts["members"]
        if ms["pinned_mismatched"]:
            raise FileNotFoundError(f"Bundle {name} under {data_dir}: the download sidecar's sha256 for "
                                    f"{ms['pinned_mismatched']} differs from the pin in {cfg.path}; the extracted "
                                    f"data is not the pinned release")
        if ms["pinned"] and not ms["all_pinned_verified"]:
            m.note(f"Bundle {name}: {ms['pinned_verified']} of {ms['pinned']} pinned member files have a verified "
                   f"sha256 in the download sidecar; run `engine exomiser-download --ref-dir {data_dir} "
                   f"--verify-extracted` to hash the extracted files against the pin.")

    if join_only:
        facts = _require_raw(paths["genes"], analysis_path, run_path, log_path, m.inputs["vcf"]["sha256"])
        on_disk = analysis_path.read_text(encoding="utf-8")
        _describe_run_as_it_was(m, facts, run_path, on_disk, case, cfg, formats, analysis_text)
        _describe_regions_as_they_were(m, facts, out_dir / REGIONS_VCF_FILENAME, regions)
        hpo = m.params["hpo"]
        m.note("join-only: Exomiser was not run; ranking.tsv and joined.json were rebuilt from the raw outputs present.")
        log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        image_used = str(facts.get("image_digest_used") or cfg.image_digest)
        finished_at = str(facts.get("finished_at") or _now())
    else:
        missing = cfg.missing_data(data_dir)
        if missing:
            raise FileNotFoundError(
                f"Exomiser data release {cfg.data_version} is incomplete under {data_dir}: {missing}. "
                f"Run `engine exomiser-download --ref-dir {data_dir}` first."
            )
        image_ref = image or cfg.image_ref
        observed = _ensure_image(image_ref, run, progress)
        m.params["image_observed"] = observed
        m.params["image_matches_pin"] = carries_digest(observed, cfg.image_digest)
        image_used = image_digest_used(observed, cfg.image_digest)
        m.params["image_digest_used"] = image_used
        if not m.params["image_matches_pin"]:
            m.note(f"Image {image_ref} does not carry the pinned digest {cfg.image_digest}: "
                   f"observed {observed.get('repo_digests')} / id {observed.get('id')}.")
        heap_value = _size_heap(m, cfg, heap, run)  # may refuse — before anything of the previous run is touched

        _clear_previous_run(out_dir, paths, log_path, run_path)
        analysis_path.write_text(analysis_text, encoding="utf-8")
        properties_path.write_text(render_properties(cfg), encoding="utf-8")

        container_vcf = case.vcf
        if regions is not None:
            m.add_tool("bcftools")
            container_vcf = _restrict_to_regions(m, case.vcf, regions, renames, mito, out_dir, run_dir, run, progress)
        elif renames:
            m.add_tool("bcftools")
            container_vcf = rename_contigs(case.vcf, renames, out_dir / INPUT_SUBDIR / vcf_name, run)
            m.params["vcf_container_sha256"] = sha256_file(container_vcf)
            m.params["vcf_container_bytes"] = container_vcf.stat().st_size
            m.note(f"The VCF spells the mitochondrion '{mito}', which Exomiser 14.0.0 (svart) cannot resolve — its "
                   f"MT records would be skipped without a warning. The container was given a copy with "
                   f"{renames} renamed in the header and records (bcftools annotate --rename-chrs); its sha256 is "
                   f"params.vcf_container_sha256, the original's is inputs.vcf. The copy was removed after the run.")
        else:
            m.params["vcf_container_sha256"] = m.inputs["vcf"]["sha256"]
            m.params["vcf_container_bytes"] = m.inputs["vcf"]["bytes"]

        cmd = docker_command(cfg, image=image_ref, data_dir=data_dir, config_dir=config_dir, results_dir=raw_dir,
                             vcf=container_vcf, vcf_container_name=vcf_name, heap=heap_value)
        m.params["heap"] = heap_value
        m.params["docker_command"] = cmd
        progress(f"exomiser {cfg.image_version} · data {cfg.data_version} · heap {heap_value} · {len(case.hpo)} HPO terms")
        t0 = time.monotonic()
        try:
            with open(log_path, "w", encoding="utf-8") as log:
                log.write("$ " + " ".join(cmd) + "\n\n")
                log.flush()
                proc = run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
                log.write(f"\n[engine] exit code {proc.returncode}\n")
        finally:
            if renames:
                shutil.rmtree(out_dir / INPUT_SUBDIR, ignore_errors=True)
        wall = round(time.monotonic() - t0, 1)
        finished_at = _now()
        m.params["wall_time_s"] = wall
        m.params["exomiser_exit_code"] = proc.returncode
        facts = {k: m.params.get(k) for k in RUN_FACT_KEYS}
        facts.update(
            finished_at=finished_at, analysis_yaml_sha256=analysis_sha, hpo=list(case.hpo), output_formats=list(formats),
            exomiser_version_pinned=cfg.image_version, data_version=cfg.data_version,
            vcf_path=str(case.vcf), vcf_sha256=m.inputs["vcf"]["sha256"], vcf_bytes=m.inputs["vcf"]["bytes"],
            vcf_mito_contig=mito, config_path=str(cfg.path), config_sha256=cfg.sha256, config_bytes=m.inputs["config"]["bytes"],
            candidates_sha256=(m.inputs["candidates"]["sha256"] if "candidates" in m.inputs else None),
        )
        run_path.write_text(json.dumps(facts, sort_keys=True, indent=1) + "\n", encoding="utf-8")  # kept even when the run failed
        log_text = log_path.read_text(encoding="utf-8")
        if proc.returncode != 0:
            raise ExomiserFailed(_failure_message(proc.returncode, log_path, log_text, formats))
        if not paths["genes"].exists():
            raise ExomiserFailed(f"Exomiser exited 0 but wrote no genes TSV at {paths['genes']}; see {log_path}")
        hpo = list(case.hpo)
        progress(f"exomiser finished in {wall}s")

    version = version_from_log(log_text)
    m.tools["exomiser"] = version or "unknown (no banner in log)"
    if version and version != cfg.image_version:
        m.note(f"Exomiser banner says v{version}; the pin says {cfg.image_version}.")

    # ---- parse the raw outputs
    genes = parse_genes_tsv(paths["genes"])
    json_genes = load_json_genes(paths["json"]) if paths["json"].exists() else None
    contributing_source = "none"
    variants: list[VariantRow] = []
    if paths["variants"].exists():
        variants = parse_variants_tsv(paths["variants"])
        contributing = contributing_from_tsv(variants)
        contributing_source = "variants_tsv"
        m.counts["variant_rows"] = len(variants)
    elif json_genes is not None:
        contributing = contributing_from_json(json_genes)
        contributing_source = "json"
        if "TSV_VARIANT" in formats:
            m.note("exomiser.variants.tsv was requested but not written (the 14.0.0 TSV_VARIANT writer aborts on a "
                   "variant whose stored frequencies are all 0.0, issue #565); contributing variants come from the JSON.")
    else:
        contributing = {}
        m.note("Neither the variants TSV nor the JSON was written: the ranking has no contributing variants.")
    gene_ids = gene_identifiers_from_json(json_genes) if json_genes is not None else {}
    del json_genes  # a genome's JSON is large; nothing below needs it
    ranking = build_ranking(genes, contributing)
    m.params["contributing_variants_source"] = contributing_source

    # ---- derived files
    ranking_path = out_dir / "ranking.tsv"
    write_ranking(ranking_path, ranking)
    joined, join_counts = join_candidates(candidates, ranking, gene_ids, top_n)
    joined["hpo"] = list(hpo)
    joined["analysis_yaml_sha256"] = m.params["analysis_yaml_sha256"]
    joined["exomiser_version"] = version
    joined["data_version"] = m.params["data_version"]
    joined["counts"] = join_counts
    cited = write_evidence(out_dir / "evidence", joined, ranking, genes, variants, gene_ids, cfg, version,
                           m.params["analysis_yaml_sha256"], hpo, image_used, m.params["data_version"], finished_at)
    joined_path = out_dir / "joined.json"
    joined_path.write_text(json.dumps(joined, sort_keys=True, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    m.counts.update({
        "gene_moi_rows": len(genes),
        "genes_ranked": len(ranking),
        "genes_with_contributing_variants": sum(1 for g in ranking if g.variants),
        "contributing_variant_keys": len({v for g in ranking for v in g.all_variants()}),
        "evidence_records": cited,
        **join_counts,
    })
    _note_regions(m, out_dir / REGIONS_VCF_FILENAME)
    for name, p in paths.items():
        if p.exists():
            m.add_output(f"exomiser_{name}", p)
    m.add_output("analysis_yaml", analysis_path)
    if log_path.exists():
        m.add_output("exomiser_log", log_path)
    if run_path.exists():
        m.add_output("exomiser_run", run_path)
    m.add_output("ranking", ranking_path)
    m.add_output("joined", joined_path)
    m.add_output("evidence_index", out_dir / "evidence" / "index.json")
    manifest_path = out_dir / "manifest.json"
    m.write(manifest_path)
    return manifest_path


# ---------------------------------------------------------------- pieces

def _naming_style(contigs: list[str], first_chrom: str | None) -> str:
    try:
        if contigs:
            return naming_style(contigs)
    except ValueError:
        pass
    if first_chrom:
        return "ucsc" if first_chrom.lower().startswith("chr") else "ensembl"
    return "unknown"


def _bundle_facts(cfg: ExomiserConfig, data_dir: Path) -> dict[str, Any]:
    """What is known about each bundle on disk: the pin, the download sidecar when
    ``engine exomiser-download`` wrote one (its verified hashes are the provenance), and
    how many of the pinned member files have a verified sha256."""
    out: dict[str, Any] = {}
    for name, b in cfg.bundles.items():
        facts: dict[str, Any] = {"filename": b.filename, "pinned_bytes": b.bytes, "pinned_md5": b.md5,
                                 "pinned_sha256": b.sha256, "extracted_dir": str(b.extracted_path(data_dir)),
                                 "missing_files": b.missing_files(data_dir), "members": members_status(b, data_dir)}
        meta = read_verified(data_dir / b.filename)
        if meta:
            facts["verified"] = {k: meta.get(k) for k in ("bytes", "md5", "sha256", "verified_at")}
        out[name] = facts
    return out


def _clear_previous_run(out_dir: Path, paths: dict[str, Path], log_path: Path, run_path: Path) -> None:
    """Nothing of an earlier run may survive into this one's directory: neither its raw
    outputs (a genes TSV alone would be mistaken for this run's) nor the files derived
    from them, nor its manifest — a failed container leaves a log and ``run.json``, not
    a stale success."""
    for p in list(paths.values()) + [log_path, run_path]:
        p.unlink(missing_ok=True)
    for name in DERIVED_FILES:
        (out_dir / name).unlink(missing_ok=True)
    for name in DERIVED_DIRS:
        shutil.rmtree(out_dir / name, ignore_errors=True)
    shutil.rmtree(out_dir / INPUT_SUBDIR, ignore_errors=True)
    for stale in out_dir.glob(REGIONS_VCF_FILENAME + "*"):  # the subset, its .tbi, rename map and temp
        stale.unlink(missing_ok=True)


def _size_heap(m: Manifest, cfg: ExomiserConfig, heap: str | None, run: Runner) -> str:
    """Pick ``-Xmx`` and record what it was sized from: the daemon's total, what its
    running containers already use, and whether another exomiser-cli is among them
    (refused unless ``--heap`` is explicit — two auto-sized JVMs cannot share a daemon)."""
    daemon_bytes = docker_info_memory_bytes(run)
    load = docker_load(run)
    m.params["docker_memory_bytes"] = daemon_bytes
    m.params["docker_memory_in_use_bytes"] = load.in_use_bytes
    m.params["docker_containers_running"] = len(load.containers)
    m.params["docker_exomiser_containers_running"] = load.exomiser_containers
    auto = heap is None and str(cfg.docker.get("heap", "auto")) == "auto"
    if load.exomiser_containers:
        if heap is None:
            raise ExomiserFailed(f"refusing to start: another exomiser-cli container is running on this daemon "
                                 f"({', '.join(load.exomiser_containers)}); its JVM will grow into the memory an "
                                 f"auto-sized heap would be cut from. Wait for it, or pass --heap explicitly.")
        m.note(f"Another exomiser-cli container was running when this one started "
               f"({', '.join(load.exomiser_containers)}); --heap {heap} was taken as given.")
    heap_value = heap or heap_for(cfg, daemon_bytes, load.in_use_bytes)
    if auto and daemon_bytes is None:
        m.note(f"`docker info` did not report the daemon's memory; the heap fell back to {heap_value} "
               "instead of being sized from it. Pass --heap to size it explicitly.")
    elif auto and load.in_use_bytes is None:
        m.note("`docker stats` could not be read: the heap was sized from the daemon's total memory alone, "
               "not from what its running containers leave free.")
    available = heap_available_bytes(cfg, daemon_bytes, load.in_use_bytes)
    wanted = heap_bytes(heap_value)
    if available is not None and wanted is not None and wanted > available:
        m.note(f"Heap {heap_value} exceeds what the daemon can back: {daemon_bytes} bytes total minus "
               f"{load.in_use_bytes or 0} in use by {len(load.containers)} running containers minus the "
               f"{cfg.docker.get('heap_reserve_mib', 1024)} MiB reserve leaves {max(available, 0)} bytes; "
               "expect exit 137 (OOM-killed) on a large VCF.")
    return heap_value


def _require_raw(genes_path: Path, analysis_path: Path, run_path: Path, log_path: Path, vcf_sha256: str) -> dict[str, Any]:
    """The raw directory must be a *completed* run of *this* VCF with *this* analysis
    file: Exomiser writes the genes TSV before the writers that can crash, so a genes
    TSV alone proves nothing; an analysis.yml edited after the run, or a case VCF swapped
    since, would attribute the raw outputs to inputs that did not produce them."""
    if not genes_path.exists():
        raise FileNotFoundError(f"--join-only needs Exomiser's genes TSV at {genes_path}; run `engine rank` without it first")
    if not analysis_path.exists():
        raise FileNotFoundError(f"--join-only needs the analysis YAML of the run at {analysis_path}")
    if not run_path.exists():
        raise FileNotFoundError(f"--join-only needs {run_path} (written when the container exits); the raw outputs "
                                f"present are from an interrupted run or an engine version that did not record it — "
                                f"run `engine rank` without --join-only")
    facts = json.loads(run_path.read_text(encoding="utf-8"))
    code = facts.get("exomiser_exit_code")
    if code != 0:
        raise ExomiserFailed(f"--join-only refused: the raw outputs are from a run that exited with code {code}; "
                             f"see {log_path}")
    recorded = facts.get("analysis_yaml_sha256")
    on_disk = sha256_text(analysis_path.read_text(encoding="utf-8"))
    if recorded and on_disk != recorded:
        raise ExomiserFailed(f"--join-only refused: {analysis_path} (sha256 {on_disk}) is not the file the run used "
                             f"(sha256 {recorded} in {run_path}); it was edited after the run")
    if facts.get("vcf_sha256") and facts["vcf_sha256"] != vcf_sha256:
        raise ExomiserFailed(f"--join-only refused: the case VCF (sha256 {vcf_sha256}) is not the file the raw run "
                             f"ranked (sha256 {facts['vcf_sha256']} in {run_path}); run `engine rank` without --join-only")
    return facts


def _describe_run_as_it_was(
    m: Manifest,
    facts: dict[str, Any],
    run_path: Path,
    on_disk: str,
    case: CaseConfig,
    cfg: ExomiserConfig,
    formats: tuple[str, ...],
    analysis_now: str,
) -> None:
    """Point the manifest's params at the run being joined — the terms, template, output
    formats, data version and config identity that produced the raw outputs — and note
    every way the current case, config or options differ from them."""
    hpo = hpo_from_analysis(on_disk)
    template, output_options = analysis_from_yaml(on_disk)
    formats_then = [str(f) for f in (facts.get("output_formats") or output_options.get("outputFormats") or [])]
    m.params["analysis_yaml_sha256"] = sha256_text(on_disk)
    m.params["hpo"] = hpo
    m.params["n_hpo"] = len(hpo)
    m.params["analysis"] = template
    m.params["output_formats"] = formats_then
    m.params["output_options"] = {k: v for k, v in output_options.items() if k != "outputFormats"}
    if facts.get("data_version"):
        m.params["data_version"] = str(facts["data_version"])
    if facts.get("config_sha256"):
        m.params["config_path"] = facts.get("config_path")
        m.params["config_sha256"] = facts["config_sha256"]
        m.inputs["config"] = {"path": facts.get("config_path"), "bytes": facts.get("config_bytes"),
                              "sha256": facts["config_sha256"], "as_of": "the run being joined (run.json)"}
        if facts["config_sha256"] != cfg.sha256:
            m.note(f"join-only: the config now at {cfg.path} (sha256 {cfg.sha256}) is not the one the run used "
                   f"(sha256 {facts['config_sha256']}); params.analysis and inputs.config describe the run as it was.")
    m.params.update({k: facts.get(k) for k in RUN_FACT_KEYS})
    m.params["run_facts_source"] = str(run_path)
    if facts.get("vcf_contig_renames") and not facts.get("regions"):
        m.note(f"join-only: the container ranked a copy of the VCF with contigs {facts['vcf_contig_renames']} renamed "
               f"(sha256 {facts.get('vcf_container_sha256')}); inputs.vcf is the original.")
    if not facts.get("vcf_sha256"):
        m.note("join-only: run.json records no VCF sha256 (an earlier engine version wrote it); whether the case VCF "
               "is still the file that was ranked could not be verified.")
    if hpo != list(case.hpo):
        m.note(f"join-only: case.yaml now lists {len(case.hpo)} HPO terms but the raw run used {len(hpo)}; "
               "the join, joined.json and the evidence records carry the terms of the run as it was.")
    elif on_disk != analysis_now:
        m.note("join-only: the analysis template differs from what the config would render now; the join is "
               "against the run as it was.")
    if formats_then and list(formats) != formats_then:
        m.note(f"join-only: output formats {list(formats)} were requested but the run wrote {formats_then}; "
               "params.output_formats is the run's.")


def _restrict_to_regions(
    m: Manifest,
    vcf: Path,
    bed: Path,
    renames: dict[str, str],
    mito: str | None,
    out_dir: Path,
    run_dir: Path,
    run: Runner,
    progress: Progress,
) -> Path:
    """Write the ``--regions`` subset the container will see and record everything about
    it: the BED, the records before and after, the rename applied to the subset, the
    bcftools lines. Refuses an empty subset — Exomiser would rank nothing, and the usual
    cause (BED and VCF spelling contigs differently) is better said here than as an
    empty genes TSV."""
    subset = subset_regions(vcf, bed, out_dir / REGIONS_VCF_FILENAME, renames, run=run)  # first: a wrong BED fails fast
    before, before_source = _records_before(vcf, run_dir, m.inputs["vcf"]["sha256"], run)
    intervals = bed_interval_count(bed)
    if subset.records == 0:
        for p in (subset.path, subset.index, subset.path.with_name(subset.path.name + ".rename.tsv")):
            p.unlink(missing_ok=True)
        raise ExomiserFailed(f"--regions {bed} selects none of the VCF's {before:,} records ({intervals:,} intervals): "
                             f"check that the BED and the VCF spell contigs the same way (the VCF's style is "
                             f"params.vcf_contig_style: {m.params.get('vcf_contig_style')})")
    m.params["vcf_container_sha256"] = sha256_file(subset.path)
    m.params["vcf_container_bytes"] = subset.path.stat().st_size
    m.params["regions"] = {
        "bed": str(bed),
        "bed_sha256": m.inputs["regions"]["sha256"],
        "bed_bytes": m.inputs["regions"]["bytes"],
        "bed_intervals": intervals,
        "vcf": str(subset.path),
        "vcf_sha256": m.params["vcf_container_sha256"],
        "vcf_bytes": m.params["vcf_container_bytes"],
        "vcf_index": str(subset.index),
        "records_before": before,
        "records_before_source": before_source,
        "records_after": subset.records,
        "regions_overlap": REGIONS_OVERLAP,
        "contig_renames": subset.renames,
        "commands": subset.commands,
    }
    if subset.renames:
        m.note(f"The VCF spells the mitochondrion '{mito}', which Exomiser 14.0.0 (svart) cannot resolve — its MT "
               f"records would be skipped without a warning. The regions subset was rewritten with {subset.renames} "
               f"renamed in the header and records (bcftools annotate --rename-chrs, on the subset only, then moved "
               f"over it) before it was indexed; the subset's sha256 is params.vcf_container_sha256, the original's "
               f"is inputs.vcf.")
    progress(f"regions: {subset.records:,} of {before:,} records kept ({intervals:,} intervals)"
             + (f" · {subset.renames} renamed in the subset" if subset.renames else ""))
    return subset.path


def _records_before(vcf: Path, run_dir: Path, vcf_sha256: str, run: Runner) -> tuple[int, str]:
    """How many records the whole VCF holds, without reading it when an index counts
    them: stage 1's own CSI (built with counts) when its manifest says it indexed this
    very file, else the index beside the VCF (a provider's tabix index may carry none),
    else one pass over the file."""
    csi = run_dir / INGEST_DIR / "input.csi"
    ingest_manifest = run_dir / INGEST_DIR / "manifest.json"
    if csi.exists() and ingest_manifest.exists():
        try:
            indexed = json.loads(ingest_manifest.read_text(encoding="utf-8")).get("inputs", {}).get("vcf", {}).get("sha256")
        except (OSError, ValueError):
            indexed = None
        if indexed == vcf_sha256:
            n = vcf_record_count(vcf, run, index=csi)
            if n is not None:
                return n, f"bcftools index -n, stage 1's index ({csi})"
    n = vcf_record_count(vcf, run)
    if n is not None:
        return n, "bcftools index -n, the index beside the VCF"
    return count_records(vcf, run), "bcftools query, one pass over the VCF (its index carries no record counts)"


def _describe_regions_as_they_were(m: Manifest, facts: dict[str, Any], subset: Path, regions_now: Path | None) -> None:
    """``--join-only``: ``params.regions`` is the run's (copied from ``run.json`` with the
    other run facts); the BED it names becomes ``inputs.regions``; the subset on disk, if
    still there, must hash to what the run ranked; a different ``--regions`` given now
    is noted, never applied."""
    r = facts.get("regions")
    if r:
        m.inputs["regions"] = {"path": r.get("bed"), "bytes": r.get("bed_bytes"), "sha256": r.get("bed_sha256"),
                               "as_of": "the run being joined (run.json)"}
        if subset.exists() and sha256_file(subset) != r.get("vcf_sha256"):
            raise ExomiserFailed(f"--join-only refused: {subset} (sha256 {sha256_file(subset)}) is not the regions "
                                 f"subset the raw run ranked (sha256 {r.get('vcf_sha256')} in run.json); it was "
                                 f"replaced after the run")
        if not subset.exists():
            m.note(f"join-only: the regions subset the container ranked ({r.get('vcf')}) is no longer on disk; its "
                   f"sha256 is params.regions.vcf_sha256 and it is reproducible from inputs.vcf and inputs.regions.")
        if regions_now is not None and sha256_file(regions_now) != r.get("bed_sha256"):
            m.note(f"join-only: --regions names {regions_now} (sha256 {sha256_file(regions_now)}) but the raw run was "
                   f"restricted to {r.get('bed')} (sha256 {r.get('bed_sha256')}); params.regions and inputs.regions "
                   f"describe the run as it was.")
    elif regions_now is not None:
        m.note(f"join-only: --regions {regions_now} was given but the raw run ranked the whole VCF; the ranking "
               f"is genome-wide and the BED was not applied.")


def _note_regions(m: Manifest, subset: Path) -> None:
    """What every manifest of a regions-restricted run says, whichever way it was made:
    the counts, the outputs, and that gene-blind is not genome-wide here."""
    r = m.params.get("regions")
    if not r:
        return
    m.counts["vcf_records"] = r.get("records_before")
    m.counts["regions_records"] = r.get("records_after")
    if subset.exists():
        m.add_output("regions_vcf", subset)
        index = subset.with_name(subset.name + ".tbi")
        if index.exists():
            m.add_output("regions_vcf_index", index)
    m.note(f"Restricted to regions: Exomiser ranked {r['records_after']:,} of {r['records_before']:,} records — those "
           f"overlapping the {r['bed_intervals']:,} intervals of {r['bed']} (sha256 {r['bed_sha256']}, inputs.regions) — "
           f"written to {r.get('vcf')} by bcftools view -R and kept with its .tbi as the file Exomiser saw. The "
           f"ranking is still gene-blind (Exomiser saw the HPO terms and no gene list or shortlist) but it is not "
           f"genome-wide: a gene with no variant inside the BED's regions cannot rank, and #RANK is relative to "
           f"the subset.")


def hpo_from_analysis(text: str) -> list[str]:
    """The ``hpoIds`` of an analysis YAML as written — what Exomiser actually saw."""
    doc = yaml.safe_load(text) or {}
    return [str(h) for h in (doc.get("analysis") or {}).get("hpoIds") or []]


def analysis_from_yaml(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The template part of an analysis YAML as written (every key but the case ones)
    and its ``outputOptions`` — the thresholds that produced a ranking, read back from
    the file Exomiser read rather than from a config that may have moved on."""
    doc = yaml.safe_load(text) or {}
    analysis = {k: v for k, v in (doc.get("analysis") or {}).items() if k not in CASE_KEYS}
    return analysis, dict(doc.get("outputOptions") or {})


def _ensure_image(image: str, run: Runner, progress: Progress) -> dict[str, Any]:
    try:
        return image_digest(image, run)
    except RuntimeError:
        progress(f"pulling {image}")
        out = run(["docker", "pull", image], capture_output=True, text=True, check=False)
        if out.returncode != 0:
            raise RuntimeError(f"docker pull {image} failed: {(out.stderr or out.stdout).strip()[:300]}")
        return image_digest(image, run)


def _failure_message(code: int, log_path: Path, log_text: str, formats: tuple[str, ...]) -> str:
    msg = f"Exomiser exited with code {code}; see {log_path}"
    if "ArrayIndexOutOfBoundsException" in log_text and "TSV_VARIANT" in formats:
        msg += (" — this is the 14.0.0 TSV_VARIANT writer bug (issue #565, fixed in 14.0.1): "
                "rerun with --formats JSON,TSV_GENE")
    elif "OutOfMemoryError" in log_text or code == 137:
        msg += (" — the JVM ran out of memory: stop other containers or raise Docker Desktop's memory, lower or "
                "raise --heap to what the daemon can back (see params.docker_memory_in_use_bytes), or pass "
                "--regions <bed> to rank only the records inside the funnel regions")
    elif "No such file" in log_text and "exomiser-data" in log_text:
        msg += " — the data release under --exomiser-data is incomplete"
    return msg


def write_ranking(path: Path, ranking: list[GeneRanking]) -> None:
    lines = ["\t".join(RANKING_COLUMNS)]
    for g in ranking:
        lines.append("\t".join((
            str(g.rank), g.gene_symbol, f"{g.combined_score:.4f}", f"{g.phenotype_score:.4f}", f"{g.variant_score:.4f}",
            g.moi, str(len(g.variants)), ";".join(g.variants),
        )))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def join_candidates(
    candidates: dict[str, Any] | None,
    ranking: list[GeneRanking],
    gene_ids: dict[str, dict[str, str]],
    top_n: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every stage-3 candidate with Exomiser's view of its gene attached, and the top-N
    ranked genes the shortlist does not contain."""
    by_symbol = {g.gene_symbol: g for g in ranking}
    by_ensembl = {gene_ids[s]["ensembl_id"]: g for s, g in by_symbol.items() if gene_ids.get(s, {}).get("ensembl_id")}
    joined_cands: list[dict[str, Any]] = []
    shortlist_genes: set[str] = set()
    n_ranked = n_variant_match = 0
    for c in (candidates or {}).get("candidates", []):
        g = by_symbol.get(c.get("gene_symbol") or "")
        match = "symbol" if g else None
        if g is None and c.get("gene_id"):
            g = by_ensembl.get(c["gene_id"])
            match = "ensembl_id" if g else None
        entry = dict(c)
        keys = [v.get("key") for v in c.get("variants", []) if v.get("key")]
        if g is None:
            entry.update(exomiser_rank=None, exomiser_score=None, phenotype_score=None, variant_score=None,
                         exomiser_moi=None, exomiser_gene_symbol=None, exomiser_variants=[], exomiser_variants_matched=[],
                         exomiser_by_moi={}, exomiser_match=None, exomiser_evidence_id=None)
        else:
            n_ranked += 1
            shortlist_genes.add(g.gene_symbol)
            matched = sorted(k for k in keys if k in g.all_variants())
            n_variant_match += bool(matched)
            entry.update(exomiser_rank=g.rank, exomiser_score=g.combined_score, phenotype_score=g.phenotype_score,
                         variant_score=g.variant_score, exomiser_moi=g.moi, exomiser_gene_symbol=g.gene_symbol,
                         exomiser_variants=list(g.variants), exomiser_variants_matched=matched, exomiser_by_moi=g.by_moi,
                         exomiser_match=match, exomiser_evidence_id=evidence_id(g.gene_symbol))
        shortlist_genes.add(c.get("gene_symbol") or "")
        joined_cands.append(entry)
    only = [g for g in ranking if g.gene_symbol not in shortlist_genes][:top_n]
    exomiser_only = [{**g.as_dict(), "evidence_id": evidence_id(g.gene_symbol)} for g in only]
    counts = {
        "candidates": len(joined_cands),
        "candidates_ranked": n_ranked,
        "candidates_unranked": len(joined_cands) - n_ranked,
        "candidates_with_variant_match": n_variant_match,
        "exomiser_only": len(exomiser_only),
        "exomiser_genes_not_in_shortlist": sum(1 for g in ranking if g.gene_symbol not in shortlist_genes),
    }
    return {"candidates": joined_cands, "exomiser_only": exomiser_only}, counts


def evidence_id(gene_symbol: str) -> str:
    return f"exomiser:{gene_symbol}"


def image_digest_used(observed: dict[str, Any], pinned: str) -> str:
    """The digest that names the image which actually ran: the pin when the daemon's
    image carries it, otherwise the first repo digest it does carry, otherwise the id."""
    if carries_digest(observed, pinned):
        return pinned
    for d in observed.get("repo_digests") or []:
        if "@" in d:
            return d.rsplit("@", 1)[1]
    return str(observed.get("id") or "unknown")


def gene_url(entrez_id: str, ensembl_id: str, symbol: str) -> str:
    """A page a judge can open for the gene: NCBI by Entrez id, else Ensembl by gene id,
    else the HGNC search for the symbol (Exomiser has no web view of a run)."""
    if entrez_id:
        return f"https://www.ncbi.nlm.nih.gov/gene/{entrez_id}"
    if ensembl_id:
        return f"https://www.ensembl.org/Homo_sapiens/Gene/Summary?g={ensembl_id}"
    return f"https://www.genenames.org/tools/search/#!/?query={symbol}"


def raw_rows_for(symbol: str, genes: list[GeneRow], variants: list[VariantRow]) -> dict[str, Any]:
    """The gene's slice of the raw TSVs: its ``genes.tsv`` row per MOI and its
    contributing ``variants.tsv`` rows, minus the sampled ``P-VALUE`` column."""
    def strip(row: dict[str, str]) -> dict[str, str]:
        return {k: v for k, v in row.items() if k not in OMITTED_COLUMNS}
    return {
        "genes_tsv": {g.moi: strip(g.raw) for g in genes if g.gene_symbol == symbol},
        "variants_tsv": [strip(v.raw) for v in variants if v.gene_symbol == symbol and v.contributing],
    }


def write_evidence(
    root: Path,
    joined: dict[str, Any],
    ranking: list[GeneRanking],
    genes: list[GeneRow],
    variants: list[VariantRow],
    gene_ids: dict[str, dict[str, str]],
    cfg: ExomiserConfig,
    version: str | None,
    analysis_sha: str,
    hpo: list[str],
    image_used: str,
    data_version: str,
    retrieved_at: str,
) -> int:
    """One record per gene in ``joined.json``, so the rank is citable; records of any
    other gene from an earlier join of the same raw directory are removed first.
    ``retrieved_at`` is when the container finished (``run.json``): the record is a
    function of the raw outputs alone, so every join of them gives the same bytes."""
    store = EvidenceStore(root)
    by_symbol = {g.gene_symbol: g for g in ranking}
    symbols = [c["exomiser_gene_symbol"] for c in joined["candidates"] if c.get("exomiser_gene_symbol")]
    symbols += [g["gene_symbol"] for g in joined["exomiser_only"]]
    wanted = [s for s in dict.fromkeys(symbols) if s in by_symbol]
    keep = {store.path_for(evidence_id(s)) for s in wanted}
    for stale in sorted((root / "exomiser").glob("*.json")) if (root / "exomiser").is_dir() else []:
        if stale not in keep:
            stale.unlink()
    source_version = f"exomiser-cli {version or cfg.image_version} · data {data_version} · image {image_used}"
    query = {"hpoIds": list(hpo), "genomeAssembly": cfg.analysis.get("genomeAssembly"), "analysis_yaml_sha256": analysis_sha}
    n = 0
    for symbol in wanted:
        g = by_symbol[symbol]
        payload = {"ranking": g.as_dict(), "raw": raw_rows_for(symbol, genes, variants),
                   "omitted_columns": OMITTED_COLUMNS, "citation": cfg.data_citation}
        rec = EvidenceRecord(
            record_id=evidence_id(symbol),
            source="exomiser",
            source_version=source_version,
            query=query,
            url=gene_url(g.entrez_id, gene_ids.get(symbol, {}).get("ensembl_id", ""), symbol),
            retrieved_at=retrieved_at,
            payload=payload,
        )
        store.put(rec)
        n += 1
    store.write_index()
    return n


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def carries_digest(observed: dict[str, Any], digest: str) -> bool:
    """Whether ``docker image inspect`` output names the pinned digest — as the image id
    (a manifest-list digest is the id when pulled by digest) or as a repo digest."""
    if observed.get("id") == digest:
        return True
    return any(d.endswith("@" + digest) for d in observed.get("repo_digests", []) or [])


def read_joined(run_dir: Path) -> dict[str, Any]:
    """``joined.json`` of a run — for later stages and tests."""
    return json.loads((Path(run_dir) / STAGE_DIR / "joined.json").read_text(encoding="utf-8"))
