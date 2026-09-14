"""Stage 2 orchestrator — ``engine retrieve``.

Reads the stage-1 table, optionally narrows it to the funnel regions, and fetches
evidence in tiers so that public APIs see only what they must:

1. **VEP** for every variant in scope (batched, 200 per request). Gives consequence,
   transcript, in-silico scores, and — as a courtesy copy — gnomAD frequencies and
   ClinVar significance for the allele.
2. **ClinVar** for every variant in scope, from a release-pinned local VCF. Free.
3. **gnomAD** (the API, one variant per query) only for variants that are
   consequential *and* not already common by VEP's copy of gnomAD. This is the
   citable frequency record with homozygote counts that ACMG PM2/BA1 needs.

The variant table written here is a projection of the evidence store: every
non-ingest column comes from a record, and ``evidence_ids`` names the records.
"""

from __future__ import annotations

import gzip
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from engine.contigs import canonical
from engine.ingest import STAGE_DIR as INGEST_DIR
from engine.ingest import read_variants
from engine.manifest import Manifest
from engine.retrieve import Retriever
from engine.retrieve.http import Http, HttpCache, RateLimiter
from engine.retrieve.intervals import IntervalIndex
from engine.retrieve.store import EvidenceRecord, EvidenceStore, VariantKey, key_str

STAGE_DIR = "02_retrieve"

RATE_LIMITS = {
    "rest.ensembl.org": 5.0,            # documented 55,000/hour; batches of 200 keep us far below
    "gnomad.broadinstitute.org": 0.15,  # observed ~10 requests/minute before 429; 25 variants per request
    "www.ebi.ac.uk": 3.0,
    "eutils.ncbi.nlm.nih.gov": 2.5,     # 3/s without an API key
}
BACKOFF_FLOOR = {
    "gnomad.broadinstitute.org": 75.0,  # a 429 stays alive while you keep probing; stop for a minute+
    "rest.ensembl.org": 15.0,           # transient 503s can last minutes; with RETRIES this waits ~4.5 min
    "www.ebi.ac.uk": 5.0,
}
RETRIES = 8

IMPACT_RANK = {"HIGH": 3, "MODERATE": 2, "SPLICE": 2, "LOW": 1, "MODIFIER": 0, "": -1}


@dataclass
class RetrieveOptions:
    cache_root: Path
    funnel_bed: Path | None = None
    clinvar_vcf: Path | None = None
    sources: tuple[str, ...] = ("vep", "clinvar", "gnomad")
    offline: bool = False
    gnomad_prefilter_impacts: tuple[str, ...] = ("HIGH", "MODERATE", "SPLICE")
    gnomad_prefilter_max_af: float = 0.05
    """Query the gnomAD API only for consequential variants whose VEP-carried gnomAD
    frequency is below this (or absent). Common variants need no citable frequency
    record: they are excluded by BA1/BS1 on VEP's copy alone."""
    max_variants: int | None = None
    """Cap the number of variants sent to APIs — for smoke runs only; recorded."""
    vep_workers: int = 4
    """Concurrent VEP batches. Ensembl allows 15 req/s; a 200-row batch takes ~1 min,
    so even 8 workers stay far below the limit."""
    progress: Callable[[str], None] = field(default=lambda s: None)


RetrieverFactory = Callable[[Http, RetrieveOptions], dict[str, Retriever]]


def default_retrievers(http: Http, opts: RetrieveOptions) -> dict[str, Retriever]:
    out: dict[str, Retriever] = {}
    if "vep" in opts.sources:
        from engine.retrieve.vep import VepRetriever
        out["vep"] = VepRetriever(http, workers=opts.vep_workers)
    if "clinvar" in opts.sources:
        if opts.clinvar_vcf is None:
            raise ValueError("clinvar source requested but no --clinvar-vcf given (see `engine clinvar-download`)")
        from engine.retrieve.clinvar import ClinvarRetriever
        out["clinvar"] = ClinvarRetriever(opts.clinvar_vcf)
    if "gnomad" in opts.sources:
        from engine.retrieve.gnomad import GnomadRetriever
        out["gnomad"] = GnomadRetriever(http)
    return out


def _chrom_order(c: str) -> tuple[int, str]:
    return (int(c), "") if c.isdigit() else (100, c)


def select_keys(rows: Iterable[dict[str, str]], funnel: IntervalIndex | None) -> tuple[list[VariantKey], dict[VariantKey, str], dict]:
    """Unique variant keys in scope, in genome order, plus the funnel region name per key."""
    seen: dict[VariantKey, str] = {}
    n_rows = 0
    n_in = 0
    for r in rows:
        n_rows += 1
        k: VariantKey = (r["chrom"], int(r["pos"]), r["ref"], r["alt"])
        if funnel is not None:
            name = funnel.lookup(k[0], k[1])
            if name is None and not funnel.overlaps(k[0], k[1], k[2]):
                continue
            region = name or "(overlap)"
        else:
            region = ""
        n_in += 1
        seen.setdefault(k, region)
    keys = sorted(seen, key=lambda k: (_chrom_order(k[0]), k[1], k[2], k[3]))
    return keys, seen, {"rows_in": n_rows, "rows_in_scope": n_in, "unique_variants": len(keys)}


def gnomad_prefilter(keys: list[VariantKey], vep_cols: dict[VariantKey, dict[str, str]], opts: RetrieveOptions) -> list[VariantKey]:
    """Variants worth a citable gnomAD record: consequential, and not already common."""
    out: list[VariantKey] = []
    for k in keys:
        c = vep_cols.get(k, {})
        if c.get("impact_any_coding", "") not in opts.gnomad_prefilter_impacts:
            continue
        afs = []
        for col in ("vep_gnomade_af", "vep_gnomadg_af"):
            v = c.get(col, "")
            if v not in ("", None):
                try:
                    afs.append(float(v))
                except ValueError:
                    pass
        if afs and max(afs) >= opts.gnomad_prefilter_max_af:
            continue
        out.append(k)
    return out


def run_retrieve(run_dir: Path, opts: RetrieveOptions, *, retrievers: RetrieverFactory = default_retrievers) -> Path:
    run_dir = Path(run_dir)
    out_dir = run_dir / STAGE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ingest_manifest = run_dir / INGEST_DIR / "manifest.json"
    if not ingest_manifest.exists():
        raise FileNotFoundError(f"stage 1 has not run in {run_dir} ({ingest_manifest} missing)")

    m = Manifest(stage="retrieve")
    m.add_input("ingest_variants", run_dir / INGEST_DIR / "variants.tsv.gz")
    m.add_input("ingest_manifest", ingest_manifest, checksum=False)
    m.add_tool("bcftools")

    funnel = None
    if opts.funnel_bed is not None:
        funnel = IntervalIndex.from_bed(opts.funnel_bed)
        m.add_input("funnel_bed", opts.funnel_bed)
        m.params["funnel"] = {"intervals": funnel.n_intervals(), "bp": funnel.total_bp()}
    if opts.clinvar_vcf is not None:
        m.add_input("clinvar_vcf", opts.clinvar_vcf, checksum=False)

    keys, region_of, counts = select_keys(read_variants(run_dir), funnel)
    if opts.max_variants is not None and len(keys) > opts.max_variants:
        m.note(f"max_variants={opts.max_variants}: only the first {opts.max_variants} of {len(keys)} variants in scope were retrieved.")
        keys = keys[: opts.max_variants]
    m.counts.update(counts)
    m.params.update({
        "sources": list(opts.sources),
        "offline": opts.offline,
        "gnomad_prefilter_impacts": list(opts.gnomad_prefilter_impacts),
        "gnomad_prefilter_max_af": opts.gnomad_prefilter_max_af,
        "max_variants": opts.max_variants,
        "vep_workers": opts.vep_workers,
        "cache_root": str(opts.cache_root),
        "rate_limits_per_second": RATE_LIMITS,
        "backoff_floor_s": BACKOFF_FLOOR,
        "retries": RETRIES,
    })

    http = Http(HttpCache(opts.cache_root), limiter=RateLimiter(RATE_LIMITS), offline=opts.offline,
                backoff_floor=BACKOFF_FLOOR, retries=RETRIES)
    store = EvidenceStore(out_dir / "evidence")
    rs = retrievers(http, opts)

    cols_by_key: dict[VariantKey, dict[str, str]] = {k: {} for k in keys}
    ids_by_key: dict[VariantKey, list[str]] = {k: [] for k in keys}
    versions: dict[str, str] = {}
    timings: dict[str, float] = {}

    def run_source(name: str, subset: list[VariantKey]) -> None:
        r = rs[name]
        t0 = time.monotonic()
        opts.progress(f"{name}: {len(subset):,} variants")
        # Stream where the retriever can (VEP): each record is stored and projected as
        # it arrives and then dropped, so memory is bounded by the workers, not the genome.
        stream = r.retrieve_stream(subset) if hasattr(r, "retrieve_stream") else r.retrieve(subset).items()
        n_rec = 0
        with_record = 0
        for k, recs in stream:
            if recs:
                with_record += 1
            for rec in recs:
                store.put(rec)
                ids_by_key[k].append(rec.record_id)
                n_rec += 1
            cols_by_key[k].update(r.extract(recs))
        versions[name] = r.version()
        timings[name] = round(time.monotonic() - t0, 1)
        m.counts[f"{name}_variants_queried"] = len(subset)
        m.counts[f"{name}_variants_with_record"] = with_record
        m.counts[f"{name}_records"] = n_rec
        opts.progress(f"{name}: {m.counts[f'{name}_variants_with_record']:,} with a record · {timings[name]}s")

    if "vep" in rs:
        run_source("vep", keys)
    if "clinvar" in rs:
        run_source("clinvar", keys)
    if "gnomad" in rs:
        subset = gnomad_prefilter(keys, cols_by_key, opts) if "vep" in rs else keys
        # gnomAD's nuclear variant() field cannot answer for MT or non-ACGT alleles; the
        # retriever refuses them loudly, so route them around it and say so.
        from engine.retrieve.gnomad import unaskable
        skipped: dict[str, int] = {}
        askable = []
        for k in subset:
            why = unaskable(k)
            if why:
                skipped[why] = skipped.get(why, 0) + 1
            else:
                askable.append(k)
        m.counts["gnomad_prefilter_selected"] = len(subset)
        m.counts["gnomad_skipped_unaskable"] = skipped
        if skipped:
            m.note(f"gnomAD not queried for {sum(skipped.values())} prefiltered variants ({skipped}); "
                   "their frequency columns are empty, not zero.")
        run_source("gnomad", askable)

    # ---- the table: ingest row + projections + evidence ids
    table_path = out_dir / "variants.annotated.tsv.gz"
    extra_cols: list[str] = []
    for name in ("vep", "clinvar", "gnomad"):
        if name in rs:
            extra_cols.extend(rs[name].columns)
    header = None
    n_written = 0
    # mtime=0: the gzip header carries no timestamp, so identical content is an identical file.
    with gzip.GzipFile(table_path, "wb", mtime=0) as raw, io.TextIOWrapper(raw, encoding="utf-8") as out:
        for r in read_variants(run_dir):
            k: VariantKey = (r["chrom"], int(r["pos"]), r["ref"], r["alt"])
            if k not in cols_by_key:
                continue
            if header is None:
                header = list(r.keys()) + ["funnel_region"] + extra_cols + ["evidence_ids"]
                out.write("\t".join(header) + "\n")
            c = cols_by_key[k]
            vals = list(r.values()) + [region_of.get(k, "")] + [c.get(col, "") for col in extra_cols] + [";".join(ids_by_key[k])]
            out.write("\t".join(v if v is not None else "" for v in vals) + "\n")
            n_written += 1
    index_path = store.write_index()

    m.counts["table_rows"] = n_written
    m.counts["evidence_records"] = store.count()
    m.counts["http"] = {"live_requests": http.live_requests, **http.cache.stats()}
    m.params["source_versions"] = versions
    m.params["timings_s"] = timings
    m.add_output("variants_annotated", table_path)
    m.add_output("evidence_index", index_path)
    manifest_path = out_dir / "manifest.json"
    m.write(manifest_path)
    return manifest_path


def read_annotated(run_dir: Path) -> Iterable[dict[str, str]]:
    path = Path(run_dir) / STAGE_DIR / "variants.annotated.tsv.gz"
    with gzip.open(path, "rt") as f:
        cols = f.readline().rstrip("\n").split("\t")
        for line in f:
            yield dict(zip(cols, line.rstrip("\n").split("\t")))
