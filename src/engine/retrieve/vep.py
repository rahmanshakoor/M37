"""Stage 2 retriever — Ensembl VEP over REST.

VEP is the first source every variant in scope goes through: it names the gene and
transcript, the consequence and its impact, HGVS, exon/intron numbering, SIFT and
PolyPhen, and — with the plugins the public endpoint honours — SpliceAI and CADD.
It also carries a *courtesy copy* of gnomAD frequencies and ClinVar significance on
the colocated dbSNP record; those are used only to pre-filter what the citable
gnomAD and ClinVar retrievers must fetch, never as the citation itself.

Why the batch endpoint and the VCF-ID trick: ``POST /vep/<species>/region`` accepts
up to 200 VCF-style lines per request and echoes column 3 of each line back as
``[i].id``. Putting the engine's own variant key there makes the result-to-key match
exact, independent of output order and of the coordinate shifting VEP applies to
indels (``7 117559590 ATCT>A`` comes back as ``TCT/-`` at 117559591). ``[i].input``
(the verbatim line) is the fallback.

Why a missing result is usually an error, not absence: VEP annotates every mappable
GRCh38 position (an empty stretch is ``intergenic_variant``), so a well-formed line
that comes back without an item was dropped server-side — and a line it cannot parse
(a symbolic ALT, an unknown contig) vanishes from a 200 without an error field. The
first is raised; the second is recorded as absence and counted, never silently.

Why the response is *not* trusted blindly (see the probe): ``transcript_consequences``,
``colocated_variants`` and ``frequencies`` are absent keys rather than empty ones when
there is nothing to say; the colocated list mixes dbSNP with HGMD/COSMIC entries that
carry no frequencies; ``clin_sig`` is merged across every allele of the dbSNP record
and only ``clin_sig_allele`` is allele-specific; an overlapping antisense lncRNA can
outrank the coding gene's own consequence; and a wrong REF is annotated without error
but silently loses the dbSNP record, CADD and SpliceAI. Stage 1 checks REF against the
assembly only when a reference FASTA is configured (``bcftools norm -c w`` warns, it
does not drop), so every allele-keyed lookup here uses the VEP-normalised
``variant_allele`` of the chosen transcript consequence, and a coding variant with
neither CADD nor SpliceAI is the wrong-REF signature to look for.
"""

from __future__ import annotations

import logging
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator, Iterable

from engine.contigs import PRIMARY
from engine.retrieve.http import Http, Response
from engine.retrieve.store import EvidenceRecord, VariantKey, key_str

log = logging.getLogger(__name__)

# Ensembl's consequence severity order, most severe first. Hardcoded from
# GET /info/variation/consequence_types?rank=1 (release 116, 2026-09-12) so that
# ranking never depends on a network call.
SEVERITY_ORDER: tuple[str, ...] = (
    "transcript_ablation",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "stop_gained",
    "frameshift_variant",
    "stop_lost",
    "start_lost",
    "transcript_amplification",
    "feature_elongation",
    "feature_truncation",
    "inframe_insertion",
    "inframe_deletion",
    "missense_variant",
    "protein_altering_variant",
    "splice_donor_5th_base_variant",
    "splice_region_variant",
    "splice_donor_region_variant",
    "splice_polypyrimidine_tract_variant",
    "incomplete_terminal_codon_variant",
    "start_retained_variant",
    "stop_retained_variant",
    "synonymous_variant",
    "coding_sequence_variant",
    "mature_miRNA_variant",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "non_coding_transcript_exon_variant",
    "intron_variant",
    "NMD_transcript_variant",
    "non_coding_transcript_variant",
    "coding_transcript_variant",
    "upstream_gene_variant",
    "downstream_gene_variant",
    "TFBS_ablation",
    "TFBS_amplification",
    "TF_binding_site_variant",
    "regulatory_region_ablation",
    "regulatory_region_amplification",
    "regulatory_region_variant",
    "intergenic_variant",
    "sequence_variant",
)
_SEVERITY_RANK = {term: i for i, term in enumerate(SEVERITY_ORDER)}

IMPACT_ORDER: tuple[str, ...] = ("HIGH", "MODERATE", "LOW", "MODIFIER")
_IMPACT_RANK = {imp: i for i, imp in enumerate(reversed(IMPACT_ORDER))}  # HIGH=3 … MODIFIER=0

BASE_PARAMS: dict[str, int] = {"canonical": 1, "mane": 1, "hgvs": 1, "numbers": 1}
"""Flags verified live: they annotate (mane_select/canonical keys, hgvsc/hgvsp,
exon/intron) but do not filter — every overlapping transcript is still returned."""

DEFAULT_PLUGINS: tuple[str, ...] = ("SpliceAI", "CADD", "REVEL", "AlphaMissense")
"""Plugins the public endpoint honours (verified live on public variants): SpliceAI
delta scores, CADD PHRED, REVEL (``revel`` per transcript) and AlphaMissense
(``alphamissense.am_pathogenicity`` / ``am_class`` per transcript). ``NMD`` is
accepted but returns nothing on the REST endpoint, so NMD is judged from the exon
number downstream. REVEL and AlphaMissense are the predictors the ClinGen SVI
computational recommendation (Pejaver et al. 2022) calibrates PP3/BP4 against."""

SPLICEAI_TAGS: tuple[str, ...] = ("AG", "AL", "DG", "DL")

_BASES = frozenset("ACGT")


class VepError(RuntimeError):
    """The API answered, but not with something this module can trust."""


def severity_rank(term: str) -> int:
    """Position in :data:`SEVERITY_ORDER`; unknown terms sort after every known one."""
    return _SEVERITY_RANK.get(term, len(SEVERITY_ORDER))


def most_severe(consequence_terms: Iterable[str]) -> str:
    """The most severe SO term of ``consequence_terms`` per Ensembl's order; ``''`` if none.

    >>> most_severe(["intron_variant", "splice_region_variant", "missense_variant"])
    'missense_variant'
    """
    terms = list(consequence_terms)
    if not terms:
        return ""
    return min(terms, key=lambda t: (severity_rank(t), t))


def vcf_line(k: VariantKey) -> str:
    """One VCF-style input line with the engine key in the ID column."""
    return f"{k[0]} {k[1]} {key_str(k)} {k[2]} {k[3]} . . ."


def well_formed(k: VariantKey) -> bool:
    """A canonical-contig, ACGT-only variant — one VEP annotates without fail, so no
    result for it means the line was dropped server-side, not that it is absent."""
    return k[0] in PRIMARY and set(k[2].upper()) <= _BASES and set(k[3].upper()) <= _BASES


class VepRetriever:
    source = "vep"
    columns: tuple[str, ...] = (
        "gene_symbol", "gene_id", "transcript_id", "mane", "consequence", "impact",
        "hgvsc", "hgvsp", "exon", "intron", "biotype", "protein_position", "amino_acids",
        "sift_pred", "sift_score", "polyphen_pred", "polyphen_score",
        "spliceai_ds_max", "spliceai_detail", "cadd_phred", "revel", "alphamissense_score", "alphamissense_class",
        "most_severe_consequence", "impact_any_coding",
        "rsid", "vep_gnomade_af", "vep_gnomadg_af", "vep_clin_sig", "vep_clinvar_ids",
    )

    def __init__(
        self,
        http: Http,
        *,
        batch_size: int = 200,
        species: str = "homo_sapiens",
        base_url: str = "https://rest.ensembl.org",
        plugins: tuple[str, ...] = DEFAULT_PLUGINS,
        workers: int = 4,
        timeout: float = 300.0,
    ):
        if not 1 <= batch_size <= 200:
            raise ValueError("VEP REST accepts at most 200 variants per POST")
        self.http = http
        self.batch_size = batch_size
        self.species = species
        self.base_url = base_url.rstrip("/")
        self.plugins = tuple(plugins)
        self.workers = max(1, workers)
        self.timeout = timeout  # a 200-row batch has been observed to take minutes
        self.params: dict[str, int] = {**BASE_PARAMS, **{p: 1 for p in self.plugins}}
        self.endpoint = f"{self.base_url}/vep/{self.species}/region"
        self._version: str | None = None
        self.dropped = 0
        """Submitted lines VEP returned nothing for and that were recorded as absent
        (non-ACGT alleles only; see :func:`well_formed`). The manifest shows the same
        number as ``vep_variants_queried - vep_variants_with_record``."""

    # ------------------------------------------------------------------ version

    def version(self) -> str:
        """``Ensembl VEP 116 (rest.ensembl.org, assembly GRCh38)`` — observed, cached
        on the instance.

        The three ``/info`` GETs replay from the HTTP cache like everything else, so
        an offline rerun stamps records with the release that produced them. Online,
        the same GETs are also made live and must agree: a cache spanning an Ensembl
        release boundary would otherwise label freshly annotated payloads with the
        old release, and nothing in a VEP payload would reveal it.
        """
        if self._version is None:
            cached = self._observe(cache_ok=True)
            if not getattr(self.http, "offline", False):
                live = self._observe(cache_ok=False)
                if live != cached:
                    raise VepError(
                        f"the HTTP cache was filled under {cached!r} but the server now reports {live!r}: "
                        "clear the cache (or run offline) rather than mixing releases")
            self._version = cached
        return self._version

    def _observe(self, *, cache_ok: bool) -> str:
        def info(path: str) -> dict[str, Any]:
            body = self.http.get(f"{self.base_url}{path}", cache_ok=cache_ok).json()
            if not isinstance(body, dict):
                raise VepError(f"GET {path} returned {type(body).__name__}, not an object")
            return body

        software = info("/info/software")
        data = info("/info/data")
        assembly = info(f"/info/assembly/{self.species}")
        release = software.get("release")
        loaded = [str(r) for r in data.get("releases", [])]
        host = urllib.parse.urlparse(self.base_url).netloc
        asm = assembly.get("default_coord_system_version") or assembly.get("assembly_name", "?")
        extra = "" if loaded == [str(release)] else f", data release {','.join(loaded) or '?'}"
        return f"Ensembl VEP {release}{extra} ({host}, assembly {asm})"

    # ----------------------------------------------------------------- retrieve

    def retrieve(self, keys: list[VariantKey]) -> dict[VariantKey, list[EvidenceRecord]]:
        """Every key's records at once. For a genome-scale key list prefer
        :meth:`retrieve_stream`: this holds every payload in memory."""
        uniq = list(dict.fromkeys(keys))
        out: dict[VariantKey, list[EvidenceRecord]] = {k: [] for k in uniq}
        for k, recs in self.retrieve_stream(uniq):
            out[k] = recs
        return out

    def retrieve_stream(self, keys: list[VariantKey]) -> Iterator[tuple[VariantKey, list[EvidenceRecord]]]:
        """Yield ``(key, records)`` batch by batch in submission order, so the caller
        can store and project each record and let it go — 176k VEP payloads kept
        at once is what exhausted an 8 GB machine."""
        uniq = list(dict.fromkeys(keys))
        batches = [uniq[i:i + self.batch_size] for i in range(0, len(uniq), self.batch_size)]
        if not batches:
            return
        version = self.version()
        log.info("vep: %d variants in %d batch(es) of up to %d", len(uniq), len(batches), self.batch_size)
        # executor.map yields in submission order, so output is independent of which
        # batch finishes first; Http is thread-safe (cache and limiter are locked).
        # The pool is created lazily per chunk of batches so at most ``workers`` responses
        # are in flight or waiting to be consumed.
        chunk = max(1, self.workers)
        with ThreadPoolExecutor(max_workers=chunk) as pool:
            for start in range(0, len(batches), chunk):
                group = batches[start:start + chunk]
                for j, (batch, resp) in enumerate(zip(group, pool.map(self._post, group)), start=start + 1):
                    found = self._match(batch, resp)
                    self._check_missing(batch, found, resp, f"batch {j}/{len(batches)}")
                    for k in batch:
                        yield k, ([self._record(k, found[k], resp, version)] if k in found else [])
                    log.info("vep: batch %d/%d: %d of %d variants returned a result%s",
                             j, len(batches), len(found), len(batch), " (cached)" if resp.from_cache else "")

    def _post(self, batch: list[VariantKey]) -> Response:
        body = {"variants": [vcf_line(k) for k in batch]}
        return self.http.post(self.endpoint, body, params=self.params, timeout=self.timeout)

    @staticmethod
    def _match(batch: list[VariantKey], resp: Response) -> dict[VariantKey, dict[str, Any]]:
        data = resp.json()
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            # deliberately without the body: a VEP error message can echo the input line
            raise VepError(f"VEP returned {type(data).__name__}, not a list of results, "
                           f"for a batch of {len(batch)} variants")
        by_id: dict[str, dict[str, Any]] = {}
        by_input: dict[str, dict[str, Any]] = {}
        for item in data:
            by_id.setdefault(str(item.get("id", "")), item)
            by_input.setdefault(str(item.get("input", "")), item)
        found: dict[VariantKey, dict[str, Any]] = {}
        for k in batch:
            item = by_id.get(key_str(k)) or by_input.get(vcf_line(k))
            if item is not None:
                found[k] = item
        return found

    def _check_missing(self, batch: list[VariantKey], found: dict, resp: Response, label: str) -> None:
        missing = [k for k in batch if k not in found]
        if not missing:
            return
        bad = sum(1 for k in missing if well_formed(k))
        if bad:
            raise VepError(
                f"{label}: {bad} of {len(batch)} well-formed variants came back without a result "
                f"(request {resp.request_key}); VEP annotates every mappable position, so this "
                "response is incomplete — remove it from the cache and retry")
        self.dropped += len(missing)
        log.warning("vep: %s: %d of %d variants (non-ACGT alleles) returned no result; recorded as absent",
                    label, len(missing), len(batch))

    def _record(self, k: VariantKey, item: dict[str, Any], resp: Response, version: str) -> EvidenceRecord:
        # request_key names the cache entry of the batch POST this line was part of
        return EvidenceRecord(
            record_id=f"vep:{key_str(k)}",
            source=self.source,
            source_version=version,
            query={"method": "POST", "url": self.endpoint, "params": self.params, "variant": vcf_line(k),
                   "request_key": resp.request_key},
            url=self.human_url(k),
            retrieved_at=resp.retrieved_at,
            payload=item,
        )

    def human_url(self, k: VariantKey) -> str:
        """Single-variant GET form of the same annotation — opens in a browser."""
        chrom, pos, ref, alt = k
        end = pos + len(ref) - 1
        query = urllib.parse.urlencode({"content-type": "application/json", **self.params})
        return f"{self.endpoint}/{chrom}:{pos}-{end}/{alt}?{query}"

    # ------------------------------------------------------------------ extract

    def extract(self, records: list[EvidenceRecord]) -> dict[str, str]:
        cols = {c: "" for c in self.columns}
        if not records:
            return cols
        payload = records[0].payload
        tcs: list[dict[str, Any]] = list(payload.get("transcript_consequences") or [])
        tc = select_transcript(tcs)
        block = tc if tc is not None else _fallback_block(payload)
        allele = (block or {}).get("variant_allele") or _alt_from_allele_string(payload)

        if tc is not None:
            cols.update(_transcript_columns(tc))
        elif block is not None:
            # intergenic / regulatory-only input: consequence but no transcript
            terms = sorted(block.get("consequence_terms") or [], key=severity_rank)
            cols["consequence"] = ",".join(terms)
            cols["impact"] = _s(block.get("impact"))

        # CADD and SpliceAI are variant-level: scan every block, not just the chosen one.
        cadd = next((b.get("cadd_phred") for b in _all_blocks(payload) if b.get("cadd_phred") is not None), None)
        cols["cadd_phred"] = _s(cadd)
        cols.update(_spliceai(tcs, tc))
        cols.update(_missense_predictors(tcs, tc))
        cols["most_severe_consequence"] = _s(payload.get("most_severe_consequence"))
        cols["impact_any_coding"] = impact_any_coding(tcs)
        cols.update(_colocated(payload, allele))
        return cols


# ---------------------------------------------------------------- selection

def _tc_rank(tc: dict[str, Any]) -> int:
    return severity_rank(most_severe(tc.get("consequence_terms") or []))


def _tc_sort_key(tc: dict[str, Any]) -> tuple:
    return (
        _tc_rank(tc),
        0 if tc.get("mane_select") else 1,
        0 if tc.get("canonical") else 1,
        0 if tc.get("biotype") == "protein_coding" else 1,
        str(tc.get("transcript_id", "")),
    )


def select_transcript(tcs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The transcript consequence to report.

    Only genes with a protein-coding transcript are candidates while any overlap the
    variant: Ensembl ranks ``non_coding_transcript_exon_variant`` above
    ``intron_variant``, so an antisense lncRNA exon would otherwise win every deep
    intronic variant of the coding gene it overlaps. Among the candidates, the gene(s)
    carrying the most severe consequence (on any of their transcripts, NMD included)
    are taken, and within them the MANE Select protein-coding transcript, else the
    canonical protein-coding one, else the most severe protein-coding one. With no
    coding gene at all, the most severe consequence of any transcript. Ties break on
    MANE, canonical, protein-coding, then transcript id, so the choice is deterministic.
    ``most_severe_consequence`` and ``impact_any_coding`` still expose the whole picture.
    """
    if not tcs:
        return None
    coding_genes = {tc.get("gene_id", "") for tc in tcs if tc.get("biotype") == "protein_coding"}
    pool = [tc for tc in tcs if tc.get("gene_id", "") in coding_genes] if coding_genes else tcs
    top = min(_tc_rank(tc) for tc in pool)
    genes = {tc.get("gene_id", "") for tc in pool if _tc_rank(tc) == top}
    coding = [tc for tc in pool if tc.get("gene_id", "") in genes and tc.get("biotype") == "protein_coding"]
    for flag in ("mane_select", "canonical"):
        hits = [tc for tc in coding if tc.get(flag)]
        if hits:
            return min(hits, key=_tc_sort_key)
    return min(coding or pool, key=_tc_sort_key)


def impact_any_coding(tcs: list[dict[str, Any]]) -> str:
    """Highest impact over every protein-coding transcript consequence; ``SPLICE`` when
    a splice-related term would otherwise leave it below MODERATE. ``''`` when the
    variant touches no protein-coding transcript. Read by the stage-3 pre-filter."""
    best = -1
    splice = False
    for tc in tcs:
        if tc.get("biotype") != "protein_coding":
            continue
        best = max(best, _IMPACT_RANK.get(str(tc.get("impact", "")), -1))
        splice = splice or any("splice" in t for t in tc.get("consequence_terms") or [])
    if splice and best < _IMPACT_RANK["MODERATE"]:
        return "SPLICE"
    return IMPACT_ORDER[len(IMPACT_ORDER) - 1 - best] if best >= 0 else ""


def _fallback_block(payload: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("intergenic_consequences", "regulatory_feature_consequences", "motif_feature_consequences"):
        blocks = payload.get(key) or []
        if blocks:
            return blocks[0]
    return None


def _all_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in ("transcript_consequences", "intergenic_consequences",
                "regulatory_feature_consequences", "motif_feature_consequences"):
        out.extend(payload.get(key) or [])
    return out


def _alt_from_allele_string(payload: dict[str, Any]) -> str:
    parts = str(payload.get("allele_string", "")).split("/")
    return parts[1] if len(parts) == 2 else ""


# ---------------------------------------------------------------- columns

def _s(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ",".join(_s(x) for x in v)
    return str(v)


def _span(start: Any, end: Any) -> str:
    if start is None and end is None:
        return ""
    if start is None or end is None or start == end:
        return _s(start if start is not None else end)
    return f"{start}-{end}"


def _mane(tc: dict[str, Any]) -> str:
    acc = tc.get("mane_select") or tc.get("mane_plus_clinical") or ""
    tags = list(tc.get("mane") or [])
    if acc and tags and tags != ["MANE_Select"]:
        return f"{acc} ({','.join(tags)})"
    return _s(acc)


def _transcript_columns(tc: dict[str, Any]) -> dict[str, str]:
    terms = sorted(tc.get("consequence_terms") or [], key=severity_rank)
    return {
        "gene_symbol": _s(tc.get("gene_symbol")),
        "gene_id": _s(tc.get("gene_id")),
        "transcript_id": _s(tc.get("transcript_id")),
        "mane": _mane(tc),
        "consequence": ",".join(terms),
        "impact": _s(tc.get("impact")),
        "hgvsc": _s(tc.get("hgvsc")),
        "hgvsp": _s(tc.get("hgvsp")),
        "exon": _s(tc.get("exon")),
        "intron": _s(tc.get("intron")),
        "biotype": _s(tc.get("biotype")),
        "protein_position": _span(tc.get("protein_start"), tc.get("protein_end")),
        "amino_acids": _s(tc.get("amino_acids")),
        "sift_pred": _s(tc.get("sift_prediction")),
        "sift_score": _s(tc.get("sift_score")),
        "polyphen_pred": _s(tc.get("polyphen_prediction")),
        "polyphen_score": _s(tc.get("polyphen_score")),
    }


def _missense_predictors(tcs: list[dict[str, Any]], chosen: dict[str, Any] | None) -> dict[str, str]:
    """REVEL and AlphaMissense from the chosen transcript, else the highest score any
    transcript carries (the predictors are per protein change, so they differ only
    when transcripts disagree on the amino acid)."""
    order = ([chosen] if chosen is not None else []) + [t for t in tcs if t is not chosen]
    revel = next((t.get("revel") for t in order if t.get("revel") is not None), None)
    if revel is None:
        revel = max((t["revel"] for t in tcs if isinstance(t.get("revel"), (int, float))), default=None)
    am = next((t.get("alphamissense") for t in order if isinstance(t.get("alphamissense"), dict)
               and t["alphamissense"].get("am_pathogenicity") is not None), None)
    if am is None:
        ams = [t["alphamissense"] for t in tcs if isinstance(t.get("alphamissense"), dict)
               and isinstance(t["alphamissense"].get("am_pathogenicity"), (int, float))]
        am = max(ams, key=lambda a: a["am_pathogenicity"], default=None)
    return {"revel": _s(revel), "alphamissense_score": _s((am or {}).get("am_pathogenicity")),
            "alphamissense_class": _s((am or {}).get("am_class"))}


def _spliceai(tcs: list[dict[str, Any]], chosen: dict[str, Any] | None) -> dict[str, str]:
    """Highest SpliceAI delta score over every transcript consequence, with the four
    scores of the block that carries it. The plugin attaches one gene-level object to
    each of a gene's transcripts (identical across them, verified live), so the chosen
    transcript's copy normally wins and a block without one (a lncRNA) is skipped; an
    overlapping gene's copy can score higher and is then labelled with that gene so
    the row stays self-explanatory."""
    empty = {"spliceai_ds_max": "", "spliceai_detail": ""}
    ordered = ([chosen] if chosen is not None else []) + [tc for tc in sorted(tcs, key=_tc_sort_key) if tc is not chosen]
    best: tuple[float, list[tuple[str, Any]], dict[str, Any]] | None = None
    for tc in ordered:
        sa = tc.get("spliceai") or {}
        scores = [(tag, sa.get(f"DS_{tag}")) for tag in SPLICEAI_TAGS]
        present = [float(v) for _, v in scores if v is not None]
        if present and (best is None or max(present) > best[0]):  # strict: the chosen block keeps ties
            best = (max(present), scores, tc)
    if best is None:
        return empty
    _, scores, tc = best
    detail = "|".join(f"{tag}:{_s(v)}" for tag, v in scores)
    if chosen is not None and tc.get("gene_id") != chosen.get("gene_id"):
        detail += f" ({tc.get('gene_symbol') or tc.get('gene_id')})"
    return {"spliceai_ds_max": _s(max((v for _, v in scores if v is not None), key=float)), "spliceai_detail": detail}


def _allele_clin_sig(clin_sig_allele: str, allele: str) -> list[str]:
    """Terms for ``allele`` from ``'A:pathogenic;T:benign'``. VEP writes an empty
    prefix for a deletion (``':pathogenic'``), which matches the ``-`` allele."""
    out: list[str] = []
    for entry in str(clin_sig_allele).split(";"):
        prefix, sep, term = entry.partition(":")
        if not sep or not term:
            continue
        if prefix == allele or (prefix == "" and allele == "-"):
            out.append(term)
    return out


def _colocated(payload: dict[str, Any], allele: str) -> dict[str, str]:
    """rsID, allele-keyed gnomAD frequencies, ClinVar terms and VCV ids from the dbSNP
    entries only — HGMD (CM…/CS…) and COSMIC (COSV…) carry none.

    ``rsid`` is the first rs entry; the other columns are a union over every rs entry
    at the position (rare; the payload keeps the full list). ClinVar terms are the
    allele-specific ``clin_sig_allele`` ones when VEP gives any for this allele, else
    the record's merged ``clin_sig`` — over-inclusion is the safe direction for a
    courtesy column, and the citable ClinVar retriever matches the exact allele."""
    rs = [cv for cv in payload.get("colocated_variants") or [] if str(cv.get("id", "")).startswith("rs")]
    cols = {"rsid": "", "vep_gnomade_af": "", "vep_gnomadg_af": "", "vep_clin_sig": "", "vep_clinvar_ids": ""}
    if not rs:
        return cols
    cols["rsid"] = str(rs[0]["id"])
    for cv in rs:
        freq = (cv.get("frequencies") or {}).get(allele)
        if freq:
            cols["vep_gnomade_af"] = _s(freq.get("gnomade"))
            cols["vep_gnomadg_af"] = _s(freq.get("gnomadg"))
            break
    sig: set[str] = set()
    vcv: set[str] = set()
    for cv in rs:
        sig.update(_allele_clin_sig(cv.get("clin_sig_allele") or "", allele) or [str(t) for t in cv.get("clin_sig") or []])
        vcv.update(x for x in (cv.get("var_synonyms") or {}).get("ClinVar") or [] if str(x).startswith("VCV"))
    cols["vep_clin_sig"] = ",".join(sorted(sig))
    cols["vep_clinvar_ids"] = ",".join(sorted(vcv))
    return cols
