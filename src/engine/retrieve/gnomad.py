"""gnomAD — population frequencies from the gnomAD GraphQL API (v4, GRCh38).

Why a retriever of its own when VEP already carries gnomAD frequencies: VEP's copy is a
courtesy annotation without homozygote counts, filters or a page to open. ACMG
PM2/BA1/BS1 want a number a judge can check, so for the variants that survive the
prefilter this module fetches gnomAD's own variant object and stores it verbatim.

How it asks. One POST per batch of up to 25 variants: GraphQL field aliases ``v0…v24``
over a shared fragment, each ``variantId`` spelled ``<chrom>-<pos>-<ref>-<alt>`` exactly
as ingest normalised it — gnomAD does not left-align or trim for you, and a
non-canonical indel comes back "not found". The server caps query cost at 25 variant
lookups and, above ~10 KB of query text, has been seen answering with an unrelated
cached response; batches are therefore also capped by text size, and every response
must echo the aliases and variant ids that were sent or the batch fails.

What it refuses to ask. gnomAD's ``variant()`` field answers nuclear ``[ACGT]+``
alleles only. The mitochondrion is served by a separate ``mitochondrial_variant`` field
with a different shape (heteroplasmy, ``ac_hom``/``ac_het``, no exome/genome/joint) that
this module does not query, so an ``MT`` key sent through ``variant()`` could only come
back "not found" — a fabricated absence. Alleles outside ``[ACGT]+`` (``N``, symbolic)
are rejected by the server as ``Invalid variant ID`` for the whole batch. Both are
therefore refused *before any request*: :meth:`GnomadRetriever.retrieve` raises with
counts only, never returning ``[]`` for a key it did not ask about. Lower-case alleles
are upper-cased, which is the server's own normalisation (its echo is upper-case).

How absence looks. ``data.vN == null`` together with exactly one absence message per
null alias — ``Variant not found`` or, for a subset dataset such as
``gnomad_r4_non_ukb``, ``Variant not found in selected subset.`` (the variant exists in
gnomAD but was not observed in that subset). The errors carry no path, so the null
alias is what attributes them; the counts must agree. Any other message (``Invalid
variant ID``, ``Multiple variants found``, a cost error, ``Something went wrong``)
raises, as does a non-JSON body — the 429 page is HTML and :mod:`engine.retrieve.http`
already retries it with a long floor for this host, so nothing here sleeps.

Cache poisoning. :class:`~engine.retrieve.http.Http` caches every 2xx as definitive,
but gnomAD reports a resolver failure with HTTP 200 too. A cached body that fails
validation here is fetched again live once (``cache_ok=False``) instead of replaying
the same failure on every rerun; that retry is not written back, so until the entry is
replaced by hand the batch costs one live request per run and its ``retrieved_at``
moves. The durable fix belongs in ``http.py``.

Cache granularity. The HTTP cache key is the whole batch body, so a warm-cache replay
is byte-identical only for the same keys in the same order: reordering the subset or
shifting a batch boundary re-fetches the affected batches live and moves their
``retrieved_at``. Nothing here sorts — the orchestrator passes keys in genome order,
which is deterministic.

What the columns mean. gnomAD v4 exposes ``ac``/``an``/``homozygote_count`` on a
*joint* (exome+genome) block but no ``af`` on it, nor on any population. So
``gnomad_af``/``gnomad_ac``/``gnomad_an``/``gnomad_nhom`` come from the joint block
when it is present, with ``af = ac / an``; when it is absent (``gnomad_r4_non_ukb``
has none) they are the exome and genome blocks combined — ``ac`` summed, ``an``
summed, ``nhom`` summed, ``af = ac / an``. Which rule applied is visible in the payload
(a ``joint`` block or not). ``gnomad_grpmax_*`` is the genetic-ancestry group with the
highest ``ac / an`` over the same blocks (gnomAD v4 calls this *grpmax*), skipping the
bottlenecked groups by gnomAD convention (ami, asj, fin, mid, oth/remaining); sex-split
and HGDP/1kG subset entries are not groups. ``gnomad_faf95_*`` is the *filtering*
allele frequency gnomAD's page labels "Popmax Filtering AF (95% CI)" — the number
BA1/BS1 thresholds are usually applied to — from the joint block when present, else
the higher of the exome and genome blocks. ``gnomad_filters`` joins the exome and
genome filters as ``exome:AC0;genome:AS_VQSR`` ('' means PASS in both); the joint
block's ``discrepant_frequencies`` is left out because it flags exome-vs-genome
disagreement on perfectly good common variants, not low quality.

Version. The API has no gnomAD release string. ``version()`` cites the dataset id
queried and the only date the API exposes (``meta.clinvar_release_date``, observed with
one extra cached request) — a function of the request alone, so every record and the
manifest carry the same string whatever was found. Every payload carries
``reference_genome``, which must be GRCh38 like the engine's keys.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any, Callable, Iterator, TypeVar

from engine.retrieve.http import Http, Response
from engine.retrieve.store import EvidenceRecord, VariantKey

log = logging.getLogger(__name__)

API_URL = "https://gnomad.broadinstitute.org/api"
MAX_ALIASES = 25        # server: "Query is too expensive (26). Maximum allowed cost is 25."
MAX_QUERY_CHARS = 8000  # server served stale unrelated data above ~10 KB; 9.3 KB was fine
REFERENCE_GENOME = "GRCh38"
NOT_FOUND = "Variant not found"
NOT_FOUND_IN_SUBSET = "Variant not found in selected subset."
ABSENT_MESSAGES = frozenset({NOT_FOUND, NOT_FOUND_IN_SUBSET})
BOTTLENECKED_POPS = frozenset({"ami", "asj", "fin", "mid", "oth", "remaining"})

_POPS = "populations { id ac an homozygote_count hemizygote_count }"
_FAF = "faf95 { popmax popmax_population }"
_BLOCK = f"{{ ac an af homozygote_count hemizygote_count filters {_FAF} {_POPS} }}"
_JOINT = f"{{ ac an homozygote_count hemizygote_count filters {_FAF} {_POPS} }}"  # joint has no af
FRAGMENT = ("fragment F on VariantDetails { variant_id reference_genome chrom pos ref alt rsids caid flags "
            f"exome {_BLOCK} genome {_BLOCK} joint {_JOINT} }}")
META_QUERY = "{ meta { clinvar_release_date } }"

_DATASET_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")  # goes into the query text unquoted
_ALLELE = re.compile(r"^[ACGTacgt]+$")                 # gnomAD's own variant-id grammar

T = TypeVar("T")


class GnomadError(RuntimeError):
    """The API answered in a shape this module does not understand, or a key was
    asked that it cannot put to the API — never absence."""


def unaskable(k: VariantKey) -> str | None:
    """Why gnomAD's ``variant()`` field cannot be asked about ``k``, or ``None``."""
    if k[0] == "MT":
        return "mitochondrial"
    if not (_ALLELE.match(k[2]) and _ALLELE.match(k[3])):
        return "non-ACGT allele"
    return None


def variant_id(k: VariantKey) -> str:
    """gnomAD's spelling of a nuclear key: ``1-11796321-G-A`` (alleles upper-cased).
    Raises :class:`GnomadError` for a key the API cannot be asked about."""
    why = unaskable(k)
    if why:
        raise GnomadError(f"{why} key cannot be asked of gnomAD's variant() field")
    return f"{k[0]}-{k[1]}-{k[2].upper()}-{k[3].upper()}"


def variant_url(vid: str, dataset: str) -> str:
    return f"https://gnomad.broadinstitute.org/variant/{vid}?dataset={dataset}"


def build_query(ids: list[str], dataset: str) -> str:
    """The batch query text: the fragment once, then one alias per variant id."""
    parts = [f"v{i}: variant(variantId: {json.dumps(vid)}, dataset: {dataset}) {{ ...F }}"
             for i, vid in enumerate(ids)]
    return FRAGMENT + " query Batch { " + " ".join(parts) + " }"


class GnomadRetriever:
    source = "gnomad"
    columns = (
        "gnomad_af",
        "gnomad_ac",
        "gnomad_an",
        "gnomad_nhom",
        "gnomad_af_exome",
        "gnomad_af_genome",
        "gnomad_nhom_exome",
        "gnomad_nhom_genome",
        "gnomad_filters",
        "gnomad_grpmax_af",
        "gnomad_grpmax_pop",
        "gnomad_faf95_popmax",
        "gnomad_faf95_pop",
        "gnomad_dataset",
    )

    def __init__(self, http: Http, *, dataset: str = "gnomad_r4", url: str = API_URL, batch_size: int = MAX_ALIASES):
        if not _DATASET_ID.match(dataset):
            raise ValueError(f"not a gnomAD dataset id: {dataset!r}")
        if not 1 <= batch_size <= MAX_ALIASES:
            raise ValueError(f"batch_size must be 1..{MAX_ALIASES} (server query-cost cap), got {batch_size}")
        self.http = http
        self.dataset = dataset
        self.url = url
        self.batch_size = batch_size
        self._clinvar_release_date: str | None = None

    # ------------------------------------------------------------------ Retriever

    def version(self) -> str:
        """``gnomad_r4 clinvar_release_date=2026-06-06`` — the dataset asked for and the
        API's only version-like field; independent of what was found."""
        return f"{self.dataset} clinvar_release_date={self._meta_date()}"

    def retrieve(self, keys: list[VariantKey]) -> dict[VariantKey, list[EvidenceRecord]]:
        out: dict[VariantKey, list[EvidenceRecord]] = {k: [] for k in keys}
        unique = list(dict.fromkeys(keys))
        rejected = Counter(why for k in unique if (why := unaskable(k)))
        if rejected:  # before any request: a refused key must never look absent
            detail = ", ".join(f"{n} {why}" for why, n in sorted(rejected.items()))
            raise GnomadError(f"{sum(rejected.values())} of {len(unique)} keys cannot be asked of gnomAD's "
                              f"variant() field ({detail}); filter them upstream")
        batches = list(self._batches(unique))
        for i, batch in enumerate(batches, 1):
            log.info("gnomad: batch %d/%d (%d variants)", i, len(batches), len(batch))
            for k, rec in self._fetch(batch):
                out[k] = [rec]
        return out

    def extract(self, records: list[EvidenceRecord]) -> dict[str, str]:
        if not records:
            return {c: "" for c in self.columns}
        rec = records[0]
        v = rec.payload
        ac, an, nhom = _totals(v)
        exome, genome = v.get("exome"), v.get("genome")
        grpmax_af, grpmax_pop = _grpmax(v)
        faf95, faf95_pop = _faf95(v)
        return {
            "gnomad_af": _ratio(ac, an),
            "gnomad_ac": _num(ac),
            "gnomad_an": _num(an),
            "gnomad_nhom": _num(nhom),
            "gnomad_af_exome": _num(_field(exome, "af")),
            "gnomad_af_genome": _num(_field(genome, "af")),
            "gnomad_nhom_exome": _num(_field(exome, "homozygote_count")),
            "gnomad_nhom_genome": _num(_field(genome, "homozygote_count")),
            "gnomad_filters": _filters(exome, genome),
            "gnomad_grpmax_af": grpmax_af,
            "gnomad_grpmax_pop": grpmax_pop,
            "gnomad_faf95_popmax": faf95,
            "gnomad_faf95_pop": faf95_pop,
            "gnomad_dataset": str(rec.query.get("dataset", "")),
        }

    # ------------------------------------------------------------------ internals

    def _batches(self, keys: list[VariantKey]) -> Iterator[list[VariantKey]]:
        """Up to ``batch_size`` keys per batch, in the order given, and never more query
        text than the server handles correctly (long indel alleles make aliases long)."""
        batch: list[VariantKey] = []
        for k in keys:
            if batch and (len(batch) >= self.batch_size
                          or len(build_query([variant_id(x) for x in batch + [k]], self.dataset)) > MAX_QUERY_CHARS):
                yield batch
                batch = []
            batch.append(k)
        if batch:
            yield batch

    def _fetch(self, batch: list[VariantKey]) -> Iterator[tuple[VariantKey, EvidenceRecord]]:
        ids = [variant_id(k) for k in batch]
        query = build_query(ids, self.dataset)
        if len(query) > MAX_QUERY_CHARS:
            raise GnomadError(f"query text is {len(query)} chars for one variant; the API misbehaves above ~10 KB")
        aliases = {f"v{i}": vid for i, vid in enumerate(ids)}
        resp, data = self._post(query, lambda body: _attribute(body, aliases))
        for k, (alias, vid) in zip(batch, aliases.items()):
            v = data[alias]
            if v is None:
                continue  # absent from gnomAD: a result, not an error
            _check_genome(v.get("reference_genome"), self.dataset)
            yield k, EvidenceRecord(
                record_id=f"{self.source}:{vid}",
                source=self.source,
                source_version=self.version(),
                query={"api": self.url, "dataset": self.dataset, "variantId": vid, "fragment": FRAGMENT},
                url=variant_url(vid, self.dataset),
                retrieved_at=resp.retrieved_at,
                payload=v,
            )

    def _post(self, query: str, parse: Callable[[dict[str, Any]], T]) -> tuple[Response, T]:
        """POST one GraphQL document and validate the body. A *cached* body that fails
        validation is fetched again live, once, so a resolver failure that came back as
        HTTP 200 (and was cached as definitive) cannot replay forever."""
        body = {"query": query, "variables": {}}
        resp = self.http.post(self.url, body)
        try:
            return resp, parse(_json_body(resp))
        except GnomadError:
            if not resp.from_cache:
                raise
            log.warning("gnomad: a cached response failed validation; fetching it again live")
            resp = self.http.post(self.url, body, cache_ok=False)
            return resp, parse(_json_body(resp))

    def _meta_date(self) -> str:
        if self._clinvar_release_date is None:
            _resp, self._clinvar_release_date = self._post(META_QUERY, _meta_date_of)
        return self._clinvar_release_date


# ---------------------------------------------------------------------- response shape

def _json_body(resp: Response) -> dict[str, Any]:
    if resp.status != 200:
        raise GnomadError(f"HTTP {resp.status} from gnomAD: {resp.text[:200]!r}")
    try:
        body = json.loads(resp.text)
    except ValueError as e:  # the 429 page and other non-API answers are HTML
        raise GnomadError(f"non-JSON body from gnomAD: {resp.text[:200]!r}") from e
    if not isinstance(body, dict):
        raise GnomadError(f"unexpected JSON from gnomAD: {resp.text[:200]!r}")
    return body


def _meta_date_of(body: dict[str, Any]) -> str:
    date = ((body.get("data") or {}).get("meta") or {}).get("clinvar_release_date")
    if not isinstance(date, str) or not date:
        raise GnomadError(f"meta query returned no clinvar_release_date: {json.dumps(body)[:200]}")
    return date


def _attribute(body: dict[str, Any], aliases: dict[str, str]) -> dict[str, dict[str, Any] | None]:
    """Map alias → variant object or ``None`` (absent), or raise for any other shape.

    Absence and only absence: a null alias matched one-for-one by an absence message.
    Everything else — another message, an unmatched null, foreign aliases (the
    stale-cache bug), a variant that echoes a different id — is an API failure.
    """
    data = body.get("data")
    if not isinstance(data, dict):
        raise GnomadError(f"gnomAD returned no data object: {json.dumps(body)[:200]}")
    if set(data) != set(aliases):
        raise GnomadError(f"response aliases {sorted(data)[:6]} do not match the {len(aliases)} sent "
                          "(stale server cache?)")
    errors = body.get("errors") or []
    messages = [e.get("message") if isinstance(e, dict) else repr(e) for e in errors]
    unexpected = [m for m in messages if m not in ABSENT_MESSAGES]
    if unexpected:
        raise GnomadError(f"gnomAD errors: {unexpected}")
    n_null = sum(1 for v in data.values() if v is None)
    if len(messages) != n_null:
        raise GnomadError(f"{len(messages)} absence errors for {n_null} null aliases")
    for alias, v in data.items():
        if v is None:
            continue
        if not isinstance(v, dict) or v.get("variant_id") != aliases[alias]:
            raise GnomadError(f"alias {alias} did not echo the variant id requested")
    return data


def _check_genome(rg: Any, dataset: str) -> None:
    """The engine's keys are GRCh38 coordinates; a dataset on another build would
    answer for the wrong position without any other sign of it."""
    if not isinstance(rg, str) or not rg:
        raise GnomadError("variant object carries no reference_genome")
    if rg != REFERENCE_GENOME:
        raise GnomadError(f"dataset {dataset} is on {rg}; the engine's keys are {REFERENCE_GENOME}")


# ---------------------------------------------------------------------- projections

def _field(block: Any, name: str) -> Any:
    return block.get(name) if isinstance(block, dict) else None


def _num(x: Any) -> str:
    return "" if x is None else repr(x)


def _ratio(ac: Any, an: Any) -> str:
    return "" if ac is None or not an else repr(ac / an)


def _sum(blocks: list[dict[str, Any]], name: str) -> int | None:
    vals = [b[name] for b in blocks if b.get(name) is not None]
    return sum(vals) if vals else None


def _blocks(v: dict[str, Any]) -> list[dict[str, Any]]:
    """The frequency blocks the totals are taken from: joint when present, else the
    exome and genome blocks that are."""
    joint = v.get("joint")
    if isinstance(joint, dict):
        return [joint]
    return [b for b in (v.get("exome"), v.get("genome")) if isinstance(b, dict)]


def _totals(v: dict[str, Any]) -> tuple[Any, Any, Any]:
    """(ac, an, nhom) — the joint block's own numbers, or exome+genome summed."""
    blocks = _blocks(v)
    return _sum(blocks, "ac"), _sum(blocks, "an"), _sum(blocks, "homozygote_count")


def _filters(exome: Any, genome: Any) -> str:
    return ";".join(f"{name}:{f}" for name, b in (("exome", exome), ("genome", genome))
                    if isinstance(b, dict) for f in (b.get("filters") or []))


def _is_group(pid: Any) -> bool:
    """A genetic-ancestry group eligible for grpmax: not sex-split (``nfe_XX``), not a
    global sex total, not an HGDP/1kG subset (``hgdp:japanese``), not the empty id
    gnomAD puts on the joint total, and not bottlenecked."""
    return (isinstance(pid, str) and pid != "" and "_" not in pid and ":" not in pid
            and pid not in ("XX", "XY") and pid not in BOTTLENECKED_POPS)


def _grpmax(v: dict[str, Any]) -> tuple[str, str]:
    ac: dict[str, int] = {}
    an: dict[str, int] = {}
    for b in _blocks(v):
        seen: set[str] = set()
        for p in b.get("populations") or []:
            pid = p.get("id")
            if not _is_group(pid) or pid in seen or p.get("ac") is None or p.get("an") is None:
                continue
            seen.add(pid)  # the joint block lists some ids twice with identical values
            ac[pid] = ac.get(pid, 0) + p["ac"]
            an[pid] = an.get(pid, 0) + p["an"]
    best: tuple[float, str] | None = None
    for pid in sorted(ac):  # alphabetical tie-break keeps the choice deterministic
        if an[pid] > 0:
            af = ac[pid] / an[pid]
            if best is None or af > best[0]:
                best = (af, pid)
    if best is None:
        return "", ""
    return repr(best[0]), best[1] if best[0] > 0 else ""


def _faf95(v: dict[str, Any]) -> tuple[str, str]:
    """gnomAD's filtering allele frequency: the joint block's, else the higher of the
    exome and genome blocks (exome wins a tie — fixed order, so deterministic)."""
    best: tuple[float, str] | None = None
    for b in _blocks(v):
        faf = b.get("faf95")
        val = faf.get("popmax") if isinstance(faf, dict) else None
        if val is not None and (best is None or val > best[0]):
            best = (val, str(faf.get("popmax_population") or ""))
    return ("", "") if best is None else (repr(best[0]), best[1])
