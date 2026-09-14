"""Literature — the "retrieve and verify papers" arm of stage 2.

A paper cited later must exist, and must be what the citation says it is. This
module fetches papers from Europe PMC's REST API (PubMed plus citation counts,
open-access flags and abstracts behind one JSON endpoint) and stores each one as an
:class:`~engine.retrieve.store.EvidenceRecord` whose payload is the raw Europe PMC
result object, so the title, journal, year, DOI and every other column can be
re-checked against the source. Identity is the PMID (``pmid:<pmid>``); results
without one (preprints, PMC-only deposits) are not citable here and are skipped.

Two kinds of record, so provenance never collides. A *paper* record (``pmid:<pmid>``,
query ``{"pmid": ...}``) is the same whichever route found it — a search, ``fetch`` or
``fetch_many`` — so the store holds one record per paper and a claim can cite it
without caring how it was found. A *search* record (``pmid-search:<sha256>``) holds
exactly what was asked (query, filter, ``max_results``), what Europe PMC answered
(hit count, version) and the PMIDs it returned in order; it is the reason a paper was
on the table. :meth:`LiteratureRetriever.search` returns both.

Absence is a result, not an error: a PMID Europe PMC does not know answers
``hitCount: 0`` and becomes ``None``; an API failure raises :class:`LiteratureError`
so a broken run cannot pass as "no papers found". Failure has three shapes — HTTP
5xx/timeouts (retried by :class:`~engine.retrieve.http.Http` first), an HTTP 200
carrying ``errCode``/``errMsg`` (empty query, page size out of 1..1000), and an HTTP
200 whose body is a bare ``{"version": "6.9"}`` (bad ``sort`` or ``cursorMark``) — so
the gate is the *presence* of ``hitCount``, never ``.get("hitCount", 0)``. A body
without ``version`` is also refused: a record's ``source_version`` must be observed,
never implied.

Rate-limit etiquette. Europe PMC publishes no hard limit and sends no rate headers;
we stay at or under 3 requests/second on ``www.ebi.ac.uk`` (pinned on the shared
limiter unless the caller set a rate already), one request in flight, pages of 25,
and ID batches kept under Europe PMC's 1500-character query limit — the service is
slow rather than strict (5-30 s responses and short 502/503 bursts were observed), so
the retries and backoff live in ``Http``. NCBI E-utilities, used only by the optional
:meth:`LiteratureRetriever.verify_ncbi`, enforce 3 requests/second per source IP
without an API key (HTTP 429 beyond that) and 10/second with one; the limiter for
``eutils.ncbi.nlm.nih.gov`` is pinned accordingly, never above a rate the orchestrator
already chose, and ``NCBI_API_KEY`` is honoured when set. E-utilities take the key
only as a URL parameter, so setting it changes the cache identity of ``verify_ncbi``
requests and the key is written into the cache entry's request record on disk.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any, Iterator

from engine.retrieve.http import Http, Response
from engine.retrieve.store import EvidenceRecord

log = logging.getLogger(__name__)

SOURCE = "pmid"
SEARCH_SOURCE = "pmid-search"
EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/"
EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
ARTICLE_URL = "https://europepmc.org/article/MED/{pmid}"
SEARCH_URL = "https://europepmc.org/search?query={query}"
PEER_REVIEWED = "SRC:MED"  # PubMed-indexed only: excludes preprints (PPR), patents, PMC-only deposits
MAX_PAGE_SIZE = 1000       # Europe PMC: "Valid size is between 1 and 1000"
MAX_QUERY_CHARS = 1400     # Europe PMC rejects a query of 1500 characters or more; keep a margin
EUROPE_PMC_PER_SECOND = 3.0
NCBI_PER_SECOND = 3.0      # per source IP without an API key
NCBI_KEYED_PER_SECOND = 10.0

COLUMNS = ("pmid", "title", "authors", "journal", "year", "doi", "pmcid", "has_abstract", "cited_by", "open_access")

_PMID = re.compile(r"^[0-9]+$")
# Only a bare <name> or </name> is markup. Europe PMC sends no attributes, and a raw
# '<' meaning "less than" ('p < 0.05', 'n<12') is left unescaped in abstracts — an
# unanchored '<[^>]+>' ate whole results sentences up to the next closing tag.
_TAG = re.compile(r"</?([A-Za-z][A-Za-z0-9]*)\s*/?>")
_INLINE = frozenset({"i", "b", "em", "strong", "u", "italic", "bold", "sup", "sub", "span", "a", "small", "mark"})
_WS = re.compile(r"\s+")


class LiteratureError(RuntimeError):
    """The API answered, but not with a result — distinct from "paper not found"."""


@dataclass(frozen=True)
class SearchResult:
    """One search: its own record (the query as asked, the hit count, the PMIDs in
    order) and one paper record per PubMed result, in Europe PMC's relevance order."""

    record: EvidenceRecord
    papers: list[EvidenceRecord]

    @property
    def pmids(self) -> list[str]:
        return list(self.record.payload["pmids"])


class LiteratureRetriever:
    """Papers by query or by PMID, as evidence records. Not a variant
    :class:`~engine.retrieve.Retriever`: literature is keyed by PMID, and no genomic
    coordinate is ever sent — searches use gene symbols and text tokens only."""

    source = SOURCE
    columns = COLUMNS

    def __init__(self, http: Http, *, base_url: str = EUROPE_PMC_URL, page_size: int = 25):
        self.http = http
        self.base_url = base_url.rstrip("/") + "/"
        self.page_size = page_size
        self._version = ""
        _pin_rate(http, _host(self.base_url), EUROPE_PMC_PER_SECOND)

    def version(self) -> str:
        """Europe PMC REST version as observed on the latest response ('' before any)."""
        return self._version

    # ---- Europe PMC

    def search(self, query: str, *, max_results: int = 25, peer_reviewed_only: bool = True) -> SearchResult:
        """Papers matching ``query``, plus the search itself as a record.

        ``peer_reviewed_only`` restricts to ``SRC:MED``; without it, results that carry
        no PMID (preprints, PMC-only deposits) are skipped and counted.
        """
        if max_results < 1:
            raise ValueError(f"max_results must be >= 1, got {max_results}")
        sent = f"({query}) AND {PEER_REVIEWED}" if peer_reviewed_only else query
        page_size = min(self.page_size, max_results)
        papers: dict[str, EvidenceRecord] = {}  # keyed by PMID: a shifting index can repeat one across pages
        skipped = 0
        head: tuple[dict, Response] | None = None
        for body, resp in self._pages(sent, page_size):
            head = head or (body, resp)
            for result in body.get("resultList", {}).get("result", []):
                pmid = result.get("pmid")
                if not pmid:
                    skipped += 1
                elif pmid not in papers:
                    papers[pmid] = self._paper(result, body, resp)
            if len(papers) >= max_results:
                break
        if skipped:
            log.info("literature search: %d results without a PMID skipped", skipped)
        assert head is not None  # _pages yields the first page or raises
        found = list(papers.values())[:max_results]
        sent_params = _params(sent, page_size, None)
        record = EvidenceRecord(
            # identity: what was sent plus how much was wanted — a different ask is a different record
            record_id=f"{SEARCH_SOURCE}:{_digest({'sent': sent_params, 'max_results': max_results})}",
            source=SEARCH_SOURCE,
            source_version=_version_label(head[0]),
            query={"query": query, "peer_reviewed_only": peer_reviewed_only, "max_results": max_results},
            url=SEARCH_URL.format(query=urllib.parse.quote(sent)),
            retrieved_at=head[1].retrieved_at,
            payload={
                "sent": sent_params,
                "hitCount": head[0]["hitCount"],
                "pmids": [p.payload["pmid"] for p in found],
                "skipped_without_pmid": skipped,
            },
        )
        return SearchResult(record, found)

    def fetch(self, pmid: str) -> EvidenceRecord | None:
        """The PubMed record for ``pmid``, or ``None`` when Europe PMC has no such paper."""
        pmid = _valid_pmid(pmid)
        return self._lookup(f"EXT_ID:{pmid} AND {PEER_REVIEWED}", [pmid])[pmid]

    def fetch_many(self, pmids: list[str]) -> dict[str, EvidenceRecord | None]:
        """Batched :meth:`fetch`: every requested PMID (normalised, deduplicated) is a
        key, absent ones map to ``None``."""
        wanted = list(dict.fromkeys(_valid_pmid(p) for p in pmids))
        batches = list(_batches(wanted))
        found: dict[str, EvidenceRecord | None] = {}
        for i, batch in enumerate(batches, 1):
            log.info("literature: PMID batch %d/%d (%d ids)", i, len(batches), len(batch))
            found.update(self._lookup(_batch_query(batch), batch))
        return found

    def verify(self, pmid: str) -> bool:
        """True when Europe PMC knows ``pmid`` as a PubMed record. Raises on API failure."""
        return self.fetch(pmid) is not None

    def _lookup(self, query: str, wanted: list[str]) -> dict[str, EvidenceRecord | None]:
        found: dict[str, EvidenceRecord | None] = {p: None for p in wanted}
        for body, resp in self._pages(query, len(wanted)):
            for result in body.get("resultList", {}).get("result", []):
                pmid = result.get("pmid")
                if pmid in found and found[pmid] is None:
                    found[pmid] = self._paper(result, body, resp)
        return found

    def _pages(self, query: str, page_size: int) -> Iterator[tuple[dict, Response]]:
        """Result pages, lazily — the next one is fetched only if the consumer keeps
        asking. Relevance order is Europe PMC's default (no ``sort``)."""
        cursor: str | None = None  # first page: Europe PMC's implicit cursorMark '*'
        page = 0
        while True:
            resp = self.http.get(self.base_url + "search", params=_params(query, page_size, cursor))
            body = self._checked(resp)
            page += 1
            results = body.get("resultList", {}).get("result", [])
            log.debug("literature: page %d, %d results", page, len(results))
            yield body, resp
            nxt = body.get("nextCursorMark")
            if not results or nxt is None or nxt == cursor:
                return  # last page: Europe PMC omits nextCursorMark (or repeats it)
            cursor = nxt

    def _checked(self, resp: Response) -> dict:
        """Parsed body, or :class:`LiteratureError` for any of the failure shapes."""
        try:
            body = resp.json()
        except ValueError as e:
            raise LiteratureError(f"Europe PMC: HTTP {resp.status} with a non-JSON body: {resp.text[:200]!r}") from e
        if not isinstance(body, dict) or "hitCount" not in body:
            detail = (f"errCode={body.get('errCode')} errMsg={body.get('errMsg')!r}"
                      if isinstance(body, dict) and "errCode" in body else f"no hitCount in {resp.text[:200]!r}")
            raise LiteratureError(f"Europe PMC: HTTP {resp.status} without a result: {detail}")
        if not body.get("version"):
            raise LiteratureError(f"Europe PMC: HTTP {resp.status} without a version to cite: {resp.text[:200]!r}")
        self._version = _version_label(body)
        return body

    def _paper(self, result: dict, body: dict, resp: Response) -> EvidenceRecord:
        pmid = result["pmid"]
        return EvidenceRecord(
            record_id=f"{SOURCE}:{pmid}",
            source=SOURCE,
            source_version=_version_label(body),
            query={"pmid": pmid},
            url=ARTICLE_URL.format(pmid=pmid),
            retrieved_at=resp.retrieved_at,
            payload=result,
        )

    # ---- NCBI (optional cross-check)

    def verify_ncbi(self, pmid: str) -> bool:
        """True when PubMed itself (E-utilities esummary) knows ``pmid``. A second opinion
        for :meth:`verify`; a missing id is an explicit per-id error in the summary."""
        pmid = _valid_pmid(pmid)
        params = {"db": "pubmed", "id": pmid, "retmode": "json", "tool": "engine"}
        key = os.environ.get("NCBI_API_KEY")
        if key:
            params["api_key"] = key
        _pin_rate(self.http, _host(EUTILS_URL), NCBI_KEYED_PER_SECOND if key else NCBI_PER_SECOND)
        resp = self.http.get(EUTILS_URL + "esummary.fcgi", params=params)
        try:
            body = resp.json()
        except ValueError as e:
            raise LiteratureError(f"NCBI esummary: HTTP {resp.status} with a non-JSON body: {resp.text[:200]!r}") from e
        summary = body.get("result", {}).get(pmid) if isinstance(body, dict) else None
        if not isinstance(summary, dict):
            raise LiteratureError(f"NCBI esummary: no entry for the requested id in {resp.text[:200]!r}")
        return "error" not in summary

    # ---- projection

    def extract(self, record: EvidenceRecord | None) -> dict[str, str]:
        """Project one paper onto :data:`COLUMNS` (strings; all '' for ``None``)."""
        if record is None:
            return {c: "" for c in COLUMNS}
        p = record.payload
        # Bookshelf chapters (GeneReviews) arrive under SRC:MED with no journalInfo, and
        # their pubYear is the book's inception (1993) — the chapter date is elsewhere.
        book = p.get("bookOrReportDetails")
        journal = _get(p, "journalInfo", "journal", "title")
        if journal is None and book:
            journal = book.get("comprisingTitle") or book.get("publisher")
        year = (p.get("firstPublicationDate") or "")[:4] if book else p.get("pubYear")
        cited = p.get("citedByCount")
        return {
            "pmid": str(p.get("pmid") or ""),
            "title": _text(p.get("title") or ""),
            "authors": p.get("authorString") or "",
            "journal": journal or "",
            "year": str(year or ""),
            "doi": p.get("doi") or "",
            "pmcid": p.get("pmcid") or "",
            "has_abstract": "Y" if abstract_text(record) else "N",
            "cited_by": "" if cited is None else str(cited),
            "open_access": p.get("isOpenAccess") or "",
        }


def abstract_text(record: EvidenceRecord) -> str:
    """The abstract as plain text — section tags such as ``<h4>Background</h4>`` removed,
    entities decoded, whitespace collapsed; '' when the paper has none."""
    return _text(record.payload.get("abstractText") or "")


def gene_query(gene: str, disease: str | None = None) -> str:
    """Europe PMC query for papers naming ``gene`` (and ``disease``) in title or abstract.

    ``TITLE_ABS:`` is a real field. ``GENE_SYMBOL:`` is not — an unknown tag degrades
    silently to free text — and the text-mined ``GENE_PROTEIN:`` covers open-access
    full text only, so it under-recalls by ~50x. Synonym expansion stays off
    (:meth:`LiteratureRetriever.search` sends ``synonym=false``) so counts reproduce.
    """
    q = f'TITLE_ABS:"{_term(gene)}"'
    if disease:
        q += f' AND TITLE_ABS:"{_term(disease)}"'
    return q


def _term(s: str) -> str:
    s = s.strip()
    if not s or '"' in s:  # Europe PMC parses unbalanced quotes leniently and returns junk, not an error
        raise ValueError(f"query term must be non-empty and free of double quotes: {s!r}")
    return s


def _valid_pmid(pmid: str | int) -> str:
    """``EXT_ID:notanumber`` gets hitCount 0 from Europe PMC — indistinguishable from a
    missing paper — so a malformed id is rejected before any request is made. Leading
    zeros are dropped: ``EXT_ID`` is an exact string match and ``07647779`` finds nothing."""
    s = str(pmid).strip()
    if not _PMID.match(s):
        raise ValueError(f"not a PMID: {pmid!r}")
    return str(int(s))


def _batch_query(pmids: list[str]) -> str:
    return "(" + " OR ".join(f"EXT_ID:{p}" for p in pmids) + f") AND {PEER_REVIEWED}"


def _batches(pmids: list[str]) -> Iterator[list[str]]:
    """Split ids so each OR-query stays within Europe PMC's page-size and length limits."""
    batch: list[str] = []
    for p in pmids:
        if batch and (len(batch) >= MAX_PAGE_SIZE or len(_batch_query(batch + [p])) > MAX_QUERY_CHARS):
            yield batch
            batch = []
        batch.append(p)
    if batch:
        yield batch


def _text(s: str) -> str:
    """Plain text from Europe PMC markup: entities first (titles arrive with their tags
    entity-encoded, ``&lt;i&gt;CFTR&lt;/i&gt;``), then tags. Inline tags vanish, as in
    PubMed's own text rendering (``Ca<sup>2+</sup>`` → ``Ca2+``); section tags become a
    space (``.<h4>Results</h4>The`` → ``. Results The``)."""
    return _WS.sub(" ", _TAG.sub(_tag_repl, html.unescape(s))).strip()


def _tag_repl(m: re.Match[str]) -> str:
    return "" if m.group(1).lower() in _INLINE else " "


def _params(query: str, page_size: int, cursor: str | None) -> dict[str, Any]:
    params: dict[str, Any] = {"query": query, "format": "json", "resultType": "core", "pageSize": page_size, "synonym": "false"}
    if cursor is not None:
        params["cursorMark"] = cursor
    return params


def _version_label(body: dict) -> str:
    return f"Europe PMC REST {body['version']}"


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _get(d: Any, *path: str) -> Any:
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc


def _pin_rate(http: Any, host: str, per_second: float) -> None:
    """Etiquette for ``host`` on the shared limiter — unless a rate is already set,
    since the orchestrator's table may be stricter for runs that share an IP."""
    limiter = getattr(http, "limiter", None)
    if limiter is not None:
        limiter.per_second.setdefault(host, per_second)
