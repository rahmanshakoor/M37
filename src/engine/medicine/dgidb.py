"""DGIdb — drug–gene interactions from the DGIdb v5 GraphQL API (dgidb.org).

Why a retriever of its own. Stage 6 must say which compounds are known to act on a
candidate gene, with what kind of action, on whose authority — and it must be able to
say "none known" with a citation rather than a shrug. DGIdb aggregates that from ~16
interaction sources (ChEMBL, PharmGKB, GuideToPharmacology, FDA, CIViC, TTD, ...) behind
one endpoint, and every interaction row carries its own source names and versions, so
one record here is one gene–drug pair exactly as DGIdb reports it.

How it asks. One POST per gene: ``genes(names: [SYMBOL])`` with the gene node and its
full, unpaginated ``interactions`` list (the ``$names`` variable, never string
interpolation). A request per gene — not one batch for all genes — keeps the HTTP cache
keyed by gene, so adding a candidate to a rerun re-fetches nothing that was already
fetched and every earlier record stays byte-identical. ``names:`` upper-cases the input
and compares it to the stored name *exactly* — observed live: ``cftr`` and ``Cftr``
answer the node ``CFTR``, but ``C9orf72`` and ``C9ORF72`` both answer ``nodes: []``
because the stored name is ``C9orf72`` — so the symbol is upper-cased before it is sent
(a caller's spelling cannot split the cache) and the dozen genes whose canonical
spelling has lowercase letters (``C9orf72``, ``C8orf44-SGK3``, ``mCR`` …) can never be
matched that way. When ``names:`` answers nothing, the symbol is looked up with the
search field ``genes(name:)`` — a case-insensitive *prefix* match (observed: ``TP53`` →
seven ``TP53…`` genes, ``FTR`` → none) that answers with DGIdb's own spelling — asking
only for ``name`` and ``conceptId`` and walking every page; the one gene whose name
equals the symbol up to case is then fetched in full by ``genes(conceptIds:)``, which is
exact and case-sensitive (``HGNC:1884`` finds nothing; ``hgnc:1884`` is CFTR). Record
ids and URLs carry DGIdb's spelling; ``query`` records the request that produced the
node — ``names:`` or ``conceptIds:`` — so it replays to the same cache key. Aliases
match neither field (``P53`` → nothing), so when a symbol is still unknown the
retriever asks ``geneMatches`` once and, on a single DIRECT match, fetches that gene by
the same route; the result says so (``resolved_via``). ``geneMatches`` echoes
``searchTerm`` upper-cased (``cftr`` → ``CFTR``), so the echo is compared
case-insensitively; an AMBIGUOUS answer counts only when exactly one of the genes is
*named* the term (``mCR`` beside NR3C2, whose alias ``MCR`` it is). ``noMatches {
searchTerm }`` is safe to select — only ``matchType`` under ``noMatches`` makes the
server answer HTTP 500 with an empty body — and is never selected.

Two kinds of record, and how absence looks. A *gene* record (``dgidb-gene:<NAME>``)
is the citation for "DGIdb knows this gene and lists N drugs", including N = 0 (BUB1B
is in the druggable genome as a kinase and has no interaction): ``payload.gene`` is the
gene node verbatim minus its ``interactions`` list, and ``payload.n_interactions`` is
the one derived value in any payload here — the length of that list, checkable against
the ``dgidb:`` records that carry it.
An *interaction* record (``dgidb:<NAME>:<drug conceptId>``) is one interaction object
verbatim: drug, approval flags, interaction types, per-source versions, PMIDs, scores.
A symbol DGIdb does not know comes back as ``nodes: []`` with HTTP 200 and no error
from every lookup — that is absence (``gene is None``, no records), distinct from a
known gene with an empty list. Anything else — ``errors[]`` (delivered with HTTP 200),
a body without ``data``, a node that does not echo the symbol or concept id asked, two
nodes for one name, a lookup that names a gene the next request then omits, an
interaction whose ``gene`` is not the node's, a drug without an id — raises
:class:`DgidbError`. HTTP 4xx/5xx raise in :mod:`engine.retrieve.http` (5xx after
retries).

What the payload cannot tell you on its own. ``interactionScore`` is
``evidenceScore × drugSpecificity × geneSpecificity`` computed against the whole release:
it shifts between releases and is not comparable across genes, and it rewards
specificity over evidence (TP53's highest-scoring drug, 1.15, has evidenceScore 10 while
broad cytotoxics with 19 claims score 0.013; a single-source CFTR compound scores 1.08)
— rank within a gene by evidenceScore first, as this module does, and cite the version
with the number. ``interactionTypes`` is usually empty and ``publications`` often
so; neither is required for an interaction to count. ``drug.name`` may be a bare
identifier (``CHEMBL:CHEMBL260560``) for an unnamed ChEMBL compound, and one drug may
appear twice under different concept ids (ELEXACAFTOR as ``rxcui:2256951`` and as
``chembl:CHEMBL4298128``) with different scores and sources — both are kept, each under
its own id, because they are different claims. Several sources are non-commercial or
unclear licence (DTC, CKB-CORE, CGI, TTD); :meth:`DgidbRetriever.sources` lists them.

Version. ``serviceInfo.dataVersion`` reads ``Dec-2023`` while the data was updated
2026-07-14 and the sources are June 2026, so the citable string is the service version
plus ``updatedAt`` — one extra cached request, the same for every record whatever was
found — and each interaction carries its sources' own versions inline. DGIdb refreshes
its data without changing the service version, and nothing in a gene response says
which refresh produced it, so — as in :mod:`engine.retrieve.vep` — when online a
*cached* observation is also made live once per instance and must agree (an
observation just made live is the live one; nothing to compare), and that check runs
*before* the first gene request of a fetch and before :meth:`DgidbRetriever.sources`:
a cache filled under an earlier refresh is refused while it is still all of one
release, never after a fresh response has been added to it. Offline, the cache
reproduces the run that filled it.

Cache poisoning. :class:`~engine.retrieve.http.Http` caches every 2xx as definitive,
but DGIdb delivers ``errors[]`` with HTTP 200 too. A cached body that fails validation
is fetched again live once and, if the live answer validates, written back under the
same key — so the next run replays the healed entry with its ``retrieved_at`` unchanged
and an offline rerun still works, instead of paying one live request per run forever.
Offline, the cached body's own error is reported as such.

Manifest. :attr:`DgidbRetriever.params` is what the stage manifest should record beside
:meth:`DgidbRetriever.version`: the endpoint, every GraphQL document, the search page
size, the alias default, the effective rate and the sort rule — everything that shapes
the output; there is no threshold.

Rate. DGIdb publishes no limit and sends no rate headers; three requests per second on
``dgidb.org`` is pinned on the shared limiter unless the orchestrator set a rate already.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any, Callable, Iterator, TypeVar

from engine.retrieve.http import Http, Response
from engine.retrieve.store import EvidenceRecord

log = logging.getLogger(__name__)

SOURCE = "dgidb"
GENE_SOURCE = "dgidb-gene"
API_URL = "https://dgidb.org/api/graphql"
GENE_URL = "https://dgidb.org/genes/{name}"
PER_SECOND = 3.0
SOURCES_PAGE = 100  # 45 sources observed; one page, but the connection is walked regardless
SEARCH_PAGE = 100   # prefix matches per page of the gene search; a symbol's prefix set is small (CF → 16)
RESOLVE_ALIASES = True
SORT_RULE = "evidenceScore desc, interactionScore desc, drug name asc, drug conceptId asc"
"""Order of a gene's interaction records — see :func:`_rank_key`."""

_NODE = (
    "{ name conceptId longName "
    "geneCategoriesWithSources { name sourceNames } "
    "interactions { id gene { name conceptId } drug { name conceptId approved antiNeoplastic immunotherapy } "
    "interactionScore evidenceScore drugSpecificity geneSpecificity "
    "interactionTypes { type directionality definition } sources { sourceDbName sourceDbVersion } "
    "publications { pmid citation } } }"
)
"""The gene node selection shared by the two full fetches, so a node is the same whichever found it."""
GENE_QUERY = "query GeneDrugs($names: [String!]!) { genes(names: $names) { nodes " + _NODE + " } }"
CONCEPT_QUERY = "query GeneDrugsById($conceptIds: [String!]!) { genes(conceptIds: $conceptIds) { nodes " + _NODE + " } }"
SEARCH_QUERY = (
    "query GeneSearch($name: String!, $first: Int!, $after: String) { genes(name: $name, first: $first, after: $after) { "
    "totalCount pageInfo { hasNextPage endCursor } nodes { name conceptId } } }"
)
MATCH_QUERY = (
    "query Match($terms: [String!]!) { geneMatches(searchTerms: $terms) { "
    "directMatches { searchTerm matchType matches { name conceptId } } "
    "ambiguousMatches { searchTerm matchType matches { name conceptId } } "
    "noMatches { searchTerm } } }"  # never matchType here: HTTP 500 for any unmatched term
)
SERVICE_QUERY = "{ serviceInfo { name version dataVersion updatedAt } }"
SOURCES_QUERY = (
    "query Sources($first: Int!, $after: String) { sources(first: $first, after: $after) { "
    "totalCount pageInfo { hasNextPage endCursor } "
    "nodes { sourceDbName sourceDbVersion fullName license citationShort interactionClaimsCount "
    "sourceTypes { type } } } }"
)

COLUMNS = (
    "dgidb_gene",
    "dgidb_drug",
    "dgidb_drug_concept_id",
    "dgidb_unnamed_compound",   # Y when drug.name is just the identifier (an unnamed ChEMBL compound)
    "dgidb_approved",
    "dgidb_anti_neoplastic",
    "dgidb_immunotherapy",
    "dgidb_interaction_types",  # sorted, ';'-joined; '' is the common case
    "dgidb_directionality",     # ACTIVATING / INHIBITORY / DIRECTIONALITY_UNCLEAR, sorted unique
    "dgidb_evidence_score",
    "dgidb_interaction_score",
    "dgidb_sources",            # Name:version;Name:version — the per-source citations
    "dgidb_pmids",
    "dgidb_interaction_id",
)

_SYMBOL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]*$")  # HGNC-shaped: no blanks, quotes or control characters

T = TypeVar("T")


class DgidbError(RuntimeError):
    """The API answered in a shape this module does not understand — never absence."""


@dataclass(frozen=True)
class GeneResult:
    """What DGIdb said about one symbol.

    ``gene`` is ``None`` when DGIdb does not know the symbol (absence). A known gene
    with nothing to list has a gene record and an empty ``interactions`` — a different,
    citable fact. ``resolved_via`` is ``"alias"`` when ``asked`` was not a gene's name
    and ``geneMatches`` mapped it to ``gene``'s; ``None`` otherwise, including when
    DGIdb spells the same symbol with lowercase letters (``C9ORF72`` → ``C9orf72``).
    """

    asked: str
    gene: EvidenceRecord | None
    interactions: list[EvidenceRecord]
    resolved_via: str | None = None

    @property
    def known(self) -> bool:
        return self.gene is not None

    @property
    def name(self) -> str | None:
        return self.gene.payload["gene"]["name"] if self.gene is not None else None


@dataclass(frozen=True)
class _Found:
    """A full gene node, the response it came in and the query that produced it."""

    node: dict[str, Any]
    resp: Response
    query: dict[str, Any]


class DgidbRetriever:
    """Drug–gene interactions by gene symbol, as evidence records. Not a variant
    :class:`~engine.retrieve.Retriever`: keyed by gene, and no genomic coordinate is
    ever sent — only HGNC symbols."""

    source = SOURCE
    columns = COLUMNS

    def __init__(self, http: Http, *, url: str = API_URL):
        self.http = http
        self.url = url
        self._service: dict[str, Any] | None = None
        _pin_rate(http, _host(url), PER_SECOND)

    # ------------------------------------------------------------------ public

    def version(self) -> str:
        """``DGIdb v.5.0.12 updatedAt=2026-07-14T18:00:15+00:00`` — the service version
        and its data timestamp; independent of what was found."""
        return _label(self.service_info())

    def service_info(self) -> dict[str, Any]:
        """The raw ``serviceInfo`` object (name, version, dataVersion, updatedAt) —
        observed once per instance. When the observation came from the cache and the
        run is online, a live observation must cite the same version and
        ``updatedAt``. :meth:`fetch` and :meth:`sources` observe it before their first
        request, so a refused cache holds nothing of the newer release."""
        if self._service is None:
            resp, info = self._post(SERVICE_QUERY, {}, _service_of)
            if resp.from_cache and not getattr(self.http, "offline", False):
                _resp, live = self._post(SERVICE_QUERY, {}, _service_of, cache_ok=False)
                if _label(live) != _label(info):
                    raise DgidbError(
                        f"the HTTP cache was filled under {_label(info)!r} but the server now reports "
                        f"{_label(live)!r}: remove the dgidb.org entries from the HTTP cache (or run offline "
                        "to reproduce the run that filled it) rather than mixing releases")
            self._service = info
        return self._service

    @property
    def params(self) -> dict[str, Any]:
        """Every setting that shapes the output, for the stage manifest's ``params``
        (JSON-ready): the endpoint, every GraphQL document, the search page size, the
        alias default, the rate the shared limiter applies to the host, the sort rule.
        The version is separate: :meth:`version`."""
        limiter = getattr(self.http, "limiter", None)
        per_second = limiter.per_second.get(_host(self.url), PER_SECOND) if limiter is not None else PER_SECOND
        return {
            "api": self.url,
            "gene_query": GENE_QUERY,
            "search_query": SEARCH_QUERY,
            "concept_query": CONCEPT_QUERY,
            "match_query": MATCH_QUERY,
            "service_query": SERVICE_QUERY,
            "sources_query": SOURCES_QUERY,
            "search_page": SEARCH_PAGE,
            "resolve_aliases_default": RESOLVE_ALIASES,
            "per_second": per_second,
            "sort": SORT_RULE,
        }

    def interactions(self, gene_symbol: str) -> list[EvidenceRecord]:
        """Interaction records for ``gene_symbol``, best-evidenced first. Empty both for a
        gene DGIdb does not know and for one it lists no drug for — :meth:`fetch` tells
        them apart."""
        return self.fetch(gene_symbol).interactions

    def fetch(self, gene_symbol: str, *, resolve_aliases: bool = RESOLVE_ALIASES) -> GeneResult:
        """Everything DGIdb reports for one symbol: the gene record and one record per
        interaction, or absence. With ``resolve_aliases`` a symbol no gene is named
        (up to case) is looked up once in ``geneMatches`` and a single DIRECT hit is
        fetched instead."""
        asked = _symbol(gene_symbol)
        version = self.version()  # first: no gene response reaches a cache that straddles a refresh
        found = self._gene(asked)
        resolved_via = None
        if found is None and resolve_aliases:
            canonical = self.resolve(asked)
            if canonical is not None:
                found = self._gene(canonical.upper())  # DGIdb's spelling (Bub1b); every lookup is by the upper-cased symbol
                if found is None:
                    raise DgidbError("geneMatches named a gene that the genes queries then did not return")
                resolved_via = "alias"
        if found is None:
            log.info("dgidb: symbol not known to DGIdb")
            return GeneResult(asked, None, [])
        node, resp, query = found.node, found.resp, found.query
        name = node["name"]  # DGIdb's canonical spelling
        records = _interaction_records(node, query, version, resp.retrieved_at)
        gene = EvidenceRecord(
            record_id=f"{GENE_SOURCE}:{name}",
            source=GENE_SOURCE,
            source_version=version,
            query=query,
            url=GENE_URL.format(name=name),
            retrieved_at=resp.retrieved_at,
            payload={"gene": {k: v for k, v in node.items() if k != "interactions"},
                     "n_interactions": len(records)},
        )
        log.info("dgidb: gene known, %d interactions%s", len(records), " (via alias)" if resolved_via else "")
        return GeneResult(asked, gene, records, resolved_via)

    def resolve(self, term: str) -> str | None:
        """The name of the gene ``term`` matches DIRECTly in DGIdb (an alias such as
        ``P53`` → ``TP53``), or — when the term is AMBIGUOUS between several genes —
        the one of them that is *named* ``term`` up to case; ``None`` when it matches
        nothing or several genes with no such tie-break."""
        term = _symbol(term)
        _resp, matches = self._post(MATCH_QUERY, {"terms": [term]}, lambda b: _matches_of(b, term))
        direct = _matched_names(matches["directMatches"], term)
        if len(direct) == 1:
            return direct[0]
        if not direct:
            named = [n for n in _matched_names(matches["ambiguousMatches"], term) if n.upper() == term]
            if len(named) == 1:
                return named[0]
        return None  # no match, or ambiguous: absence, not a guess

    def sources(self, *, page_size: int = SOURCES_PAGE) -> list[dict[str, Any]]:
        """Every DGIdb source (name, version, licence, citation, claim counts), sorted
        by name — the per-source citations and licence terms behind the interactions.
        Subject to the same release agreement as :meth:`fetch`."""
        if page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {page_size}")
        self.version()
        nodes = [n for page in self._pages(SOURCES_QUERY, {"first": page_size}, _sources_of) for n in page["nodes"]]
        return sorted(nodes, key=lambda n: str(n.get("sourceDbName", "")))

    def extract(self, record: EvidenceRecord | None) -> dict[str, str]:
        """Project one interaction record onto :data:`COLUMNS` (strings; all '' for ``None``)."""
        if record is None:
            return {c: "" for c in COLUMNS}
        p = record.payload
        if not isinstance(p, dict) or not isinstance(p.get("drug"), dict):
            raise ValueError(f"not a dgidb interaction record: {record.record_id}")
        drug = p["drug"]
        types = [t for t in p.get("interactionTypes") or [] if isinstance(t, dict)]
        return {
            "dgidb_gene": str(_get(p, "gene", "name") or ""),
            "dgidb_drug": str(drug.get("name") or ""),
            "dgidb_drug_concept_id": str(drug.get("conceptId") or ""),
            "dgidb_unnamed_compound": "Y" if is_unnamed(drug) else "N",
            "dgidb_approved": _flag(drug.get("approved")),
            "dgidb_anti_neoplastic": _flag(drug.get("antiNeoplastic")),
            "dgidb_immunotherapy": _flag(drug.get("immunotherapy")),
            "dgidb_interaction_types": ";".join(sorted({str(t["type"]) for t in types if t.get("type")})),
            "dgidb_directionality": ";".join(sorted({str(t["directionality"]) for t in types if t.get("directionality")})),
            "dgidb_evidence_score": _num(p.get("evidenceScore")),
            "dgidb_interaction_score": _num(p.get("interactionScore")),
            "dgidb_sources": ";".join(f"{s.get('sourceDbName')}:{s.get('sourceDbVersion') or ''}"
                                      for s in _sorted_sources(p)),
            "dgidb_pmids": ";".join(str(n) for n in sorted({int(x["pmid"]) for x in p.get("publications") or []
                                                            if isinstance(x, dict) and x.get("pmid") is not None})),
            "dgidb_interaction_id": str(p.get("id") or ""),
        }

    # ------------------------------------------------------------------ internals

    def _gene(self, symbol: str) -> _Found | None:
        """The full node of the gene named ``symbol`` up to case — by ``names:`` first
        (exact on the upper-cased spelling, one request), else by the prefix search and
        the concept id it names — or ``None`` when no gene is so named."""
        variables: dict[str, Any] = {"names": [symbol]}
        resp, node = self._post(GENE_QUERY, variables, lambda b: _node_of(b, symbol))
        if node is not None:
            return _Found(node, resp, self._query(GENE_QUERY, variables))
        hit = self._search(symbol)
        if hit is None:
            return None
        variables = {"conceptIds": [hit["conceptId"]]}
        resp, node = self._post(CONCEPT_QUERY, variables, lambda b: _node_of(b, symbol, hit["conceptId"]))
        if node is None:
            raise DgidbError("the gene search named a concept id that genes(conceptIds:) then did not return")
        return _Found(node, resp, self._query(CONCEPT_QUERY, variables))

    def _search(self, symbol: str, *, page_size: int = SEARCH_PAGE) -> dict[str, str] | None:
        """``{name, conceptId}`` of the gene named ``symbol`` up to case, from the
        prefix search ``genes(name:)`` — every page is walked, so two genes differing
        only by case would be noticed rather than one chosen — or ``None``."""
        hits = [n for page in self._pages(SEARCH_QUERY, {"name": symbol, "first": page_size}, _search_page_of)
                for n in page["nodes"] if n["name"].upper() == symbol]
        if len(hits) > 1:
            raise DgidbError(f"{len(hits)} genes are named {symbol} up to case")
        return hits[0] if hits else None

    def _pages(self, query: str, variables: dict[str, Any], parse: Callable[[dict[str, Any]], dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Every page of a connection, ``variables`` plus the ``after`` cursor, until
        ``hasNextPage`` is false."""
        after: str | None = None
        while True:
            _resp, page = self._post(query, {**variables, "after": after}, parse)
            yield page
            info = page["pageInfo"]
            if not info.get("hasNextPage"):
                return
            if not info.get("endCursor") or info["endCursor"] == after:
                raise DgidbError("hasNextPage without a new endCursor")
            after = info["endCursor"]

    def _query(self, graphql: str, variables: dict[str, Any]) -> dict[str, Any]:
        return {"api": self.url, "graphql": graphql, "variables": variables}

    def _post(self, query: str, variables: dict[str, Any], parse: Callable[[dict[str, Any]], T],
              *, cache_ok: bool = True) -> tuple[Response, T]:
        """POST one GraphQL document and validate the body. A *cached* body that fails
        validation is fetched again live, once, and the live answer — if it validates —
        replaces the cached one, so a failure the server delivered with HTTP 200 (and
        the cache kept as definitive) neither replays nor costs a live request forever.
        Offline, the cached body's own error is reported as such."""
        body = {"query": query, "variables": variables}
        resp = self.http.post(self.url, body, cache_ok=cache_ok)
        try:
            return resp, parse(_json_body(resp))
        except DgidbError as e:
            if not resp.from_cache:
                raise
            if getattr(self.http, "offline", False):
                raise DgidbError(f"a cached DGIdb response failed validation and offline forbids fetching it again: {e}") from e
            log.warning("dgidb: a cached response failed validation; fetching it again live")
            resp = self.http.post(self.url, body, cache_ok=False)
            parsed = parse(_json_body(resp))
            self._heal(resp, body)
            return resp, parsed

    def _heal(self, resp: Response, body: dict[str, Any]) -> None:
        """Write a validated live answer into the HTTP cache under the key
        :class:`~engine.retrieve.http.Http` used for it — the same entry shape ``Http``
        writes, so the next run replays it (same ``retrieved_at``; works offline)."""
        cache = getattr(self.http, "cache", None)
        if cache is None or resp.from_cache:
            return
        cache.put(resp.request_key, resp, {"method": "POST", "url": self.url, "params": None, "body": body})


# ---------------------------------------------------------------------- records

def _interaction_records(node: dict[str, Any], query: dict[str, Any], version: str, retrieved_at: str) -> list[EvidenceRecord]:
    """One record per interaction object, verbatim, best-evidenced first. A drug's
    concept id is the identity (its name is not unique: one drug can be two DGIdb
    records); two interactions under one id would be an API shape this module does
    not understand."""
    name = node["name"]
    records: dict[str, EvidenceRecord] = {}
    for i in node["interactions"]:
        drug_id = _drug_id(i, name)
        rid = f"{SOURCE}:{name}:{drug_id}"
        if rid in records:
            raise DgidbError(f"two interactions for {rid}")
        records[rid] = EvidenceRecord(
            record_id=rid,
            source=SOURCE,
            source_version=version,
            query=query,
            url=GENE_URL.format(name=name),
            retrieved_at=retrieved_at,
            payload=i,
        )
    return sorted(records.values(), key=_rank_key)


def _drug_id(i: Any, name: str) -> str:
    if not isinstance(i, dict) or not isinstance(i.get("drug"), dict):
        raise DgidbError(f"an interaction of {name} has no drug object")
    echoed = _get(i, "gene", "name")
    if echoed is not None and str(echoed).upper() != name.upper():
        raise DgidbError(f"an interaction listed under {name} names another gene")
    drug = i["drug"]
    drug_id = drug.get("conceptId") or drug.get("name")
    if not isinstance(drug_id, str) or not drug_id:
        raise DgidbError(f"an interaction of {name} has a drug with neither conceptId nor name")
    return drug_id


def _rank_key(rec: EvidenceRecord) -> tuple[float, float, str, str]:
    """:data:`SORT_RULE` — best evidence first, then DGIdb's own score, then name and id
    so the order is a pure function of the payload (the server's order is not promised)."""
    p = rec.payload
    return (-_float(p.get("evidenceScore")), -_float(p.get("interactionScore")),
            str(p["drug"].get("name") or ""), str(p["drug"].get("conceptId") or ""))


def is_unnamed(drug: dict[str, Any]) -> bool:
    """True when DGIdb has no name for the compound and shows its identifier instead
    (``CHEMBL:CHEMBL260560`` for ``chembl:CHEMBL260560``)."""
    name, cid = drug.get("name"), drug.get("conceptId")
    return isinstance(name, str) and isinstance(cid, str) and name.upper() == cid.upper()


# ---------------------------------------------------------------------- response shape

def _json_body(resp: Response) -> dict[str, Any]:
    """The GraphQL envelope, or :class:`DgidbError`: a non-JSON body, ``errors[]``
    (which DGIdb delivers with HTTP 200), or a body without a ``data`` object."""
    if resp.status != 200:
        raise DgidbError(f"HTTP {resp.status} from DGIdb: {resp.text[:200]!r}")
    try:
        body = resp.json()
    except ValueError as e:
        raise DgidbError(f"non-JSON body from DGIdb: {resp.text[:200]!r}") from e
    if not isinstance(body, dict):
        raise DgidbError(f"unexpected JSON from DGIdb: {resp.text[:200]!r}")
    errors = body.get("errors")
    if errors:
        messages = [e.get("message") if isinstance(e, dict) else repr(e) for e in errors]
        raise DgidbError(f"DGIdb errors: {messages}")
    if not isinstance(body.get("data"), dict):
        raise DgidbError(f"DGIdb returned no data object: {resp.text[:200]!r}")
    return body


def _node_of(body: dict[str, Any], symbol: str, concept_id: str | None = None) -> dict[str, Any] | None:
    """The one gene node named ``symbol`` up to case (and carrying ``concept_id`` when
    one was asked for), ``None`` for ``nodes: []``, :class:`DgidbError` for any other
    shape."""
    nodes = _get(body, "data", "genes", "nodes")
    if not isinstance(nodes, list):
        raise DgidbError("genes query returned no nodes list")
    if not nodes:
        return None
    if len(nodes) != 1:
        raise DgidbError(f"{len(nodes)} gene nodes for one gene")
    node = nodes[0]
    if not isinstance(node, dict) or not isinstance(node.get("name"), str):
        raise DgidbError("gene node without a name")
    if node["name"].upper() != symbol.upper():
        raise DgidbError("gene node did not echo the symbol asked")
    if concept_id is not None and node.get("conceptId") != concept_id:
        raise DgidbError("gene node did not echo the concept id asked")
    if not isinstance(node.get("interactions"), list):
        raise DgidbError("gene node without an interactions list")
    return node


def _search_page_of(body: dict[str, Any]) -> dict[str, Any]:
    """One page of the gene search: ``nodes`` of ``{name, conceptId}`` and ``pageInfo``."""
    page = _get(body, "data", "genes")
    if not isinstance(page, dict) or not isinstance(page.get("nodes"), list) or not isinstance(page.get("pageInfo"), dict):
        raise DgidbError("gene search returned no connection")
    for n in page["nodes"]:
        if not isinstance(n, dict) or not isinstance(n.get("name"), str) or not isinstance(n.get("conceptId"), str) or not n["conceptId"]:
            raise DgidbError("gene search node without a name and conceptId")
    return page


def _matches_of(body: dict[str, Any], term: str) -> dict[str, Any]:
    m = _get(body, "data", "geneMatches")
    if not isinstance(m, dict):
        raise DgidbError("geneMatches query returned no result object")
    for key in ("directMatches", "ambiguousMatches", "noMatches"):
        if not isinstance(m.get(key), list):
            raise DgidbError(f"geneMatches without a {key} list")
    return m


def _matched_names(matches: list[Any], term: str) -> list[str]:
    """The distinct gene names under the entries that echo ``term`` (case-insensitively), sorted."""
    mine = [m for m in matches if isinstance(m, dict) and str(m.get("searchTerm", "")).upper() == term]
    return sorted({str(g["name"]) for m in mine for g in m.get("matches", []) if isinstance(g, dict) and g.get("name")})


def _service_of(body: dict[str, Any]) -> dict[str, Any]:
    info = _get(body, "data", "serviceInfo")
    if not isinstance(info, dict) or not info.get("version") or not info.get("updatedAt"):
        raise DgidbError("serviceInfo without a version and updatedAt to cite")
    return info


def _label(info: dict[str, Any]) -> str:
    """``DGIdb v.5.0.12 updatedAt=2026-07-14T18:00:15+00:00`` — never ``dataVersion``."""
    return f"DGIdb {info['version']} updatedAt={info['updatedAt']}"


def _sources_of(body: dict[str, Any]) -> dict[str, Any]:
    page = _get(body, "data", "sources")
    if not isinstance(page, dict) or not isinstance(page.get("nodes"), list) or not isinstance(page.get("pageInfo"), dict):
        raise DgidbError("sources query returned no connection")
    return page


# ---------------------------------------------------------------------- helpers

def _symbol(s: str) -> str:
    """The symbol as sent: stripped and upper-cased (DGIdb compares ``names:`` after
    upper-casing and answers with its own spelling, observed live with ``cftr`` →
    ``CFTR``). A blank or malformed symbol is refused before any request — ``genes(names:
    [""])`` answers ``nodes: []``, a fabricated absence. The rejected text is not
    echoed: it could be anything a caller passed by mistake."""
    sym = str(s).strip().upper()
    if not _SYMBOL.match(sym):
        raise ValueError(f"not a gene symbol ({len(sym)} chars)")
    return sym


def _sorted_sources(p: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted((s for s in p.get("sources") or [] if isinstance(s, dict)),
                  key=lambda s: (str(s.get("sourceDbName") or ""), str(s.get("sourceDbVersion") or "")))


def _flag(x: Any) -> str:
    return "" if x is None else ("Y" if x else "N")


def _num(x: Any) -> str:
    return "" if x is None else repr(x)


def _float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("-inf")


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
