"""The three tools the reasoning agent may call, and what they leave behind.

``get_record`` reads; ``search_literature`` and ``get_paper`` retrieve — and every paper
either returns becomes a ``pmid:<n>`` record in ``05_reason/evidence/`` before the model
sees it, so the citation vocabulary grows only through the store. A search also
leaves its own ``pmid-search:<sha>`` record (the query as sent, the hit count, the
PMIDs in order): the reason a paper was on the table is as auditable as the paper.

What the model may read is what it may cite. ``get_record`` serves the candidate's
own records (the bundle's citable list) and the papers the literature tools returned
in this conversation — not another candidate's variant records, not a paper an
earlier run fetched and the model happens to remember. A paper is read with
``get_paper``, which serves it from any stage's store when one already holds it and
fetches it otherwise; either way it is then on this candidate's list.

The log keeps one fact per record — that a tool returned it in this conversation —
and not whether the bytes came from Europe PMC or from a store, because the latter
depends on what earlier runs left behind and the transcript must be the same on a
rerun. What this run wrote to the store is kept apart (``written``) for the manifest's
counts only.

Two kinds of refusal, kept apart as :mod:`engine.agents.client` expects. A bad
argument — an unknown record id, a PMID Europe PMC does not know, a malformed query, a
query carrying a genomic coordinate — raises ``KeyError``/``ValueError`` and goes back
to the model as an error result it can route around. A failure of the service behind
the tool (:class:`~engine.retrieve.literature.LiteratureError`, HTTP errors, an offline
cache miss) propagates and aborts the run, because a chain written around a silently
missing paper is exactly what this engine refuses to produce.

Privacy: a search query never carries a genomic coordinate. Papers are found by gene
symbol, protein or cDNA change, disease or phenotype — the way a geneticist searches —
and a query that spells a position, in any form, is refused before any request is
made. On GRCh38 a bare run of five or more digits is a position until an identifier
prefix (``rs``, ``PMID``, ``VCV``, ``NCT``, ``HP:``, ``c.``, …) claims it, so such a
run is refused too.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from engine.agents.client import ToolSpec
from engine.agents.validator import EvidenceIndex
from engine.retrieve.literature import LiteratureRetriever, abstract_text
from engine.retrieve.store import EvidenceRecord, EvidenceStore

DEFAULT_MAX_RESULTS = 10
MAX_RESULTS_CAP = 25
"""One Europe PMC page; a reasoning agent needs the best few papers, not a corpus."""
ABSTRACT_CHARS = 6000
"""Abstracts are a few thousand characters at most; the cap only guards the prompt."""

COORDINATE_RULE = ("refused: chrom:pos, chrom-pos and g.<pos> forms, and any bare run of 5+ digits (thousands "
                   "separators removed) that no identifier prefix claims (rs, PMID, PMC, NCT, VCV/SCV/RCV, HP:, "
                   "OMIM, MONDO, Orphanet, EXT_ID, DOI, c./p./m./n./r., an accession such as NM_000492)")
"""The rule as the manifest records it (``params.literature.coordinates_in_queries``)."""

_COORDINATE = re.compile(
    r"(?<![A-Za-z0-9_])(?:chr)?(?:(?:[1-9]|1[0-9]|2[0-2]|X|Y)[:\-]\s?\d{5,}|MT?[:\-]\s?\d{3,})(?![A-Za-z0-9])"  # 7:117559590, chr7-117559590, MT:8993
    r"|\bg\.\d{4,}",                                                                                          # NC_…:g.117559590del
    re.IGNORECASE,
)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
"""``117,559,590`` is the same number as ``117559590``."""
_ID_PREFIX = re.compile(r"\b(?:OMIM|MIM|PMID|PMCID|PMC|NCT|EXT_ID|HP|MONDO|ORPHA|Orphanet|rs|VCV|SCV|RCV|DOI)\s*:?\s*(?=\d)",
                        re.IGNORECASE)
"""Prefixes that claim the digits after them (``OMIM 219700``, ``PMID: 7647779``)."""
_BARE_POSITION = re.compile(r"(?<![A-Za-z0-9_.:\-/])\d{5,}(?![A-Za-z0-9])")
"""Five or more digits with nothing claiming them: no letter (``rs1801133``,
``NM_000492``), no ``.`` (``c.1521``), no ``_`` (``c.1521_1523del``), no ``:`` (``HP:0002205``)."""
_PMID = re.compile(r"^\s*(?:pmid:|PMID\s*:?\s*)?(\d{1,9})\s*$", re.IGNORECASE)


@dataclass
class ToolLog:
    """What the tools returned during one candidate's run — the ids the chain may cite
    beyond the bundle's own, and what the manifest reports."""
    searches: list[str] = field(default_factory=list)
    """``pmid-search:`` record ids, in call order."""
    papers: list[str] = field(default_factory=list)
    """``pmid:`` record ids ``search_literature`` or ``get_paper`` returned, first
    appearance only — wherever the record was served from."""
    records_read: list[str] = field(default_factory=list)
    """Ids ``get_record`` returned, first appearance only."""
    written: list[str] = field(default_factory=list)
    """Records this run added to the stage store (absent before the call). Not in
    :meth:`as_dict`: whether a record was new depends on what earlier runs left in
    the store, and the transcript must not."""

    def ids(self) -> list[str]:
        """Every record id a tool returned, in first-appearance order."""
        return list(dict.fromkeys([*self.searches, *self.papers, *self.records_read]))

    def as_dict(self) -> dict[str, Any]:
        return {"searches": list(self.searches), "papers": list(self.papers), "records_read": list(self.records_read)}


class ReasonTools:
    """Handlers bound to one run's evidence index, the stage store that receives new
    records, a literature retriever (``None`` in a dry run, where no tool is ever
    called) and the candidate's citable list (``None``: any record the index holds —
    for tests and ad-hoc use; the stage always passes the bundle's)."""

    def __init__(self, index: EvidenceIndex, store: EvidenceStore, literature: LiteratureRetriever | None, *,
                 citable: Iterable[str] | None = None,
                 default_max_results: int = DEFAULT_MAX_RESULTS, max_results_cap: int = MAX_RESULTS_CAP):
        self.index = index
        self.store = store
        self.literature = literature
        self.citable = None if citable is None else frozenset(citable)
        self.default_max_results = default_max_results
        self.max_results_cap = max_results_cap
        self.log = ToolLog()

    def citable_ids(self) -> list[str]:
        """The candidate's list plus everything the tools returned so far, sorted."""
        return sorted(set(self.citable or ()) | set(self.log.ids()))

    # ---- handlers

    def get_record(self, inputs: dict[str, Any]) -> dict[str, Any]:
        rid = str(inputs.get("record_id", "")).strip()
        if self.citable is not None and rid not in self.citable and rid not in self.log.ids():
            raise KeyError(f"no record {rid!r} in this candidate's bundle or among this conversation's tool results; "
                           "cite only the bundle's listed ids and papers returned by search_literature or get_paper")
        rec = self.index.get(rid)
        if rec is None:
            raise KeyError(f"no record {rid!r} in the evidence store; cite only ids listed in the bundle or returned by a tool")
        if rid not in self.log.records_read:
            self.log.records_read.append(rid)
        return asdict(rec)

    def search_literature(self, inputs: dict[str, Any]) -> dict[str, Any]:
        query = str(inputs.get("query", "")).strip()
        if not query:
            raise ValueError("query must not be empty")
        check_query(query)
        n = inputs.get("max_results")
        n = self.default_max_results if n is None else int(n)
        if not 1 <= n <= self.max_results_cap:
            raise ValueError(f"max_results must be between 1 and {self.max_results_cap}, got {n}")
        lit = self._literature()
        result = lit.search(query, max_results=n)
        self._put(result.record)
        self.log.searches.append(result.record.record_id)
        papers = [self._keep(p) for p in result.papers]
        return {
            "query": query,
            "sent": result.record.payload["sent"]["query"],
            "hit_count": result.record.payload["hitCount"],
            "returned": len(papers),
            "search_record": result.record.record_id,
            "papers": [self._summary(p, lit) for p in papers],
            "note": ("Cite a paper by its record_id. Read an abstract with get_paper before relying on a "
                     "paper for a specific claim; a title is not evidence of a result."),
        }

    def get_paper(self, inputs: dict[str, Any]) -> dict[str, Any]:
        pmid = parse_pmid(inputs.get("pmid"))
        rid = f"pmid:{pmid}"
        rec = self.index.get(rid)  # any stage's store: a paper is the same record wherever it was fetched
        if rec is None:
            rec = self._literature().fetch(pmid)
            if rec is None:
                raise KeyError(f"no PubMed record for PMID {pmid}; it does not exist in Europe PMC and cannot be cited")
        self._keep(rec)
        lit = self.literature or LiteratureRetriever(None)  # extract() is a pure projection
        out = self._summary(rec, lit)
        out["abstract"] = abstract_text(rec)[:ABSTRACT_CHARS] or "(no abstract in Europe PMC)"
        out["doi"] = lit.extract(rec)["doi"]
        return out

    # ---- specs

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                "get_record",
                "Return one of this candidate's evidence records by its id (e.g. gnomad:7-117559590-ATCT-A, "
                "clinvar:VCV000007105, vep:7:117559590:ATCT:A, exomiser:CFTR): source, version, the query that "
                "produced it, its URL and the full raw payload. Use it to quote a number exactly (subpopulation "
                "frequencies, homozygote counts, ClinVar review status, transcript details). Serves only the ids "
                "listed in the bundle and papers already returned by search_literature or get_paper; errors for "
                "any other id.",
                {"type": "object", "properties": {"record_id": {"type": "string", "description": "A record id exactly as listed in the bundle or returned by a tool."}}},
                self.get_record,
            ),
            ToolSpec(
                "search_literature",
                "Search Europe PMC for PubMed-indexed papers. Query in Europe PMC syntax; the reliable field "
                "is TITLE_ABS, e.g. TITLE_ABS:\"CFTR\" AND TITLE_ABS:\"F508del\" or TITLE_ABS:\"TP53\" AND "
                "TITLE_ABS:\"Li-Fraumeni syndrome\". Use gene symbols, protein or cDNA changes, disease "
                "or phenotype terms — never a genomic coordinate or a bare number of five or more digits (such a "
                "query is refused). Returns paper records pmid:<n> with title, authors, journal, year and citation "
                "count; each returned paper becomes citable. Read an abstract with get_paper before relying on a paper.",
                {"type": "object", "properties": {
                    "query": {"type": "string", "description": "Europe PMC query, e.g. TITLE_ABS:\"CFTR\" AND TITLE_ABS:\"F508del\"."},
                    "max_results": {"anyOf": [{"type": "integer"}, {"type": "null"}],
                                    "description": f"Papers to return, 1–{MAX_RESULTS_CAP}; null for the default of {DEFAULT_MAX_RESULTS}."},
                }},
                self.search_literature,
            ),
            ToolSpec(
                "get_paper",
                "Return one paper's metadata and abstract by PMID (a number, e.g. 7647779), and make it "
                "citable as pmid:<n>. Errors when Europe PMC has no PubMed record for that PMID — such a "
                "PMID cannot be cited.",
                {"type": "object", "properties": {"pmid": {"type": "string", "description": "The PubMed id, digits only."}}},
                self.get_paper,
            ),
        ]

    def handlers(self) -> dict[str, Any]:
        return {t.name: t.handler for t in self.specs()}

    # ---- helpers

    def _literature(self) -> LiteratureRetriever:
        if self.literature is None:
            raise RuntimeError("no literature service is configured (dry run)")
        return self.literature

    def _put(self, rec: EvidenceRecord) -> None:
        """Write a record into the stage store; the index sees the file at once."""
        if not self.store.exists(rec.record_id):
            self.log.written.append(rec.record_id)
        self.store.put(rec)

    def _keep(self, rec: EvidenceRecord) -> EvidenceRecord:
        """A paper the model was shown: on the candidate's list from now on, and in the
        stage store — unless a store already holds the record, in which case that
        record is the one shown and cited, so the model reads what a judge opens."""
        held = self.index.get(rec.record_id)
        if held is None:
            self._put(rec)
        if rec.record_id not in self.log.papers:
            self.log.papers.append(rec.record_id)
        return held or rec

    @staticmethod
    def _summary(rec: EvidenceRecord, lit: LiteratureRetriever) -> dict[str, Any]:
        cols = lit.extract(rec)
        return {
            "record_id": rec.record_id,
            "title": cols["title"],
            "authors": cols["authors"],
            "journal": cols["journal"],
            "year": cols["year"],
            "cited_by": cols["cited_by"],
            "has_abstract": cols["has_abstract"],
            "open_access": cols["open_access"],
            "url": rec.url,
        }


def check_query(query: str) -> None:
    """Refuse a literature query that spells a genomic position — as ``chrom:pos``,
    ``chrom-pos``, ``g.<pos>``, or a bare run of five or more digits that no identifier
    prefix claims. Gene symbols, rsIDs, protein and cDNA changes and phenotype terms
    are how papers are found; a coordinate identifies the proband's variant to a
    third party and finds nothing anyway."""
    plain = _THOUSANDS.sub("", query)
    if _COORDINATE.search(plain):
        raise ValueError("a literature query must not contain a genomic coordinate; search by gene symbol, "
                         "protein or cDNA change (p.Phe508del, c.1521_1523del), rsID, disease or phenotype term")
    if _BARE_POSITION.search(_ID_PREFIX.sub("id", plain)):
        raise ValueError("a literature query must not contain a bare number of five or more digits (it reads as a "
                         "genomic position); search by gene symbol, protein or cDNA change, rsID, disease or "
                         "phenotype term, or spell an identifier with its prefix (rs…, PMID …, NCT…)")


def parse_pmid(value: Any) -> str:
    """``'7647779'``, ``'pmid:7647779'``, ``'PMID 7647779'`` → ``'7647779'`` (no leading zeros)."""
    m = _PMID.match(str(value if value is not None else ""))
    if not m:
        raise ValueError(f"not a PMID: {value!r} (expected digits, e.g. 7647779)")
    return str(int(m.group(1)))
