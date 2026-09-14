"""ClinicalTrials.gov — registered trials, from the REST API v2 (clinicaltrials.gov/api/v2).

Why stage 6 needs it. A drug candidate in a medicine report is only as good as what
has actually been tried: which trials exist for the condition and the compound, in
what phase, with what status, why any were stopped, and whether results were posted.
ClinicalTrials.gov is the registry of record for that, and the ``DrugCandidate``
schema (see CONTRACTS.md) cites trials by ``nct:<NCT id>`` — so every trial a report
names must exist here as a record whose payload is the registry's own study object.

Two kinds of record. A *study* record (``nct:<NCT id>``) is one registry entry, the
same record whichever route found it — :meth:`TrialsRetriever.study` or a search —
because its ``query`` is the detail request that reproduces it (the API returns the
identical object on both routes for the same ``fields`` list; verified live), not the
search that happened to list it. The one field the route decides is ``retrieved_at``:
it is the time of the response the study came in on, so a trial reached by a search
and again by :meth:`~TrialsRetriever.study` gives two records that differ only there;
an orchestrator that writes both should keep the first (the store's ``put`` is
last-writer-wins). A *search* record (``nct-search:<sha256>``) holds what was asked
(condition, intervention, free-text term, status filter, sort, page size, how many
were wanted), what the registry answered (``totalCount`` — the number of matching
trials, which is usually more than were fetched) and the NCT ids returned in order;
it is the reason a trial was on the table. :meth:`TrialsRetriever.find` returns both;
:meth:`TrialsRetriever.search` returns the study records alone.

How it asks. ``GET /studies?query.cond=…&query.intr=…&query.term=…`` with an explicit
``fields`` list (a results-bearing trial is ~73 KB in full; the modules asked for here
are ~7 KB and hold everything a report needs: identification with aliases, status and
every date, sponsor, conditions, design with phases and enrolment, arms and
interventions, eligibility, brief summary, primary outcomes, MeSH terms, and the
snapshot stamp) and an explicit ``sort`` — the server's default order is not stable
between calls, and a ``pageToken`` is *not* bound to the query it came from (a token
reused with different parameters returns HTTP 200 and the wrong rows), so every page
carries byte-identical query/filter/fields/sort parameters and only ``countTotal``
(first page) and ``pageToken`` (later pages) differ. ``pageSize`` is capped at the
server's 1000; 0 or a larger value is silently coerced, never an error. One shape of
ask is refused before any request: a query part that is nothing but NCT ids
(``NCT00909532``, ``NCT00909532 NCT02971839``, comma-separated, any case — in
``cond``, ``intr`` or ``term`` alike) switches the registry into id lookup, and it
then silently ignores ``filter.overallStatus`` *and the other query parts* (verified
live: ``cond=asthma&term=NCT00909532&filter.overallStatus=RECRUITING`` answers the
completed cystic-fibrosis trial), so a search record would claim a filter the answer
does not honour; :meth:`TrialsRetriever.study` is the lookup by id. As a second
guard every study a filtered search lists is checked against the filter, and one
outside it raises. The pieces the module itself reads — ``NCTId``, ``OverallStatus``
and ``VersionHolder`` — are always requested, whatever ``fields`` list is given.

Absence versus failure. A missing NCT id is HTTP 404 whose plain-text body names the
id (``NCT number NCT99999999 not found``) — that, and only that, is "no such study"
(``None``); a 404 of any other shape (an empty body from a wrong API path, an
edge/CDN or nginx page) is a failure and raises :class:`CtgovError`, because a 404 is
cached as an answer and a wrong endpoint must never replay as a permanent absence.
An empty search is HTTP 200 with ``totalCount: 0`` and an empty ``studies`` list; a
first page that claims matches but lists none, or claims fewer than it lists, is
inconsistent and raises. Error
bodies are text, not JSON: a malformed id, an unknown field name, a bad status value
or a bad page token are all HTTP 400 text, and :class:`~engine.retrieve.http.Http`
raises on those; anything that answers HTTP 200 but is not a study list — a non-JSON
body, no ``studies``, a study without an NCT id or without its snapshot stamp, a page
token that repeats — raises :class:`CtgovError`. Ids and status values are validated
*before any request* so a typo cannot come back as a fabricated absence. An *alias*
id (a trial that was re-registered) redirects to the canonical entry with the query
string dropped, so the answer is the full record under another id: the retriever
checks the id echoed, accepts it only when the asked id is in ``nctIdAliases``, and
then fetches the canonical entry with the normal field list, so the record is the
canonical one.

Version. The registry reloads nightly and every study object carries the snapshot
it was served from (``derivedSection.miscInfoModule.versionHolder``, a date — the
same value on every record of one snapshot, not the study's own change date, which
is ``lastUpdatePostDateStruct``). A record cites what was *observed on its own
payload*: a study record its stamp, a search record the set of stamps on the studies
it lists (``ClinicalTrials.gov API 2.0.5 data 2026-09-11``); an empty search observed
no stamp and cites the API version alone. That keeps every record byte-identical on a
warm cache whatever night it is replayed. :meth:`TrialsRetriever.version` — the
manifest's line — is the ``/version`` object the HTTP cache holds (``ClinicalTrials.gov
API 2.0.5 data 2026-09-11T09:00:04``): what the cache was filled under, so it too is
byte-identical on a warm cache, online or offline, and the same convention as the
Open Targets, DGIdb and ChEMBL retrievers. On a cold cache that first observation is
itself live and no second request is made; on a warm cache, online, the live service
is asked once more and its API version must agree with the cache's (a cache filled
under one API version is not replayed under another — the same refusal as the
siblings); a different *data* stamp is the nightly reload and is only logged, since
refusing it would fail every day, and the records say which night each came from.
The knobs that shape the output but are in no record — page size, sort, fields, the
rate — are :attr:`TrialsRetriever.params`, for the manifest beside the version.

Etiquette. No key, no rate-limit headers; the limiter is pinned to 3 requests/second
on ``clinicaltrials.gov`` unless the orchestrator already applies a rate for the host,
or a default at least as strict.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from engine.retrieve.http import Http, Response
from engine.retrieve.store import EvidenceRecord

log = logging.getLogger(__name__)

SOURCE = "nct"
SEARCH_SOURCE = "nct-search"
API_URL = "https://clinicaltrials.gov/api/v2/"
STUDY_URL = "https://clinicaltrials.gov/study/{nct_id}"
SEARCH_URL = "https://clinicaltrials.gov/search?{query}"
PER_SECOND = 3.0
MAX_PAGE_SIZE = 1000   # the server silently coerces anything larger
DEFAULT_PAGE_SIZE = 50
DEFAULT_SORT = "LastUpdatePostDate:desc,StudyFirstPostDate:desc"
"""Most recently updated first. Only date/numeric fields (or ``@relevance``) may be
sorted on, at most two keys; ``NCTId`` is not sortable (HTTP 400)."""

FIELDS = (
    "IdentificationModule",          # nctId, nctIdAliases, orgStudyIdInfo, secondaryIdInfos, titles, organization
    "StatusModule",                  # overallStatus, whyStopped, every date struct (partial dates 'YYYY-MM')
    "SponsorCollaboratorsModule",
    "ConditionsModule",              # conditions, keywords
    "DesignModule",                  # studyType, phases (absent for observational/expanded access), enrollmentInfo
    "ArmsInterventionsModule",
    "EligibilityModule",             # eligibilityCriteria, sex, minimumAge, maximumAge, stdAges
    "BriefSummary",
    "PrimaryOutcomeMeasure",
    "PrimaryOutcomeDescription",
    "PrimaryOutcomeTimeFrame",
    "ConditionMeshTerm",             # derived from conditions AND keywords: broad terms included
    "ConditionMeshId",
    "InterventionMeshTerm",
    "InterventionMeshId",
    "VersionHolder",                 # the nightly snapshot stamp — a record's data version
    "HasResults",
)
"""``fields=`` pieces: module names return the whole module, piece names one leaf.
Names are case-sensitive; an unknown one is HTTP 400."""

REQUIRED_FIELDS = (
    ("NCTId", "IdentificationModule"),     # protocolSection.identificationModule.nctId — the record id
    ("OverallStatus", "StatusModule"),     # protocolSection.statusModule.overallStatus — checked against a status filter
    ("VersionHolder", "MiscInfoModule"),   # derivedSection.miscInfoModule.versionHolder — the record's data version
)
"""The leaves this module reads on every study, as ``(piece, module that contains it)``:
a ``fields`` list that names neither gets the piece appended, so a reduced list still
yields records (a dotted path to the same leaf is not recognised and just gets the
piece as well — the API returns one leaf once, verified live)."""

STATUSES = frozenset({
    "ACTIVE_NOT_RECRUITING", "COMPLETED", "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING", "RECRUITING",
    "SUSPENDED", "TERMINATED", "WITHDRAWN", "AVAILABLE", "NO_LONGER_AVAILABLE", "TEMPORARILY_NOT_AVAILABLE",
    "APPROVED_FOR_MARKETING", "WITHHELD", "UNKNOWN",
})
"""``overallStatus`` values (the API's ``Status`` enum). ``WITHHELD`` entries carry a
placeholder title and almost nothing else."""

SITE_STATUS_CODES = {
    "NOT_YET_RECRUITING": "not", "RECRUITING": "rec", "ACTIVE_NOT_RECRUITING": "act", "COMPLETED": "com",
    "TERMINATED": "ter", "ENROLLING_BY_INVITATION": "enr", "SUSPENDED": "sus", "WITHDRAWN": "wit",
    "UNKNOWN": "unk", "AVAILABLE": "ava", "NO_LONGER_AVAILABLE": "nla", "TEMPORARILY_NOT_AVAILABLE": "tna",
    "APPROVED_FOR_MARKETING": "afm",
}
"""How the registry's own search page spells a status filter (``aggFilters=status:rec com``),
read from the site's bundle on 2026-09-13. ``WITHHELD`` has no code there: a filter that
names it cannot be put in a page URL, so the judge-openable URL is then the unfiltered ask."""

COLUMNS = (
    "nct_id",
    "aliases",                  # ';'-joined former NCT ids of a re-registered trial
    "title",
    "status",
    "why_stopped",
    "study_type",
    "phases",                   # ';'-joined; '' for observational/expanded-access (not applicable)
    "conditions",
    "interventions",            # 'TYPE: name;TYPE: name'
    "start_date",               # partial dates as given: 'YYYY-MM' or 'YYYY-MM-DD'
    "primary_completion_date",
    "completion_date",
    "enrollment",
    "enrollment_type",          # ACTUAL / ESTIMATED
    "sponsor",
    "sponsor_class",            # NIH, FED, OTHER_GOV, INDUSTRY, INDIV, NETWORK, AMBIG, OTHER, UNKNOWN
    "primary_outcomes",
    "sex",
    "min_age",
    "max_age",
    "has_results",
    "results_first_post_date",
    "last_update_post_date",
    "condition_mesh",
    "intervention_mesh",
)

_NCT = re.compile(r"^NCT(?!00000000$)[0-9]{8}$")   # the API's own pattern is NCT0*[1-9]\d{0,7}: all zeros is not an id
_ONLY_NCT_IDS = re.compile(r"^[\s,]*NCT[0-9]{1,8}(?:[\s,]+NCT[0-9]{1,8})*[\s,]*$", re.IGNORECASE)
"""A query part the registry takes as an id lookup rather than text (see the module
docstring); ``NCT00909532 OR NCT02971839`` and ``ivacaftor NCT00909532`` are text."""
_FIELD = re.compile(r"^[A-Za-z][A-Za-z0-9.]*$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class CtgovError(RuntimeError):
    """The API answered, but not with something this module can trust — distinct
    from "no such study" (``None``) and "no matching trials" (an empty list)."""


@dataclass(frozen=True)
class SearchResult:
    """One search: its own record (what was asked, the registry's total, the NCT ids
    in order) and one study record per trial returned, in the sort order sent."""

    record: EvidenceRecord
    studies: list[EvidenceRecord]

    @property
    def nct_ids(self) -> list[str]:
        return list(self.record.payload["nctIds"])

    @property
    def total_count(self) -> int:
        """How many trials matched in the registry — usually more than were fetched."""
        return int(self.record.payload["totalCount"])


class TrialsRetriever:
    """Registered trials by condition/intervention or by NCT id, as evidence records.
    Not a variant :class:`~engine.retrieve.Retriever`: nothing genomic is ever sent —
    condition names, intervention names, gene symbols and NCT ids only."""

    source = SOURCE
    columns = COLUMNS

    def __init__(self, http: Http, *, base_url: str = API_URL, page_size: int = DEFAULT_PAGE_SIZE,
                 fields: Iterable[str] = FIELDS, sort: str = DEFAULT_SORT):
        if not 1 <= page_size <= MAX_PAGE_SIZE:
            raise ValueError(f"page_size must be 1..{MAX_PAGE_SIZE} (server cap), got {page_size}")
        self.http = http
        self.base_url = base_url.rstrip("/") + "/"
        self.page_size = page_size
        given = list(fields)
        for f in given:
            if not _FIELD.match(f):
                raise ValueError(f"not a ClinicalTrials.gov field name: {f!r}")
        if not given:
            raise ValueError("fields must name at least one module or piece")
        # What the module itself reads on every study must be in the answer, whatever was asked for.
        given += [piece for piece, module in REQUIRED_FIELDS if piece not in given and module not in given]
        self.fields = tuple(given)
        self.sort = sort
        self._info: dict[str, str] | None = None
        self._warned_snapshot = False
        _pin_rate(http, _host(self.base_url), PER_SECOND)

    # ------------------------------------------------------------------ version

    def version(self) -> str:
        """``ClinicalTrials.gov API 2.0.5 data 2026-09-11T09:00:04`` — the ``/version``
        object (API version and nightly data load) the HTTP cache holds, i.e. what the
        cache was filled under: the manifest's line, byte-identical on a warm cache.
        Records cite the snapshot stamps on their own payloads instead; see the module
        docstring."""
        info = self.version_info()
        return f"ClinicalTrials.gov API {info['apiVersion']} data {info['dataTimestamp']}"

    def version_info(self) -> dict[str, str]:
        """The raw ``/version`` object: ``apiVersion`` and ``dataTimestamp``, both required;
        observed once per instance. The cached observation is what is cited. When it was
        not itself live (a warm cache, online) the live service is asked once more and
        must report the same API version — a cache filled under one API version is not
        replayed under another; a different data stamp is the nightly reload and is
        only logged, since every record names the night it came from."""
        if self._info is None:
            resp, cached = self._observe_version(cache_ok=True)
            if resp.from_cache and not getattr(self.http, "offline", False):
                _resp, live = self._observe_version(cache_ok=False)
                if live["apiVersion"] != cached["apiVersion"]:
                    raise CtgovError(
                        f"the HTTP cache was filled under ClinicalTrials.gov API {cached['apiVersion']} but the "
                        f"server now reports {live['apiVersion']}: remove the clinicaltrials.gov entries from the "
                        "HTTP cache (or run offline to reproduce the run that filled it) rather than mixing API versions")
                if live["dataTimestamp"] != cached["dataTimestamp"]:
                    log.info("ctgov: the cache's /version observation is data %s; the registry now reports data %s "
                             "(nightly reload; each record cites its own snapshot stamp)",
                             cached["dataTimestamp"], live["dataTimestamp"])
            self._info = cached
        return self._info

    def _observe_version(self, *, cache_ok: bool) -> tuple[Response, dict[str, str]]:
        resp = self.http.get(self.base_url + "version", cache_ok=cache_ok)
        body = self._json(resp)
        api, stamp = body.get("apiVersion"), body.get("dataTimestamp")
        if not (isinstance(api, str) and api and isinstance(stamp, str) and stamp):
            raise CtgovError(f"/version names no apiVersion/dataTimestamp to cite: {resp.text[:200]!r}")
        return resp, {"apiVersion": api, "dataTimestamp": stamp}

    @property
    def params(self) -> dict[str, Any]:
        """Every setting that shapes the output and is in no record, JSON-ready, for the
        stage manifest's ``params`` beside :meth:`version`: the API, the fields asked
        for on every study, the sort, the page size, the effective rate."""
        limiter = getattr(self.http, "limiter", None)
        # the rate in force: the per-host entry, else the limiter's default (already at least as strict; see _pin_rate)
        per_second = limiter.per_second.get(_host(self.base_url), float(limiter.default)) if limiter is not None else PER_SECOND
        return {
            "api_url": self.base_url,
            "fields": list(self.fields),
            "max_page_size": MAX_PAGE_SIZE,
            "page_size": self.page_size,
            "per_second": per_second,
            "sort": self.sort,
        }

    # ------------------------------------------------------------------ search

    def search(self, condition: str | None, intervention: str | None = None, max_results: int = 50, *,
               term: str | None = None, statuses: Iterable[str] | None = None) -> list[EvidenceRecord]:
        """Study records (``nct:<id>``) matching ``condition`` and, when given,
        ``intervention`` (drug/device name) and ``term`` (free text — the place for a
        gene symbol, which has no field of its own), optionally restricted to
        ``statuses``; at most ``max_results``, most recently updated first. ``[]`` when
        nothing matches. :meth:`find` also returns the search itself as a record."""
        return self.find(condition, intervention, max_results, term=term, statuses=statuses).studies

    def find(self, condition: str | None, intervention: str | None = None, max_results: int = 50, *,
             term: str | None = None, statuses: Iterable[str] | None = None) -> SearchResult:
        """:meth:`search` plus the search record: the parameters as sent, the
        registry's ``totalCount``, the NCT ids returned in order and the snapshot
        stamps they were served from. The record's identity is the ask — query, filter,
        fields, sort and ``max_results`` — not the page size, which is transport: the
        same ask at another page size is the same record (``payload.sent.pageSize``
        and ``payload.pages`` say how it was fetched)."""
        if max_results < 1:
            raise ValueError(f"max_results must be >= 1, got {max_results}")
        asked = {"condition": _text(condition), "intervention": _text(intervention), "term": _text(term),
                 "statuses": _statuses(statuses)}
        if not (asked["condition"] or asked["intervention"] or asked["term"]):
            raise ValueError("a search needs a condition, an intervention or a term")
        self.version_info()  # first: a refused cache must hold no page fetched under the newer API version
        sent = self._search_params(asked, page_size=min(self.page_size, max_results))
        studies: dict[str, EvidenceRecord] = {}  # keyed by NCT id: a shifting index can repeat one across pages
        head: tuple[dict[str, Any], Response] | None = None
        pages = 0
        for body, resp in self._pages(sent):
            head = head or (body, resp)
            pages += 1
            for study in body["studies"]:
                rec = self._study_record(study, resp)
                studies.setdefault(rec.record_id, rec)
            if len(studies) >= max_results:
                break
        if head is None:
            raise CtgovError("studies search yielded no page")
        total = head[0].get("totalCount")
        if not isinstance(total, int) or isinstance(total, bool):
            raise CtgovError(f"first page carries no totalCount: {head[1].text[:200]!r}")
        if total > 0 and not head[0]["studies"]:
            raise CtgovError(f"studies search reports totalCount {total} but its first page lists no study")
        if total < len(head[0]["studies"]):
            raise CtgovError(f"studies search reports totalCount {total} but its first page lists {len(head[0]['studies'])} studies")
        found = list(studies.values())[:max_results]
        if asked["statuses"]:
            _check_statuses(found, asked["statuses"])
        holders = sorted({_holder_of(r.payload) for r in found})
        log.info("ctgov: %d of %d matching trials fetched over %d page(s)", len(found), total, pages)
        ask = {k: v for k, v in sent.items() if k != "pageSize"}
        record = EvidenceRecord(
            # identity: what was asked plus how much was wanted — a different ask is a different record;
            # the page size is how it was fetched, not what was asked
            record_id=f"{SEARCH_SOURCE}:{_digest({'ask': ask, 'max_results': max_results})}",
            source=SEARCH_SOURCE,
            source_version=self._data_version(holders),
            query={**asked, "max_results": max_results, "api": self.base_url + "studies"},
            url=_search_url(asked),
            retrieved_at=head[1].retrieved_at,
            payload={"sent": sent, "totalCount": total, "nctIds": [_nct_of(r.payload) for r in found],
                     "pages": pages, "versionHolders": holders},
        )
        return SearchResult(record, found)

    # ------------------------------------------------------------------ one study

    def study(self, nct_id: str) -> EvidenceRecord | None:
        """The registry entry for ``nct_id``, or ``None`` when there is no such study.
        An alias id yields the canonical entry (its ``record_id`` is the canonical id)."""
        asked = valid_nct_id(nct_id)
        self.version_info()  # first: a refused cache must hold no study fetched under the newer API version
        resp = self.http.get(self._study_api(asked), params={"fields": self._fields_param()}, cache_404=True)
        if resp.status == 404:
            if not _names_missing(resp, asked):
                # Cached like any 404, so a wrong endpoint must raise on every replay, never read as absence.
                raise CtgovError(f"studies/{asked}: HTTP 404 that is not the registry's 'not found' (a wrong API "
                                 f"path, or an edge/CDN page); body {resp.text[:120]!r}")
            log.info("ctgov: no such study")
            return None
        body = self._json(resp)
        echoed = _nct_of(body)
        if echoed != asked:
            aliases = _get(body, "protocolSection", "identificationModule", "nctIdAliases") or []
            if asked not in aliases:
                raise CtgovError(f"studies/{asked} answered for {echoed}, which does not list it as an alias")
            # The redirect dropped ``fields``; fetch the canonical entry the normal way so the
            # record is the same one a search would produce.
            log.info("ctgov: %s is an alias of %s; fetching the canonical entry", asked, echoed)
            resp = self.http.get(self._study_api(echoed), params={"fields": self._fields_param()}, cache_404=True)
            if resp.status == 404:
                raise CtgovError(f"studies/{asked} redirected to {echoed}, which is then not found")
            body = self._json(resp)
            if _nct_of(body) != echoed:
                raise CtgovError(f"studies/{echoed} answered for {_nct_of(body)}")
        return self._study_record(body, resp)

    # ------------------------------------------------------------------ projection

    def extract(self, record: EvidenceRecord | None) -> dict[str, str]:
        """Project one study record onto :data:`COLUMNS` (strings; all '' for ``None``).
        Absent keys are '' — the fields-limited payload cannot distinguish "not
        applicable" (phases on an observational study) from "not given"."""
        if record is None:
            return {c: "" for c in COLUMNS}
        p = record.payload
        if record.source != SOURCE or not isinstance(p, dict) or not isinstance(p.get("protocolSection"), dict):
            raise ValueError(f"not a ClinicalTrials.gov study record: {record.record_id}")
        proto = p["protocolSection"]
        ident, status = proto.get("identificationModule") or {}, proto.get("statusModule") or {}
        design, elig = proto.get("designModule") or {}, proto.get("eligibilityModule") or {}
        sponsor = _get(proto, "sponsorCollaboratorsModule", "leadSponsor") or {}
        enrollment = design.get("enrollmentInfo") or {}
        interventions = [i for i in _get(proto, "armsInterventionsModule", "interventions") or [] if isinstance(i, dict)]
        outcomes = [o for o in _get(proto, "outcomesModule", "primaryOutcomes") or [] if isinstance(o, dict)]
        return {
            "nct_id": _s(ident.get("nctId")),
            "aliases": ";".join(_s(a) for a in ident.get("nctIdAliases") or []),
            "title": _s(ident.get("briefTitle")),
            "status": _s(status.get("overallStatus")),
            "why_stopped": _s(status.get("whyStopped")),
            "study_type": _s(design.get("studyType")),
            "phases": ";".join(_s(x) for x in design.get("phases") or []),
            "conditions": ";".join(_s(c) for c in _get(proto, "conditionsModule", "conditions") or []),
            "interventions": ";".join(f"{_s(i.get('type'))}: {_s(i.get('name'))}" for i in interventions),
            "start_date": _s(_get(status, "startDateStruct", "date")),
            "primary_completion_date": _s(_get(status, "primaryCompletionDateStruct", "date")),
            "completion_date": _s(_get(status, "completionDateStruct", "date")),
            "enrollment": _s(enrollment.get("count")),
            "enrollment_type": _s(enrollment.get("type")),
            "sponsor": _s(sponsor.get("name")),
            "sponsor_class": _s(sponsor.get("class")),
            "primary_outcomes": ";".join(_s(o.get("measure")) for o in outcomes),
            "sex": _s(elig.get("sex")),
            "min_age": _s(elig.get("minimumAge")),
            "max_age": _s(elig.get("maximumAge")),
            "has_results": _flag(p.get("hasResults")),
            "results_first_post_date": _s(_get(status, "resultsFirstPostDateStruct", "date")),
            "last_update_post_date": _s(_get(status, "lastUpdatePostDateStruct", "date")),
            "condition_mesh": ";".join(_mesh_terms(p, "conditionBrowseModule")),
            "intervention_mesh": ";".join(_mesh_terms(p, "interventionBrowseModule")),
        }

    # ------------------------------------------------------------------ internals

    def _fields_param(self) -> str:
        return ",".join(self.fields)

    def _study_api(self, nct_id: str) -> str:
        return f"{self.base_url}studies/{nct_id}"

    def _search_params(self, asked: dict[str, Any], *, page_size: int) -> dict[str, Any]:
        """The parameters every page carries, byte-identical from page to page."""
        params: dict[str, Any] = {}
        if asked["condition"]:
            params["query.cond"] = asked["condition"]
        if asked["intervention"]:
            params["query.intr"] = asked["intervention"]
        if asked["term"]:
            params["query.term"] = asked["term"]
        if asked["statuses"]:
            params["filter.overallStatus"] = ",".join(asked["statuses"])
        params["fields"] = self._fields_param()
        params["sort"] = self.sort
        params["pageSize"] = page_size
        return params

    def _pages(self, sent: dict[str, Any]) -> Iterator[tuple[dict[str, Any], Response]]:
        """Result pages, lazily — the next one is fetched only if the consumer keeps
        asking. ``countTotal`` goes on the first page only (it is ignored on later
        ones); ``pageToken`` on every later page; nothing else changes."""
        token: str | None = None
        seen: set[str] = set()
        page = 0
        while True:
            params = dict(sent)
            if token is None:
                params["countTotal"] = "true"
            else:
                params["pageToken"] = token
            resp = self.http.get(self.base_url + "studies", params=params)
            body = self._json(resp)
            studies = body.get("studies")
            if not isinstance(studies, list):
                raise CtgovError(f"studies search returned no studies list: {resp.text[:200]!r}")
            page += 1
            log.debug("ctgov: page %d, %d studies", page, len(studies))
            yield body, resp
            nxt = body.get("nextPageToken")
            if not nxt or not studies:
                return  # last page: no token (or an empty page)
            if not isinstance(nxt, str) or nxt in seen or nxt == token:
                raise CtgovError("studies search repeated a pageToken: paging would never end")
            seen.add(nxt)
            token = nxt

    def _json(self, resp: Response) -> dict[str, Any]:
        """The parsed object, or :class:`CtgovError`. Error bodies are plain text (and
        one — an oversized URL — is an nginx HTML page), so status is checked first."""
        if resp.status != 200:
            raise CtgovError(f"ClinicalTrials.gov: HTTP {resp.status}: {resp.text[:200]!r}")
        try:
            body = resp.json()
        except ValueError as e:
            raise CtgovError(f"ClinicalTrials.gov: non-JSON body: {resp.text[:200]!r}") from e
        if not isinstance(body, dict):
            raise CtgovError(f"ClinicalTrials.gov: expected an object, got {type(body).__name__}")
        return body

    def _data_version(self, holders: Iterable[str]) -> str:
        """A record's ``source_version``: the API version with the snapshot stamp(s)
        observed on the payload — ``… data 2026-09-11``; several, comma-joined, when a
        search's pages came from different nights; none when nothing was observed."""
        api = self.version_info()["apiVersion"]
        stamps = sorted(set(holders))
        return f"ClinicalTrials.gov API {api}" + (f" data {','.join(stamps)}" if stamps else "")

    def _study_record(self, study: Any, resp: Response) -> EvidenceRecord:
        """One study object → its record. The snapshot stamp is required: a record's
        data version must be observed, never implied."""
        nct = _nct_of(study)
        holder = _holder_of(study)
        info = self.version_info()
        if holder != info["dataTimestamp"][:10] and not self._warned_snapshot:
            self._warned_snapshot = True
            log.warning("ctgov: a study is stamped with snapshot %s while /version reports %s; the cache spans "
                        "nightly snapshots and each record cites its own", holder, info["dataTimestamp"])
        return EvidenceRecord(
            record_id=f"{SOURCE}:{nct}",
            source=SOURCE,
            source_version=self._data_version([holder]),
            query={"nctId": nct, "fields": self._fields_param(),
                   "api": self._study_api(nct) + "?" + urllib.parse.urlencode({"fields": self._fields_param()})},
            url=STUDY_URL.format(nct_id=nct),
            retrieved_at=resp.retrieved_at,
            payload=study,
        )


# ---------------------------------------------------------------------- helpers

def valid_nct_id(nct_id: str) -> str:
    """``NCT00909532`` — stripped and upper-cased (the path is case-insensitive). Anything
    else is refused before a request: a malformed id is HTTP 400, never a 404, but a
    short or padded one could resolve to a different trial than the caller meant."""
    s = str(nct_id).strip().upper()
    if not _NCT.match(s):
        raise ValueError(f"not an NCT id: {nct_id!r}")
    return s


def _nct_of(study: Any) -> str:
    nct = _get(study, "protocolSection", "identificationModule", "nctId")
    if not isinstance(nct, str) or not _NCT.match(nct):
        raise CtgovError(f"a study object carries no NCT id: {str(study)[:120]!r}")
    return nct


def _holder_of(study: Any) -> str:
    holder = _get(study, "derivedSection", "miscInfoModule", "versionHolder")
    if not isinstance(holder, str) or not holder:
        raise CtgovError(f"study {_nct_of(study)} carries no versionHolder snapshot stamp")
    return holder


def _names_missing(resp: Response, nct_id: str) -> bool:
    """The registry's own "no such study": a 404 whose text body names the id asked
    (``NCT number NCT99999999 not found``). Any other 404 body is not an answer."""
    return resp.status == 404 and nct_id in resp.text and "not found" in resp.text.lower()


def _text(s: str | None) -> str | None:
    """A query part as sent: stripped; ``None`` when blank. Control characters are
    refused — Essie would take them as text and answer an empty, plausible-looking
    result. A part that is only NCT ids is refused too: the registry then ignores the
    filter and the other parts (see the module docstring), and :meth:`TrialsRetriever.study`
    is the lookup by id."""
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    if _CONTROL.search(t):
        raise ValueError(f"query text must not contain control characters: {s!r}")
    if _ONLY_NCT_IDS.match(t):
        raise ValueError(f"a query part that is only NCT ids makes the registry ignore every filter and the other "
                         f"parts; look a trial up by id with study() instead: {s!r}")
    return t


def _check_statuses(found: Iterable[EvidenceRecord], statuses: list[str]) -> None:
    """Every listed study must carry a status the filter asked for; the registry has
    been seen to drop the filter silently (an id-only query), and an answer that
    contradicts the record's own claim is not one to cite."""
    for r in found:
        status = _get(r.payload, "protocolSection", "statusModule", "overallStatus")
        if status not in statuses:
            raise CtgovError(f"studies search with filter.overallStatus={','.join(statuses)} listed {_nct_of(r.payload)} "
                             f"with overallStatus {status!r}: the registry did not apply the filter")


def _statuses(statuses: Iterable[str] | None) -> list[str]:
    """Upper-cased, deduplicated, in the order given; an unknown value is refused
    before a request (the server answers HTTP 400 for it)."""
    out: list[str] = []
    for s in statuses or []:
        v = str(s).strip().upper()
        if v not in STATUSES:
            raise ValueError(f"not an overallStatus value: {s!r} (one of {sorted(STATUSES)})")
        if v not in out:
            out.append(v)
    return out


def _search_url(asked: dict[str, Any]) -> str:
    """The registry's own search page for the same ask — what a judge opens. The status
    filter goes along as the site spells it (``aggFilters=status:rec com``) when every
    status has a site code; otherwise the page shows the unfiltered ask (a superset),
    and the exact filter is in the record's ``payload.sent``. Spaces are ``%20`` — the
    site's own spelling, and unambiguous where a ``+`` might be read literally."""
    q = {k: v for k, v in (("cond", asked["condition"]), ("intr", asked["intervention"]), ("term", asked["term"])) if v}
    codes = [SITE_STATUS_CODES.get(s) for s in asked["statuses"]]
    if codes and all(codes):
        q["aggFilters"] = "status:" + " ".join(codes)
    return SEARCH_URL.format(query=urllib.parse.urlencode(q, safe=":", quote_via=urllib.parse.quote))


def _mesh_terms(p: dict[str, Any], module: str) -> list[str]:
    return [_s(m.get("term")) for m in _get(p, "derivedSection", module, "meshes") or [] if isinstance(m, dict)]


def _s(x: Any) -> str:
    return "" if x is None else str(x)


def _flag(x: Any) -> str:
    return "" if x is None else ("Y" if x else "N")


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
    """Etiquette for ``host`` on the shared limiter — unless the orchestrator already
    applies a rate for it: an explicit per-host entry (its call, either way), or a
    default at least as strict. A default of 0 means unlimited and is not "stricter"."""
    limiter = getattr(http, "limiter", None)
    if limiter is None or host in limiter.per_second:
        return
    default = float(getattr(limiter, "default", 0.0) or 0.0)
    if 0 < default <= per_second:
        return
    limiter.per_second[host] = per_second
