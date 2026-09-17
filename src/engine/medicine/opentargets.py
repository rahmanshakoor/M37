"""Open Targets Platform — gene-centric evidence for stage 6 (medicine).

Why a gene-centric retriever. Stage 2 asks "what is known about this *variant*"; stage 6
asks "what is known about this *gene* as a drug target": which diseases it is
associated with and on what kind of evidence, which pathways it sits in, whether it is
tractable by any modality, and which drugs or clinical candidates already act on it.
The Open Targets Platform answers all four from one GraphQL endpoint
(``https://api.platform.opentargets.org/api/v4/graphql``), and every answer is stored
here as an :class:`~engine.retrieve.store.EvidenceRecord` whose payload is the raw
server object, so any number a report quotes can be checked against it.

Three record kinds, one per thing a claim may cite:

* ``opentargets:<ENSG>`` — the target itself: symbol, name, biotype, GRCh38 location,
  every tractability flag (``{modality,label,value}``; the schema has no ``id`` field
  on Tractability), Reactome pathways, and the *counts* of drug rows and associated
  diseases so the record says how much it left out. :meth:`OpenTargetsRetriever.target`.
* ``opentargets:drug:<ENSG>:<CHEMBL>`` — one drug/clinical-candidate row for that
  target, from ``drugAndClinicalCandidates`` (``knownDrugs`` no longer exists in API
  26.6.x; asking for it is HTTP 400, so a rename shows up as a loud failure, never as an
  empty list). :meth:`OpenTargetsRetriever.known_drugs`.
* ``opentargets:association:<ENSG>:<disease id>`` — one target–disease association
  with its overall score and per-datatype/per-datasource breakdown, direct associations
  only (``enableIndirect: false``, the server default, sent explicitly because ``true``
  inflates a Mendelian gene's list ~9x through ontology propagation and reorders it).
  :meth:`OpenTargetsRetriever.associated_diseases`.
* ``opentargets:disease:<disease id>`` — the public disease object for a case-file
  disease id (``MONDO_0009061``): name, description, synonyms, cross-references,
  therapeutic areas and the first 100 HPO annotations with their labels. It is the
  disease context stage 6 shows the model as a record rather than as free text, so a
  sentence about the disease has an id to cite. ``phenotypes`` may be null for a term
  the platform has no annotations for; it is kept as served.
  :meth:`OpenTargetsRetriever.disease`. Verified live 2026-09-17: an unknown well-formed
  id answers ``{"data":{"disease":null}}`` with no ``errors`` — absence, as for a target.

Absence versus failure. An unknown Ensembl id is HTTP 200 ``{"data":{"target":null}}``
with no ``errors`` key — that, and only that, is "not in Open Targets" (``None`` /
``[]``). A pagination violation is *also* HTTP 200 with ``data.target`` null, but with an
``errors`` array; a schema error is HTTP 400 with ``errors`` and no ``data``; a request
without JSON content type is HTTP 415 with an HTML page. So ``errors`` is checked before
``data``, a non-JSON body raises, and :class:`~engine.retrieve.http.Http` raises on the
4xx itself. Versioned ids (``ENSG00000001626.16``), lower-case ids and symbols passed as
ids all come back null — a fabricated absence — so ids are normalised (version stripped,
upper-cased) and anything that is not ``ENSG`` + 11 digits is refused *before any
request*; symbols go through :meth:`OpenTargetsRetriever.resolve_symbol` first.

Symbol resolution. ``mapIds`` returns ``score: 1`` for case-insensitive, synonym and
Ensembl-id matches alike (``P53`` → TP53), so a mapping is accepted only when the hit's
``approvedSymbol`` equals the input case-insensitively; a synonym match is returned
only when the caller opts in.

Two things the projection does that the payload does not. ``drug.mechanismsOfAction``
is the drug's complete list across *all* its targets (under CFTR, CROFELEMER also
carries an ANO1 mechanism), so :func:`extract_drug` keeps the rows whose ``targets[].id``
is the queried gene and counts the rest. A kept row may still name a protein complex
rather than the gene alone — under TP53 the MDM2 inhibitors are "p53/Mdm2 inhibitor"
rows whose targets are TP53 *and* MDM2 — so the flat view also carries every target
symbol and the target name of the kept rows (``moa_targets`` = ``MDM2,TP53``,
``moa_target_name``), and a reader of the table can tell a complex from a direct
binder. ``rows[].diseases`` is the union of every trial condition and outcome, not an
indication list; the projection reports approved indications from clinical reports at
stage ``APPROVAL`` instead — mapped diseases only, as the platform's own
``drug.indications``; an approval term the platform could not map survives in
``conditions``.

Counts are checked, not trusted. Both list blocks carry a server ``count``; a block
whose ``rows`` do not match it (``min(count, size)`` for page 0 of the paged
association block) raises, because the alternative — a silently truncated drug list if
a later release starts paging ``drugAndClinicalCandidates`` at 25 rows — would look
exactly like a gene with fewer drugs.

Version. ``meta { apiVersion dataVersion dataPrefix }`` (never ``meta.downloads``, a
~360 KB JSON-LD blob) is observed once per retriever and cited on every record as
``Open Targets Platform 26.06 (API 26.6.3, platform2606)``. All values are strings on
the wire, ``month`` zero-padded; they are never compared numerically. The observation
replays from the HTTP cache like everything else, so an offline rerun stamps records
with the release that produced them; online, ``meta`` is also asked live and must
agree, because a cache spanning a quarterly release boundary would otherwise label
freshly fetched payloads with the old release and nothing in a target, drug or
association payload reveals which release it came from. The version is observed
before a retriever's first gene request, so a refused cache holds no gene response of
the newer release.

The cache can hold a failure. The API reports an execution-time failure (a resolver
error, a pagination violation) as HTTP 200 with an ``errors`` array, and
:class:`~engine.retrieve.http.Http` caches every 2xx as definitive — so such a body
would replay on every rerun until someone deleted the entry by hand. A *cached* body
this module rejects is therefore fetched again live, once; a live answer that
validates is written back into the cache under the same key (the entry ``Http`` would
have written), and a live answer that fails raises. Offline, the cached body's own
error is raised as it is.

Etiquette. No auth, no rate-limit headers, no 429 observed; the limiter is pinned to
3 requests/second for this host unless the orchestrator set a rate already. Page size
is capped at the server's 3000 and search latency grows with page size, so the page is
always sent explicitly — the server silently answers 25 rows when no page is given.
Associations are the top ``disease_size`` by score on page 0, never paged further; the
server's ``count`` is stored on every association record's query slice so a judge sees
"5 of 1987". ``disease_size`` is part of the exact query and so of every association
record's bytes; it is a constructor knob so that :meth:`OpenTargetsRetriever.params`
reports the value a run used. Every knob this module applies (URL, rate, page size,
stage order, datatype vocabulary, ``enableIndirect``) is returned by ``params`` for the
stage-6 orchestrator to copy into its manifest next to
:meth:`OpenTargetsRetriever.version`; this module writes no manifest itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.parse
from typing import Any, Callable, TypeVar

from engine.retrieve.http import Http, Response
from engine.retrieve.store import EvidenceRecord

log = logging.getLogger(__name__)

T = TypeVar("T")

SOURCE = "opentargets"
API_URL = "https://api.platform.opentargets.org/api/v4/graphql"
PLATFORM_URL = "https://platform.opentargets.org"
PER_SECOND = 3.0
MAX_PAGE_SIZE = 3000  # server: "the size must be between 0 and 3000"
DEFAULT_DISEASE_SIZE = 25

META_QUERY = ("{ meta { name product apiVersion { x y z suffix } "
              "dataVersion { year month iteration } dataPrefix } }")

TARGET_QUERY = (
    "query Target($id: String!) { target(ensemblId: $id) { "
    "id approvedSymbol approvedName biotype "
    "genomicLocation { chromosome start end strand } "
    "tractability { modality label value } "
    "pathways { pathwayId pathway topLevelTerm } "
    "drugAndClinicalCandidates { count } "
    "associatedDiseases(page: { index: 0, size: 0 }, enableIndirect: false) { count } } }"
)

DRUGS_QUERY = (
    "query KnownDrugs($id: String!) { target(ensemblId: $id) { id "
    "drugAndClinicalCandidates { count rows { id maxClinicalStage "
    "drug { id name drugType maximumClinicalStage "
    "mechanismsOfAction { rows { mechanismOfAction actionType targetName targets { id approvedSymbol } } } } "
    "diseases { diseaseFromSource disease { id name } } "
    "clinicalReports { id type source url trialPhase clinicalStage trialOverallStatus year trialWhyStopped "
    "diseases { disease { id name } } } } } } }"
)

ASSOCIATIONS_QUERY = (
    "query AssociatedDiseases($id: String!, $size: Int!) { target(ensemblId: $id) { id "
    "associatedDiseases(page: { index: 0, size: $size }, enableIndirect: false) { count "
    "rows { disease { id name } score datatypeScores { id score } datasourceScores { id score } } } } }"
)

MAP_QUERY = (
    'query MapSymbols($terms: [String!]!) { mapIds(queryTerms: $terms, entityNames: ["target"]) { '
    "total mappings { term hits { id entity name score object { ... on Target { id approvedSymbol } } } } } }"
)

DISEASE_QUERY = (
    "query Disease($id: String!) { disease(efoId: $id) { id name description "
    "synonyms { relation terms } dbXRefs therapeuticAreas { id name } "
    "phenotypes(page: { index: 0, size: 100 }) { count rows { phenotypeHPO { id name } } } } }"
)
DISEASE_PHENOTYPE_PAGE = 100
"""HPO annotations asked for per disease: page 0 only; ``phenotypes.count`` says how
many there are in all."""

# Clinical stage strings as the server spells them, best first. They are plain Strings in
# the schema (not an enum), so anything unlisted sorts last rather than failing.
STAGE_ORDER = {
    "APPROVAL": 9, "PHASE_4": 8, "PHASE_3": 7, "PHASE_2_3": 6, "PHASE_2": 5, "PHASE_1_2": 4,
    "PHASE_1": 3, "EARLY_PHASE_1": 2, "PRECLINICAL": 1, "UNKNOWN": 0,
}
APPROVAL_STAGE = "APPROVAL"
TRIAL_REPORT_TYPE = "CLINICAL_TRIAL"

# Association datatypes observed over 11,999 rows for four public genes. The API cannot
# enumerate them (associationDatasources() is empty in this release), so an unlisted id
# is data — it appears in the joined ``datatype_scores`` column, never an error.
DATATYPES = ("genetic_association", "somatic_mutation", "clinical", "affected_pathway",
             "literature", "genetic_literature", "animal_model", "rna_expression")

TARGET_COLUMNS = (
    "ensembl_id", "symbol", "name", "biotype", "chromosome", "start", "end", "strand",
    "tractability", "tractable_modalities", "pathway_ids", "pathways",
    "n_pathways", "n_drug_rows", "n_associated_diseases",
)
DRUG_COLUMNS = (
    "chembl_id", "name", "drug_type", "max_clinical_stage", "drug_max_clinical_stage",
    "mechanism_of_action", "action_type", "moa_targets", "moa_target_name", "n_mechanisms_other_targets",
    "approved_indications", "approved_indication_ids", "conditions",
    "n_reports", "report_types", "report_sources", "report_statuses", "trial_ids",
)
ASSOCIATION_COLUMNS = ("ensembl_id", "disease_id", "disease_name", "score", *DATATYPES,
                       "datatype_scores", "datasource_scores")
DISEASE_COLUMNS = ("disease_id", "name", "description", "synonyms", "omim", "orphanet", "n_phenotypes",
                   "phenotype_ids", "phenotype_names")

_ENSG = re.compile(r"^ENSG\d{11}$")
_DISEASE_ID = re.compile(r"^(MONDO|EFO|Orphanet|DOID|HP|NCIT|OTAR)_\d+$")
"""The platform's spelling of a disease id — the ontology prefix, an underscore, digits.
``MONDO:0009061`` (a colon) and a lower-case prefix answer null live: refused before any
request, as :func:`normalise_id` refuses a versioned Ensembl id."""
_NCT = re.compile(r"^NCT\d{8}$")


class OpenTargetsError(RuntimeError):
    """The API answered in a shape this module does not understand — never absence."""


def normalise_id(s: str) -> str:
    """The unversioned, upper-case Ensembl gene id the API accepts, or :class:`ValueError`.

    ``ENSG00000001626.16`` and ``ensg00000001626`` both answer ``null`` live; a symbol
    or anything else would too. Refusing them here keeps a bad id from looking absent."""
    core = str(s).strip().upper().split(".", 1)[0]
    if not _ENSG.match(core):
        raise ValueError(f"not an Ensembl gene id: {s!r} (resolve a symbol with resolve_symbol first)")
    return core


def target_url(ensg: str) -> str:
    return f"{PLATFORM_URL}/target/{ensg}"


def drug_url(chembl_id: str) -> str:
    return f"{PLATFORM_URL}/drug/{urllib.parse.quote(chembl_id, safe='')}"


def association_url(ensg: str, disease_id: str) -> str:
    return f"{PLATFORM_URL}/evidence/{ensg}/{urllib.parse.quote(disease_id, safe='')}"


def disease_url(disease_id: str) -> str:
    return f"{PLATFORM_URL}/disease/{urllib.parse.quote(disease_id, safe='')}"


def valid_disease_id(s: str) -> str:
    """``MONDO_0009061`` — stripped; :class:`ValueError` for any other spelling, before
    any request (the API answers ``null`` for a colon or a lower-case prefix, which
    would read as absence)."""
    core = str(s).strip()
    if not _DISEASE_ID.match(core):
        raise ValueError(f"not an Open Targets disease id: {s!r} (expected e.g. MONDO_0009061, EFO_0000400, Orphanet_586)")
    return core


class OpenTargetsRetriever:
    """Target, drug and association records for one gene at a time. Not a variant
    :class:`~engine.retrieve.Retriever`: everything here is keyed by Ensembl gene id,
    and no genomic coordinate is ever sent."""

    source = SOURCE

    def __init__(self, http: Http, *, url: str = API_URL, disease_size: int = DEFAULT_DISEASE_SIZE):
        _check_size(disease_size)
        self.http = http
        self.url = url
        self.disease_size = disease_size
        """Page size for :meth:`associated_diseases` when the call names none —
        reported by :meth:`params`, since it is in every association record's bytes."""
        self._version: str | None = None
        _pin_rate(http, urllib.parse.urlparse(url).netloc, PER_SECOND)

    # ------------------------------------------------------------------ public

    def version(self) -> str:
        """``Open Targets Platform 26.06 (API 26.6.3, platform2606)`` — from ``meta``,
        observed once per retriever and cited on every record it makes. When online the
        cached observation must agree with a live one (see the module docstring)."""
        if self._version is None:
            _resp, cached = self._post(META_QUERY, {}, _version_of)
            if not getattr(self.http, "offline", False):
                _resp, live = self._post(META_QUERY, {}, _version_of, cache_ok=False)
                if live != cached:
                    raise OpenTargetsError(
                        f"the HTTP cache was filled under {cached!r} but the server now reports {live!r}: "
                        f"remove the {urllib.parse.urlparse(self.url).netloc} entries from the HTTP cache (or run "
                        "offline to reproduce the run that filled it) rather than mixing releases")
            self._version = cached
        return self._version

    def target(self, ensembl_id: str) -> EvidenceRecord | None:
        """The target record, or ``None`` when Open Targets has no such gene."""
        ensg = normalise_id(ensembl_id)
        version = self.version()
        resp, t = self._post(TARGET_QUERY, {"id": ensg}, lambda body: _target_of(body, ensg))
        if t is None:
            return None
        return EvidenceRecord(
            record_id=f"{SOURCE}:{ensg}",
            source=SOURCE,
            source_version=version,
            query=self._query(TARGET_QUERY, {"id": ensg}),
            url=target_url(ensg),
            retrieved_at=resp.retrieved_at,
            payload=t,
        )

    def known_drugs(self, ensembl_id: str) -> list[EvidenceRecord]:
        """One record per drug/clinical candidate acting on the gene, best clinical
        stage first (the server orders rows by hash). ``[]`` when the gene has none or
        is unknown — :meth:`target` tells the two apart."""
        ensg = normalise_id(ensembl_id)
        version = self.version()
        resp, rows = self._post(DRUGS_QUERY, {"id": ensg}, lambda body: _drug_rows(body, ensg))
        return [
            EvidenceRecord(
                record_id=f"{SOURCE}:drug:{ensg}:{chembl}",
                source=SOURCE,
                source_version=version,
                query=self._query(DRUGS_QUERY, {"id": ensg}, chemblId=chembl),
                url=drug_url(chembl),
                retrieved_at=resp.retrieved_at,
                payload=row,
            )
            for chembl, row in rows or []
        ]

    def associated_diseases(self, ensembl_id: str, size: int | None = None) -> list[EvidenceRecord]:
        """The top ``size`` direct target–disease associations, in the server's
        score-descending order — page 0 only, so a gene with more associations than
        ``size`` is cut there and every record's query slice carries the server's
        ``count``. ``[]`` for an unknown gene.

        ``size`` defaults to the retriever's ``disease_size``, the one :meth:`params`
        reports; an explicit ``size`` is for ad-hoc use and is on the records' query
        slice only."""
        size = self.disease_size if size is None else _check_size(size)
        ensg = normalise_id(ensembl_id)
        version = self.version()
        variables = {"id": ensg, "size": size}
        resp, page = self._post(ASSOCIATIONS_QUERY, variables, lambda body: _association_page(body, ensg, size))
        if page is None:
            return []
        rows, count = page
        return [
            EvidenceRecord(
                record_id=f"{SOURCE}:association:{ensg}:{did}",
                source=SOURCE,
                source_version=version,
                query=self._query(ASSOCIATIONS_QUERY, variables, diseaseId=did, count=count),
                url=association_url(ensg, did),
                retrieved_at=resp.retrieved_at,
                payload=row,
            )
            for did, row in rows
        ]

    def disease(self, disease_id: str) -> EvidenceRecord | None:
        """The disease record for a platform disease id, or ``None`` when Open Targets
        has no such disease (``data.disease`` null with no errors — the one verified
        shape of absence). ``phenotypes`` is page 0 of up to
        :data:`DISEASE_PHENOTYPE_PAGE` HPO annotations, or null when the platform has
        none; the payload is the object as served either way."""
        did = valid_disease_id(disease_id)
        version = self.version()
        resp, d = self._post(DISEASE_QUERY, {"id": did}, lambda body: _disease_of(body, did))
        if d is None:
            return None
        return EvidenceRecord(
            record_id=f"{SOURCE}:disease:{did}",
            source=SOURCE,
            source_version=version,
            query=self._query(DISEASE_QUERY, {"id": did}),
            url=disease_url(did),
            retrieved_at=resp.retrieved_at,
            payload=d,
        )

    def resolve_symbol(self, symbol: str, *, allow_synonym: bool = False) -> str | None:
        """Ensembl gene id for a gene symbol, or ``None``.

        Accepted when a ``mapIds`` hit's ``approvedSymbol`` equals ``symbol``
        case-insensitively. With ``allow_synonym`` the top hit is accepted even when it
        matched through a synonym or previous symbol (``P53`` → TP53)."""
        term = symbol.strip()
        if not term:
            raise ValueError("symbol must be non-empty")
        self.version()
        _resp, hits = self._post(MAP_QUERY, {"terms": [term]}, lambda body: _hits_for(_field(body, "mapIds"), term))
        hits = [h for h in hits if isinstance(h, dict)]
        for h in hits:
            obj = h.get("object")
            if isinstance(obj, dict) and str(obj.get("approvedSymbol", "")).upper() == term.upper():
                return _hit_id(obj.get("id") or h.get("id"))
        if allow_synonym and hits and hits[0].get("id"):
            return _hit_id(hits[0]["id"])
        return None

    def extract(self, record: EvidenceRecord | None) -> dict[str, str]:
        """Project any of this source's records by kind (see the module-level helpers)."""
        if record is None:
            return {}
        kind = record.record_id.split(":")[1] if record.record_id.count(":") >= 2 else "target"
        if kind == "drug":
            return extract_drug(record)
        if kind == "association":
            return extract_association(record)
        if kind == "disease":
            return extract_disease(record)
        return extract_target(record)

    def params(self) -> dict[str, Any]:
        """Every knob this retriever applies, JSON-ready, for the orchestrator's
        ``Manifest.params`` next to :meth:`version` — none of it is recoverable from the
        records alone (the ordering rule, the projection vocabulary, the rate, the
        default page)."""
        host = urllib.parse.urlparse(self.url).netloc
        limiter = getattr(self.http, "limiter", None)
        per_second = limiter.per_second.get(host, PER_SECOND) if limiter is not None else PER_SECOND
        return {
            "api_url": self.url,
            "approval_stage": APPROVAL_STAGE,
            "datatypes": list(DATATYPES),
            "disease_phenotype_page": DISEASE_PHENOTYPE_PAGE,
            "disease_query_sha256": hashlib.sha256(DISEASE_QUERY.encode()).hexdigest(),
            "disease_size": self.disease_size,
            "enable_indirect": False,
            "max_page_size": MAX_PAGE_SIZE,
            "page_index": 0,
            "per_second": per_second,
            "stage_order": dict(STAGE_ORDER),
            "trial_report_type": TRIAL_REPORT_TYPE,
        }

    # ------------------------------------------------------------------ internals

    def _post(self, query: str, variables: dict[str, Any], parse: Callable[[dict[str, Any]], T],
              *, cache_ok: bool = True) -> tuple[Response, T]:
        """POST one GraphQL document and validate the body with ``parse``. A *cached*
        body that fails validation is fetched again live, once, and the live answer —
        if it validates — replaces the cached one, so a failure the server delivered
        with HTTP 200 (and the cache kept as definitive) neither replays nor costs a
        live request forever. Offline, the cached body's own error stands."""
        body = {"query": query, "variables": variables}
        resp = self.http.post(self.url, body, cache_ok=cache_ok)
        try:
            return resp, parse(_checked(resp))
        except OpenTargetsError:
            if not resp.from_cache or getattr(self.http, "offline", False):
                raise
            log.warning("opentargets: a cached response failed validation; fetching it again live")
            resp = self.http.post(self.url, body, cache_ok=False)
            parsed = parse(_checked(resp))
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

    def _query(self, graphql: str, variables: dict[str, Any], **slice_: Any) -> dict[str, Any]:
        """What was sent, plus which slice of the answer this record is."""
        return {"api": self.url, "graphql": graphql, "variables": dict(variables), **slice_}


def _check_size(size: Any) -> int:
    """A page size the server accepts (an ``Int!`` in 1..3000), or :class:`ValueError`
    — before any request. ``5.0`` is refused although it means 5: it would be a
    different cache key from ``5``."""
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError(f"size must be an int (a GraphQL Int!), got {size!r}")
    if not 1 <= size <= MAX_PAGE_SIZE:
        raise ValueError(f"size must be 1..{MAX_PAGE_SIZE} (server page limit), got {size}")
    return size


# ---------------------------------------------------------------------- response shape

def _checked(resp: Response) -> dict[str, Any]:
    """The parsed body, with ``data`` a dict and no ``errors`` — or :class:`OpenTargetsError`.
    ``errors`` is checked first: a validation error also leaves ``data.target`` null."""
    if resp.status != 200:
        raise OpenTargetsError(f"HTTP {resp.status} from Open Targets: {resp.text[:200]!r}")
    try:
        body = json.loads(resp.text)
    except ValueError as e:  # the 415/500 answers are HTML pages
        raise OpenTargetsError(f"non-JSON body from Open Targets: {resp.text[:200]!r}") from e
    if not isinstance(body, dict):
        raise OpenTargetsError(f"unexpected JSON from Open Targets: {resp.text[:200]!r}")
    errors = body.get("errors")
    if errors:
        messages = [e.get("message") if isinstance(e, dict) else repr(e) for e in errors]
        raise OpenTargetsError(f"Open Targets errors: {messages}")
    if not isinstance(body.get("data"), dict):
        raise OpenTargetsError(f"Open Targets returned no data object: {resp.text[:200]!r}")
    return body


def _field(body: dict[str, Any], name: str) -> Any:
    data = body["data"]
    if name not in data:
        raise OpenTargetsError(f"response has no data.{name}: {json.dumps(data)[:200]}")
    return data[name]


def _target_of(body: dict[str, Any], ensg: str) -> dict[str, Any] | None:
    """``data.target`` — ``None`` for absence; a dict that echoes the id asked for."""
    t = _field(body, "target")
    if t is None:
        return None
    if not isinstance(t, dict) or t.get("id") != ensg:
        raise OpenTargetsError("data.target did not echo the Ensembl id requested")
    return t


def _disease_of(body: dict[str, Any], did: str) -> dict[str, Any] | None:
    """``data.disease`` — ``None`` for absence; a dict that echoes the id asked for."""
    d = _field(body, "disease")
    if d is None:
        return None
    if not isinstance(d, dict) or d.get("id") != did:
        raise OpenTargetsError("data.disease did not echo the disease id requested")
    if d.get("phenotypes") is not None:
        block = _dict(d.get("phenotypes"), "disease.phenotypes")
        rows = _list(block.get("rows"), "disease.phenotypes.rows")
        _check_count(block, len(rows), "disease.phenotypes", page_size=DISEASE_PHENOTYPE_PAGE)
    return d


def _version_of(body: dict[str, Any]) -> str:
    return _version_label(_field(body, "meta"))


def _drug_rows(body: dict[str, Any], ensg: str) -> list[tuple[str, dict[str, Any]]] | None:
    """``(chembl id, row)`` per citable drug row, best clinical stage first — or ``None``
    for an unknown gene. Every shape check happens here so that a cached body failing
    any of them is fetched again live (see :meth:`OpenTargetsRetriever._post`)."""
    t = _target_of(body, ensg)
    if t is None:
        return None
    block = _dict(t.get("drugAndClinicalCandidates"), "drugAndClinicalCandidates")
    rows = _list(block.get("rows"), "drugAndClinicalCandidates.rows")
    _check_count(block, len(rows), "drugAndClinicalCandidates")
    best: dict[str, dict[str, Any]] = {}
    skipped = 0
    for row in rows:
        drug = _dict(row, "row").get("drug")
        chembl = drug.get("id") if isinstance(drug, dict) else None
        if not isinstance(chembl, str) or not chembl:
            skipped += 1  # the schema allows a null drug; a row without an id cannot be cited
            continue
        # Rows are per (drug, target), so a repeat is unexpected; if one appears, keep the
        # best-staged row rather than whichever the server's hash order delivered first.
        if chembl not in best or _drug_sort_key(chembl, row) < _drug_sort_key(chembl, best[chembl]):
            best[chembl] = row
    if skipped:
        log.info("opentargets: %d drug rows without a drug id skipped", skipped)
    return sorted(best.items(), key=lambda kv: _drug_sort_key(kv[0], kv[1]))


def _association_page(body: dict[str, Any], ensg: str, size: int) -> tuple[list[tuple[str, dict[str, Any]]], int] | None:
    """``([(disease id, row)…], server count)`` for page 0 of the association block, in
    the server's order — or ``None`` for an unknown gene. A disease id repeated on one
    page (never seen live; every row is one disease) keeps its first row, and the
    number dropped is logged so the record count can be reconciled with ``count``."""
    t = _target_of(body, ensg)
    if t is None:
        return None
    block = _dict(t.get("associatedDiseases"), "associatedDiseases")
    rows = _list(block.get("rows"), "associatedDiseases.rows")
    count = _check_count(block, len(rows), "associatedDiseases", page_size=size)
    out: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for row in rows:
        did = _dict(_dict(row, "row").get("disease"), "row.disease").get("id")
        if not isinstance(did, str) or not did:
            raise OpenTargetsError("an association row carries no disease id")
        if did in seen:
            continue
        seen.add(did)
        out.append((did, row))
    if len(out) < len(rows):
        log.info("opentargets: %d association rows repeating a disease id on the page dropped (%d kept of %d delivered)",
                 len(rows) - len(out), len(out), len(rows))
    return out, count


def _hits_for(map_ids: Any, term: str) -> list[Any]:
    mappings = _list(_dict(map_ids, "mapIds").get("mappings"), "mapIds.mappings")
    for m in mappings:
        if isinstance(m, dict) and m.get("term") == term:
            return m.get("hits") or []  # hits is nullable in the schema
    raise OpenTargetsError("mapIds did not echo the term asked for")


def _hit_id(x: Any) -> str:
    """A target hit's id, which must itself be an Ensembl gene id the API will accept."""
    try:
        return normalise_id(str(x))
    except ValueError as e:
        raise OpenTargetsError(f"mapIds target hit is not an Ensembl gene id: {x!r}") from e


def _dict(x: Any, what: str) -> dict[str, Any]:
    if not isinstance(x, dict):
        raise OpenTargetsError(f"{what} is not an object in the response")
    return x


def _list(x: Any, what: str) -> list[Any]:
    if not isinstance(x, list):
        raise OpenTargetsError(f"{what} is not a list in the response")
    return x


def _check_count(block: dict[str, Any], n_rows: int, what: str, page_size: int | None = None) -> int:
    """The block's server ``count``, once the rows delivered are shown to match it: all
    of them for an unpaged block, ``min(count, page_size)`` for page 0 of a paged one.
    A shortfall is a truncated answer (a new server-side page default, say), never a
    gene with fewer results, so it raises rather than returning a shorter list."""
    count = block.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise OpenTargetsError(f"{what}.count is not a non-negative integer: {count!r}")
    expected = count if page_size is None else min(count, page_size)
    if n_rows != expected:
        raise OpenTargetsError(f"{what}: server count {count} but {n_rows} rows delivered (expected {expected})")
    return count


def _version_label(meta: Any) -> str:
    meta = _dict(meta, "meta")
    api = _dict(meta.get("apiVersion"), "meta.apiVersion")
    data = _dict(meta.get("dataVersion"), "meta.dataVersion")
    for d, keys in ((api, ("x", "y", "z")), (data, ("year", "month"))):
        if any(not isinstance(d.get(k), str) or not d[k] for k in keys):
            raise OpenTargetsError(f"meta is missing a version part: {json.dumps(meta)[:200]}")
    api_v = ".".join(api[k] for k in ("x", "y", "z")) + (f"-{api['suffix']}" if api.get("suffix") else "")
    data_v = f"{data['year']}.{data['month']}" + (f".{data['iteration']}" if data.get("iteration") else "")
    prefix = meta.get("dataPrefix")
    if not isinstance(prefix, str) or not prefix:
        raise OpenTargetsError("meta carries no dataPrefix")
    return f"Open Targets Platform {data_v} (API {api_v}, {prefix})"


def _drug_sort_key(chembl: str, row: dict[str, Any]) -> tuple[int, str, str]:
    drug = row.get("drug") if isinstance(row.get("drug"), dict) else {}
    return (-STAGE_ORDER.get(str(row.get("maxClinicalStage")), -1), str(drug.get("name") or ""), chembl)


# ---------------------------------------------------------------------- projections

def extract_target(record: EvidenceRecord) -> dict[str, str]:
    """Flat view of a target record (:data:`TARGET_COLUMNS`)."""
    t = record.payload
    loc = t.get("genomicLocation") if isinstance(t.get("genomicLocation"), dict) else {}
    positive = [f"{x.get('modality')}:{x.get('label')}" for x in _rows(t.get("tractability")) if x.get("value") is True]
    modalities = sorted({p.split(":", 1)[0] for p in positive})
    pathways = _rows(t.get("pathways"))
    return {
        "ensembl_id": _s(t.get("id")),
        "symbol": _s(t.get("approvedSymbol")),
        "name": _s(t.get("approvedName")),
        "biotype": _s(t.get("biotype")),
        "chromosome": _s(loc.get("chromosome")),
        "start": _s(loc.get("start")),
        "end": _s(loc.get("end")),
        "strand": _s(loc.get("strand")),
        "tractability": ";".join(positive),
        "tractable_modalities": ";".join(modalities),
        "pathway_ids": ";".join(_s(p.get("pathwayId")) for p in pathways),
        "pathways": ";".join(_s(p.get("pathway")) for p in pathways),
        "n_pathways": str(len(pathways)),
        "n_drug_rows": _s(_count(t.get("drugAndClinicalCandidates"))),
        "n_associated_diseases": _s(_count(t.get("associatedDiseases"))),
    }


def extract_drug(record: EvidenceRecord) -> dict[str, str]:
    """Flat view of a drug record (:data:`DRUG_COLUMNS`). Mechanisms are the rows that
    name the queried gene among their targets; the rest are only counted. A kept row's
    full target list is ``moa_targets`` (``MDM2,TP53``: a complex, not the gene alone)
    and its ``targetName`` is ``moa_target_name``. ``approved_indications`` are the
    *mapped* diseases on ``APPROVAL`` reports, as in the platform's ``drug.indications``;
    an approval term without a mapping is visible only in ``conditions``."""
    row = record.payload
    ensg = str(record.query.get("variables", {}).get("id", ""))
    drug = row.get("drug") if isinstance(row.get("drug"), dict) else {}
    moa_rows = _rows((drug.get("mechanismsOfAction") or {}).get("rows"))
    mine = [m for m in moa_rows if any(t.get("id") == ensg for t in _rows(m.get("targets")))]
    reports = _rows(row.get("clinicalReports"))
    approved = _diseases_on(r for r in reports if r.get("clinicalStage") == APPROVAL_STAGE)
    conditions = sorted({_condition_name(d) for d in _rows(row.get("diseases"))} - {""})
    trial_ids = sorted({r["id"].upper() for r in reports
                        if r.get("type") == TRIAL_REPORT_TYPE and isinstance(r.get("id"), str) and _NCT.match(r["id"].upper())})
    return {
        "chembl_id": _s(drug.get("id")),
        "name": _s(drug.get("name")),
        "drug_type": _s(drug.get("drugType")),
        "max_clinical_stage": _s(row.get("maxClinicalStage")),
        "drug_max_clinical_stage": _s(drug.get("maximumClinicalStage")),
        "mechanism_of_action": ";".join(_unique(_s(m.get("mechanismOfAction")) for m in mine)),
        "action_type": ";".join(_unique(_s(m.get("actionType")) for m in mine)),
        "moa_targets": ";".join(_unique(_target_symbols(m) for m in mine)),
        "moa_target_name": ";".join(_unique(_s(m.get("targetName")) for m in mine)),
        "n_mechanisms_other_targets": str(len(moa_rows) - len(mine)),
        "approved_indications": ";".join(name for _id, name in approved),
        "approved_indication_ids": ";".join(_id for _id, _name in approved),
        "conditions": ";".join(conditions),
        "n_reports": str(len(reports)),
        "report_types": ";".join(sorted({_s(r.get("type")) for r in reports} - {""})),
        "report_sources": ";".join(sorted({_s(r.get("source")) for r in reports} - {""})),
        "report_statuses": ";".join(sorted({_s(r.get("trialOverallStatus")) for r in reports} - {""})),
        "trial_ids": ";".join(trial_ids),
    }


def extract_association(record: EvidenceRecord) -> dict[str, str]:
    """Flat view of an association record (:data:`ASSOCIATION_COLUMNS`): one column per
    known datatype, plus every datatype and datasource score joined as ``id=score``."""
    row = record.payload
    disease = row.get("disease") if isinstance(row.get("disease"), dict) else {}
    by_type = {str(x.get("id")): x.get("score") for x in _rows(row.get("datatypeScores"))}
    out = {
        "ensembl_id": str(record.query.get("variables", {}).get("id", "")),
        "disease_id": _s(disease.get("id")),
        "disease_name": _s(disease.get("name")),
        "score": _s(row.get("score")),
    }
    out.update({dt: _s(by_type.get(dt)) for dt in DATATYPES})
    out["datatype_scores"] = _scores(row.get("datatypeScores"))
    out["datasource_scores"] = _scores(row.get("datasourceScores"))
    return out


def extract_disease(record: EvidenceRecord) -> dict[str, str]:
    """Flat view of a disease record (:data:`DISEASE_COLUMNS`): the exact synonyms
    ``;``-joined, the OMIM and Orphanet cross-references, and the HPO annotations of
    page 0 as ids and labels (``n_phenotypes`` is the server's total)."""
    d = record.payload
    synonyms = [t for s in _rows(d.get("synonyms")) if s.get("relation") == "hasExactSynonym"
                for t in (s.get("terms") or []) if isinstance(t, str) and t]
    xrefs = [str(x) for x in (d.get("dbXRefs") or []) if isinstance(x, str)]
    ph = d.get("phenotypes") if isinstance(d.get("phenotypes"), dict) else {}
    terms = [r.get("phenotypeHPO") for r in _rows(ph.get("rows"))]
    terms = [t for t in terms if isinstance(t, dict)]
    return {
        "disease_id": _s(d.get("id")),
        "name": _s(d.get("name")),
        "description": _s(d.get("description")),
        "synonyms": ";".join(_unique(synonyms)),
        "omim": ";".join(x.split(":", 1)[1] for x in xrefs if x.startswith(("OMIM:", "OMIMPS:"))),
        "orphanet": ";".join(x.split(":", 1)[1] for x in xrefs if x.startswith("Orphanet:")),
        "n_phenotypes": _s(ph.get("count")) if ph else "",
        "phenotype_ids": ";".join(_s(t.get("id")) for t in terms),
        "phenotype_names": ";".join(_s(t.get("name")) for t in terms),
    }


def _rows(x: Any) -> list[dict[str, Any]]:
    return [r for r in x if isinstance(r, dict)] if isinstance(x, list) else []


def _count(block: Any) -> Any:
    return block.get("count") if isinstance(block, dict) else None


def _target_symbols(moa_row: dict[str, Any]) -> str:
    """Every target a mechanism row names, sorted: ``MDM2,TP53`` — more than one is a
    complex. The Ensembl id stands in where a symbol is missing."""
    symbols = {_s(t.get("approvedSymbol")) or _s(t.get("id")) for t in _rows(moa_row.get("targets"))}
    return ",".join(sorted(symbols - {""}))


def _condition_name(d: dict[str, Any]) -> str:
    """The mapped disease name, else the source's own text (unmapped terms keep only that)."""
    dis = d.get("disease")
    mapped = _s(dis.get("name")) if isinstance(dis, dict) else ""
    return mapped or _s(d.get("diseaseFromSource"))


def _diseases_on(reports: Any) -> list[tuple[str, str]]:
    """Unique ``(id, name)`` of mapped diseases on the given reports, by name then id."""
    seen: dict[str, str] = {}
    for r in reports:
        for d in _rows(r.get("diseases")):
            dis = d.get("disease")
            if isinstance(dis, dict) and isinstance(dis.get("id"), str) and dis["id"]:
                seen.setdefault(dis["id"], _s(dis.get("name")))
    return sorted(seen.items(), key=lambda kv: (kv[1], kv[0]))


def _scores(x: Any) -> str:
    return ";".join(f"{_s(s.get('id'))}={_s(s.get('score'))}" for s in _rows(x))


def _unique(items: Any) -> list[str]:
    return [x for x in dict.fromkeys(items) if x]


def _s(x: Any) -> str:
    """Payload scalar as a table string: '' for null, ``repr`` for floats (no rounding)."""
    if x is None:
        return ""
    return repr(x) if isinstance(x, float) else str(x)


def _pin_rate(http: Any, host: str, per_second: float) -> None:
    """Etiquette for ``host`` on the shared limiter — unless a rate is already set."""
    limiter = getattr(http, "limiter", None)
    if limiter is not None:
        limiter.per_second.setdefault(host, per_second)
