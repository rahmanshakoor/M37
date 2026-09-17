"""ChEMBL — compounds, curated mechanisms of action, clinical phase, indications, warnings.

Why ChEMBL for stage 6: it is the one public source where "drug X acts on target Y"
is a *curated* row with a controlled action type, the highest clinical phase the
compound reached, the approval year, the regulator's label as the reference and —
for a mutation-specific mechanism such as lumacaftor on CFTR — the protein variant
it was shown to act on. Open Targets and DGIdb aggregate it; citing ChEMBL itself
gives the judge a row to open.

What becomes a record. Five kinds, one source directory (``chembl/``):

* ``chembl:<CHEMBL id>`` — a molecule (``molecule/{id}.json``) or a target
  (``target/{id}.json``). ChEMBL ids are unique across entity types, so the id alone
  names the object; ``query["endpoint"]`` says which kind it is.
* ``chembl:mechanism:<mec_id>`` — one curated mechanism row (``mechanism/{mec_id}.json``).
* ``chembl:indication:<drugind_id>`` — one drug-indication row (``drug_indication/{id}.json``).
* ``chembl:warning:<warning_id>`` — one withdrawal / black-box row (``drug_warning/{id}.json``).

A record's ``query`` is its identity plus the API detail URL that reproduces it (every
one verified to resolve), never the list request that happened to find it, so the same
object is the same record whichever route found it (compare paper records in
:mod:`engine.retrieve.literature`); only ``retrieved_at`` is the list response's, so
two routes to one row can differ in that field alone. The payload is the raw object as
served; nothing is projected away.

How a gene becomes drugs (:meth:`ChemblRetriever.drugs_for_target_symbol`). The human
targets whose component carries the symbol as a ``GENE_SYMBOL`` synonym — kept when
that component *is* the target (``SINGLE PROTEIN``), a subunit of it (``PROTEIN
SUBUNIT``: a GABA-A receptor for GABRA1) or a member of it (``GROUP MEMBER``: the
sodium-channel family for SCN1A, where every curated blocker sits), and dropped when
it is one side of a protein-protein-interaction target (``INTERACTING PROTEIN``: a
p53/Mdm2 inhibitor acts on the interface and belongs to neither gene). Then every
mechanism row against those targets — regardless of therapeutic intent, so crofelemer
(an antidiarrhoeal CFTR inhibitor) sits beside the CF modulators and the agent must
read ``action_type``; each molecule's own record; and, because a mechanism is often
registered on a salt, ester or deuterated form while the ATC codes, the withdrawal
flag and most indication rows sit on the *parent* (phentermine hydrochloride carries
none of phentermine's 1981 withdrawal), the parent's record too whenever it differs,
and the indication and warning rows of the whole family (``parent_molecule_chembl_id``,
which every such row echoes). Warning rows are fetched for every family, flagged or
not — ChEMBL's ``withdrawn_flag`` is occasionally false on a molecule that has
``Withdrawn`` rows — and are the only place the withdrawal year, country and class
went (they are no longer on the molecule); the indication rows are the only place
ChEMBL_37 states what a drug treats (``indication_class`` is gone from the molecule).
Absence is a result: MTHFR is not a ChEMBL target and BUB1B has a target but no
curated mechanism; both come back without error.

How a mechanism or a disease becomes drugs without a gene (:meth:`ChemblRetriever.
search_mechanisms`, :meth:`ChemblRetriever.search_indications`). Stage 6's ladder
asks for interventions that act on the *consequence* of a defect, not on the gene
symbol, so two text searches are offered: the curated mechanism table filtered on
``mechanism_of_action__icontains`` (``conductance regulator`` → 13 rows; ``CFTR`` → 0,
a real answer) and the drug-indication table on ``mesh_heading__icontains`` (``cystic
fibrosis`` → 135 rows), verified live 2026-09-17 against ChEMBL_37. One page only,
``limit`` 1–:data:`MAX_CHEMBL_SEARCH`, ordered by the primary key; the page's rows
become the same ``chembl:mechanism:`` / ``chembl:indication:`` records the gene route
makes, each row's molecule (and its parent when different) is fetched so a candidate
has a record that names the drug and states ``max_phase`` / ``first_approval``, and a
``chembl-search:<sha256>`` record holds what was asked, the server's ``total_count``
and the ids returned — the reason a row was on the table. Because an unknown filter
name is silently ignored and the whole table comes back (``mechanism_of_actionx__
icontains`` → 7561 rows, HTTP 200), every row must contain the needle,
case-insensitively, in the filtered field, or the answer is refused. A needle is text
the model wrote: :func:`engine.reason.tools.check_query` refuses a genomic position in
it before any request, and it must be at least :data:`MIN_NEEDLE_CHARS` long.

What the live API does that this module defends against (probed 2026-09-12/13
against ChEMBL_37): list filters are case-sensitive exact matches that silently drop
a lower-case or padded id, so ids are upper-cased and validated before any request; a
misspelled filter name is *silently ignored* and the whole collection comes back with
HTTP 200, so every row must echo the filter it was asked with (``molecule_chembl_id``,
``parent_molecule_chembl_id``, ``target_chembl_id``, ``organism``) or the response is
refused, every target row must carry the symbol asked for, and a symbol that
"matches" more targets than any gene could is refused too; an unknown ``only=`` field
is silently dropped from the rows, so ``only=`` is never sent; a 404 on a detail
endpoint for an unknown id has an *empty* ``text/html`` body, and only that shape is
absence — a 404 with a body (a mistyped path, a wrong prefix) is a routing failure and
raises, since a cached 404 would otherwise replay as a reproducibly empty answer;
``max_phase`` is the string ``"4.0"`` on a molecule, the integer ``4`` on a mechanism,
``"-1.0"`` for "unknown" and null for >99% of molecules, so :func:`phase_label`
normalises before anything compares; an indication row's ``max_phase_for_ind`` lags
the molecule (elexacaftor: molecule ``"4.0"``, approved 2019; its cystic-fibrosis row
``"3.0"``), so approval status is read from the molecule and the indication phase says
only how far that pairing was documented; ``page_meta.limit`` is silently capped at
1000 (and ``limit=0`` means 1000); pages are walked with an explicit ``order_by`` on
the primary key and a row served twice is refused; and because a list answer is HTTP
200 whatever it holds — and so cached, and so replayed identically on every rerun —
the rows walked across pages must add up to ``page_meta.total_count`` or the answer is
refused: a short page would otherwise be reproducibly wrong rather than loudly wrong.

Version. ``status.json`` names the release (``ChEMBL_37``) and its date, and every
record carries it; it also says whether the service is ``UP``, and anything else is
refused rather than cited (a maintenance window is not a release). As in
:mod:`engine.retrieve.vep`, the cached observation is also checked live once per
instance when online, and *before* any data request: a cache spanning a ChEMBL
release would otherwise label fresh rows with the old release, and nothing in a row
reveals it. Mechanism, indication and warning record ids are ChEMBL primary keys,
stable within a release and pinned by ``source_version`` —
:attr:`ChemblRetriever.params` says so for the manifest, along with every other
setting that shapes the output.

Logging and error text name counts, endpoints and filter names, never the gene asked
about: for a rare-disease proband the candidate gene is identifying, and neither an
INFO log file nor a traceback on stderr is a run directory.

Rate. ChEMBL publishes no quota and sends no rate headers; the shared limiter keeps
``www.ebi.ac.uk`` at 3 requests/second unless the orchestrator set a rate already.
Molecules are fetched one detail GET each rather than batched: a target has tens of
drugs at most, each GET is its own cache entry, and so :meth:`molecule` and
:meth:`drugs_for_target_symbol` replay the identical record for the same compound.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.parse
from typing import Any, Iterator

from engine.reason.tools import check_query
from engine.retrieve.http import Http, Response
from engine.retrieve.store import EvidenceRecord

log = logging.getLogger(__name__)

SOURCE = "chembl"
API_URL = "https://www.ebi.ac.uk/chembl/api/data/"
COMPOUND_URL = "https://www.ebi.ac.uk/chembl/explore/compound/{id}"
TARGET_URL = "https://www.ebi.ac.uk/chembl/explore/target/{id}"
EBI_PER_SECOND = 3.0
MAX_PAGE_SIZE = 1000   # larger values are silently capped by the server
MAX_TARGETS_PER_SYMBOL = 100
"""A gene symbol resolves to a few human targets: one single protein plus the
complexes, families and interactions it takes part in (18 for GABRG2, 17 for HDAC1,
the most seen). Thousands means a filter name was ignored and the whole target table
is being served — a failure, not a rich answer."""

ORGANISM = "Homo sapiens"
GENE_SYMBOL = "GENE_SYMBOL"
ACCEPTED_RELATIONSHIPS = ("SINGLE PROTEIN", "PROTEIN SUBUNIT", "GROUP MEMBER")
"""How the component carrying the symbol may relate to its target for a mechanism
against that target to count as acting on the gene product. ``INTERACTING PROTEIN``
and ``FUSION PROTEIN`` are not on the list: the drug acts on an interface or a
chimera, and the mechanism belongs to neither partner gene."""
ORDER_BY = {"mechanism": "mec_id", "drug_indication": "drugind_id", "drug_warning": "warning_id",
            "target": "target_chembl_id"}
"""Primary key each list endpoint is walked by, so a page boundary is the same on
every rerun and a repeated key is detectable."""
FAMILY_FILTER = "parent_molecule_chembl_id"
SEARCH_SOURCE = "chembl-search"
SEARCH_FIELDS = {"mechanism": "mechanism_of_action__icontains", "drug_indication": "mesh_heading__icontains"}
"""The one text filter each search endpoint takes: a substring match, case-insensitive
on the server (``icontains``), checked again on every row here."""
SEARCH_KEYS = {"mechanism": "mechanisms", "drug_indication": "drug_indications"}
DEFAULT_CHEMBL_SEARCH = 10
MAX_CHEMBL_SEARCH = 25
"""Rows one text search returns (one page): enough to name the curated drugs for a
mechanism phrase or an indication; the record carries the server's total so the
report can say how many were left out."""
MIN_NEEDLE_CHARS = 3
"""A shorter needle matches most of a table by accident and names nothing."""

COLUMNS = (
    "chembl_id",
    "name",
    "parent_chembl_id",
    "molecule_type",
    "max_phase",
    "first_approval",
    "withdrawn",
    "black_box_warning",
    "warnings",
    "atc_codes",
    "target_chembl_id",
    "target_name",
    "target_type",
    "action_type",
    "mechanism_of_action",
    "variant_mutation",
    "indications",
    "evidence_ids",
)
"""One row per curated mechanism (drug × target), see :func:`drug_rows`."""

_CHEMBL_ID = re.compile(r"^CHEMBL[0-9]+$")
_SYMBOL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]*$")


class ChemblError(RuntimeError):
    """The API answered, but not with something this module can trust — distinct
    from "ChEMBL has no such object", which is ``None`` or an empty list."""


class ChemblRetriever:
    """Compounds, mechanisms, targets, indications and warnings from the ChEMBL REST
    API, as evidence records. Not a variant retriever: nothing genomic is ever sent —
    ChEMBL ids and gene symbols only."""

    source = SOURCE
    columns = COLUMNS

    def __init__(self, http: Http, *, base_url: str = API_URL, page_size: int = MAX_PAGE_SIZE):
        if not 1 <= page_size <= MAX_PAGE_SIZE:
            raise ValueError(f"page_size must be 1..{MAX_PAGE_SIZE} (server cap), got {page_size}")
        self.http = http
        self.base_url = base_url.rstrip("/") + "/"
        self.page_size = page_size
        self._version: str | None = None
        _pin_rate(http, _host(self.base_url), EBI_PER_SECOND)

    # ------------------------------------------------------------------ version

    def version(self) -> str:
        """``ChEMBL_37 (2026-05-01)`` from ``status.json`` — observed, cached on the
        instance; when online the cached observation must agree with a live one. Every
        data request calls this first, so a release boundary is caught before a page
        served under the new release can be cached under the old one."""
        if self._version is None:
            cached = self._status(cache_ok=True)
            if not getattr(self.http, "offline", False):
                live = self._status(cache_ok=False)
                if live != cached:
                    raise ChemblError(
                        f"the HTTP cache was filled under {cached!r} but the server now reports {live!r}: "
                        "delete the HTTP cache rather than mixing releases")
            self._version = cached
        return self._version

    def _status(self, *, cache_ok: bool) -> str:
        body = self._json(self.http.get(self.base_url + "status.json", cache_ok=cache_ok))
        release, date = body.get("chembl_db_version"), body.get("chembl_release_date")
        if not (isinstance(release, str) and release and isinstance(date, str) and date):
            raise ChemblError(f"status.json names no release to cite: {str(body)[:200]}")
        if body.get("status") != "UP":
            raise ChemblError(f"status.json reports {release} but status {body.get('status')!r}, not 'UP': "
                              "nothing served now can be cited")
        return f"{release} ({date})"

    @property
    def params(self) -> dict[str, Any]:
        """Every setting that shapes the output, JSON-ready, for the stage manifest's
        ``params`` — none of it is recoverable from the records, whose ``query`` names
        the detail URL that reproduces the object rather than the list request that
        found it. The version is separate: :meth:`version`."""
        limiter = getattr(self.http, "limiter", None)
        per_second = limiter.per_second.get(_host(self.base_url), EBI_PER_SECOND) if limiter is not None else EBI_PER_SECOND
        return {
            "api_url": self.base_url,
            "page_size": self.page_size,
            "max_page_size": MAX_PAGE_SIZE,
            "max_targets_per_symbol": MAX_TARGETS_PER_SYMBOL,
            "organism": ORGANISM,
            "syn_type": GENE_SYMBOL,
            "accepted_relationships": list(ACCEPTED_RELATIONSHIPS),
            "order_by": dict(ORDER_BY),
            "family_filter": FAMILY_FILTER,
            "per_second": per_second,
            "release_scoped_record_ids": ["chembl:mechanism:<mec_id>", "chembl:indication:<drugind_id>",
                                          "chembl:warning:<warning_id>"],
            "search_fields": dict(SEARCH_FIELDS),
            "search_default_limit": DEFAULT_CHEMBL_SEARCH,
            "search_max_limit": MAX_CHEMBL_SEARCH,
            "search_min_needle_chars": MIN_NEEDLE_CHARS,
        }

    # ------------------------------------------------------------------ molecules

    def molecule(self, chembl_id: str) -> EvidenceRecord | None:
        """The molecule object for ``chembl_id``, or ``None`` when ChEMBL has no such id
        (a 404 with an empty body — the one shape verified to mean that)."""
        cid = valid_chembl_id(chembl_id)
        self.version()
        resp = self.http.get(self._detail("molecule", cid), cache_404=True)
        if resp.status == 404:
            if resp.text != "":
                raise ChemblError(f"molecule/{cid}: a 404 with a body ({len(resp.text)} chars) is not ChEMBL saying "
                                  "no such molecule — the path or prefix is wrong")
            return None
        body = self._json(resp)
        if body.get("molecule_chembl_id") != cid:
            raise ChemblError(f"molecule/{cid} answered for {body.get('molecule_chembl_id')!r}")
        return self._record(cid, "molecule", cid, COMPOUND_URL.format(id=cid), body, resp)

    def mechanisms(self, chembl_id: str) -> list[EvidenceRecord]:
        """Every curated mechanism row for the molecule ``chembl_id`` (``[]`` if none),
        in ``mec_id`` order."""
        cid = valid_chembl_id(chembl_id)
        rows = self._rows("mechanism", "mechanisms", {"molecule_chembl_id": cid})
        return sorted((self._mechanism(row, resp) for row, resp in rows), key=lambda r: r.payload["mec_id"])

    def indications(self, parent_chembl_id: str) -> list[EvidenceRecord]:
        """Every drug-indication row of the family whose parent is ``parent_chembl_id``
        — the parent itself and each salt, ester or deuterated form filed under it
        (``molecule_hierarchy.parent_chembl_id``; a child id matches nothing) — in
        ``drugind_id`` order, ``[]`` if none. Any phase: ``max_phase_for_ind`` says how
        far each pairing got, and ``molecule_chembl_id`` which form it was documented on."""
        pid = valid_chembl_id(parent_chembl_id)
        rows = self._rows("drug_indication", "drug_indications", {FAMILY_FILTER: pid})
        out = []
        for row, resp in rows:
            did = _int_field(row, "drugind_id")
            out.append(self._record(f"indication:{did}", "drug_indication", str(did),
                                    COMPOUND_URL.format(id=_id_field(row, "molecule_chembl_id")), row, resp))
        return sorted(out, key=lambda r: r.payload["drugind_id"])

    def warnings(self, parent_chembl_id: str) -> list[EvidenceRecord]:
        """Every withdrawal / black-box row of the family whose parent is
        ``parent_chembl_id`` (``[]`` if none — every CFTR modulator), in ``warning_id``
        order. The molecule's own ``withdrawn_flag`` and ``black_box_warning`` only say
        *that* a warning exists, and not always that; the year, country, class and text
        are here. One withdrawal is often several near-duplicate rows differing only
        in the cited reference — see :func:`warning_summary`."""
        pid = valid_chembl_id(parent_chembl_id)
        rows = self._rows("drug_warning", "drug_warnings", {FAMILY_FILTER: pid})
        out = []
        for row, resp in rows:
            wid = _int_field(row, "warning_id")
            out.append(self._record(f"warning:{wid}", "drug_warning", str(wid),
                                    COMPOUND_URL.format(id=_id_field(row, "molecule_chembl_id")), row, resp))
        return sorted(out, key=lambda r: r.payload["warning_id"])

    # ------------------------------------------------------------------ targets

    def targets_for_symbol(self, symbol: str) -> list[EvidenceRecord]:
        """Human targets whose component carrying ``symbol`` as its ``GENE_SYMBOL``
        synonym is the target, a subunit of it or a member of it
        (:data:`ACCEPTED_RELATIONSHIPS`), in id order; ``[]`` when the gene is not a
        ChEMBL target or only takes part in protein-protein-interaction targets."""
        sym = valid_symbol(symbol)
        params = {
            "target_components__target_component_synonyms__component_synonym__iexact": sym,
            "target_components__target_component_synonyms__syn_type": GENE_SYMBOL,
            "organism": ORGANISM,
        }
        out = []
        excluded = 0
        for row, resp in self._rows("target", "targets", params, max_total=MAX_TARGETS_PER_SYMBOL):
            tid = _id_field(row, "target_chembl_id")
            relationships = symbol_relationships(row, sym)
            if not relationships:
                raise ChemblError(f"target {tid} carries no GENE_SYMBOL synonym equal to the one asked for: "
                                  "the synonym filter was not applied")
            if not relationships & set(ACCEPTED_RELATIONSHIPS):
                excluded += 1
                continue
            out.append(self._record(tid, "target", tid, TARGET_URL.format(id=tid), row, resp))
        log.info("chembl: %d target(s) carry the symbol, %d kept (gene product is the target, a subunit or a member)",
                 len(out) + excluded, len(out))
        return sorted(out, key=lambda r: r.record_id)

    def mechanisms_for_target(self, target_chembl_id: str) -> list[EvidenceRecord]:
        """Every curated mechanism row against the target (``[]`` if none), in ``mec_id``
        order — all action types and phases; the caller decides what counts as a drug."""
        tid = valid_chembl_id(target_chembl_id)
        rows = self._rows("mechanism", "mechanisms", {"target_chembl_id": tid})
        return sorted((self._mechanism(row, resp) for row, resp in rows), key=lambda r: r.payload["mec_id"])

    def drugs_for_target_symbol(self, symbol: str) -> list[EvidenceRecord]:
        """Everything ChEMBL knows about drugs acting on the gene product ``symbol``:
        the target record(s), then every mechanism row against them, then each
        molecule's own record and — for a salt, ester or deuterated form — its parent's,
        then the indication rows of every family, then the warning rows of every family.
        ``[]`` when the gene is not a ChEMBL target; the target record alone when it is
        one but no mechanism is curated against it (BUB1B) — citable either way.
        Project with :func:`drug_rows`."""
        targets = self.targets_for_symbol(symbol)
        mechanisms: list[EvidenceRecord] = []
        for t in targets:
            mechanisms.extend(self.mechanisms_for_target(t.payload["target_chembl_id"]))
        molecules: dict[str, EvidenceRecord] = {}
        for cid in sorted({m.payload["molecule_chembl_id"] for m in mechanisms}):
            molecules[cid] = self._named_molecule(cid)
        # the parent carries the ATC codes, the withdrawal flag and most indication rows
        parents = sorted({parent_id(m.payload) for m in molecules.values()})
        for pid in parents:
            if pid not in molecules:
                molecules[pid] = self._named_molecule(pid)
        indications: list[EvidenceRecord] = []
        warnings: list[EvidenceRecord] = []
        for pid in parents:
            indications.extend(self.indications(pid))
            warnings.extend(self.warnings(pid))
        log.info("chembl: %d target(s), %d mechanism row(s), %d molecule(s) in %d famil(ies), %d indication row(s), %d warning row(s)",
                 len(targets), len(mechanisms), len(molecules), len(parents), len(indications), len(warnings))
        return (targets + mechanisms + [molecules[cid] for cid in sorted(molecules)]
                + sorted(indications, key=lambda r: r.payload["drugind_id"])
                + sorted(warnings, key=lambda r: r.payload["warning_id"]))

    # ------------------------------------------------------------------ text searches

    def search_mechanisms(self, text: str, *, limit: int = DEFAULT_CHEMBL_SEARCH) -> tuple[list[EvidenceRecord], int]:
        """Curated mechanism rows whose ``mechanism_of_action`` contains ``text``
        (case-insensitively), one page of at most ``limit`` in ``mec_id`` order, and the
        server's total. Returns ``([search record, *rows, *molecules], total_count)``:
        the ``chembl-search:`` record first, the rows as ``chembl:mechanism:<mec_id>``,
        then each row's molecule and its parent when different, sorted by id."""
        return self._search("mechanism", text, limit)

    def search_indications(self, text: str, *, limit: int = DEFAULT_CHEMBL_SEARCH) -> tuple[list[EvidenceRecord], int]:
        """Drug-indication rows whose MeSH heading contains ``text``, one page of at
        most ``limit`` in ``drugind_id`` order, and the server's total — shaped as
        :meth:`search_mechanisms` (rows as ``chembl:indication:<drugind_id>``)."""
        return self._search("drug_indication", text, limit)

    def _search(self, endpoint: str, text: str, limit: int) -> tuple[list[EvidenceRecord], int]:
        needle = valid_needle(text)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_CHEMBL_SEARCH:
            raise ValueError(f"limit must be an int between 1 and {MAX_CHEMBL_SEARCH}, got {limit!r}")
        field, key, pk = SEARCH_FIELDS[endpoint], SEARCH_KEYS[endpoint], ORDER_BY[endpoint]
        self.version()
        url = f"{self.base_url}{endpoint}.json"
        params: dict[str, Any] = {field: needle, "limit": limit, "order_by": pk}
        resp = self.http.get(url, params=params)
        body = self._json(resp)
        rows, meta = body.get(key), body.get("page_meta")
        if not isinstance(rows, list) or not isinstance(meta, dict):
            raise ChemblError(f"{endpoint}.json returned no {key!r}/page_meta: {resp.text[:200]!r}")
        total = _int_field(meta, "total_count")
        if len(rows) > limit:
            raise ChemblError(f"{endpoint}.json served {len(rows)} rows for limit {limit}: the page cannot be cited")
        ids: set[Any] = set()
        records: list[EvidenceRecord] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ChemblError(f"{endpoint}.json row is not an object: {str(row)[:100]!r}")
            if needle.lower() not in str(row.get(field.removesuffix("__icontains")) or "").lower():
                raise ChemblError(f"{endpoint}.json ignored the {field} filter: a row does not contain the needle")
            if row.get(pk) is None or row[pk] in ids:
                raise ChemblError(f"{endpoint}.json served {pk}={row.get(pk)!r} twice (or without it): the page cannot be cited")
            ids.add(row[pk])
            if endpoint == "mechanism":
                records.append(self._mechanism(row, resp))
            else:
                did = _int_field(row, "drugind_id")
                records.append(self._record(f"indication:{did}", "drug_indication", str(did),
                                            COMPOUND_URL.format(id=_id_field(row, "molecule_chembl_id")), row, resp))
        molecules: dict[str, EvidenceRecord] = {}
        for row in rows:
            for name in ("molecule_chembl_id", FAMILY_FILTER):
                cid = row.get(name)
                if cid and cid not in molecules:
                    molecules[valid_chembl_id(str(cid))] = self._named_molecule(valid_chembl_id(str(cid)))
        sent = {"endpoint": endpoint, "field": field, "needle": needle, "limit": limit}
        search = EvidenceRecord(
            record_id=f"{SEARCH_SOURCE}:{_digest(sent)}",
            source=SEARCH_SOURCE,
            source_version=self.version(),
            query={**sent, "api": url + "?" + urllib.parse.urlencode(params)},
            url=url + "?" + urllib.parse.urlencode(params),
            retrieved_at=resp.retrieved_at,
            payload={"sent": sent, "total_count": total, "ids": [r.record_id for r in records]},
        )
        log.info("chembl: %s search matched %d row(s) in all, %d returned, %d molecule(s)", endpoint, total, len(records), len(molecules))
        return [search, *records, *(molecules[c] for c in sorted(molecules))], total

    # ------------------------------------------------------------------ internals

    def _named_molecule(self, cid: str) -> EvidenceRecord:
        """A molecule another row names: "no such molecule" is then a broken answer."""
        mol = self.molecule(cid)
        if mol is None:
            raise ChemblError(f"ChEMBL rows name {cid} but molecule/{cid} is 404")
        return mol

    def _detail(self, endpoint: str, ident: str) -> str:
        return f"{self.base_url}{endpoint}/{ident}.json"

    def _rows(self, endpoint: str, key: str, filters: dict[str, str], *,
              max_total: int | None = None) -> Iterator[tuple[dict[str, Any], Response]]:
        """Rows of a list endpoint across pages in primary-key order, each checked to
        echo every filter whose name is a field of the row — the server ignores a filter
        it does not recognise and serves the whole table instead, with HTTP 200 — and,
        once the last page is in, the rows walked must add up to ``page_meta.total_count``:
        a short page is served with HTTP 200 too, and would be cached and replayed as
        the truth. Error text names endpoints, filter names, counts and ChEMBL ids —
        never the symbol asked about."""
        self.version()  # the release cross-check runs before any data page is cached
        url = f"{self.base_url}{endpoint}.json"
        pk = ORDER_BY[endpoint]
        names = sorted(filters)
        offset = seen = 0
        ids: set[Any] = set()
        while True:
            params: dict[str, Any] = {**filters, "order_by": pk, "limit": self.page_size, "offset": offset}
            resp = self.http.get(url, params=params)
            body = self._json(resp)
            rows, meta = body.get(key), body.get("page_meta")
            if not isinstance(rows, list) or not isinstance(meta, dict):
                raise ChemblError(f"{endpoint}.json returned no {key!r}/page_meta: {resp.text[:200]!r}")
            total = _int_field(meta, "total_count")
            if max_total is not None and total > max_total:
                raise ChemblError(f"{endpoint}.json matched {total} rows for filters {names}; "
                                  f"more than {max_total} means a filter was ignored")
            for row in rows:
                if not isinstance(row, dict):
                    raise ChemblError(f"{endpoint}.json row is not an object: {str(row)[:100]!r}")
                for name, value in filters.items():
                    if name in row and row[name] != value:
                        raise ChemblError(f"{endpoint}.json ignored {name}={value!r}: a row carries {row[name]!r}")
                if row.get(pk) is None or row[pk] in ids:
                    raise ChemblError(f"{endpoint}.json served {pk}={row.get(pk)!r} twice (or without it) across pages: "
                                      "the walk cannot be cited")
                ids.add(row[pk])
                yield row, resp
            seen += len(rows)
            if not meta.get("next"):
                if seen != total:
                    raise ChemblError(f"{endpoint}.json served {seen} row(s) for filters {names} but page_meta.total_count "
                                      f"is {total}: a truncated page cannot be cited")
                return
            if not rows:
                raise ChemblError(f"{endpoint}.json page at offset {offset} is empty yet names a next page "
                                  f"({seen} of {total} rows seen): a truncated walk cannot be cited")
            offset += int(meta.get("limit") or self.page_size)

    def _json(self, resp: Response) -> dict[str, Any]:
        if resp.status != 200:
            raise ChemblError(f"ChEMBL: HTTP {resp.status}: {resp.text[:200]!r}")
        try:
            body = resp.json()
        except ValueError as e:  # the XML default and HTML error pages
            raise ChemblError(f"ChEMBL: non-JSON body: {resp.text[:200]!r}") from e
        if not isinstance(body, dict):
            raise ChemblError(f"ChEMBL: expected an object, got {type(body).__name__}")
        return body

    def _mechanism(self, row: dict[str, Any], resp: Response) -> EvidenceRecord:
        mec_id = _int_field(row, "mec_id")
        cid = _id_field(row, "molecule_chembl_id")
        return self._record(f"mechanism:{mec_id}", "mechanism", str(mec_id), COMPOUND_URL.format(id=cid), row, resp)

    def _record(self, ident: str, endpoint: str, api_id: str, url: str, payload: Any, resp: Response) -> EvidenceRecord:
        return EvidenceRecord(
            record_id=f"{SOURCE}:{ident}",
            source=SOURCE,
            source_version=self.version(),
            query={"endpoint": endpoint, "id": api_id, "api": self._detail(endpoint, api_id)},
            url=url,
            retrieved_at=resp.retrieved_at,
            payload=payload,
        )


# ---------------------------------------------------------------------- projection

def drug_rows(records: list[EvidenceRecord]) -> list[dict[str, str]]:
    """Project the records of :meth:`ChemblRetriever.drugs_for_target_symbol` onto
    :data:`COLUMNS`: one row per mechanism, joined with its molecule, that molecule's
    parent when it has one, its target, and the family's indication and warning rows;
    ``evidence_ids`` names every record the row was built from. Name, type, phase and
    approval year are the mechanism's own molecule's; ``withdrawn`` and
    ``black_box_warning`` are set if either the molecule or its parent says so;
    ``atc_codes`` is the union; an indication documented on another member of the
    family says so (``via CHEMBL…``); ``target_name`` carries the target type in
    brackets unless it is the single protein, so a family- or complex-level mechanism
    reads as one. Sorted by molecule id then ``mec_id``; a mechanism whose molecule
    record is absent still yields a row with blank molecule columns."""
    by_kind: dict[str, list[EvidenceRecord]] = {}
    for r in records:
        by_kind.setdefault(str(r.query.get("endpoint")), []).append(r)
    molecules = {r.payload["molecule_chembl_id"]: r for r in by_kind.get("molecule", [])}
    targets = {r.payload["target_chembl_id"]: r for r in by_kind.get("target", [])}
    indications: dict[str, list[EvidenceRecord]] = {}
    for r in by_kind.get("drug_indication", []):
        indications.setdefault(_family_of(r.payload), []).append(r)
    warnings: dict[str, list[EvidenceRecord]] = {}
    for r in by_kind.get("drug_warning", []):
        warnings.setdefault(_family_of(r.payload), []).append(r)

    rows = []
    mechanisms = sorted(by_kind.get("mechanism", []), key=lambda r: (r.payload["molecule_chembl_id"], r.payload["mec_id"]))
    for mech in mechanisms:
        m = mech.payload
        cid = m["molecule_chembl_id"]
        mol = molecules.get(cid)
        p = mol.payload if mol else {}
        pid = str(_get(p, "molecule_hierarchy", "parent_chembl_id") or m.get("parent_molecule_chembl_id") or cid)
        parent = molecules.get(pid) if pid != cid else None
        pp = parent.payload if parent else {}
        target = targets.get(m.get("target_chembl_id"))
        t = target.payload if target else {}
        inds = sorted(indications.get(pid, []), key=lambda r: (-_phase(r.payload.get("max_phase_for_ind")),
                                                              str(r.payload.get("mesh_heading") or ""), r.payload["drugind_id"]))
        warns = sorted(warnings.get(pid, []), key=lambda r: r.payload["warning_id"])
        variant = m.get("variant_sequence")
        target_type = str(t.get("target_type") or "")
        target_name = str(t.get("pref_name") or "")
        if target and target_type != "SINGLE PROTEIN":
            target_name = f"{target_name} [{target_type}]"
        rows.append({
            "chembl_id": cid,
            "name": str(p.get("pref_name") or ""),
            "parent_chembl_id": pid,
            "molecule_type": str(p.get("molecule_type") or ""),
            "max_phase": phase_label(p["max_phase"] if p.get("max_phase") is not None else m.get("max_phase")),
            "first_approval": "" if p.get("first_approval") is None else str(p["first_approval"]),
            "withdrawn": _flag(p.get("withdrawn_flag"), pp.get("withdrawn_flag")),
            "black_box_warning": _flag(p.get("black_box_warning"), pp.get("black_box_warning")),
            "warnings": "; ".join(warning_summary(warns)),
            "atc_codes": ";".join(sorted({str(a) for src in (p, pp) for a in (src.get("atc_classifications") or [])})),
            "target_chembl_id": str(m.get("target_chembl_id") or ""),
            "target_name": target_name,
            "target_type": target_type,
            "action_type": str(m.get("action_type") or ""),
            "mechanism_of_action": str(m.get("mechanism_of_action") or ""),
            "variant_mutation": str(variant.get("mutation") or "") if isinstance(variant, dict) else "",
            "indications": "; ".join(_indication_text(r.payload, cid) for r in inds),
            "evidence_ids": ";".join([mech.record_id] + [r.record_id for r in (mol, parent, target) if r]
                                     + [r.record_id for r in inds] + [r.record_id for r in warns]),
        })
    return rows


def phase_label(value: Any) -> str:
    """ChEMBL's clinical phase as one spelling: ``"4"`` for the molecule's ``"4.0"``
    and the mechanism's ``4`` alike, ``"0.5"`` for early phase 1, ``"-1"`` for
    ChEMBL's "unknown", ``""`` for null (the norm for a research compound)."""
    if value is None or value == "":
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(f)) if f == int(f) else str(f)


def gene_symbols(target: dict[str, Any]) -> list[str]:
    """The ``GENE_SYMBOL`` synonyms across a target's components, in order of appearance.
    ``GENE_SYMBOL_OTHER`` is left out: it carries aliases and curation junk."""
    out = []
    for comp in target.get("target_components") or []:
        for syn in comp.get("target_component_synonyms") or []:
            if syn.get("syn_type") == GENE_SYMBOL and syn.get("component_synonym"):
                out.append(str(syn["component_synonym"]))
    return out


def symbol_relationships(target: dict[str, Any], symbol: str) -> set[str]:
    """The ``relationship`` of every component of ``target`` that carries ``symbol``
    (case-insensitively) as a ``GENE_SYMBOL`` synonym — ``{"SINGLE PROTEIN"}`` for the
    protein itself, ``{"PROTEIN SUBUNIT"}`` in a complex, ``{"GROUP MEMBER"}`` in a
    family, ``{"INTERACTING PROTEIN"}`` in a protein-protein interaction; empty when
    no component carries it."""
    sym = symbol.upper()
    out: set[str] = set()
    for comp in target.get("target_components") or []:
        for syn in comp.get("target_component_synonyms") or []:
            if syn.get("syn_type") == GENE_SYMBOL and str(syn.get("component_synonym") or "").upper() == sym:
                out.add(str(comp.get("relationship") or ""))
    return out


def parent_id(molecule: dict[str, Any]) -> str:
    """``molecule_hierarchy.parent_chembl_id`` — the molecule itself when it is the
    parent or ChEMBL states no hierarchy."""
    return str(_get(molecule, "molecule_hierarchy", "parent_chembl_id") or molecule["molecule_chembl_id"])


def warning_summary(records: list[EvidenceRecord]) -> list[str]:
    """One line per distinct ``(warning_type, warning_year, warning_country)`` in
    ``warning_id`` order — rofecoxib's single 2004 withdrawal is ten rows that differ
    only in the reference cited and the EFO term, and reads as one."""
    seen: dict[tuple[Any, Any, Any], str] = {}
    for r in sorted(records, key=lambda r: r.payload["warning_id"]):
        w = r.payload
        key = (w.get("warning_type"), w.get("warning_year"), w.get("warning_country"))
        if key not in seen:
            seen[key] = " ".join(str(x) for x in key if x not in (None, ""))
    return list(seen.values())


def valid_chembl_id(chembl_id: str) -> str:
    """``CHEMBL2010601`` — upper-cased and stripped, because list filters are
    case-sensitive exact matches and drop a lower-case or padded id without a word."""
    s = str(chembl_id).strip().upper()
    if not _CHEMBL_ID.match(s):
        raise ValueError(f"not a ChEMBL id: {chembl_id!r}")
    return s


def valid_symbol(symbol: str) -> str:
    """``CFTR`` — stripped and upper-cased. The server matches the synonym
    case-insensitively, so this only keeps one HTTP-cache key per gene whatever a
    caller's spelling; the records carry ChEMBL's own spelling regardless. The error
    does not echo the input: a traceback is not a run directory."""
    s = str(symbol).strip().upper()
    if not _SYMBOL.match(s):
        raise ValueError(f"not a gene symbol: {len(s)} character(s) after stripping, "
                         "expected letters/digits then letters, digits, '.', '_', '@' or '-'")
    return s


def valid_needle(text: Any) -> str:
    """The search text stripped, at least :data:`MIN_NEEDLE_CHARS` long and free of a
    genomic position (:func:`~engine.reason.tools.check_query`) — or ``ValueError``
    before any request. The error never echoes the text."""
    s = str(text if text is not None else "").strip()
    if len(s) < MIN_NEEDLE_CHARS:
        raise ValueError(f"a ChEMBL search needs at least {MIN_NEEDLE_CHARS} characters of text")
    check_query(s)
    return s


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _phase(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -2.0  # sorts after every real phase, including ChEMBL's -1 "unknown"


def _flag(*values: Any) -> str:
    """``withdrawn_flag`` is a boolean, ``black_box_warning`` an integer; both mean the
    same yes/no. Across a molecule and its parent: yes if either says yes, no if one
    says no and neither says yes, unknown (null, ChEMBL's -1) otherwise."""
    if any(v is True or v == 1 for v in values):
        return "Y"
    if any(v is False or v == 0 for v in values):
        return "N"
    return ""


def _family_of(row: dict[str, Any]) -> str:
    return str(row.get(FAMILY_FILTER) or row.get("molecule_chembl_id") or "")


def _indication_text(ind: dict[str, Any], own_chembl_id: str) -> str:
    name = ind.get("mesh_heading") or ind.get("efo_term") or "?"
    ids = ", ".join(str(x) for x in (ind.get("efo_id"), ind.get("mesh_id")) if x)
    phase = phase_label(ind.get("max_phase_for_ind"))
    via = "" if ind.get("molecule_chembl_id") in (None, own_chembl_id) else f" via {ind['molecule_chembl_id']}"
    return f"{name} [{ids}] phase {phase or '?'}{via}"


def _int_field(obj: dict[str, Any], name: str) -> int:
    v = obj.get(name)
    if not isinstance(v, int) or isinstance(v, bool):
        raise ChemblError(f"no integer {name} in {str(obj)[:120]!r}")
    return v


def _id_field(row: dict[str, Any], name: str) -> str:
    try:
        return valid_chembl_id(str(row.get(name)))
    except ValueError as e:
        raise ChemblError(f"row has no ChEMBL id in {name}: {str(row)[:120]!r}") from e


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
