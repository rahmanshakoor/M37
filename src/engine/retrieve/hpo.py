"""HPO term records — the case's phenotype terms as citable evidence.

Stage 5 used to hand the model the case's HPO ids bare (``HP:0012236``), and a model
that knows the ontology by heart attaches a label from memory; when it misremembers,
the wrong label reaches the report and nothing can dispute it, because no record in
the run says what the term is called. This module fetches each term from the JAX
ontology API and stores it as an :class:`~engine.retrieve.store.EvidenceRecord`
(``hpo:HP:0012236``) whose payload is the API's term object verbatim — ``name`` (the
label), ``definition``, ``synonyms``, ``xrefs`` — so the bundle can show the model the
label beside the id, the model can cite the record, and
:mod:`engine.agents.terms` can hold every label it writes against the record.

Absence is a result, not an error, and only one response shape means it. Verified live
on 2026-09-17: a well-formed id the ontology lacks (``HP:9999999``) answers HTTP 404
with an *empty* body, and that alone is "no such term" (``None``). A 404 carrying a
body, a non-JSON body, or an object whose ``id`` is not the one asked is
:class:`HpoError` — the API answered with something this module cannot cite.
Lower-case ``hp:0012236`` (404 empty at the API) and ``HP_0012236`` (400 "TermId
construction error") are refused before any request by validating against
:data:`engine.config.HPO_ID` after ``strip().upper()``, so a spelling slip never
reads as absence.

The service exposes no release identifier — no version endpoint, no ``ETag`` or
``Last-Modified`` header — so :meth:`HpoRetriever.version` is a constant that says
so and ``params`` documents it; a record is dated by its ``retrieved_at`` only.

:func:`case_terms` is the "fetched once per run" mechanism stages 5 and 6 share: a
term already held by any store of the run is served from there (no request), only
the missing ones are fetched, and the ``Http`` is built lazily so a run whose terms
are all in the store — or a test — never touches the network or the cache.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from engine.config import HPO_ID
from engine.retrieve.http import Http, HttpError
from engine.retrieve.store import EvidenceRecord, EvidenceStore

SOURCE = "hpo"
API_URL = "https://ontology.jax.org/api/hp/terms/"
TERM_URL = "https://hpo.jax.org/browse/term/{id}"
JAX_PER_SECOND = 3.0
"""JAX publishes no rate limit and sends no rate headers; three requests a second is
the etiquette every other public host in the engine gets."""
COLUMNS = ("hpo_id", "label", "definition", "synonyms", "descendant_count")
VERSION = "JAX ontology API hp (the service exposes no release identifier)"
VERSION_NOTE = "the JAX API exposes no HPO release; records are dated by retrieved_at only"

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


class HpoError(RuntimeError):
    """The API answered with a shape this module cannot cite: a 404 with a body, a
    non-JSON body, or a term object whose id is not the one asked."""


class HpoRetriever:
    """One HPO term per request, as an evidence record. Not a variant
    :class:`~engine.retrieve.Retriever`: terms are keyed by HPO id and nothing about
    the proband beyond the id itself is ever sent."""

    source = SOURCE
    columns = COLUMNS

    def __init__(self, http: Http, *, base_url: str = API_URL):
        self.http = http
        self.base_url = base_url.rstrip("/") + "/"
        _pin_rate(http, _host(self.base_url), JAX_PER_SECOND)

    def version(self) -> str:
        """Constant: the JAX API exposes no release (see the module docstring)."""
        return VERSION

    @property
    def params(self) -> dict[str, Any]:
        return {"api_url": self.base_url, "per_second": JAX_PER_SECOND, "version_note": VERSION_NOTE}

    def term(self, hpo_id: str) -> EvidenceRecord | None:
        """The term record for ``hpo_id``, or ``None`` when the ontology has no such
        term (HTTP 404 with an empty body — the only absence shape). Raises
        ``ValueError`` for a malformed id before any request, :class:`HpoError` for an
        answer that is neither a term nor that absence."""
        hpo_id = valid_hpo_id(hpo_id)
        api = self.base_url + hpo_id
        try:
            resp = self.http.get(api, cache_404=True)
        except HttpError as e:  # a transport that raises on 404 instead of returning it
            if e.status != 404:
                raise
            status, text, retrieved_at = 404, e.body or "", ""
        else:
            status, text, retrieved_at = resp.status, resp.text, resp.retrieved_at
        if status == 404:
            if text.strip():
                raise HpoError(f"JAX HPO API: HTTP 404 with a body for {hpo_id} — not the empty-body absence: {text[:200]!r}")
            return None
        try:
            body = json.loads(text)
        except ValueError as e:
            raise HpoError(f"JAX HPO API: HTTP {status} with a non-JSON body for {hpo_id}: {text[:200]!r}") from e
        if not isinstance(body, dict) or body.get("id") != hpo_id:
            raise HpoError(f"JAX HPO API: the answer for {hpo_id} carries id {body.get('id') if isinstance(body, dict) else None!r}")
        return EvidenceRecord(
            record_id=f"{SOURCE}:{hpo_id}",
            source=SOURCE,
            source_version=self.version(),
            query={"id": hpo_id, "api": api},
            url=TERM_URL.format(id=hpo_id),
            retrieved_at=retrieved_at,
            payload=body,
        )

    def terms(self, ids: Iterable[str]) -> dict[str, EvidenceRecord | None]:
        """One :meth:`term` per id — validated, deduplicated, in sorted order."""
        wanted = sorted({valid_hpo_id(i) for i in ids})
        return {i: self.term(i) for i in wanted}

    def extract(self, record: EvidenceRecord | None) -> dict[str, str]:
        """Project one term onto :data:`COLUMNS` (strings; all '' for ``None``)."""
        if record is None:
            return {c: "" for c in COLUMNS}
        p = record.payload if isinstance(record.payload, dict) else {}
        count = p.get("descendantCount")
        return {
            "hpo_id": str(p.get("id") or ""),
            "label": str(p.get("name") or ""),
            "definition": str(p.get("definition") or ""),
            "synonyms": ";".join(str(s) for s in (p.get("synonyms") or [])),
            "descendant_count": "" if count is None else str(count),
        }


def valid_hpo_id(hpo_id: str) -> str:
    """``HP:nnnnnnn`` after ``strip().upper()``; ``ValueError`` otherwise — before any
    request, since the API reads a lower-case id as absent and an underscore as a 400."""
    s = str(hpo_id).strip().upper()
    if not HPO_ID.match(s):
        raise ValueError(f"not an HPO id (expected HP:0000000): {hpo_id!r}")
    return s


def label_of(record: EvidenceRecord) -> str:
    """The term's label — the API's ``name``."""
    return str(record.payload.get("name") or "") if isinstance(record.payload, dict) else ""


def synonyms_of(record: EvidenceRecord) -> list[str]:
    return [str(s) for s in (record.payload.get("synonyms") or [])] if isinstance(record.payload, dict) else []


def normalise_label(text: str) -> str:
    """The comparison form: lower-case, every run of non-alphanumerics one space,
    stripped — so ``Elevated sweat Cl-`` and ``elevated sweat cl`` compare equal."""
    return _NON_ALNUM.sub(" ", str(text).lower()).strip()


@dataclass
class CaseTerms:
    """What :func:`case_terms` did for one run."""

    records: list[EvidenceRecord] = field(default_factory=list)
    """One per case term the store or the API knows, sorted by id."""
    served: list[str] = field(default_factory=list)
    """Record ids found in the index before any request."""
    fetched: list[str] = field(default_factory=list)
    """Record ids fetched by this call and written into the store."""
    missing: list[str] = field(default_factory=list)
    """HP ids the API has no term for (404): visible, never silent."""


def case_terms(ids: Iterable[str], index: Any, store: EvidenceStore, http_factory: Callable[[], Http]) -> CaseTerms:
    """The case's term records, fetched once per run: for each id (validated,
    deduplicated, sorted) ``hpo:<id>`` is looked up in ``index`` — every store of the
    run the stage may read — and only the ids the index lacks trigger
    ``http_factory()`` (called at most once, lazily) and one request each; every
    record fetched is ``store.put``. A term already held is served from the store, so
    stage 6 fetches nothing stage 5 fetched and a rerun on a warm cache is
    byte-identical."""
    out = CaseTerms()
    retriever: HpoRetriever | None = None
    for hpo_id in sorted({valid_hpo_id(i) for i in ids}):
        rid = f"{SOURCE}:{hpo_id}"
        rec = index.get(rid)
        if rec is not None:
            out.records.append(rec)
            out.served.append(rid)
            continue
        if retriever is None:
            retriever = HpoRetriever(http_factory())
        rec = retriever.term(hpo_id)
        if rec is None:
            out.missing.append(hpo_id)
            continue
        store.put(rec)
        out.records.append(rec)
        out.fetched.append(rid)
    return out


def _host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc


def _pin_rate(http: Any, host: str, per_second: float) -> None:
    """Etiquette for ``host`` on the shared limiter — unless a rate is already set,
    since the orchestrator's table may be stricter for runs that share an IP."""
    limiter = getattr(http, "limiter", None)
    if limiter is not None:
        limiter.per_second.setdefault(host, per_second)
