"""The HPO term retriever (``engine.retrieve.hpo``) over the recorded JAX answers in
``tests/fixtures/hpo/requests`` — public ids only (the five cystic-fibrosis terms of the
public demo case and one id the ontology lacks). The stub ``Http`` serves the exact
request the retriever makes and returns a recorded 404 the way the real ``Http`` does
with ``cache_404``; one live test talks to the JAX API when ``ENGINE_LIVE_TESTS`` is set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from engine.agents.validator import EvidenceIndex
from engine.retrieve import hpo
from engine.retrieve.hpo import (
    API_URL, COLUMNS, JAX_PER_SECOND, TERM_URL, CaseTerms, HpoError, HpoRetriever, case_terms, label_of, normalise_label,
    synonyms_of, valid_hpo_id,
)
from engine.retrieve.http import Http, HttpCache, HttpError, RateLimiter, Response
from engine.retrieve.store import EvidenceRecord, EvidenceStore

FIXTURES = Path(__file__).parent / "fixtures" / "hpo"
PUBLIC = ["HP:0012236", "HP:0001738", "HP:0002205", "HP:0006528", "HP:0002110"]
ABSENT = "HP:9999999"


def fixture(hpo_id: str) -> dict:
    return json.loads((FIXTURES / "requests" / f"term_{hpo_id.replace(':', '_')}.json").read_text())


class StubHttp:
    """Serves the recorded answers keyed by the exact request; refuses anything else.
    A recorded 404 is returned as a ``Response`` when the caller passed ``cache_404``
    (as ``Http`` does) and raised as ``HttpError`` otherwise."""

    def __init__(self, *fixtures: dict):
        self.limiter = RateLimiter(default_per_second=0)
        self.calls: list[tuple[str, str]] = []
        self._by_key: dict[str, dict] = {}
        for fx in fixtures:
            r = fx["request"]
            self._by_key[HttpCache.key(r["method"], r["url"], r["params"], r["body"])] = fx

    def request(self, method: str, url: str, *, params=None, json_body=None, cache_404: bool = False, **kw: Any) -> Response:
        self.calls.append((method, url))
        fx = self._by_key.get(HttpCache.key(method, url, params, json_body))
        if fx is None:
            raise AssertionError(f"no fixture for {method} {url}")
        text = fx["text"] if fx.get("body") is None else json.dumps(fx["body"], ensure_ascii=False)
        if fx["status"] == 404 and cache_404:
            return Response(404, text, fx["retrieved_at"], False, "stub", fx.get("headers", {}))
        if fx["status"] >= 400:
            raise HttpError(fx["status"], url, text)
        return Response(fx["status"], text, fx["retrieved_at"], False, "stub", fx.get("headers", {}))

    def get(self, url: str, **kw: Any) -> Response:
        return self.request("GET", url, **kw)


def stub(*ids: str) -> StubHttp:
    return StubHttp(*(fixture(i) for i in ids))


# ------------------------------------------------------------------------- term()

def test_term_builds_the_record_from_the_recorded_answer():
    http = stub("HP:0012236")
    r = HpoRetriever(http)
    rec = r.term("HP:0012236")
    assert rec is not None and rec.record_id == "hpo:HP:0012236" and rec.source == "hpo"
    assert rec.source_version == "JAX ontology API hp (the service exposes no release identifier)" == r.version()
    assert rec.query == {"id": "HP:0012236", "api": "https://ontology.jax.org/api/hp/terms/HP:0012236"}
    assert rec.url == "https://hpo.jax.org/browse/term/HP:0012236" == TERM_URL.format(id="HP:0012236")
    assert rec.retrieved_at == fixture("HP:0012236")["retrieved_at"] and "2026-" in rec.retrieved_at
    # the payload is the JAX object verbatim
    assert rec.payload == fixture("HP:0012236")["body"]
    assert set(rec.payload) == {"id", "name", "definition", "comment", "descendantCount", "synonyms", "xrefs",
                                "publicationReferences", "translations"}
    assert rec.payload["name"] == "Elevated sweat chloride" and rec.payload["definition"] == "An increased concentration of chloride in the sweat."
    assert rec.payload["synonyms"] == ["Elevated sweat Cl", "Elevated sweat Cl-", "Elevated sweat chloride"] and rec.payload["xrefs"] == ["UMLS:C1856646"]
    assert http.calls == [("GET", "https://ontology.jax.org/api/hp/terms/HP:0012236")]
    assert label_of(rec) == "Elevated sweat chloride" and synonyms_of(rec) == rec.payload["synonyms"]
    # the id is normalised before the request: the same record, one more call
    assert r.term(" hp:0012236 ").record_id == "hpo:HP:0012236" and len(http.calls) == 2
    # the limiter is pinned for the host, never above a rate the orchestrator set
    assert http.limiter.per_second == {"ontology.jax.org": JAX_PER_SECOND} and JAX_PER_SECOND == 3.0
    strict = StubHttp(fixture("HP:0012236"))
    strict.limiter.per_second["ontology.jax.org"] = 1.0
    HpoRetriever(strict)
    assert strict.limiter.per_second["ontology.jax.org"] == 1.0


def test_absence_is_the_empty_404_and_nothing_else():
    http = stub(ABSENT)
    assert HpoRetriever(http).term(ABSENT) is None
    assert http.calls == [("GET", f"https://ontology.jax.org/api/hp/terms/{ABSENT}")]
    # a transport that raises on the 404 instead of returning it is read the same way
    raising = StubHttp(fixture(ABSENT))
    raising.request = lambda method, url, **kw: (_ for _ in ()).throw(HttpError(404, url, ""))  # type: ignore[assignment]
    assert HpoRetriever(raising).term(ABSENT) is None
    # a 404 with a body is not absence
    with_body = dict(fixture(ABSENT), text='{"message": "not found"}')
    with pytest.raises(HpoError, match="404 with a body"):
        HpoRetriever(StubHttp(with_body)).term(ABSENT)
    # a non-JSON 200, or a term object for a different id, cannot be cited
    junk = dict(fixture("HP:0012236"), body=None, text="<html>maintenance</html>")
    with pytest.raises(HpoError, match="non-JSON"):
        HpoRetriever(StubHttp(junk)).term("HP:0012236")
    other = dict(fixture("HP:0012236"), body=dict(fixture("HP:0012236")["body"], id="HP:0012237"))
    with pytest.raises(HpoError, match="carries id 'HP:0012237'"):
        HpoRetriever(StubHttp(other)).term("HP:0012236")
    # a server error is the transport's error, not absence
    broken = dict(fixture("HP:0012236"), status=503, body=None, text="unavailable")
    with pytest.raises(HttpError, match="HTTP 503"):
        HpoRetriever(StubHttp(broken)).term("HP:0012236")


def test_malformed_ids_are_refused_before_any_request(tmp_path: Path):
    http = StubHttp()
    r = HpoRetriever(http)
    for bad in ("hp:12", "HP_0012236", "HP:12", "HP:00122360", "0012236", "", "HP: 0012236", "hp_0012236"):
        with pytest.raises(ValueError, match="not an HPO id"):
            r.term(bad)
    with pytest.raises(ValueError):
        r.terms(["HP:0012236", "HP_0012236"])
    with pytest.raises(ValueError):
        case_terms(["HP:12"], EvidenceIndex(), EvidenceStore(tmp_path / "evidence"), http_factory=lambda: http)  # type: ignore[arg-type]
    assert http.calls == []
    assert valid_hpo_id(" hp:0012236\n") == "HP:0012236"


def test_terms_deduplicates_and_sorts_and_extract_projects_the_columns():
    http = stub("HP:0012236", "HP:0002205", ABSENT)
    r = HpoRetriever(http)
    found = r.terms(["HP:0012236", "hp:0002205", ABSENT, "HP:0012236"])
    assert list(found) == ["HP:0002205", "HP:0012236", ABSENT] and found[ABSENT] is None
    assert [u for _, u in http.calls] == [f"{API_URL}HP:0002205", f"{API_URL}HP:0012236", f"{API_URL}{ABSENT}"]
    assert r.columns == COLUMNS == ("hpo_id", "label", "definition", "synonyms", "descendant_count")
    assert r.extract(found["HP:0002205"]) == {
        "hpo_id": "HP:0002205", "label": "Recurrent respiratory infections",
        "definition": "An increased susceptibility to respiratory infections as manifested by a history of recurrent respiratory infections.",
        "synonyms": "Frequent respiratory infections;Multiple respiratory infections;Recurrent respiratory infections;"
                    "Susceptibility to respiratory infections;respiratory infections, recurrent",
        "descendant_count": "16",
    }
    assert r.extract(found["HP:0012236"])["descendant_count"] == "0"
    assert r.extract(None) == {c: "" for c in COLUMNS}


def test_params_and_version_say_the_api_is_unversioned():
    r = HpoRetriever(StubHttp(), base_url="https://ontology.jax.org/api/hp/terms")
    assert r.params == {"api_url": "https://ontology.jax.org/api/hp/terms/", "per_second": 3.0,
                        "version_note": "the JAX API exposes no HPO release; records are dated by retrieved_at only"}
    assert r.version() == "JAX ontology API hp (the service exposes no release identifier)"
    assert "no release" in r.version() and hpo.VERSION_NOTE == r.params["version_note"]


def test_normalise_label_is_the_comparison_form():
    assert normalise_label("Elevated sweat Cl-") == "elevated sweat cl"
    assert normalise_label("  Recurrent respiratory infections. ") == "recurrent respiratory infections"
    assert normalise_label("respiratory infections, recurrent") == "respiratory infections recurrent"
    assert normalise_label("Chronic  lung—disease") == "chronic lung disease" and normalise_label("") == ""


# --------------------------------------------------------------------- case_terms

def test_case_terms_serves_from_the_index_and_fetches_only_the_missing(tmp_path: Path):
    """Two terms already in a store of the run (stage 2's, here), one to fetch, one the
    API lacks: the factory is called once, one request per missing id, every fetched
    record written, and a second call over the same stores fetches nothing."""
    earlier = EvidenceStore(tmp_path / "02_retrieve" / "evidence")
    for hpo_id in ("HP:0012236", "HP:0002205"):
        earlier.put(HpoRetriever(stub(hpo_id)).term(hpo_id))
    stage = EvidenceStore(tmp_path / "05_reason" / "evidence")
    index = EvidenceIndex([earlier, stage])
    http = stub("HP:0006528", ABSENT)
    built: list[StubHttp] = []

    def factory() -> StubHttp:
        built.append(http)
        return http

    out = case_terms(["HP:0006528", "HP:0012236", ABSENT, "hp:0002205", "HP:0012236"], index, stage, http_factory=factory)
    assert isinstance(out, CaseTerms) and len(built) == 1
    assert [r.record_id for r in out.records] == ["hpo:HP:0002205", "hpo:HP:0006528", "hpo:HP:0012236"]
    assert out.served == ["hpo:HP:0002205", "hpo:HP:0012236"] and out.fetched == ["hpo:HP:0006528"] and out.missing == [ABSENT]
    assert http.calls == [("GET", f"{API_URL}HP:0006528"), ("GET", f"{API_URL}{ABSENT}")]
    assert [r.record_id for r in stage.iter()] == ["hpo:HP:0006528"] and not earlier.exists("hpo:HP:0006528")
    assert out.records[0] is not None and out.records[0].payload["name"] == "Recurrent respiratory infections"
    written = stage.path_for("hpo:HP:0006528").read_bytes()
    assert written == (FIXTURES / "records" / "hpo" / stage.path_for("hpo:HP:0006528").name).read_bytes()  # the fixture's bytes

    def never() -> StubHttp:
        raise AssertionError("every term is held: no Http should be built")
    again = case_terms(["HP:0006528", "HP:0012236", "HP:0002205"], index, stage, http_factory=never)
    assert again.served == ["hpo:HP:0002205", "hpo:HP:0006528", "hpo:HP:0012236"] and again.fetched == [] and again.missing == []
    assert [r.record_id for r in again.records] == [r.record_id for r in out.records]
    assert stage.path_for("hpo:HP:0006528").read_bytes() == written
    assert case_terms([], index, stage, http_factory=never) == CaseTerms()


def test_the_record_fixtures_are_what_the_retriever_writes():
    """``tests/fixtures/hpo/records`` (and the public case's copy) seed run directories;
    each file must be the record the retriever builds from the recorded answer."""
    store = EvidenceStore(FIXTURES / "records")
    ids = sorted(r.record_id for r in store.iter())
    assert ids == sorted(f"hpo:{t}" for t in PUBLIC)
    for hpo_id in PUBLIC:
        rec = HpoRetriever(stub(hpo_id)).term(hpo_id)
        assert rec is not None and store.get(f"hpo:{hpo_id}") == rec
        assert store.path_for(rec.record_id).read_text() == rec.to_json()
    public = EvidenceStore(Path(__file__).parent / "fixtures" / "public_case" / "05_reason" / "evidence")
    assert {r.record_id: r for r in public.iter("hpo")} == {r.record_id: r for r in store.iter("hpo")}
    index = json.loads((public.root / "index.json").read_text())
    assert sorted(index) == ids and all(index[i]["source"] == "hpo" and index[i]["path"].startswith("hpo/") for i in ids)
    assert not any(p.suffix == ".json" for p in (FIXTURES / "requests").iterdir() if "HP_" not in p.name)


# ------------------------------------------------------------------------------ live

@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ENGINE_LIVE_TESTS"), reason="set ENGINE_LIVE_TESTS=1 to hit the JAX ontology API")
def test_live_term_and_absence(tmp_path: Path):
    """Public ids only: HP:0012236 exists, HP:9999999 does not."""
    http = Http(HttpCache(tmp_path / "cache"), limiter=RateLimiter())
    r = HpoRetriever(http)
    rec = r.term("HP:0012236")
    assert rec is not None and rec.payload["id"] == "HP:0012236" and rec.payload["name"] == "Elevated sweat chloride"
    assert rec.url == "https://hpo.jax.org/browse/term/HP:0012236" and "synonyms" in rec.payload
    assert r.term(ABSENT) is None and http.live_requests == 2
    assert r.term(ABSENT) is None and http.live_requests == 2  # the empty 404 is cached
