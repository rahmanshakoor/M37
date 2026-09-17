"""The UniProt retriever over the recorded fixtures in ``tests/fixtures/uniprot/``
(public entries only: CFTR P13569, BUB1B O60566, a symbol UniProt does not know, a
404 and a 400). Every request is served by a stub ``Http`` keyed on the exact request;
nothing here talks to the network except the one test under ``ENGINE_LIVE_TESTS``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from engine.retrieve import http as http_mod
from engine.retrieve.http import Http, HttpCache, HttpError, RateLimiter, Response
from engine.retrieve.store import EvidenceRecord
from engine.retrieve.uniprot import (ACCESSION, API_URL, COLUMNS, ENTRY_URL, FIELDS, MAP_FEATURES, PROBE_QUERY, SEARCH_FIELDS,
                                     UNIPROT_PER_SECOND, UniprotError, UniprotRetriever, disease_names, disease_texts,
                                     feature_map, function_text, gene_of, length_of, natural_variants_at, protein_name,
                                     reference_residue, regions_at, release_of, residue_at, residue_of)

FIXTURES = Path(__file__).parent / "fixtures" / "uniprot"
RELEASE = "UniProt release 2026_03 (02-September-2026)"


class StubHttp:
    """Serves the recorded fixtures keyed by the exact request; refuses anything else.
    ``cache_ok=False`` (the live half of the version check) is served from the same
    recordings unless ``live`` overrides a request's headers."""

    def __init__(self, *dirs: Path, offline: bool = False):
        self.limiter = RateLimiter(default_per_second=0)
        self.offline = offline
        self.calls: list[tuple[str, str, Any, bool]] = []
        self.live_headers: dict[str, str] | None = None
        self._by_key: dict[str, dict] = {}
        for d in dirs:
            for p in sorted(Path(d).glob("*.json")):
                self.add(json.loads(p.read_text()))

    def add(self, fx: dict) -> None:
        r = fx["request"]
        self._by_key[HttpCache.key(r["method"], r["url"], r.get("params"), r.get("body"))] = fx

    def request(self, method: str, url: str, *, params=None, json_body=None, cache_404: bool = False,
                cache_ok: bool = True, **kw) -> Response:
        self.calls.append((method, url, params, cache_ok))
        fx = self._by_key.get(HttpCache.key(method, url, params, json_body))
        if fx is None:
            raise AssertionError(f"no fixture for {method} {url} {params}")
        text = fx["text"] if fx.get("body") is None else json.dumps(fx["body"], ensure_ascii=False)
        headers = dict(fx.get("headers", {}))
        if not cache_ok and self.live_headers is not None:
            headers.update(self.live_headers)
        if fx["status"] == 404 and cache_404:
            return Response(404, text, fx["retrieved_at"], False, "stub", headers)
        if fx["status"] >= 400:
            raise HttpError(fx["status"], url, text)
        return Response(fx["status"], text, fx["retrieved_at"], False, "stub", headers)

    def get(self, url: str, **kw) -> Response:
        return self.request("GET", url, **kw)


def stub(**kw: Any) -> StubHttp:
    return StubHttp(FIXTURES, **kw)


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


@pytest.fixture
def entry() -> EvidenceRecord:
    return UniprotRetriever(stub()).entry("P13569")


# ---------------------------------------------------------------------- version

def test_version_is_observed_from_the_probe_headers_and_pins_the_limiter():
    http = stub()
    r = UniprotRetriever(http)
    assert r.version() == RELEASE
    assert http.limiter.per_second["rest.uniprot.org"] == UNIPROT_PER_SECOND
    # the probe: a cached observation and a live one, both the fixed query that names no gene
    assert [(u, p["query"], ok) for _, u, p, ok in http.calls] == [
        (API_URL + "search", PROBE_QUERY, True), (API_URL + "search", PROBE_QUERY, False)]
    assert r.version() == RELEASE and len(http.calls) == 2  # cached on the instance


def test_version_refuses_a_release_boundary_and_a_response_without_the_headers():
    http = stub()
    http.live_headers = {"x-uniprot-release": "2026_04", "x-uniprot-release-date": "01-October-2026"}
    with pytest.raises(UniprotError, match="cache was filled under .*2026_03.* but the server now reports .*2026_04"):
        UniprotRetriever(http).version()
    offline = stub(offline=True)
    offline.live_headers = {"x-uniprot-release": "2026_04", "x-uniprot-release-date": "01-October-2026"}
    assert UniprotRetriever(offline).version() == RELEASE  # offline: the cached observation stands, no live check
    assert [ok for *_, ok in offline.calls] == [True]
    with pytest.raises(UniprotError, match="carries no x-uniprot-release"):
        release_of(Response(200, "{}", "t", False, "k", {"content-type": "application/json"}))


def test_a_data_response_under_another_release_is_refused():
    http = stub()
    r = UniprotRetriever(http)
    r.version()
    fx = fixture("search_CFTR")
    fx["headers"] = {**fx["headers"], "x-uniprot-release": "2026_04"}
    http.add(fx)
    with pytest.raises(UniprotError, match="release changed mid-run"):
        r.accession_for_symbol("CFTR")


def test_a_transport_that_drops_the_release_headers_is_refused_by_name():
    """UniProt publishes its release only in response headers, and the shared ``Http``
    keeps only the headers on its ``_KEEP`` tuple — which does not yet name them (the
    integrator's request 1). Until it does, a live run fails loudly here rather than
    recording entries under an unobserved release."""
    http = stub()
    for fx in list(http._by_key.values()):
        fx["headers"] = {"content-type": "application/json"}
    with pytest.raises(UniprotError, match="carries no x-uniprot-release / x-uniprot-release-date header"):
        UniprotRetriever(http).version()


# ----------------------------------------------------------------------- lookup

def test_accession_for_symbol_and_absence():
    http = stub()
    r = UniprotRetriever(http)
    assert r.accession_for_symbol("CFTR") == "P13569"
    assert r.accession_for_symbol("cftr") == "P13569"  # upper-cased before asking
    assert r.accession_for_symbol("BUB1B") == "O60566"
    assert r.accession_for_symbol("NOTAGENE1") is None
    sent = [p for _, u, p, _ in http.calls if u == API_URL + "search" and p["query"] != PROBE_QUERY]
    assert sent[0] == {"query": "(gene_exact:CFTR) AND (organism_id:9606) AND (reviewed:true)", "format": "json", "fields": SEARCH_FIELDS}
    with pytest.raises(ValueError, match="not a gene symbol"):
        r.accession_for_symbol("7:117559590")
    with pytest.raises(ValueError, match="not a gene symbol"):
        r.accession_for_symbol("")


def test_ambiguity_and_a_synonym_match_raise():
    http = stub()
    two = fixture("search_CFTR")
    other = json.loads(json.dumps(two["body"]["results"][0]))
    other["primaryAccession"] = "O60566"
    two["body"]["results"].append(other)
    http.add(two)
    with pytest.raises(UniprotError, match="names 2 reviewed human entries .*P13569, O60566.*refusing"):
        UniprotRetriever(http).accession_for_symbol("CFTR")
    http = stub()
    synonym = fixture("search_CFTR")
    synonym["request"]["params"]["query"] = "(gene_exact:ABCC7) AND (organism_id:9606) AND (reviewed:true)"
    http.add(synonym)
    with pytest.raises(UniprotError, match="none with the symbol asked as its primary gene name"):
        UniprotRetriever(http).accession_for_symbol("ABCC7")


def test_entry_record_shape(entry: EvidenceRecord):
    assert entry.record_id == "uniprot:P13569" and entry.source == "uniprot"
    assert entry.source_version == RELEASE
    assert entry.url == ENTRY_URL.format(acc="P13569") == "https://www.uniprot.org/uniprotkb/P13569/entry"
    assert entry.query == {"accession": "P13569", "fields": FIELDS, "api": API_URL + "P13569.json?fields=" + FIELDS.replace(",", "%2C")}
    assert entry.retrieved_at == fixture("entry_P13569")["retrieved_at"]
    assert entry.payload == fixture("entry_P13569")["body"]  # verbatim
    assert entry.payload["primaryAccession"] == "P13569" and entry.payload["uniProtkbId"] == "CFTR_HUMAN"
    assert entry.payload["sequence"]["length"] == 1480 and len(entry.payload["sequence"]["value"]) == 1480
    assert set(entry.payload) == {"comments", "entryAudit", "entryType", "extraAttributes", "features", "genes", "keywords",
                                  "primaryAccession", "proteinDescription", "sequence", "uniProtKBCrossReferences", "uniProtkbId"}


def test_entry_refuses_a_404_an_inactive_entry_a_redirect_and_a_bad_accession():
    http = stub()
    r = UniprotRetriever(http)
    with pytest.raises(UniprotError, match="404 Resource not found.*vanished"):
        r.entry("Q00000")
    assert fixture("entry_Q00000_404")["body"] == {"url": "http://rest.uniprot.org/uniprotkb/Q00000", "messages": ["Resource not found"]}
    odd = fixture("entry_Q00000_404")
    odd["body"] = {"messages": ["something else"]}
    http.add(odd)
    with pytest.raises(UniprotError, match="not UniProt's 'Resource not found' shape"):
        r.entry("Q00000")
    inactive = fixture("entry_P13569")
    inactive["request"]["params"] = {"fields": FIELDS}
    inactive["request"]["url"] = API_URL + "A9A9A9.json"
    inactive["body"] = {"entryType": "Inactive", "primaryAccession": "A9A9A9", "inactiveReason": {"inactiveReasonType": "DELETED"}}
    http.add(inactive)
    with pytest.raises(UniprotError, match="inactive \\(DELETED\\)"):
        r.entry("A9A9A9")
    redirect = fixture("entry_P13569")
    redirect["request"]["url"] = API_URL + "Q2M3G0.json"  # a well-formed accession answered with another entry
    http.add(redirect)
    with pytest.raises(UniprotError, match="answered for 'P13569'"):
        r.entry("Q2M3G0")
    before = len(http.calls)
    with pytest.raises(ValueError, match="not a UniProtKB accession"):
        r.entry("notanaccession")
    assert len(http.calls) == before  # refused before any request
    assert fixture("entry_notanaccession_400")["status"] == 400
    assert "invalid format" in fixture("entry_notanaccession_400")["body"]["messages"][0]
    for acc in ("P13569", "O60566", "A0A0C5B5G6", "Q9UDN9"):
        assert ACCESSION.match(acc), acc
    for bad in ("P1356", "p13569", "CFTR", "ENSP00000003084", "VCV000007105", "PF00005"):
        assert not ACCESSION.match(bad), bad


# --------------------------------------------------------------------- helpers

def test_feature_map_holds_every_map_feature_in_position_order(entry: EvidenceRecord):
    fm = feature_map(entry)
    assert [f["start"] for f in fm] == sorted(f["start"] for f in fm)
    counts = {t: sum(1 for f in fm if f["type"] == t) for t in MAP_FEATURES}
    assert counts == {"Domain": 4, "Region": 3, "Motif": 1, "Topological domain": 13, "Transmembrane": 12,
                      "Binding site": 6, "Active site": 0, "Site": 0}
    assert {"type": "Domain", "start": 423, "end": 646, "description": "ABC transporter 1"} in fm
    assert {"type": "Motif", "start": 1478, "end": 1480, "description": "PDZ-binding"} in fm
    assert all(set(f) == {"type", "start", "end", "description"} for f in fm)


def test_regions_and_natural_variants_at_the_public_residues(entry: EvidenceRecord):
    assert regions_at(entry, 508) == ["Topological domain: Cytoplasmic (359–858)", "Domain: ABC transporter 1 (423–646)"]
    assert regions_at(entry, 542) == regions_at(entry, 508)
    assert regions_at(entry, 5000) == [] and natural_variants_at(entry, 5000) == []
    v508 = natural_variants_at(entry, 508)
    assert len(v508) == 2
    assert v508[0].startswith("VAR_000171 F→del: in CF and CBAVD; most common mutation in Caucasian CF chromosomes; ")
    assert v508[0].endswith("… (20 UniProt evidence references)")
    assert v508[1] == "VAR_000172 F→C: in dbSNP:rs74571530 (1 UniProt evidence reference)"
    assert natural_variants_at(entry, 542) == ["VAR_080305 G542–L1480→del: in CF (1 UniProt evidence reference)"]
    for line in v508 + natural_variants_at(entry, 542):
        assert "PubMed" not in line and len(line.split(": ", 1)[1].rsplit(" (", 1)[0]) <= 121
    assert residue_at(entry, 508) == "F" and residue_at(entry, 542) == "G" and residue_at(entry, 0) is None and residue_at(entry, 1481) is None


def test_function_disease_and_names(entry: EvidenceRecord):
    assert disease_names(entry) == ["Cystic fibrosis", "Congenital bilateral absence of the vas deferens"]
    assert disease_texts(entry)[0].startswith("Cystic fibrosis: A common generalized disorder of the exocrine glands")
    text = function_text(entry)
    assert text.startswith("Epithelial ion channel that plays an important role") and "PubMed" not in text
    assert "(PubMed:" in entry.payload["comments"][0]["texts"][0]["value"]  # stripped from the text, kept in the record
    assert protein_name(entry) == "Cystic fibrosis transmembrane conductance regulator"
    assert gene_of(entry) == "CFTR" and length_of(entry) == 1480


@pytest.mark.parametrize("hgvsp, residue, ref", [
    ("ENSP00000003084.6:p.Phe508del", 508, "F"),
    ("p.Gly542Ter", 542, "G"),
    ("p.Met1?", 1, "M"),
    ("ENSP00000269305.4:p.Arg175His", 175, "R"),
    ("p.(Arg175His)", 175, "R"),
    ("p.?", None, None),
    ("-", None, None),
    ("", None, None),
    (None, None, None),
])
def test_residue_of(hgvsp, residue, ref):
    assert residue_of(hgvsp) == residue
    assert reference_residue(hgvsp) == ref


def test_extract_and_params(entry: EvidenceRecord):
    r = UniprotRetriever(stub())
    cols = r.extract(entry)
    assert list(cols) == list(COLUMNS)
    assert cols["accession"] == "P13569" and cols["entry_name"] == "CFTR_HUMAN" and cols["gene"] == "CFTR"
    assert cols["length"] == "1480" and cols["release"] == RELEASE and cols["entry_version"] == "286"
    assert cols["n_features"] == "39" and cols["n_natural_variants"] == "209"
    assert cols["diseases"] == "Cystic fibrosis;Congenital bilateral absence of the vas deferens"
    assert r.extract(None) == {c: "" for c in COLUMNS}
    p = r.params
    assert p["api_url"] == API_URL and p["fields"] == FIELDS and p["per_second"] == 3.0
    assert p["map_features"] == list(MAP_FEATURES) and p["release_headers"] == ["x-uniprot-release", "x-uniprot-release-date"]
    assert "<SYMBOL>" in p["search_query"]


def test_fixtures_are_recorded_with_the_headers_and_the_date():
    for name in ("probe_release", "search_CFTR", "search_NOTAGENE1", "search_BUB1B", "entry_P13569", "entry_Q00000_404",
                 "entry_notanaccession_400"):
        fx = fixture(name)
        assert fx["headers"]["x-uniprot-release"] == "2026_03" and fx["headers"]["x-uniprot-release-date"] == "02-September-2026"
        assert fx["retrieved_at"].startswith("2026-09-17")
    assert fixture("search_CFTR")["headers"]["x-total-results"] == "1"
    assert fixture("search_NOTAGENE1")["headers"]["x-total-results"] == "0" and fixture("search_NOTAGENE1")["body"] == {"results": []}
    assert fixture("search_BUB1B")["body"]["results"][0]["uniProtkbId"] == "BUB1B_HUMAN"
    assert fixture("search_BUB1B")["body"]["results"][0]["sequence"]["length"] == 1050
    readme = (FIXTURES / "README.md").read_text()
    assert "2026-09-17" in readme and "2026_03" in readme


# ------------------------------------------------------------------------- live

@pytest.mark.skipif(not os.environ.get("ENGINE_LIVE_TESTS"), reason="set ENGINE_LIVE_TESTS=1 to talk to UniProt")
def test_live_cftr_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """One live fetch of the public CFTR entry through the shared ``Http``. The release
    headers must survive its header filter (``_KEEP``); until the integrator adds
    them, this test extends the tuple itself."""
    monkeypatch.setattr(http_mod, "_KEEP", tuple(http_mod._KEEP) + ("x-uniprot-release", "x-uniprot-release-date", "x-total-results"))
    http = Http(HttpCache(tmp_path / "cache"), limiter=RateLimiter())
    r = UniprotRetriever(http)
    assert r.version().startswith("UniProt release 20")
    assert r.accession_for_symbol("CFTR") == "P13569"
    rec = r.entry("P13569")
    assert rec.payload["primaryAccession"] == "P13569" and rec.payload["sequence"]["length"] == 1480
    assert "Domain: ABC transporter 1 (423–646)" in regions_at(rec, 508)
    assert any(v.startswith("VAR_000171 F→del") for v in natural_variants_at(rec, 508))
    assert r.accession_for_symbol("NOTAGENE1") is None
