"""ChEMBL retriever against recorded real responses (public entities only: the CFTR,
BUB1B and MTHFR gene symbols; ivacaftor CHEMBL2010601, lumacaftor CHEMBL2103870 and the
other CFTR-mechanism molecules; rofecoxib CHEMBL122 for the withdrawal path; phentermine
hydrochloride CHEMBL1200912 and its parent phentermine CHEMBL1574 for the salt-form path;
and the unknown id CHEMBL999999999).

Every fixture in ``tests/fixtures/chembl/`` is one of the engine's own HTTP cache entries,
byte-for-byte as :class:`engine.retrieve.http.HttpCache` wrote it — ``request`` (method,
URL, params, body), ``status``, ``headers``, ``retrieved_at`` and the served ``text``
verbatim — from a live request the retriever itself made on 2026-09-13 (UTC) against
ChEMBL_37. So the stub Http serves a fixture only when the retriever asks for exactly that
request and refuses anything else, and the same files, filed under the client's own key,
replay through the real ``Http(offline=True)`` unchanged
(``test_fixtures_are_cache_entries_the_real_client_replays_offline``). The shapes ChEMBL
delivers with HTTP 200 that must be refused (an ignored filter, a row for another id, a
short page, a row served twice) and the code paths real public data does not reach with a
small fixture set (a parent absent from the mechanism list, a complex-level target) are
exercised by swapping a fixture's text, each edit labelled; nothing synthetic is ever sent.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
from pathlib import Path

import pytest

from engine.medicine.chembl import (ACCEPTED_RELATIONSHIPS, API_URL, COLUMNS, DEFAULT_CHEMBL_SEARCH, MAX_CHEMBL_SEARCH,
                                    MAX_PAGE_SIZE, MAX_TARGETS_PER_SYMBOL, MIN_NEEDLE_CHARS, SEARCH_FIELDS,
                                    ChemblError, ChemblRetriever, drug_rows, gene_symbols, parent_id, phase_label,
                                    symbol_relationships, valid_chembl_id, valid_needle, valid_symbol, warning_summary)
from engine.retrieve.http import Http, HttpCache, HttpError, RateLimiter, Response
from engine.retrieve.store import EvidenceRecord, EvidenceStore

FIXTURES = Path(__file__).parent / "fixtures" / "chembl"
VERSION = "ChEMBL_37 (2026-05-01)"
IVACAFTOR, LUMACAFTOR, DEUTIVACAFTOR, CROFELEMER = "CHEMBL2010601", "CHEMBL2103870", "CHEMBL4297603", "CHEMBL2108184"
ROFECOXIB, UNKNOWN = "CHEMBL122", "CHEMBL999999999"
PHENTERMINE_HCL, PHENTERMINE = "CHEMBL1200912", "CHEMBL1574"
CFTR_TARGET, CFTR_PPI_TARGET, BUB1B_TARGET = "CHEMBL4051", "CHEMBL3885559", "CHEMBL4295998"
CFTR_MEC_IDS = [964, 965, 2346, 5165, 5167, 7457, 8823, 8905, 9140, 9263, 9363, 9954, 9955]
CFTR_MOLECULES = ["CHEMBL2010601", "CHEMBL2103870", "CHEMBL2108184", "CHEMBL3544914", "CHEMBL4101487",
                  "CHEMBL4297392", "CHEMBL4297603", "CHEMBL4297649", "CHEMBL4297849", "CHEMBL4298128",
                  "CHEMBL4650318", "CHEMBL4802150", "CHEMBL5314934"]
CFTR_FAMILIES = [c for c in CFTR_MOLECULES if c != DEUTIVACAFTOR]  # deutivacaftor is filed under ivacaftor
MECHANISM_SEARCH = "search_mechanism_conductance_regulator_limit3"
INDICATION_SEARCH = "search_indication_cystic_fibrosis_limit3"
"""The two text searches recorded live on 2026-09-17 (``limit=3``): the curated
mechanism table for ``conductance regulator`` (13 rows in all) and the drug-indication
table for ``cystic fibrosis`` (135), with the molecules their rows name."""
ATALUREN, LIPROTAMASE = "CHEMBL256997", "CHEMBL2108703"
CFTR_INDICATION_IDS = [25197, 26648, 28071, 54198, 56193, 56194, 56195, 56375, 116606, 121645, 125577, 137003, 137004,
                       137105, 141838, 142771, 142780, 142818, 143021, 143203, 143501, 143643, 147820, 148245, 149451,
                       155729, 156570]


def fixture(name: str) -> dict:
    """The cache entry: ``request``, ``status``, ``headers``, ``retrieved_at``, ``text``."""
    return json.loads((FIXTURES / f"{name}.json").read_text())


def body_of(name: str) -> dict:
    """The served text, parsed."""
    return json.loads(fixture(name)["text"])


def served(fx: dict, *, body=None, text: str | None = None, status: int | None = None) -> dict:
    """A copy of a recorded fixture with its text/status swapped — for shapes that must
    be served in reply to a request the retriever actually makes."""
    out = copy.deepcopy(fx)
    if body is not None:
        out["text"] = json.dumps(body, ensure_ascii=False)
    if text is not None:
        out["text"] = text
    if status is not None:
        out["status"] = status
    return out


class StubHttp:
    """Serves recorded responses keyed by the exact request (method, url, params, body),
    honouring ``cache_404`` the way :class:`engine.retrieve.http.Http` does."""

    def __init__(self, *fixtures: dict, offline: bool = False):
        self.limiter = RateLimiter(default_per_second=0)
        self.offline = offline
        self.calls: list[dict] = []
        self._by_key: dict[str, dict] = {}
        for fx in fixtures:
            self.add(fx)

    def add(self, fx: dict) -> None:
        r = fx["request"]
        self._by_key[HttpCache.key(r["method"], r["url"], r["params"], r["body"])] = fx

    def request(self, method: str, url: str, *, params=None, json_body=None, **kw) -> Response:
        self.calls.append({"method": method, "url": url, "params": params, **kw})
        fx = self._by_key.get(HttpCache.key(method, url, params, json_body))
        if fx is None:
            raise AssertionError(f"no fixture for {method} {url} {params}")
        if fx["status"] == 404 and kw.get("cache_404"):
            return Response(404, fx["text"], fx["retrieved_at"], False, "stub", fx["headers"])
        if fx["status"] >= 400:
            raise HttpError(fx["status"], url, fx["text"])
        return Response(fx["status"], fx["text"], fx["retrieved_at"], False, "stub", fx["headers"])

    def get(self, url: str, **kw) -> Response:
        return self.request("GET", url, **kw)

    def post(self, url: str, json_body, **kw) -> Response:
        return self.request("POST", url, json_body=json_body, **kw)


def _all_fixtures() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(FIXTURES.glob("*.json"))]


def _stub(*names: str, offline: bool = False) -> StubHttp:
    return StubHttp(fixture("status"), *(fixture(n) for n in names), offline=offline)


def _cftr_stub(*replacements: dict) -> StubHttp:
    """Every fixture, with ``replacements`` (edited copies) taking the place of the
    recorded entry for the same request."""
    stub = StubHttp(*_all_fixtures())
    for fx in replacements:
        stub.add(fx)
    return stub


def _cache_from_fixtures(root: Path) -> HttpCache:
    """File every fixture under the key the real client computes for its request. Each
    file :meth:`HttpCache.put` writes is then byte-identical to the fixture it came from —
    which is what makes a fixture a cache entry rather than a description of one."""
    cache = HttpCache(root)
    for path in sorted(FIXTURES.glob("*.json")):
        fx = json.loads(path.read_text())
        r = fx["request"]
        key = HttpCache.key(r["method"], r["url"], r["params"], r["body"])
        cache.put(key, Response(fx["status"], fx["text"], fx["retrieved_at"], False, key, fx["headers"]), r)
    assert sorted(q.read_bytes() for q in root.rglob("*.json")) == sorted(q.read_bytes() for q in FIXTURES.glob("*.json"))
    return cache


def _paths(stub: StubHttp) -> list[str]:
    """Endpoints asked, in order, without the two status.json calls the first data request triggers."""
    return [p for p in (c["url"].removeprefix(API_URL) for c in stub.calls) if p != "status.json"]


def _call(stub: StubHttp, endpoint: str) -> dict:
    """The last call to ``endpoint`` (``molecule/CHEMBL2010601.json``, ``mechanism.json``, ...)."""
    return [c for c in stub.calls if c["url"] == API_URL + endpoint][-1]


def _list_params(**filters) -> dict:
    """The exact query a one-page list walk sends: the filters, the primary-key order, the page."""
    pk = {"mechanism.json": "mec_id", "drug_indication.json": "drugind_id", "drug_warning.json": "warning_id",
          "target.json": "target_chembl_id"}[filters.pop("_endpoint")]
    return {**filters, "order_by": pk, "limit": MAX_PAGE_SIZE, "offset": 0}


# ---------------------------------------------------------------- version


def test_version_is_the_release_and_date_from_status_json():
    stub = _stub()
    r = ChemblRetriever(stub)
    assert r.version() == VERSION == r.version()
    # asked once from the cache and once live to detect a release boundary; then remembered
    assert [(c["url"], c.get("cache_ok", True)) for c in stub.calls] == [(API_URL + "status.json", True), (API_URL + "status.json", False)]
    offline = _stub(offline=True)
    assert ChemblRetriever(offline).version() == VERSION and len(offline.calls) == 1


class _Drift(StubHttp):
    """The cache holds ChEMBL_37's status.json; the server now answers ChEMBL_38."""

    def request(self, method, url, *, params=None, json_body=None, **kw):
        resp = super().request(method, url, params=params, json_body=json_body, **kw)
        if url.endswith("status.json") and kw.get("cache_ok", True) is False:
            return Response(200, '{"chembl_db_version": "ChEMBL_38", "chembl_release_date": "2026-11-01", "status": "UP"}', resp.retrieved_at, False, "live", {})
        return resp


def test_version_mismatch_between_cache_and_server_raises():
    with pytest.raises(ChemblError, match="ChEMBL_38.*delete the HTTP cache"):
        ChemblRetriever(_Drift(fixture("status"))).version()
    with pytest.raises(ChemblError, match="no release"):
        ChemblRetriever(StubHttp(served(fixture("status"), body={"status": "UP"}))).version()


def test_release_cross_check_runs_before_any_data_request_is_cached():
    # a page served under the new release must never be cached under the old one, so the
    # status cross-check is the first request of every fetching method — and the only one made
    for call in (lambda r: r.targets_for_symbol("CFTR"), lambda r: r.molecule(IVACAFTOR), lambda r: r.mechanisms(IVACAFTOR),
                 lambda r: r.mechanisms_for_target(CFTR_TARGET), lambda r: r.indications(IVACAFTOR), lambda r: r.warnings(IVACAFTOR),
                 lambda r: r.drugs_for_target_symbol("CFTR")):
        stub = _Drift(*_all_fixtures())
        with pytest.raises(ChemblError, match="ChEMBL_38"):
            call(ChemblRetriever(stub))
        assert [c["url"].removeprefix(API_URL) for c in stub.calls] == ["status.json", "status.json"]


def test_status_other_than_up_is_refused_not_cited():
    assert body_of("status")["status"] == "UP"
    for state in ("DOWN", "MAINTENANCE", None):
        down = dict(body_of("status"), status=state)
        with pytest.raises(ChemblError, match=f"ChEMBL_37 but status {state!r}, not 'UP'"):
            ChemblRetriever(StubHttp(served(fixture("status"), body=down))).version()


def test_params_name_every_setting_for_the_manifest():
    stub = StubHttp()
    r = ChemblRetriever(stub, page_size=5)
    assert r.params == {
        "api_url": API_URL, "page_size": 5, "max_page_size": MAX_PAGE_SIZE, "max_targets_per_symbol": MAX_TARGETS_PER_SYMBOL,
        "organism": "Homo sapiens", "syn_type": "GENE_SYMBOL",
        "accepted_relationships": ["SINGLE PROTEIN", "PROTEIN SUBUNIT", "GROUP MEMBER"],
        "order_by": {"mechanism": "mec_id", "drug_indication": "drugind_id", "drug_warning": "warning_id", "target": "target_chembl_id"},
        "family_filter": "parent_molecule_chembl_id", "per_second": 3.0,
        "release_scoped_record_ids": ["chembl:mechanism:<mec_id>", "chembl:indication:<drugind_id>", "chembl:warning:<warning_id>"],
        "search_fields": {"mechanism": "mechanism_of_action__icontains", "drug_indication": "mesh_heading__icontains"},
        "search_default_limit": 10, "search_max_limit": 25, "search_min_needle_chars": 3,
    }
    assert json.loads(json.dumps(r.params, sort_keys=True)) == r.params  # manifest-ready
    assert stub.calls == []  # a property: no request; the version is asked separately
    stub.limiter.per_second["www.ebi.ac.uk"] = 1.0
    assert r.params["per_second"] == 1.0  # the orchestrator's rate once it set one


# ---------------------------------------------------------------- molecule


def test_molecule_ivacaftor_is_the_object_verbatim():
    stub = _stub("molecule_CHEMBL2010601")
    rec = ChemblRetriever(stub).molecule(IVACAFTOR)
    fx = fixture("molecule_CHEMBL2010601")
    assert rec is not None
    assert rec.record_id == "chembl:CHEMBL2010601" and rec.source == "chembl" and rec.source_version == VERSION
    assert rec.query == {"endpoint": "molecule", "id": IVACAFTOR, "api": API_URL + "molecule/CHEMBL2010601.json"}
    assert rec.url == "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL2010601"
    assert rec.retrieved_at == fx["retrieved_at"]  # from the Response, not the clock
    assert rec.payload == json.loads(fx["text"])
    p = rec.payload
    assert (p["pref_name"], p["max_phase"], p["first_approval"], p["molecule_type"]) == ("IVACAFTOR", "4.0", 2012, "Small molecule")
    assert p["atc_classifications"] == ["R07AX02"] and p["withdrawn_flag"] is False and p["black_box_warning"] == 0
    assert p["molecule_hierarchy"]["parent_chembl_id"] == IVACAFTOR and parent_id(p) == IVACAFTOR
    assert "indication_class" not in p and "withdrawn_year" not in p  # gone in ChEMBL_37: use indications / warnings
    call = _call(stub, "molecule/CHEMBL2010601.json")
    assert call["cache_404"] is True and call["params"] is None


def test_molecule_unknown_id_is_none_not_an_error():
    stub = _stub("molecule_CHEMBL999999999")
    fx = fixture("molecule_CHEMBL999999999")
    assert fx["status"] == 404 and fx["text"] == "" and fx["headers"]["content-type"].startswith("text/html")  # the verified shape
    assert ChemblRetriever(stub).molecule(UNKNOWN) is None
    assert _paths(stub) == ["molecule/CHEMBL999999999.json"]


def test_molecule_404_with_a_body_is_a_routing_failure_not_absence():
    # observed live for a mistyped path (text/html, 122,869 bytes) and a wrong prefix (application/json, "{}"):
    # neither is ChEMBL saying "no such molecule", and a cached one would replay as absence forever
    for text in ("{}", "<!DOCTYPE html><html><body>Not Found</body></html>", " "):
        stub = StubHttp(fixture("status"), served(fixture("molecule_CHEMBL999999999"), text=text))
        with pytest.raises(ChemblError, match=r"404 with a body \(\d+ chars\) is not ChEMBL saying no such molecule"):
            ChemblRetriever(stub).molecule(UNKNOWN)


def test_molecule_id_is_normalised_before_the_request_and_junk_is_refused():
    a = ChemblRetriever(_stub("molecule_CHEMBL2010601")).molecule(IVACAFTOR)
    b = ChemblRetriever(_stub("molecule_CHEMBL2010601")).molecule("  chembl2010601 ")
    assert a is not None and a.to_json() == b.to_json()
    stub = StubHttp()
    r = ChemblRetriever(stub)
    for bad in ("", "2010601", "CHEMBL", "CHEMBL 2010601", "ivacaftor", "CHEMBL2010601;CHEMBL2103870"):
        with pytest.raises(ValueError, match="ChEMBL id"):
            r.molecule(bad)
        with pytest.raises(ValueError):
            r.mechanisms(bad)
        with pytest.raises(ValueError):
            r.indications(bad)
    assert stub.calls == []
    assert valid_chembl_id("chembl4051") == "CHEMBL4051"


def test_molecule_that_answers_for_another_id_raises():
    other = dict(body_of("molecule_CHEMBL2010601"), molecule_chembl_id=LUMACAFTOR)
    stub = StubHttp(fixture("status"), served(fixture("molecule_CHEMBL2010601"), body=other))
    with pytest.raises(ChemblError, match="answered for 'CHEMBL2103870'"):
        ChemblRetriever(stub).molecule(IVACAFTOR)


def test_molecule_deuterated_and_salt_forms_name_their_parent():
    rec = ChemblRetriever(_stub("molecule_CHEMBL4297603")).molecule(DEUTIVACAFTOR)
    assert rec.payload["pref_name"] == "DEUTIVACAFTOR" and rec.payload["first_approval"] == 2024
    assert rec.payload["molecule_hierarchy"] == {"active_chembl_id": IVACAFTOR, "molecule_chembl_id": DEUTIVACAFTOR, "parent_chembl_id": IVACAFTOR}
    assert parent_id(rec.payload) == IVACAFTOR
    salt = ChemblRetriever(_stub("molecule_CHEMBL1200912")).molecule(PHENTERMINE_HCL).payload
    assert (salt["pref_name"], parent_id(salt), salt["withdrawn_flag"], salt["atc_classifications"]) == ("PHENTERMINE HYDROCHLORIDE", PHENTERMINE, False, [])
    assert parent_id({"molecule_chembl_id": "CHEMBL1", "molecule_hierarchy": None}) == "CHEMBL1"


# ---------------------------------------------------------------- mechanisms


def test_mechanisms_ivacaftor_and_lumacaftor():
    stub = _stub("mechanism_molecule_CHEMBL2010601", "mechanism_molecule_CHEMBL2103870")
    r = ChemblRetriever(stub)
    (iva,) = r.mechanisms(IVACAFTOR)
    fx = fixture("mechanism_molecule_CHEMBL2010601")
    assert iva.record_id == "chembl:mechanism:965" and iva.source == "chembl" and iva.source_version == VERSION
    assert iva.query == {"endpoint": "mechanism", "id": "965", "api": API_URL + "mechanism/965.json"}
    assert iva.url == "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL2010601"
    assert iva.retrieved_at == fx["retrieved_at"] and iva.payload == json.loads(fx["text"])["mechanisms"][0]
    m = iva.payload
    assert (m["action_type"], m["max_phase"], m["target_chembl_id"], m["variant_sequence"]) == ("POSITIVE MODULATOR", 4, CFTR_TARGET, None)
    assert m["mechanism_of_action"] == "Cystic fibrosis transmembrane conductance regulator positive modulator"
    assert m["mechanism_refs"][0]["ref_type"] == "DailyMed"
    assert _call(stub, "mechanism.json")["params"] == _list_params(_endpoint="mechanism.json", molecule_chembl_id=IVACAFTOR)

    (luma,) = r.mechanisms(LUMACAFTOR)
    assert luma.record_id == "chembl:mechanism:2346" and luma.payload["action_type"] == "STABILISER"
    v = luma.payload["variant_sequence"]
    assert (v["accession"], v["mutation"], v["organism"], v["tax_id"]) == ("P13569", "F508del", "Homo sapiens", 9606)
    assert "F508del" in luma.payload["mechanism_comment"]


def test_mechanisms_of_an_unknown_molecule_is_empty():
    stub = _stub("mechanism_molecule_CHEMBL999999999")
    assert ChemblRetriever(stub).mechanisms(UNKNOWN) == []
    assert _call(stub, "mechanism.json")["params"]["molecule_chembl_id"] == UNKNOWN


def test_mechanisms_for_target_all_thirteen_in_mec_id_order():
    r = ChemblRetriever(_stub("mechanism_target_CHEMBL4051"))
    recs = r.mechanisms_for_target(CFTR_TARGET)
    assert [x.payload["mec_id"] for x in recs] == CFTR_MEC_IDS
    assert [x.record_id for x in recs] == [f"chembl:mechanism:{i}" for i in CFTR_MEC_IDS]
    assert all(x.payload["target_chembl_id"] == CFTR_TARGET for x in recs)
    assert sum(1 for x in recs if x.payload["max_phase"] == 4) == 7
    assert sum(1 for x in recs if x.payload["variant_sequence"]) == 9
    assert all(x.payload["variant_sequence"]["mutation"] == "F508del" for x in recs if x.payload["variant_sequence"])
    assert {x.payload["action_type"] for x in recs} == {"ACTIVATOR", "INHIBITOR", "POSITIVE MODULATOR", "STABILISER"}
    (cro,) = [x for x in recs if x.payload["molecule_chembl_id"] == CROFELEMER]
    assert cro.payload["action_type"] == "INHIBITOR"  # an antidiarrhoeal, not a CF drug: kept, labelled
    (deu,) = [x for x in recs if x.payload["molecule_chembl_id"] == DEUTIVACAFTOR]
    assert deu.payload["parent_molecule_chembl_id"] == IVACAFTOR


def test_the_same_mechanism_is_the_same_record_whichever_route_found_it():
    r = ChemblRetriever(_stub("mechanism_molecule_CHEMBL2010601", "mechanism_target_CHEMBL4051"))
    (by_mol,) = r.mechanisms(IVACAFTOR)
    (by_target,) = [x for x in r.mechanisms_for_target(CFTR_TARGET) if x.payload["mec_id"] == 965]
    assert by_mol.record_id == by_target.record_id and by_mol.query == by_target.query and by_mol.payload == by_target.payload
    # only retrieved_at is the list response's — documented; the stage tool keeps the first copy it stores
    assert dataclasses.replace(by_mol, retrieved_at="") == dataclasses.replace(by_target, retrieved_at="")


# ---------------------------------------------------------------- indications and warnings


def test_indications_are_the_family_s_rows_in_id_order():
    stub = _stub("drug_indication_parent_CHEMBL2010601")
    recs = ChemblRetriever(stub).indications(IVACAFTOR)
    # ivacaftor's seven rows and deutivacaftor's one, which ChEMBL files under the parent
    assert [x.payload["drugind_id"] for x in recs] == [26648, 56193, 56194, 56195, 116606, 137003, 137004, 142771]
    assert all(x.payload["parent_molecule_chembl_id"] == IVACAFTOR for x in recs)
    assert [x.payload["molecule_chembl_id"] for x in recs] == [IVACAFTOR] * 7 + [DEUTIVACAFTOR]
    cf = recs[5]
    assert cf.record_id == "chembl:indication:137003"
    assert cf.query == {"endpoint": "drug_indication", "id": "137003", "api": API_URL + "drug_indication/137003.json"}
    assert cf.url == "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL2010601"
    assert (cf.payload["mesh_heading"], cf.payload["mesh_id"], cf.payload["efo_id"], cf.payload["max_phase_for_ind"]) == ("Cystic Fibrosis", "D003550", "MONDO:0009061", "4.0")
    assert {ref["ref_type"] for ref in cf.payload["indication_refs"]} == {"ClinicalTrials", "DailyMed", "EMA", "FDA"}
    assert recs[-1].url == "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL4297603"  # the row's own compound page
    assert {x.payload["max_phase_for_ind"] for x in recs} == {"0.5", "1.0", "2.0", "3.0", "4.0"}  # strings, as served
    assert _call(stub, "drug_indication.json")["params"] == _list_params(_endpoint="drug_indication.json", parent_molecule_chembl_id=IVACAFTOR)


def test_warnings_rofecoxib_and_none_for_ivacaftor():
    stub = _stub("drug_warning_parent_CHEMBL122", "drug_warning_parent_CHEMBL2010601")
    r = ChemblRetriever(stub)
    recs = r.warnings(ROFECOXIB)
    assert len(recs) == 11 and [x.payload["warning_id"] for x in recs] == sorted(x.payload["warning_id"] for x in recs)
    assert recs[0].record_id == "chembl:warning:1662"
    assert recs[0].query == {"endpoint": "drug_warning", "id": "1662", "api": API_URL + "drug_warning/1662.json"}
    assert recs[0].url == "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL122"
    assert (recs[0].payload["warning_type"], recs[0].payload["warning_year"], recs[0].payload["warning_country"]) == ("Black Box Warning", None, "United States")
    assert (recs[1].payload["warning_type"], recs[1].payload["warning_year"], recs[1].payload["warning_country"], recs[1].payload["warning_class"]) == ("Withdrawn", 2004, "Worldwide", "cardiotoxicity")
    assert {x.payload["warning_type"] for x in recs} == {"Black Box Warning", "Withdrawn"}
    assert warning_summary(recs) == ["Black Box Warning United States", "Withdrawn 2004 Worldwide"]
    assert r.warnings(IVACAFTOR) == []
    assert _call(stub, "drug_warning.json")["params"] == _list_params(_endpoint="drug_warning.json", parent_molecule_chembl_id=IVACAFTOR)


def test_warnings_of_a_withdrawn_parent_sit_on_the_parent_not_the_salt():
    r = ChemblRetriever(_stub("drug_warning_parent_CHEMBL1574", "drug_warning_parent_CHEMBL122"))
    recs = r.warnings(PHENTERMINE)
    assert [(x.payload["warning_id"], x.payload["molecule_chembl_id"], x.payload["warning_type"], x.payload["warning_year"]) for x in recs] == [
        (3989, PHENTERMINE, "Withdrawn", 1981), (4125, PHENTERMINE, "Withdrawn", 1981)]
    assert warning_summary(recs) == ["Withdrawn 1981 Sweden; United Kingdom; Mauritius"]


# ---------------------------------------------------------------- targets


def test_targets_for_symbol_cftr_bub1b_mthfr():
    stub = _stub("target_symbol_CFTR", "target_symbol_BUB1B", "target_symbol_MTHFR")
    r = ChemblRetriever(stub)
    (cftr,) = r.targets_for_symbol("CFTR")
    fx = fixture("target_symbol_CFTR")
    both = json.loads(fx["text"])["targets"]
    # ChEMBL returns two: the protein, and a CFTR/GOPC protein-protein-interaction target the gene is only a partner in
    assert [(t["target_chembl_id"], t["target_type"]) for t in both] == [(CFTR_PPI_TARGET, "PROTEIN-PROTEIN INTERACTION"), (CFTR_TARGET, "SINGLE PROTEIN")]
    assert symbol_relationships(both[0], "CFTR") == {"INTERACTING PROTEIN"} and gene_symbols(both[0]) == ["CFTR", "GOPC"]
    assert cftr.record_id == "chembl:CHEMBL4051" and cftr.source_version == VERSION
    assert cftr.query == {"endpoint": "target", "id": CFTR_TARGET, "api": API_URL + "target/CHEMBL4051.json"}
    assert cftr.url == "https://www.ebi.ac.uk/chembl/explore/target/CHEMBL4051"
    assert cftr.retrieved_at == fx["retrieved_at"] and cftr.payload == both[1]
    t = cftr.payload
    assert (t["pref_name"], t["target_type"], t["organism"], t["tax_id"]) == ("Cystic fibrosis transmembrane conductance regulator", "SINGLE PROTEIN", "Homo sapiens", 9606)
    assert t["target_components"][0]["accession"] == "P13569" and gene_symbols(t) == ["CFTR"]
    assert symbol_relationships(t, "cftr") == {"SINGLE PROTEIN"} and symbol_relationships(t, "GOPC") == set()
    assert _call(stub, "target.json")["params"] == {
        "target_components__target_component_synonyms__component_synonym__iexact": "CFTR",
        "target_components__target_component_synonyms__syn_type": "GENE_SYMBOL",
        "organism": "Homo sapiens", "order_by": "target_chembl_id", "limit": MAX_PAGE_SIZE, "offset": 0,
    }  # no target_type: complexes and families the gene product is part of are wanted; interactions are dropped by relationship
    (bub1b,) = r.targets_for_symbol("BUB1B")
    assert bub1b.record_id == "chembl:CHEMBL4295998" and bub1b.payload["pref_name"].startswith("Mitotic checkpoint")
    assert r.targets_for_symbol("MTHFR") == []  # not a ChEMBL target: absence, HTTP 200, total_count 0


def test_targets_keep_complexes_and_families_the_gene_product_is_part_of():
    # the recorded CFTR/GOPC interaction target served as if CFTR were a subunit of a complex (edit of a public
    # record, never sent): the relationship, not the target type, decides — and its mechanisms are then fetched
    body = copy.deepcopy(body_of("target_symbol_CFTR"))
    ppi = body["targets"][0]
    assert ppi["target_chembl_id"] == CFTR_PPI_TARGET
    ppi["target_type"] = "PROTEIN COMPLEX"
    for comp in ppi["target_components"]:
        comp["relationship"] = "PROTEIN SUBUNIT"
    stub = _cftr_stub(served(fixture("target_symbol_CFTR"), body=body))
    r = ChemblRetriever(stub)
    assert [t.record_id for t in r.targets_for_symbol("CFTR")] == [f"chembl:{CFTR_PPI_TARGET}", "chembl:CHEMBL4051"]
    recs = r.drugs_for_target_symbol("CFTR")
    assert [x.record_id for x in recs if x.query["endpoint"] == "target"] == [f"chembl:{CFTR_PPI_TARGET}", "chembl:CHEMBL4051"]
    assert len(recs) == 55  # one more target; the recorded mechanism list for it is empty
    assert [c["params"]["target_chembl_id"] for c in stub.calls if c["url"].endswith("mechanism.json")] == [CFTR_PPI_TARGET, CFTR_TARGET]
    for relationship in ACCEPTED_RELATIONSHIPS:
        for comp in ppi["target_components"]:
            comp["relationship"] = relationship
        assert len(ChemblRetriever(_cftr_stub(served(fixture("target_symbol_CFTR"), body=body))).targets_for_symbol("CFTR")) == 2
    for relationship in ("INTERACTING PROTEIN", "FUSION PROTEIN", "", None):
        for comp in ppi["target_components"]:
            comp["relationship"] = relationship
        assert len(ChemblRetriever(_cftr_stub(served(fixture("target_symbol_CFTR"), body=body))).targets_for_symbol("CFTR")) == 1


def test_symbol_is_upper_cased_before_it_is_sent_and_junk_is_refused():
    stub = _stub("target_symbol_CFTR")
    (a,) = ChemblRetriever(stub).targets_for_symbol("  cftr ")
    assert _call(stub, "target.json")["params"]["target_components__target_component_synonyms__component_synonym__iexact"] == "CFTR"
    assert a.to_json() == ChemblRetriever(_stub("target_symbol_CFTR")).targets_for_symbol("CFTR")[0].to_json()
    empty = StubHttp()
    r = ChemblRetriever(empty)
    for bad in ("", "   ", "CF TR", 'A"B', "TP\n53", "{}", "-A", "CFTR,TP53"):
        with pytest.raises(ValueError, match="gene symbol") as e:
            r.targets_for_symbol(bad)
        assert "CFTR" not in str(e.value) and "TP" not in str(e.value)  # the error never echoes what was asked
    assert empty.calls == []
    assert valid_symbol(" HLA-A ") == "HLA-A" and valid_symbol("C1orf123") == "C1ORF123"


def test_target_that_does_not_carry_the_symbol_is_refused():
    body = copy.deepcopy(body_of("target_symbol_CFTR"))
    for t in body["targets"]:
        for comp in t["target_components"]:
            comp["target_component_synonyms"] = [s for s in comp["target_component_synonyms"] if s["syn_type"] != "GENE_SYMBOL"]
    with pytest.raises(ChemblError, match="carries no GENE_SYMBOL synonym equal to the one asked for") as e:
        ChemblRetriever(StubHttp(fixture("status"), served(fixture("target_symbol_CFTR"), body=body))).targets_for_symbol("CFTR")
    assert "CFTR" not in str(e.value)


def test_ignored_filter_is_refused_not_served_as_a_rich_answer():
    # a typo'd filter name makes the server answer the whole table with HTTP 200
    whole_table = dict(body_of("target_symbol_CFTR"))
    whole_table["page_meta"] = dict(whole_table["page_meta"], total_count=18552, next="/chembl/api/data/target.json?limit=1000&offset=1000")
    with pytest.raises(ChemblError, match=f"18552 rows for filters \\[.*'organism'.*\\]; more than {MAX_TARGETS_PER_SYMBOL}") as e:
        ChemblRetriever(StubHttp(fixture("status"), served(fixture("target_symbol_CFTR"), body=whole_table))).targets_for_symbol("CFTR")
    assert "CFTR" not in str(e.value)  # filter names, never the gene
    foreign = copy.deepcopy(body_of("target_symbol_CFTR"))
    foreign["targets"][1]["organism"] = "Rattus norvegicus"
    with pytest.raises(ChemblError, match="ignored organism='Homo sapiens'"):
        ChemblRetriever(StubHttp(fixture("status"), served(fixture("target_symbol_CFTR"), body=foreign))).targets_for_symbol("CFTR")
    # a mechanism list whose rows are not for the molecule asked
    other = copy.deepcopy(body_of("mechanism_molecule_CHEMBL2010601"))
    other["mechanisms"][0]["molecule_chembl_id"] = LUMACAFTOR
    with pytest.raises(ChemblError, match="ignored molecule_chembl_id='CHEMBL2010601'"):
        ChemblRetriever(StubHttp(fixture("status"), served(fixture("mechanism_molecule_CHEMBL2010601"), body=other))).mechanisms(IVACAFTOR)
    # an indication list whose rows belong to another family
    other = copy.deepcopy(body_of("drug_indication_parent_CHEMBL2010601"))
    other["drug_indications"][0]["parent_molecule_chembl_id"] = LUMACAFTOR
    with pytest.raises(ChemblError, match="ignored parent_molecule_chembl_id='CHEMBL2010601'"):
        ChemblRetriever(StubHttp(fixture("status"), served(fixture("drug_indication_parent_CHEMBL2010601"), body=other))).indications(IVACAFTOR)


# ---------------------------------------------------------------- drugs for a gene


def test_drugs_for_target_symbol_cftr_end_to_end():
    stub = _cftr_stub()
    recs = ChemblRetriever(stub).drugs_for_target_symbol("CFTR")
    kinds = [r.query["endpoint"] for r in recs]
    assert kinds == ["target"] + ["mechanism"] * 13 + ["molecule"] * 13 + ["drug_indication"] * 27
    assert len(recs) == 54 and len({r.record_id for r in recs}) == 54
    assert recs[0].record_id == "chembl:CHEMBL4051"  # the interaction target is not among them
    assert [r.payload["mec_id"] for r in recs if r.query["endpoint"] == "mechanism"] == CFTR_MEC_IDS
    mols = [r for r in recs if r.query["endpoint"] == "molecule"]
    assert [r.payload["molecule_chembl_id"] for r in mols] == CFTR_MOLECULES  # id order, one detail GET each
    assert [parent_id(r.payload) for r in mols] == [IVACAFTOR if c == DEUTIVACAFTOR else c for c in CFTR_MOLECULES]
    assert {r.payload["pref_name"] for r in mols} >= {"IVACAFTOR", "LUMACAFTOR", "TEZACAFTOR", "ELEXACAFTOR", "CROFELEMER", "VANZACAFTOR"}
    assert sum(1 for r in mols if r.payload["max_phase"] == "4.0") == 7
    assert sum(1 for r in mols if r.payload["first_approval"] is None) == 6  # phase 2/3 candidates
    inds = [r for r in recs if r.query["endpoint"] == "drug_indication"]
    assert [r.payload["drugind_id"] for r in inds] == CFTR_INDICATION_IDS
    assert {r.payload["parent_molecule_chembl_id"] for r in inds} == set(CFTR_FAMILIES)
    # two rows are documented on other forms of a family (bamocaftor's and vanzacaftor's), fetched with the family
    assert sorted({r.payload["molecule_chembl_id"] for r in inds} - set(CFTR_MOLECULES)) == ["CHEMBL4298159", "CHEMBL6068396"]
    cf = {r.payload["molecule_chembl_id"]: r.payload["max_phase_for_ind"] for r in inds if r.payload["efo_id"] == "MONDO:0009061"}
    assert len(cf) == 13 and sum(1 for v in cf.values() if v == "4.0") == 3
    # the indication's phase lags the molecule's: elexacaftor is approved (max_phase "4.0", 2019) but its CF row says 3.0
    assert cf["CHEMBL4298128"] == "3.0" and cf["CHEMBL5314934"] == "2.0"
    assert {r.payload["molecule_chembl_id"]: r.payload["max_phase"] for r in mols}["CHEMBL4298128"] == "4.0"
    assert all(r.source_version == VERSION for r in recs)
    # warnings are asked for every family, flagged or not (ChEMBL's flag is not always set): all empty for CFTR drugs
    assert [c["params"]["parent_molecule_chembl_id"] for c in stub.calls if c["url"].endswith("drug_warning.json")] == CFTR_FAMILIES
    assert _paths(stub)[:2] == ["target.json", "mechanism.json"]
    assert _paths(stub)[2:15] == [f"molecule/{cid}.json" for cid in CFTR_MOLECULES]  # every parent is already among them
    assert _paths(stub)[15:] == ["drug_indication.json", "drug_warning.json"] * 12
    assert len(stub.calls) == 2 + 2 + 13 + 2 * 12  # status twice (cache + live), target, mechanism, molecules, per family


def test_drugs_for_symbol_absence_shapes():
    stub = _stub("target_symbol_MTHFR", "target_symbol_BUB1B", "mechanism_target_CHEMBL4295998")
    r = ChemblRetriever(stub)
    assert r.drugs_for_target_symbol("MTHFR") == []  # not a target
    (only_target,) = r.drugs_for_target_symbol("BUB1B")  # a target with no curated mechanism: still citable
    assert only_target.record_id == "chembl:CHEMBL4295998" and only_target.query["endpoint"] == "target"
    assert _paths(stub) == ["target.json", "target.json", "mechanism.json"]


def test_parent_absent_from_the_mechanism_list_is_fetched_for_its_salt_or_child():
    # the recorded CFTR mechanism list served without ivacaftor's own row (mec_id 965; an edit of a public
    # record, never sent): deutivacaftor still names ivacaftor as its parent, so ivacaftor's record — the one
    # carrying the ATC code — is fetched, and the family's indications are asked once, under the parent
    full = body_of("mechanism_target_CHEMBL4051")
    rows = [m for m in full["mechanisms"] if m["mec_id"] != 965]
    without = dict(full, mechanisms=rows, page_meta=dict(full["page_meta"], total_count=12))
    stub = _cftr_stub(served(fixture("mechanism_target_CHEMBL4051"), body=without))
    recs = ChemblRetriever(stub).drugs_for_target_symbol("CFTR")
    assert len(recs) == 53 and [r.query["endpoint"] for r in recs].count("mechanism") == 12
    mols = [r.payload["molecule_chembl_id"] for r in recs if r.query["endpoint"] == "molecule"]
    assert mols == CFTR_MOLECULES  # ivacaftor is there as deutivacaftor's parent, in id order with the rest
    assert _paths(stub)[2:15] == [f"molecule/{c}.json" for c in CFTR_MOLECULES if c != IVACAFTOR] + ["molecule/CHEMBL2010601.json"]
    assert [c["params"]["parent_molecule_chembl_id"] for c in stub.calls if c["url"].endswith("drug_indication.json")] == CFTR_FAMILIES
    (deu,) = [row for row in drug_rows(recs) if row["chembl_id"] == DEUTIVACAFTOR]
    assert (deu["parent_chembl_id"], deu["atc_codes"], deu["first_approval"], deu["withdrawn"]) == (IVACAFTOR, "R07AX02", "2024", "N")
    assert deu["indications"].startswith("Cystic Fibrosis [MONDO:0009061, D003550] phase 4 via CHEMBL2010601; ")
    assert "Cystic Fibrosis [MONDO:0009061, D003550] phase 3;" in deu["indications"]  # its own row, unmarked
    assert deu["evidence_ids"].split(";")[:4] == ["chembl:mechanism:9954", "chembl:CHEMBL4297603", "chembl:CHEMBL2010601", "chembl:CHEMBL4051"]


def test_mechanism_naming_a_molecule_the_api_then_404s_raises():
    gone = served(fixture("molecule_CHEMBL2010601"), status=404, text="")
    with pytest.raises(ChemblError, match="CHEMBL2010601 is 404"):
        ChemblRetriever(_cftr_stub(gone)).drugs_for_target_symbol("CFTR")
    # the same for a parent a mechanism's molecule names
    full = body_of("mechanism_target_CHEMBL4051")
    without = dict(full, mechanisms=[m for m in full["mechanisms"] if m["mec_id"] != 965], page_meta=dict(full["page_meta"], total_count=12))
    with pytest.raises(ChemblError, match="CHEMBL2010601 is 404"):
        ChemblRetriever(_cftr_stub(served(fixture("mechanism_target_CHEMBL4051"), body=without), gone)).drugs_for_target_symbol("CFTR")


# ---------------------------------------------------------------- pagination


def test_pagination_walks_every_page_in_key_order_and_agrees_with_one_page():
    stub = _stub("mechanism_target_CHEMBL4051_limit5_offset0", "mechanism_target_CHEMBL4051_limit5_offset5",
                 "mechanism_target_CHEMBL4051_limit5_offset10")
    paged = ChemblRetriever(stub, page_size=5).mechanisms_for_target(CFTR_TARGET)
    assert [(c["params"]["order_by"], c["params"]["limit"], c["params"]["offset"]) for c in stub.calls if "mechanism" in c["url"]] == [("mec_id", 5, 0), ("mec_id", 5, 5), ("mec_id", 5, 10)]
    assert "order_by=mec_id" in body_of("mechanism_target_CHEMBL4051_limit5_offset0")["page_meta"]["next"]  # the server honours it
    one = ChemblRetriever(_stub("mechanism_target_CHEMBL4051")).mechanisms_for_target(CFTR_TARGET)
    assert [x.record_id for x in paged] == [x.record_id for x in one]
    assert [x.payload for x in paged] == [x.payload for x in one]


def test_a_row_served_twice_across_pages_is_refused():
    # page 2 served with page 1's rows again (an unordered walk shifting under us) — the totals would still add up
    p0, p5 = body_of("mechanism_target_CHEMBL4051_limit5_offset0"), body_of("mechanism_target_CHEMBL4051_limit5_offset5")
    stub = StubHttp(fixture("status"), fixture("mechanism_target_CHEMBL4051_limit5_offset0"),
                    served(fixture("mechanism_target_CHEMBL4051_limit5_offset5"), body=dict(p5, mechanisms=p0["mechanisms"])),
                    fixture("mechanism_target_CHEMBL4051_limit5_offset10"))
    with pytest.raises(ChemblError, match=r"served mec_id=964 twice"):
        ChemblRetriever(stub, page_size=5).mechanisms_for_target(CFTR_TARGET)
    missing = dict(p0, mechanisms=[{k: v for k, v in p0["mechanisms"][0].items() if k != "mec_id"}] + p0["mechanisms"][1:])
    with pytest.raises(ChemblError, match=r"served mec_id=None twice \(or without it\)"):
        ChemblRetriever(StubHttp(fixture("status"), served(fixture("mechanism_target_CHEMBL4051_limit5_offset0"), body=missing)), page_size=5).mechanisms_for_target(CFTR_TARGET)


def test_short_or_truncated_list_answer_is_refused_not_cached_as_truth():
    # one page, next=None, five of the thirteen rows the same page_meta announces
    full = body_of("mechanism_target_CHEMBL4051")
    short = dict(full, mechanisms=full["mechanisms"][:5])
    with pytest.raises(ChemblError, match=r"served 5 row\(s\) for filters \['target_chembl_id'\] but page_meta\.total_count is 13"):
        ChemblRetriever(StubHttp(fixture("status"), served(fixture("mechanism_target_CHEMBL4051"), body=short))).mechanisms_for_target(CFTR_TARGET)
    # an empty middle page that still names a next page
    p5 = body_of("mechanism_target_CHEMBL4051_limit5_offset5")
    stub = StubHttp(fixture("status"), fixture("mechanism_target_CHEMBL4051_limit5_offset0"),
                    served(fixture("mechanism_target_CHEMBL4051_limit5_offset5"), body=dict(p5, mechanisms=[])),
                    fixture("mechanism_target_CHEMBL4051_limit5_offset10"))
    with pytest.raises(ChemblError, match=r"offset 5 is empty yet names a next page \(5 of 13 rows seen\)"):
        ChemblRetriever(stub, page_size=5).mechanisms_for_target(CFTR_TARGET)
    assert [c["params"]["offset"] for c in stub.calls if "mechanism" in c["url"]] == [0, 5]  # stopped there
    # a short last page
    p10 = body_of("mechanism_target_CHEMBL4051_limit5_offset10")
    stub = StubHttp(fixture("status"), fixture("mechanism_target_CHEMBL4051_limit5_offset0"),
                    fixture("mechanism_target_CHEMBL4051_limit5_offset5"),
                    served(fixture("mechanism_target_CHEMBL4051_limit5_offset10"), body=dict(p10, mechanisms=p10["mechanisms"][:1])))
    with pytest.raises(ChemblError, match=r"served 11 row\(s\)"):
        ChemblRetriever(stub, page_size=5).mechanisms_for_target(CFTR_TARGET)
    # the empty answer is whole: zero rows, total_count 0, next None
    assert ChemblRetriever(_stub("target_symbol_MTHFR")).targets_for_symbol("MTHFR") == []
    # a string total_count is schema drift, reported as this module's error, not a TypeError
    t = body_of("target_symbol_CFTR")
    drifted = dict(t, page_meta=dict(t["page_meta"], total_count="1"))
    with pytest.raises(ChemblError, match="no integer total_count"):
        ChemblRetriever(StubHttp(fixture("status"), served(fixture("target_symbol_CFTR"), body=drifted))).targets_for_symbol("CFTR")


def test_page_size_is_bounded_by_the_server_cap():
    for bad in (0, -1, MAX_PAGE_SIZE + 1):
        with pytest.raises(ValueError, match="page_size"):
            ChemblRetriever(StubHttp(), page_size=bad)
    assert ChemblRetriever(StubHttp(), page_size=MAX_PAGE_SIZE).page_size == 1000


# ---------------------------------------------------------------- failure is never absence


def _iva_served(**kw) -> ChemblRetriever:
    return ChemblRetriever(StubHttp(fixture("status"), served(fixture("mechanism_molecule_CHEMBL2010601"), **kw)))


def test_non_json_and_non_object_bodies_raise():
    with pytest.raises(ChemblError, match="non-JSON"):  # the XML default / an HTML error page
        _iva_served(text='<?xml version="1.0"?><response><mechanisms/></response>').mechanisms(IVACAFTOR)
    with pytest.raises(ChemblError, match="expected an object"):
        _iva_served(text="[]").mechanisms(IVACAFTOR)
    with pytest.raises(ChemblError, match="page_meta"):
        _iva_served(body={"mechanisms": []}).mechanisms(IVACAFTOR)
    with pytest.raises(ChemblError, match="not an object"):
        _iva_served(body={"mechanisms": ["x"], "page_meta": {"total_count": 1, "next": None}}).mechanisms(IVACAFTOR)
    with pytest.raises(ChemblError, match="integer mec_id"):
        _iva_served(body={"mechanisms": [{"molecule_chembl_id": IVACAFTOR, "mec_id": "965"}], "page_meta": {"total_count": 1, "next": None}}).mechanisms(IVACAFTOR)
    with pytest.raises(HttpError):  # 5xx: raised by Http after its retries, never absence
        _iva_served(status=500, text="").mechanisms(IVACAFTOR)
    with pytest.raises(HttpError):  # a 404 on a list endpoint is not the molecule's absence (that is an empty list)
        _iva_served(status=404, text="").mechanisms(IVACAFTOR)


# ---------------------------------------------------------------- determinism and the store


def test_records_are_byte_identical_and_survive_the_store(tmp_path: Path):
    a = ChemblRetriever(_cftr_stub()).drugs_for_target_symbol("CFTR")
    b = ChemblRetriever(_cftr_stub()).drugs_for_target_symbol("cftr")  # iexact on the server; same records
    assert [x.to_json() for x in a] == [x.to_json() for x in b]
    store = EvidenceStore(tmp_path / "evidence")
    for rec in a:
        p = store.put(rec)
        assert p.parent.name == "chembl" and store.get(rec.record_id) == rec
    assert store.count("chembl") == 54
    index = json.loads(store.write_index().read_text())
    assert index["chembl:mechanism:2346"]["source_version"] == VERSION
    assert index["chembl:CHEMBL2010601"]["url"] == "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL2010601"
    assert index["chembl:CHEMBL4051"]["url"] == "https://www.ebi.ac.uk/chembl/explore/target/CHEMBL4051"
    assert index["chembl:indication:142771"]["url"] == "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL4297603"
    assert set(index) == {x.record_id for x in a}


def test_fixtures_are_cache_entries_the_real_client_replays_offline(tmp_path: Path):
    cache = _cache_from_fixtures(tmp_path / "cache")
    http = Http(cache, offline=True)
    r = ChemblRetriever(http)
    assert r.version() == VERSION
    recs = r.drugs_for_target_symbol("CFTR")
    assert [x.to_json() for x in recs] == [x.to_json() for x in ChemblRetriever(_cftr_stub()).drugs_for_target_symbol("CFTR")]
    assert r.molecule(UNKNOWN) is None  # the recorded 404 replays as absence
    assert r.drugs_for_target_symbol("MTHFR") == [] and len(r.warnings(ROFECOXIB)) == 11
    assert len(r.mechanisms(PHENTERMINE_HCL)) == 1 and len(r.indications(PHENTERMINE)) == 8 and len(r.warnings(PHENTERMINE)) == 2
    assert http.live_requests == 0 and cache.misses == 0
    with pytest.raises(RuntimeError, match="offline"):  # anything unrecorded is refused, never fetched
        r.molecule("CHEMBL25")


def test_rate_is_pinned_politely_but_never_overrides_the_orchestrator():
    stub = StubHttp()
    ChemblRetriever(stub)
    assert stub.limiter.per_second["www.ebi.ac.uk"] == 3.0
    stub.limiter.per_second["www.ebi.ac.uk"] = 1.0
    ChemblRetriever(stub)
    assert stub.limiter.per_second["www.ebi.ac.uk"] == 1.0
    assert ChemblRetriever(StubHttp(), base_url="https://example.org/api/data").base_url == "https://example.org/api/data/"


def test_log_lines_carry_counts_never_the_gene_or_an_id(caplog):
    with caplog.at_level(logging.DEBUG, logger="engine.medicine.chembl"):
        ChemblRetriever(_cftr_stub()).drugs_for_target_symbol("CFTR")
    messages = [rec.getMessage() for rec in caplog.records]
    assert messages == ["chembl: 2 target(s) carry the symbol, 1 kept (gene product is the target, a subunit or a member)",
                        "chembl: 1 target(s), 13 mechanism row(s), 13 molecule(s) in 12 famil(ies), 27 indication row(s), 0 warning row(s)"]
    assert not any("CFTR" in m or "CHEMBL" in m for m in messages)


# ---------------------------------------------------------------- projection


def test_drug_rows_one_row_per_mechanism():
    recs = ChemblRetriever(_cftr_stub()).drugs_for_target_symbol("CFTR")
    rows = drug_rows(recs)
    assert len(rows) == 13 and all(tuple(row) == COLUMNS for row in rows)
    assert [row["chembl_id"] for row in rows] == CFTR_MOLECULES  # molecule id, then mec_id
    by_id = {row["chembl_id"]: row for row in rows}
    iva = by_id[IVACAFTOR]
    assert iva == {
        "chembl_id": IVACAFTOR, "name": "IVACAFTOR", "parent_chembl_id": IVACAFTOR, "molecule_type": "Small molecule",
        "max_phase": "4", "first_approval": "2012", "withdrawn": "N", "black_box_warning": "N", "warnings": "",
        "atc_codes": "R07AX02", "target_chembl_id": CFTR_TARGET,
        "target_name": "Cystic fibrosis transmembrane conductance regulator", "target_type": "SINGLE PROTEIN",
        "action_type": "POSITIVE MODULATOR",
        "mechanism_of_action": "Cystic fibrosis transmembrane conductance regulator positive modulator",
        "variant_mutation": "",
        # the family's rows, highest phase first; deutivacaftor's CF row is marked as documented on that form
        "indications": "Cystic Fibrosis [MONDO:0009061, D003550] phase 4; Respiratory Tract Diseases [EFO:0000684, D012140] phase 4; "
                       "Cystic Fibrosis [MONDO:0009061, D003550] phase 3 via CHEMBL4297603; "
                       "Bronchitis, Chronic [EFO:0006505, D029481] phase 2; Ciliary Motility Disorders [MONDO:0016575, D002925] phase 2; "
                       "Pulmonary Disease, Chronic Obstructive [EFO:0000341, D029424] phase 2; Liver Diseases [EFO:0001421, D008107] phase 1; "
                       "Sinusitis [EFO:0007486, D012852] phase 0.5",
        "evidence_ids": "chembl:mechanism:965;chembl:CHEMBL2010601;chembl:CHEMBL4051;chembl:indication:137003;chembl:indication:116606;"
                        "chembl:indication:142771;chembl:indication:56194;chembl:indication:137004;chembl:indication:26648;"
                        "chembl:indication:56193;chembl:indication:56195",
    }
    luma = by_id[LUMACAFTOR]
    assert (luma["action_type"], luma["variant_mutation"], luma["atc_codes"], luma["first_approval"]) == ("STABILISER", "F508del", "", "2015")
    deu = by_id[DEUTIVACAFTOR]  # its own phase, approval and name; the parent's ATC code; the family's indications
    assert (deu["parent_chembl_id"], deu["max_phase"], deu["first_approval"], deu["action_type"], deu["atc_codes"]) == (IVACAFTOR, "4", "2024", "ACTIVATOR", "R07AX02")
    assert deu["indications"].split("; ")[:3] == ["Cystic Fibrosis [MONDO:0009061, D003550] phase 4 via CHEMBL2010601",
                                                  "Respiratory Tract Diseases [EFO:0000684, D012140] phase 4 via CHEMBL2010601",
                                                  "Cystic Fibrosis [MONDO:0009061, D003550] phase 3"]
    assert deu["evidence_ids"].split(";")[:4] == ["chembl:mechanism:9954", "chembl:CHEMBL4297603", "chembl:CHEMBL2010601", "chembl:CHEMBL4051"]
    bam = by_id["CHEMBL4297849"]
    assert bam["indications"] == "Cystic Fibrosis [MONDO:0009061, D003550] phase 3; Cystic Fibrosis [MONDO:0009061, D003550] phase 1 via CHEMBL4298159"
    ice = by_id["CHEMBL4650318"]
    assert (ice["name"], ice["max_phase"], ice["first_approval"]) == ("ICENTICAFTOR", "2", "")
    assert by_id[CROFELEMER]["action_type"] == "INHIBITOR" and "Diarrhea" in by_id[CROFELEMER]["indications"]
    assert all(row["evidence_ids"].split(";")[0].startswith("chembl:mechanism:") for row in rows)
    cited = {i for row in rows for i in row["evidence_ids"].split(";")}
    assert cited == {r.record_id for r in recs}  # every record is cited by some row; nothing cited that does not exist


def test_drug_rows_salt_form_reads_the_parent_s_withdrawal_atc_and_indications():
    # phentermine hydrochloride's mechanism row: the salt itself is not flagged and has no ATC code; phentermine,
    # its parent, is withdrawn (1981) and carries A08AA01 — the row must say so, citing both molecule records
    r = ChemblRetriever(_stub("mechanism_molecule_CHEMBL1200912", "molecule_CHEMBL1200912", "molecule_CHEMBL1574",
                              "drug_indication_parent_CHEMBL1574", "drug_warning_parent_CHEMBL1574"))
    (mech,) = r.mechanisms(PHENTERMINE_HCL)
    assert mech.payload["parent_molecule_chembl_id"] == PHENTERMINE
    salt, parent = r.molecule(PHENTERMINE_HCL), r.molecule(PHENTERMINE)
    inds, warns = r.indications(PHENTERMINE), r.warnings(PHENTERMINE)
    assert (salt.payload["withdrawn_flag"], salt.payload["atc_classifications"], parent.payload["withdrawn_flag"], parent.payload["atc_classifications"]) == (False, [], True, ["A08AA01"])
    (row,) = drug_rows([mech, salt, parent, *inds, *warns])
    assert (row["chembl_id"], row["name"], row["parent_chembl_id"], row["max_phase"], row["first_approval"]) == (PHENTERMINE_HCL, "PHENTERMINE HYDROCHLORIDE", PHENTERMINE, "4", "1973")
    assert (row["withdrawn"], row["black_box_warning"], row["warnings"], row["atc_codes"]) == ("Y", "N", "Withdrawn 1981 Sweden; United Kingdom; Mauritius", "A08AA01")
    assert row["indications"].startswith("Obesity [") and row["indications"].count("via CHEMBL1574") == 7 and "phase 4;" in row["indications"]
    assert row["evidence_ids"].split(";")[:3] == ["chembl:mechanism:886", f"chembl:{PHENTERMINE_HCL}", f"chembl:{PHENTERMINE}"]
    assert set(row["evidence_ids"].split(";")[3:]) == {x.record_id for x in inds + warns}
    # without the parent record the salt's own flags stand, and the family rows still join through the parent id
    (row,) = drug_rows([mech, salt, *inds, *warns])
    assert (row["withdrawn"], row["atc_codes"], row["warnings"]) == ("N", "", "Withdrawn 1981 Sweden; United Kingdom; Mauritius")
    # a parent that says unknown (-1) never hides the salt's own answer, and vice versa
    unknown = dataclasses.replace(parent, payload=dict(parent.payload, withdrawn_flag=None, black_box_warning=-1))
    assert [(x["withdrawn"], x["black_box_warning"]) for x in drug_rows([mech, salt, unknown])] == [("N", "N")]
    flagged = dataclasses.replace(salt, payload=dict(salt.payload, black_box_warning=1))
    assert [(x["withdrawn"], x["black_box_warning"]) for x in drug_rows([mech, flagged, parent])] == [("Y", "Y")]


def test_drug_rows_marks_a_complex_or_family_level_target():
    # the CFTR target served as a family (edit of a public record; the agent must see that a family-level
    # mechanism is a weaker claim about the gene)
    r = ChemblRetriever(_stub("mechanism_molecule_CHEMBL2010601", "target_symbol_CFTR"))
    (mech,) = r.mechanisms(IVACAFTOR)
    (target,) = r.targets_for_symbol("CFTR")
    (row,) = drug_rows([mech, target])
    assert (row["target_name"], row["target_type"]) == ("Cystic fibrosis transmembrane conductance regulator", "SINGLE PROTEIN")
    family = dataclasses.replace(target, payload=dict(target.payload, target_type="PROTEIN FAMILY"))
    (row,) = drug_rows([mech, family])
    assert (row["target_name"], row["target_type"]) == ("Cystic fibrosis transmembrane conductance regulator [PROTEIN FAMILY]", "PROTEIN FAMILY")
    (row,) = drug_rows([mech])
    assert (row["target_name"], row["target_type"], row["target_chembl_id"]) == ("", "", CFTR_TARGET)


def test_drug_rows_without_a_molecule_record_and_with_warnings():
    r = ChemblRetriever(_stub("mechanism_molecule_CHEMBL2010601", "drug_warning_parent_CHEMBL122"))
    (row,) = drug_rows(r.mechanisms(IVACAFTOR))  # mechanism alone: molecule columns blank, phase from the mechanism
    assert (row["name"], row["max_phase"], row["parent_chembl_id"], row["withdrawn"], row["target_name"], row["indications"]) == ("", "4", IVACAFTOR, "", "", "")
    assert row["evidence_ids"] == "chembl:mechanism:965"
    # warnings de-duplicated on (type, year, country): rofecoxib's eleven rows read as two lines
    warns = r.warnings(ROFECOXIB)
    (iva,) = r.mechanisms(IVACAFTOR)
    mech = dataclasses.replace(iva, payload=dict(iva.payload, molecule_chembl_id=ROFECOXIB, parent_molecule_chembl_id=ROFECOXIB))
    (row,) = drug_rows([mech, *warns])
    assert row["warnings"] == "Black Box Warning United States; Withdrawn 2004 Worldwide"
    assert row["evidence_ids"].split(";")[1:] == [w.record_id for w in warns]
    assert drug_rows([]) == []


def test_drug_rows_phase_is_the_molecule_s_unless_null_then_the_mechanism_s():
    r = ChemblRetriever(_stub("mechanism_molecule_CHEMBL2010601", "molecule_CHEMBL2010601"))
    (mech,) = r.mechanisms(IVACAFTOR)
    mol = r.molecule(IVACAFTOR)
    assert mol is not None and mol.payload["max_phase"] == "4.0" and mech.payload["max_phase"] == 4
    (row,) = drug_rows([mech, mol])
    assert row["max_phase"] == "4"
    nulled = dataclasses.replace(mol, payload=dict(mol.payload, max_phase=None))
    (row,) = drug_rows([mech, nulled])
    assert (row["max_phase"], row["name"]) == ("4", "IVACAFTOR")  # the mechanism's denormalised copy, not ""
    unknown = dataclasses.replace(mol, payload=dict(mol.payload, max_phase="-1.0"))
    assert drug_rows([mech, unknown])[0]["max_phase"] == "-1"  # ChEMBL's explicit "unknown" is kept as said


# ---------------------------------------------------------------- text searches


def test_search_mechanisms_by_text_returns_the_rows_their_molecules_and_the_search_record():
    """The mechanism table filtered on words of the mechanism itself — how stage 6
    finds a class of intervention without a gene symbol."""
    stub = _stub(MECHANISM_SEARCH, "molecule_CHEMBL2010601", "molecule_CHEMBL2103870", "molecule_CHEMBL2108184")
    records, total = ChemblRetriever(stub).search_mechanisms("conductance regulator", limit=3)
    fx = fixture(MECHANISM_SEARCH)
    assert _call(stub, "mechanism.json")["params"] == fx["request"]["params"] == {
        "mechanism_of_action__icontains": "conductance regulator", "limit": 3, "order_by": "mec_id"}
    assert total == json.loads(fx["text"])["page_meta"]["total_count"] == 13  # 3 of 13: the record says so
    search, *rest = records
    assert search.record_id.startswith("chembl-search:") and search.source == "chembl-search"
    assert search.source_version == VERSION and search.retrieved_at == fx["retrieved_at"]
    assert search.query == {"endpoint": "mechanism", "field": "mechanism_of_action__icontains",
                            "needle": "conductance regulator", "limit": 3,
                            "api": search.url}
    assert search.url == (API_URL + "mechanism.json?mechanism_of_action__icontains=conductance+regulator"
                          "&limit=3&order_by=mec_id")
    assert search.payload == {"sent": search.query["endpoint"] and {"endpoint": "mechanism",
                                                                    "field": "mechanism_of_action__icontains",
                                                                    "needle": "conductance regulator", "limit": 3},
                              "total_count": 13,
                              "ids": ["chembl:mechanism:964", "chembl:mechanism:965", "chembl:mechanism:2346"]}
    rows = [r for r in rest if r.query["endpoint"] == "mechanism"]
    molecules = [r for r in rest if r.query["endpoint"] == "molecule"]
    assert [r.record_id for r in rows] == ["chembl:mechanism:964", "chembl:mechanism:965", "chembl:mechanism:2346"]
    assert [r.record_id for r in molecules] == [f"chembl:{c}" for c in (IVACAFTOR, LUMACAFTOR, CROFELEMER)]  # sorted by id
    assert all("conductance regulator" in r.payload["mechanism_of_action"].lower() for r in rows)
    # each row is the same record the gene route makes, with the same detail URL
    (by_gene,) = [r for r in ChemblRetriever(_cftr_stub()).mechanisms_for_target(CFTR_TARGET) if r.record_id == "chembl:mechanism:965"]
    assert by_gene.query == rows[1].query and by_gene.payload == rows[1].payload
    assert rows[1].url == "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL2010601"
    # a search record identifies its request: same needle, same limit, same id
    again, _ = ChemblRetriever(_stub(MECHANISM_SEARCH, "molecule_CHEMBL2010601", "molecule_CHEMBL2103870",
                                     "molecule_CHEMBL2108184")).search_mechanisms("conductance regulator", limit=3)
    assert again[0].record_id == search.record_id and [r.to_json() for r in again] == [r.to_json() for r in records]


def test_search_indications_by_disease_name():
    stub = _stub(INDICATION_SEARCH, "molecule_CHEMBL2010601", f"molecule_{ATALUREN}", f"molecule_{LIPROTAMASE}")
    records, total = ChemblRetriever(stub).search_indications("cystic fibrosis", limit=3)
    fx = fixture(INDICATION_SEARCH)
    assert _call(stub, "drug_indication.json")["params"] == fx["request"]["params"] == {
        "mesh_heading__icontains": "cystic fibrosis", "limit": 3, "order_by": "drugind_id"}
    assert total == 135
    search, *rest = records
    assert search.payload["total_count"] == 135 and search.payload["sent"]["field"] == "mesh_heading__icontains"
    rows = [r for r in rest if r.query["endpoint"] == "drug_indication"]
    assert [r.record_id for r in rows] == ["chembl:indication:136867", "chembl:indication:136913", "chembl:indication:137003"]
    assert all(r.payload["mesh_heading"].lower() == "cystic fibrosis" for r in rows)
    molecules = {r.payload["molecule_chembl_id"]: r for r in rest if r.query["endpoint"] == "molecule"}
    assert set(molecules) == {ATALUREN, LIPROTAMASE, IVACAFTOR}
    assert molecules[ATALUREN].payload["pref_name"] == "ATALUREN" and phase_label(molecules[ATALUREN].payload["max_phase"]) == "4"
    assert rows[0].url == f"https://www.ebi.ac.uk/chembl/explore/compound/{ATALUREN}"


def test_a_row_without_the_needle_means_the_filter_was_ignored_and_is_refused():
    """An unknown filter name is silently ignored and the whole table comes back with
    HTTP 200 (``mechanism_of_actionx__icontains`` → 7561 rows live), so every row must
    contain the needle in the filtered field or nothing here can be cited."""
    fx = fixture(MECHANISM_SEARCH)
    body = json.loads(fx["text"])
    body["mechanisms"][1]["mechanism_of_action"] = "Sodium channel blocker"  # a row from another part of the table
    stub = _stub("molecule_CHEMBL2010601", "molecule_CHEMBL2103870", "molecule_CHEMBL2108184")
    stub.add(served(fx, body=body))
    with pytest.raises(ChemblError, match="ignored the mechanism_of_action__icontains filter"):
        ChemblRetriever(stub).search_mechanisms("conductance regulator", limit=3)
    # the needle matches case-insensitively, as the server's own icontains does: the same
    # rows answer a mixed-case needle (served here under its own request key, as the cache would)
    mixed = copy.deepcopy(fx)
    mixed["request"]["params"] = {**fx["request"]["params"], "mechanism_of_action__icontains": "CONDUCTANCE Regulator"}
    stub = _stub("molecule_CHEMBL2010601", "molecule_CHEMBL2103870", "molecule_CHEMBL2108184")
    stub.add(mixed)
    records, _ = ChemblRetriever(stub).search_mechanisms("CONDUCTANCE Regulator", limit=3)
    assert len(records) == 7 and records[0].payload["sent"]["needle"] == "CONDUCTANCE Regulator"


def test_a_short_needle_a_coordinate_or_a_bad_limit_is_refused_before_any_request():
    stub = _stub(MECHANISM_SEARCH)
    r = ChemblRetriever(stub)
    assert valid_needle("  conductance regulator ") == "conductance regulator" and MIN_NEEDLE_CHARS == 3
    for bad in ("", "  ", "cf", None):
        with pytest.raises(ValueError, match="at least 3 characters"):
            r.search_mechanisms(bad)  # type: ignore[arg-type]
    for coordinate in ("cystic fibrosis 7:117559590", "chr7-117559590", "g.117559590", "117559590"):
        with pytest.raises(ValueError, match="genomic coordinate|five or more digits"):
            r.search_indications(coordinate)
    for limit in (0, -1, 26, 5.0, True, "5"):
        with pytest.raises(ValueError, match="between 1 and 25"):
            r.search_mechanisms("conductance regulator", limit=limit)  # type: ignore[arg-type]
    assert _paths(stub) == []  # nothing was sent
    assert (DEFAULT_CHEMBL_SEARCH, MAX_CHEMBL_SEARCH) == (10, 25)


def test_a_page_longer_than_the_limit_or_a_repeated_key_is_refused():
    fx = fixture(MECHANISM_SEARCH)
    body = json.loads(fx["text"])
    stub = _stub("molecule_CHEMBL2010601", "molecule_CHEMBL2103870", "molecule_CHEMBL2108184")
    stub.add(served(fx, body={**body, "mechanisms": body["mechanisms"] + [body["mechanisms"][0]]}))
    with pytest.raises(ChemblError, match="served 4 rows for limit 3"):
        ChemblRetriever(stub).search_mechanisms("conductance regulator", limit=3)
    twice = {**body, "mechanisms": [body["mechanisms"][0], body["mechanisms"][0], body["mechanisms"][1]]}
    stub = _stub("molecule_CHEMBL2010601", "molecule_CHEMBL2103870", "molecule_CHEMBL2108184")
    stub.add(served(fx, body=twice))
    with pytest.raises(ChemblError, match="served mec_id=964 twice"):
        ChemblRetriever(stub).search_mechanisms("conductance regulator", limit=3)


def test_search_fixtures_replay_offline_through_the_real_client(tmp_path: Path):
    cache = _cache_from_fixtures(tmp_path / "cache")
    r = ChemblRetriever(Http(cache, offline=True))
    records, total = r.search_mechanisms("conductance regulator", limit=3)
    assert total == 13 and [x.record_id for x in records][:2] == [records[0].record_id, "chembl:mechanism:964"]
    indications, total = r.search_indications("cystic fibrosis", limit=3)
    assert total == 135 and len(indications) == 7
    assert SEARCH_FIELDS == {"mechanism": "mechanism_of_action__icontains", "drug_indication": "mesh_heading__icontains"}


def test_phase_label_unifies_the_three_spellings():
    assert phase_label("4.0") == phase_label(4) == phase_label(4.0) == "4"
    assert phase_label("0.5") == "0.5" and phase_label("-1.0") == "-1" and phase_label(2) == "2"
    assert phase_label(None) == "" and phase_label("") == "" and phase_label("n/a") == "n/a"


# ---------------------------------------------------------------- live (opt-in)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ENGINE_LIVE_TESTS"), reason="set ENGINE_LIVE_TESTS=1 to hit www.ebi.ac.uk")
def test_live_public_entities(tmp_path: Path):
    http = Http(HttpCache(tmp_path / "cache"), limiter=RateLimiter(default_per_second=3.0))
    r = ChemblRetriever(http)
    assert r.version().startswith("ChEMBL_") and "(" in r.version()
    iva = r.molecule(IVACAFTOR)
    assert iva is not None and iva.payload["pref_name"] == "IVACAFTOR" and phase_label(iva.payload["max_phase"]) == "4"
    assert r.molecule(UNKNOWN) is None
    (luma,) = r.mechanisms(LUMACAFTOR)
    assert luma.payload["variant_sequence"]["mutation"] == "F508del"
    recs = r.drugs_for_target_symbol("CFTR")
    rows = drug_rows(recs)
    names = {row["name"] for row in rows}
    assert {"IVACAFTOR", "LUMACAFTOR", "TEZACAFTOR", "ELEXACAFTOR"} <= names
    assert any(row["variant_mutation"] == "F508del" for row in rows)
    assert all(row["target_type"] == "SINGLE PROTEIN" for row in rows)  # the CFTR/GOPC interaction target is excluded
    assert r.drugs_for_target_symbol("MTHFR") == []
    (bub1b,) = r.drugs_for_target_symbol("BUB1B")
    assert bub1b.query["endpoint"] == "target"
    assert len(r.warnings(ROFECOXIB)) >= 2 and warning_summary(r.warnings(ROFECOXIB))
    # a salt form reads its parent's withdrawal and ATC code
    (mech,) = r.mechanisms(PHENTERMINE_HCL)
    (row,) = drug_rows([mech, r.molecule(PHENTERMINE_HCL), r.molecule(PHENTERMINE), *r.indications(PHENTERMINE), *r.warnings(PHENTERMINE)])
    assert row["withdrawn"] == "Y" and row["warnings"].startswith("Withdrawn 1981") and "A08AA01" in row["atc_codes"]
    # the text searches (public terms only)
    mechanisms, total = r.search_mechanisms("conductance regulator", limit=3)
    assert total >= 13 and len(mechanisms) >= 4
    assert all("conductance regulator" in m.payload["mechanism_of_action"].lower()
               for m in mechanisms if m.query["endpoint"] == "mechanism")
    indications, total = r.search_indications("cystic fibrosis", limit=3)
    assert total >= 100 and any(i.query["endpoint"] == "molecule" for i in indications)
    # a rerun with the warm cache is byte-identical and needs nothing from outside
    again = ChemblRetriever(Http(HttpCache(tmp_path / "cache"), offline=True)).drugs_for_target_symbol("CFTR")
    assert [x.to_json() for x in again] == [x.to_json() for x in recs]
