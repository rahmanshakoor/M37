"""Open Targets retriever against recorded real responses (public genes only).

The fixtures in ``tests/fixtures/opentargets/`` are the engine's own HTTP cache entries
from two live sessions on 2026-09-12 (public genes only): ``meta``, the target block for
CFTR and TP53 and for the unknown id ENSG00000000000, the drug block for CFTR (13 rows),
TP53 (9 rows, four of them MDM2-complex inhibitors) and MTHFR (0 rows), the top-5
associations for CFTR and TP53, and ``mapIds`` for CFTR, cftr, TP53, P53 (a synonym) and
a made-up symbol — nothing else was ever sent. A stub Http answers each request with the
recorded bytes for the exact body the module builds, so a change to a query text fails
here before it fails live; an id that was never recorded gets the server's verified
answer for an unknown gene, ``{"data":{"target":null}}``. Tests that need the real
:class:`~engine.retrieve.http.Http` seed a real :class:`~engine.retrieve.http.HttpCache`
from the same files: one replays offline, proving a warm cache needs no network; others
poison one entry and stand in for ``urlopen`` with the recorded bytes, proving a cached
failure is re-fetched once and healed, and that a cache spanning a release is refused.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from engine.medicine.opentargets import (API_URL, ASSOCIATION_COLUMNS, ASSOCIATIONS_QUERY, DEFAULT_DISEASE_SIZE, DRUG_COLUMNS,
                                         DRUGS_QUERY, MAP_QUERY, META_QUERY, PER_SECOND, TARGET_COLUMNS, TARGET_QUERY,
                                         OpenTargetsError, OpenTargetsRetriever, association_url, drug_url,
                                         extract_association, extract_drug, extract_target, normalise_id, target_url)
from engine.retrieve.http import Http, HttpCache, HttpError, RateLimiter, Response
from engine.retrieve.store import EvidenceRecord

FIXTURES = Path(__file__).parent / "fixtures" / "opentargets"
CFTR, TP53, MTHFR, ABSENT = "ENSG00000001626", "ENSG00000141510", "ENSG00000177000", "ENSG00000000000"
VERSION = "Open Targets Platform 26.06 (API 26.6.3, platform2606)"
HOST = "api.platform.opentargets.org"
QUERIES = {META_QUERY, TARGET_QUERY, DRUGS_QUERY, ASSOCIATIONS_QUERY, MAP_QUERY}
NEXT_META = {"data": {"meta": {"name": "Open Targets Platform", "product": "platform",
                               "apiVersion": {"x": "26", "y": "9", "z": "0", "suffix": None},
                               "dataVersion": {"year": "26", "month": "09", "iteration": None}, "dataPrefix": "platform2609"}}}
NEXT_VERSION = "Open Targets Platform 26.09 (API 26.9.0, platform2609)"


def _canon(body: dict) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _recorded() -> dict[str, dict]:
    """canonical request body → recorded cache entry."""
    out: dict[str, dict] = {}
    for p in sorted(FIXTURES.glob("*.json")):
        d = json.loads(p.read_text())
        out[_canon(d["request"]["body"])] = d
    return out


class StubHttp:
    """Serves the recorded fixtures for the exact bodies recorded; an unrecorded id on a
    target-shaped query gets the server's verified unknown-gene shape; anything else
    (a changed query text, another URL, a GET) is an error, never a network call."""

    def __init__(self):
        self.recorded = _recorded()
        self.calls: list[dict] = []
        self.limiter = RateLimiter()

    def request(self, method: str, url: str, *, params=None, json_body=None, **kw) -> Response:
        self.calls.append({"method": method, "url": url, "body": json_body, **kw})
        assert method == "POST" and url == API_URL and params is None
        assert json_body["query"] in QUERIES, "query text drifted from what was recorded live"
        d = self.recorded.get(_canon(json_body))
        if d is None:
            assert json_body["query"] in (TARGET_QUERY, DRUGS_QUERY, ASSOCIATIONS_QUERY), "unrecorded request"
            return Response(200, '{"data":{"target":null}}', "2026-09-12T18:54:28+00:00", False, "stub-null", {})
        return Response(d["status"], d["text"], d["retrieved_at"], False, "stub-recorded", d["headers"])

    def get(self, url: str, **kw) -> Response:
        return self.request("GET", url, **kw)

    def post(self, url: str, json_body, **kw) -> Response:
        return self.request("POST", url, json_body=json_body, **kw)


class CannedHttp(StubHttp):
    """Answers every non-meta request with one fixed body (or raises), for error shapes."""

    def __init__(self, text: str = "", status: int = 200, raise_: Exception | None = None, canned_meta: bool = False):
        super().__init__()
        self.text, self.status, self.raise_, self.canned_meta = text, status, raise_, canned_meta

    def request(self, method, url, *, params=None, json_body=None, **kw) -> Response:
        if json_body["query"] == META_QUERY and not self.canned_meta:
            return super().request(method, url, params=params, json_body=json_body, **kw)
        self.calls.append({"method": method, "url": url, "body": json_body, **kw})
        if self.raise_ is not None:
            raise self.raise_
        return Response(self.status, self.text, "2026-09-12T00:00:00+00:00", False, "canned", {})


def _canned(body: dict | str, status: int = 200, **kw) -> OpenTargetsRetriever:
    text = body if isinstance(body, str) else json.dumps(body)
    return OpenTargetsRetriever(CannedHttp(text=text, status=status, **kw))


def _sent(stub: StubHttp, query: str) -> list[dict]:
    return [c["body"] for c in stub.calls if c["body"]["query"] == query]


# ---------------------------------------------------------------- version

def test_version_from_meta_once():
    stub = StubHttp()
    r = OpenTargetsRetriever(stub)
    assert r.version() == VERSION
    r.target(CFTR)
    r.known_drugs(CFTR)
    assert r.version() == VERSION
    meta = [c for c in stub.calls if c["body"]["query"] == META_QUERY]
    # observed once per retriever: the cache's answer and, online, a live one that must agree
    assert [c.get("cache_ok", True) for c in meta] == [True, False]
    stub.offline = True
    OpenTargetsRetriever(stub).version()
    assert [c.get("cache_ok", True) for c in stub.calls if c["body"]["query"] == META_QUERY] == [True, False, True]


class ReleaseHttp(StubHttp):
    """The recorded 26.06 ``meta`` from the cache, a later release when asked live."""

    def request(self, method, url, *, params=None, json_body=None, cache_ok=True, **kw) -> Response:
        if json_body["query"] == META_QUERY and not cache_ok:
            self.calls.append({"method": method, "url": url, "body": json_body, "cache_ok": cache_ok, **kw})
            return Response(200, json.dumps(NEXT_META), "2026-12-01T00:00:00+00:00", False, "live-meta", {})
        return super().request(method, url, params=params, json_body=json_body, cache_ok=cache_ok, **kw)


def test_version_refuses_a_cache_that_spans_a_release():
    """Nothing in a target/drug/association payload says which release made it, so a
    cache filled under 26.06 must not stamp 26.09 payloads — and the check runs before
    the first gene request, so the refused cache holds no 26.09 gene response."""
    h = ReleaseHttp()
    r = OpenTargetsRetriever(h)
    with pytest.raises(OpenTargetsError, match=r"filled under .*26\.06.* but the server now reports .*26\.09.*remove the api\.platform\.opentargets\.org entries"):
        r.target(CFTR)
    assert [c["body"]["query"] for c in h.calls] == [META_QUERY, META_QUERY]
    for call in (lambda: r.known_drugs(CFTR), lambda: r.associated_diseases(CFTR, size=5), lambda: r.resolve_symbol("CFTR")):
        with pytest.raises(OpenTargetsError, match="mixing releases"):
            call()
    assert {c["body"]["query"] for c in h.calls} == {META_QUERY}  # every entry point observes the version first
    h.offline = True  # offline: the cache's release is the one cited, and the live check is not made
    assert OpenTargetsRetriever(h).version() == VERSION
    assert OpenTargetsRetriever(h).target(CFTR).source_version == VERSION


def test_version_check_is_a_live_observation_not_a_replay():
    """The live observation must bypass the cache: the same meta body under
    ``cache_ok=True`` twice would not detect a release boundary."""
    stub = StubHttp()
    OpenTargetsRetriever(stub).version()
    live = [c for c in stub.calls if c["body"]["query"] == META_QUERY and c.get("cache_ok", True) is False]
    assert len(live) == 1 and live[0]["body"] == {"query": META_QUERY, "variables": {}}


def test_version_parts_must_be_present():
    with pytest.raises(OpenTargetsError, match="version part"):
        _canned({"data": {"meta": {"apiVersion": {"x": "26", "y": "6", "z": None, "suffix": None},
                                   "dataVersion": {"year": "26", "month": "06", "iteration": None},
                                   "dataPrefix": "platform2606"}}}, canned_meta=True).version()
    with pytest.raises(OpenTargetsError, match="dataPrefix"):
        _canned({"data": {"meta": {"apiVersion": {"x": "26", "y": "6", "z": "3", "suffix": None},
                                   "dataVersion": {"year": "26", "month": "06", "iteration": None},
                                   "dataPrefix": None}}}, canned_meta=True).version()


def test_version_label_carries_suffix_and_iteration_when_present():
    r = _canned({"data": {"meta": {"apiVersion": {"x": "27", "y": "1", "z": "0", "suffix": "rc1"},
                                   "dataVersion": {"year": "27", "month": "01", "iteration": "2"},
                                   "dataPrefix": "platform2701"}}}, canned_meta=True)
    assert r.version() == "Open Targets Platform 27.01.2 (API 27.1.0-rc1, platform2701)"


# ---------------------------------------------------------------- target

def test_target_record_from_recorded_response():
    stub = StubHttp()
    r = OpenTargetsRetriever(stub)
    rec = r.target(CFTR)
    assert rec is not None
    fx = _fixture(f"target_{CFTR}")
    assert _sent(stub, TARGET_QUERY) == [fx["request"]["body"]]  # byte-identical to the live request
    assert rec.record_id == f"opentargets:{CFTR}"
    assert rec.source == "opentargets"
    assert rec.source_version == VERSION
    assert rec.query == {"api": API_URL, "graphql": TARGET_QUERY, "variables": {"id": CFTR}}
    assert rec.url == target_url(CFTR) == f"https://platform.opentargets.org/target/{CFTR}"
    assert rec.retrieved_at == fx["retrieved_at"]  # from the Response, not the clock
    assert rec.payload == json.loads(fx["text"])["data"]["target"]  # verbatim: no derived keys
    assert rec.payload["approvedSymbol"] == "CFTR"
    assert rec.payload["tractability"][0] == {"modality": "SM", "label": "Approved Drug", "value": True}


def test_extract_target_cftr():
    r = OpenTargetsRetriever(StubHttp())
    cols = extract_target(r.target(CFTR))
    assert tuple(cols) == TARGET_COLUMNS
    assert cols["ensembl_id"] == CFTR and cols["symbol"] == "CFTR"
    assert cols["name"] == "CF transmembrane conductance regulator" and cols["biotype"] == "protein_coding"
    assert (cols["chromosome"], cols["start"], cols["end"], cols["strand"]) == ("7", "117287120", "117715971", "1")
    assert cols["tractability"].startswith("SM:Approved Drug;SM:Structure with Ligand;")
    assert "AB:UniProt loc high conf" in cols["tractability"]
    assert cols["tractable_modalities"] == "AB;PR;SM"  # only modalities with a true flag; OC has none for CFTR
    assert cols["n_pathways"] == "11" and cols["pathway_ids"].count(";") == 10
    assert "R-HSA-5678895" in cols["pathway_ids"].split(";")
    assert "Defective CFTR causes cystic fibrosis" in cols["pathways"].split(";")
    assert cols["n_drug_rows"] == "13" and cols["n_associated_diseases"] == "1987"
    assert r.extract(r.target(CFTR)) == cols  # the dispatching method agrees


def test_extract_target_tp53_minus_strand_and_other_modalities():
    cols = extract_target(OpenTargetsRetriever(StubHttp()).target(TP53))
    assert (cols["symbol"], cols["chromosome"], cols["start"], cols["end"], cols["strand"]) == ("TP53", "17", "7661779", "7687546", "-1")
    assert cols["tractable_modalities"] == "OC;PR;SM"
    assert "OC:Advanced Clinical" in cols["tractability"] and "SM:Approved Drug" not in cols["tractability"]
    assert (cols["n_pathways"], cols["n_drug_rows"], cols["n_associated_diseases"]) == ("46", "9", "5638")


def test_absent_target_is_none_not_an_error():
    stub = StubHttp()
    r = OpenTargetsRetriever(stub)
    assert r.target(ABSENT) is None
    assert _sent(stub, TARGET_QUERY) == [_fixture(f"target_{ABSENT}")["request"]["body"]]
    assert r.known_drugs(ABSENT) == []
    assert r.associated_diseases(ABSENT, size=5) == []
    assert r.extract(None) == {}


def test_ensembl_ids_are_normalised_before_asking():
    stub = StubHttp()
    r = OpenTargetsRetriever(stub)
    assert normalise_id("ENSG00000001626.16") == CFTR == normalise_id(" ensg00000001626 ")
    a = r.target("ENSG00000001626.16")
    b = r.target("ensg00000001626")
    assert a is not None and a.to_json() == b.to_json()
    assert {c["body"]["variables"]["id"] for c in stub.calls if c["body"]["query"] == TARGET_QUERY} == {CFTR}


def test_symbols_and_junk_are_refused_before_any_request():
    stub = StubHttp()
    r = OpenTargetsRetriever(stub)
    for bad in ("CFTR", "", "ENSG1", "ENSG00000001626X", "ENST00000003084", "NM_000492"):
        with pytest.raises(ValueError, match="not an Ensembl gene id"):
            r.target(bad)
        with pytest.raises(ValueError):
            r.known_drugs(bad)
        with pytest.raises(ValueError):
            r.associated_diseases(bad)
    assert stub.calls == []


# ---------------------------------------------------------------- known drugs

def test_known_drugs_cftr_best_stage_first():
    stub = StubHttp()
    r = OpenTargetsRetriever(stub)
    drugs = r.known_drugs(CFTR)
    fx = _fixture(f"drugs_{CFTR}")
    assert _sent(stub, DRUGS_QUERY) == [fx["request"]["body"]]
    names = [(extract_drug(d)["name"], extract_drug(d)["max_clinical_stage"]) for d in drugs]
    assert names == [
        ("CROFELEMER", "APPROVAL"), ("DEUTIVACAFTOR", "APPROVAL"), ("ELEXACAFTOR", "APPROVAL"), ("IVACAFTOR", "APPROVAL"),
        ("LUMACAFTOR", "APPROVAL"), ("TEZACAFTOR", "APPROVAL"), ("VANZACAFTOR", "APPROVAL"), ("BAMOCAFTOR", "PHASE_3"),
        ("GALICAFTOR", "PHASE_2"), ("ICENTICAFTOR", "PHASE_2"), ("IOWH-032", "PHASE_2"), ("NAVOCAFTOR", "PHASE_2"),
        ("OLACAFTOR", "PHASE_2"),
    ]
    rows = {row["drug"]["id"]: row for row in json.loads(fx["text"])["data"]["target"]["drugAndClinicalCandidates"]["rows"]}
    assert rows[list(rows)[0]]["drug"]["name"] == "GALICAFTOR"  # the server's order is by hash, not by stage
    for rec in drugs:
        chembl = rec.payload["drug"]["id"]
        assert rec.record_id == f"opentargets:drug:{CFTR}:{chembl}"
        assert rec.source == "opentargets" and rec.source_version == VERSION
        assert rec.query == {"api": API_URL, "graphql": DRUGS_QUERY, "variables": {"id": CFTR}, "chemblId": chembl}
        assert rec.url == drug_url(chembl) == f"https://platform.opentargets.org/drug/{chembl}"
        assert rec.retrieved_at == fx["retrieved_at"]
        assert rec.payload == rows[chembl]  # the raw row, verbatim


def test_extract_drug_scopes_mechanisms_to_the_queried_gene():
    r = OpenTargetsRetriever(StubHttp())
    by_name = {extract_drug(d)["name"]: extract_drug(d) for d in r.known_drugs(CFTR)}
    assert tuple(by_name["CROFELEMER"]) == DRUG_COLUMNS
    c = by_name["CROFELEMER"]  # carries an ANO1 mechanism too; that one must not land on CFTR
    assert c["chembl_id"] == "CHEMBL2108184" and c["drug_type"] == "Small molecule"
    assert c["mechanism_of_action"] == "Cystic fibrosis transmembrane conductance regulator inhibitor"
    assert c["action_type"] == "INHIBITOR"
    assert (c["moa_targets"], c["moa_target_name"]) == ("CFTR", "Cystic fibrosis transmembrane conductance regulator")
    assert c["n_mechanisms_other_targets"] == "1"
    assert (c["approved_indications"], c["approved_indication_ids"]) == ("Diarrhea", "HP_0002014")
    assert c["report_types"] == "CLINICAL_TRIAL;CURATED_RESOURCE;DRUG_LABEL;REGULATORY_AGENCY"
    assert c["report_sources"] == "AACT;DailyMed;FDA;TTD"
    assert c["report_statuses"] == "COMPLETED;RECRUITING;TERMINATED"
    assert c["n_reports"] == "16" and len(c["trial_ids"].split(";")) == 13
    assert all(t.startswith("NCT") and len(t) == 11 for t in c["trial_ids"].split(";"))  # upper-cased, no curated ids
    assert "hiv-associated diarrhoea" in c["conditions"].split(";")  # unmapped source text survives as a condition

    i = by_name["IVACAFTOR"]  # two mechanism rows against the same target: both kept, in order
    assert i["mechanism_of_action"] == ("Cystic fibrosis transmembrane conductance regulator positive modulator;"
                                        "Cystic fibrosis transmembrane conductance regulator activator")
    assert i["action_type"] == "POSITIVE MODULATOR;ACTIVATOR" and i["n_mechanisms_other_targets"] == "0"
    assert i["moa_targets"] == "CFTR"  # the same single target on both rows, reported once
    assert (i["max_clinical_stage"], i["drug_max_clinical_stage"]) == ("APPROVAL", "APPROVAL")
    assert i["approved_indication_ids"] == "MONDO_0015796;MONDO_0009061;MONDO_0005087"  # by name: acute lung injury, CF, ...
    assert "cystic fibrosis" in i["approved_indications"].split(";")
    assert i["n_reports"] == "145" and "NCT00457821" in i["trial_ids"].split(";")
    assert "APPROVED_FOR_MARKETING" in i["report_statuses"].split(";")

    t = by_name["TEZACAFTOR"]  # trial conditions are not indications: 'F508del mutation' is a condition only
    assert "F508del mutation" in t["conditions"].split(";") and "F508del mutation" not in t["approved_indications"]

    o = by_name["OLACAFTOR"]
    assert (o["approved_indications"], o["approved_indication_ids"], o["n_reports"], o["trial_ids"]) == ("", "", "1", "NCT02951182")
    assert o["conditions"] == "cystic fibrosis" and o["report_statuses"] == "COMPLETED"


def test_known_drugs_empty_for_a_gene_without_any():
    stub = StubHttp()
    r = OpenTargetsRetriever(stub)
    assert r.known_drugs(MTHFR) == []
    assert _sent(stub, DRUGS_QUERY) == [_fixture(f"drugs_{MTHFR}")["request"]["body"]]


def test_extract_drug_shows_a_protein_complex_target_tp53():
    """The MDM2 inhibitors reach TP53 only through the p53/Mdm2 complex: the row is kept
    (TP53 is among its targets) but the table must say the drug binds MDM2 too."""
    stub = StubHttp()
    r = OpenTargetsRetriever(stub)
    drugs = r.known_drugs(TP53)
    assert _sent(stub, DRUGS_QUERY) == [_fixture(f"drugs_{TP53}")["request"]["body"]]
    by_name = {extract_drug(d)["name"]: extract_drug(d) for d in drugs}
    assert len(by_name) == 9 and extract_target(r.target(TP53))["n_drug_rows"] == "9"
    for name in ("IDASANUTLIN", "NAVTEMADLIN", "SIREMADLIN", "ALRIZOMADLIN"):
        d = by_name[name]
        assert d["mechanism_of_action"] == "Tumour suppressor p53/oncoprotein Mdm2 inhibitor"
        assert d["action_type"] == "INHIBITOR" and d["n_mechanisms_other_targets"] == "0"
        assert d["moa_targets"] == "MDM2,TP53"  # sorted symbols; two of them = a complex, not TP53 alone
        assert d["moa_target_name"] == "Tumour suppressor p53/oncoprotein Mdm2"
    assert by_name["IDASANUTLIN"]["chembl_id"] == "CHEMBL2402737"
    e = by_name["EPRENETAPOPT"]  # a direct p53 binder, for contrast
    assert (e["moa_targets"], e["moa_target_name"], e["action_type"]) == ("TP53", "Cellular tumor antigen p53", "STABILISER")
    assert by_name["TEPRASIRAN"]["moa_targets"] == "TP53" and by_name["TEPRASIRAN"]["moa_target_name"] == "p53 mRNA"
    stages = [extract_drug(d)["max_clinical_stage"] for d in drugs]
    assert stages == ["PHASE_3"] * 5 + ["PHASE_2"] * 4


def test_moa_targets_fall_back_to_the_id_and_skip_empty_targets():
    row = _drug_row("CHEMBL5", "Q", "PHASE_1")
    row["drug"]["mechanismsOfAction"] = {"rows": [
        {"mechanismOfAction": "X inhibitor", "actionType": "INHIBITOR", "targetName": "X complex",
         "targets": [{"id": CFTR, "approvedSymbol": None}, {"id": TP53, "approvedSymbol": "TP53"}, {"id": None, "approvedSymbol": None}]},
        {"mechanismOfAction": "Y agonist", "actionType": "AGONIST", "targetName": "Y", "targets": None},
    ]}
    (rec,) = _canned(_drugs_body([row])).known_drugs(CFTR)
    cols = extract_drug(rec)
    assert cols["moa_targets"] == f"{CFTR},TP53" and cols["moa_target_name"] == "X complex"
    assert cols["n_mechanisms_other_targets"] == "1"  # the row with no targets does not name CFTR


def test_count_and_rows_must_agree():
    """A server that starts paging (or drops rows) must not read as a gene with fewer
    drugs or associations: count is checked against the rows delivered."""
    with pytest.raises(OpenTargetsError, match="count 13 but 0 rows"):
        _canned({"data": {"target": {"id": CFTR, "drugAndClinicalCandidates": {"count": 13, "rows": []}}}}).known_drugs(CFTR)
    body = _drugs_body([_drug_row("CHEMBL1", "A"), _drug_row("CHEMBL2", "B")])
    body["data"]["target"]["drugAndClinicalCandidates"]["count"] = 1
    with pytest.raises(OpenTargetsError, match="count 1 but 2 rows"):
        _canned(body).known_drugs(CFTR)
    assoc_row = {"disease": {"id": "MONDO_0000001", "name": "x"}, "score": 0.5, "datatypeScores": [], "datasourceScores": []}
    with pytest.raises(OpenTargetsError, match=r"count 1987 but 0 rows delivered \(expected 5\)"):
        _canned({"data": {"target": {"id": CFTR, "associatedDiseases": {"count": 1987, "rows": []}}}}).associated_diseases(CFTR, size=5)
    # page 0 of a paged block: min(count, size) rows are the full answer
    ok = {"data": {"target": {"id": CFTR, "associatedDiseases": {"count": 617, "rows": [assoc_row]}}}}
    assert len(_canned(ok).associated_diseases(CFTR, size=1)) == 1
    ok["data"]["target"]["associatedDiseases"]["count"] = 1
    assert len(_canned(ok).associated_diseases(CFTR, size=700)) == 1
    for bad_count in (None, "13", 13.0, True, -1):
        with pytest.raises(OpenTargetsError, match="count is not a non-negative integer"):
            _canned({"data": {"target": {"id": CFTR, "drugAndClinicalCandidates": {"count": bad_count, "rows": []}}}}).known_drugs(CFTR)


def _drug_row(chembl: str, name: str, stage: str = "PHASE_2", drug: bool = True) -> dict:
    return {"id": f"row-{chembl}", "maxClinicalStage": stage,
            "drug": {"id": chembl, "name": name, "drugType": "Small molecule", "maximumClinicalStage": stage,
                     "mechanismsOfAction": {"rows": []}} if drug else None,
            "diseases": [], "clinicalReports": []}


def _drugs_body(rows: list[dict], ensg: str = CFTR) -> dict:
    return {"data": {"target": {"id": ensg, "drugAndClinicalCandidates": {"count": len(rows), "rows": rows}}}}


def test_known_drugs_rows_without_a_drug_are_skipped_and_stages_ordered():
    rows = [_drug_row("CHEMBL2", "B", "PRECLINICAL"), _drug_row("CHEMBL9", "Z", "NOVEL_STAGE"), _drug_row("CHEMBL3", "C", drug=False),
            _drug_row("CHEMBL1", "A", "PHASE_2_3"), _drug_row("CHEMBL4", "A", "PHASE_2_3"), _drug_row("CHEMBL1", "A", "APPROVAL")]
    drugs = _canned(_drugs_body(rows)).known_drugs(CFTR)
    ids = [d.record_id for d in drugs]
    # PHASE_2_3 and PRECLINICAL are real stages; an unknown one sorts last; ties break by name then id;
    # the duplicate CHEMBL1 keeps its best-staged row whichever the server sent first
    # (per (drug, target) rows never repeat live; the server's own order is by hash).
    assert ids == [f"opentargets:drug:{CFTR}:CHEMBL1", f"opentargets:drug:{CFTR}:CHEMBL4",
                   f"opentargets:drug:{CFTR}:CHEMBL2", f"opentargets:drug:{CFTR}:CHEMBL9"]
    assert drugs[0].payload["maxClinicalStage"] == "APPROVAL"
    assert extract_drug(drugs[3])["max_clinical_stage"] == "NOVEL_STAGE"
    rows.reverse()  # the APPROVAL row first: same answer
    assert [d.payload for d in _canned(_drugs_body(rows)).known_drugs(CFTR)] == [d.payload for d in drugs]


def test_extract_drug_tolerates_null_leaves():
    row = _drug_row("CHEMBL7", "X", "APPROVAL")
    row["drug"]["mechanismsOfAction"] = None
    row["diseases"] = [{"diseaseFromSource": "solid tumour/cancer", "disease": None}]
    row["clinicalReports"] = [
        {"id": "nct00000001", "type": "CLINICAL_TRIAL", "source": "AACT", "url": None, "trialPhase": None, "clinicalStage": "APPROVAL",
         "trialOverallStatus": None, "year": None, "trialWhyStopped": None, "diseases": [{"disease": None}]},
        {"id": "label/x.pdf", "type": "REGULATORY_AGENCY", "source": "FDA", "url": None, "trialPhase": None, "clinicalStage": "APPROVAL",
         "trialOverallStatus": None, "year": None, "trialWhyStopped": None, "diseases": []},
    ]
    (rec,) = _canned(_drugs_body([row])).known_drugs(CFTR)
    cols = extract_drug(rec)
    assert cols["mechanism_of_action"] == "" and cols["n_mechanisms_other_targets"] == "0"
    assert cols["conditions"] == "solid tumour/cancer"
    assert (cols["approved_indications"], cols["approved_indication_ids"]) == ("", "")
    assert cols["trial_ids"] == "NCT00000001" and cols["report_statuses"] == "" and cols["n_reports"] == "2"


# ---------------------------------------------------------------- associated diseases

def test_associated_diseases_cftr_top5():
    stub = StubHttp()
    r = OpenTargetsRetriever(stub)
    assoc = r.associated_diseases(CFTR, size=5)
    fx = _fixture(f"associations_{CFTR}_5")
    assert _sent(stub, ASSOCIATIONS_QUERY) == [fx["request"]["body"]]
    rows = json.loads(fx["text"])["data"]["target"]["associatedDiseases"]["rows"]
    assert [a.payload["disease"]["id"] for a in assoc] == ["MONDO_0009061", "MONDO_0010178", "MONDO_0008185", "MONDO_0008887", "MONDO_0018801"]
    for rec, row in zip(assoc, rows):
        did = row["disease"]["id"]
        assert rec.record_id == f"opentargets:association:{CFTR}:{did}"
        assert rec.query == {"api": API_URL, "graphql": ASSOCIATIONS_QUERY, "variables": {"id": CFTR, "size": 5},
                             "diseaseId": did, "count": 1987}  # 5 of 1987: the truncation is on the record
        assert rec.url == association_url(CFTR, did) == f"https://platform.opentargets.org/evidence/{CFTR}/{did}"
        assert rec.source_version == VERSION and rec.retrieved_at == fx["retrieved_at"]
        assert rec.payload == row
    scores = [a.payload["score"] for a in assoc]
    assert scores == sorted(scores, reverse=True)


def test_extract_association():
    r = OpenTargetsRetriever(StubHttp())
    cf = extract_association(r.associated_diseases(CFTR, size=5)[0])
    assert tuple(cf) == ASSOCIATION_COLUMNS
    assert (cf["ensembl_id"], cf["disease_id"], cf["disease_name"]) == (CFTR, "MONDO_0009061", "cystic fibrosis")
    assert cf["score"] == "0.9182039330240603" and float(cf["score"]) == 0.9182039330240603  # repr, no rounding
    assert cf["genetic_association"] == "0.9678771377226557" and cf["clinical"] == "0.993501369004063"
    assert cf["somatic_mutation"] == "" and cf["rna_expression"] == ""  # a datatype the row lacks is blank
    assert cf["datatype_scores"].startswith("clinical=0.993501369004063;genetic_literature=")
    assert "eva=0.9989021794951988" in cf["datasource_scores"].split(";")
    tp = [extract_association(a) for a in r.associated_diseases(TP53, size=5)]
    assert tp[0]["disease_name"] == "Li-Fraumeni syndrome" and tp[0]["somatic_mutation"] == "0.4377101742803672"
    assert r.extract(r.associated_diseases(TP53, size=5)[0]) == tp[0]


def test_associated_diseases_size_is_guarded_before_any_request():
    stub = StubHttp()
    r = OpenTargetsRetriever(stub)
    for size in (0, -1, 3001):
        with pytest.raises(ValueError, match="1..3000"):
            r.associated_diseases(CFTR, size=size)
        with pytest.raises(ValueError, match="1..3000"):
            OpenTargetsRetriever(stub, disease_size=size)
    for size in (5.0, True, "5"):  # a GraphQL Int! — and 5.0 would be a different cache key from 5
        with pytest.raises(ValueError, match="must be an int"):
            r.associated_diseases(CFTR, size=size)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="must be an int"):
            OpenTargetsRetriever(stub, disease_size=size)  # type: ignore[arg-type]
    assert stub.calls == []


def test_disease_size_is_an_instance_knob_reported_by_params():
    """The page size is in every association record's bytes, so the one a run uses is
    set on the retriever and reported by ``params`` — a manifest reproduces the records."""
    stub = StubHttp()
    r = OpenTargetsRetriever(stub, disease_size=5)
    assert r.params()["disease_size"] == 5 and OpenTargetsRetriever(stub).params()["disease_size"] == DEFAULT_DISEASE_SIZE == 25
    assoc = r.associated_diseases(CFTR)  # no size named: the instance's
    assert [b["variables"] for b in _sent(stub, ASSOCIATIONS_QUERY)] == [{"id": CFTR, "size": 5}]
    assert len(assoc) == 5 and assoc[0].query["variables"]["size"] == 5
    assert [a.to_json() for a in assoc] == [a.to_json() for a in OpenTargetsRetriever(StubHttp()).associated_diseases(CFTR, size=5)]
    assert OpenTargetsRetriever(stub).params()["disease_size"] == 25 and r.params()["disease_size"] == 5  # an explicit size changes no knob


def test_association_rows_need_a_disease_id_and_duplicates_are_logged(caplog: pytest.LogCaptureFixture):
    row = {"disease": {"id": "MONDO_0000001", "name": "x"}, "score": 0.5, "datatypeScores": [], "datasourceScores": []}
    body = {"data": {"target": {"id": CFTR, "associatedDiseases": {"count": 2, "rows": [row, dict(row, score=0.4)]}}}}
    with caplog.at_level(logging.INFO, logger="engine.medicine.opentargets"):
        (rec,) = _canned(body).associated_diseases(CFTR, size=2)
    assert rec.payload["score"] == 0.5  # the first (best-scored) row is the record
    assert [m for m in caplog.messages if "repeating a disease id" in m] == \
        ["opentargets: 1 association rows repeating a disease id on the page dropped (1 kept of 2 delivered)"]
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="engine.medicine.opentargets"):
        assert len(OpenTargetsRetriever(StubHttp()).associated_diseases(CFTR, size=5)) == 5
    assert caplog.messages == []  # the recorded page has no repeat, so nothing is said
    bad = {"data": {"target": {"id": CFTR, "associatedDiseases": {"count": 1, "rows": [{"disease": None, "score": 0.5}]}}}}
    with pytest.raises(OpenTargetsError, match="disease"):
        _canned(bad).associated_diseases(CFTR, size=1)


# ---------------------------------------------------------------- resolve symbol

def test_resolve_symbol_exact_and_case_insensitive():
    stub = StubHttp()
    r = OpenTargetsRetriever(stub)
    assert r.resolve_symbol("CFTR") == CFTR
    assert r.resolve_symbol("cftr") == CFTR  # the server matches case-insensitively; approvedSymbol still equals
    assert r.resolve_symbol("TP53") == TP53
    assert r.resolve_symbol("NOTAREALGENEXYZ") is None
    assert r.resolve_symbol("NOTAREALGENEXYZ", allow_synonym=True) is None
    assert [c["body"]["variables"] for c in stub.calls if c["body"]["query"] == MAP_QUERY][0] == {"terms": ["CFTR"]}


def test_resolve_symbol_synonym_only_on_request():
    r = OpenTargetsRetriever(StubHttp())
    assert r.resolve_symbol("P53") is None  # mapIds says score 1, but approvedSymbol is TP53
    assert r.resolve_symbol("P53", allow_synonym=True) == TP53
    with pytest.raises(ValueError, match="non-empty"):
        r.resolve_symbol("   ")


def test_resolve_symbol_guards_the_response_shape():
    null_hits = {"data": {"mapIds": {"total": 0, "mappings": [{"term": "CFTR", "hits": None}]}}}
    assert _canned(null_hits).resolve_symbol("CFTR") is None  # hits is nullable in the schema
    wrong_term = {"data": {"mapIds": {"total": 1, "mappings": [{"term": "TP53", "hits": []}]}}}
    with pytest.raises(OpenTargetsError, match="echo"):
        _canned(wrong_term).resolve_symbol("CFTR")
    with pytest.raises(OpenTargetsError, match="no data.mapIds"):
        _canned({"data": {"target": None}}).resolve_symbol("CFTR")
    odd_id = {"data": {"mapIds": {"total": 1, "mappings": [{"term": "CFTR", "hits": [
        {"id": "CFTR", "entity": "target", "name": "CFTR", "score": 1, "object": {"id": "CFTR", "approvedSymbol": "CFTR"}}]}]}}}
    with pytest.raises(OpenTargetsError, match="not an Ensembl gene id"):
        _canned(odd_id).resolve_symbol("CFTR")


# ---------------------------------------------------------------- determinism

def test_records_are_deterministic_and_round_trip():
    a = OpenTargetsRetriever(StubHttp())
    b = OpenTargetsRetriever(StubHttp())
    b.associated_diseases(TP53, size=5)  # a different call history changes nothing
    for make in (lambda r: [r.target(CFTR)], lambda r: r.known_drugs(CFTR), lambda r: r.associated_diseases(CFTR, size=5)):
        for x, y in zip(make(a), make(b)):
            assert x.to_json() == y.to_json()
            assert EvidenceRecord.from_json(x.to_json()) == x
    assert a.version() == b.version() == VERSION


def _seed_cache(root: Path) -> HttpCache:
    """The fixtures *are* engine cache entries: file them under the key Http would compute."""
    cache = HttpCache(root)
    for fx in FIXTURES.glob("*.json"):
        d = json.loads(fx.read_text())
        req = d["request"]
        key = HttpCache.key(req["method"], req["url"], req["params"], req["body"])
        cache.put(key, Response(d["status"], d["text"], d["retrieved_at"], False, key, d["headers"]), req)
    return cache


def test_offline_replay_through_the_real_http_cache(tmp_path: Path):
    """Warm cache => byte-identical records and zero network, through the real Http."""
    cache = _seed_cache(tmp_path / "cache")
    passes = []
    for _ in range(2):
        http = Http(cache, offline=True)
        r = OpenTargetsRetriever(http)
        recs = [r.target(CFTR), *r.known_drugs(CFTR), *r.associated_diseases(CFTR, size=5), *r.known_drugs(TP53)]
        assert r.resolve_symbol("CFTR") == CFTR and r.target(ABSENT) is None
        assert http.live_requests == 0
        passes.append([x.to_json() for x in recs])
    assert passes[0] == passes[1] and len(passes[0]) == 1 + 13 + 5 + 9
    assert cache.stats()["misses"] == 0
    assert http.limiter.per_second[HOST] == PER_SECOND
    with pytest.raises(RuntimeError, match="offline"):  # anything unrecorded is a hard stop, never a fabricated absence
        OpenTargetsRetriever(Http(cache, offline=True)).associated_diseases(CFTR, size=6)


class _Served:
    """A stand-in for ``urllib.request.urlopen`` that answers with the recorded bytes for
    the exact body sent (or a scripted override), and remembers what went live."""

    def __init__(self, override: dict[str, str] | None = None):
        self.recorded = _recorded()
        self.override = override or {}  # canonical body → text
        self.live: list[dict] = []

    def __call__(self, req, timeout=None):
        body = json.loads(req.data.decode())
        self.live.append(body)
        text = self.override.get(_canon(body))
        if text is None:
            text = self.recorded[_canon(body)]["text"]
        return _Reply(text)


class _Reply:
    status = 200
    headers: dict[str, str] = {"content-type": "application/json"}

    def __init__(self, text: str):
        self._text = text

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._text.encode()


POISON = {"data": {"target": None}, "errors": [{"message": "Internal server error"}]}


def _poison(cache: HttpCache, body: dict, text: str = json.dumps(POISON)) -> str:
    """File a failure body under the key ``Http`` computes for ``body`` — what ``Http``
    itself would have written had the server answered so with HTTP 200."""
    key = HttpCache.key("POST", API_URL, None, body)
    cache.put(key, Response(200, text, "2026-09-12T00:00:00+00:00", False, key, {}),
              {"method": "POST", "url": API_URL, "params": None, "body": body})
    return key


def _online(cache: HttpCache) -> Http:
    return Http(cache, limiter=RateLimiter({HOST: 0}))  # no pacing in a test; _pin_rate leaves a set rate alone


def test_a_cached_failure_body_is_refetched_live_once_and_healed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                 caplog: pytest.LogCaptureFixture):
    """The API reports an execution-time failure as HTTP 200 + errors[], which Http caches
    as definitive. Such an entry must not replay forever: one live re-fetch, and the
    validated answer takes its place in the cache."""
    cache = _seed_cache(tmp_path / "cache")
    body = {"query": TARGET_QUERY, "variables": {"id": CFTR}}
    key = _poison(cache, body)
    served = _Served()
    monkeypatch.setattr("engine.retrieve.http.urllib.request.urlopen", served)
    http = _online(cache)
    with caplog.at_level(logging.WARNING, logger="engine.medicine.opentargets"):
        rec = OpenTargetsRetriever(http).target(CFTR)
    assert rec is not None and rec.payload["approvedSymbol"] == "CFTR" and rec.source_version == VERSION
    assert [b["query"] for b in served.live] == [META_QUERY, TARGET_QUERY]  # the live version check, then the re-fetch
    assert http.live_requests == 2 and served.live[1] == body
    assert caplog.messages == ["opentargets: a cached response failed validation; fetching it again live"]
    healed = cache.get(key)
    assert healed is not None and json.loads(healed.text)["data"]["target"]["approvedSymbol"] == "CFTR"  # the poison is gone
    assert healed.retrieved_at == rec.retrieved_at
    # the next run replays the healed entry: same bytes, no gene request, and it works offline
    http2 = _online(cache)
    again = OpenTargetsRetriever(http2).target(CFTR)
    assert again.to_json() == rec.to_json() and http2.live_requests == 1 and [b["query"] for b in served.live[2:]] == [META_QUERY]
    assert OpenTargetsRetriever(Http(cache, offline=True)).target(CFTR).to_json() == rec.to_json()


def test_a_failure_that_persists_live_raises_and_is_not_healed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cache = _seed_cache(tmp_path / "cache")
    body = {"query": DRUGS_QUERY, "variables": {"id": CFTR}}
    key = _poison(cache, body)
    served = _Served(override={_canon(body): json.dumps(POISON)})
    monkeypatch.setattr("engine.retrieve.http.urllib.request.urlopen", served)
    http = _online(cache)
    with pytest.raises(OpenTargetsError, match="Internal server error"):
        OpenTargetsRetriever(http).known_drugs(CFTR)
    assert [b["query"] for b in served.live] == [META_QUERY, DRUGS_QUERY]  # exactly one re-fetch, never a loop
    assert json.loads(cache.get(key).text) == POISON  # a failing live answer is not written back
    # a live (uncached) failure is raised at once: no second request for it
    assert http.live_requests == 2


def test_a_cached_failure_offline_raises_its_own_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Offline there is nothing to re-fetch: the cached body's error is the answer, not a
    misleading 'no cached response'."""
    cache = _seed_cache(tmp_path / "cache")
    _poison(cache, {"query": TARGET_QUERY, "variables": {"id": CFTR}})
    served = _Served()
    monkeypatch.setattr("engine.retrieve.http.urllib.request.urlopen", served)
    http = Http(cache, offline=True)
    with pytest.raises(OpenTargetsError, match="Internal server error"):
        OpenTargetsRetriever(http).target(CFTR)
    assert served.live == [] and http.live_requests == 0


def test_a_cached_body_the_module_rejects_for_shape_is_also_refetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Not only errors[]: a cached page whose rows disagree with its count (a truncated
    answer) gets the same single live re-fetch, so validation and re-fetch share one path."""
    cache = _seed_cache(tmp_path / "cache")
    body = {"query": ASSOCIATIONS_QUERY, "variables": {"id": CFTR, "size": 5}}
    truncated = json.loads(_fixture(f"associations_{CFTR}_5")["text"])
    truncated["data"]["target"]["associatedDiseases"]["rows"] = truncated["data"]["target"]["associatedDiseases"]["rows"][:2]
    key = _poison(cache, body, json.dumps(truncated))
    served = _Served()
    monkeypatch.setattr("engine.retrieve.http.urllib.request.urlopen", served)
    http = _online(cache)
    assoc = OpenTargetsRetriever(http).associated_diseases(CFTR, size=5)
    assert len(assoc) == 5 and [b["query"] for b in served.live] == [META_QUERY, ASSOCIATIONS_QUERY]
    assert json.loads(cache.get(key).text) == json.loads(_fixture(f"associations_{CFTR}_5")["text"])


def test_params_are_manifest_ready():
    stub = StubHttp()
    r = OpenTargetsRetriever(stub)
    params = r.params()
    assert json.loads(json.dumps(params, sort_keys=True)) == params  # JSON round-trips, nothing exotic
    assert set(params) == {"api_url", "approval_stage", "datatypes", "disease_size", "enable_indirect",
                           "max_page_size", "page_index", "per_second", "stage_order", "trial_report_type"}
    assert params["api_url"] == API_URL and params["per_second"] == PER_SECOND
    assert params["disease_size"] == 25 and params["max_page_size"] == 3000 and params["page_index"] == 0
    assert params["enable_indirect"] is False and "enableIndirect: false" in ASSOCIATIONS_QUERY
    assert "page: { index: 0, size: $size }" in ASSOCIATIONS_QUERY
    assert params["stage_order"]["APPROVAL"] > params["stage_order"]["PHASE_3"] > params["stage_order"]["PRECLINICAL"]
    assert list(params["datatypes"]) == list(ASSOCIATION_COLUMNS[4:-2])
    assert stub.calls == []  # params never asks the server
    stub.limiter.per_second[HOST] = 1.0  # an orchestrator's rate is the one reported
    assert OpenTargetsRetriever(stub).params()["per_second"] == 1.0


# ---------------------------------------------------------------- absence is narrow; everything else raises

def test_pagination_error_with_null_target_is_a_failure_not_absence():
    body = {"data": {"target": None}, "errors": [{"message": "Argument 'page' has invalid value: ... size must be between 0 and 3000"}]}
    with pytest.raises(OpenTargetsError, match="between 0 and 3000"):
        _canned(body).associated_diseases(CFTR, size=5)
    with pytest.raises(OpenTargetsError, match="errors"):
        _canned(body).target(CFTR)


def test_null_data_with_errors_raises():
    body = {"data": None, "errors": [{"message": "Query is too expensive."}]}
    with pytest.raises(OpenTargetsError, match="too expensive"):
        _canned(body).associated_diseases(CFTR, size=3000)


def test_data_without_the_field_raises():
    with pytest.raises(OpenTargetsError, match="no data.target"):
        _canned({"data": {"search": {"total": 0}}}).target(CFTR)
    with pytest.raises(OpenTargetsError, match="no data object"):
        _canned({"errors": []}).target(CFTR)


def test_echoed_id_mismatch_raises():
    with pytest.raises(OpenTargetsError, match="echo"):
        _canned({"data": {"target": {"id": TP53, "approvedSymbol": "TP53"}}}).target(CFTR)


def test_html_and_non_200_bodies_raise_and_never_read_as_absence():
    html = "<!DOCTYPE html><html><body>Bad Request</body></html>"
    with pytest.raises(OpenTargetsError, match="non-JSON"):
        _canned(html).target(CFTR)
    with pytest.raises(OpenTargetsError, match="HTTP 415"):
        _canned(html, status=415).target(CFTR)
    with pytest.raises(OpenTargetsError, match="unexpected JSON"):
        _canned("[]").target(CFTR)


def test_http_error_propagates():
    err = HttpError(400, API_URL, '{"errors":[{"message":"Cannot query field \'knownDrugs\' on type \'Target\'."}]}')
    with pytest.raises(HttpError, match="knownDrugs"):
        OpenTargetsRetriever(CannedHttp(raise_=err)).known_drugs(CFTR)


def test_malformed_blocks_raise():
    with pytest.raises(OpenTargetsError, match="drugAndClinicalCandidates"):
        _canned({"data": {"target": {"id": CFTR, "drugAndClinicalCandidates": None}}}).known_drugs(CFTR)
    with pytest.raises(OpenTargetsError, match="rows"):
        _canned({"data": {"target": {"id": CFTR, "associatedDiseases": {"count": 1, "rows": None}}}}).associated_diseases(CFTR, size=1)


# ---------------------------------------------------------------- etiquette

def test_rate_is_pinned_unless_the_orchestrator_set_one():
    stub = StubHttp()
    OpenTargetsRetriever(stub)
    assert stub.limiter.per_second[HOST] == PER_SECOND
    stub.limiter.per_second[HOST] = 1.0
    OpenTargetsRetriever(stub)
    assert stub.limiter.per_second[HOST] == 1.0


def test_requests_are_post_json_with_variables_only():
    stub = StubHttp()
    r = OpenTargetsRetriever(stub)
    r.target(CFTR)
    r.known_drugs(CFTR)
    r.associated_diseases(CFTR, size=5)
    r.resolve_symbol("CFTR")
    for c in stub.calls:
        assert c["method"] == "POST" and c["url"] == API_URL
        assert set(c["body"]) == {"query", "variables"}
        assert CFTR not in c["body"]["query"] and "CFTR" not in c["body"]["query"]  # ids travel as variables, never in the text


# ---------------------------------------------------------------- live (public genes only)

@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ENGINE_LIVE_TESTS"), reason="set ENGINE_LIVE_TESTS=1 to hit the Open Targets API")
def test_live_public_genes(tmp_path: Path):
    from engine.retrieve.http import Http, HttpCache

    http = Http(HttpCache(tmp_path / "cache"))
    r = OpenTargetsRetriever(http)
    assert r.version().startswith("Open Targets Platform ") and "(API " in r.version()
    assert r.resolve_symbol("CFTR") == CFTR and r.resolve_symbol("NOTAREALGENEXYZ") is None
    rec = r.target(CFTR)
    assert rec is not None and extract_target(rec)["symbol"] == "CFTR" and extract_target(rec)["chromosome"] == "7"
    assert r.target(ABSENT) is None
    drugs = {extract_drug(d)["name"]: extract_drug(d) for d in r.known_drugs(CFTR)}
    assert drugs["IVACAFTOR"]["chembl_id"] == "CHEMBL2010601" and drugs["IVACAFTOR"]["max_clinical_stage"] == "APPROVAL"
    assert "positive modulator" in drugs["IVACAFTOR"]["mechanism_of_action"].lower()
    assert drugs["CROFELEMER"]["n_mechanisms_other_targets"] == "1"  # the ANO1 row, filtered
    tp53 = {extract_drug(d)["name"]: extract_drug(d) for d in r.known_drugs(TP53)}
    assert tp53["IDASANUTLIN"]["moa_targets"] == "MDM2,TP53"  # a complex, visible in the flat view
    assert r.params()["per_second"] == PER_SECOND
    assoc = [extract_association(a) for a in r.associated_diseases(CFTR, size=5)]
    assert "MONDO_0009061" in [a["disease_id"] for a in assoc]
    n = http.live_requests
    assert r.target(CFTR) == rec and http.live_requests == n  # warm cache: same bytes, no request
