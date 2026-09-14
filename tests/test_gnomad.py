"""gnomAD retriever against recorded real responses (public textbook variants only).

The fixtures in ``tests/fixtures/gnomad/`` are the engine's own HTTP cache entries from
one live batch for MTHFR rs1801133, CFTR p.Phe508del and TP53 p.Arg175His, and one
``meta`` query — nothing else was ever sent. A stub Http re-assembles batches from them
alias by alias, exactly the way the server shapes a response (``null`` plus one
"Variant not found" for an unknown id), so absence and any batch composition can be
exercised with synthetic keys that never leave the process.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from engine.retrieve import Retriever
from engine.retrieve.gnomad import (FRAGMENT, MAX_QUERY_CHARS, META_QUERY, NOT_FOUND_IN_SUBSET, GnomadError,
                                    GnomadRetriever, build_query, unaskable, variant_id)
from engine.retrieve.http import HttpError, Response
from engine.retrieve.store import EvidenceRecord

FIXTURES = Path(__file__).parent / "fixtures" / "gnomad"
BATCH = json.loads((FIXTURES / "batch_gnomad_r4.json").read_text())
META = json.loads((FIXTURES / "meta.json").read_text())

MTHFR = ("1", 11796321, "G", "A")
CFTR = ("7", 117559590, "ATCT", "A")
TP53 = ("17", 7675088, "C", "T")
ABSENT = ("1", 11796321, "G", "T")  # synthetic: another alt at the MTHFR site, stub only, never sent
API = "https://gnomad.broadinstitute.org/api"
VERSION = "gnomad_r4 clinvar_release_date=2026-06-06"

_ALIAS = re.compile(r'(v\d+): variant\(variantId: "([^"]+)", dataset: (\w+)\)')


def _found() -> dict[str, dict]:
    """variant_id → recorded variant object, from the fixture's batch response."""
    data = json.loads(BATCH["text"])["data"]
    return {v["variant_id"]: v for v in data.values() if v is not None}


class StubHttp:
    """Serves the recorded fixtures. A batch request is answered alias by alias — a
    recorded object for a known id, ``null`` plus one "Variant not found" error for
    anything else — which is the shape the live server produced."""

    def __init__(self, dataset: str = "gnomad_r4"):
        self.found = _found()
        self.dataset = dataset
        self.calls: list[dict] = []

    def request(self, method: str, url: str, *, params=None, json_body=None, **kw) -> Response:
        self.calls.append({"method": method, "url": url, "body": json_body, **kw})
        assert method == "POST" and url == API
        query = json_body["query"]
        if query == META_QUERY:
            return Response(META["status"], META["text"], META["retrieved_at"], False, "stub-meta", {})
        if json_body == BATCH["request"]["body"]:
            # byte-for-byte the request recorded live: answer with the recorded bytes
            return Response(BATCH["status"], BATCH["text"], BATCH["retrieved_at"], False, "stub-recorded", {})
        data, errors = {}, []
        for alias, vid, dataset in _ALIAS.findall(query):
            assert dataset == self.dataset
            data[alias] = self.found.get(vid)
            if data[alias] is None:
                errors.append({"message": "Variant not found"})
        body = {"errors": errors, "data": data} if errors else {"data": data}
        return Response(200, json.dumps(body), BATCH["retrieved_at"], False, "stub-assembled", {})

    def get(self, url: str, **kw) -> Response:
        return self.request("GET", url, **kw)

    def post(self, url: str, json_body, **kw) -> Response:
        return self.request("POST", url, json_body=json_body, **kw)


class CannedHttp(StubHttp):
    """Answers every variant batch with one fixed body (or raises), for error shapes."""

    def __init__(self, text: str | None = None, status: int = 200, raise_: Exception | None = None):
        super().__init__()
        self.text, self.status, self.raise_ = text, status, raise_

    def request(self, method, url, *, params=None, json_body=None, **kw) -> Response:
        if json_body["query"] == META_QUERY:
            return super().request(method, url, params=params, json_body=json_body, **kw)
        self.calls.append({"method": method, "url": url, "body": json_body, **kw})
        if self.raise_ is not None:
            raise self.raise_
        return Response(self.status, self.text or "", BATCH["retrieved_at"], False, "canned", {})


class PoisonedHttp(StubHttp):
    """A cache that holds a 200 body gnomAD produced for a transient resolver failure:
    every cache-consulting call replays it; only ``cache_ok=False`` reaches the server."""

    def __init__(self, poison_meta: bool = False):
        super().__init__()
        self.poison_meta = poison_meta

    def request(self, method, url, *, params=None, json_body=None, **kw) -> Response:
        is_meta = json_body["query"] == META_QUERY
        if kw.get("cache_ok", True) and (self.poison_meta or not is_meta):
            self.calls.append({"method": method, "url": url, "body": json_body, **kw})
            body = {"errors": [{"message": "Something went wrong"}], "data": {"meta": None} if is_meta else {"v0": None}}
            return Response(200, json.dumps(body), "2026-09-01T00:00:00+00:00", True, "poisoned", {})
        return super().request(method, url, params=params, json_body=json_body, **kw)


def _batch_queries(stub: StubHttp) -> list[str]:
    return [c["body"]["query"] for c in stub.calls if c["body"]["query"] != META_QUERY]


# ---------------------------------------------------------------- retrieve

def test_implements_protocol():
    r = GnomadRetriever(StubHttp())
    assert isinstance(r, Retriever)
    assert r.source == "gnomad" and len(r.columns) == 14 and r.columns[0] == "gnomad_af"


def test_retrieve_from_recorded_batch():
    stub = StubHttp()
    r = GnomadRetriever(stub)
    found = r.retrieve([MTHFR, CFTR, TP53])

    # The query our code builds is byte-identical to the one recorded live.
    assert _batch_queries(stub) == [BATCH["request"]["body"]["query"]]
    assert len(_batch_queries(stub)) == 1  # three keys, one POST

    for k in (MTHFR, CFTR, TP53):
        (rec,) = found[k]
        vid = variant_id(k)
        assert rec.record_id == f"gnomad:{vid}"
        assert rec.source == "gnomad"
        assert rec.source_version == VERSION
        assert rec.query == {"api": API, "dataset": "gnomad_r4", "variantId": vid, "fragment": FRAGMENT}
        assert rec.url == f"https://gnomad.broadinstitute.org/variant/{vid}?dataset=gnomad_r4"
        assert rec.retrieved_at == BATCH["retrieved_at"]  # from the Response, not the clock
        p = rec.payload
        assert p["variant_id"] == vid and p["reference_genome"] == "GRCh38"
        assert all(b in p for b in ("exome", "genome", "joint", "rsids", "flags", "caid"))
    assert found[MTHFR][0].payload["rsids"] == ["rs1801133"]
    assert found[CFTR][0].payload["rsids"] == ["rs113993960"]
    assert found[TP53][0].payload["caid"] == "CA251"
    assert r.version() == VERSION


def test_absent_is_a_result_not_an_error():
    stub = StubHttp()
    r = GnomadRetriever(stub)
    found = r.retrieve([MTHFR, ABSENT])
    assert len(_batch_queries(stub)) == 1  # one POST for both, absence attributed by alias
    absent = found[ABSENT]
    assert len(absent) == 1 and absent[0].payload["absent"] is True and absent[0].record_id == "gnomad:1-11796321-G-T"
    assert absent[0].url.startswith("https://gnomad.broadinstitute.org/variant/1-11796321-G-T")  # a citable observation of absence
    assert found[MTHFR][0].record_id == "gnomad:1-11796321-G-A"
    assert r.extract(absent) == {**{c: "" for c in r.columns}, "gnomad_dataset": "gnomad_r4"}


def test_payload_is_the_raw_variant_object():
    r = GnomadRetriever(StubHttp())
    (rec,) = r.retrieve([TP53])[TP53]
    assert rec.payload == _found()["17-7675088-C-T"]  # verbatim: no derived keys
    assert rec.payload["genome"]["faf95"] == {"popmax": None, "popmax_population": None}  # nullable leaves survive


def test_records_are_deterministic():
    a = GnomadRetriever(StubHttp()).retrieve([MTHFR, CFTR, TP53, ABSENT])
    b = GnomadRetriever(StubHttp()).retrieve([TP53, ABSENT, MTHFR, CFTR])  # other order, other batch
    for k in (MTHFR, CFTR, TP53):
        assert a[k][0].to_json() == b[k][0].to_json()
        assert EvidenceRecord.from_json(a[k][0].to_json()) == a[k][0]


def test_batches_of_25_and_query_size():
    stub = StubHttp()
    r = GnomadRetriever(stub)
    synthetic = [("2", 1000 + i, "A", "C") for i in range(58)]  # never sent anywhere: stub only
    found = r.retrieve(synthetic + [MTHFR])
    queries = _batch_queries(stub)
    assert [len(_ALIAS.findall(q)) for q in queries] == [25, 25, 9]
    assert all(len(q) < MAX_QUERY_CHARS for q in queries)
    assert all(len(q) < 9000 for q in queries)
    assert sum(1 for k in synthetic if found[k] and found[k][0].payload.get("absent")) == 58
    assert found[MTHFR][0].record_id == "gnomad:1-11796321-G-A"
    assert set(re.findall(r"v(\d+):", queries[0])) == {str(i) for i in range(25)}


def test_long_alleles_split_batches_by_text_size():
    stub = StubHttp()
    r = GnomadRetriever(stub)
    long_ref = "A" * 700
    keys = [("3", 5000 + i, long_ref, "A") for i in range(20)]
    r.retrieve(keys)
    queries = _batch_queries(stub)
    assert len(queries) > 1
    assert all(len(q) <= MAX_QUERY_CHARS for q in queries)
    assert sum(len(_ALIAS.findall(q)) for q in queries) == 20


def test_one_variant_too_long_to_query_raises():
    r = GnomadRetriever(StubHttp())
    with pytest.raises(GnomadError, match="chars"):
        r.retrieve([("3", 5000, "A" * 9000, "A")])


def test_duplicate_keys_are_queried_once():
    stub = StubHttp()
    r = GnomadRetriever(stub)
    found = r.retrieve([MTHFR, MTHFR])
    assert len(_ALIAS.findall(_batch_queries(stub)[0])) == 1
    assert found[MTHFR][0].record_id == "gnomad:1-11796321-G-A"


def test_empty_keys_makes_no_request():
    stub = StubHttp()
    assert GnomadRetriever(stub).retrieve([]) == {}
    assert stub.calls == []


def test_query_text_and_constructor_guards():
    q = build_query(["17-7675088-C-T"], "gnomad_r4_non_ukb")
    assert 'v0: variant(variantId: "17-7675088-C-T", dataset: gnomad_r4_non_ukb) { ...F }' in q
    with pytest.raises(ValueError):
        GnomadRetriever(StubHttp(), dataset="gnomad_r4) { } #")
    with pytest.raises(ValueError):
        GnomadRetriever(StubHttp(), batch_size=26)


def test_non_default_dataset_flows_into_url_column_and_version():
    stub = StubHttp(dataset="gnomad_r4_non_ukb")
    r = GnomadRetriever(stub, dataset="gnomad_r4_non_ukb")
    (rec,) = r.retrieve([TP53])[TP53]
    assert rec.url.endswith("?dataset=gnomad_r4_non_ukb")
    assert rec.query["dataset"] == "gnomad_r4_non_ukb"
    assert r.extract([rec])["gnomad_dataset"] == "gnomad_r4_non_ukb"
    assert r.version() == "gnomad_r4_non_ukb clinvar_release_date=2026-06-06"


def test_version_is_a_function_of_the_request_not_of_what_was_found():
    r = GnomadRetriever(StubHttp())
    before = r.version()
    r.retrieve([ABSENT])
    after_absent = r.version()
    found = r.retrieve([MTHFR])
    assert before == after_absent == r.version() == found[MTHFR][0].source_version == VERSION


# ---------------------------------------------------------------- keys the API cannot be asked about

def test_mitochondrial_keys_are_refused_before_any_request():
    stub = StubHttp()
    r = GnomadRetriever(stub)
    mito = ("MT", 3243, "A", "G")
    assert unaskable(mito) == "mitochondrial"
    with pytest.raises(GnomadError, match="mitochondrial"):
        variant_id(mito)
    with pytest.raises(GnomadError, match=r"1 of 2 keys .*1 mitochondrial.*filter them upstream") as e:
        r.retrieve([MTHFR, mito])
    assert stub.calls == []  # nothing was sent, so nothing can be reported absent
    assert "3243" not in str(e.value) and "M-" not in str(e.value)  # counts only, never an allele


def test_non_acgt_alleles_are_refused_before_any_request():
    stub = StubHttp()
    r = GnomadRetriever(stub)
    n_ref = ("1", 11796321, "N", "A")
    symbolic = ("2", 5000, "A", "<DEL>")
    assert unaskable(n_ref) == unaskable(symbolic) == "non-ACGT allele"
    assert unaskable(MTHFR) is None and unaskable(CFTR) is None
    with pytest.raises(GnomadError, match=r"2 of 3 keys .*2 non-ACGT allele") as e:
        r.retrieve([MTHFR, n_ref, symbolic])
    assert stub.calls == []
    assert "<DEL>" not in str(e.value) and "11796321" not in str(e.value)
    with pytest.raises(GnomadError, match=r"1 mitochondrial, 1 non-ACGT"):
        r.retrieve([n_ref, ("MT", 3243, "A", "G")])


def test_lowercase_alleles_are_uppercased_like_the_server_does():
    stub = StubHttp()
    r = GnomadRetriever(stub)
    lower = ("1", 11796321, "g", "a")
    assert variant_id(lower) == "1-11796321-G-A"
    found = r.retrieve([lower])
    (rec,) = found[lower]  # keyed by the key as given
    assert rec.record_id == "gnomad:1-11796321-G-A" and rec.query["variantId"] == "1-11796321-G-A"
    assert 'variantId: "1-11796321-G-A"' in _batch_queries(stub)[0]


# ---------------------------------------------------------------- absence is narrow; everything else raises

def _canned(body: dict | str, status: int = 200, dataset: str = "gnomad_r4") -> GnomadRetriever:
    text = body if isinstance(body, str) else json.dumps(body)
    return GnomadRetriever(CannedHttp(text=text, status=status), dataset=dataset)


def test_not_found_in_selected_subset_is_absence():
    r = _canned({"errors": [{"message": NOT_FOUND_IN_SUBSET}], "data": {"v0": None}}, dataset="gnomad_r4_non_ukb")
    found = r.retrieve([MTHFR])
    assert len(found[MTHFR]) == 1 and found[MTHFR][0].payload["absent"] is True and found[MTHFR][0].query["dataset"] == "gnomad_r4_non_ukb"
    assert r.extract(found[MTHFR]) == {**{c: "" for c in r.columns}, "gnomad_dataset": "gnomad_r4_non_ukb"}
    # still one message per null alias, and the two absence messages may mix
    v = _found()["1-11796321-G-A"]
    r = _canned({"errors": [{"message": "Variant not found"}, {"message": NOT_FOUND_IN_SUBSET}],
                 "data": {"v0": None, "v1": v, "v2": None}}, dataset="gnomad_r4_non_ukb")
    found = r.retrieve([ABSENT, MTHFR, TP53])
    assert found[ABSENT][0].payload["absent"] and found[TP53][0].payload["absent"] and found[MTHFR][0].record_id == "gnomad:1-11796321-G-A"


def test_invalid_variant_id_raises_not_absent():
    r = _canned({"errors": [{"message": "Invalid variant ID"}], "data": {"v0": None}})
    with pytest.raises(GnomadError, match="Invalid variant ID"):
        r.retrieve([MTHFR])


def test_multiple_variants_found_raises():
    msg = "Multiple variants found, query using variant ID to select one."
    r = _canned({"errors": [{"message": msg}], "data": {"v0": None}})
    with pytest.raises(GnomadError, match="Multiple variants"):
        r.retrieve([TP53])


def test_null_without_matching_error_raises():
    r = _canned({"data": {"v0": None}})
    with pytest.raises(GnomadError, match="null aliases"):
        r.retrieve([MTHFR])


def test_error_count_mismatch_raises():
    v = _found()["1-11796321-G-A"]
    r = _canned({"errors": [{"message": "Variant not found"}, {"message": "Variant not found"}],
                 "data": {"v0": v, "v1": None}})
    with pytest.raises(GnomadError, match="null aliases"):
        r.retrieve([MTHFR, ABSENT])


def test_mixed_error_messages_raise_even_with_a_matching_not_found():
    r = _canned({"errors": [{"message": "Variant not found"}, {"message": "Query is too expensive (26). Maximum allowed cost is 25."}],
                 "data": {"v0": None, "v1": None}})
    with pytest.raises(GnomadError, match="too expensive"):
        r.retrieve([MTHFR, ABSENT])


def test_stale_cache_foreign_aliases_raise():
    r = _canned({"data": {"g0": {"gene_id": "ENSG00000073464", "symbol": "CLCN4"}, "g1": {"symbol": "ARHGEF9"}}})
    with pytest.raises(GnomadError, match="aliases"):
        r.retrieve([MTHFR])


def test_echoed_variant_id_mismatch_raises():
    v = dict(_found()["1-11796321-G-A"], variant_id="1-11796321-G-C")
    r = _canned({"data": {"v0": v}})
    with pytest.raises(GnomadError, match="echo"):
        r.retrieve([MTHFR])


def test_other_reference_genome_raises():
    v = dict(_found()["1-11796321-G-A"], reference_genome="GRCh37")
    with pytest.raises(GnomadError, match="GRCh37"):
        _canned({"data": {"v0": v}}).retrieve([MTHFR])
    v = dict(_found()["1-11796321-G-A"], reference_genome=None)
    with pytest.raises(GnomadError, match="reference_genome"):
        _canned({"data": {"v0": v}}).retrieve([MTHFR])


def test_html_body_raises_and_is_never_parsed_as_absence():
    html = '<!doctype html><meta charset="utf-8"><title>429</title>429 Too Many Requests'
    with pytest.raises(GnomadError, match="non-JSON"):
        _canned(html).retrieve([MTHFR])
    with pytest.raises(GnomadError, match="HTTP 429"):
        _canned(html, status=429).retrieve([MTHFR])


def test_data_null_or_missing_raises():
    with pytest.raises(GnomadError, match="no data"):
        _canned({"errors": [{"message": "Syntax Error"}], "data": None}).retrieve([MTHFR])
    with pytest.raises(GnomadError):
        _canned([]).retrieve([MTHFR])


def test_http_error_propagates():
    r = GnomadRetriever(CannedHttp(raise_=HttpError(400, API, '{"errors":[{"message":"Query is too expensive (26)."}]}')))
    with pytest.raises(HttpError):
        r.retrieve([MTHFR])


def test_meta_without_date_raises_on_version():
    class NoMeta(StubHttp):
        def request(self, method, url, *, params=None, json_body=None, **kw):
            if json_body["query"] == META_QUERY:
                return Response(200, '{"data":{"meta":{"clinvar_release_date":null}}}', "t", False, "k", {})
            return super().request(method, url, params=params, json_body=json_body, **kw)
    with pytest.raises(GnomadError, match="clinvar_release_date"):
        GnomadRetriever(NoMeta()).version()


# ---------------------------------------------------------------- a poisoned cache entry is refetched live once

def test_poisoned_cached_batch_is_refetched_live_once():
    stub = PoisonedHttp()
    r = GnomadRetriever(stub)
    found = r.retrieve([MTHFR])
    (rec,) = found[MTHFR]
    assert rec.record_id == "gnomad:1-11796321-G-A"
    assert rec.retrieved_at == BATCH["retrieved_at"]  # from the live answer, not the poisoned entry
    batch_calls = [c for c in stub.calls if c["body"]["query"] != META_QUERY]
    assert [c.get("cache_ok", True) for c in batch_calls] == [True, False]  # cached replay, then one live
    assert batch_calls[0]["body"] == batch_calls[1]["body"]  # the very same request


def test_poisoned_cached_meta_is_refetched_live_once():
    stub = PoisonedHttp(poison_meta=True)
    assert GnomadRetriever(stub).version() == VERSION
    meta_calls = [c for c in stub.calls if c["body"]["query"] == META_QUERY]
    assert [c.get("cache_ok", True) for c in meta_calls] == [True, False]


def test_uncached_failure_is_not_retried():
    stub = CannedHttp(text=json.dumps({"errors": [{"message": "Something went wrong"}], "data": {"v0": None}}))
    with pytest.raises(GnomadError, match="Something went wrong"):
        GnomadRetriever(stub).retrieve([MTHFR])
    assert len([c for c in stub.calls if c["body"]["query"] != META_QUERY]) == 1


def test_poisoned_entry_that_fails_again_live_raises():
    class StillBroken(PoisonedHttp):
        def request(self, method, url, *, params=None, json_body=None, **kw):
            if not kw.get("cache_ok", True):
                return Response(200, json.dumps({"errors": [{"message": "Something went wrong"}], "data": {"v0": None}}),
                                "t", False, "live", {})
            return super().request(method, url, params=params, json_body=json_body, **kw)
    with pytest.raises(GnomadError, match="Something went wrong"):
        GnomadRetriever(StillBroken()).retrieve([MTHFR])


# ---------------------------------------------------------------- extract

def _records(*keys):
    r = GnomadRetriever(StubHttp())
    found = r.retrieve(list(keys))
    return r, found


def test_extract_absent_is_all_blank():
    r, found = _records(ABSENT)
    assert r.extract(found[ABSENT]) == {**{c: "" for c in r.columns}, "gnomad_dataset": "gnomad_r4"}
    assert r.extract([]) == {c: "" for c in r.columns}


def test_extract_mthfr_uses_joint_block():
    r, found = _records(MTHFR)
    cols = r.extract(found[MTHFR])
    assert cols == {
        "gnomad_af": "0.3182137576943525",      # 513548 / 1613846, joint has no af field
        "gnomad_ac": "513548",
        "gnomad_an": "1613846",
        "gnomad_nhom": "87723",
        "gnomad_af_exome": "0.3226922650671315",
        "gnomad_af_genome": "0.27516963863026667",
        "gnomad_nhom_exome": "80805",
        "gnomad_nhom_genome": "6918",
        "gnomad_filters": "",                    # joint's discrepant_frequencies is not a QC filter
        "gnomad_grpmax_af": "0.4777440650840224",  # amr 28657 / 59984 in the joint block
        "gnomad_grpmax_pop": "amr",
        "gnomad_faf95_popmax": "0.4731112799999998",  # joint.faf95.popmax, the browser's "Popmax Filtering AF"
        "gnomad_faf95_pop": "amr",
        "gnomad_dataset": "gnomad_r4",
    }
    assert found[MTHFR][0].payload["joint"]["filters"] == ["discrepant_frequencies"]
    assert float(cols["gnomad_af"]) == 513548 / 1613846  # repr precision, no rounding
    assert float(cols["gnomad_faf95_popmax"]) < float(cols["gnomad_grpmax_af"])  # a filtering AF is the CI lower bound


def test_extract_cftr_and_tp53():
    r, found = _records(CFTR, TP53)
    c = r.extract(found[CFTR])
    assert (c["gnomad_af"], c["gnomad_ac"], c["gnomad_an"], c["gnomad_nhom"]) == ("0.011931254341569912", "19237", "1612320", "58")
    assert (c["gnomad_af_exome"], c["gnomad_af_genome"]) == ("0.012353145028401891", "0.007884051877061352")
    assert (c["gnomad_nhom_exome"], c["gnomad_nhom_genome"]) == ("57", "1")
    assert (c["gnomad_grpmax_af"], c["gnomad_grpmax_pop"]) == ("0.01494254629134656", "nfe")  # 17610 / 1178514
    assert (c["gnomad_faf95_popmax"], c["gnomad_faf95_pop"]) == ("0.01475724", "nfe")
    t = r.extract(found[TP53])
    assert (t["gnomad_af"], t["gnomad_ac"], t["gnomad_an"], t["gnomad_nhom"]) == ("4.336884208908951e-06", "7", "1614062", "0")
    assert (t["gnomad_af_exome"], t["gnomad_af_genome"]) == ("4.104365813916263e-06", "6.570129562954982e-06")
    assert (t["gnomad_grpmax_af"], t["gnomad_grpmax_pop"]) == ("5.93204252088079e-06", "nfe")  # 7 / 1180032
    assert (t["gnomad_faf95_popmax"], t["gnomad_faf95_pop"]) == ("2.47e-06", "nfe")
    assert t["gnomad_filters"] == ""


def _with(rec: EvidenceRecord, **blocks) -> EvidenceRecord:
    payload = dict(rec.payload)
    payload.update(blocks)
    return EvidenceRecord(**{**rec.__dict__, "payload": payload})


def test_extract_without_joint_combines_exome_and_genome():
    r, found = _records(MTHFR)
    rec = _with(found[MTHFR][0], joint=None)
    cols = r.extract([rec])
    # exome 471698/1461758 hom 80805 + genome 41850/152088 hom 6918
    assert (cols["gnomad_ac"], cols["gnomad_an"], cols["gnomad_nhom"]) == ("513548", "1613846", "87723")
    assert cols["gnomad_af"] == repr(513548 / 1613846)
    # per-population sums across the two blocks: amr (21885+6772)/(44724+15260)
    assert (cols["gnomad_grpmax_af"], cols["gnomad_grpmax_pop"]) == (repr(28657 / 59984), "amr")
    # filtering AF: the higher of exome (0.4839…) and genome (0.4349…)
    assert (cols["gnomad_faf95_popmax"], cols["gnomad_faf95_pop"]) == ("0.48390636999999964", "amr")


def test_extract_single_block_only():
    r, found = _records(MTHFR)
    exome_only = _with(found[MTHFR][0], joint=None, genome=None)
    cols = r.extract([exome_only])
    assert (cols["gnomad_af"], cols["gnomad_ac"], cols["gnomad_an"], cols["gnomad_nhom"]) == (repr(471698 / 1461758), "471698", "1461758", "80805")
    assert cols["gnomad_af_genome"] == "" and cols["gnomad_nhom_genome"] == ""
    assert cols["gnomad_af_exome"] == "0.3226922650671315"
    assert (cols["gnomad_grpmax_af"], cols["gnomad_grpmax_pop"]) == (repr(21885 / 44724), "amr")
    genome_only = _with(found[MTHFR][0], joint=None, exome=None)
    cols = r.extract([genome_only])
    assert (cols["gnomad_ac"], cols["gnomad_an"], cols["gnomad_af_exome"]) == ("41850", "152088", "")
    assert (cols["gnomad_faf95_popmax"], cols["gnomad_faf95_pop"]) == ("0.43494121999999985", "amr")
    nothing = _with(found[MTHFR][0], joint=None, exome=None, genome=None)
    cols = r.extract([nothing])
    assert cols == {**{c: "" for c in r.columns}, "gnomad_dataset": "gnomad_r4"}


def test_no_joint_payload_stays_verbatim_and_combines():
    stub = StubHttp()
    v = dict(_found()["17-7675088-C-T"], joint=None)  # gnomad_r4_non_ukb shape: no joint block
    stub.found["17-7675088-C-T"] = v
    r = GnomadRetriever(stub)
    (rec,) = r.retrieve([TP53])[TP53]
    assert rec.payload == v
    cols = r.extract([rec])
    assert cols["gnomad_ac"] == "7"  # 6 exome + 1 genome
    assert (cols["gnomad_faf95_popmax"], cols["gnomad_faf95_pop"]) == ("1.94e-06", "nfe")  # genome faf95 is null


def test_filters_joined_from_exome_and_genome():
    r, found = _records(TP53)
    rec = found[TP53][0]
    ex = dict(rec.payload["exome"], filters=["AC0"])
    ge = dict(rec.payload["genome"], filters=["AS_VQSR", "InbreedingCoeff"])
    assert r.extract([_with(rec, exome=ex, genome=ge)])["gnomad_filters"] == "exome:AC0;genome:AS_VQSR;genome:InbreedingCoeff"
    assert r.extract([_with(rec, exome=ex, genome=None)])["gnomad_filters"] == "exome:AC0"


def test_grpmax_skips_bottlenecked_and_subset_populations():
    r, found = _records(TP53)
    rec = found[TP53][0]
    pops = [
        {"id": "fin", "ac": 50, "an": 100, "homozygote_count": 0, "hemizygote_count": 0},        # bottlenecked
        {"id": "asj", "ac": 50, "an": 100, "homozygote_count": 0, "hemizygote_count": 0},
        {"id": "remaining", "ac": 50, "an": 100, "homozygote_count": 0, "hemizygote_count": 0},
        {"id": "nfe_XX", "ac": 90, "an": 100, "homozygote_count": 0, "hemizygote_count": 0},    # sex-split
        {"id": "XX", "ac": 90, "an": 100, "homozygote_count": 0, "hemizygote_count": 0},
        {"id": "hgdp:japanese", "ac": 90, "an": 100, "homozygote_count": 0, "hemizygote_count": 0},
        {"id": "", "ac": 900, "an": 1000, "homozygote_count": 0, "hemizygote_count": 0},        # joint total
        {"id": "afr", "ac": 3, "an": 1000, "homozygote_count": 0, "hemizygote_count": 0},
        {"id": "eas", "ac": 4, "an": 1000, "homozygote_count": 0, "hemizygote_count": 0},
        {"id": "eas", "ac": 4, "an": 1000, "homozygote_count": 0, "hemizygote_count": 0},        # duplicate entry
        {"id": "sas", "ac": 0, "an": 0, "homozygote_count": 0, "hemizygote_count": 0},          # no calls
    ]
    joint = dict(rec.payload["joint"], populations=pops)
    cols = r.extract([_with(rec, joint=joint)])
    assert (cols["gnomad_grpmax_af"], cols["gnomad_grpmax_pop"]) == (repr(4 / 1000), "eas")
    # every eligible population at zero: max af is 0.0 but no population "has" it
    zero = [dict(p, ac=0) for p in pops]
    cols = r.extract([_with(rec, joint=dict(joint, populations=zero))])
    assert (cols["gnomad_grpmax_af"], cols["gnomad_grpmax_pop"]) == ("0.0", "")
    cols = r.extract([_with(rec, joint=dict(joint, populations=[]))])
    assert (cols["gnomad_grpmax_af"], cols["gnomad_grpmax_pop"]) == ("", "")


def test_extract_guards_null_leaves():
    r, found = _records(TP53)
    rec = found[TP53][0]
    joint = dict(rec.payload["joint"], ac=None, an=None, homozygote_count=None, populations=None, faf95=None)
    cols = r.extract([_with(rec, joint=joint, exome=dict(rec.payload["exome"], af=None, filters=None))])
    assert (cols["gnomad_af"], cols["gnomad_ac"], cols["gnomad_an"], cols["gnomad_nhom"]) == ("", "", "", "")
    assert cols["gnomad_af_exome"] == "" and cols["gnomad_filters"] == ""
    assert (cols["gnomad_faf95_popmax"], cols["gnomad_faf95_pop"]) == ("", "")
    assert cols["gnomad_nhom_exome"] == "0"


# ---------------------------------------------------------------- live

@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ENGINE_LIVE_TESTS"), reason="set ENGINE_LIVE_TESTS=1 to hit gnomAD")
def test_live_mthfr(tmp_path: Path):
    from engine.retrieve.http import Http, HttpCache, RateLimiter

    http = Http(HttpCache(tmp_path / "cache"), limiter=RateLimiter({"gnomad.broadinstitute.org": 0.15}),
                backoff_floor={"gnomad.broadinstitute.org": 75.0})
    r = GnomadRetriever(http)
    found = r.retrieve([MTHFR])
    (rec,) = found[MTHFR]
    assert rec.record_id == "gnomad:1-11796321-G-A"
    assert rec.payload["rsids"] == ["rs1801133"] and rec.payload["reference_genome"] == "GRCh38"
    cols = r.extract([rec])
    assert 0.2 < float(cols["gnomad_af"]) < 0.5 and int(cols["gnomad_nhom"]) > 10000
    assert cols["gnomad_grpmax_pop"] == "amr" and 0 < float(cols["gnomad_faf95_popmax"]) < float(cols["gnomad_grpmax_af"])
    assert r.version().startswith("gnomad_r4 clinvar_release_date=")
    # warm cache → identical bytes, no network
    again = GnomadRetriever(Http(HttpCache(tmp_path / "cache"), offline=True)).retrieve([MTHFR])
    assert again[MTHFR][0].to_json() == rec.to_json()
