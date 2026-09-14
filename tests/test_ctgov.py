"""ClinicalTrials.gov retriever against recorded real responses (public queries only:
cystic fibrosis / ivacaftor / CFTR, the public trials NCT00909532, NCT02971839,
NCT03278314, NCT06191640, the alias pair NCT07048704 → NCT07280598, the missing id
NCT99999999 and a nonsense condition).

Every fixture in ``tests/fixtures/ctgov/`` is the engine's own HTTP cache entry from one
live request the retriever itself made — request, status, headers, retrieval time, body
text — so the stub Http serves a fixture only when the retriever asks for exactly that
request and refuses anything else. Failure shapes are exercised by swapping a fixture's
body; nothing synthetic is ever sent anywhere.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path

import pytest

from engine.medicine.ctgov import (API_URL, COLUMNS, DEFAULT_SORT, FIELDS, SITE_STATUS_CODES, STATUSES, CtgovError,
                                   SearchResult, TrialsRetriever, _search_url, valid_nct_id)
from engine.retrieve.http import Http, HttpCache, HttpError, RateLimiter, Response
from engine.retrieve.store import EvidenceRecord, EvidenceStore

FIXTURES = Path(__file__).parent / "fixtures" / "ctgov"
VERSION = "ClinicalTrials.gov API 2.0.5 data 2026-09-11T09:00:04"
RECORD_VERSION = "ClinicalTrials.gov API 2.0.5 data 2026-09-11"
FIELDS_PARAM = ",".join(FIELDS)
PAGE1 = ["NCT06191640", "NCT05331183", "NCT07809867"]
PAGE2 = ["NCT04602468", "NCT05668741", "NCT06237335"]
STRIVE = "NCT00909532"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def body_of(name: str) -> dict:
    return json.loads(fixture(name)["text"])


def served(fx: dict, *, body=None, text: str | None = None, status: int | None = None) -> dict:
    """A copy of a recorded fixture with its body/status swapped — for failure shapes
    that must be served in reply to a request the retriever actually makes."""
    out = copy.deepcopy(fx)
    if body is not None:
        out["text"] = json.dumps(body)
    if text is not None:
        out["text"] = text
    if status is not None:
        out["status"] = status
    return out


class StubHttp:
    """Serves recorded responses keyed by the exact request (method, url, params, body),
    honouring ``cache_404`` the way :class:`Http` does. The stub *is* the HTTP cache: a
    response it serves reports ``from_cache=True`` like a replay, except one asked for
    with ``cache_ok=False`` (live), or — with ``cold=True`` — the first time a request is
    served, as a cache that has just been filled would. A fixture added with ``live=True``
    answers only requests made with ``cache_ok=False`` (what the server says *now*), so
    a cache that disagrees with the live service can be staged; ``offline`` makes any
    ``cache_ok=False`` request fail the way :class:`Http` does."""

    def __init__(self, *fixtures: dict, per_second: float = 0, offline: bool = False, cold: bool = False):
        self.limiter = RateLimiter(default_per_second=per_second)
        self.offline = offline
        self.cold = cold
        self.calls: list[dict] = []
        self._by_key: dict[str, dict] = {}
        self._live_by_key: dict[str, dict] = {}
        self._served: set[str] = set()
        for fx in fixtures:
            self.add(fx)

    def add(self, fx: dict, *, live: bool = False) -> None:
        r = fx["request"]
        (self._live_by_key if live else self._by_key)[HttpCache.key(r["method"], r["url"], r["params"], r["body"])] = fx

    def request(self, method: str, url: str, *, params=None, json_body=None, cache_404: bool = False, **kw) -> Response:
        self.calls.append({"method": method, "url": url, "params": params, "cache_404": cache_404, **kw})
        key = HttpCache.key(method, url, params, json_body)
        live = kw.get("cache_ok", True) is False
        if live:
            if self.offline:
                raise RuntimeError(f"offline: no cached response for {method} {url}")
            fx = self._live_by_key.get(key) or self._by_key.get(key)
        else:
            fx = self._by_key.get(key)
        if fx is None:
            raise AssertionError(f"no fixture for {method} {url} {json.dumps(params)[:160]}")
        from_cache = not live and not (self.cold and key not in self._served)
        self._served.add(key)
        if fx["status"] == 404 and cache_404:
            return Response(404, fx["text"], fx["retrieved_at"], from_cache, "stub", fx.get("headers", {}))
        if fx["status"] >= 400:
            raise HttpError(fx["status"], url, fx.get("text") or "")
        return Response(fx["status"], fx["text"], fx["retrieved_at"], from_cache, "stub", fx.get("headers", {}))

    def get(self, url: str, **kw) -> Response:
        return self.request("GET", url, **kw)


def _search_calls(stub: StubHttp) -> list[dict]:
    return [c["params"] for c in stub.calls if c["url"] == API_URL + "studies"]


def _version_calls(stub: StubHttp) -> list[bool]:
    """``cache_ok`` of every /version request, in order."""
    return [c.get("cache_ok", True) for c in stub.calls if c["url"] == API_URL + "version"]


def _cf() -> tuple[TrialsRetriever, StubHttp]:
    stub = StubHttp(fixture("version"), fixture("search_cf_ivacaftor_size3_page1"), fixture("search_cf_ivacaftor_size3_page2"))
    return TrialsRetriever(stub, page_size=3), stub


def _study_fixture(nct: str) -> dict:
    return body_of(f"study_{nct}")


# ---------------------------------------------------------------- search: pages, records, the search record


def test_search_pages_until_max_results_and_records_are_verbatim():
    r, stub = _cf()
    recs = r.search("cystic fibrosis", "ivacaftor", max_results=5)
    assert [x.record_id for x in recs] == [f"nct:{n}" for n in PAGE1 + PAGE2[:2]]
    p1, p2 = _search_calls(stub)
    assert p1 == {"query.cond": "cystic fibrosis", "query.intr": "ivacaftor", "fields": FIELDS_PARAM, "sort": DEFAULT_SORT,
                  "pageSize": 3, "countTotal": "true"}
    # page 2: the same query/fields/sort/pageSize byte for byte; only countTotal goes and pageToken comes
    assert p2 == {**{k: v for k, v in p1.items() if k != "countTotal"}, "pageToken": body_of("search_cf_ivacaftor_size3_page1")["nextPageToken"]}

    first = recs[0]
    raw = body_of("search_cf_ivacaftor_size3_page1")["studies"][0]
    assert first.record_id == "nct:NCT06191640" and first.source == "nct"
    assert first.source_version == RECORD_VERSION  # the record's own snapshot stamp, a date
    assert first.query == {"nctId": "NCT06191640", "fields": FIELDS_PARAM,
                           "api": f"{API_URL}studies/NCT06191640?fields=" + FIELDS_PARAM.replace(",", "%2C")}
    assert first.url == "https://clinicaltrials.gov/study/NCT06191640"
    assert first.retrieved_at == fixture("search_cf_ivacaftor_size3_page1")["retrieved_at"]  # from the Response, not the clock
    assert first.payload == raw  # verbatim: the study object exactly as served
    assert first.payload["derivedSection"]["miscInfoModule"]["versionHolder"] == "2026-09-11"
    assert recs[3].retrieved_at == fixture("search_cf_ivacaftor_size3_page2")["retrieved_at"]
    # most recently updated first, as the explicit sort asks
    dates = [x.payload["protocolSection"]["statusModule"]["lastUpdatePostDateStruct"]["date"] for x in recs]
    assert dates == sorted(dates, reverse=True)


def test_search_stops_when_the_first_page_has_enough():
    r, stub = _cf()
    recs = r.search("cystic fibrosis", "ivacaftor", max_results=3)
    assert [x.record_id for x in recs] == [f"nct:{n}" for n in PAGE1]
    assert len(_search_calls(stub)) == 1


def test_find_returns_the_search_itself_as_a_record():
    r, stub = _cf()
    res = r.find("cystic fibrosis", "ivacaftor", max_results=5)
    assert isinstance(res, SearchResult) and res.nct_ids == PAGE1 + PAGE2[:2] and res.total_count == 187
    rec = res.record
    assert rec.record_id.startswith("nct-search:") and len(rec.record_id) == len("nct-search:") + 64
    assert rec.source == "nct-search"
    assert rec.source_version == RECORD_VERSION  # the stamps observed on the studies it lists, not /version
    assert rec.query == {"condition": "cystic fibrosis", "intervention": "ivacaftor", "term": None, "statuses": [],
                         "max_results": 5, "api": API_URL + "studies"}
    assert rec.url == "https://clinicaltrials.gov/search?cond=cystic%20fibrosis&intr=ivacaftor"  # %20: the site's own spelling
    assert rec.retrieved_at == fixture("search_cf_ivacaftor_size3_page1")["retrieved_at"]
    sent = {k: v for k, v in _search_calls(stub)[0].items() if k != "countTotal"}  # what every page carried
    assert rec.payload == {"sent": sent, "totalCount": 187, "nctIds": PAGE1 + PAGE2[:2], "pages": 2,
                           "versionHolders": ["2026-09-11"]}
    # a different ask is a different record; the same ask is the same bytes on another instance
    other = _cf()[0].find("cystic fibrosis", "ivacaftor", max_results=3)
    assert other.record.record_id != rec.record_id and other.record.payload["pages"] == 1
    assert _cf()[0].find("cystic fibrosis", "ivacaftor", max_results=5).record.to_json() == rec.to_json()


def test_search_record_identity_is_the_ask_not_the_page_size():
    """The page size is transport: the same ask fetched in pages of 5 is the same record
    as in pages of 3 (``payload.sent.pageSize`` and ``payload.pages`` say how), so two
    orchestrators that page differently do not mint two records for one search."""
    by_three = _cf()[0].find("cystic fibrosis", "ivacaftor", max_results=5)
    # the recorded first page, as a single page of 5 (the token to a second page dropped): nothing is sent anywhere
    page = fixture("search_cf_ivacaftor_size3_page1")
    one_page = served(page, body={k: v for k, v in body_of("search_cf_ivacaftor_size3_page1").items() if k != "nextPageToken"})
    one_page["request"]["params"] = {**page["request"]["params"], "pageSize": 5}
    by_five = TrialsRetriever(StubHttp(fixture("version"), one_page), page_size=5).find("cystic fibrosis", "ivacaftor", max_results=5)
    assert by_five.record.record_id == by_three.record.record_id
    assert by_five.record.payload["sent"]["pageSize"] == 5 and by_three.record.payload["sent"]["pageSize"] == 3
    assert by_five.record.payload["pages"] == 1 and by_five.nct_ids == PAGE1
    # while the sort is part of the ask: the same page served for another order is another record
    by_relevance = copy.deepcopy(one_page)
    by_relevance["request"]["params"]["sort"] = "@relevance"
    r = TrialsRetriever(StubHttp(fixture("version"), by_relevance), page_size=5, sort="@relevance")
    assert r.find("cystic fibrosis", "ivacaftor", max_results=5).record.record_id != by_three.record.record_id


def test_empty_search_is_absence_with_a_citable_zero():
    stub = StubHttp(fixture("version"), fixture("search_nonsense_size3_page1"))
    r = TrialsRetriever(stub, page_size=3)
    res = r.find("zzqxnotacondition987654", max_results=3)
    assert res.studies == [] and res.nct_ids == [] and res.total_count == 0 and res.record.payload["pages"] == 1
    # no study, no snapshot stamp observed: the record claims the API version only, never a guessed night
    assert res.record.payload["versionHolders"] == [] and res.record.source_version == "ClinicalTrials.gov API 2.0.5"
    assert r.search("zzqxnotacondition987654", max_results=3) == []
    assert len(_search_calls(stub)) == 2  # the stub has no cache; each search is one request


def test_a_first_page_that_claims_matches_but_lists_none_raises():
    r = TrialsRetriever(StubHttp(fixture("version"), served(fixture("search_nonsense_size3_page1"), body={"totalCount": 187, "studies": []})), page_size=3)
    with pytest.raises(CtgovError, match="totalCount 187 but its first page lists no study"):
        r.find("zzqxnotacondition987654", max_results=3)


def test_term_and_status_filter_are_sent():
    stub = StubHttp(fixture("version"), fixture("search_cf_term_cftr_recruiting_size3_page1"))
    r = TrialsRetriever(stub, page_size=3)
    res = r.find("cystic fibrosis", term="CFTR", statuses=["recruiting", "RECRUITING"], max_results=3)
    assert res.total_count == 44 and res.nct_ids == ["NCT06191640", "NCT06560463", "NCT04509050"]
    assert _search_calls(stub)[0] == {"query.cond": "cystic fibrosis", "query.term": "CFTR", "filter.overallStatus": "RECRUITING",
                                      "fields": FIELDS_PARAM, "sort": DEFAULT_SORT, "pageSize": 3, "countTotal": "true"}
    assert res.record.query["statuses"] == ["RECRUITING"] and res.record.query["term"] == "CFTR"
    assert res.record.url == "https://clinicaltrials.gov/search?cond=cystic%20fibrosis&term=CFTR&aggFilters=status:rec"
    assert all(x.payload["protocolSection"]["statusModule"]["overallStatus"] == "RECRUITING" for x in res.studies)


def test_a_listed_study_outside_the_status_filter_raises():
    """The registry has been seen to drop ``filter.overallStatus`` silently (an id-only
    query); a study the answer lists with another status contradicts what the search
    record would claim and is not one to cite."""
    page = body_of("search_cf_term_cftr_recruiting_size3_page1")
    page["studies"][1]["protocolSection"]["statusModule"]["overallStatus"] = "COMPLETED"
    stub = StubHttp(fixture("version"), served(fixture("search_cf_term_cftr_recruiting_size3_page1"), body=page))
    with pytest.raises(CtgovError, match="filter.overallStatus=RECRUITING listed NCT06560463 with overallStatus 'COMPLETED'"):
        TrialsRetriever(stub, page_size=3).find("cystic fibrosis", term="CFTR", statuses=["RECRUITING"], max_results=3)
    del page["studies"][2]["protocolSection"]["statusModule"]["overallStatus"]  # absent is not "in the filter" either
    page["studies"][1]["protocolSection"]["statusModule"]["overallStatus"] = "RECRUITING"
    stub = StubHttp(fixture("version"), served(fixture("search_cf_term_cftr_recruiting_size3_page1"), body=page))
    with pytest.raises(CtgovError, match="listed NCT04509050 with overallStatus None"):
        TrialsRetriever(stub, page_size=3).find("cystic fibrosis", term="CFTR", statuses=["RECRUITING"], max_results=3)


def test_a_query_part_that_is_only_nct_ids_is_refused_before_any_request():
    """Verified live: ``cond=asthma&term=NCT00909532&filter.overallStatus=RECRUITING``
    answers the completed cystic-fibrosis trial — an id-only part switches the registry
    to id lookup and the filter and the other parts are ignored. Any of the three parts
    does it, in any case, whitespace- or comma-separated; text around an id does not."""
    stub = StubHttp()
    r = TrialsRetriever(stub)
    for text in ("NCT00909532", "nct00909532", " NCT00909532 ", "NCT00909532 NCT02971839", "NCT00909532, NCT02971839",
                 "NCT00909532,NCT02971839", "NCT909532"):
        for part in ("condition", "intervention", "term"):
            ask = {"condition": "cystic fibrosis", "intervention": None, "term": None, part: text}
            with pytest.raises(ValueError, match="only NCT ids.*study\\(\\)"):
                r.find(**ask, statuses=["RECRUITING"])
    assert stub.calls == []
    for text in ("NCT00909532 OR NCT02971839", "ivacaftor NCT00909532", "NCT-00909532"):  # text, which the registry searches
        with pytest.raises(AssertionError, match="no fixture"):  # passed validation and reached the (empty) stub
            r.find(None, term=text)


def test_search_url_spells_the_status_filter_the_way_the_site_does():
    ask = {"condition": "cystic fibrosis", "intervention": None, "term": None}
    assert _search_url({**ask, "statuses": []}) == "https://clinicaltrials.gov/search?cond=cystic%20fibrosis"
    assert _search_url({**ask, "statuses": ["RECRUITING", "COMPLETED"]}) == "https://clinicaltrials.gov/search?cond=cystic%20fibrosis&aggFilters=status:rec%20com"
    assert _search_url({**ask, "statuses": ["APPROVED_FOR_MARKETING", "NO_LONGER_AVAILABLE", "TEMPORARILY_NOT_AVAILABLE"]}).endswith("aggFilters=status:afm%20nla%20tna")
    # WITHHELD has no code on the site: the page URL is then the unfiltered ask (a superset), never a wrong subset
    assert _search_url({**ask, "statuses": ["RECRUITING", "WITHHELD"]}) == "https://clinicaltrials.gov/search?cond=cystic%20fibrosis"
    # never a '+' for a space (a decoder may keep it literal); other reserved characters are escaped as before
    assert _search_url({**ask, "intervention": "VX-770 / ivacaftor", "statuses": []}) == "https://clinicaltrials.gov/search?cond=cystic%20fibrosis&intr=VX-770%20%2F%20ivacaftor"
    assert set(STATUSES) - {"WITHHELD"} == set(SITE_STATUS_CODES)


def test_bad_asks_are_refused_before_any_request():
    stub = StubHttp()
    r = TrialsRetriever(stub)
    with pytest.raises(ValueError, match="overallStatus"):
        r.search("cystic fibrosis", statuses=["BOGUS"])
    with pytest.raises(ValueError, match="condition, an intervention or a term"):
        r.search("   ", None)
    with pytest.raises(ValueError, match="condition, an intervention or a term"):
        r.search(None)
    with pytest.raises(ValueError, match="control characters"):
        r.search("cystic\nfibrosis")
    with pytest.raises(ValueError, match="max_results"):
        r.search("cystic fibrosis", max_results=0)
    assert stub.calls == []
    assert STATUSES >= {"RECRUITING", "COMPLETED", "TERMINATED", "WITHHELD", "APPROVED_FOR_MARKETING"}


def test_same_trial_is_the_same_record_whichever_route_found_it():
    stub = StubHttp(fixture("version"), fixture("search_cf_ivacaftor_size3_page1"),
                    fixture("search_cf_term_cftr_recruiting_size3_page1"), fixture("study_NCT06191640"))
    r = TrialsRetriever(stub, page_size=3)
    via_a = r.search("cystic fibrosis", "ivacaftor", max_results=3)[0]
    via_b = r.search("cystic fibrosis", term="CFTR", statuses=["RECRUITING"], max_results=3)[0]
    direct = r.study("NCT06191640")
    assert via_a.record_id == via_b.record_id == direct.record_id == "nct:NCT06191640"
    for a, b in ((via_a, via_b), (via_a, direct)):
        assert a.payload == b.payload and a.query == b.query and a.url == b.url and a.source_version == b.source_version
        assert EvidenceRecord(**{**a.__dict__, "retrieved_at": b.retrieved_at}).to_json() == b.to_json()


# ---------------------------------------------------------------- one study


def test_study_by_id():
    stub = StubHttp(fixture("version"), fixture("study_NCT00909532"))
    r = TrialsRetriever(stub)
    rec = r.study(" nct00909532 ")
    assert rec is not None and rec.record_id == f"nct:{STRIVE}" and rec.source == "nct"
    detail = [c for c in stub.calls if "/studies/" in c["url"]]
    assert len(detail) == 1 and detail[0]["url"] == f"{API_URL}studies/{STRIVE}" and detail[0]["params"] == {"fields": FIELDS_PARAM}
    assert detail[0]["cache_404"] is True
    assert _version_calls(stub) == [True, False] and stub.calls.index(detail[0]) == 2  # the version is observed first
    assert rec.source_version == RECORD_VERSION
    assert rec.query == {"nctId": STRIVE, "fields": FIELDS_PARAM, "api": f"{API_URL}studies/{STRIVE}?fields=" + FIELDS_PARAM.replace(",", "%2C")}
    assert rec.url == f"https://clinicaltrials.gov/study/{STRIVE}"
    assert rec.retrieved_at == fixture("study_NCT00909532")["retrieved_at"]
    assert rec.payload == _study_fixture(STRIVE)
    ident = rec.payload["protocolSection"]["identificationModule"]
    assert ident["acronym"] == "STRIVE" and ident["orgStudyIdInfo"]["id"] == "VX08-770-102"
    assert rec.payload["hasResults"] is True and "resultsSection" not in rec.payload  # fields-limited: never the 73 KB record


def test_missing_study_is_none_and_malformed_ids_never_reach_the_api():
    stub = StubHttp(fixture("version"), fixture("study_NCT99999999"))
    r = TrialsRetriever(stub)
    assert r.study("NCT99999999") is None
    assert stub.calls[-1]["cache_404"] is True  # the 404 is definitive and cached like any answer
    n = len(stub.calls)
    for bad in ("FOO", "", "NCT909532", "NCT0909532", "NCT009095321", "NCT-00909532", "nct 00909532", "N CT00909532",
                "NCT00000000"):  # all zeros: the API's own id pattern is NCT0*[1-9]\d{0,7}; it answers HTTP 400
        with pytest.raises(ValueError, match="NCT id"):
            r.study(bad)
    assert len(stub.calls) == n
    assert valid_nct_id("nct00909532") == STRIVE


def test_only_the_registrys_own_not_found_is_absence():
    """A 404 is cached as an answer, so a 404 of another shape (wrong API path, edge/CDN
    page) must raise on every replay rather than read as a permanent absence."""
    registry = fixture("study_NCT99999999")
    assert registry["status"] == 404 and registry["text"] == "NCT number NCT99999999 not found"
    assert TrialsRetriever(StubHttp(fixture("version"), registry)).study("NCT99999999") is None
    for text in ("", "<html><head><title>404 Not Found</title></head><body><center><h1>404 Not Found</h1></center></body></html>",
                 "NCT number NCT00000001 not found"):  # names another id
        stub = StubHttp(fixture("version"), served(registry, text=text))
        with pytest.raises(CtgovError, match="HTTP 404 that is not the registry's 'not found'"):
            TrialsRetriever(stub).study("NCT99999999")
        assert stub.calls[-1]["cache_404"] is True


def test_alias_resolves_to_the_canonical_entry():
    stub = StubHttp(fixture("version"), fixture("study_NCT07048704"), fixture("study_NCT07280598"))
    r = TrialsRetriever(stub)
    rec = r.study("NCT07048704")
    assert rec is not None and rec.record_id == "nct:NCT07280598"
    assert [c["url"].rsplit("/", 1)[1] for c in stub.calls if "/studies/" in c["url"]] == ["NCT07048704", "NCT07280598"]
    # the redirect answered with the full record (no fields); the record is the fields-limited canonical one
    assert "oversightModule" in _study_fixture("NCT07048704")["protocolSection"]
    assert "oversightModule" not in rec.payload["protocolSection"]
    assert rec.payload == _study_fixture("NCT07280598")
    assert rec.payload["protocolSection"]["identificationModule"]["nctIdAliases"] == ["NCT07048704"]
    assert rec.query["nctId"] == "NCT07280598" and rec.url.endswith("/NCT07280598")
    assert rec.to_json() == TrialsRetriever(StubHttp(fixture("version"), fixture("study_NCT07280598"))).study("NCT07280598").to_json()


def test_an_answer_for_another_id_that_is_not_an_alias_raises():
    body = _study_fixture("NCT07048704")
    del body["protocolSection"]["identificationModule"]["nctIdAliases"]
    stub = StubHttp(fixture("version"), served(fixture("study_NCT07048704"), body=body))
    with pytest.raises(CtgovError, match="does not list it as an alias"):
        TrialsRetriever(stub).study("NCT07048704")
    # alias acknowledged, but the canonical entry is then not found, or answers for a third id
    stub = StubHttp(fixture("version"), fixture("study_NCT07048704"), served(fixture("study_NCT07280598"), status=404, text="not found"))
    with pytest.raises(CtgovError, match="then not found"):
        TrialsRetriever(stub).study("NCT07048704")
    third = _study_fixture("NCT07280598")
    third["protocolSection"]["identificationModule"]["nctId"] = "NCT00000001"
    stub = StubHttp(fixture("version"), fixture("study_NCT07048704"), served(fixture("study_NCT07280598"), body=third))
    with pytest.raises(CtgovError, match="answered for NCT00000001"):
        TrialsRetriever(stub).study("NCT07048704")


# ---------------------------------------------------------------- failure is never absence


def _search_served(**kw) -> TrialsRetriever:
    return TrialsRetriever(StubHttp(fixture("version"), served(fixture("search_cf_ivacaftor_size3_page1"), **kw)), page_size=3)


def test_non_json_and_non_object_bodies_raise():
    with pytest.raises(CtgovError, match="non-JSON"):
        _search_served(text="<html><title>414 Request-URI Too Large</title></html>").search("cystic fibrosis", "ivacaftor", 3)
    with pytest.raises(CtgovError, match="expected an object"):
        _search_served(text="[]").search("cystic fibrosis", "ivacaftor", 3)
    with pytest.raises(CtgovError, match="no studies list"):
        _search_served(body={"totalCount": 1}).search("cystic fibrosis", "ivacaftor", 3)
    with pytest.raises(CtgovError, match="no totalCount"):
        _search_served(body={"studies": []}).search("cystic fibrosis", "ivacaftor", 3)


def test_a_total_below_what_the_first_page_lists_raises():
    """The first page is counted under the same query: a total smaller than it (a zero
    with studies present, say) is a self-contradictory answer, not one to record."""
    page = body_of("search_cf_ivacaftor_size3_page1")
    for total in (0, 1, 2):
        with pytest.raises(CtgovError, match=f"totalCount {total} but its first page lists 3 studies"):
            _search_served(body={**page, "totalCount": total}).search("cystic fibrosis", "ivacaftor", 3)
    assert len(_search_served(body={**page, "totalCount": 3}).search("cystic fibrosis", "ivacaftor", 3)) == 3


def test_study_shapes_that_raise():
    page = body_of("search_cf_ivacaftor_size3_page1")
    no_id = copy.deepcopy(page)
    del no_id["studies"][0]["protocolSection"]["identificationModule"]["nctId"]
    with pytest.raises(CtgovError, match="no NCT id"):
        _search_served(body=no_id).search("cystic fibrosis", "ivacaftor", 3)
    no_stamp = copy.deepcopy(page)
    del no_stamp["studies"][1]["derivedSection"]["miscInfoModule"]
    with pytest.raises(CtgovError, match="no versionHolder"):
        _search_served(body=no_stamp).search("cystic fibrosis", "ivacaftor", 3)
    stuck = copy.deepcopy(page)
    stub = StubHttp(fixture("version"), served(fixture("search_cf_ivacaftor_size3_page1"), body=stuck))
    # page 2 hands back the very token that fetched it
    p2 = copy.deepcopy(fixture("search_cf_ivacaftor_size3_page2"))
    p2_body = json.loads(p2["text"])
    p2_body["nextPageToken"] = p2["request"]["params"]["pageToken"]
    stub.add(served(p2, body=p2_body))
    with pytest.raises(CtgovError, match="repeated a pageToken"):
        TrialsRetriever(stub, page_size=3).search("cystic fibrosis", "ivacaftor", 50)


def test_http_errors_propagate():
    with pytest.raises(HttpError):  # 5xx after retries, 4xx at once — Http raises, the retriever does not catch
        _search_served(status=500, text="").search("cystic fibrosis", "ivacaftor", 3)
    with pytest.raises(HttpError, match="400"):
        TrialsRetriever(StubHttp(fixture("version"), served(fixture("study_NCT00909532"), status=400, text="Parameter `nctId` has incorrect format"))).study(STRIVE)
    # a 404 on the *search* endpoint is not "no trials": the stub raises as Http would without cache_404
    with pytest.raises(HttpError, match="404"):
        _search_served(status=404, text="not found").search("cystic fibrosis", "ivacaftor", 3)


def test_last_page_without_a_token_and_an_empty_page_end_paging():
    p1 = body_of("search_cf_ivacaftor_size3_page1")
    no_token = {k: v for k, v in p1.items() if k != "nextPageToken"}
    r = _search_served(body=no_token)
    assert len(r.search("cystic fibrosis", "ivacaftor", 50)) == 3
    stub = StubHttp(fixture("version"), fixture("search_cf_ivacaftor_size3_page1"),
                    served(fixture("search_cf_ivacaftor_size3_page2"), body={"studies": [], "nextPageToken": "whatever"}))
    res = TrialsRetriever(stub, page_size=3).find("cystic fibrosis", "ivacaftor", 50)
    assert res.nct_ids == PAGE1 and res.record.payload["pages"] == 2 and len(_search_calls(stub)) == 2


def test_a_trial_repeated_across_pages_is_one_record():
    p2 = copy.deepcopy(fixture("search_cf_ivacaftor_size3_page2"))
    p2_body = json.loads(p2["text"])
    p2_body["studies"][0] = body_of("search_cf_ivacaftor_size3_page1")["studies"][0]  # NCT06191640 again
    stub = StubHttp(fixture("version"), fixture("search_cf_ivacaftor_size3_page1"), served(p2, body=p2_body))
    res = TrialsRetriever(stub, page_size=3).find("cystic fibrosis", "ivacaftor", 5)
    assert res.nct_ids == PAGE1 + PAGE2[1:]


def _seed(cache: HttpCache, name: str, **over) -> None:
    """Put a recorded fixture into a real HTTP cache exactly as :class:`Http` would have."""
    d = {**fixture(name), **over}
    r = d["request"]
    key = HttpCache.key(r["method"], r["url"], r["params"], r["body"])
    cache.put(key, Response(d["status"], d["text"], d["retrieved_at"], False, key, d.get("headers", {})), r)


def test_a_cached_404_replays_through_the_real_http_as_it_was_answered(tmp_path: Path):
    cache = HttpCache(tmp_path / "cache")
    _seed(cache, "version")
    _seed(cache, "study_NCT99999999")
    r = TrialsRetriever(Http(cache, offline=True))
    assert r.study("NCT99999999") is None  # the registry's own 'not found', replayed: still absence
    bogus = HttpCache(tmp_path / "bogus")
    _seed(bogus, "version")
    _seed(bogus, "study_NCT99999999", text="", headers={})  # what a wrong API path answers: 404, empty body
    with pytest.raises(CtgovError, match="HTTP 404 that is not the registry's"):
        TrialsRetriever(Http(bogus, offline=True)).study("NCT99999999")


# ---------------------------------------------------------------- version and snapshot stamps


def test_version_is_observed_once_cached_then_checked_live_and_required():
    stub = StubHttp(fixture("version"))
    r = TrialsRetriever(stub)
    assert r.version() == VERSION == r.version()
    assert r.version_info() == {"apiVersion": "2.0.5", "dataTimestamp": "2026-09-11T09:00:04"}
    # online, warm cache: the cache's observation, then the live one (a check, never stored) — once per instance
    assert [c["url"] for c in stub.calls] == [API_URL + "version"] * 2 and _version_calls(stub) == [True, False]
    offline = StubHttp(fixture("version"), offline=True)
    assert TrialsRetriever(offline).version() == VERSION and _version_calls(offline) == [True]
    # cold cache: the first observation was itself live, so there is nothing to check it against — one request
    cold = StubHttp(fixture("version"), fixture("study_NCT00909532"), cold=True)
    assert TrialsRetriever(cold).study(STRIVE) is not None and _version_calls(cold) == [True]
    with pytest.raises(CtgovError, match="apiVersion/dataTimestamp"):
        TrialsRetriever(StubHttp(served(fixture("version"), body={"apiVersion": "2.0.5"}))).version()
    with pytest.raises(CtgovError, match="non-JSON"):
        TrialsRetriever(StubHttp(served(fixture("version"), text="oops"))).version()


def test_version_cites_the_cache_it_replays_and_the_live_load_only_logs(caplog):
    """The manifest's line is what the cache was filled under — the same string on a
    warm cache whatever night it is replayed, online or offline (the sibling
    retrievers' convention). The registry's newer nightly load is logged, not cited;
    the study records cite the stamp on their own payload either way."""
    stale = served(fixture("version"), body={"apiVersion": "2.0.5", "dataTimestamp": "2026-09-04T09:00:04"})
    stub = StubHttp(stale, fixture("search_cf_ivacaftor_size3_page1"))
    stub.add(fixture("version"), live=True)  # the registry now reports the 2026-09-11 load
    r = TrialsRetriever(stub, page_size=3)
    with caplog.at_level(logging.INFO, logger="engine.medicine.ctgov"):
        res = r.find("cystic fibrosis", "ivacaftor", max_results=3)
    assert r.version() == "ClinicalTrials.gov API 2.0.5 data 2026-09-04T09:00:04"  # the cache's, not the live load
    assert _version_calls(stub) == [True, False]
    assert res.record.source_version == RECORD_VERSION and all(x.source_version == RECORD_VERSION for x in res.studies)
    assert [m for m in caplog.messages if "nightly reload" in m and "2026-09-04T09:00:04" in m and "2026-09-11T09:00:04" in m]
    assert [m for m in caplog.messages if "spans" in m]  # the studies came from a later night than the cache's /version
    # offline, and online on another night, the same string and the same record bytes
    again = TrialsRetriever(StubHttp(stale, fixture("search_cf_ivacaftor_size3_page1"), offline=True), page_size=3)
    assert again.version() == r.version()
    assert again.find("cystic fibrosis", "ivacaftor", max_results=3).record.to_json() == res.record.to_json()
    later = StubHttp(stale, fixture("search_cf_ivacaftor_size3_page1"))
    later.add(served(fixture("version"), body={"apiVersion": "2.0.5", "dataTimestamp": "2026-09-20T09:00:04"}), live=True)
    assert TrialsRetriever(later, page_size=3).version() == r.version()


def test_params_names_every_knob_for_the_manifest():
    stub = StubHttp()
    r = TrialsRetriever(stub, page_size=3)
    assert r.params == {"api_url": API_URL, "fields": list(FIELDS), "max_page_size": 1000, "page_size": 3,
                        "per_second": 3.0, "sort": DEFAULT_SORT}
    assert json.loads(json.dumps(r.params, sort_keys=True)) == r.params and "version" not in r.params
    assert stub.calls == []  # settings, not observations
    stub.limiter.per_second["clinicaltrials.gov"] = 1.0  # the orchestrator's rate is the one in force
    assert TrialsRetriever(stub, page_size=3).params["per_second"] == 1.0
    assert TrialsRetriever(StubHttp(per_second=2.0)).params["per_second"] == 2.0  # a stricter default: no pin, and it is the rate
    assert TrialsRetriever(stub, fields=["NCTId"], sort="@relevance").params["sort"] == "@relevance"


def test_a_cache_filled_under_another_api_version_is_refused():
    stub = StubHttp(fixture("version"), fixture("study_NCT00909532"))
    stub.add(served(fixture("version"), body={"apiVersion": "2.0.6", "dataTimestamp": "2026-09-11T09:00:04"}), live=True)
    with pytest.raises(CtgovError, match="filled under ClinicalTrials.gov API 2.0.5 but the server now reports 2.0.6"):
        TrialsRetriever(stub).study(STRIVE)
    assert not [c for c in stub.calls if "/studies/" in c["url"]]  # refused before any study request


def test_records_cite_their_own_snapshot_and_a_newer_version_only_warns(caplog):
    moved = served(fixture("version"), body={"apiVersion": "2.0.5", "dataTimestamp": "2026-09-12T09:00:04"})
    stub = StubHttp(moved, fixture("study_NCT00909532"), fixture("study_NCT02971839"))
    r = TrialsRetriever(stub)
    with caplog.at_level(logging.WARNING, logger="engine.medicine.ctgov"):
        a = r.study(STRIVE)
        b = r.study("NCT02971839")
    assert r.version() == "ClinicalTrials.gov API 2.0.5 data 2026-09-12T09:00:04"
    assert a.source_version == b.source_version == RECORD_VERSION  # what the payloads say, not what /version says
    warnings = [m for m in caplog.messages if "spans" in m]
    assert len(warnings) == 1 and "2026-09-11" in warnings[0] and "2026-09-12T09:00:04" in warnings[0]


def test_version_does_not_depend_on_what_was_found():
    stub = StubHttp(fixture("version"), fixture("search_nonsense_size3_page1"), fixture("study_NCT00909532"))
    r = TrialsRetriever(stub, page_size=3)
    empty = r.find("zzqxnotacondition987654", max_results=3)
    assert r.version() == VERSION
    assert empty.record.source_version == "ClinicalTrials.gov API 2.0.5"  # nothing observed: no stamp claimed
    assert r.study(STRIVE).source_version == RECORD_VERSION
    assert _version_calls(stub) == [True, False]  # observed once per instance, however many records


# ---------------------------------------------------------------- records survive the store, byte for byte


def test_records_are_byte_identical_and_survive_the_store(tmp_path: Path):
    a = _cf()[0].find("cystic fibrosis", "ivacaftor", max_results=5)
    b = _cf()[0].find("cystic fibrosis", "ivacaftor", max_results=5)
    assert a.record.to_json() == b.record.to_json()
    assert [x.to_json() for x in a.studies] == [x.to_json() for x in b.studies]
    store = EvidenceStore(tmp_path / "evidence")
    for rec in [a.record, *a.studies]:
        p = store.put(rec)
        assert p.parent.name == rec.source  # the source is what precedes the first colon
        assert store.get(rec.record_id) == rec
    assert store.count("nct") == 5 and store.count("nct-search") == 1
    index = json.loads(store.write_index().read_text())
    assert index["nct:NCT06191640"]["source_version"] == RECORD_VERSION
    assert index["nct:NCT06191640"]["url"] == "https://clinicaltrials.gov/study/NCT06191640"
    assert index[a.record.record_id]["source_version"] == RECORD_VERSION


def test_rate_is_pinned_politely_but_never_overrides_the_orchestrator():
    stub = StubHttp()  # default 0 = unlimited: pin
    TrialsRetriever(stub)
    assert stub.limiter.per_second["clinicaltrials.gov"] == 3.0
    stub.limiter.per_second["clinicaltrials.gov"] = 1.0  # an explicit per-host rate is the orchestrator's call
    TrialsRetriever(stub)
    assert stub.limiter.per_second["clinicaltrials.gov"] == 1.0
    stub.limiter.per_second["clinicaltrials.gov"] = 10.0
    TrialsRetriever(stub)
    assert stub.limiter.per_second["clinicaltrials.gov"] == 10.0
    stricter = StubHttp(per_second=2.0)  # a default already stricter than 3/s stays the effective rate
    TrialsRetriever(stricter)
    assert stricter.limiter.per_second == {} and stricter.limiter.default == 2.0
    looser = StubHttp(per_second=5.0)
    TrialsRetriever(looser)
    assert looser.limiter.per_second["clinicaltrials.gov"] == 3.0


def test_constructor_guards():
    stub = StubHttp()
    for bad in (0, 1001):
        with pytest.raises(ValueError, match="page_size"):
            TrialsRetriever(stub, page_size=bad)
    with pytest.raises(ValueError, match="field name"):
        TrialsRetriever(stub, fields=("NCTId", "Not A Field"))
    with pytest.raises(ValueError, match="at least one"):
        TrialsRetriever(stub, fields=())
    r = TrialsRetriever(stub, fields=["NCTId", "BriefTitle"], sort="@relevance")
    assert r.fields == ("NCTId", "BriefTitle", "OverallStatus", "VersionHolder") and r.sort == "@relevance"
    assert stub.calls == []


def test_a_reduced_field_list_still_yields_records():
    """The three leaves the module reads on every study are appended to any ``fields``
    list that names neither the piece nor its module (the default list names them all,
    so it is sent as is); recorded live with ``fields=NCTId,BriefTitle``."""
    assert TrialsRetriever(StubHttp()).fields == FIELDS
    assert TrialsRetriever(StubHttp(), fields=["IdentificationModule", "StatusModule", "MiscInfoModule"]).fields == \
        ("IdentificationModule", "StatusModule", "MiscInfoModule")
    assert TrialsRetriever(StubHttp(), fields=["BriefTitle", "VersionHolder"]).fields == ("BriefTitle", "VersionHolder", "NCTId", "OverallStatus")
    stub = StubHttp(fixture("version"), fixture("study_NCT00909532_reduced_fields"))
    r = TrialsRetriever(stub, fields=["NCTId", "BriefTitle"])
    rec = r.study(STRIVE)
    assert [c["params"] for c in stub.calls if "/studies/" in c["url"]] == [{"fields": "NCTId,BriefTitle,OverallStatus,VersionHolder"}]
    assert rec is not None and rec.record_id == f"nct:{STRIVE}" and rec.source_version == RECORD_VERSION
    assert rec.query["fields"] == "NCTId,BriefTitle,OverallStatus,VersionHolder"
    assert rec.payload == {"protocolSection": {"identificationModule": {"nctId": STRIVE, "briefTitle": "Study of Ivacaftor in Cystic Fibrosis Subjects Aged 12 Years and Older With the G551D Mutation"},
                                               "statusModule": {"overallStatus": "COMPLETED"}},
                           "derivedSection": {"miscInfoModule": {"versionHolder": "2026-09-11"}}}
    cols = r.extract(rec)
    assert (cols["nct_id"], cols["status"], cols["phases"], cols["has_results"]) == (STRIVE, "COMPLETED", "", "")
    assert cols["title"].startswith("Study of Ivacaftor")


# ---------------------------------------------------------------- extract


def test_extract_strive():
    r = TrialsRetriever(StubHttp(fixture("version"), fixture("study_NCT00909532")))
    cols = r.extract(r.study(STRIVE))
    assert tuple(cols) == COLUMNS == r.columns
    assert cols == {
        "nct_id": STRIVE,
        "aliases": "",
        "title": "Study of Ivacaftor in Cystic Fibrosis Subjects Aged 12 Years and Older With the G551D Mutation",
        "status": "COMPLETED",
        "why_stopped": "",
        "study_type": "INTERVENTIONAL",
        "phases": "PHASE3",
        "conditions": "Cystic Fibrosis",
        "interventions": "DRUG: Ivacaftor;DRUG: Placebo",
        "start_date": "2009-06",  # a partial date, as the registry gives it
        "primary_completion_date": "2010-07",
        "completion_date": "2012-11",
        "enrollment": "167",
        "enrollment_type": "ACTUAL",
        "sponsor": "Vertex Pharmaceuticals Incorporated",
        "sponsor_class": "INDUSTRY",
        "primary_outcomes": "Absolute Mean Change From Baseline in Percent Predicted Forced Expiratory Volume in 1 Second (FEV1) Through Week 24",
        "sex": "ALL",
        "min_age": "12 Years",
        "max_age": "",
        "has_results": "Y",
        "results_first_post_date": "2012-08-21",
        "last_update_post_date": "2013-01-18",
        "condition_mesh": "Cystic Fibrosis;Fibrosis;Pancreatic Diseases;Lung Diseases;Respiratory Tract Diseases;Genetic Diseases, Inborn;Infant, Newborn, Diseases;Pathologic Processes",
        "intervention_mesh": "ivacaftor",
    }


def test_extract_expanded_access_terminated_and_alias():
    stub = StubHttp(fixture("version"), fixture("study_NCT03278314"), fixture("study_NCT02971839"), fixture("study_NCT07280598"))
    r = TrialsRetriever(stub)
    ea = r.extract(r.study("NCT03278314"))  # expanded access: no phases, no enrolment — blank, not an error
    assert (ea["study_type"], ea["status"], ea["phases"], ea["enrollment"], ea["enrollment_type"]) == ("EXPANDED_ACCESS", "APPROVED_FOR_MARKETING", "", "", "")
    assert ea["interventions"].startswith("DRUG: ") and ea["has_results"] == "N"
    te = r.extract(r.study("NCT02971839"))
    assert (te["status"], te["why_stopped"], te["phases"], te["has_results"], te["results_first_post_date"]) == ("TERMINATED", "Decision by Sponsor.", "PHASE2", "Y", "2020-08-26")
    al = r.extract(r.study("NCT07280598"))
    assert al["nct_id"] == "NCT07280598" and al["aliases"] == "NCT07048704"


def test_extract_none_and_foreign_records():
    r, _ = _cf()
    assert r.extract(None) == {c: "" for c in COLUMNS}
    res = r.find("cystic fibrosis", "ivacaftor", max_results=3)
    with pytest.raises(ValueError, match="not a ClinicalTrials.gov study record"):
        r.extract(res.record)
    hollow = EvidenceRecord(**{**res.studies[0].__dict__, "payload": {"protocolSection": {}, "hasResults": None}})
    assert r.extract(hollow) == {c: "" for c in COLUMNS}


# ---------------------------------------------------------------- live (opt-in)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ENGINE_LIVE_TESTS"), reason="set ENGINE_LIVE_TESTS=1 to hit clinicaltrials.gov")
def test_live_public_queries(tmp_path: Path):
    http = Http(HttpCache(tmp_path / "cache"), limiter=RateLimiter(default_per_second=2.0))
    r = TrialsRetriever(http, page_size=3)
    assert r.version().startswith("ClinicalTrials.gov API ") and " data 20" in r.version()
    res = r.find("cystic fibrosis", "ivacaftor", max_results=5)
    assert res.total_count >= 100 and len(res.studies) == 5 and res.record.payload["pages"] == 2
    assert all(x.record_id.startswith("nct:NCT") and x.source_version.startswith("ClinicalTrials.gov API ") for x in res.studies)
    assert all("ivacaftor" in r.extract(x)["interventions"].lower() or "ivacaftor" in x.payload["protocolSection"]["identificationModule"]["briefTitle"].lower()
               or "ivacaftor" in r.extract(x)["intervention_mesh"].lower() or "ivacaftor" in json.dumps(x.payload).lower() for x in res.studies)
    strive = r.study(STRIVE)
    assert strive is not None and r.extract(strive)["phases"] == "PHASE3" and r.extract(strive)["has_results"] == "Y"
    assert r.study("NCT99999999") is None
    assert r.study("NCT07048704").record_id == "nct:NCT07280598"
    assert r.search("zzqxnotacondition987654", max_results=3) == []
    # a rerun with the warm cache is byte-identical and needs nothing from outside
    again = TrialsRetriever(Http(HttpCache(tmp_path / "cache"), offline=True), page_size=3)
    assert again.find("cystic fibrosis", "ivacaftor", max_results=5).record.to_json() == res.record.to_json()
    assert again.study(STRIVE).to_json() == strive.to_json()
