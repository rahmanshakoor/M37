"""Literature retriever against recorded Europe PMC / NCBI responses (public papers only:
Frosst 1995 on MTHFR 677C>T, Kerem 1989 on CFTR F508del, Zelicha 2026 on MTHFR
rs1801133, and a CFTR/F508del search).

Every fixture in tests/fixtures/literature/ is a real response, stored with the exact
request that produced it; the stub Http below serves a fixture when the retriever
asks for exactly that request, and refuses anything else.
"""

import copy
import json
import os
import urllib.parse
from pathlib import Path

import pytest

from engine.retrieve.http import Http, HttpCache, HttpError, RateLimiter, Response
from engine.retrieve.literature import (COLUMNS, MAX_QUERY_CHARS, LiteratureError, LiteratureRetriever, SearchResult,
                                        _batch_query, _batches, abstract_text, gene_query)
from engine.retrieve.store import EvidenceStore

FIXTURES = Path(__file__).parent / "fixtures" / "literature"
ASKED = 'TITLE_ABS:"CFTR" AND TITLE_ABS:"F508del"'
SEARCH = f"({ASKED}) AND SRC:MED"
PAGE1_IDS = ["pmid:42616613", "pmid:41925277", "pmid:42680797", "pmid:41576103", "pmid:41869727"]


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


class StubHttp:
    """Serves recorded responses keyed by the exact request (method, url, params, body)."""

    def __init__(self, *fixtures: dict):
        self.limiter = RateLimiter(default_per_second=0)
        self.calls: list[tuple[str, str, dict | None]] = []
        self._by_key: dict[str, dict] = {}
        for fx in fixtures:
            self.add(fx)

    def add(self, fx: dict, *, request: dict | None = None) -> None:
        r = request or fx["request"]
        self._by_key[HttpCache.key(r["method"], r["url"], r["params"], r["body"])] = fx

    def request(self, method: str, url: str, *, params=None, json_body=None, **kw) -> Response:
        self.calls.append((method, url, params))
        fx = self._by_key.get(HttpCache.key(method, url, params, json_body))
        if fx is None:
            raise AssertionError(f"no fixture for {method} {url} {params}")
        if fx["status"] >= 400:
            raise HttpError(fx["status"], url, fx.get("text") or "")
        text = fx["text"] if fx.get("body") is None else json.dumps(fx["body"], ensure_ascii=False)
        return Response(fx["status"], text, fx["retrieved_at"], True, "stub", fx.get("headers", {}))

    def get(self, url: str, **kw) -> Response:
        return self.request("GET", url, **kw)

    def post(self, url: str, json_body, **kw) -> Response:
        return self.request("POST", url, json_body=json_body, **kw)


def served(fx: dict, *, body=None, status: int | None = None, request: dict | None = None) -> dict:
    """A copy of a recorded fixture with its body/status swapped — for failure shapes
    that must be served in reply to a request the retriever actually makes."""
    out = copy.deepcopy(fx)
    if body is not None:
        out["body"] = body
    if status is not None:
        out["status"] = status
    if request is not None:
        out["request"] = request
    return out


# ---- search


def test_search_one_record_per_paper_from_first_page():
    fx = fixture("search_page1")
    http = StubHttp(fx)
    lit = LiteratureRetriever(http, page_size=5)
    res = lit.search(ASKED, max_results=5)

    assert isinstance(res, SearchResult) and res.pmids == [i.split(":")[1] for i in PAGE1_IDS]
    assert [r.record_id for r in res.papers] == PAGE1_IDS
    assert http.calls == [("GET", "https://www.ebi.ac.uk/europepmc/webservices/rest/search", {
        "query": SEARCH, "format": "json", "resultType": "core", "pageSize": 5, "synonym": "false"})]
    r = res.papers[0]
    assert r.source == "pmid" and r.source_version == "Europe PMC REST 6.9" and lit.version() == "Europe PMC REST 6.9"
    assert r.query == {"pmid": "42616613"}  # the paper's identity, whichever route found it
    assert r.url == "https://europepmc.org/article/MED/42616613"
    assert r.retrieved_at == fx["retrieved_at"]
    assert r.payload is not None and r.payload == fx["body"]["resultList"]["result"][0]  # raw result object, untouched
    assert lit.extract(r) == {
        "pmid": "42616613",
        "title": "Functional and biochemical characterization of Alyftrek® components reveals high-potency rescue of F508del-CFTR.",
        "authors": r.payload["authorString"],
        "journal": "American journal of physiology. Lung cellular and molecular physiology",
        "year": "2026", "doi": "10.1152/ajplung.00103.2026", "pmcid": "",
        "has_abstract": "Y", "cited_by": "0", "open_access": "N",
    }


def test_search_record_holds_the_question_and_the_answer():
    fx = fixture("search_page1")
    lit = LiteratureRetriever(StubHttp(fx), page_size=5)
    s = lit.search(ASKED, max_results=5).record

    assert s.record_id.startswith("pmid-search:") and len(s.record_id) == len("pmid-search:") + 64
    assert s.source == "pmid-search" and s.source_version == "Europe PMC REST 6.9"
    assert s.query == {"query": ASKED, "peer_reviewed_only": True, "max_results": 5}  # exactly what was asked
    assert s.url == "https://europepmc.org/search?query=" + urllib.parse.quote(SEARCH)
    assert s.retrieved_at == fx["retrieved_at"]
    assert s.payload == {
        "sent": {"query": SEARCH, "format": "json", "resultType": "core", "pageSize": 5, "synonym": "false"},
        "hitCount": fx["body"]["hitCount"],
        "pmids": ["42616613", "41925277", "42680797", "41576103", "41869727"],
        "skipped_without_pmid": 0,
    }
    # the id is a function of what was sent, so the same search is the same record...
    again = LiteratureRetriever(StubHttp(fixture("search_page1")), page_size=5).search(ASKED, max_results=5).record
    assert again.to_json() == s.to_json()
    # ...and a different ask (more results wanted) is a different record, not an overwrite
    more = LiteratureRetriever(StubHttp(fixture("search_page1"), fixture("search_page2")), page_size=5)
    m = more.search(ASKED, max_results=7).record
    assert m.record_id != s.record_id and m.query["max_results"] == 7
    assert len(m.payload["pmids"]) == 7 and m.payload["pmids"][:5] == s.payload["pmids"]


def test_search_with_no_hits_is_a_record_with_no_papers():
    empty = fixture("fetch_99999999")["body"]
    assert empty["hitCount"] == 0
    req = dict(fixture("search_page1")["request"])
    lit = LiteratureRetriever(StubHttp(served(fixture("search_page1"), body=empty, request=req)), page_size=5)
    res = lit.search(ASKED, max_results=5)
    assert res.papers == [] and res.pmids == []
    assert res.record.payload["hitCount"] == 0 and res.record.source_version == "Europe PMC REST 6.9"


def test_search_pages_with_cursor_until_max_results():
    p1, p2 = fixture("search_page1"), fixture("search_page2")
    http = StubHttp(p1, p2)
    lit = LiteratureRetriever(http, page_size=5)
    res = lit.search(ASKED, max_results=8)

    assert len(res.papers) == 8 and len(http.calls) == 2
    assert http.calls[1][2]["cursorMark"] == p1["body"]["nextCursorMark"]
    assert "cursorMark" not in http.calls[0][2]
    assert [r.record_id for r in res.papers[5:]] == ["pmid:42350209", "pmid:41469310", "pmid:41998105"]
    assert res.papers[7].retrieved_at == p2["retrieved_at"]  # each record dates from its own page
    assert res.record.retrieved_at == p1["retrieved_at"] and len(res.pmids) == 8


def test_search_stops_on_last_page_without_next_cursor():
    p1 = fixture("search_page1")
    body = copy.deepcopy(p1["body"])
    del body["nextCursorMark"]  # Europe PMC omits it on the final page
    lit = LiteratureRetriever(StubHttp(served(p1, body=body)), page_size=5)
    assert len(lit.search(ASKED, max_results=8).papers) == 5


def test_search_does_not_repeat_a_paper_the_index_shifted_across_pages():
    p1, p2 = fixture("search_page1"), fixture("search_page2")
    body2 = copy.deepcopy(p2["body"])
    body2["resultList"]["result"][0] = copy.deepcopy(p1["body"]["resultList"]["result"][4])  # page 1's last paper again
    del body2["nextCursorMark"]
    res = LiteratureRetriever(StubHttp(p1, served(p2, body=body2)), page_size=5).search(ASKED, max_results=10)
    ids = [r.record_id for r in res.papers]
    assert len(ids) == len(set(ids)) == 9 and ids[:5] == PAGE1_IDS
    assert res.pmids == [i.split(":")[1] for i in ids]


def test_search_without_peer_review_filter_skips_results_lacking_a_pmid():
    p1 = fixture("search_page1")
    body = copy.deepcopy(p1["body"])
    del body["nextCursorMark"]
    del body["resultList"]["result"][1]["pmid"]  # what a preprint (SRC:PPR) looks like
    req = dict(p1["request"], params=dict(p1["request"]["params"], query=ASKED))
    lit = LiteratureRetriever(StubHttp(served(p1, body=body, request=req)), page_size=5)
    res = lit.search(ASKED, max_results=5, peer_reviewed_only=False)
    assert [r.record_id for r in res.papers] == ["pmid:42616613", "pmid:42680797", "pmid:41576103", "pmid:41869727"]
    assert res.record.query["peer_reviewed_only"] is False and res.record.payload["skipped_without_pmid"] == 1
    assert res.record.payload["sent"]["query"] == ASKED


def test_search_rejects_nonsense_max_results():
    with pytest.raises(ValueError):
        LiteratureRetriever(StubHttp()).search("x", max_results=0)


# ---- fetch / verify: absence vs failure


def test_fetch_existing_pmid_and_extract():
    fx = fixture("fetch_7647779")
    lit = LiteratureRetriever(StubHttp(fx))
    rec = lit.fetch("7647779")
    assert rec is not None
    assert rec.record_id == "pmid:7647779" and rec.query == {"pmid": "7647779"}
    assert rec.url == "https://europepmc.org/article/MED/7647779" and rec.retrieved_at == fx["retrieved_at"]
    assert lit.extract(rec) == {
        "pmid": "7647779",
        "title": "A candidate genetic risk factor for vascular disease: a common mutation in methylenetetrahydrofolate reductase.",
        "authors": "Frosst P, Blom HJ, Milos R, Goyette P, Sheppard CA, Matthews RG, Boers GJ, den Heijer M, Kluijtmans LA, van den Heuvel LP.",
        "journal": "Nature genetics", "year": "1995", "doi": "10.1038/ng0595-111", "pmcid": "",
        "has_abstract": "Y", "cited_by": "3899", "open_access": "N",
    }
    assert list(lit.extract(rec)) == list(COLUMNS)
    assert abstract_text(rec).startswith("Hyperhomocysteinaemia has been identified as a risk factor")


def test_fetch_missing_pmid_is_none_and_extract_is_blank():
    lit = LiteratureRetriever(StubHttp(fixture("fetch_99999999")))
    assert lit.fetch("99999999") is None
    assert lit.verify("99999999") is False
    assert lit.extract(None) == {c: "" for c in COLUMNS}


def test_verify_true_for_existing_pmid():
    assert LiteratureRetriever(StubHttp(fixture("fetch_7647779"))).verify("7647779") is True


def test_fetch_rejects_malformed_pmid_before_any_request():
    http = StubHttp()
    lit = LiteratureRetriever(http)
    for bad in ("notanumber", "PMC3148255", "", "12 34", "-1"):
        with pytest.raises(ValueError):
            lit.fetch(bad)
    assert http.calls == []


def test_fetch_drops_leading_zeros_because_ext_id_is_an_exact_match():
    http = StubHttp(fixture("fetch_7647779"))
    lit = LiteratureRetriever(http)
    rec = lit.fetch("0007647779")
    assert rec is not None and rec.record_id == "pmid:7647779" and rec.query == {"pmid": "7647779"}
    assert http.calls[0][2]["query"] == "EXT_ID:7647779 AND SRC:MED"
    assert lit.verify(" 07647779 ") is True


def test_http_200_with_errcode_is_a_failure_not_an_absence():
    fx = fixture("fetch_7647779")
    lit = LiteratureRetriever(StubHttp(served(fx, body=fixture("error_errcode")["body"])))
    with pytest.raises(LiteratureError, match="errCode=404"):
        lit.fetch("7647779")


def test_http_200_bare_version_body_is_a_failure_not_an_absence():
    fx = fixture("fetch_7647779")
    bare = fixture("error_bare_version")["body"]
    assert bare == {"version": "6.9"}  # what Europe PMC sends for a bad sort/cursor: no hitCount at all
    lit = LiteratureRetriever(StubHttp(served(fx, body=bare)))
    with pytest.raises(LiteratureError, match="no hitCount"):
        lit.verify("7647779")


def test_body_without_a_version_cannot_be_cited():
    fx = fixture("fetch_7647779")
    body = copy.deepcopy(fx["body"])
    del body["version"]
    lit = LiteratureRetriever(StubHttp(served(fx, body=body)))
    with pytest.raises(LiteratureError, match="without a version"):
        lit.fetch("7647779")
    assert lit.version() == ""  # nothing was observed, nothing is implied


def test_http_5xx_propagates_as_http_error():
    fx = fixture("fetch_7647779")
    lit = LiteratureRetriever(StubHttp(served(fx, status=503)))
    with pytest.raises(HttpError):
        lit.fetch("7647779")


def test_non_json_body_is_a_failure():
    fx = served(fixture("fetch_7647779"))
    fx["body"], fx["text"] = None, "<html>Service Temporarily Unavailable</html>"
    with pytest.raises(LiteratureError, match="non-JSON"):
        LiteratureRetriever(StubHttp(fx)).fetch("7647779")


# ---- fetch_many


def test_fetch_many_batches_with_or_query_and_reports_absence():
    fx = fixture("fetch_many_3")
    http = StubHttp(fx)
    lit = LiteratureRetriever(http)
    found = lit.fetch_many(["7647779", "2570460", "99999999", "07647779"])  # duplicate (zero-padded) collapses

    assert list(found) == ["7647779", "2570460", "99999999"]
    assert found["99999999"] is None
    assert http.calls[0][2]["query"] == "(EXT_ID:7647779 OR EXT_ID:2570460 OR EXT_ID:99999999) AND SRC:MED"
    assert http.calls[0][2]["pageSize"] == 3 and len(http.calls) == 1
    kerem = found["2570460"]
    assert kerem is not None and kerem.query == {"pmid": "2570460"}
    assert lit.extract(kerem)["title"] == "Identification of the cystic fibrosis gene: genetic analysis."
    assert lit.extract(kerem)["journal"] == "Science (New York, N.Y.)" and lit.extract(kerem)["year"] == "1989"
    assert lit.extract(found["7647779"])["doi"] == "10.1038/ng0595-111"


def test_batches_respect_query_length_and_page_size():
    ids = [str(30000000 + i) for i in range(400)]
    batches = list(_batches(ids))
    assert [p for b in batches for p in b] == ids
    assert len(batches) > 1
    assert all(len(_batch_query(b)) <= MAX_QUERY_CHARS for b in batches)
    assert all(len(b) <= 1000 for b in batches)
    assert list(_batches([])) == []


# ---- extract edge cases and helpers


def test_abstract_text_strips_section_and_inline_tags_and_entities():
    fx = fixture("search_page1")
    lit = LiteratureRetriever(StubHttp(fx), page_size=5)
    by_id = {r.record_id: r for r in lit.search(ASKED, max_results=5).papers}
    raw = by_id["pmid:42680797"].payload["abstractText"]
    assert raw.startswith("<h4>Impact statement</h4>")
    text = abstract_text(by_id["pmid:42680797"])
    assert text.startswith("Impact statement This commentary") and "<" not in text
    # inline markup vanishes without inserting spaces, as in PubMed's own text rendering
    assert "Cftr<sup>F508del/F508del</sup> mouse" in by_id["pmid:41925277"].payload["abstractText"]
    assert "CftrF508del/F508del mouse" in abstract_text(by_id["pmid:41925277"])
    assert "(<i>CFTR</i>) variant" in by_id["pmid:41576103"].payload["abstractText"]
    assert "(CFTR) variant" in abstract_text(by_id["pmid:41576103"])
    assert "Cl<sup>-</sup> channels" in by_id["pmid:41576103"].payload["abstractText"]
    assert "Cl- channels" in abstract_text(by_id["pmid:41576103"])


def test_abstract_text_keeps_a_raw_less_than_sign():
    # Europe PMC leaves 'p < 0.05' unescaped inside abstractText; a tag regex that is
    # not anchored to a tag name eats everything from there to the next closing tag.
    fx = fixture("fetch_42269400")
    rec = LiteratureRetriever(StubHttp(fx)).fetch("42269400")
    assert rec is not None
    raw = rec.payload["abstractText"]
    assert "< 0.05 between groups" in raw and "</h4>" in raw
    text = abstract_text(rec)
    assert "p = 0.003" in text and "TT-genotype" in text
    assert "< 0.05 between groups" in text and "</h4>" not in text and "<h4>" not in text
    assert len(text) >= len(raw) - raw.count("<") * len("</h4>")  # tags are the only thing removed


def test_extract_title_decodes_entity_encoded_tags():
    fx = fixture("search_page1")
    lit = LiteratureRetriever(StubHttp(fx), page_size=5)
    rec = lit.search(ASKED, max_results=5).papers[3]
    assert "&lt;i&gt;CFTR&lt;/i&gt;" in rec.payload["title"]  # how Europe PMC serves markup in titles
    title = lit.extract(rec)["title"]
    assert title == ("Chronic and acute modulator treatment restore wild-type-like activity and stability to the "
                     "primary cystic fibrosis-causing CFTR variant.")
    assert "<" not in title and "&" not in title


def test_extract_tolerates_absent_fields_and_bookshelf_records():
    fx = fixture("fetch_7647779")
    lit = LiteratureRetriever(StubHttp(fx))
    rec = lit.fetch("7647779")
    assert rec is not None
    # A GeneReviews-style Bookshelf record under SRC:MED: no journalInfo, no
    # authorString, pubYear is the book's inception, the chapter date is elsewhere.
    payload = {k: v for k, v in rec.payload.items() if k not in ("journalInfo", "authorString", "abstractText", "citedByCount", "doi")}
    payload.update({"pubYear": "1993", "firstPublicationDate": "2024-08-08",
                    "bookOrReportDetails": {"comprisingTitle": "GeneReviews®", "publisher": "University of Washington", "yearOfPublication": 1993}})
    book = type(rec)(**{**rec.__dict__, "payload": payload})
    cols = lit.extract(book)
    assert cols["journal"] == "GeneReviews®" and cols["year"] == "2024"
    assert cols["authors"] == "" and cols["has_abstract"] == "N" and cols["cited_by"] == "" and cols["doi"] == ""


def test_gene_query_uses_title_abs_fields():
    assert gene_query("CFTR") == 'TITLE_ABS:"CFTR"'
    assert gene_query("CFTR", "cystic fibrosis") == 'TITLE_ABS:"CFTR" AND TITLE_ABS:"cystic fibrosis"'
    assert gene_query(" MTHFR ", None) == 'TITLE_ABS:"MTHFR"'
    assert "GENE_SYMBOL" not in gene_query("TP53")
    for bad in ('TP"53', "", "  "):
        with pytest.raises(ValueError):
            gene_query(bad)
    with pytest.raises(ValueError):
        gene_query("TP53", 'Li-"Fraumeni')


# ---- NCBI cross-check


def test_verify_ncbi_true_and_false_from_esummary(monkeypatch):
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    http = StubHttp(fixture("esummary_7647779"), fixture("esummary_99999999"))
    lit = LiteratureRetriever(http)
    assert lit.verify_ncbi("7647779") is True
    assert lit.verify_ncbi("99999999") is False
    assert http.calls[0][2] == {"db": "pubmed", "id": "7647779", "retmode": "json", "tool": "engine"}
    assert http.limiter.per_second["eutils.ncbi.nlm.nih.gov"] == 3.0
    with pytest.raises(ValueError):
        lit.verify_ncbi("PMC1")


def test_verify_ncbi_honours_api_key_and_raises_the_rate(monkeypatch):
    monkeypatch.setenv("NCBI_API_KEY", "TESTKEY")
    fx = fixture("esummary_7647779")
    req = dict(fx["request"], params=dict(fx["request"]["params"], api_key="TESTKEY"))
    http = StubHttp(served(fx, request=req))
    lit = LiteratureRetriever(http)
    assert lit.verify_ncbi("7647779") is True
    assert http.calls[0][2]["api_key"] == "TESTKEY"
    assert http.limiter.per_second["eutils.ncbi.nlm.nih.gov"] == 10.0


def test_verify_ncbi_unexpected_body_is_a_failure(monkeypatch):
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    fx = fixture("esummary_7647779")
    lit = LiteratureRetriever(StubHttp(served(fx, body={"header": {"type": "esummary"}})))
    with pytest.raises(LiteratureError):
        lit.verify_ncbi("7647779")


def test_rate_etiquette_never_overrides_the_orchestrators_table():
    http = StubHttp()
    http.limiter.per_second["www.ebi.ac.uk"] = 1.0
    LiteratureRetriever(http)
    assert http.limiter.per_second["www.ebi.ac.uk"] == 1.0
    assert LiteratureRetriever(StubHttp()).http.limiter.per_second["www.ebi.ac.uk"] == 3.0


# ---- determinism


def test_same_fixture_gives_byte_identical_records(tmp_path: Path):
    a = LiteratureRetriever(StubHttp(fixture("fetch_7647779"))).fetch("7647779")
    b = LiteratureRetriever(StubHttp(fixture("fetch_7647779"))).fetch("7647779")
    assert a is not None and b is not None and a.to_json() == b.to_json() and a.sha256 == b.sha256
    store = EvidenceStore(tmp_path / "evidence")
    p = store.put(a)
    assert p.parent.name == "pmid" and store.get("pmid:7647779") == a
    # fetch and fetch_many agree on the record for the same paper (query and id alike)
    many = LiteratureRetriever(StubHttp(fixture("fetch_many_3"))).fetch_many(["7647779", "2570460", "99999999"])
    m = many["7647779"]
    assert m is not None and (m.record_id, m.query, m.url) == (a.record_id, a.query, a.url)


def test_a_paper_is_the_same_record_whether_searched_or_fetched(tmp_path: Path):
    """Search and fetch stamp the same identity on a paper, so storing both routes'
    records never makes one paper's provenance depend on call order; the search
    keeps its own record under its own source."""
    res = LiteratureRetriever(StubHttp(fixture("search_page1")), page_size=5).search(ASKED, max_results=5)
    fx = copy.deepcopy(fixture("fetch_7647779"))
    fx["body"]["resultList"]["result"] = [copy.deepcopy(res.papers[0].payload)]  # the same paper by EXT_ID, same time
    fx["body"]["request"]["queryString"] = fx["body"]["request"]["internalQuery"] = "EXT_ID:42616613 AND SRC:MED"
    fx["request"]["params"]["query"], fx["retrieved_at"] = "EXT_ID:42616613 AND SRC:MED", res.papers[0].retrieved_at
    fetched = LiteratureRetriever(StubHttp(fx)).fetch("42616613")
    assert fetched is not None and fetched.to_json() == res.papers[0].to_json()
    store = EvidenceStore(tmp_path / "evidence")
    for r in (res.record, *res.papers, fetched):
        store.put(r)
    assert store.count("pmid") == 5 and store.count("pmid-search") == 1
    assert store.get(res.record.record_id) == res.record and store.get("pmid:42616613") == fetched


# ---- live (opt-in)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ENGINE_LIVE_TESTS"), reason="set ENGINE_LIVE_TESTS=1 to hit Europe PMC")
def test_live_public_variant_search_and_fetch(tmp_path: Path):
    http = Http(HttpCache(tmp_path / "cache"), limiter=RateLimiter(default_per_second=2.0))
    lit = LiteratureRetriever(http)
    # MTHFR 1:11796321 G>A (rs1801133) — as a text token only, never as a coordinate
    res = lit.search(gene_query("MTHFR") + ' AND TITLE_ABS:"rs1801133"', max_results=3)
    assert 1 <= len(res.papers) <= 3 and res.record.payload["hitCount"] >= len(res.papers)
    for r in res.papers:
        cols = lit.extract(r)
        assert cols["pmid"] and cols["title"] and cols["year"].isdigit()
        assert r.source_version.startswith("Europe PMC REST ") and "<" not in cols["title"]
    rec = lit.fetch("7647779")
    assert rec is not None and lit.extract(rec)["journal"] == "Nature genetics"
    assert lit.fetch("99999999") is None
    # a rerun with the warm cache is byte-identical
    again = LiteratureRetriever(Http(HttpCache(tmp_path / "cache"), offline=True)).fetch("7647779")
    assert again is not None and again.to_json() == rec.to_json()
