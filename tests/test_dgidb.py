"""DGIdb retriever against recorded real responses (public genes only: CFTR, MTHFR,
BUB1B, TP53's gene-search pages, the alias CF and the nonsense symbol NOTAREALGENE123).

Every fixture in ``tests/fixtures/dgidb/`` is the engine's own HTTP cache entry from one
live request the retriever itself made — request body, status, headers, retrieval time,
body text — so the stub Http serves a fixture only when the retriever asks for exactly
that request and refuses anything else. Failure shapes that DGIdb delivers with HTTP 200,
and the one thing no public gene shows (a canonical spelling with lowercase letters),
are exercised by swapping a fixture's body; nothing synthetic is ever sent anywhere.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from engine.medicine.dgidb import (API_URL, COLUMNS, CONCEPT_QUERY, GENE_QUERY, MATCH_QUERY, PER_SECOND, RESOLVE_ALIASES,
                                   SEARCH_PAGE, SEARCH_QUERY, SERVICE_QUERY, SORT_RULE, SOURCES_QUERY, DgidbError,
                                   DgidbRetriever, GeneResult, is_unnamed)
from engine.retrieve.http import Http, HttpCache, HttpError, RateLimiter, Response
from engine.retrieve.store import EvidenceRecord, EvidenceStore

FIXTURES = Path(__file__).parent / "fixtures" / "dgidb"
VERSION = "DGIdb v.5.0.12 updatedAt=2026-07-14T18:00:15+00:00"
TEZACAFTOR = "dgidb:CFTR:rxcui:1999382"
EMPTY_NODES = {"data": {"genes": {"nodes": []}}}
"""What ``genes(names:)`` and ``genes(conceptIds:)`` answer for nothing — the recorded shape."""


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


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


def requested(fx: dict, variables: dict) -> dict:
    """A copy of a recorded fixture keyed to other GraphQL variables — for a request the
    retriever makes that was never recorded (a synthetic reply; never sent anywhere)."""
    out = copy.deepcopy(fx)
    out["request"]["body"] = {**out["request"]["body"], "variables": variables}
    return out


def search_vars(name: str, first: int = SEARCH_PAGE, after: str | None = None) -> dict:
    return {"name": name, "first": first, "after": after}


def search_page(nodes: list[dict]) -> dict:
    """A one-page gene-search body in the recorded shape."""
    return {"data": {"genes": {"totalCount": len(nodes), "nodes": nodes,
                               "pageInfo": {"hasNextPage": False, "endCursor": str(len(nodes)) if nodes else None}}}}


class StubHttp:
    """Serves recorded responses keyed by the exact request (method, url, params, body).
    ``from_cache`` marks the answers to ``cache_ok=True`` requests as cache hits — a
    warm cache, which the retriever must then confirm live; the default is a cold one,
    every answer live."""

    def __init__(self, *fixtures: dict, per_second: float = 0, offline: bool = False, from_cache: bool = False):
        self.limiter = RateLimiter(default_per_second=per_second)
        self.offline = offline
        """Like ``Http.offline``: when set, the retriever must not make its live version check."""
        self.from_cache = from_cache
        self.calls: list[dict] = []
        self._by_key: dict[str, dict] = {}
        for fx in fixtures:
            self.add(fx)

    def add(self, fx: dict) -> None:
        r = fx["request"]
        self._by_key[HttpCache.key(r["method"], r["url"], r["params"], r["body"])] = fx

    def request(self, method: str, url: str, *, params=None, json_body=None, **kw) -> Response:
        self.calls.append({"method": method, "url": url, "body": json_body, **kw})
        fx = self._by_key.get(HttpCache.key(method, url, params, json_body))
        if fx is None:
            raise AssertionError(f"no fixture for {method} {url} {json.dumps(json_body)[:120]}")
        if fx["status"] >= 400:
            raise HttpError(fx["status"], url, fx.get("text") or "")
        hit = self.from_cache and kw.get("cache_ok", True)
        return Response(fx["status"], fx["text"], fx["retrieved_at"], hit, "stub", fx.get("headers", {}))

    def get(self, url: str, **kw) -> Response:
        return self.request("GET", url, **kw)

    def post(self, url: str, json_body, **kw) -> Response:
        return self.request("POST", url, json_body=json_body, **kw)


def _queries(stub: StubHttp) -> list[tuple[str, dict]]:
    return [(c["body"]["query"], c["body"]["variables"]) for c in stub.calls]


def _asked(stub: StubHttp) -> list[tuple[str, dict]]:
    """The gene, search, concept and match queries in order, without the version observation every fetch starts with."""
    return [(q, v) for q, v in _queries(stub) if q != SERVICE_QUERY]


def _cache_ok(stub: StubHttp) -> list[tuple[str, bool]]:
    return [(c["body"]["query"], c.get("cache_ok", True)) for c in stub.calls]


SERVICE_COLD = [(SERVICE_QUERY, True)]
"""What a retriever over a cold cache asks for its version: one observation, live."""
SERVICE_WARM = [(SERVICE_QUERY, True), (SERVICE_QUERY, False)]
"""Over a warm cache, online: the cached observation, then a live one that must agree."""


def _cftr() -> DgidbRetriever:
    return DgidbRetriever(StubHttp(fixture("genes_CFTR"), fixture("service_info")))


def _node(name: str) -> dict:
    return json.loads(fixture(f"genes_{name}")["text"])["data"]["genes"]["nodes"][0]


# ---------------------------------------------------------------- a known gene with interactions


def test_fetch_cftr_one_record_per_interaction_plus_gene_record():
    stub = StubHttp(fixture("genes_CFTR"), fixture("service_info"))
    r = DgidbRetriever(stub)
    res = r.fetch("CFTR")
    assert isinstance(res, GeneResult) and res.known and res.name == "CFTR" and res.resolved_via is None
    assert _asked(stub) == [(GENE_QUERY, {"names": ["CFTR"]})]  # names: matched: no search, no alias lookup
    assert _cache_ok(stub) == SERVICE_COLD + [(GENE_QUERY, True)]  # version observed before the gene POST
    assert len(res.interactions) == 49 == res.gene.payload["n_interactions"]
    assert sum(1 for x in res.interactions if x.payload["drug"]["approved"]) == 19

    fx = fixture("genes_CFTR")
    gene = res.gene
    assert gene.record_id == "dgidb-gene:CFTR" and gene.source == "dgidb-gene"
    assert gene.source_version == VERSION
    assert gene.query == {"api": API_URL, "graphql": GENE_QUERY, "variables": {"names": ["CFTR"]}}
    assert gene.url == "https://dgidb.org/genes/CFTR"
    assert gene.retrieved_at == fx["retrieved_at"]  # from the Response, not the clock
    assert gene.payload["gene"] == {k: v for k, v in _node("CFTR").items() if k != "interactions"}
    assert gene.payload["gene"]["conceptId"] == "hgnc:1884"
    assert {c["name"] for c in gene.payload["gene"]["geneCategoriesWithSources"]} >= {"ION CHANNEL", "DRUGGABLE GENOME"}

    by_id = {x.record_id: x for x in res.interactions}
    tez = by_id[TEZACAFTOR]
    assert tez.source == "dgidb" and tez.source_version == VERSION and tez.query == gene.query
    assert tez.url == "https://dgidb.org/genes/CFTR" and tez.retrieved_at == fx["retrieved_at"]
    # verbatim: the interaction object exactly as the server sent it, no derived keys
    (raw,) = [i for i in _node("CFTR")["interactions"] if i["drug"]["conceptId"] == "rxcui:1999382"]
    assert tez.payload == raw
    assert tez.payload["gene"] == {"name": "CFTR", "conceptId": "hgnc:1884"}
    assert tez.payload["evidenceScore"] == 10 and tez.payload["interactionScore"] == 10.80244683536382
    assert {s["sourceDbName"] for s in tez.payload["sources"]} == {"PharmGKB", "GuideToPharmacology", "FDA", "TTD", "ChEMBL"}
    assert r.version() == VERSION


def test_interactions_is_the_records_only_and_symbol_case_does_not_matter():
    stub = StubHttp(fixture("genes_CFTR"), fixture("service_info"))
    r = DgidbRetriever(stub)
    recs = r.interactions("  cftr ")
    assert [x.record_id for x in recs] == [x.record_id for x in _cftr().fetch("CFTR").interactions]
    assert _asked(stub) == [(GENE_QUERY, {"names": ["CFTR"]})]  # upper-cased before it is sent
    assert all(x.source == "dgidb" for x in recs)  # the gene record is not in this list


def test_records_are_best_evidenced_first_then_deterministic():
    recs = _cftr().interactions("CFTR")
    ev = [x.payload["evidenceScore"] for x in recs]
    assert ev == sorted(ev, reverse=True) and ev[0] == 10 and recs[0].record_id == TEZACAFTOR
    # within a tie on evidence, DGIdb's own score decides; within that, name then id
    for a, b in zip(recs, recs[1:]):
        if a.payload["evidenceScore"] == b.payload["evidenceScore"]:
            assert a.payload["interactionScore"] >= b.payload["interactionScore"]


def test_one_drug_two_concept_ids_gives_two_records():
    by_name: dict[str, list[EvidenceRecord]] = {}
    for x in _cftr().interactions("CFTR"):
        by_name.setdefault(x.payload["drug"]["name"], []).append(x)
    elexa = by_name["ELEXACAFTOR"]
    assert sorted(x.record_id for x in elexa) == ["dgidb:CFTR:chembl:CHEMBL4298128", "dgidb:CFTR:rxcui:2256951"]
    assert len({x.payload["interactionScore"] for x in elexa}) == 2  # different claims, different scores
    assert {n for n, v in by_name.items() if len(v) > 1} == {"ELEXACAFTOR", "ICENTICAFTOR", "BAMOCAFTOR"}


def test_records_are_byte_identical_and_survive_the_store(tmp_path: Path):
    a = _cftr().fetch("CFTR")
    b = _cftr().fetch("cftr")
    assert a.gene.to_json() == b.gene.to_json()
    assert [x.to_json() for x in a.interactions] == [x.to_json() for x in b.interactions]
    store = EvidenceStore(tmp_path / "evidence")
    for rec in [a.gene, *a.interactions]:
        p = store.put(rec)
        assert p.parent.name == rec.source  # the source is what precedes the first colon
        assert store.get(rec.record_id) == rec
    assert store.count("dgidb") == 49 and store.count("dgidb-gene") == 1
    index = json.loads(store.write_index().read_text())
    assert index[TEZACAFTOR]["source_version"] == VERSION


# ---------------------------------------------------------------- absence: unknown symbol vs known gene without drugs


def test_known_gene_with_no_interactions_is_a_citable_record():
    stub = StubHttp(fixture("genes_BUB1B"), fixture("service_info"))
    res = DgidbRetriever(stub).fetch("BUB1B")
    assert res.known and res.name == "BUB1B" and res.interactions == []
    assert res.gene.record_id == "dgidb-gene:BUB1B" and res.gene.payload["n_interactions"] == 0
    assert {c["name"] for c in res.gene.payload["gene"]["geneCategoriesWithSources"]} >= {"KINASE", "DRUGGABLE GENOME"}
    assert _cache_ok(stub) == SERVICE_COLD + [(GENE_QUERY, True)]  # no search, no alias lookup for a known gene


def test_unknown_symbol_is_absence_after_the_search_and_one_alias_lookup():
    stub = StubHttp(fixture("genes_NOTAREALGENE123"), fixture("search_NOTAREALGENE123"), fixture("match_NOTAREALGENE123"),
                    fixture("service_info"))
    r = DgidbRetriever(stub)
    res = r.fetch("NOTAREALGENE123")
    assert not res.known and res.gene is None and res.interactions == [] and res.name is None
    assert res.asked == "NOTAREALGENE123" and res.resolved_via is None
    assert _asked(stub) == [(GENE_QUERY, {"names": ["NOTAREALGENE123"]}),
                            (SEARCH_QUERY, search_vars("NOTAREALGENE123")),
                            (MATCH_QUERY, {"terms": ["NOTAREALGENE123"]})]
    assert r.interactions("NOTAREALGENE123") == []


def test_alias_lookup_can_be_switched_off():
    stub = StubHttp(fixture("genes_NOTAREALGENE123"), fixture("search_NOTAREALGENE123"), fixture("service_info"))
    res = DgidbRetriever(stub).fetch("NOTAREALGENE123", resolve_aliases=False)
    assert not res.known
    assert _asked(stub) == [(GENE_QUERY, {"names": ["NOTAREALGENE123"]}), (SEARCH_QUERY, search_vars("NOTAREALGENE123"))]
    assert RESOLVE_ALIASES is True  # the default, and what the manifest records


def test_alias_is_resolved_and_the_records_are_those_of_the_canonical_gene():
    # CF: names: finds nothing, the prefix search lists 16 CF… genes but none named CF,
    # geneMatches says CF → CFTR, and CFTR is fetched by the ordinary route
    stub = StubHttp(fixture("genes_CF"), fixture("search_CF"), fixture("match_CF"), fixture("genes_CFTR"), fixture("service_info"))
    res = DgidbRetriever(stub).fetch("CF")
    assert res.known and res.asked == "CF" and res.name == "CFTR" and res.resolved_via == "alias"
    assert _asked(stub) == [(GENE_QUERY, {"names": ["CF"]}), (SEARCH_QUERY, search_vars("CF")),
                            (MATCH_QUERY, {"terms": ["CF"]}), (GENE_QUERY, {"names": ["CFTR"]})]
    assert res.gene.query["variables"] == {"names": ["CFTR"]}  # the request that produced the node
    direct = _cftr().fetch("CFTR")
    # the same record whichever route found it: identity and bytes are the canonical gene's
    assert res.gene.to_json() == direct.gene.to_json()
    assert [x.to_json() for x in res.interactions] == [x.to_json() for x in direct.interactions]


def test_resolve_alone():
    r = DgidbRetriever(StubHttp(fixture("match_CF"), fixture("match_NOTAREALGENE123")))
    assert r.resolve("cf") == "CFTR"
    assert r.resolve("NOTAREALGENE123") is None


def test_ambiguous_match_is_absence_unless_one_gene_is_named_the_term():
    def matches(kind: str, names: list[str]) -> dict:
        return {"data": {"geneMatches": {"directMatches": [], "ambiguousMatches": [], "noMatches": [],
                                         kind: [{"searchTerm": "NOTAREALGENE123", "matchType": kind[:-7].upper(),
                                                 "matches": [{"name": n, "conceptId": f"x:{n}"} for n in names]}]}}}
    # two genes the term is an alias of: absence, not a guess
    stub = StubHttp(fixture("genes_NOTAREALGENE123"), fixture("search_NOTAREALGENE123"),
                    served(fixture("match_NOTAREALGENE123"), body=matches("ambiguousMatches", ["CFTR", "MTHFR"])),
                    fixture("service_info"))
    res = DgidbRetriever(stub).fetch("NOTAREALGENE123")
    assert not res.known and [q for q, _ in _asked(stub)] == [GENE_QUERY, SEARCH_QUERY, MATCH_QUERY]
    # two DIRECT genes: the same
    stub = StubHttp(fixture("genes_NOTAREALGENE123"), fixture("search_NOTAREALGENE123"),
                    served(fixture("match_NOTAREALGENE123"), body=matches("directMatches", ["CFTR", "MTHFR"])),
                    fixture("service_info"))
    assert not DgidbRetriever(stub).fetch("NOTAREALGENE123").known
    # AMBIGUOUS between a gene the term is an alias of and a gene *named* the term
    # (observed live for MCR: NR3C2 and mCR): the one named so is the answer
    named = matches("ambiguousMatches", ["CFTR", "NotARealGene123"])
    r = DgidbRetriever(StubHttp(served(fixture("match_NOTAREALGENE123"), body=named)))
    assert r.resolve("NOTAREALGENE123") == "NotARealGene123"
    # ... but a DIRECT match is decided by the DIRECT list alone
    both = matches("ambiguousMatches", ["CFTR", "NotARealGene123"])
    both["data"]["geneMatches"]["directMatches"] = matches("directMatches", ["MTHFR"])["data"]["geneMatches"]["directMatches"]
    assert DgidbRetriever(StubHttp(served(fixture("match_NOTAREALGENE123"), body=both))).resolve("NOTAREALGENE123") == "MTHFR"


def test_match_that_names_a_gene_the_api_then_omits_raises():
    # geneMatches says CF → CFTR, but neither names:["CFTR"] nor the search returns it — a contradiction, not absence
    stub = StubHttp(fixture("genes_CF"), fixture("search_CF"), fixture("match_CF"),
                    served(fixture("genes_CFTR"), body=EMPTY_NODES), served(fixture("search_CFTR"), body=search_page([])),
                    fixture("service_info"))
    with pytest.raises(DgidbError, match="did not return"):
        DgidbRetriever(stub).fetch("CF")


def test_direct_match_on_the_very_symbol_the_search_did_not_find_raises():
    # names: and the prefix search both say no gene is named NOTAREALGENE123, then
    # geneMatches DIRECT-matches it to a gene of that very name: the same contradiction
    same = {"data": {"geneMatches": {"directMatches": [{"searchTerm": "NOTAREALGENE123", "matchType": "DIRECT",
                                                        "matches": [{"name": "NOTAREALGENE123", "conceptId": "x"}]}],
                                     "ambiguousMatches": [], "noMatches": []}}}
    stub = StubHttp(fixture("genes_NOTAREALGENE123"), fixture("search_NOTAREALGENE123"),
                    served(fixture("match_NOTAREALGENE123"), body=same), fixture("service_info"))
    with pytest.raises(DgidbError, match="did not return"):
        DgidbRetriever(stub).fetch("NOTAREALGENE123")
    assert [q for q, _ in _asked(stub)] == [GENE_QUERY, SEARCH_QUERY, MATCH_QUERY, GENE_QUERY, SEARCH_QUERY]


# ---------------------------------------------------------------- the search + concept-id route


def test_search_picks_the_gene_named_exactly_out_of_the_prefix_matches_and_walks_every_page():
    stub = StubHttp(fixture("search_TP53"), fixture("search_CF"), fixture("search_NOTAREALGENE123"),
                    fixture("search_TP53_first3_afternone"), fixture("search_TP53_first3_afterMw"), fixture("search_TP53_first3_afterNg"))
    r = DgidbRetriever(stub)
    # genes(name: "TP53") is a prefix search: seven TP53… genes, TP53 sixth — the exact name is the one
    assert r._search("TP53") == {"name": "TP53", "conceptId": "hgnc:11998"}
    assert r._search("CF") is None  # 16 genes start with CF; none is named CF
    assert r._search("NOTAREALGENE123") is None
    # three pages of three: found on the second, the third still read (a second gene of that name would raise)
    assert r._search("TP53", page_size=3) == {"name": "TP53", "conceptId": "hgnc:11998"}
    assert _queries(stub)[-3:] == [(SEARCH_QUERY, search_vars("TP53", 3)), (SEARCH_QUERY, search_vars("TP53", 3, "Mw")),
                                   (SEARCH_QUERY, search_vars("TP53", 3, "Ng"))]


def test_search_and_concept_id_route_yields_the_same_node_as_names():
    # names:["CFTR"] answering nothing (as it does for a gene spelt with lowercase letters)
    # sends the retriever through the search and genes(conceptIds:) — recorded live for CFTR
    stub = StubHttp(served(fixture("genes_CFTR"), body=EMPTY_NODES), fixture("search_CFTR"), fixture("concept_CFTR"),
                    fixture("service_info"))
    res = DgidbRetriever(stub).fetch("CFTR")
    assert res.known and res.name == "CFTR" and res.resolved_via is None
    assert _asked(stub) == [(GENE_QUERY, {"names": ["CFTR"]}), (SEARCH_QUERY, search_vars("CFTR")),
                            (CONCEPT_QUERY, {"conceptIds": ["hgnc:1884"]})]
    direct = _cftr().fetch("CFTR")
    assert res.gene.record_id == direct.gene.record_id and res.gene.url == direct.gene.url
    assert res.gene.payload == direct.gene.payload  # the node is the same whichever request found it
    assert [x.record_id for x in res.interactions] == [x.record_id for x in direct.interactions]
    assert [x.payload for x in res.interactions] == [x.payload for x in direct.interactions]
    assert res.gene.query == {"api": API_URL, "graphql": CONCEPT_QUERY, "variables": {"conceptIds": ["hgnc:1884"]}}
    assert res.gene.retrieved_at == fixture("concept_CFTR")["retrieved_at"]
    assert all(x.query == res.gene.query for x in res.interactions)
    # the query replays to the very cache entry that answered
    fx = fixture("concept_CFTR")["request"]
    replay = {"query": res.gene.query["graphql"], "variables": res.gene.query["variables"]}
    assert HttpCache.key("POST", API_URL, None, replay) == HttpCache.key(fx["method"], fx["url"], fx["params"], fx["body"])


def _lowercase_node() -> dict:
    """A gene whose canonical spelling has lowercase letters, as DGIdb has a dozen of
    (C9orf72, C8orf44-SGK3, mCR …). Stand-in: the public BUB1B node re-spelled ``Bub1b``
    — a synthetic edit of a recorded public record, never sent anywhere."""
    return dict(_node("BUB1B"), name="Bub1b")


def _lowercase_stub(**extra) -> StubHttp:
    """The shapes observed live for such a gene: ``names:`` upper-cases the input and
    compares it exactly, so it answers ``nodes: []`` for any spelling; the search answers
    the canonical spelling; ``conceptIds:`` returns the node."""
    return StubHttp(served(fixture("genes_BUB1B"), body=EMPTY_NODES),
                    requested(served(fixture("search_CFTR"), body=search_page([{"name": "Bub1b", "conceptId": "hgnc:1149"}])),
                              search_vars("BUB1B")),
                    requested(served(fixture("concept_CFTR"), body={"data": {"genes": {"nodes": [_lowercase_node()]}}}),
                              {"conceptIds": ["hgnc:1149"]}),
                    fixture("service_info"), **extra)


def test_lowercase_canonical_spelling_is_found_by_the_search_not_a_false_absence():
    stub = _lowercase_stub()
    res = DgidbRetriever(stub).fetch("bub1b")
    assert res.known and res.asked == "BUB1B" and res.name == "Bub1b" and res.resolved_via is None  # the same symbol
    assert res.gene.record_id == "dgidb-gene:Bub1b" and res.gene.url == "https://dgidb.org/genes/Bub1b"
    assert res.gene.query["variables"] == {"conceptIds": ["hgnc:1149"]} and res.gene.query["graphql"] == CONCEPT_QUERY
    assert _asked(stub) == [(GENE_QUERY, {"names": ["BUB1B"]}), (SEARCH_QUERY, search_vars("BUB1B")),
                            (CONCEPT_QUERY, {"conceptIds": ["hgnc:1149"]})]  # no alias lookup
    # the same gene reached through an alias: geneMatches names the canonical spelling, the same route fetches it
    match = {"data": {"geneMatches": {"directMatches": [{"searchTerm": "NOTAREALGENE123", "matchType": "DIRECT",
                                                         "matches": [{"name": "Bub1b", "conceptId": "hgnc:1149"}]}],
                                      "ambiguousMatches": [], "noMatches": []}}}
    stub = _lowercase_stub()
    for fx in (fixture("genes_NOTAREALGENE123"), fixture("search_NOTAREALGENE123"), served(fixture("match_NOTAREALGENE123"), body=match)):
        stub.add(fx)
    via = DgidbRetriever(stub).fetch("NOTAREALGENE123")
    assert via.known and via.name == "Bub1b" and via.resolved_via == "alias"
    assert via.gene.to_json() == res.gene.to_json()
    assert [q for q, _ in _asked(stub)] == [GENE_QUERY, SEARCH_QUERY, MATCH_QUERY, GENE_QUERY, SEARCH_QUERY, CONCEPT_QUERY]


def test_search_and_concept_shapes_that_raise():
    two = search_page([{"name": "BUB1B", "conceptId": "hgnc:1149"}, {"name": "Bub1b", "conceptId": "x:1"}])
    stub = _lowercase_stub()
    stub.add(requested(served(fixture("search_CFTR"), body=two), search_vars("BUB1B")))
    with pytest.raises(DgidbError, match="2 genes are named BUB1B"):
        DgidbRetriever(stub).fetch("BUB1B")
    # the search named a concept id that genes(conceptIds:) then does not return
    stub = _lowercase_stub()
    stub.add(requested(served(fixture("concept_CFTR"), body=EMPTY_NODES), {"conceptIds": ["hgnc:1149"]}))
    with pytest.raises(DgidbError, match="did not return"):
        DgidbRetriever(stub).fetch("BUB1B")
    # ... or returns a node under another concept id
    stub = _lowercase_stub()
    stub.add(requested(served(fixture("concept_CFTR"), body={"data": {"genes": {"nodes": [dict(_lowercase_node(), conceptId="hgnc:0")]}}}),
                       {"conceptIds": ["hgnc:1149"]}))
    with pytest.raises(DgidbError, match="concept id"):
        DgidbRetriever(stub).fetch("BUB1B")
    # a search page without the connection, a node without a concept id, a cursor that does not move
    for body, message in ((({"data": {"genes": None}}), "no connection"),
                          (search_page([{"name": "Bub1b", "conceptId": None}]), "conceptId"),
                          ({"data": {"genes": {"totalCount": 9, "nodes": [], "pageInfo": {"hasNextPage": True, "endCursor": None}}}}, "endCursor")):
        stub = _lowercase_stub()
        stub.add(requested(served(fixture("search_CFTR"), body=body), search_vars("BUB1B")))
        with pytest.raises(DgidbError, match=message):
            DgidbRetriever(stub).fetch("BUB1B")


def test_resolve_compares_the_echoed_search_term_case_insensitively():
    # geneMatches echoes searchTerm upper-cased (observed live: cftr → 'CFTR'); a lowercase
    # echo, should the server ever send one, must not turn a DIRECT match into absence
    lower = {"data": {"geneMatches": {"directMatches": [{"searchTerm": "cf", "matchType": "DIRECT",
                                                         "matches": [{"name": "CFTR", "conceptId": "hgnc:1884"}]}],
                                      "ambiguousMatches": [], "noMatches": []}}}
    assert DgidbRetriever(StubHttp(served(fixture("match_CF"), body=lower))).resolve("CF") == "CFTR"


def test_malformed_symbols_are_refused_before_any_request_without_echoing_them():
    stub = StubHttp()
    r = DgidbRetriever(stub)
    for bad in ("", "   ", "CF TR", 'A"B', "TP\n53", "{}", "-A", "7:117559590:ATCT:A"):
        with pytest.raises(ValueError, match="not a gene symbol") as e:
            r.fetch(bad)
        assert bad.strip() not in str(e.value) or not bad.strip()  # whatever was passed by mistake stays out of the message
        with pytest.raises(ValueError):
            r.resolve(bad)
    assert stub.calls == []


# ---------------------------------------------------------------- failure is never absence


def _cftr_served(**kw) -> DgidbRetriever:
    return DgidbRetriever(StubHttp(served(fixture("genes_CFTR"), **kw), fixture("service_info")))


def test_graphql_errors_with_http_200_raise():
    r = _cftr_served(text=fixture("error_undefined_field")["text"])  # a real errors[] body, HTTP 200
    with pytest.raises(DgidbError, match="bogusField"):
        r.fetch("CFTR")


def test_body_without_data_raises():
    with pytest.raises(DgidbError, match="No query string"):  # what DGIdb answers when the JSON body is not seen
        _cftr_served(body={"errors": [{"message": "No query string was present"}]}).fetch("CFTR")
    with pytest.raises(DgidbError, match="no data"):
        _cftr_served(body={"data": None}).fetch("CFTR")
    with pytest.raises(DgidbError, match="nodes"):
        _cftr_served(body={"data": {"genes": None}}).fetch("CFTR")


def test_non_json_and_non_200_bodies_raise():
    with pytest.raises(DgidbError, match="non-JSON"):
        _cftr_served(text="<html><title>502 Bad Gateway</title></html>").fetch("CFTR")
    with pytest.raises(DgidbError, match="unexpected JSON"):
        _cftr_served(text="[]").fetch("CFTR")
    with pytest.raises(HttpError):  # the stub raises like Http does for a 4xx/5xx it will not retry
        _cftr_served(status=500, text="").fetch("CFTR")


def test_node_that_does_not_echo_the_symbol_raises():
    node = dict(_node("CFTR"), name="MTHFR")
    with pytest.raises(DgidbError, match="echo"):
        _cftr_served(body={"data": {"genes": {"nodes": [node]}}}).fetch("CFTR")
    with pytest.raises(DgidbError, match="2 gene nodes"):
        _cftr_served(body={"data": {"genes": {"nodes": [_node("CFTR"), _node("CFTR")]}}}).fetch("CFTR")


def test_interaction_shapes_that_raise():
    node = _node("CFTR")
    foreign = copy.deepcopy(node)
    foreign["interactions"][0]["gene"] = {"name": "MTHFR", "conceptId": "hgnc:7436"}
    with pytest.raises(DgidbError, match="another gene"):
        _cftr_served(body={"data": {"genes": {"nodes": [foreign]}}}).fetch("CFTR")
    no_drug = copy.deepcopy(node)
    no_drug["interactions"][0]["drug"] = None
    with pytest.raises(DgidbError, match="no drug object"):
        _cftr_served(body={"data": {"genes": {"nodes": [no_drug]}}}).fetch("CFTR")
    no_id = copy.deepcopy(node)
    no_id["interactions"][0]["drug"] = {"name": "", "conceptId": None, "approved": True}
    with pytest.raises(DgidbError, match="neither conceptId nor name"):
        _cftr_served(body={"data": {"genes": {"nodes": [no_id]}}}).fetch("CFTR")
    twice = copy.deepcopy(node)
    twice["interactions"].append(copy.deepcopy(twice["interactions"][0]))
    with pytest.raises(DgidbError, match="two interactions for dgidb:CFTR:"):
        _cftr_served(body={"data": {"genes": {"nodes": [twice]}}}).fetch("CFTR")
    not_a_list = dict(node, interactions=None)
    with pytest.raises(DgidbError, match="interactions list"):
        _cftr_served(body={"data": {"genes": {"nodes": [not_a_list]}}}).fetch("CFTR")


def test_drug_without_concept_id_falls_back_to_its_name():
    node = copy.deepcopy(_node("CFTR"))
    node["interactions"][0]["drug"] = {"name": "NAMEONLY", "conceptId": None, "approved": None}
    res = _cftr_served(body={"data": {"genes": {"nodes": [node]}}}).fetch("CFTR")
    assert "dgidb:CFTR:NAMEONLY" in {x.record_id for x in res.interactions}


def test_cached_body_that_fails_validation_is_refetched_live_once():
    class Poisoned(StubHttp):
        def request(self, method, url, *, params=None, json_body=None, **kw):
            if json_body["query"] == GENE_QUERY and kw.get("cache_ok", True):
                self.calls.append({"method": method, "url": url, "body": json_body, **kw})
                return Response(200, '{"errors":[{"message":"Something went wrong"}]}', "2026-09-01T00:00:00+00:00", True, "poisoned", {})
            return super().request(method, url, params=params, json_body=json_body, **kw)

    stub = Poisoned(fixture("genes_CFTR"), fixture("service_info"))
    res = DgidbRetriever(stub).fetch("CFTR")
    assert res.known and len(res.interactions) == 49
    assert res.gene.retrieved_at == fixture("genes_CFTR")["retrieved_at"]  # the live answer's time
    gene_calls = [c for c in stub.calls if c["body"]["query"] == GENE_QUERY]
    assert [c.get("cache_ok", True) for c in gene_calls] == [True, False]
    assert gene_calls[0]["body"] == gene_calls[1]["body"]


def test_live_failure_is_not_retried():
    stub = StubHttp(served(fixture("genes_CFTR"), body={"errors": [{"message": "Something went wrong"}]}),
                    fixture("service_info"))
    with pytest.raises(DgidbError, match="Something went wrong"):
        DgidbRetriever(stub).fetch("CFTR")
    assert _asked(stub) == [(GENE_QUERY, {"names": ["CFTR"]})]


def test_offline_a_cached_failure_is_reported_as_such_not_refetched():
    stub = StubHttp(served(fixture("genes_CFTR"), body={"errors": [{"message": "Something went wrong"}]}),
                    fixture("service_info"), from_cache=True, offline=True)
    with pytest.raises(DgidbError, match="offline forbids fetching it again.*Something went wrong"):
        DgidbRetriever(stub).fetch("CFTR")
    assert _cache_ok(stub) == [(SERVICE_QUERY, True), (GENE_QUERY, True)]  # nothing live was attempted


def _urlopen_serving(*fixtures: dict):
    """A stand-in for ``urllib.request.urlopen`` that answers a real :class:`Http` from
    recorded public fixtures, by exact request — so the on-disk cache behaviour can be
    exercised end to end without the network. Anything unrecorded is an error."""
    by_key = {HttpCache.key(f["request"]["method"], f["request"]["url"], f["request"]["params"], f["request"]["body"]): f
              for f in fixtures}

    class _Reply:
        def __init__(self, fx: dict):
            self.status, self.headers, self._text = fx["status"], fx.get("headers", {}), fx["text"]

        def read(self) -> bytes:
            return self._text.encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(req, timeout=None):
        fx = by_key.get(HttpCache.key(req.get_method(), req.full_url, None, json.loads(req.data.decode())))
        if fx is None:
            raise AssertionError(f"no fixture for {req.get_method()} {req.full_url}")
        urlopen.calls.append(req.full_url)
        return _Reply(fx)

    urlopen.calls = []
    return urlopen


def test_poisoned_cache_entry_is_healed_so_reruns_replay_it_and_work_offline(tmp_path: Path, monkeypatch):
    # The engine's own cache holds, for the CFTR gene request, a failure DGIdb delivered
    # with HTTP 200 (cached as definitive by Http). Without a write-back every rerun would
    # pay a live request, retrieved_at would move, and an offline rerun would fail.
    monkeypatch.setattr("urllib.request.urlopen", _urlopen_serving(fixture("genes_CFTR"), fixture("service_info")))
    root = tmp_path / "cache"
    gene_body = fixture("genes_CFTR")["request"]["body"]
    key = HttpCache.key("POST", API_URL, None, gene_body)
    seed = HttpCache(root)
    poisoned = Response(200, '{"errors":[{"message":"Something went wrong"}]}', "2026-09-01T00:00:00+00:00", False, key, {})
    seed.put(key, poisoned, {"method": "POST", "url": API_URL, "params": None, "body": gene_body})
    quiet = {"dgidb.org": 0.0}  # no pacing in a test; the retriever's setdefault leaves this alone

    first = Http(HttpCache(root), limiter=RateLimiter(quiet))
    a = DgidbRetriever(first).fetch("CFTR")
    assert a.known and len(a.interactions) == 49
    assert first.live_requests == 2  # serviceInfo (cold, so no agreement check to make) and the gene refetch
    healed = HttpCache(root).get(key)
    assert healed is not None and healed.text == fixture("genes_CFTR")["text"]
    assert healed.retrieved_at == a.gene.retrieved_at
    entry = json.loads(next(root.glob(f"{key[:2]}/{key}.json")).read_text())
    assert entry["request"] == {"method": "POST", "url": API_URL, "params": None, "body": gene_body}  # the shape Http writes

    second = Http(HttpCache(root), limiter=RateLimiter(quiet))
    b = DgidbRetriever(second).fetch("CFTR")
    assert second.live_requests == 1  # only the live agreement check; the gene came from the healed entry
    assert b.gene.to_json() == a.gene.to_json()
    assert [x.to_json() for x in b.interactions] == [x.to_json() for x in a.interactions]

    offline = Http(HttpCache(root), offline=True)
    c = DgidbRetriever(offline).fetch("CFTR")
    assert offline.live_requests == 0 and c.gene.to_json() == a.gene.to_json()


def test_a_live_refetch_that_also_fails_heals_nothing(tmp_path: Path, monkeypatch):
    poisoned = served(fixture("genes_CFTR"), body={"errors": [{"message": "Something went wrong"}]})
    monkeypatch.setattr("urllib.request.urlopen", _urlopen_serving(poisoned, fixture("service_info")))
    root = tmp_path / "cache"
    key = HttpCache.key("POST", API_URL, None, poisoned["request"]["body"])
    HttpCache(root).put(key, Response(200, poisoned["text"], "2026-09-01T00:00:00+00:00", False, key, {}), poisoned["request"])
    with pytest.raises(DgidbError, match="Something went wrong"):
        DgidbRetriever(Http(HttpCache(root), limiter=RateLimiter({"dgidb.org": 0.0}))).fetch("CFTR")
    assert HttpCache(root).get(key).retrieved_at == "2026-09-01T00:00:00+00:00"  # untouched


# ---------------------------------------------------------------- version and sources


def test_version_is_the_service_version_and_data_timestamp():
    stub = StubHttp(fixture("service_info"))
    r = DgidbRetriever(stub)
    assert r.version() == VERSION == r.version()
    assert _cache_ok(stub) == SERVICE_COLD  # observed once, live: nothing cached to confirm; then remembered
    info = r.service_info()
    assert info["dataVersion"] == "Dec-2023" and "Dec-2023" not in r.version()  # stale; never cited alone
    warm = StubHttp(fixture("service_info"), from_cache=True)
    assert DgidbRetriever(warm).version() == VERSION
    assert _cache_ok(warm) == SERVICE_WARM  # a cached observation is confirmed live, once
    with pytest.raises(DgidbError, match="version"):
        DgidbRetriever(StubHttp(served(fixture("service_info"), body={"data": {"serviceInfo": {"version": "v.5", "updatedAt": None}}}))).version()


class MovedOn(StubHttp):
    """The cache holds v.5.0.12 / 2026-07-14, but DGIdb has refreshed since: a live
    serviceInfo now reports a later ``updatedAt`` (same service version, as a data
    refresh looks) — the shape that would mislabel fresh rows if it went unnoticed."""

    LIVE = {"data": {"serviceInfo": {"name": "DGIdb", "version": "v.5.0.12", "dataVersion": "Dec-2023",
                                     "updatedAt": "2026-10-01T00:00:00+00:00"}}}

    def __init__(self, *fixtures: dict, **kw):
        super().__init__(*fixtures, from_cache=True, **kw)

    def request(self, method, url, *, params=None, json_body=None, **kw):
        if json_body["query"] == SERVICE_QUERY and not kw.get("cache_ok", True):
            self.calls.append({"method": method, "url": url, "body": json_body, **kw})
            return Response(200, json.dumps(self.LIVE), "2026-10-02T00:00:00+00:00", False, "live", {})
        return super().request(method, url, params=params, json_body=json_body, **kw)


def test_cached_version_must_agree_with_live_when_online_and_reproduces_offline():
    offline = StubHttp(fixture("service_info"), fixture("genes_CFTR"), offline=True, from_cache=True)
    r = DgidbRetriever(offline)
    assert r.version() == VERSION
    assert _cache_ok(offline) == [(SERVICE_QUERY, True)]  # offline: the cache alone, nothing live
    assert len(r.interactions("CFTR")) == 49

    with pytest.raises(DgidbError, match="dgidb.org entries"):  # not the whole shared cache
        DgidbRetriever(MovedOn(fixture("service_info"))).version()
    stale = MovedOn(fixture("service_info"), fixture("genes_CFTR"))
    with pytest.raises(DgidbError, match="2026-10-01"):  # a gene fetch is refused too: no record is stamped wrongly
        DgidbRetriever(stale).fetch("CFTR")
    # ... and refused before the gene POST, so the stale cache gains no response of the newer release
    assert _queries(stale) == [(SERVICE_QUERY, {}), (SERVICE_QUERY, {})]
    # the sources page is under the same rule: per-source versions of the earlier refresh are not replayed online
    with pytest.raises(DgidbError, match="2026-10-01"):
        DgidbRetriever(MovedOn(fixture("service_info"), fixture("sources_first100_afternone"))).sources()
    # the same stale cache is fine offline: it reproduces the run that filled it
    assert DgidbRetriever(MovedOn(fixture("service_info"), offline=True)).version() == VERSION


def test_version_does_not_depend_on_what_was_found():
    stub = StubHttp(fixture("genes_BUB1B"), fixture("genes_CFTR"), fixture("service_info"))
    r = DgidbRetriever(stub)
    empty = r.fetch("BUB1B")
    full = r.fetch("CFTR")
    assert empty.gene.source_version == full.gene.source_version == full.interactions[0].source_version == VERSION


def test_sources_single_page_and_paginated_agree():
    stub = StubHttp(fixture("sources_first100_afternone"), fixture("service_info"))
    one = DgidbRetriever(stub).sources()
    assert _queries(stub)[0] == (SERVICE_QUERY, {})  # the release is observed before the page is asked for
    assert len(one) == 45 and [s["sourceDbName"] for s in one] == sorted(s["sourceDbName"] for s in one)
    v = {s["sourceDbName"]: s["sourceDbVersion"] for s in one}
    assert v["ChEMBL"] == "37" and v["PharmGKB"] == "20260622" and v["CIViC"] == "08-June-2026" and v["HGNC"] == "20260619"
    lic = {s["sourceDbName"]: s["license"] for s in one}
    assert "CC BY-NC-SA" in lic["DTC"] and "CC0" in lic["CIViC"]
    stub = StubHttp(fixture("sources_first30_afternone"), fixture("sources_first30_afterMzA"), fixture("service_info"))
    paged = DgidbRetriever(stub).sources(page_size=30)
    assert paged == one
    assert _asked(stub) == [(SOURCES_QUERY, {"first": 30, "after": None}), (SOURCES_QUERY, {"first": 30, "after": "MzA"})]


def test_sources_guards():
    stub = StubHttp()
    with pytest.raises(ValueError):
        DgidbRetriever(stub).sources(page_size=0)
    assert stub.calls == []
    stuck = {"data": {"sources": {"totalCount": 45, "pageInfo": {"hasNextPage": True, "endCursor": None}, "nodes": []}}}
    with pytest.raises(DgidbError, match="endCursor"):
        DgidbRetriever(StubHttp(served(fixture("sources_first100_afternone"), body=stuck), fixture("service_info"))).sources()
    with pytest.raises(DgidbError, match="connection"):
        DgidbRetriever(StubHttp(served(fixture("sources_first100_afternone"), body={"data": {"sources": None}}),
                                fixture("service_info"))).sources()


def test_params_name_everything_that_shapes_the_output_and_are_json():
    stub = StubHttp()
    r = DgidbRetriever(stub, url="https://example.invalid/graphql")
    p = r.params
    assert p == {"api": "https://example.invalid/graphql", "gene_query": GENE_QUERY, "search_query": SEARCH_QUERY,
                 "concept_query": CONCEPT_QUERY, "match_query": MATCH_QUERY, "service_query": SERVICE_QUERY,
                 "sources_query": SOURCES_QUERY, "search_page": SEARCH_PAGE, "resolve_aliases_default": True,
                 "per_second": PER_SECOND, "sort": SORT_RULE}
    assert json.loads(json.dumps(p, sort_keys=True)) == p
    assert "evidenceScore desc" in SORT_RULE and "interactionScore desc" in SORT_RULE
    # the rate recorded is the one the shared limiter applies to the host, not the module's default
    stub.limiter.per_second["example.invalid"] = 0.5
    assert r.params["per_second"] == 0.5
    assert DgidbRetriever(StubHttp(), url="https://example.invalid/graphql").params["per_second"] == PER_SECOND


def test_rate_is_pinned_politely_but_never_overrides_the_orchestrator():
    stub = StubHttp()
    DgidbRetriever(stub)
    assert stub.limiter.per_second["dgidb.org"] == 3.0
    stub.limiter.per_second["dgidb.org"] = 1.0
    r = DgidbRetriever(stub)
    assert stub.limiter.per_second["dgidb.org"] == 1.0 and r.params["per_second"] == 1.0


# ---------------------------------------------------------------- extract


def test_extract_tezacaftor():
    r = _cftr()
    (tez,) = [x for x in r.interactions("CFTR") if x.record_id == TEZACAFTOR]
    cols = r.extract(tez)
    assert tuple(cols) == COLUMNS == r.columns
    assert cols == {
        "dgidb_gene": "CFTR",
        "dgidb_drug": "TEZACAFTOR",
        "dgidb_drug_concept_id": "rxcui:1999382",
        "dgidb_unnamed_compound": "N",
        "dgidb_approved": "Y",
        "dgidb_anti_neoplastic": "N",
        "dgidb_immunotherapy": "N",
        "dgidb_interaction_types": "activator;positive modulator",
        "dgidb_directionality": "ACTIVATING",
        "dgidb_evidence_score": "10",
        "dgidb_interaction_score": "10.80244683536382",
        "dgidb_sources": "ChEMBL:37;FDA:08-June-2026;GuideToPharmacology:2026.2;PharmGKB:20260622;TTD:2020.06.01",
        "dgidb_pmids": "28930490;29099333;29099344;30334692;40785054",
        "dgidb_interaction_id": "6f500fd3-ca6f-44c0-8a65-1af2bf436baf",
    }
    assert float(cols["dgidb_interaction_score"]) == tez.payload["interactionScore"]  # repr precision, no rounding


def test_extract_empty_types_and_publications_are_blank_not_missing():
    r = DgidbRetriever(StubHttp(fixture("genes_MTHFR"), fixture("service_info")))
    recs = r.interactions("MTHFR")
    assert len(recs) == 34 and recs[0].payload["drug"]["name"] == "PEMETREXED DISODIUM"
    rows = [r.extract(x) for x in recs]
    assert all(row["dgidb_interaction_types"] == "" and row["dgidb_directionality"] == "" for row in rows)
    assert sum(1 for row in rows if row["dgidb_pmids"]) == 33
    assert all(row["dgidb_gene"] == "MTHFR" and row["dgidb_sources"] for row in rows)


def test_extract_none_and_nullable_flags_and_unnamed_compound():
    r = _cftr()
    assert r.extract(None) == {c: "" for c in COLUMNS}
    (tez,) = [x for x in r.interactions("CFTR") if x.record_id == TEZACAFTOR]
    # an unnamed ChEMBL compound as DGIdb shows it (synthetic edit of a public record; never sent)
    payload = copy.deepcopy(tez.payload)
    payload["drug"] = {"name": "CHEMBL:CHEMBL260560", "conceptId": "chembl:CHEMBL260560",
                       "approved": None, "antiNeoplastic": None, "immunotherapy": True}
    payload["interactionTypes"] = [{"type": "inhibitor", "directionality": "INHIBITORY", "definition": None},
                                   {"type": "modulator", "directionality": None, "definition": None}]
    payload["publications"] = []
    rec = EvidenceRecord(**{**tez.__dict__, "record_id": "dgidb:CFTR:chembl:CHEMBL260560", "payload": payload})
    cols = r.extract(rec)
    assert is_unnamed(payload["drug"]) and not is_unnamed(tez.payload["drug"])
    assert (cols["dgidb_unnamed_compound"], cols["dgidb_approved"], cols["dgidb_anti_neoplastic"], cols["dgidb_immunotherapy"]) == ("Y", "", "", "Y")
    assert cols["dgidb_interaction_types"] == "inhibitor;modulator" and cols["dgidb_directionality"] == "INHIBITORY"
    assert cols["dgidb_pmids"] == ""
    with pytest.raises(ValueError, match="not a dgidb interaction record"):
        r.extract(_cftr().fetch("CFTR").gene)


# ---------------------------------------------------------------- live (opt-in)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ENGINE_LIVE_TESTS"), reason="set ENGINE_LIVE_TESTS=1 to hit dgidb.org")
def test_live_public_genes(tmp_path: Path):
    http = Http(HttpCache(tmp_path / "cache"), limiter=RateLimiter(default_per_second=2.0))
    r = DgidbRetriever(http)
    assert r.version().startswith("DGIdb v") and "updatedAt=" in r.version()
    cftr = r.fetch("CFTR")
    assert cftr.known and cftr.name == "CFTR" and cftr.gene.payload["gene"]["conceptId"] == "hgnc:1884"
    names = {x.payload["drug"]["name"] for x in cftr.interactions}
    assert {"TEZACAFTOR", "IVACAFTOR", "LUMACAFTOR"} <= names
    assert any(r.extract(x)["dgidb_approved"] == "Y" for x in cftr.interactions)
    assert r.fetch("BUB1B").known
    assert not r.fetch("NOTAREALGENE123").known
    p53 = r.fetch("P53")  # names: nothing, the prefix search nothing, geneMatches → TP53
    assert p53.name == "TP53" and p53.resolved_via == "alias" and p53.gene.query["variables"] == {"names": ["TP53"]}
    assert r._search("TP53") == {"name": "TP53", "conceptId": "hgnc:11998"}  # the exact name among the TP53… prefix matches
    assert {s["sourceDbName"] for s in r.sources()} >= {"ChEMBL", "PharmGKB", "FDA", "CIViC"}
    # a rerun with the warm cache is byte-identical and needs nothing from outside
    again = DgidbRetriever(Http(HttpCache(tmp_path / "cache"), offline=True)).fetch("CFTR")
    assert again.gene.to_json() == cftr.gene.to_json()
    assert [x.to_json() for x in again.interactions] == [x.to_json() for x in cftr.interactions]
