"""VEP retriever tests.

Recorded responses under ``tests/fixtures/vep/`` hold only the three public textbook
variants (MTHFR rs1801133, CFTR p.Phe508del, TP53 p.Arg175His). Structural edge cases
the public set cannot show (intergenic input, absent keys, splice bumping, transcript
choice, an overlapping lncRNA, dropped lines, API failures) use small hand-built
payloads with made-up coordinates; nothing here touches the network unless
``ENGINE_LIVE_TESTS`` is set.
"""

import json
import logging
import os
from pathlib import Path

import pytest

from engine.retrieve.http import Http, HttpCache, HttpError, RateLimiter, Response
from engine.retrieve.store import EvidenceRecord, key_str
from engine.retrieve.vep import (
    SEVERITY_ORDER, VepError, VepRetriever, impact_any_coding, most_severe, select_transcript, vcf_line,
    well_formed,
)

FIX = Path(__file__).parent / "fixtures" / "vep"
BASE = "https://rest.ensembl.org"
MTHFR = ("1", 11796321, "G", "A")
CFTR = ("7", 117559590, "ATCT", "A")
TP53 = ("17", 7675088, "C", "T")
PUBLIC = [MTHFR, CFTR, TP53]
SYMBOLIC = ("1", 1000, "A", "<DEL>")  # made up: a line VEP cannot parse
VERSION = "Ensembl VEP 116 (rest.ensembl.org, assembly GRCh38)"


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text())


def _resp(fx: dict, body, key: str) -> Response:
    return Response(fx["status"], json.dumps(body), fx["retrieved_at"], False, key, fx["headers"])


class StubHttp:
    """Serves the recorded fixtures. The batch response is sliced to the ids actually
    requested, so smaller batch sizes see exactly what the API would have returned.

    ``live_release`` is what the server reports when asked with ``cache_ok=False``
    (a release boundary crossed since the cache was filled); ``drop`` are ids the
    server returns nothing for; ``fail_post`` raises on the n-th POST (1-based) and
    ``bad_shape`` answers a POST with an object instead of a list."""

    def __init__(self, *, blank_ids: bool = False, drop: set | None = None, offline: bool = False,
                 live_release: int | None = None, fail_post: int = 0, bad_shape: bool = False):
        self.calls: list[tuple] = []
        self.batch = _load("region_post_public3.json")
        self.info = {
            "/info/software": _load("info_software.json"),
            "/info/data": _load("info_data.json"),
            "/info/assembly/homo_sapiens": _load("info_assembly.json"),
        }
        self.blank_ids = blank_ids  # simulate a server that does not echo column 3
        self.drop = drop or set()
        self.offline = offline
        self.live_release = live_release
        self.fail_post = fail_post
        self.bad_shape = bad_shape
        self.posts = 0

    def request(self, method, url, *, params=None, json_body=None, headers=None,
                cache_404=False, cache_ok=True, timeout=None) -> Response:
        self.calls.append((method, url, params, json_body, timeout, cache_ok))
        key = HttpCache.key(method, url, params, json_body)
        path = url.replace(BASE, "")
        if method == "GET":
            fx = self.info[path]
            body = fx["response"]
            if path == "/info/software" and not cache_ok and self.live_release is not None:
                body = {**body, "release": self.live_release}
            return _resp(fx, body, key)
        assert method == "POST" and path == "/vep/homo_sapiens/region"
        assert params == self.batch["request"]["params"]
        assert len(json_body["variants"]) <= 200
        self.posts += 1
        if self.posts == self.fail_post:
            raise HttpError(503, url, "Service Unavailable")
        if self.bad_shape:
            return _resp(self.batch, {"error": "no"}, key)
        by_id = {it["id"]: it for it in self.batch["response"]}
        items = []
        for line in json_body["variants"]:
            vid = line.split()[2]
            if vid in by_id and vid not in self.drop:
                item = dict(by_id[vid])
                if self.blank_ids:
                    item["id"] = "."
                items.append(item)
        return _resp(self.batch, items, key)

    def get(self, url, **kw):
        return self.request("GET", url, **kw)

    def post(self, url, json_body, **kw):
        return self.request("POST", url, json_body=json_body, **kw)


def _gets(http: StubHttp) -> list[tuple[str, bool]]:
    return [(c[1].replace(BASE, ""), c[5]) for c in http.calls if c[0] == "GET"]


# ---------------------------------------------------------------- pure helpers

def test_most_severe_follows_ensembl_order():
    assert most_severe(["intron_variant", "splice_region_variant", "missense_variant"]) == "missense_variant"
    assert most_severe(["synonymous_variant", "splice_donor_variant"]) == "splice_donor_variant"
    assert most_severe([]) == ""
    assert most_severe(["made_up_term", "intergenic_variant"]) == "intergenic_variant"  # unknown sorts last
    assert SEVERITY_ORDER[0] == "transcript_ablation" and SEVERITY_ORDER[-1] == "sequence_variant"
    assert len(SEVERITY_ORDER) == len(set(SEVERITY_ORDER)) == 41


def test_vcf_line_and_well_formed():
    assert vcf_line(CFTR) == "7 117559590 7:117559590:ATCT:A ATCT A . . ."
    assert all(well_formed(k) for k in PUBLIC)
    assert well_formed(("MT", 100, "a", "g"))            # case does not matter to VEP
    assert not well_formed(SYMBOLIC)
    assert not well_formed(("1", 1000, "A", "N"))
    assert not well_formed(("chr1", 11796321, "G", "A"))  # not canonical: ingest would never send it


# ---------------------------------------------------------------- version

def test_version_is_observed_live_and_replayed_from_cache():
    http = StubHttp()
    r = VepRetriever(http)
    assert r.version() == VERSION
    assert r.version() == r.version()
    paths = ["/info/software", "/info/data", "/info/assembly/homo_sapiens"]
    # cached copy first (an offline rerun needs it), then the same three live
    assert _gets(http) == [(p, True) for p in paths] + [(p, False) for p in paths]

    offline = StubHttp(offline=True)
    assert VepRetriever(offline).version() == VERSION
    assert _gets(offline) == [(p, True) for p in paths]


def test_version_refuses_a_cache_that_spans_a_release():
    http = StubHttp(live_release=117)
    r = VepRetriever(http)
    with pytest.raises(VepError, match="Ensembl VEP 116.*Ensembl VEP 117.*clear the cache"):
        r.version()
    with pytest.raises(VepError):
        r.retrieve(PUBLIC)                        # checked before any POST goes out
    assert not [c for c in http.calls if c[0] == "POST"]
    # the same stale cache is fine offline: it reproduces the run that filled it
    assert VepRetriever(StubHttp(live_release=117, offline=True)).version() == VERSION


# ---------------------------------------------------------------- retrieve

def test_retrieve_builds_one_record_per_key():
    http = StubHttp()
    r = VepRetriever(http)
    out = r.retrieve(PUBLIC)
    assert list(out) == PUBLIC
    posts = [c for c in http.calls if c[0] == "POST"]
    assert len(posts) == 1
    _, url, params, body, timeout, _ = posts[0]
    assert url == f"{BASE}/vep/homo_sapiens/region"
    assert params == {"canonical": 1, "mane": 1, "hgvs": 1, "numbers": 1, "SpliceAI": 1, "CADD": 1}
    assert body == {"variants": [vcf_line(k) for k in PUBLIC]}
    assert timeout == 300.0

    fx = _load("region_post_public3.json")
    request_key = HttpCache.key("POST", url, params, body)
    for k in PUBLIC:
        recs = out[k]
        assert len(recs) == 1
        rec = recs[0]
        assert rec.record_id == f"vep:{key_str(k)}"
        assert rec.source == "vep"
        assert rec.source_version == VERSION
        assert rec.query == {"method": "POST", "url": url, "params": params, "variant": vcf_line(k),
                             "request_key": request_key}
        assert rec.retrieved_at == fx["retrieved_at"]
        assert rec.payload["id"] == key_str(k) and rec.payload["input"] == vcf_line(k)
        assert rec.payload["assembly_name"] == "GRCh38"
        # payload is the full raw item: every number extract() quotes is in there
        assert "transcript_consequences" in rec.payload and "colocated_variants" in rec.payload
    assert out[MTHFR][0].url == (f"{BASE}/vep/homo_sapiens/region/1:11796321-11796321/A"
                                 "?content-type=application%2Fjson&canonical=1&mane=1&hgvs=1&numbers=1&SpliceAI=1&CADD=1")
    assert out[CFTR][0].url.startswith(f"{BASE}/vep/homo_sapiens/region/7:117559590-117559593/A?")
    assert r.dropped == 0


def _without_request_key(rec: EvidenceRecord) -> str:
    q = {k: v for k, v in rec.query.items() if k != "request_key"}
    return EvidenceRecord(**{**rec.__dict__, "query": q}).to_json()


def test_batches_preserve_order_and_bytes():
    one = VepRetriever(StubHttp()).retrieve(PUBLIC)
    assert one[MTHFR][0].to_json() == VepRetriever(StubHttp()).retrieve(PUBLIC)[MTHFR][0].to_json()
    http = StubHttp()
    small = VepRetriever(http, batch_size=2, workers=4).retrieve(PUBLIC + [MTHFR])  # duplicate key
    assert len([c for c in http.calls if c[0] == "POST"]) == 2
    assert list(small) == PUBLIC
    for k in PUBLIC:
        # only the batch identity (request_key) may differ with a different batch size
        assert _without_request_key(small[k][0]) == _without_request_key(one[k][0])
        assert small[k][0].query["request_key"] != one[k][0].query["request_key"]
    serial = VepRetriever(StubHttp(), batch_size=1, workers=1).retrieve(PUBLIC)
    assert {k: v[0].payload for k, v in serial.items()} == {k: v[0].payload for k, v in one.items()}
    with pytest.raises(ValueError):
        VepRetriever(StubHttp(), batch_size=201)


def test_dropped_well_formed_line_is_an_error(caplog):
    r = VepRetriever(StubHttp(drop={key_str(TP53)}))
    with pytest.raises(VepError, match=r"1 of 3 well-formed variants came back without a result") as e:
        r.retrieve(PUBLIC)
    assert "7675088" not in str(e.value) and "C:T" not in str(e.value)
    assert r.retrieve([]) == {}


def test_dropped_symbolic_line_is_absence_and_counted(caplog):
    http = StubHttp()  # the fixture has no item for the symbolic key, as the server would not
    r = VepRetriever(http)
    with caplog.at_level(logging.WARNING, logger="engine.retrieve.vep"):
        out = r.retrieve([MTHFR, SYMBOLIC])
    assert out[SYMBOLIC] == [] and len(out[MTHFR]) == 1
    assert r.extract(out[SYMBOLIC]) == {c: "" for c in r.columns}
    assert r.dropped == 1
    warnings = [m for m in caplog.messages if "returned no result" in m]
    assert len(warnings) == 1 and "1 of 2" in warnings[0]
    assert "<DEL>" not in warnings[0] and "1000" not in warnings[0]  # counts, not alleles

    r2 = VepRetriever(StubHttp(blank_ids=True))
    out2 = r2.retrieve(PUBLIC)
    assert all(len(out2[k]) == 1 for k in PUBLIC)
    assert out2[CFTR][0].payload["input"] == vcf_line(CFTR)  # input line is the fallback match


def test_api_failure_raises_out_of_the_pool():
    # second of three concurrent single-variant batches fails: the error propagates, nothing is []
    with pytest.raises(HttpError) as e:
        VepRetriever(StubHttp(fail_post=2), batch_size=1, workers=3).retrieve(PUBLIC)
    assert e.value.status == 503
    with pytest.raises(HttpError):
        VepRetriever(StubHttp(fail_post=1)).retrieve(PUBLIC)
    with pytest.raises(VepError, match="returned dict, not a list of results, for a batch of 3") as e2:
        VepRetriever(StubHttp(bad_shape=True)).retrieve(PUBLIC)
    for k in PUBLIC:
        assert key_str(k) not in str(e2.value) and str(k[1]) not in str(e2.value)


# ---------------------------------------------------------------- extract (recorded)

def _cols(k):
    r = VepRetriever(StubHttp())
    return r.extract(r.retrieve([k])[k])


def test_extract_mthfr_snv_on_reverse_strand_gene():
    c = _cols(MTHFR)
    assert c["gene_symbol"] == "MTHFR" and c["gene_id"] == "ENSG00000177000"
    assert c["transcript_id"] == "ENST00000376590" and c["mane"] == "NM_005957.5"
    assert c["consequence"] == "missense_variant" and c["impact"] == "MODERATE"
    assert c["hgvsc"] == "ENST00000376590.9:c.665C>T" and c["hgvsp"] == "ENSP00000365775.3:p.Ala222Val"
    assert c["exon"] == "5/12" and c["intron"] == "" and c["biotype"] == "protein_coding"
    assert c["protein_position"] == "222" and c["amino_acids"] == "A/V"
    assert c["sift_pred"] == "deleterious" and c["sift_score"] == "0"
    assert c["polyphen_pred"] == "probably_damaging" and c["polyphen_score"] == "0.943"
    assert c["spliceai_ds_max"] == "0" and c["spliceai_detail"] == "AG:0|AL:0|DG:0|DL:0"
    assert c["cadd_phred"] == "28.3"
    assert c["most_severe_consequence"] == "missense_variant" and c["impact_any_coding"] == "MODERATE"
    assert c["rsid"] == "rs1801133"
    assert c["vep_gnomade_af"] == "0.3227" and c["vep_gnomadg_af"] == "0.2752"
    # allele-specific terms (clin_sig_allele), not the record's merged clin_sig
    assert c["vep_clin_sig"] == "benign,conflicting_classifications_of_pathogenicity,drug_response,likely_benign,uncertain_significance"
    assert c["vep_clinvar_ids"] == "VCV000003520"
    assert set(c) == set(VepRetriever.columns)


def test_extract_cftr_deletion_uses_dash_allele_and_ignores_the_antisense_lncrna():
    c = _cols(CFTR)
    assert c["gene_symbol"] == "CFTR" and c["transcript_id"] == "ENST00000003084" and c["mane"] == "NM_000492.4"
    assert c["consequence"] == "inframe_deletion" and c["impact"] == "MODERATE"
    assert c["hgvsc"] == "ENST00000003084.11:c.1521_1523del" and c["hgvsp"] == "ENSP00000003084.6:p.Phe508del"
    assert c["exon"] == "11/27" and c["protein_position"] == "507-508" and c["amino_acids"] == "IF/I"
    assert c["sift_pred"] == "" and c["sift_score"] == "" and c["polyphen_pred"] == "" and c["polyphen_score"] == ""
    assert c["cadd_phred"] == "17.55" and c["spliceai_ds_max"] == "0" and c["spliceai_detail"] == "AG:0|AL:0|DG:0|DL:0"
    assert c["rsid"] == "rs113993960"
    # frequencies keyed by VEP's '-' allele, clin_sig_allele with an empty prefix
    assert c["vep_gnomade_af"] == "0.01235" and c["vep_gnomadg_af"] == "0.007884"
    assert c["vep_clin_sig"] == "drug_response,likely_pathogenic,pathogenic,risk_factor"
    assert c["vep_clinvar_ids"] == "VCV000007105,VCV000634837"


def test_extract_tp53_is_allele_specific_and_tolerates_missing_scores():
    c = _cols(TP53)
    assert c["gene_symbol"] == "TP53" and c["transcript_id"] == "ENST00000269305" and c["mane"] == "NM_000546.6"
    assert c["hgvsp"] == "ENSP00000269305.4:p.Arg175His" and c["exon"] == "5/11" and c["protein_position"] == "175"
    assert c["sift_pred"] == "tolerated" and c["sift_score"] == "0.08"
    assert c["polyphen_pred"] == "" and c["polyphen_score"] == ""   # absent on the MANE transcript
    assert c["spliceai_ds_max"] == "0.01" and c["spliceai_detail"] == "AG:0|AL:0|DG:0.01|DL:0"
    assert c["cadd_phred"] == "25.9" and c["impact_any_coding"] == "MODERATE"
    assert c["rsid"] == "rs28934578"
    assert c["vep_gnomade_af"] == "4.104e-06" and c["vep_gnomadg_af"] == "6.57e-06"
    # only the T-allele terms; 'pathogenic/likely_pathogenic' belongs to the A allele
    assert c["vep_clin_sig"] == "likely_pathogenic,pathogenic"
    assert c["vep_clinvar_ids"] == "VCV000012374,VCV000182963"


# ---------------------------------------------------------------- extract (synthetic shapes)

def _rec(payload) -> list[EvidenceRecord]:
    return [EvidenceRecord(record_id="vep:1:1000:A:G", source="vep", source_version="v", query={},
                           url="u", retrieved_at="t", payload=payload)]


def _tc(**kw):
    base = {"transcript_id": "ENST1", "gene_id": "ENSG1", "gene_symbol": "G1", "biotype": "protein_coding",
            "consequence_terms": ["intron_variant"], "impact": "MODIFIER", "variant_allele": "G"}
    base.update(kw)
    return base


def test_extract_intergenic_and_absent_keys():
    r = VepRetriever(StubHttp())
    payload = {"id": "1:1000:A:G", "input": "1 1000 1:1000:A:G A G . . .", "allele_string": "A/G",
               "most_severe_consequence": "intergenic_variant",
               "intergenic_consequences": [{"impact": "MODIFIER", "variant_allele": "G",
                                            "consequence_terms": ["intergenic_variant"]}]}
    c = r.extract(_rec(payload))
    assert c["consequence"] == "intergenic_variant" and c["impact"] == "MODIFIER"
    assert c["most_severe_consequence"] == "intergenic_variant" and c["impact_any_coding"] == ""
    assert c["gene_symbol"] == "" and c["rsid"] == "" and c["vep_gnomade_af"] == "" and c["vep_clin_sig"] == ""
    assert c["spliceai_ds_max"] == "" and c["spliceai_detail"] == ""
    # rs entry without a 'frequencies' key, HGMD entry without anything
    payload2 = {"allele_string": "A/G", "most_severe_consequence": "intron_variant",
                "transcript_consequences": [_tc(mane_select="NM_1", canonical=1)],
                "colocated_variants": [{"id": "CM000001", "allele_string": "HGMD_MUTATION"},
                                       {"id": "rs1", "allele_string": "A/G/C", "clin_sig_allele": "C:benign;G:pathogenic",
                                        "clin_sig": ["benign", "pathogenic", "other"],
                                        "var_synonyms": {"ClinVar": ["RCV000000001", "VCV000000002"]}}]}
    c2 = r.extract(_rec(payload2))
    assert c2["rsid"] == "rs1" and c2["vep_gnomade_af"] == "" and c2["vep_gnomadg_af"] == ""
    assert c2["vep_clin_sig"] == "pathogenic" and c2["vep_clinvar_ids"] == "VCV000000002"
    assert c2["mane"] == "NM_1" and c2["exon"] == "" and c2["cadd_phred"] == ""


def test_clin_sig_falls_back_to_the_merged_list():
    r = VepRetriever(StubHttp())
    base = {"allele_string": "A/G", "transcript_consequences": [_tc()]}
    # no clin_sig_allele key at all → the merged list, rather than a false empty
    c = r.extract(_rec({**base, "colocated_variants": [{"id": "rs1", "clin_sig": ["likely_pathogenic", "pathogenic"]}]}))
    assert c["vep_clin_sig"] == "likely_pathogenic,pathogenic"
    # clin_sig_allele present but silent on this allele → the merged list (over-inclusive by design)
    c = r.extract(_rec({**base, "colocated_variants": [
        {"id": "rs1", "clin_sig": ["benign"], "clin_sig_allele": "C:benign"}]}))
    assert c["vep_clin_sig"] == "benign"
    # allele-specific terms win whenever VEP gives any for the allele
    c = r.extract(_rec({**base, "colocated_variants": [
        {"id": "rs1", "clin_sig": ["benign", "pathogenic"], "clin_sig_allele": "C:pathogenic;G:benign"}]}))
    assert c["vep_clin_sig"] == "benign"
    # neither → ''
    c = r.extract(_rec({**base, "colocated_variants": [{"id": "rs1"}]}))
    assert c["vep_clin_sig"] == "" and c["rsid"] == "rs1"


def test_impact_any_coding_and_splice_bump():
    assert impact_any_coding([]) == ""
    assert impact_any_coding([_tc(biotype="lncRNA", impact="HIGH", consequence_terms=["splice_donor_variant"])]) == ""
    assert impact_any_coding([_tc(impact="LOW", consequence_terms=["splice_region_variant", "synonymous_variant"]),
                              _tc(impact="MODIFIER")]) == "SPLICE"
    assert impact_any_coding([_tc(impact="MODERATE", consequence_terms=["missense_variant", "splice_region_variant"])]) == "MODERATE"
    assert impact_any_coding([_tc(impact="HIGH", consequence_terms=["splice_donor_variant"]), _tc(impact="LOW")]) == "HIGH"
    assert impact_any_coding([_tc(impact="LOW"), _tc(impact="MODIFIER")]) == "LOW"


def test_transcript_choice_prefers_mane_of_most_severe_coding_gene():
    mane_a = _tc(transcript_id="ENST_A_MANE", gene_id="A", mane_select="NM_A", canonical=1)
    other_a = _tc(transcript_id="ENST_A_2", gene_id="A", consequence_terms=["splice_donor_variant"], impact="HIGH")
    mane_b = _tc(transcript_id="ENST_B_MANE", gene_id="B", mane_select="NM_B", canonical=1,
                 consequence_terms=["missense_variant"], impact="MODERATE")
    # gene A carries the most severe consequence → A's MANE is reported even though it is intronic there
    assert select_transcript([mane_b, other_a, mane_a])["transcript_id"] == "ENST_A_MANE"
    # no MANE in the top gene → its canonical protein-coding transcript
    canon_a = _tc(transcript_id="ENST_A_CANON", gene_id="A", canonical=1)
    assert select_transcript([mane_b, other_a, canon_a])["transcript_id"] == "ENST_A_CANON"
    # neither MANE nor canonical in the top gene → its most severe protein-coding transcript
    assert select_transcript([mane_b, other_a])["transcript_id"] == "ENST_A_2"
    # a lncRNA outranking every coding consequence still loses to the coding gene's MANE
    nc = _tc(transcript_id="ENST_NC", gene_id="C", gene_symbol="LINC1", biotype="lncRNA",
             consequence_terms=["splice_acceptor_variant"], impact="HIGH")
    assert select_transcript([mane_b, nc])["transcript_id"] == "ENST_B_MANE"
    # nothing coding at all → the most severe of any, MANE first on ties
    assert select_transcript([nc, _tc(transcript_id="ENST_NC2", gene_id="D", biotype="lncRNA")])["transcript_id"] == "ENST_NC"
    assert select_transcript([_tc(transcript_id="ENST_Z", biotype="lncRNA"),
                              _tc(transcript_id="ENST_Y", biotype="lncRNA", mane_select="NM_Y")])["transcript_id"] == "ENST_Y"
    assert select_transcript([]) is None

    r = VepRetriever(StubHttp())
    c = r.extract(_rec({"allele_string": "A/G", "most_severe_consequence": "splice_donor_variant",
                        "transcript_consequences": [mane_b, other_a, mane_a]}))
    assert c["transcript_id"] == "ENST_A_MANE" and c["consequence"] == "intron_variant"
    assert c["most_severe_consequence"] == "splice_donor_variant" and c["impact_any_coding"] == "HIGH"
    c2 = r.extract(_rec({"allele_string": "A/G", "transcript_consequences": [
        _tc(mane_select="NM_1", consequence_terms=["splice_region_variant", "intron_variant"], impact="LOW",
            mane=["MANE_Plus_Clinical"], spliceai={"DS_AG": 0, "DS_AL": 0.12, "DS_DG": 0.03, "DS_DL": 0.4})]}))
    assert c2["consequence"] == "splice_region_variant,intron_variant"
    assert c2["spliceai_ds_max"] == "0.4" and c2["spliceai_detail"] == "AG:0|AL:0.12|DG:0.03|DL:0.4"
    assert c2["impact_any_coding"] == "SPLICE" and c2["mane"] == "NM_1 (MANE_Plus_Clinical)"


def test_deep_intronic_variant_under_an_antisense_lncrna_exon():
    """The live CFTR shape: every coding transcript carries the gene's SpliceAI object,
    the antisense lncRNA carries none, and its exon term outranks intron_variant."""
    sa = {"DS_AG": 0, "DS_AL": 0.02, "DS_DG": 0, "DS_DL": 0.55}
    cftr_mane = _tc(transcript_id="ENST_CFTR_MANE", gene_id="ENSG_CFTR", gene_symbol="CFTR", mane_select="NM_000492.4",
                    canonical=1, intron="12/26", spliceai=sa)
    cftr_nmd = _tc(transcript_id="ENST_CFTR_NMD", gene_id="ENSG_CFTR", gene_symbol="CFTR", biotype="nonsense_mediated_decay",
                   consequence_terms=["intron_variant", "NMD_transcript_variant"], spliceai=sa)
    as1 = _tc(transcript_id="ENST_AS1", gene_id="ENSG_AS1", gene_symbol="CFTR-AS1", biotype="lncRNA",
              consequence_terms=["non_coding_transcript_exon_variant"])
    payload = {"allele_string": "A/G", "most_severe_consequence": "non_coding_transcript_exon_variant",
               "transcript_consequences": [as1, cftr_nmd, cftr_mane]}
    c = VepRetriever(StubHttp()).extract(_rec(payload))
    assert c["gene_symbol"] == "CFTR" and c["transcript_id"] == "ENST_CFTR_MANE" and c["mane"] == "NM_000492.4"
    assert c["consequence"] == "intron_variant" and c["intron"] == "12/26" and c["biotype"] == "protein_coding"
    assert c["spliceai_ds_max"] == "0.55" and c["spliceai_detail"] == "AG:0|AL:0.02|DG:0|DL:0.55"
    assert c["most_severe_consequence"] == "non_coding_transcript_exon_variant" and c["impact_any_coding"] == "MODIFIER"


def test_spliceai_is_the_variant_level_maximum():
    r = VepRetriever(StubHttp())
    low = {"DS_AG": 0, "DS_AL": 0, "DS_DG": 0.01, "DS_DL": 0}
    high = {"DS_AG": 0.8, "DS_AL": 0, "DS_DG": 0, "DS_DL": 0}
    # the chosen (MANE) transcript's block wins ties …
    a = _tc(transcript_id="ENST_A", gene_id="A", gene_symbol="GA", mane_select="NM_A", spliceai=low)
    b = _tc(transcript_id="ENST_B", gene_id="B", gene_symbol="GB", spliceai={**low, "DS_DG": 0.01})
    c = r.extract(_rec({"allele_string": "A/G", "transcript_consequences": [b, a]}))
    assert c["transcript_id"] == "ENST_A" and c["spliceai_ds_max"] == "0.01" and c["spliceai_detail"] == "AG:0|AL:0|DG:0.01|DL:0"
    # … but an overlapping gene's higher score is reported, labelled with that gene
    b2 = {**b, "spliceai": high}
    c = r.extract(_rec({"allele_string": "A/G", "transcript_consequences": [b2, a]}))
    assert c["transcript_id"] == "ENST_A" and c["spliceai_ds_max"] == "0.8"
    assert c["spliceai_detail"] == "AG:0.8|AL:0|DG:0|DL:0 (GB)"
    # chosen transcript without a block: the score still comes from the one that has it
    c = r.extract(_rec({"allele_string": "A/G", "transcript_consequences": [b2, {**a, "spliceai": None}]}))
    assert c["spliceai_ds_max"] == "0.8"
    # a partial block (some DS_ keys missing) is still usable
    c = r.extract(_rec({"allele_string": "A/G", "transcript_consequences": [{**a, "spliceai": {"DS_AL": 0.3}}]}))
    assert c["spliceai_ds_max"] == "0.3" and c["spliceai_detail"] == "AG:|AL:0.3|DG:|DL:"


# ---------------------------------------------------------------- live

@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ENGINE_LIVE_TESTS"), reason="set ENGINE_LIVE_TESTS=1 to hit rest.ensembl.org")
def test_live_public_variant(tmp_path: Path):
    http = Http(HttpCache(tmp_path / "cache"), limiter=RateLimiter({"rest.ensembl.org": 5.0}))
    r = VepRetriever(http)
    out = r.retrieve([MTHFR])
    assert len(out[MTHFR]) == 1
    c = r.extract(out[MTHFR])
    assert c["gene_symbol"] == "MTHFR" and c["hgvsp"].endswith("p.Ala222Val") and c["rsid"] == "rs1801133"
    assert c["mane"].startswith("NM_005957") and c["impact_any_coding"] == "MODERATE"
    assert r.version().startswith("Ensembl VEP ") and "assembly GRCh38" in r.version()
    assert r.dropped == 0
    # warm cache → byte-identical record, online (live version check agrees) and offline
    again = VepRetriever(http).retrieve([MTHFR])
    assert again[MTHFR][0].to_json() == out[MTHFR][0].to_json()
    offline = Http(HttpCache(tmp_path / "cache"), offline=True)
    assert VepRetriever(offline).retrieve([MTHFR])[MTHFR][0].to_json() == out[MTHFR][0].to_json()
