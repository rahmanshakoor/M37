"""ClinVar retriever against a tiny bgzipped ClinVar-shaped VCF.

The three public textbook variants come verbatim from a recorded release
(``tests/fixtures/clinvar/records.json``). Every decoy — the other ALT at the same
position, the overlapping large deletion, the multi-allelic row — is synthetic, with
a VariationID in the 9xxxxxxxx range so it can never be mistaken for a real record.
"""

import functools
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import urllib.request
from datetime import timezone
from email.utils import parsedate_to_datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest

import engine.retrieve.clinvar as clinvar_module
from engine.retrieve import Retriever
from engine.retrieve.clinvar import (
    INFO_TAGS, RETRIEVED_SUFFIX, ClinvarRelease, ClinvarRetriever, pathogenicity, stars,
)
from engine.retrieve.http import Http, HttpCache, RateLimiter, Response
from engine.retrieve.store import EvidenceRecord

FIXTURES = Path(__file__).parent / "fixtures" / "clinvar"
RECORDS = json.loads((FIXTURES / "records.json").read_text())

MTHFR = ("1", 11796321, "G", "A")
CFTR = ("7", 117559590, "ATCT", "A")
TP53 = ("17", 7675088, "C", "T")

# The Last-Modified NCBI served for the pinned release, and its ISO form every record cites.
LAST_MODIFIED = "Sun, 06 Sep 2026 16:25:09 GMT"
RETRIEVED_AT = "2026-09-06T16:25:09+00:00"
LAST_MODIFIED_EPOCH = parsedate_to_datetime(LAST_MODIFIED).timestamp()

needs_htslib = pytest.mark.skipif(
    not (shutil.which("bcftools") and shutil.which("bgzip") and shutil.which("tabix")),
    reason="bcftools/htslib not installed",
)

# Synthetic decoys (IDs 9xxxxxxxx). Same INFO grammar as the real rows.
SYNTHETIC = [
    # other ALT at the TP53 position
    ("17", 7675088, "900000001", "C", "G",
     "ALLELEID=900000001;CLNDN=synthetic_decoy;CLNHGVS=NC_000017.11:g.7675088C>G;"
     "CLNREVSTAT=criteria_provided,_single_submitter;CLNSIG=Likely_benign;CLNVC=single_nucleotide_variant;"
     "GENEINFO=TP53:7157;ORIGIN=1"),
    # SNV at the CFTR indel position (same POS, different REF/ALT)
    ("7", 117559590, "900000002", "A", "G",
     "ALLELEID=900000002;CLNDN=synthetic_decoy;CLNHGVS=NC_000007.14:g.117559590A>G;"
     "CLNREVSTAT=criteria_provided,_multiple_submitters,_no_conflicts;CLNSIG=Benign/Likely_benign;"
     "CLNVC=single_nucleotide_variant;GENEINFO=CFTR:1080;ORIGIN=1"),
    # multi-allelic row (ClinVar never writes one; the retriever must refuse it)
    ("17", 7675200, "900000003", "A", "C,G",
     "ALLELEID=900000003;CLNDN=synthetic_multiallelic;CLNREVSTAT=no_assertion_criteria_provided;"
     "CLNSIG=Uncertain_significance;CLNVC=single_nucleotide_variant;GENEINFO=TP53:7157;ORIGIN=0"),
    # 44 bp deletion whose span covers the TP53 position — a region query returns it
    ("17", 7675047, "900000004", "GATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGAT", "G",
     "ALLELEID=900000004;CLNDN=synthetic_overlapping_deletion;CLNHGVS=NC_000017.11:g.7675048_7675090del;"
     "CLNREVSTAT=criteria_provided,_single_submitter;CLNSIG=Pathogenic;CLNVC=Deletion;"
     "GENEINFO=TP53:7157;MC=SO:0001575|splice_donor_variant;ORIGIN=1"),
]


def _spell(chrom: str, style: str) -> str:
    if style == "ucsc":
        return "chrM" if chrom == "MT" else f"chr{chrom}"
    return chrom


def write_sidecar(gz: Path, *, retrieved_at: str = RETRIEVED_AT, md5: str | None = None) -> Path:
    """What ``ClinvarRelease.download`` leaves beside the VCF."""
    p = Path(str(gz) + RETRIEVED_SUFFIX)
    p.write_text(json.dumps({
        "retrieved_at": retrieved_at, "last_modified": LAST_MODIFIED, "recorded_at": "2026-09-12T16:52:00+00:00",
        "url": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/" + gz.name,
        "md5": md5 or hashlib.md5(gz.read_bytes()).hexdigest(),
    }, sort_keys=True, indent=1) + "\n")
    return p


def make_vcf(tmp: Path, *, style: str = "ensembl", header_edits=None, name: str = "clinvar_20260905.vcf.gz",
             synthetic: bool = True, sidecar: bool = True) -> Path:
    header = []
    for line in RECORDS["header"]:
        if line.startswith("##contig=<ID="):
            cid = line[13:].rstrip(">")
            line = f"##contig=<ID={_spell(cid, style) if cid in {*map(str, range(1, 23)), 'X', 'Y', 'MT'} else cid}>"
        header.append(line)
    if header_edits:
        header = header_edits(header)
    rows = [(r["CHROM"], r["POS"], r["ID"], r["REF"], r["ALT"], r["INFO"]) for r in RECORDS["rows"]]
    if synthetic:
        rows += SYNTHETIC
    order = {c: i for i, c in enumerate([*map(str, range(1, 23)), "X", "Y", "MT"])}
    rows.sort(key=lambda r: (order[r[0]], r[1]))
    body = "".join(f"{_spell(c, style)}\t{p}\t{i}\t{ref}\t{alt}\t.\t.\t{info}\n" for c, p, i, ref, alt, info in rows)
    raw = tmp / name.removesuffix(".gz")
    raw.write_text("\n".join(header) + "\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n" + body)
    gz = tmp / name
    subprocess.run(["bgzip", "-c", str(raw)], stdout=open(gz, "wb"), check=True)
    subprocess.run(["tabix", "-p", "vcf", str(gz)], check=True)
    if sidecar:
        write_sidecar(gz)
    return gz


def _jsons(found, keys):
    return [[x.to_json() for x in found[k]] for k in keys]


# ---------------------------------------------------------------- pure helpers

@pytest.mark.parametrize("clnsig, expected", [
    # single classes
    ("Pathogenic", "Pathogenic"),
    ("Likely_pathogenic", "Likely_pathogenic"),
    ("Uncertain_significance", "Uncertain_significance"),
    ("Conflicting_classifications_of_pathogenicity", "Conflicting_classifications_of_pathogenicity"),
    # '|' joins distinct aggregate classifications: only the pathogenicity term is returned
    ("Pathogenic|drug_response", "Pathogenic"),
    ("drug_response|Pathogenic", "Pathogenic"),
    ("Benign|Affects|association|other", "Benign"),
    ("Conflicting_classifications_of_pathogenicity|other|risk_factor", "Conflicting_classifications_of_pathogenicity"),
    # combined categories and qualifiers are a different claim from their strongest member: verbatim
    ("Pathogenic/Likely_pathogenic", "Pathogenic/Likely_pathogenic"),
    ("Benign/Likely_benign", "Benign/Likely_benign"),
    ("Pathogenic/Likely_pathogenic|other", "Pathogenic/Likely_pathogenic"),
    ("Benign/Likely_benign|drug_response|other", "Benign/Likely_benign"),
    ("Pathogenic,_low_penetrance", "Pathogenic,_low_penetrance"),
    ("Likely_pathogenic,_low_penetrance", "Likely_pathogenic,_low_penetrance"),
    ("Pathogenic/Likely_pathogenic/Pathogenic,_low_penetrance", "Pathogenic/Likely_pathogenic/Pathogenic,_low_penetrance"),
    ("Pathogenic/Pathogenic,_low_penetrance|risk_factor", "Pathogenic/Pathogenic,_low_penetrance"),
    ("Likely_pathogenic/Likely_risk_allele", "Likely_pathogenic/Likely_risk_allele"),
    ("Uncertain_significance/Uncertain_risk_allele", "Uncertain_significance/Uncertain_risk_allele"),
    ("Pathogenic/Likely_pathogenic/Pathogenic,_low_penetrance/Established_risk_allele|risk_factor",
     "Pathogenic/Likely_pathogenic/Pathogenic,_low_penetrance/Established_risk_allele"),
    # VUS tiers are uncertain-significance terms
    ("VUS-high", "VUS-high"),
    ("VUS-low", "VUS-low"),
    ("Uncertain_significance/VUS-mid", "Uncertain_significance/VUS-mid"),
    # pre-2024 spelling normalised
    ("Conflicting_interpretations_of_pathogenicity", "Conflicting_classifications_of_pathogenicity"),
    ("Conflicting_interpretations_of_pathogenicity|other", "Conflicting_classifications_of_pathogenicity"),
    # not a pathogenicity classification
    ("drug_response", ""),
    ("not_provided", ""),
    ("no_classification_for_the_single_variant", ""),
    ("no_classifications_from_unflagged_records", ""),
    ("Uncertain_risk_allele", ""),
    ("Likely_risk_allele|risk_factor", ""),
    ("Established_risk_allele|association", ""),
    ("association|drug_response|risk_factor", ""),
    (".", ""),
    ("", ""),
])
def test_pathogenicity_is_the_germline_term_verbatim(clnsig, expected):
    assert pathogenicity(clnsig) == expected


@pytest.mark.parametrize("revstat, expected", [
    ("practice_guideline", "4"),
    ("reviewed_by_expert_panel", "3"),
    ("criteria_provided,_multiple_submitters,_no_conflicts", "2"),
    ("criteria_provided,_conflicting_classifications", "1"),
    ("criteria_provided,_single_submitter", "1"),
    ("no_assertion_criteria_provided", "0"),
    ("no_classification_provided", "0"),
    ("no_classification_for_the_single_variant", "0"),
    ("no_classifications_from_unflagged_records", "0"),
    ("", ""),
    ("something_new", ""),
])
def test_stars_table(revstat, expected):
    assert stars(revstat) == expected


# ---------------------------------------------------------------- retrieve + extract

@needs_htslib
def test_exact_allele_match_and_extract(tmp_path: Path):
    r = ClinvarRetriever(make_vcf(tmp_path))
    assert isinstance(r, Retriever)
    assert r.version() == "ClinVar 2026-09-05 GRCh38 VCF"
    assert r.retrieved_at == RETRIEVED_AT  # from the download sidecar, not the file's mtime

    decoy_tp53 = ("17", 7675088, "C", "G")
    absent_same_pos = ("17", 7675088, "C", "A")
    absent_elsewhere = ("1", 11796999, "G", "A")
    keys = [TP53, decoy_tp53, absent_same_pos, CFTR, MTHFR, absent_elsewhere]
    found = r.retrieve(keys)
    assert set(found) == set(keys)

    # TP53: the C>T row only — not the C>G decoy at the same POS, not the overlapping deletion.
    (rec,) = found[TP53]
    assert rec.record_id == "clinvar:VCV000012374"
    assert rec.source == "clinvar" and rec.source_version == "ClinVar 2026-09-05 GRCh38 VCF"
    assert rec.url == "https://www.ncbi.nlm.nih.gov/clinvar/variation/12374/"
    assert rec.query == {"variant": "17:7675088:C:T", "chrom": "17", "pos": 7675088, "ref": "C", "alt": "T",
                         "vcf": "clinvar_20260905.vcf.gz"}
    assert rec.retrieved_at == RETRIEVED_AT
    p = rec.payload
    assert (p["CHROM"], p["POS"], p["ID"], p["REF"], p["ALT"]) == ("17", "7675088", "12374", "C", "T")
    assert p["CLNSIG"] == "Pathogenic" and p["CLNREVSTAT"] == "reviewed_by_expert_panel"
    assert p["ALLELEID"] == "27413" and p["RS"] == "28934578" and p["GENEINFO"] == "TP53:7157"
    assert p["CLNSIGCONF"] == "." and p["ONC"] == "Oncogenic" and p["SCI"] == "Tier_I_-_Strong"
    assert p["clinvar_release"] == "2026-09-05"
    assert set(INFO_TAGS) <= set(p)
    cols = r.extract(found[TP53])
    assert tuple(cols) == r.columns
    assert cols["clinvar_vcv"] == "VCV000012374"
    assert cols["clinvar_clnsig"] == "Pathogenic" and cols["clinvar_pathogenicity"] == "Pathogenic"
    assert cols["clinvar_revstat"] == "reviewed_by_expert_panel" and cols["clinvar_stars"] == "3"
    assert cols["clinvar_conditions"].startswith("TP53-related_disorder|Colorectal_cancer|")
    assert cols["clinvar_alleleid"] == "27413" and cols["clinvar_hgvs"] == "NC_000017.11:g.7675088C>T"
    assert cols["clinvar_release"] == "2026-09-05"

    # The decoy is a hit for its own allele, and only for its own allele.
    (d,) = found[decoy_tp53]
    assert d.record_id == "clinvar:VCV900000001" and d.payload["CLNSIG"] == "Likely_benign"
    assert found[absent_same_pos] == [] and found[absent_elsewhere] == []
    assert r.extract([]) == {c: "" for c in r.columns}

    # CFTR: left-anchored indel matched as given; CLNHGVS is the 3'-shifted text and is reported, never matched on.
    (c,) = found[CFTR]
    assert c.record_id == "clinvar:VCV000007105" and c.payload["CLNSIGINCL"] == "634837:Pathogenic"
    cc = r.extract(found[CFTR])
    assert cc["clinvar_hgvs"] == "NC_000007.14:g.117559592_117559594del"
    assert cc["clinvar_stars"] == "4" and cc["clinvar_revstat"] == "practice_guideline"
    assert cc["clinvar_pathogenicity"] == "Pathogenic"

    # MTHFR: a non-pathogenicity classification stays verbatim; the class column is empty.
    (m,) = found[MTHFR]
    assert m.record_id == "clinvar:VCV000003520"
    mc = r.extract(found[MTHFR])
    assert mc["clinvar_clnsig"] == "drug_response" and mc["clinvar_pathogenicity"] == ""
    assert mc["clinvar_stars"] == "3" and mc["clinvar_alleleid"] == "18559"

    # a combined category is projected as its own value, never as its strongest member
    (s,) = r.retrieve([("7", 117559590, "A", "G")])[("7", 117559590, "A", "G")]
    assert r.extract([s])["clinvar_pathogenicity"] == "Benign/Likely_benign"


@needs_htslib
def test_overlapping_deletion_is_returned_by_the_region_query_but_filtered(tmp_path: Path):
    r = ClinvarRetriever(make_vcf(tmp_path))
    rows = r._query([("17", 7675088)])
    assert {row["ID"] for row in rows} == {"12374", "900000001", "900000004"}  # the deletion overlaps
    found = r.retrieve([TP53])
    assert [rec.record_id for rec in found[TP53]] == ["clinvar:VCV000012374"]


@needs_htslib
def test_multiallelic_row_is_refused(tmp_path: Path):
    """One VariationID names one allele; a row with two ALTs would give two queries the
    same record id, so it is a malformed release, not a match."""
    r = ClinvarRetriever(make_vcf(tmp_path))
    with pytest.raises(ValueError, match="VCV900000003 is multi-allelic"):
        r.retrieve([("17", 7675200, "A", "G")])
    assert r.retrieve([TP53])[TP53][0].record_id == "clinvar:VCV000012374"  # rows elsewhere are unaffected


@needs_htslib
def test_lowercase_alleles_and_chr_prefixed_file_still_match(tmp_path: Path):
    (tmp_path / "e").mkdir()
    (tmp_path / "u").mkdir()
    ensembl = ClinvarRetriever(make_vcf(tmp_path / "e", style="ensembl"))
    ucsc = ClinvarRetriever(make_vcf(tmp_path / "u", style="ucsc"))
    lower = ("17", 7675088, "c", "t")
    for r in (ensembl, ucsc):
        found = r.retrieve([TP53, lower, MTHFR])
        assert [x.record_id for x in found[TP53]] == ["clinvar:VCV000012374"]
        assert [x.record_id for x in found[lower]] == ["clinvar:VCV000012374"]
        assert found[lower][0].query["ref"] == "c"  # the query stays exactly what was asked
        assert found[MTHFR][0].payload["CHROM"] in ("1", "chr1")


@needs_htslib
def test_chromosome_absent_from_the_index_is_absent_not_an_error(tmp_path: Path):
    # Like the real release: no ##contig lines at all, so the spellings come from the
    # ##contig lines htslib synthesises from the .tbi — only chromosomes with rows.
    r = ClinvarRetriever(make_vcf(tmp_path, header_edits=lambda h: [l for l in h if not l.startswith("##contig=")]))
    assert r._spelling == {"1": "1", "7": "7", "17": "17"}
    found = r.retrieve([("MT", 100, "A", "G"), ("X", 1000, "A", "G"), TP53])
    assert found[("MT", 100, "A", "G")] == [] and found[("X", 1000, "A", "G")] == []
    assert found[TP53][0].record_id == "clinvar:VCV000012374"
    assert r.retrieve([("MT", 100, "A", "G")]) == {("MT", 100, "A", "G"): []}  # nothing to ask bcftools

    # a chr-prefixed file declaring no chrM behaves the same
    (tmp_path / "u").mkdir()
    no_mt = ClinvarRetriever(make_vcf(tmp_path / "u", style="ucsc",
                                      header_edits=lambda h: [l for l in h if l != "##contig=<ID=chrM>"]))
    assert no_mt.retrieve([TP53, ("MT", 100, "A", "G")])[("MT", 100, "A", "G")] == []


@needs_htslib
def test_records_are_byte_identical_across_runs_and_duplicate_keys_collapse(tmp_path: Path):
    vcf = make_vcf(tmp_path)
    a = ClinvarRetriever(vcf).retrieve([TP53, CFTR, MTHFR])
    b = ClinvarRetriever(vcf).retrieve([MTHFR, CFTR, TP53, TP53])
    for k in (TP53, CFTR, MTHFR):
        assert [x.to_json() for x in a[k]] == [x.to_json() for x in b[k]]
        assert EvidenceRecord.from_json(a[k][0].to_json()) == a[k][0]
    assert len(b[TP53]) == 1  # a key passed twice is one record, not two copies


@needs_htslib
def test_retrieved_at_comes_from_the_sidecar_and_survives_a_copy(tmp_path: Path, caplog):
    (tmp_path / "a").mkdir()
    src = make_vcf(tmp_path / "a")
    a = ClinvarRetriever(src)
    first = _jsons(a.retrieve([TP53, CFTR, MTHFR]), [TP53, CFTR, MTHFR])

    # cp without -p: new mtimes, same bytes, same sidecar → identical records
    (tmp_path / "b").mkdir()
    for suffix in ("", ".tbi", RETRIEVED_SUFFIX):
        shutil.copyfile(str(src) + suffix, str(tmp_path / "b" / src.name) + suffix)
    os.utime(tmp_path / "b" / src.name, (LAST_MODIFIED_EPOCH + 86400 * 30,) * 2)
    b = ClinvarRetriever(tmp_path / "b" / src.name)
    assert b.retrieved_at == RETRIEVED_AT
    assert _jsons(b.retrieve([TP53, CFTR, MTHFR]), [TP53, CFTR, MTHFR]) == first

    # without the sidecar the only date left is the mtime — used, but said out loud
    (tmp_path / "c").mkdir()
    for suffix in ("", ".tbi"):
        shutil.copyfile(str(src) + suffix, str(tmp_path / "c" / src.name) + suffix)
    os.utime(tmp_path / "c" / src.name, (LAST_MODIFIED_EPOCH + 86400 * 30,) * 2)
    with caplog.at_level(logging.WARNING, logger="engine.retrieve.clinvar"):
        c = ClinvarRetriever(tmp_path / "c" / src.name)
    assert c.retrieved_at == "2026-10-06T16:25:09+00:00"
    assert any("falls back to the file mtime" in m for m in caplog.messages)

    # a sidecar whose md5 is not this file's: the copy is not the release that was downloaded
    write_sidecar(tmp_path / "b" / src.name, md5="0" * 32)
    with pytest.raises(ValueError, match="recorded at download"):
        ClinvarRetriever(tmp_path / "b" / src.name)


@needs_htslib
def test_query_failure_raises_instead_of_reporting_absence(tmp_path: Path):
    vcf = make_vcf(tmp_path)
    r = ClinvarRetriever(vcf)
    tbi = Path(str(vcf) + ".tbi")
    good = tbi.read_bytes()
    tbi.write_bytes(good[: len(good) // 2])
    with pytest.raises(RuntimeError, match=r"bcftools query failed \(\d+\)"):
        r.retrieve([TP53])
    tbi.write_bytes(good)
    assert r.retrieve([TP53])[TP53][0].record_id == "clinvar:VCV000012374"


@needs_htslib
def test_streaming_path_matches_index_path(tmp_path: Path, monkeypatch):
    vcf = make_vcf(tmp_path)
    keys = [TP53, ("17", 7675088, "C", "G"), ("17", 7675088, "C", "A"), CFTR, MTHFR, ("1", 11796999, "G", "A")]
    r = ClinvarRetriever(vcf)
    seeks = _jsons(r.retrieve(keys), keys)

    modes: list[str] = []
    real_run = subprocess.run

    def spy(cmd, **kw):
        if cmd[:2] == ["bcftools", "query"]:
            modes.append(cmd[2])
            assert cmd[3] == "-" and "input" in kw  # positions on stdin, never in a file
        return real_run(cmd, **kw)

    monkeypatch.setattr(clinvar_module.subprocess, "run", spy)
    monkeypatch.setattr(clinvar_module, "STREAM_THRESHOLD", 0)
    assert _jsons(r.retrieve(keys), keys) == seeks
    assert modes == ["-T"]
    assert not list(tmp_path.glob("**/regions*"))


@needs_htslib
def test_missing_info_tags_become_dots(tmp_path: Path):
    old = make_vcf(tmp_path, header_edits=lambda h: [l for l in h if not l.startswith(("##INFO=<ID=ONC", "##INFO=<ID=SCI"))],
                   synthetic=False, name="clinvar_20260905.vcf.gz")
    # rows still carry ONC=/SCI= for TP53; strip them so the file is consistent with its header
    txt = subprocess.run(["bcftools", "view", str(old)], capture_output=True, text=True, check=True).stdout
    lines = []
    for line in txt.splitlines():
        if not line.startswith("#"):
            f = line.split("\t")
            f[7] = ";".join(kv for kv in f[7].split(";") if not kv.startswith(("ONC", "SCI")))
            line = "\t".join(f)
        lines.append(line)
    raw = tmp_path / "old.vcf"
    raw.write_text("\n".join(lines) + "\n")
    gz = tmp_path / "clinvar_20260905.vcf.gz"
    subprocess.run(["bgzip", "-f", "-c", str(raw)], stdout=open(gz, "wb"), check=True)
    subprocess.run(["tabix", "-f", "-p", "vcf", str(gz)], check=True)
    write_sidecar(gz)
    r = ClinvarRetriever(gz)
    (rec,) = r.retrieve([TP53])[TP53]
    assert rec.payload["ONC"] == "." and rec.payload["SCIREVSTAT"] == "."
    assert rec.payload["CLNSIG"] == "Pathogenic"


@needs_htslib
def test_release_and_reference_are_checked(tmp_path: Path):
    with pytest.raises(ValueError, match="fileDate"):
        ClinvarRetriever(make_vcf(tmp_path, name="clinvar_20260829.vcf.gz"))
    with pytest.raises(ValueError, match="GRCh38"):
        ClinvarRetriever(make_vcf(tmp_path, header_edits=lambda h: [l.replace("GRCh38", "GRCh37") for l in h]))
    with pytest.raises(FileNotFoundError):
        ClinvarRetriever(tmp_path / "nope.vcf.gz")
    # an undated filename falls back to ##fileDate
    r = ClinvarRetriever(make_vcf(tmp_path, name="clinvar.vcf.gz"))
    assert r.release_date == "2026-09-05"


# ---------------------------------------------------------------- release pin + download

class _StubHttp:
    def __init__(self, text: str):
        self.text = text
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, **kw):
        self.calls.append((url, kw))
        return Response(200, self.text, "2026-09-12T19:05:00+00:00", False, "stub", {})

    def request(self, method, url, **kw):
        return self.get(url, **kw)

    def post(self, url, json_body, **kw):
        raise AssertionError("no POST expected")


def test_latest_filename_from_recorded_listing():
    listing = json.loads((FIXTURES / "listing.json").read_text())
    http = _StubHttp(listing["text"])
    assert ClinvarRelease.latest_filename(http) == "clinvar_20260905.vcf.gz"  # not the _papu file
    assert http.calls == [("https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/", {"cache_ok": False})]
    with pytest.raises(RuntimeError):
        ClinvarRelease.latest_filename(_StubHttp("<html>nothing here</html>"))


def test_pinned_config_matches_the_probe(tmp_path: Path):
    rel = ClinvarRelease.load()
    assert rel.date == "20260905" and rel.filename == "clinvar_20260905.vcf.gz"
    assert rel.url == "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar_20260905.vcf.gz"
    assert rel.md5 == "ece04fe2ee72db1dd988d8b188df34b9" and rel.bytes == 193476678 and rel.tbi_bytes == 610243
    assert rel.candidate_urls() == [
        rel.url,
        "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly/clinvar_20260905.vcf.gz",
        "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2026/clinvar_20260905.vcf.gz",
    ]
    with pytest.raises(FileNotFoundError, match="release pin not found"):
        ClinvarRelease.load(tmp_path / "missing.yaml")


class _Quiet(SimpleHTTPRequestHandler):
    requests: list[tuple[str, str]] = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        _Quiet.requests.append(("GET", self.path))
        super().do_GET()

    def do_HEAD(self):
        _Quiet.requests.append(("HEAD", self.path))
        super().do_HEAD()


@pytest.fixture
def file_server(tmp_path: Path, monkeypatch):
    # urllib would route 127.0.0.1 through a bogus http_proxy from the environment
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    root = tmp_path / "srv"
    (root / "weekly").mkdir(parents=True)
    _Quiet.requests = []
    srv = HTTPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(root)))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield root, f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_download_falls_back_verifies_md5_and_skips_when_present(file_server, tmp_path: Path):
    root, base = file_server
    vcf_bytes = b"\x1f\x8b" + b"fake bgzip payload " * 100
    tbi_bytes = b"fake index"
    md5 = hashlib.md5(vcf_bytes).hexdigest()
    # the top-level dated file has "rotated": only weekly/ holds it
    (root / "weekly" / "clinvar_20260905.vcf.gz").write_bytes(vcf_bytes)
    (root / "weekly" / "clinvar_20260905.vcf.gz.tbi").write_bytes(tbi_bytes)
    (root / "weekly" / "clinvar_20260905.vcf.gz.md5").write_text(f"{md5}  /ncbi/internal/path/clinvar_20260905.vcf.gz\n")
    for f in (root / "weekly").iterdir():
        os.utime(f, (LAST_MODIFIED_EPOCH, LAST_MODIFIED_EPOCH))
    rel = ClinvarRelease(date="20260905", url=f"{base}/clinvar_20260905.vcf.gz", md5=md5, bytes=len(vcf_bytes),
                         tbi_bytes=len(tbi_bytes), fallback_dirs=("weekly/", "archive_2.0/{year}/"))
    dest = tmp_path / "ref"
    out = rel.download(dest)
    assert out == dest / "clinvar_20260905.vcf.gz" and out.read_bytes() == vcf_bytes
    assert (dest / "clinvar_20260905.vcf.gz.tbi").read_bytes() == tbi_bytes
    assert (dest / "clinvar_20260905.vcf.gz.md5").read_text().split()[0] == md5
    assert not list(dest.glob("*.part"))
    # the small .tbi is tried first, so the rotated top-level directory costs no big transfer
    assert _Quiet.requests[:3] == [("GET", "/clinvar_20260905.vcf.gz.tbi"), ("GET", "/weekly/clinvar_20260905.vcf.gz.tbi"),
                                   ("GET", "/weekly/clinvar_20260905.vcf.gz")]
    sidecar = dest / ("clinvar_20260905.vcf.gz" + RETRIEVED_SUFFIX)
    meta = json.loads(sidecar.read_text())
    assert meta["retrieved_at"] == RETRIEVED_AT and meta["last_modified"] == LAST_MODIFIED and meta["md5"] == md5
    assert meta["url"] == f"{base}/weekly/clinvar_20260905.vcf.gz"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", meta["recorded_at"])

    # present and verified → no request at all
    n = len(_Quiet.requests)
    assert rel.download(dest) == out and len(_Quiet.requests) == n

    # present and verified but the sidecar is gone → rebuilt from one HEAD, same instant
    sidecar.unlink()
    assert rel.download(dest) == out
    assert _Quiet.requests[n:] == [("HEAD", "/clinvar_20260905.vcf.gz"), ("HEAD", "/weekly/clinvar_20260905.vcf.gz")]
    assert json.loads(sidecar.read_text())["retrieved_at"] == RETRIEVED_AT
    n = len(_Quiet.requests)

    # a .tbi of the wrong size beside a verified VCF is not "present": fetched again
    (dest / "clinvar_20260905.vcf.gz.tbi").write_bytes(b"x")
    assert rel.download(dest) == out
    assert (dest / "clinvar_20260905.vcf.gz.tbi").read_bytes() == tbi_bytes and len(_Quiet.requests) > n

    # a pin that disagrees with the bytes is refused and leaves nothing behind
    bad = ClinvarRelease(date="20260905", url=rel.url, md5="0" * 32, bytes=len(vcf_bytes), fallback_dirs=("weekly/",))
    with pytest.raises(ValueError, match="md5"):
        bad.download(tmp_path / "bad")
    assert not list((tmp_path / "bad").iterdir())

    # a sidecar that disagrees with the pin is an error too (the pin was mistyped)
    (root / "weekly" / "clinvar_20260905.vcf.gz.md5").write_text("deadbeef  x\n")
    with pytest.raises(ValueError, match="sidecar"):
        rel.download(tmp_path / "side")

    # nothing anywhere → FileNotFoundError naming every candidate
    gone = ClinvarRelease(date="20260905", url=f"{base}/clinvar_20260905.vcf.gz", md5=md5, bytes=1,
                          fallback_dirs=("archive_2.0/{year}/",))
    with pytest.raises(FileNotFoundError, match="archive_2.0/2026/"):
        gone.download(tmp_path / "gone")


@needs_htslib
def test_downloaded_release_is_cited_by_its_last_modified(file_server, tmp_path: Path):
    """End to end: download() → sidecar → retriever, and a copy of the download is
    byte-identical in what it retrieves."""
    root, base = file_server
    served = make_vcf(root / "weekly", sidecar=False)
    for f in (root / "weekly").iterdir():
        os.utime(f, (LAST_MODIFIED_EPOCH, LAST_MODIFIED_EPOCH))
    rel = ClinvarRelease(date="20260905", url=f"{base}/clinvar_20260905.vcf.gz",
                         md5=hashlib.md5(served.read_bytes()).hexdigest(), bytes=served.stat().st_size,
                         tbi_bytes=Path(str(served) + ".tbi").stat().st_size, fallback_dirs=("weekly/",))
    vcf = rel.download(tmp_path / "ref")
    r = ClinvarRetriever(vcf)
    assert r.retrieved_at == RETRIEVED_AT
    found = r.retrieve([TP53, CFTR, MTHFR])
    assert [x.record_id for x in found[CFTR]] == ["clinvar:VCV000007105"]

    shutil.copytree(tmp_path / "ref", tmp_path / "copy")
    os.utime(tmp_path / "copy" / vcf.name, (LAST_MODIFIED_EPOCH + 3600,) * 2)  # a copy that lost its mtime
    again = ClinvarRetriever(tmp_path / "copy" / vcf.name).retrieve([TP53, CFTR, MTHFR])
    assert _jsons(again, [TP53, CFTR, MTHFR]) == _jsons(found, [TP53, CFTR, MTHFR])


# ---------------------------------------------------------------- live

@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ENGINE_LIVE_TESTS"), reason="set ENGINE_LIVE_TESTS=1 to hit NCBI")
@needs_htslib
def test_live_release_listing_and_remote_lookup(tmp_path: Path, monkeypatch):
    """One public textbook variant against the pinned release over htslib range requests."""
    monkeypatch.chdir(tmp_path)  # where htslib would drop the remote index if the retriever let it
    http = Http(HttpCache(tmp_path / "cache"), limiter=RateLimiter(default_per_second=1))
    assert re.fullmatch(r"clinvar_\d{8}\.vcf\.gz", ClinvarRelease.latest_filename(http))
    ClinvarRelease.latest_filename(http)
    assert http.live_requests == 2  # the listing is never answered from cache
    rel = ClinvarRelease.load()
    url = rel.resolve_url()
    r = ClinvarRetriever(url)
    assert r.version() == "ClinVar 2026-09-05 GRCh38 VCF"
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as resp:
        last_modified = resp.headers["Last-Modified"]
    assert r.retrieved_at == parsedate_to_datetime(last_modified).astimezone(timezone.utc).isoformat(timespec="seconds")
    found = r.retrieve([MTHFR])
    (rec,) = found[MTHFR]
    assert rec.record_id == "clinvar:VCV000003520"
    cols = r.extract(found[MTHFR])
    assert cols["clinvar_clnsig"] == "drug_response" and cols["clinvar_stars"] == "3"
    assert cols["clinvar_alleleid"] == "18559" and cols["clinvar_pathogenicity"] == ""
    assert not list(tmp_path.glob("*.tbi"))
