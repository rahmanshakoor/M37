"""Orchestrator test with stub retrievers — no network, no real sources."""

import json
from pathlib import Path

import pytest

from engine.retrieve.run import (RetrieveOptions, gnomad_prefilter, read_annotated, run_retrieve, select_keys)
from engine.retrieve.store import EvidenceRecord, key_str
from tests.test_ingest import make_case, make_vcf, run_ingest

pytestmark = pytest.mark.skipif(
    __import__("shutil").which("bcftools") is None, reason="bcftools not installed")


class StubVep:
    source = "vep"
    columns = ("gene_symbol", "impact_any_coding", "vep_gnomade_af", "vep_gnomadg_af")

    def __init__(self):
        self.calls = []

    def version(self):
        return "Ensembl VEP stub"

    def retrieve(self, keys):
        self.calls.append(list(keys))
        out = {}
        for k in keys:
            rec = EvidenceRecord(record_id=f"vep:{key_str(k)}", source="vep", source_version="stub",
                                 query={"variant": key_str(k)}, url="https://example/vep", retrieved_at="t",
                                 payload={"gene": "GENE" + k[0], "impact": "HIGH" if k[0] == "15" else "MODIFIER",
                                          "af": 0.5 if k[1] == 100 else None})
            out[k] = [rec]
        return out

    def extract(self, records):
        if not records:
            return {c: "" for c in self.columns}
        p = records[0].payload
        return {"gene_symbol": p["gene"], "impact_any_coding": p["impact"],
                "vep_gnomade_af": "" if p["af"] is None else str(p["af"]), "vep_gnomadg_af": ""}


class StubGnomad:
    source = "gnomad"
    columns = ("gnomad_af", "gnomad_nhom")

    def __init__(self):
        self.calls = []

    def version(self):
        return "gnomad_r4 stub"

    def retrieve(self, keys):
        self.calls.append(list(keys))
        # first key present, others absent
        out = {k: [] for k in keys}
        if keys:
            k = keys[0]
            out[k] = [EvidenceRecord(record_id=f"gnomad:{k[0]}-{k[1]}-{k[2]}-{k[3]}", source="gnomad", source_version="stub",
                                     query={}, url="https://example/gnomad", retrieved_at="t", payload={"af": 1e-5, "nhom": 0})]
        return out

    def extract(self, records):
        if not records:
            return {"gnomad_af": "", "gnomad_nhom": ""}
        return {"gnomad_af": repr(records[0].payload["af"]), "gnomad_nhom": "0"}


def test_orchestrator_end_to_end(tmp_path: Path):
    vcf = make_vcf(tmp_path)
    case = make_case(tmp_path, vcf)
    run = tmp_path / "run"
    run_ingest(case, run, threads=1)

    bed = tmp_path / "funnel.bed"
    bed.write_text("15\t250\t450\tGENEA\n1\t50\t150\tGENEB\n")  # excludes 1:200, X, MT
    stubs = {"vep": StubVep(), "gnomad": StubGnomad()}
    opts = RetrieveOptions(cache_root=tmp_path / "cache", funnel_bed=bed, sources=("vep", "gnomad"))
    manifest = run_retrieve(run, opts, retrievers=lambda http, o: stubs)

    m = json.loads(manifest.read_text())
    c = m["counts"]
    assert c["rows_in"] == 7 and c["rows_in_scope"] == 4 and c["unique_variants"] == 4
    assert c["vep_variants_queried"] == 4 and c["vep_records"] == 4
    # gnomAD prefilter: HIGH impact only on chrom 15 → 3 variants (15:300 G, 15:300 A, 15:400)
    assert c["gnomad_prefilter_selected"] == 3 and c["gnomad_variants_queried"] == 3
    assert c["gnomad_variants_with_record"] == 1 and c["gnomad_records"] == 1
    assert c["evidence_records"] == 5 and c["table_rows"] == 4
    assert m["params"]["source_versions"] == {"vep": "Ensembl VEP stub", "gnomad": "gnomad_r4 stub"}
    assert stubs["gnomad"].calls[0][0] == ("15", 300, "T", "A")  # genome order, alt-sorted

    rows = list(read_annotated(run))
    assert len(rows) == 4
    by = {(r["chrom"], r["pos"], r["alt"]): r for r in rows}
    assert by[("1", "100", "G")]["funnel_region"] == "GENEB"
    assert by[("1", "100", "G")]["impact_any_coding"] == "MODIFIER"
    assert by[("1", "100", "G")]["gnomad_af"] == ""  # not prefiltered in
    assert by[("15", "300", "A")]["gnomad_af"] == "1e-05"
    assert by[("15", "300", "A")]["evidence_ids"] == "vep:15:300:T:A;gnomad:15-300-T-A"
    assert by[("15", "400", "A")]["evidence_ids"] == "vep:15:400:ATT:A"  # gnomAD absent → no id
    # ingest columns carried through
    assert by[("15", "400", "A")]["gt"] == "1/1" and by[("15", "400", "A")]["quality_flag"] == ""

    idx = json.loads((run / "02_retrieve" / "evidence" / "index.json").read_text())
    assert set(idx) == {"vep:1:100:A:G", "vep:15:300:T:G", "vep:15:300:T:A", "vep:15:400:ATT:A", "gnomad:15-300-T-A"}


def test_prefilter_rules():
    keys = [("1", 1, "A", "G"), ("1", 2, "A", "G"), ("1", 3, "A", "G"), ("1", 4, "A", "G")]
    cols = {
        keys[0]: {"impact_any_coding": "HIGH", "vep_gnomade_af": "0.2"},        # common → skip
        keys[1]: {"impact_any_coding": "HIGH", "vep_gnomade_af": "", "vep_gnomadg_af": "1e-5"},  # rare → query
        keys[2]: {"impact_any_coding": "MODIFIER", "vep_gnomade_af": ""},      # not consequential → skip
        keys[3]: {"impact_any_coding": "SPLICE"},                              # absent from gnomAD → query
    }
    opts = RetrieveOptions(cache_root=Path("/dev/null"))
    assert gnomad_prefilter(keys, cols, opts) == [keys[1], keys[3]]


def test_select_keys_without_funnel_dedupes_and_orders():
    rows = [
        {"chrom": "X", "pos": "5", "ref": "A", "alt": "G"},
        {"chrom": "2", "pos": "9", "ref": "A", "alt": "T"},
        {"chrom": "2", "pos": "9", "ref": "A", "alt": "T"},
        {"chrom": "10", "pos": "1", "ref": "C", "alt": "G"},
    ]
    keys, region, counts = select_keys(rows, None)
    assert keys == [("2", 9, "A", "T"), ("10", 1, "C", "G"), ("X", 5, "A", "G")]
    assert counts == {"rows_in": 4, "rows_in_scope": 4, "unique_variants": 3}
    assert all(v == "" for v in region.values())
