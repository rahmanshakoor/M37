"""Stage 8 over a synthetic run directory built from public variants only."""

import csv
import json
from pathlib import Path

import pytest

from engine.submit.run import (MAX_ROWS, dry_run, is_artefact_family, plan, run_submit, write_csv)

SCORER = Path(__file__).resolve().parents[2] / "space" / "evaluation.py"

CFTR_A = "7:117559590:ATCT:A"   # p.Phe508del
CFTR_B = "7:117587778:G:T"      # p.Gly542Ter
TP53 = "17:7675088:C:T"         # p.Arg175His
MTHFR = "1:11796321:G:A"        # common


def _variant(key, impact="HIGH", af="1e-05", **kw):
    d = {"key": key, "impact": impact, "consequence": "stop_gained", "hgvsc": "c.1G>T", "hgvsp": "p.X1Ter",
         "af_used": af, "af_source": "gnomad_af", "clinvar_pathogenicity": "", "clinvar_stars": "", "clinvar_vcv": "",
         "gt": "0/1", "evidence_ids": []}
    d.update(kw)
    return d


def _run_dir(tmp_path: Path, *, with_rank=True, with_chains=True, extra=()) -> Path:
    run = tmp_path / "run"
    cands = [
        {"candidate_id": "CFTR:comphet", "gene_symbol": "CFTR", "gene_id": "ENSG1", "model": "comphet", "priority": 1,
         "phase": {"status": "unknown", "evidence": "no shared PID"}, "rule_hits": [], "caveats": [],
         "variants": [_variant(CFTR_A, "MODERATE", "0.012", hgvsp="p.Phe508del", clinvar_pathogenicity="Pathogenic"),
                      _variant(CFTR_B, "HIGH", "0.00036", hgvsp="p.Gly542Ter", clinvar_pathogenicity="Pathogenic")]},
        {"candidate_id": "TP53:het_single", "gene_symbol": "TP53", "gene_id": "ENSG2", "model": "het_single", "priority": 2,
         "phase": {"status": "not_applicable", "evidence": ""}, "rule_hits": [], "caveats": [],
         "variants": [_variant(TP53, "MODERATE", "4e-06", hgvsp="p.Arg175His", clinvar_pathogenicity="Pathogenic")]},
        {"candidate_id": "MTHFR:hom", "gene_symbol": "MTHFR", "gene_id": "ENSG3", "model": "hom", "priority": 3,
         "phase": {"status": "not_applicable", "evidence": ""}, "rule_hits": [], "caveats": [],
         "variants": [_variant(MTHFR, "MODERATE", "0", gt="1/1", hgvsp="p.Ala222Val")]},
        {"candidate_id": "HLA-DRB1:comphet", "gene_symbol": "HLA-DRB1", "gene_id": "ENSG4", "model": "comphet", "priority": 4,
         "phase": {"status": "unknown", "evidence": ""}, "rule_hits": [], "caveats": [],
         "variants": [_variant("6:32581836:G:A", "MODERATE"), _variant("6:32589725:G:GC", "HIGH")]},
    ] + list(extra)
    (run / "03_filter").mkdir(parents=True)
    (run / "03_filter" / "candidates.json").write_text(json.dumps({"candidates": cands, "config": {}, "counts": {}}))
    if with_rank:
        (run / "04_rank").mkdir()
        (run / "04_rank" / "joined.json").write_text(json.dumps({"candidates": [
            {"candidate_id": "CFTR:comphet", "exomiser_rank": 1}, {"candidate_id": "TP53:het_single", "exomiser_rank": 3},
            {"candidate_id": "HLA-DRB1:comphet", "exomiser_rank": 2}, {"candidate_id": "MTHFR:hom", "exomiser_rank": None}]}))
    if with_chains:
        (run / "05_reason" / "chains").mkdir(parents=True)
        for cid, key, cls in (("MTHFR:hom", MTHFR, "benign"), ("TP53:het_single", TP53, "pathogenic")):
            (run / "05_reason" / "chains" / f"{cid}.json").write_text(json.dumps(
                {"candidate_id": cid, "variants": [{"key": key, "classification": cls, "criteria": [], "summary": ""}],
                 "phase_statement": "", "mechanism_hypothesis": "", "limits": [], "what_would_change_the_call": [], "literature": []}))
    return run


def test_plan_lead_pair_secondary_and_benign_skip(tmp_path: Path):
    p = plan(_run_dir(tmp_path))
    assert [r.candidate_id for r in p.rows] == ["CFTR:comphet", "TP53:het_single", "HLA-DRB1:comphet"]
    assert p.rows[0].variants == [("7", 117587778, "G", "T"), ("7", 117559590, "ATCT", "A")]  # HIGH allele first
    assert [r.finding_type for r in p.rows] == ["primary", "secondary", "primary"]
    assert p.skipped == [{"candidate_id": "MTHFR:hom", "why": "stage-5 chain classified benign/likely benign"}]
    assert [r.epcr for r in p.rows] == [0.95, 0.90, 0.85]
    assert "ACMG" not in p.rows[0].notes and "ClinVar Pathogenic" in p.rows[0].notes and "phase unknown" in p.rows[0].notes


def test_artefact_families_go_last_and_ranks_order_backups(tmp_path: Path):
    extra = [{"candidate_id": "PKD1:comphet", "gene_symbol": "PKD1", "gene_id": "ENSG5", "model": "comphet", "priority": 5,
              "phase": {"status": "unknown", "evidence": ""}, "rule_hits": [], "caveats": [],
              "variants": [_variant("16:2100000:A:G"), _variant("16:2100100:C:T")]}]
    p = plan(_run_dir(tmp_path, extra=extra))
    ids = [r.candidate_id for r in p.rows]
    assert ids[0] == "CFTR:comphet"
    assert ids.index("HLA-DRB1:comphet") == len(ids) - 1  # ranked #2 by Exomiser, but an artefact family: last
    assert ids.index("TP53:het_single") < ids.index("PKD1:comphet")  # ranked (3) before unranked
    assert is_artefact_family("HLA-A") and is_artefact_family("MUC4") and is_artefact_family("OR2T4")
    assert not is_artefact_family("ORC1") and not is_artefact_family("CFTR")


def test_also_pairs_follow_the_lead(tmp_path: Path):
    p = plan(_run_dir(tmp_path), also_pairs=[(CFTR_B, "7:117530975:G:A")])
    assert p.rows[1].candidate_id == "CFTR:comphet:alt" and len(p.rows[1].variants) == 2
    assert p.rows[1].epcr == 0.90 and p.rows[2].epcr == 0.85


def test_csv_is_in_the_scorers_conventions(tmp_path: Path):
    p = plan(_run_dir(tmp_path))
    out = write_csv(p, "PUBLIC01", tmp_path / "s.csv")
    rows = list(csv.DictReader(open(out)))
    assert rows[0]["chrom_1"] == "chr7" and rows[0]["chrom_2"] == "chr7" and rows[0]["pos_2"] == "117559590"
    assert rows[1]["chrom_2"] == "" and rows[1]["finding_type"] == "secondary"
    epcrs = [float(r["epcr"]) for r in rows]
    assert epcrs == sorted(epcrs, reverse=True) and len(set(epcrs)) == len(epcrs)
    assert all(0 < e <= 1 for e in epcrs) and len(rows) <= MAX_ROWS
    assert all("," not in r["notes"] for r in rows)


@pytest.mark.skipif(not SCORER.exists(), reason="challenge scorer not beside the checkout")
def test_dry_run_against_the_real_scorer(tmp_path: Path):
    run = _run_dir(tmp_path)
    csv_path = run_submit(run, proband_id="PUBLIC01", scorer=SCORER)
    m = json.loads((run / "08_submit" / "manifest.json").read_text())
    d = m["params"]["dry_run"]
    assert d["if_lead_is_the_key"] == {"rank_points": 100.0, "f_max": 1.0, "full_match_rank": 1, "partial_match_rank": None,
                                       "f_max_threshold": 0.95, "rows": 3}
    assert d["if_only_first_allele_is_right"]["rank_points"] == 50.0
    assert d["if_key_is_absent"]["rank_points"] == 0.0 and d["if_key_is_absent"]["f_max"] == 0.0
    # a split pair would only earn half credit — proven with the scorer, not assumed
    split = dry_run(csv_path, SCORER, "PUBLIC01", [("7", 117587778, "G", "T"), ("7", 999, "A", "C")])
    assert split["rank_points"] == 50.0 and split["full_match_rank"] is None and split["partial_match_rank"] == 1


def test_clinvar_benign_pairs_and_male_x_comphets_take_no_row(tmp_path: Path):
    benign = {"candidate_id": "SERPINA1:comphet", "gene_symbol": "SERPINA1", "gene_id": "ENSG6", "model": "comphet", "priority": 5,
              "phase": {"status": "unknown", "evidence": ""}, "rule_hits": [], "caveats": [],
              "variants": [_variant("14:94379548:C:CTA", clinvar_pathogenicity="Benign", clinvar_stars="1"),
                           _variant("14:94379541:G:GA", clinvar_pathogenicity="Likely_benign", clinvar_stars="1")]}
    x_pair = {"candidate_id": "DMD:comphet", "gene_symbol": "DMD", "gene_id": "ENSG7", "model": "comphet", "priority": 6,
              "phase": {"status": "unknown", "evidence": ""}, "rule_hits": [], "caveats": [],
              "variants": [_variant("X:32000000:G:A"), _variant("X:32000500:C:T")]}
    run = _run_dir(tmp_path, extra=[benign, x_pair])
    p = plan(run)
    assert "SERPINA1:comphet" not in [r.candidate_id for r in p.rows]
    assert "DMD:comphet" in [r.candidate_id for r in p.rows]  # sex unknown: a stage-3 comphet stands
    assert {"candidate_id": "SERPINA1:comphet", "why": "every allele ClinVar Benign/Likely_benign"} in p.skipped
    (run / "01_ingest").mkdir()
    (run / "01_ingest" / "manifest.json").write_text(json.dumps({"params": {"sex": "male"}}))
    p = plan(run)
    assert "DMD:comphet" not in [r.candidate_id for r in p.rows]
    assert {"candidate_id": "DMD:comphet", "why": "compound heterozygote on X in a male"} in p.skipped
    # one benign allele beside a pathogenic one is not a benign pair
    mixed = dict(benign, candidate_id="MIX:comphet", gene_symbol="MIX",
                 variants=[_variant("14:94379548:C:CTA", clinvar_pathogenicity="Benign"), _variant("14:94379541:G:GA")])
    assert "MIX:comphet" in [r.candidate_id for r in plan(_run_dir(tmp_path / "b", extra=[mixed])).rows]


def test_without_rank_or_chains_still_writes(tmp_path: Path):
    run = _run_dir(tmp_path, with_rank=False, with_chains=False)
    csv_path = run_submit(run, proband_id="PUBLIC01", scorer=None)
    rows = list(csv.DictReader(open(csv_path)))
    # MTHFR kept (no chain says benign); TP53 is primary (nobody classified it, so not an incidental finding)
    assert [r["finding_type"] for r in rows] == ["primary", "primary", "primary", "primary"]
