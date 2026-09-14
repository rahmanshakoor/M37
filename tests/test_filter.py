"""Stage 3 tests — pure rules over a synthetic, public-only annotated table.

``tests/fixtures/filter/annotated_rows.json`` holds fourteen rows: five real public
variants (MTHFR rs1801133, CFTR p.Phe508del and p.Gly542Ter, TP53 p.Arg213Ter, plus
an exact duplicate) with the projections the stage-2 retrievers produce for them,
and nine invented rows that each trip one rule — a cis pair sharing a PID and
PGT inside the public BUB1B locus (both dropped as ``phase:cis``; the HIGH member is
recorded as het_single-eligible), a homozygote, a non-carrier ``0/0``, a common
ClinVar-Benign variant, a synonymous variant, a splice-region variant with SpliceAI
0.4, an X hemizygote, a mitochondrial call. Every genotype is invented; nothing
comes from a patient, and no gene outside the public four (CFTR, TP53, MTHFR, BUB1B)
is named — the X and MT rows use placeholder gene labels.

The table is materialised into a temporary run directory exactly as stage 2 writes
it (the real header from the retriever classes, gzip with ``mtime=0``), so the
tests exercise ``read_annotated`` → rules → the three output files → the manifest.
"""

from __future__ import annotations

import gzip
import io
import json
import os
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from engine.cli import main
from engine.filter import rules
from engine.filter.config import DEFAULT_CONFIG, IMPACT_RANK, FilterConfig
from engine.filter.rules import (
    ANNOTATED_COLUMNS, DECISION_COLUMNS, HET_SINGLE_ELIGIBLE, REQUIRED_COLUMNS, SHORTLIST_EXTRA_COLUMNS, Hit,
    alt_haplotype, clinvar_in, genotype_model, opposite_pgt, pair_phase, rule_family, screen_row, screen_rows,
    zygosity,
)
from engine.filter.run import STAGE_DIR, read_candidates, read_decisions, read_shortlist, run_filter
from engine.ingest import COLUMNS as INGEST_COLUMNS
from engine.retrieve.clinvar import ClinvarRetriever
from engine.retrieve.gnomad import GnomadRetriever
from engine.retrieve.vep import IMPACT_ORDER, VepRetriever, impact_any_coding, select_transcript

FIX = Path(__file__).parent / "fixtures" / "filter" / "annotated_rows.json"

MTHFR_COMMON = "1:11796321:G:A"
MTHFR_SYN = "1:11796400:C:T"
MTHFR_SPLICE = "1:11797000:A:G"
F508DEL = "7:117559590:ATCT:A"
G542X = "7:117587778:G:T"
CIS_A = "15:40170100:G:A"
CIS_B = "15:40170150:C:T"
NON_CARRIER = "15:40170200:T:C"
BENIGN = "15:40175000:C:T"
HOM = "15:40180000:CA:C"
R213X = "17:7674894:G:A"
X_HEMI = "X:100000000:A:G"
MITO = "MT:1000:A:G"

PUBLIC_GENES = {"MTHFR", "CFTR", "TP53", "BUB1B"}
PLACEHOLDER_GENES = {"GENEX", "GENEMT"}

CANDIDATE_FIELDS = {"candidate_id", "gene_symbol", "gene_id", "model", "priority", "variants", "phase", "rule_hits", "caveats"}
VARIANT_FIELDS = {
    "key", "consequence", "impact", "hgvsc", "hgvsp", "transcript_id", "mane", "gt", "ad", "dp", "gq", "quality_flag",
    "af_used", "af_source", "gnomad_nhom", "clinvar_vcv", "clinvar_pathogenicity", "clinvar_stars", "spliceai_ds_max",
    "sift_pred", "polyphen_pred", "caveats", "evidence_ids",
}
ELIGIBLE_NOTE = "het_single_eligible"


# ---------------------------------------------------------------- helpers

def fixture_rows() -> list[dict[str, str]]:
    rows = json.loads(FIX.read_text())["rows"]
    return [{c: r.get(c, "") for c in ANNOTATED_COLUMNS} for r in rows]


def write_table(run: Path, cols: list[str], rows: list[dict[str, str]]) -> Path:
    (run / "02_retrieve").mkdir(parents=True, exist_ok=True)
    path = run / "02_retrieve" / "variants.annotated.tsv.gz"
    with gzip.GzipFile(path, "wb", mtime=0) as raw, io.TextIOWrapper(raw, encoding="utf-8") as out:
        out.write("\t".join(cols) + "\n")
        for r in rows:
            out.write("\t".join(r[c] for c in cols) + "\n")
    return path


def make_run(tmp_path: Path, rows: list[dict[str, str]] | None = None, name: str = "run") -> Path:
    run = tmp_path / name
    write_table(run, list(ANNOTATED_COLUMNS), rows if rows is not None else fixture_rows())
    return run


def write_config(tmp_path: Path, **overrides: dict) -> Path:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    for section, kv in overrides.items():
        raw[section].update(kv)
    p = tmp_path / "filter.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def decisions_by_key(run: Path) -> dict[str, dict[str, str]]:
    """First decision per key (the duplicate row is reached by position instead)."""
    out: dict[str, dict[str, str]] = {}
    for d in read_decisions(run):
        out.setdefault(f"{d['chrom']}:{d['pos']}:{d['ref']}:{d['alt']}", d)
    return out


def row(**kw: str) -> dict[str, str]:
    """A hand-made annotated row: a rare heterozygous BUB1B nonsense unless overridden
    (BUB1B, ENSG00000156970, GRCh38 15:40160984-40221137; the position is invented)."""
    base = {c: "" for c in ANNOTATED_COLUMNS}
    base.update({
        "chrom": "15", "pos": "40170100", "ref": "G", "alt": "A", "gt": "0/1", "ad": "15,14", "dp": "29", "gq": "99",
        "gene_symbol": "BUB1B", "gene_id": "ENSG00000156970", "consequence": "stop_gained", "impact": "HIGH",
        "impact_any_coding": "HIGH", "spliceai_ds_max": "0", "gnomad_af": "1e-05", "gnomad_nhom": "0",
    })
    base.update(kw)
    return base


def with_(cfg: FilterConfig, section: str, **kv) -> FilterConfig:
    """A config with one section's fields overridden, for the rule-level tests."""
    return cfg.model_copy(update={section: getattr(cfg, section).model_copy(update=kv)})


def messages(exc: BaseException | None) -> list[str]:
    out = []
    while exc is not None:
        out.append(str(exc))
        exc = exc.__cause__
    return out


@pytest.fixture(scope="module")
def cfg() -> FilterConfig:
    return FilterConfig.load(DEFAULT_CONFIG)


# ---------------------------------------------------------------- columns and fixture hygiene

def test_header_is_built_from_the_retriever_classes():
    expected = (list(INGEST_COLUMNS) + ["funnel_region"] + list(VepRetriever.columns) + list(ClinvarRetriever.columns)
                + list(GnomadRetriever.columns) + ["evidence_ids"])
    assert list(ANNOTATED_COLUMNS) == expected
    # Every column the rules read is one the classes write.
    for name in ("GENE_SYMBOL", "GENE_ID", "IMPACT", "IMPACT_ANY_CODING", "SPLICEAI_DS_MAX", "CLINVAR_PATHOGENICITY",
                 "CLINVAR_STARS", "GNOMAD_NHOM", "PID", "PGT", "GT", "DP", "GQ", "QUALITY_FLAG"):
        assert getattr(rules, name) in ANNOTATED_COLUMNS
    assert set(REQUIRED_COLUMNS) <= set(INGEST_COLUMNS)
    assert DECISION_COLUMNS[:11] == ("chrom", "pos", "ref", "alt", "gene_symbol", "kept", "rule", "model", "caveats",
                                     "af_used", "af_source")


def test_a_renamed_upstream_column_breaks_stage3_at_import(monkeypatch: pytest.MonkeyPatch):
    """Every column the rules read is checked at import against the class that writes
    it: rename ``gnomad_af`` upstream and ``engine.filter.rules`` refuses to import,
    naming the column and the owner — rather than reading '' for every row."""
    import importlib
    import sys

    renamed = tuple("gnomad_n_hom" if c == "gnomad_nhom" else c for c in GnomadRetriever.columns)
    monkeypatch.setattr(GnomadRetriever, "columns", renamed)
    monkeypatch.delitem(sys.modules, "engine.filter.rules")
    with pytest.raises(ImportError, match="column 'gnomad_nhom' is not written by"):
        importlib.import_module("engine.filter.rules")
    monkeypatch.undo()
    monkeypatch.delitem(sys.modules, "engine.filter.rules", raising=False)
    fresh = importlib.import_module("engine.filter.rules")  # a fresh, working module under the real columns
    assert fresh is not rules and fresh.GNOMAD_NHOM == rules.GNOMAD_NHOM == "gnomad_nhom"
    monkeypatch.setitem(sys.modules, "engine.filter.rules", rules)  # the module the rest of the suite imported stays the one in use


def test_impact_rank_follows_the_stage2_order():
    """The priority sort ranks impacts as VEP orders them; SPLICE (stage 2's marker for a
    splice term below MODERATE) sits with MODERATE; absent ranks below everything."""
    assert [IMPACT_RANK[i] for i in IMPACT_ORDER] == [3, 2, 1, 0]
    assert IMPACT_RANK["SPLICE"] == IMPACT_RANK["MODERATE"] and IMPACT_RANK[""] == -1


def test_fixture_rows_use_only_known_columns_and_public_material():
    doc = json.loads(FIX.read_text())
    assert "patient" in doc["_provenance"]["public_material_only"].lower()
    assert len(doc["rows"]) == 14
    for r in doc["rows"]:
        unknown = set(r) - set(ANNOTATED_COLUMNS) - {"_note"}
        assert not unknown, unknown
        assert r["_note"].startswith(("real public variant", "synthetic", "exact duplicate"))
        assert r["gene_symbol"] in PUBLIC_GENES | PLACEHOLDER_GENES, r["gene_symbol"]
        # the row is internally consistent: source spelling, evidence ids and key agree
        assert r["chrom_source"] == ("chrM" if r["chrom"] == "MT" else "chr" + r["chrom"])
        for rid in r.get("evidence_ids", "").split(";"):
            if rid.startswith("vep:"):
                assert rid == f"vep:{r['chrom']}:{r['pos']}:{r['ref']}:{r['alt']}"
            elif rid.startswith("gnomad:"):
                assert rid == f"gnomad:{r['chrom']}-{r['pos']}-{r['ref']}-{r['alt']}"


# ---------------------------------------------------------------- end to end

def test_every_row_gets_a_decision_with_the_contract_rule(tmp_path: Path):
    run = make_run(tmp_path)
    manifest = run_filter(run, DEFAULT_CONFIG)
    decisions = list(read_decisions(run))
    assert len(decisions) == 14
    assert list(decisions[0].keys()) == list(DECISION_COLUMNS)
    by = decisions_by_key(run)

    # rule 3 on VEP's gnomAD copy (the API was not queried for a common variant)
    d = by[MTHFR_COMMON]
    assert (d["kept"], d["rule"], d["model"]) == ("0", "rarity:af=0.3227", "")
    assert (d["af_used"], d["af_source"]) == ("0.3227", "vep_gnomade_af")
    assert d["rule_hits"] == "consequence:MODERATE;rarity:af=0.3227"
    # the second copy of the same key is a duplicate, whatever else it is
    assert decisions[1]["rule"] == "duplicate" and decisions[1]["kept"] == "0"
    # rule 1
    assert by[MTHFR_SYN]["rule"] == "consequence:LOW"
    # kept by SpliceAI, then alone in its gene and not HIGH → no model
    d = by[MTHFR_SPLICE]
    assert d["rule"] == "model:het_single" and d["kept"] == "0"
    assert d["rule_hits"] == "consequence:splice=0.4;rarity:af=0;model:het_single"
    assert (d["af_used"], d["af_source"]) == ("0", "absent")
    # ClinVar P above the recessive ceiling: rescued, and the rescue is a recorded hit
    d = by[F508DEL]
    assert (d["kept"], d["rule"], d["model"]) == ("1", "", "comphet")
    assert d["rule_hits"] == ("consequence:MODERATE;rarity:af=0.011931254341569912;"
                              "clinvar:always_keep=Pathogenic;model:comphet")
    d = by[G542X]
    assert (d["kept"], d["model"]) == ("1", "comphet")
    assert d["rule_hits"] == "consequence:HIGH;rarity:af=0.00036287621268888174;model:comphet"
    # phase: same PID → cis → both dropped (rule 5); the HIGH member's dominant eligibility is on record
    d = by[CIS_A]
    assert (d["kept"], d["rule"], d["model"]) == ("0", "phase:cis", "")
    assert d["rule_hits"] == f"consequence:HIGH;rarity:af=1e-05;{HET_SINGLE_ELIGIBLE};phase:cis"
    d = by[CIS_B]
    assert (d["kept"], d["rule"]) == ("0", "phase:cis")
    assert d["rule_hits"] == "consequence:MODERATE;rarity:af=2e-05;phase:cis"
    # pre-rules
    assert by[NON_CARRIER]["rule"] == "genotype:gt=0/0"
    # rule 2 fires before rule 3 on a common benign variant
    assert by[BENIGN]["rule"] == "clinvar_benign:Benign,stars=2"
    # models
    assert (by[HOM]["kept"], by[HOM]["model"]) == ("1", "hom")
    assert (by[X_HEMI]["kept"], by[X_HEMI]["model"]) == ("1", "hemi")
    assert (by[MITO]["kept"], by[MITO]["model"], by[MITO]["af_source"]) == ("1", "mito", "absent")
    d = by[R213X]
    assert (d["kept"], d["model"]) == ("1", "het_single")
    assert d["caveats"] == "dp<10;gq<20;flagged:LowQual"
    # nothing kept has a rule; nothing dropped has a model
    for d in decisions:
        assert (d["kept"] == "1") == (d["rule"] == "")
        assert (d["kept"] == "1") == (d["model"] != "")

    m = json.loads(manifest.read_text())
    c = m["counts"]
    assert c["rows_in"] == 14 and c["kept"] == 6 and c["dropped"] == 8 and c["rows_after_row_rules"] == 9
    assert c["dropped_by_rule"] == {
        "clinvar_benign": 1, "consequence:LOW": 1, "duplicate": 1, "genotype": 1, "model:het_single": 1,
        "phase:cis": 2, "rarity": 1,
    }
    assert c["candidates"] == 5 and c["genes_with_candidates"] == 5
    assert c["candidates_by_model"] == {"hom": 1, "comphet": 1, "hemi": 1, "mito": 1, "het_single": 1}
    assert c["rows_with_caveats"] == 1
    assert c["phase"] == {"cis_rows_dropped": 2, "cis_rows_dropped_het_single_eligible": 1,
                          "shared_pid_opposite_pgt_pairs": 0, "unknown": 1}
    assert c["rarity"] == {"rows_by_af_source": {"absent": 3, "gnomad_af": 8, "vep_gnomade_af": 3},
                           "af_fallback_first_masked_rows": 0}


def test_candidates_json_shape_priority_and_phase(tmp_path: Path):
    run = make_run(tmp_path)
    run_filter(run, DEFAULT_CONFIG)
    doc = read_candidates(run)
    assert set(doc) == {"candidates", "config", "counts"}
    cands = doc["candidates"]
    assert [c["candidate_id"] for c in cands] == ["CFTR:comphet", "TP53:het_single", "BUB1B:hom", "GENEX:hemi", "GENEMT:mito"]
    assert [c["priority"] for c in cands] == [1, 2, 3, 4, 5]
    for c in cands:
        assert set(c) == CANDIDATE_FIELDS
        for v in c["variants"]:
            assert set(v) == VARIANT_FIELDS
            assert isinstance(v["caveats"], list) and isinstance(v["evidence_ids"], list)
        if c["model"] != "comphet":
            assert len(c["variants"]) == 1 and c["phase"]["status"] == "not_applicable"

    cftr = cands[0]
    assert cftr["gene_symbol"] == "CFTR" and cftr["gene_id"] == "ENSG00000001626" and cftr["model"] == "comphet"
    assert [v["key"] for v in cftr["variants"]] == [F508DEL, G542X]
    assert cftr["phase"] == {"status": "unknown", "evidence": "no shared PID; 28.2 kb apart"}
    assert cftr["rule_hits"] == [
        "consequence:MODERATE", "rarity:af=0.011931254341569912", "clinvar:always_keep=Pathogenic", "model:comphet",
        "consequence:HIGH", "rarity:af=0.00036287621268888174",
    ]
    assert cftr["caveats"] == []
    v = cftr["variants"][0]
    assert v["consequence"] == "inframe_deletion" and v["impact"] == "MODERATE"
    assert v["hgvsp"] == "ENSP00000003084.6:p.Phe508del" and v["mane"] == "NM_000492.4"
    assert (v["gt"], v["ad"], v["dp"], v["gq"], v["quality_flag"]) == ("0/1", "18,16", "34", "99", "")
    assert (v["af_used"], v["af_source"], v["gnomad_nhom"]) == ("0.011931254341569912", "gnomad_af", "58")
    assert (v["clinvar_vcv"], v["clinvar_pathogenicity"], v["clinvar_stars"]) == ("VCV000007105", "Pathogenic", "4")
    assert v["evidence_ids"] == ["vep:7:117559590:ATCT:A", "clinvar:VCV000007105", "gnomad:7-117559590-ATCT-A"]
    v = cftr["variants"][1]
    assert v["consequence"] == "stop_gained" and v["hgvsp"] == "ENSP00000003084.6:p.Gly542Ter"

    tp53 = cands[1]
    assert tp53["model"] == "het_single" and tp53["caveats"] == ["dp<10", "gq<20", "flagged:LowQual"]
    assert tp53["variants"][0]["caveats"] == ["dp<10", "gq<20", "flagged:LowQual"]
    assert tp53["variants"][0]["quality_flag"] == "LowQual"

    hom = cands[2]
    assert hom["gene_symbol"] == "BUB1B" and hom["gene_id"] == "ENSG00000156970"
    assert hom["variants"][0]["gt"] == "1/1" and hom["variants"][0]["gnomad_nhom"] == "3"
    assert cands[3]["model"] == "hemi" and cands[4]["model"] == "mito"
    assert cands[4]["variants"][0]["af_used"] == "0" and cands[4]["variants"][0]["af_source"] == "absent"

    assert doc["config"] == FilterConfig.load(DEFAULT_CONFIG).as_params()
    assert doc["counts"]["candidates"] == 5


def test_shortlist_has_every_annotated_column_plus_the_model_columns(tmp_path: Path):
    run = make_run(tmp_path)
    run_filter(run, DEFAULT_CONFIG)
    rows = list(read_shortlist(run))
    assert list(rows[0].keys()) == list(ANNOTATED_COLUMNS) + list(SHORTLIST_EXTRA_COLUMNS)
    keys = [f"{r['chrom']}:{r['pos']}:{r['ref']}:{r['alt']}" for r in rows]
    assert keys == [F508DEL, G542X, HOM, R213X, X_HEMI, MITO]  # input order
    by = dict(zip(keys, rows))
    assert by[F508DEL]["partner_keys"] == G542X and by[G542X]["partner_keys"] == F508DEL
    assert by[F508DEL]["phase"] == "unknown" and by[F508DEL]["candidate_id"] == "CFTR:comphet"
    # the phase column says the same as candidates.json for every model
    for k in (HOM, R213X, X_HEMI, MITO):
        assert by[k]["partner_keys"] == "" and by[k]["phase"] == "not_applicable"
    assert by[HOM]["candidate_id"] == "BUB1B:hom" and by[R213X]["candidate_id"] == "TP53:het_single"
    assert by[R213X]["caveats"] == "dp<10;gq<20;flagged:LowQual"
    assert by[F508DEL]["hgvsp"] == "ENSP00000003084.6:p.Phe508del"  # annotated columns carried through


def test_manifest_records_inputs_thresholds_and_outputs(tmp_path: Path):
    run = make_run(tmp_path)
    cfg_path = write_config(tmp_path)
    manifest = run_filter(run, cfg_path)
    assert manifest == run / STAGE_DIR / "manifest.json"
    m = json.loads(manifest.read_text())
    assert m["stage"] == "filter"
    assert set(m["inputs"]) == {"variants_annotated", "config"}
    assert m["inputs"]["variants_annotated"]["sha256"] and m["inputs"]["config"]["sha256"]
    assert m["tools"]["python"]
    p = m["params"]
    assert p["config_path"] == str(cfg_path)
    for section in ("rarity", "consequence", "clinvar", "quality", "phase"):
        assert p[section] == FilterConfig.load(cfg_path).as_params()[section]
    for key in ("rescue_max_af", "af_fallback_pick"):
        assert key in p["rarity"]
    assert "rescue_min_stars" in p["clinvar"] and "trust_pgt" in p["phase"]
    assert p["columns_by_source"]["vep"] == list(VepRetriever.columns)
    assert p["columns_by_source"]["clinvar"] == list(ClinvarRetriever.columns)
    assert p["columns_by_source"]["gnomad"] == list(GnomadRetriever.columns)
    assert p["af_source_columns_present"] == ["gnomad_af", "vep_gnomade_af", "vep_gnomadg_af"]
    assert p["required_columns"] == list(REQUIRED_COLUMNS)
    assert p["model_rank"] == ["hom", "comphet", "hemi", "mito", "het_single"]
    assert set(m["outputs"]) == {"decisions", "shortlist", "candidates"}
    for o in m["outputs"].values():
        assert Path(o["path"]).exists() and o["sha256"]
    # the one thing worth a note in this run: a dominant-eligible het hidden behind its cis partner
    assert len(m["notes"]) == 1 and m["notes"][0].startswith("1 heterozygous row(s) dropped as phase:cis")
    assert ELIGIBLE_NOTE in m["notes"][0]


def test_outputs_are_byte_identical_across_reruns(tmp_path: Path):
    a = make_run(tmp_path, name="a")
    b = make_run(tmp_path, name="b")
    run_filter(a, DEFAULT_CONFIG)
    run_filter(b, DEFAULT_CONFIG)
    for name in ("decisions.tsv.gz", "shortlist.tsv.gz", "candidates.json"):
        assert (a / STAGE_DIR / name).read_bytes() == (b / STAGE_DIR / name).read_bytes(), name


def test_missing_stage2_table_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="stage 2"):
        run_filter(tmp_path / "empty", DEFAULT_CONFIG)


def test_config_is_strict():
    with pytest.raises(FileNotFoundError):
        FilterConfig.load(Path("/nonexistent/filter.yaml"))
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["rarity"]["recesive_max_af"] = 0.5  # typo must not silently do nothing
    with pytest.raises(ValueError):
        FilterConfig(**raw)


@pytest.mark.parametrize("section,key,value", [
    ("rarity", "af_source_order", ["gnomad_AF", "vep_gnomade_AF"]),   # case: would read AF 0 for every row
    ("rarity", "af_source_order", ["gene_symbol_af"]),
    ("consequence", "keep_impacts", ["High", "MODERATE"]),
    ("clinvar", "always_keep", ["pathogenic"]),
    ("rarity", "af_fallback_pick", "largest"),
])
def test_config_vocabularies_are_checked_against_the_producing_code(section, key, value):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw[section][key] = value
    with pytest.raises(ValueError, match=key):
        FilterConfig(**raw)


def test_shipped_defaults_are_the_contract_rules():
    """CONTRACTS.md as written: AF is the first non-empty source, a shared PID is cis.
    The other readings exist as switches and are echoed when used."""
    cfg = FilterConfig.load(DEFAULT_CONFIG)
    assert cfg.rarity.af_fallback_pick == "first" and cfg.phase.trust_pgt is False
    assert cfg.rarity.rescue_max_af == 0.05 and cfg.clinvar.rescue_min_stars == 1


def test_contract_yaml_block_loads_and_is_the_shipped_config():
    """The stage-3 ``yaml`` block in CONTRACTS.md is a complete FilterConfig (every key the
    strict model requires, no unknown key) and carries the same values as the shipped
    ``configs/filter.yaml`` — the contract and the defaults cannot drift apart unnoticed."""
    text = (Path(__file__).resolve().parents[1] / "CONTRACTS.md").read_text()
    stage3 = text.split("## Stage 3", 1)[1]
    block = stage3.split("```yaml\n", 1)[1].split("```", 1)[0]
    contract = FilterConfig(**yaml.safe_load(block))
    assert contract.as_params() == FilterConfig.load(DEFAULT_CONFIG).as_params()


def test_table_without_any_af_column_is_refused(tmp_path: Path):
    """No AF anywhere means every row passes rarity at AF 0 — a wrong config, not a result."""
    cols = [c for c in ANNOTATED_COLUMNS if c not in ("gnomad_af", "vep_gnomade_af", "vep_gnomadg_af")]
    run = tmp_path / "run"
    write_table(run, cols, [row()])
    with pytest.raises(ValueError, match="af_source_order"):
        run_filter(run, DEFAULT_CONFIG)
    assert not (run / STAGE_DIR / "candidates.json").exists()


@pytest.mark.parametrize("missing", ["gt", "pid", "pos", "quality_flag"])
def test_table_without_a_stage1_column_the_rules_read_is_refused(tmp_path: Path, missing: str):
    """Without ``gt`` every row would be dropped as a non-carrier and called a result;
    the table is refused by name instead, like a table with no AF column."""
    cols = [c for c in ANNOTATED_COLUMNS if c != missing]
    run = tmp_path / "run"
    write_table(run, cols, [row(clinvar_pathogenicity="Pathogenic", clinvar_stars="4")])
    with pytest.raises(ValueError, match=rf"lacks stage-1 column\(s\) \['{missing}'\]"):
        run_filter(run, DEFAULT_CONFIG)
    assert not (run / STAGE_DIR / "candidates.json").exists()


def test_empty_stage2_table_is_refused_with_its_cause(tmp_path: Path):
    """Stage 2 writes its header with the first row, so no variant in scope leaves an
    empty file; that is reported as such, not as a missing AF column. A header-only
    table is a table and filters to nothing."""
    run = tmp_path / "run"
    (run / "02_retrieve").mkdir(parents=True)
    with gzip.GzipFile(run / "02_retrieve" / "variants.annotated.tsv.gz", "wb", mtime=0):
        pass
    with pytest.raises(ValueError, match="stage 2 wrote no rows"):
        run_filter(run, DEFAULT_CONFIG)
    write_table(run, list(ANNOTATED_COLUMNS), [])
    m = json.loads(run_filter(run, DEFAULT_CONFIG).read_text())
    assert m["counts"]["rows_in"] == 0 and m["counts"]["candidates"] == 0
    assert list(read_decisions(run)) == [] and read_candidates(run)["candidates"] == []


# ---------------------------------------------------------------- every threshold is live

ROUND_TRIP = [
    ("rarity", "recessive_max_af", 0.5, MTHFR_COMMON, "kept", "1"),
    ("rarity", "dominant_max_af", 1e-7, R213X, "rule", "model:het_single"),
    ("rarity", "homozygote_max_nhom", 2, HOM, "rule", "model:hom:nhom=3"),
    ("rarity", "af_source_order", ["vep_gnomade_af", "gnomad_af", "vep_gnomadg_af"], F508DEL, "af_used", "0.01235"),
    ("consequence", "keep_impacts", ["HIGH"], MITO, "rule", "consequence:MODERATE"),
    ("consequence", "keep_splice_min_ds", 0.5, MTHFR_SPLICE, "rule", "consequence:SPLICE"),
    ("rarity", "rescue_max_af", 0.01, F508DEL, "rule", "rarity:af=0.011931254341569912"),
    ("clinvar", "always_keep", [], F508DEL, "rule", "rarity:af=0.011931254341569912"),
    ("clinvar", "always_keep", [], G542X, "rule", "model:het_single"),
    ("clinvar", "drop_if_benign_min_stars", 3, BENIGN, "rule", "rarity:af=0.021"),
    ("quality", "caveat_min_dp", 5, R213X, "caveats", "gq<20;flagged:LowQual"),
    ("quality", "caveat_min_gq", 10, R213X, "caveats", "dp<10;flagged:LowQual"),
    ("quality", "caveat_flagged_filters", False, R213X, "caveats", "dp<10;gq<20"),
]


@pytest.mark.parametrize("section,key,value,variant,column,expected", ROUND_TRIP,
                         ids=[f"{s}.{k}" for s, k, *_ in ROUND_TRIP])
def test_changing_a_threshold_changes_the_decision_and_is_echoed(tmp_path: Path, section, key, value, variant, column, expected):
    run = make_run(tmp_path)
    cfg_path = write_config(tmp_path, **{section: {key: value}})
    m = json.loads(run_filter(run, cfg_path).read_text())
    assert m["params"][section][key] == value
    assert read_candidates(run)["config"][section][key] == value
    assert decisions_by_key(run)[variant][column] == expected


def test_keep_impacts_without_moderate_still_keeps_clinvar_pathogenic(tmp_path: Path):
    run = make_run(tmp_path)
    run_filter(run, write_config(tmp_path, consequence={"keep_impacts": ["HIGH"]}))
    d = decisions_by_key(run)[F508DEL]
    assert d["kept"] == "1" and d["rule_hits"].startswith("consequence:clinvar=Pathogenic;")


def test_af_source_order_is_echoed_in_af_source(tmp_path: Path):
    run = make_run(tmp_path)
    run_filter(run, write_config(tmp_path, rarity={"af_source_order": ["vep_gnomadg_af", "gnomad_af"]}))
    d = decisions_by_key(run)[F508DEL]
    assert (d["af_used"], d["af_source"]) == ("0.007884", "vep_gnomadg_af")
    m = json.loads((run / STAGE_DIR / "manifest.json").read_text())
    assert m["params"]["af_source_columns_present"] == ["vep_gnomadg_af", "gnomad_af"]


def test_drop_terms_veto_only_the_transcript_that_earned_the_impact(cfg: FilterConfig):
    """Stage 2 reports the MANE transcript of the most severe gene whatever that
    transcript's own term is (``select_transcript``), while ``impact_any_coding``
    ranges over every coding transcript — so a synonymous MANE call with a
    stop_gained on another transcript is a real row shape, and rule 1 keeps it."""
    mane = {"gene_id": "ENSG00000156970", "transcript_id": "ENST_MANE", "biotype": "protein_coding",
            "mane_select": "NM_001211.6", "consequence_terms": ["synonymous_variant"], "impact": "LOW"}
    other = {"gene_id": "ENSG00000156970", "transcript_id": "ENST_ALT", "biotype": "protein_coding",
             "consequence_terms": ["stop_gained"], "impact": "HIGH"}
    assert select_transcript([mane, other]) is mane and impact_any_coding([mane, other]) == "HIGH"

    r = row(consequence="synonymous_variant", impact="LOW", impact_any_coding="HIGH", most_severe_consequence="stop_gained")
    d = screen_row(0, r, cfg)
    assert d.rule == "" and d.hits == ["consequence:HIGH", "rarity:af=1e-05"]
    # a synonymous-everywhere row is LOW and dropped by its impact, not by the term
    syn = row(consequence="synonymous_variant", impact="LOW", impact_any_coding="LOW", most_severe_consequence="synonymous_variant")
    assert screen_row(0, syn, cfg).rule == "consequence:LOW"
    # the term bites once LOW is a kept impact and the reported transcript is the one that earned it ...
    low = with_(cfg, "consequence", keep_impacts=["HIGH", "MODERATE", "LOW"])
    assert screen_row(0, syn, low).rule == "consequence:synonymous_variant"
    # ... unless the list is empty, or ClinVar / SpliceAI rescue the row
    assert screen_row(0, syn, with_(low, "consequence", drop_terms=[])).hits == ["consequence:LOW", "rarity:af=1e-05"]
    assert screen_row(0, {**syn, "clinvar_pathogenicity": "Likely_pathogenic"}, low).hits[0] == "consequence:clinvar=Likely_pathogenic"
    assert screen_row(0, {**syn, "spliceai_ds_max": "0.9"}, low).hits[0] == "consequence:splice=0.9"
    # a configured MODERATE term vetoes a MODERATE row but not one that is HIGH elsewhere
    mis = with_(cfg, "consequence", drop_terms=["missense_variant"])
    assert screen_row(0, row(consequence="missense_variant", impact="MODERATE", impact_any_coding="MODERATE"), mis).rule == \
        "consequence:missense_variant"
    assert screen_row(0, row(consequence="missense_variant", impact="MODERATE", impact_any_coding="HIGH"), mis).rule == ""


# ---------------------------------------------------------------- rules on hand-made rows

def test_zygosity_and_genotype_model():
    assert zygosity("0/1") == "het" and zygosity("1|0") == "het"
    assert zygosity("1/1") == "hom" and zygosity("1|1") == "hom" and zygosity("1") == "hap"
    assert zygosity("0/0") == "none" and zygosity("./.") == "none" and zygosity("") == "none"
    assert zygosity("./1") == "het"
    # after the multi-allelic split only allele 1 is this row's ALT (bcftools norm -m -any rewrites
    # 1/2 → 1/0 + 0/1 anyway, so these are defensive): 1/2 is het for it, 2/2 does not carry it
    assert zygosity("1/2") == "het" and zygosity("1|2") == "het" and zygosity("2/2") == "none" and zygosity("0/2") == "none"
    assert genotype_model("15", "1/1") == "hom" and genotype_model("15", "0/1") == "het"
    assert genotype_model("X", "1/1") == "hemi" and genotype_model("X", "1") == "hemi" and genotype_model("Y", "1") == "hemi"
    assert genotype_model("X", "0/1") == "het"
    assert genotype_model("MT", "1/1") == "mito" and genotype_model("MT", "0/1") == "mito" and genotype_model("MT", "1") == "mito"
    assert genotype_model("15", "0/0") == "" and genotype_model("MT", "0/0") == ""


def test_clinvar_terms_match_member_wise():
    plp = ["Pathogenic", "Likely_pathogenic"]
    assert clinvar_in("Pathogenic", plp) and clinvar_in("Likely_pathogenic", plp)
    assert clinvar_in("Pathogenic/Likely_pathogenic", plp)
    assert clinvar_in("Pathogenic,_low_penetrance", plp)
    assert not clinvar_in("Conflicting_classifications_of_pathogenicity", plp)
    assert not clinvar_in("Uncertain_significance", plp) and not clinvar_in("", plp)
    assert not clinvar_in("Pathogenic/Uncertain_significance", plp)  # not every member qualifies
    assert clinvar_in("Benign/Likely_benign", rules.BENIGN_CLASSES)


def test_rule_family_strips_values_but_keeps_named_qualifiers():
    assert rule_family("rarity:af=0.3227") == "rarity"
    assert rule_family("consequence:LOW") == "consequence:LOW"
    assert rule_family("consequence:synonymous_variant") == "consequence:synonymous_variant"
    assert rule_family("model:hom:nhom=12") == "model:hom"
    assert rule_family("model:het_single") == "model:het_single"
    assert rule_family("phase:cis") == "phase:cis"
    assert rule_family("genotype:gt=0/0") == "genotype"
    assert rule_family("clinvar_benign:Benign,stars=2") == "clinvar_benign"
    assert rule_family("duplicate") == "duplicate"
    assert str(Hit("rarity", "af=1e-05")) == "rarity:af=1e-05" and str(Hit("duplicate")) == "duplicate"


def test_consequence_rule_paths(cfg: FilterConfig):
    assert screen_row(0, row(impact_any_coding="MODIFIER", consequence="intron_variant"), cfg).rule == "consequence:MODIFIER"
    assert screen_row(0, row(impact_any_coding="", consequence="intergenic_variant"), cfg).rule == "consequence:none"
    assert screen_row(0, row(impact_any_coding="SPLICE", consequence="splice_region_variant", spliceai_ds_max="0.19"), cfg).rule == "consequence:SPLICE"
    assert screen_row(0, row(impact_any_coding="SPLICE", consequence="splice_region_variant", spliceai_ds_max="0.2"), cfg).hits[0] == "consequence:splice=0.2"
    assert screen_row(0, row(impact_any_coding="LOW", clinvar_pathogenicity="Pathogenic/Likely_pathogenic"), cfg).hits[0] == \
        "consequence:clinvar=Pathogenic/Likely_pathogenic"
    with pytest.raises(ValueError, match="not a number"):
        screen_row(0, row(impact_any_coding="LOW", spliceai_ds_max="high"), cfg)


def test_rules_run_lazily_so_a_dropped_row_is_not_parsed_further(cfg: FilterConfig):
    # rule 1 drops before rule 2 would read the malformed star count
    d = screen_row(0, row(impact_any_coding="LOW", clinvar_pathogenicity="Benign", clinvar_stars="two"), cfg)
    assert d.rule == "consequence:LOW" and d.hits == ["consequence:LOW"]
    # a kept impact never reads SpliceAI at all
    assert screen_row(0, row(impact_any_coding="MODERATE", spliceai_ds_max="high"), cfg).rule == ""
    with pytest.raises(ValueError):  # ... but a benign row that reaches rule 2 still fails loudly
        screen_row(0, row(clinvar_pathogenicity="Benign", clinvar_stars="two"), cfg)


def test_malformed_cells_are_reported_without_the_variant(tmp_path: Path, cfg: FilterConfig):
    """A traceback pasted into a bug report must say where, never what."""
    with pytest.raises(ValueError) as info:
        list(screen_rows([row(), row(pos="40170150", ref="C", alt="T", gnomad_af="abc")], cfg))
    assert messages(info.value)[0] == "table row 1: gnomad_af: not a number: 'abc'"
    with pytest.raises(ValueError) as info:
        list(screen_rows([row(pos="4017010x")], cfg))
    assert messages(info.value)[0] == "table row 0: pos: not an integer"
    assert "4017010x" not in "".join(messages(info.value))
    with pytest.raises(ValueError) as info:
        run_filter(make_run(tmp_path, [row(gt="1/1", gnomad_nhom="many")]), DEFAULT_CONFIG)
    assert messages(info.value)[0] == "table row 0: gnomad_nhom: not a number: 'many'"
    for m in messages(info.value):
        for secret in ("15:40170100", "40170100", "BUB1B", "1/1"):
            assert secret not in m
    result = CliRunner().invoke(main, ["filter", "--run", str(tmp_path / "run")])
    assert result.exit_code != 0
    assert "40170100" not in result.output and "40170100" not in "".join(messages(result.exception))


def test_clinvar_benign_under_the_star_threshold_is_a_hit_not_a_drop(cfg: FilterConfig):
    d = screen_row(0, row(clinvar_pathogenicity="Likely_benign", clinvar_stars="1"), cfg)
    assert d.rule == "" and "clinvar_benign:Likely_benign,stars=1" in d.hits
    d = screen_row(0, row(clinvar_pathogenicity="Benign/Likely_benign", clinvar_stars=""), cfg)
    assert d.rule == "" and "clinvar_benign:Benign/Likely_benign,stars=?" in d.hits
    assert screen_row(0, row(clinvar_pathogenicity="Benign", clinvar_stars="2"), cfg).rule == "clinvar_benign:Benign,stars=2"


def test_rarity_rescues_reviewed_clinvar_plp_and_records_a_refusal(cfg: FilterConfig):
    d = screen_row(0, row(gnomad_af="", vep_gnomade_af="", vep_gnomadg_af="0.5"), cfg)
    assert d.rule == "rarity:af=0.5" and d.af_source == "vep_gnomadg_af"
    d = screen_row(0, row(gnomad_af="0.02", clinvar_pathogenicity="Likely_pathogenic", clinvar_stars="1"), cfg)
    assert d.rule == "" and d.hits == ["consequence:HIGH", "rarity:af=0.02", "clinvar:always_keep=Likely_pathogenic"]
    d = screen_row(0, row(gnomad_af="0.02", clinvar_pathogenicity="Likely_pathogenic", clinvar_stars=""), cfg)
    assert d.rule == "rarity:af=0.02" and d.hits[1] == "clinvar:rescue_refused=Likely_pathogenic,stars=0<1"
    d = screen_row(0, row(gnomad_af="0.01"), cfg)  # at the ceiling is still rare
    assert d.rule == "" and d.af == 0.01


def test_af_is_the_first_non_empty_source_and_masked_rows_are_counted(tmp_path: Path, cfg: FilterConfig):
    """CONTRACTS.md rule 3 as written. Stage 2 skips the gnomAD API when the *larger*
    VEP copy is common, so the first copy can be rare while a later one is not; the
    run counts and notes such rows, and ``af_fallback_pick: max`` is the switch."""
    split = dict(gnomad_af="", vep_gnomade_af="0.005", vep_gnomadg_af="0.30")
    d = screen_row(0, row(**split), cfg)
    assert d.rule == "" and (d.af_used, d.af_source) == ("0.005", "vep_gnomade_af") and d.af_masked
    # the first column, when present, is never second-guessed
    d = screen_row(0, row(gnomad_af="0.004", vep_gnomade_af="0.005", vep_gnomadg_af="0.30"), cfg)
    assert d.rule == "" and (d.af_used, d.af_source) == ("0.004", "gnomad_af") and not d.af_masked

    run = make_run(tmp_path, [row(**split), row(pos="40170150", ref="C", alt="T", gnomad_af="", vep_gnomadg_af="0.30")])
    m = json.loads(run_filter(run, DEFAULT_CONFIG).read_text())
    by = decisions_by_key(run)
    assert (by[CIS_A]["af_used"], by[CIS_A]["af_source"]) == ("0.005", "vep_gnomade_af")
    assert by[CIS_B]["rule"] == "rarity:af=0.30"
    assert m["counts"]["rarity"] == {"rows_by_af_source": {"vep_gnomade_af": 1, "vep_gnomadg_af": 1},
                                     "af_fallback_first_masked_rows": 1}
    assert any("af_fallback_pick=first" in n for n in m["notes"])

    m = json.loads(run_filter(run, write_config(tmp_path, rarity={"af_fallback_pick": "max"})).read_text())
    assert m["params"]["rarity"]["af_fallback_pick"] == "max"
    by = decisions_by_key(run)
    assert by[CIS_A]["rule"] == "rarity:af=0.30" and (by[CIS_A]["af_used"], by[CIS_A]["af_source"]) == ("0.30", "vep_gnomadg_af")
    assert m["counts"]["rarity"]["af_fallback_first_masked_rows"] == 0 and m["notes"] == []
    # under max, ties go to the earlier column
    d = screen_row(0, row(gnomad_af="", vep_gnomade_af="0.3", vep_gnomadg_af="0.30"), with_(cfg, "rarity", af_fallback_pick="max"))
    assert (d.af_used, d.af_source) == ("0.3", "vep_gnomade_af")


def test_quality_caveats_never_drop_and_split_back_from_the_tsv(tmp_path: Path, cfg: FilterConfig):
    """A caller FILTER string is ';'-separated, and so is the caveats cell: each filter
    code is its own caveat, so the cell splits back into exactly the caveats."""
    d = screen_row(0, row(dp="9", gq="19", quality_flag="LowQual;QD2"), cfg)
    assert d.rule == "" and d.caveats == ["dp<10", "gq<20", "flagged:LowQual", "flagged:QD2"]
    assert screen_row(0, row(dp="", gq=".", quality_flag=""), cfg).caveats == []  # unknown is not low
    run = make_run(tmp_path, [row(dp="9", gq="19", quality_flag="LowQual;QD2")])
    run_filter(run, DEFAULT_CONFIG)
    (d,) = read_decisions(run)
    assert d["kept"] == "1" and d["caveats"].split(";") == ["dp<10", "gq<20", "flagged:LowQual", "flagged:QD2"]
    (s,) = read_shortlist(run)
    assert s["caveats"] == d["caveats"] and s["quality_flag"] == "LowQual;QD2"
    (c,) = read_candidates(run)["candidates"]
    assert c["caveats"] == ["dp<10", "gq<20", "flagged:LowQual", "flagged:QD2"]


def test_alt_haplotype_reads_a_phased_pgt():
    assert alt_haplotype("0|1") == 1 and alt_haplotype("1|0") == 0
    assert alt_haplotype("0/1") is None and alt_haplotype("") is None and alt_haplotype(".|.") is None
    assert alt_haplotype("1|1") is None and alt_haplotype("1|2") == 0 and alt_haplotype("0|1|1") is None


def test_pair_phase_from_pid_and_pgt():
    a, b = row(pid="p1", pgt="0|1"), row(pos="40170150", pid="p1", pgt="1|0")
    # the contract: a shared PID is cis, whatever the PGT says — and what it says is quoted
    assert pair_phase(a, b) == ("cis", "shared PID p1 (PGT 0|1, 1|0); 50 bp apart")
    assert opposite_pgt(a, b)
    # trust_pgt: the caller phased the ALTs onto opposite haplotypes → trans
    assert pair_phase(a, b, trust_pgt=True) == ("trans", "shared PID p1 with opposite PGT (PGT 0|1, 1|0); 50 bp apart")
    # the same haplotype is cis under either reading
    same = row(pos="40170150", pid="p1", pgt="0|1")
    assert pair_phase(a, same) == ("cis", "shared PID p1 (PGT 0|1, 0|1); 50 bp apart")
    assert pair_phase(a, same, trust_pgt=True) == ("cis", "shared PID p1 with the same PGT (PGT 0|1, 0|1); 50 bp apart")
    assert not opposite_pgt(a, same)
    # a shared PID without a usable PGT on either side is cis under both readings
    assert pair_phase(row(pid="p1"), row(pos="40170150", pid="p1"), trust_pgt=True) == ("cis", "shared PID p1; 50 bp apart")
    assert pair_phase(a, row(pos="40170150", pid="p1", pgt="0/1"), trust_pgt=True) == ("cis", "shared PID p1 (PGT 0|1, 0/1); 50 bp apart")
    assert pair_phase(row(pid="p1"), row(pos="40170150", pid="p2")) == ("unknown", "different PIDs (p1, p2); 50 bp apart")
    assert pair_phase(row(), row(pos="40181000")) == ("unknown", "no shared PID; 10.9 kb apart")
    assert not opposite_pgt(row(pgt="0|1"), row(pgt="1|0"))  # no PID, nothing shared


# Labelled with the public demo gene rather than the row() default: the hand-made comphets below then carry a
# candidate id (CFTR:comphet) that names nothing about any case; the positions are invented either way.
DEMO_GENE = dict(gene_symbol="CFTR", gene_id="ENSG00000001626")
OPPOSITE_PGT_PAIR = [dict(pos="40170100", pid="p1", pgt="0|1", **DEMO_GENE),
                     dict(pos="40170150", ref="C", alt="T", pid="p1", pgt="1|0", **DEMO_GENE)]


def test_shared_pid_is_cis_by_default_and_the_callers_trans_is_noted(tmp_path: Path):
    """CONTRACTS.md rule 5: shared PID → cis, so the pair is dropped as phase:cis even
    though the caller's PGT phased it in trans; the run says so, twice over."""
    run = make_run(tmp_path, [row(**r) for r in OPPOSITE_PGT_PAIR])
    m = json.loads(run_filter(run, DEFAULT_CONFIG).read_text())
    assert m["params"]["phase"]["trust_pgt"] is False
    assert m["counts"]["phase"] == {"cis_rows_dropped": 2, "cis_rows_dropped_het_single_eligible": 2,
                                    "shared_pid_opposite_pgt_pairs": 1}
    assert m["counts"]["dropped_by_rule"] == {"phase:cis": 2}
    assert any("opposite PGT" in n and "trust_pgt is false" in n for n in m["notes"])
    assert any(n.startswith("2 heterozygous row(s) dropped as phase:cis") and ELIGIBLE_NOTE in n for n in m["notes"])
    assert read_candidates(run)["candidates"] == [] and list(read_shortlist(run)) == []
    for d in read_decisions(run):
        assert (d["kept"], d["rule"], d["model"]) == ("0", "phase:cis", "")
        assert d["rule_hits"] == f"consequence:HIGH;rarity:af=1e-05;{HET_SINGLE_ELIGIBLE};phase:cis"


def test_trust_pgt_is_a_config_choice_that_forms_a_trans_comphet(tmp_path: Path):
    run = make_run(tmp_path, [row(**r) for r in OPPOSITE_PGT_PAIR])
    m = json.loads(run_filter(run, write_config(tmp_path, phase={"trust_pgt": True})).read_text())
    assert m["params"]["phase"]["trust_pgt"] is True
    assert m["counts"]["phase"] == {"cis_rows_dropped": 0, "cis_rows_dropped_het_single_eligible": 0,
                                    "shared_pid_opposite_pgt_pairs": 1, "trans": 1}
    assert m["notes"] == []
    cands = read_candidates(run)["candidates"]
    assert [c["candidate_id"] for c in cands] == ["CFTR:comphet"]
    assert cands[0]["phase"] == {"status": "trans",
                                 "evidence": "shared PID p1 with opposite PGT (PGT 0|1, 1|0); 50 bp apart"}
    assert all(d["kept"] == "1" and d["model"] == "comphet" for d in read_decisions(run))
    assert all(r["phase"] == "trans" for r in read_shortlist(run))
    # the same PGT on both is cis under trust_pgt too
    run = make_run(tmp_path, [row(pos="40170100", pid="p1", pgt="0|1"), row(pos="40170150", ref="C", alt="T", pid="p1", pgt="0|1")], name="same")
    m = json.loads(run_filter(run, write_config(tmp_path, phase={"trust_pgt": True})).read_text())
    assert m["counts"]["dropped_by_rule"] == {"phase:cis": 2} and m["counts"]["phase"]["shared_pid_opposite_pgt_pairs"] == 0


def test_cis_only_het_is_dropped_and_its_dominant_eligibility_is_recorded(tmp_path: Path):
    """A dominant P/LP het that shares its phasing window with a surviving VUS is one
    allele with it and is dropped as phase:cis (rule 5) — but not silently: the hit,
    the count and the manifest note say a het_single candidate is hiding there."""
    p = dict(chrom="17", pos="7674894", ref="G", alt="A", gene_symbol="TP53", gene_id="ENSG00000141510",
             clinvar_pathogenicity="Pathogenic", clinvar_stars="2", gnomad_af="1e-06")
    vus = dict(chrom="17", pos="7674900", ref="C", alt="T", gene_symbol="TP53", gene_id="ENSG00000141510",
               impact_any_coding="MODERATE", consequence="missense_variant", gnomad_af="2e-06")
    alone = make_run(tmp_path, [row(**p)], name="alone")
    m = json.loads(run_filter(alone, DEFAULT_CONFIG).read_text())
    assert [c["candidate_id"] for c in read_candidates(alone)["candidates"]] == ["TP53:het_single"] and m["notes"] == []

    paired = make_run(tmp_path, [row(**p, pid="p1", pgt="0|1"), row(**vus, pid="p1", pgt="0|1")], name="paired")
    m = json.loads(run_filter(paired, DEFAULT_CONFIG).read_text())
    by = decisions_by_key(paired)
    assert (by[R213X]["kept"], by[R213X]["rule"], by[R213X]["model"]) == ("0", "phase:cis", "")
    assert by[R213X]["rule_hits"] == f"consequence:HIGH;rarity:af=1e-06;{HET_SINGLE_ELIGIBLE};phase:cis"
    assert (by["17:7674900:C:T"]["kept"], by["17:7674900:C:T"]["rule"]) == ("0", "phase:cis")
    assert by["17:7674900:C:T"]["rule_hits"] == "consequence:MODERATE;rarity:af=2e-06;phase:cis"
    assert read_candidates(paired)["candidates"] == []
    assert m["counts"]["phase"] == {"cis_rows_dropped": 2, "cis_rows_dropped_het_single_eligible": 1,
                                    "shared_pid_opposite_pgt_pairs": 0}
    assert m["counts"]["dropped_by_rule"] == {"phase:cis": 2}
    assert len(m["notes"]) == 1 and m["notes"][0].startswith("1 heterozygous row(s) dropped as phase:cis") and ELIGIBLE_NOTE in m["notes"][0]


def test_three_hets_with_one_cis_pair_all_join_the_comphet(tmp_path: Path):
    rows = [row(pos="40170100", pid="p1", **DEMO_GENE), row(pos="40170150", ref="C", alt="T", pid="p1", **DEMO_GENE),
            row(pos="40175000", ref="T", alt="C", impact_any_coding="MODERATE", consequence="missense_variant", **DEMO_GENE)]
    run = make_run(tmp_path, rows)
    m = json.loads(run_filter(run, DEFAULT_CONFIG).read_text())
    cands = read_candidates(run)["candidates"]
    assert len(cands) == 1 and cands[0]["candidate_id"] == "CFTR:comphet"
    assert [v["key"] for v in cands[0]["variants"]] == [CIS_A, CIS_B, "15:40175000:T:C"]
    ev = cands[0]["phase"]["evidence"]
    assert f"{CIS_A}~{CIS_B}: cis, shared PID p1" in ev and f"{CIS_A}~15:40175000:T:C: unknown" in ev
    by = {r["pos"]: r for r in read_shortlist(run)}
    assert by["40170100"]["partner_keys"] == "15:40175000:T:C"  # only non-cis partners are listed
    assert by["40175000"]["partner_keys"] == f"{CIS_A};{CIS_B}"
    assert all(r["phase"] == "unknown" for r in by.values())
    assert m["counts"]["phase"] == {"cis_rows_dropped": 0, "cis_rows_dropped_het_single_eligible": 0,
                                    "shared_pid_opposite_pgt_pairs": 0, "unknown": 1}


def test_hom_ceiling_exempts_reviewed_clinvar_plp_with_a_trace(tmp_path: Path):
    rows = [row(gt="1/1", gnomad_nhom="12"),
            row(pos="40175000", ref="T", alt="C", gt="1/1", gnomad_nhom="58", clinvar_pathogenicity="Pathogenic",
                clinvar_stars="4", gene_symbol="CFTR", gene_id="ENSG00000001626"),
            row(pos="40176000", ref="T", alt="G", gt="1/1", gnomad_nhom="", gene_symbol="TP53", gene_id="ENSG00000141510")]
    run = make_run(tmp_path, rows)
    run_filter(run, DEFAULT_CONFIG)
    by = decisions_by_key(run)
    assert by[CIS_A]["rule"] == "model:hom:nhom=12"
    d = by["15:40175000:T:C"]
    assert d["kept"] == "1"
    # the exceeded ceiling and the rescue that overrode it are both on record, once each
    assert d["rule_hits"] == "consequence:HIGH;rarity:af=1e-05;model:hom:nhom=58;clinvar:always_keep_nhom=Pathogenic;model:hom"
    assert by["15:40176000:T:G"]["kept"] == "1"  # unknown homozygote count is not evidence of homozygotes
    ids = [c["candidate_id"] for c in read_candidates(run)["candidates"]]
    assert ids == ["CFTR:hom", "TP53:hom"]  # ClinVar P/LP first; then the same model, impact and AF → gene name


def test_clinvar_rescue_is_bounded_by_stars_and_af(tmp_path: Path):
    """A 0-star 'Pathogenic' at 30% with thousands of homozygotes is not a priority-1
    candidate; F508del-like entries (reviewed, ~1%) still are. Every refusal is traced."""
    common_p = dict(gnomad_af="0.30", clinvar_pathogenicity="Pathogenic")
    rows = [row(gt="1|1", gnomad_nhom="5000", clinvar_stars="0", **common_p),                        # refused: stars
            row(pos="40175000", ref="T", alt="C", gt="1|1", gnomad_nhom="5000", clinvar_stars="4", **common_p,
                gene_symbol="CFTR", gene_id="g2"),                                                  # refused: AF
            row(pos="40176000", ref="T", alt="G", gt="1|1", gnomad_nhom="58", gnomad_af="0.0119",
                clinvar_pathogenicity="Pathogenic", clinvar_stars="4", gene_symbol="TP53", gene_id="g3"),  # rescued twice
            row(pos="40177000", ref="A", alt="C", gt="1|1", gnomad_nhom="58", gnomad_af="0.005",
                clinvar_pathogenicity="Pathogenic", clinvar_stars="", gene_symbol="MTHFR", gene_id="g4")]  # nhom: unknown stars
    run = make_run(tmp_path, rows)
    run_filter(run, DEFAULT_CONFIG)
    by = decisions_by_key(run)
    d = by[CIS_A]
    assert d["rule"] == "rarity:af=0.30"
    assert d["rule_hits"] == "consequence:HIGH;clinvar:rescue_refused=Pathogenic,stars=0<1;rarity:af=0.30"
    d = by["15:40175000:T:C"]
    assert d["rule"] == "rarity:af=0.30"
    assert d["rule_hits"] == "consequence:HIGH;clinvar:rescue_refused=Pathogenic,af>0.05;rarity:af=0.30"
    d = by["15:40176000:T:G"]
    assert (d["kept"], d["model"]) == ("1", "hom")
    assert d["rule_hits"] == ("consequence:HIGH;rarity:af=0.0119;clinvar:always_keep=Pathogenic;"
                              "model:hom:nhom=58;clinvar:always_keep_nhom=Pathogenic;model:hom")
    d = by["15:40177000:A:C"]
    assert d["rule"] == "model:hom:nhom=58"
    assert d["rule_hits"] == "consequence:HIGH;rarity:af=0.005;clinvar:rescue_refused=Pathogenic,stars=0<1;model:hom:nhom=58"
    assert [c["candidate_id"] for c in read_candidates(run)["candidates"]] == ["TP53:hom"]
    # the bounds are live thresholds: loosen both and the refused rows come back
    loose = write_config(tmp_path, rarity={"rescue_max_af": 0.5}, clinvar={"rescue_min_stars": 0})
    m = json.loads(run_filter(run, loose).read_text())
    assert m["params"]["rarity"]["rescue_max_af"] == 0.5 and m["params"]["clinvar"]["rescue_min_stars"] == 0
    assert all(d["kept"] == "1" for d in read_decisions(run))


def test_het_single_needs_high_or_clinvar_and_the_dominant_ceiling(tmp_path: Path):
    rows = [row(pos="40170100", gnomad_af="0.0001"),                                                   # HIGH at the ceiling
            row(pos="40170150", ref="C", alt="T", gnomad_af="0.00011", gene_symbol="CFTR", gene_id="g2"),  # just above
            row(pos="40170200", ref="T", alt="C", impact_any_coding="MODERATE", consequence="missense_variant",
                clinvar_pathogenicity="Likely_pathogenic", gene_symbol="TP53", gene_id="g3"),        # moderate but ClinVar LP
            row(pos="40170250", ref="A", alt="C", impact_any_coding="MODERATE", consequence="missense_variant",
                gene_symbol="MTHFR", gene_id="g4")]                                                   # moderate, nothing else
    run = make_run(tmp_path, rows)
    run_filter(run, DEFAULT_CONFIG)
    by = decisions_by_key(run)
    assert by[CIS_A]["model"] == "het_single"
    assert by[CIS_B]["rule"] == "model:het_single"
    assert by[NON_CARRIER]["model"] == "het_single"
    assert by["15:40170250:A:C"]["rule"] == "model:het_single"
    ids = [c["candidate_id"] for c in read_candidates(run)["candidates"]]
    assert ids == ["TP53:het_single", "BUB1B:het_single"]  # ClinVar LP outranks a HIGH impact without it


def test_gene_grouping_falls_back_to_gene_id_then_key(tmp_path: Path):
    rows = [row(pos="40170100", gene_symbol="", gene_id="ENSG_A"),
            row(pos="40170150", ref="C", alt="T", gene_symbol="", gene_id="ENSG_A"),
            row(pos="40170200", ref="T", alt="C", gene_symbol="", gene_id="")]
    run = make_run(tmp_path, rows)
    run_filter(run, DEFAULT_CONFIG)
    ids = [c["candidate_id"] for c in read_candidates(run)["candidates"]]
    assert ids == ["ENSG_A:comphet", f"{NON_CARRIER}:het_single"]


def test_missing_source_columns_are_noted_not_fatal(tmp_path: Path):
    """A table from a stage-2 run without gnomAD still filters; the manifest says so."""
    cols = [c for c in ANNOTATED_COLUMNS if c not in GnomadRetriever.columns]
    run = tmp_path / "run"
    write_table(run, cols, [row(vep_gnomade_af="0.3")])
    m = json.loads(run_filter(run, DEFAULT_CONFIG).read_text())
    assert any(n.startswith("gnomad:") for n in m["notes"])
    assert m["params"]["af_source_columns_present"] == ["vep_gnomade_af", "vep_gnomadg_af"]
    d = next(iter(read_decisions(run)))
    assert d["rule"] == "rarity:af=0.3" and d["af_source"] == "vep_gnomade_af"
    assert list(read_shortlist(run)) == []


# ---------------------------------------------------------------- CLI

def test_cli_prints_counts_and_manifest_but_no_variant(tmp_path: Path):
    run = make_run(tmp_path)
    cfg_path = write_config(tmp_path)
    result = CliRunner().invoke(main, ["filter", "--run", str(run), "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "rows in: 14 · kept: 6 · dropped: 8 · candidates: 5" in result.output
    assert "hom 1, comphet 1, hemi 1, mito 1, het_single 1" in result.output
    assert str(run / STAGE_DIR / "manifest.json") in result.output
    for secret in ("117559590", "40170100", "CFTR", "TP53", "BUB1B", "0/1", "rs1801133"):
        assert secret not in result.output
    assert "filter" in CliRunner().invoke(main, ["--help"]).output


def test_cli_default_config_is_the_source_tree_file(tmp_path: Path):
    run = make_run(tmp_path)
    result = CliRunner().invoke(main, ["filter", "--run", str(run)])
    assert result.exit_code == 0, result.output
    assert json.loads((run / STAGE_DIR / "manifest.json").read_text())["params"]["config_path"] == str(DEFAULT_CONFIG)


# ---------------------------------------------------------------- live

@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ENGINE_LIVE_TESTS"), reason="set ENGINE_LIVE_TESTS=1 to hit rest.ensembl.org")
def test_live_vep_still_agrees_with_the_real_fixture_rows(tmp_path: Path):
    """Stage 3 has no API of its own; this pins the fixture's real rows to the public
    source they were taken from, so a VEP release that re-annotates CFTR p.Gly542Ter
    or TP53 p.Arg213Ter is noticed rather than silently tested against."""
    from engine.retrieve.http import Http, HttpCache
    from engine.retrieve.store import parse_key

    vep = VepRetriever(Http(HttpCache(tmp_path / "cache")), workers=1)
    keys = [parse_key(G542X), parse_key(R213X)]
    found = vep.retrieve(keys)
    fixture = {f"{r['chrom']}:{r['pos']}:{r['ref']}:{r['alt']}": r for r in fixture_rows()}
    for k, ks in zip(keys, (G542X, R213X)):
        cols = vep.extract(found[k])
        for c in ("gene_symbol", "gene_id", "transcript_id", "consequence", "impact", "impact_any_coding", "hgvsp"):
            assert cols[c] == fixture[ks][c], (ks, c, cols[c])
