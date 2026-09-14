"""The public demo case — ``tests/fixtures/public_case/`` (CONTRACTS.md, last section).

Twelve public alleles in one hand-written VCF (two ClinVar-pathogenic CFTR alleles in
trans-or-unknown phase, a dominant TP53 pathogenic het, and common decoys), plus a
``case.yaml`` with cystic-fibrosis HPO terms. Nothing here derives from a person.

Network-free: the fixture files are checked as written, stage 1 runs on the VCF
(bcftools), stages 3 and 5 (dry run) run on the recorded stage-2 output under
``fixtures/public_case/02_retrieve/``, and stage 6 (dry run) on the recorded stage-5
chain under ``fixtures/public_case/05_reason/`` — the real stage 5 over that same
evidence with a scripted client, which is what the script copies into the run before
stage 6. The live test runs ``scripts/run_public_case.sh`` — Ensembl VEP, gnomAD, the
local ClinVar release and, when Docker and the verified Exomiser bundle are present,
stage 4 — and needs ``ENGINE_LIVE_TESTS=1``.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from engine.config import CaseConfig
from engine.filter.run import read_candidates, run_filter
from engine.ingest import COLUMNS, read_variants, run_ingest

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "public_case"
SCRIPT = REPO / "scripts" / "run_public_case.sh"
CHAIN = FIX / "05_reason" / "chain_CFTR_comphet.json"

F508DEL = "7:117559590:ATCT:A"
G542X = "7:117587778:G:T"
R175H = "17:7675088:C:T"

# Every allele in the fixture, in file order, as (key, rsID, gene, GT). All public.
ALLELES = [
    ("1:11794419:T:G", "rs1801131", "MTHFR", "0/1"),
    ("1:11796321:G:A", "rs1801133", "MTHFR", "0/1"),
    ("7:117559403:A:G", "rs34855237", "CFTR", "0/1"),
    ("7:117559479:G:A", "rs213950", "CFTR", "0/1"),
    (F508DEL, "rs113993960", "CFTR", "0/1"),
    ("7:117578521:G:A", "rs213953", "CFTR", "1/1"),
    (G542X, "rs113993959", "CFTR", "0/1"),
    ("7:117589483:T:A", "rs213965", "CFTR", "0/1"),
    ("7:117595001:T:G", "rs1042077", "CFTR", "0/1"),
    ("15:40185630:G:A", "rs1801376", "BUB1B", "1/1"),
    (R175H, "rs28934578", "TP53", "0/1"),
    ("17:7676154:G:C", "rs1042522", "TP53", "0/1"),
]
HPO = ["HP:0012236", "HP:0001738", "HP:0002205", "HP:0006528", "HP:0002110"]
# What the CFTR:comphet bundle may cite: the recorded stage-2 records of its two variants.
CFTR_PAIR_IDS = ["clinvar:VCV000007105", "clinvar:VCV000007115", "gnomad:7-117559590-ATCT-A", "gnomad:7-117587778-G-T",
                 f"vep:{F508DEL}", f"vep:{G542X}"]

needs_htslib = pytest.mark.skipif(
    not (shutil.which("bcftools") and shutil.which("bgzip") and shutil.which("tabix")),
    reason="bcftools/htslib not installed",
)


def _records(text: str) -> list[list[str]]:
    return [line.split("\t") for line in text.splitlines() if line and not line.startswith("#")]


def _sample_fields(rec: list[str]) -> dict[str, str]:
    return dict(zip(rec[8].split(":"), rec[9].split(":")))


# ------------------------------------------------------------------ the files as written

def test_vcf_holds_exactly_the_public_alleles():
    text = (FIX / "public.vcf").read_text()
    header = [line for line in text.splitlines() if line.startswith("#")]
    assert header[0] == "##fileformat=VCFv4.2"
    assert header[-1].split("\t") == ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "PUBLIC01"]
    contigs = [line[13:].split(",", 1)[0] for line in header if line.startswith("##contig=<ID=")]
    assert contigs == ["1", "7", "15", "17"]  # GRCh38, bare names, declared
    assert any(line.startswith("##reference=GRCh38") for line in header)

    recs = _records(text)
    assert [(f"{r[0]}:{r[1]}:{r[3]}:{r[4]}", r[2]) for r in recs] == [(k, rs) for k, rs, _, _ in ALLELES]
    assert all(r[6] == "PASS" for r in recs)
    assert all("," not in r[4] for r in recs)  # no multi-allelic sites: stage 1 must not have to split
    positions = [(int(r[0]) if r[0].isdigit() else 99, int(r[1])) for r in recs]
    assert positions == sorted(positions)  # sorted, as tabix requires

    fields = {f"{r[0]}:{r[1]}:{r[3]}:{r[4]}": _sample_fields(r) for r in recs}
    assert [fields[k]["GT"] for k, *_ in ALLELES] == [gt for *_, gt in ALLELES]
    for f in fields.values():
        assert {"GT", "AD", "DP", "GQ", "PL"} <= set(f)
        ref_depth, alt_depth = (int(x) for x in f["AD"].split(","))
        assert ref_depth + alt_depth == int(f["DP"]) and int(f["GQ"]) >= 20
        assert (ref_depth == 0) == (f["GT"] == "1/1")

    # the compound-het pair: both het, and no PID shared between them
    a, b = fields[F508DEL], fields[G542X]
    assert a["GT"] == "0/1" and b["GT"] == "0/1"
    assert a["PID"] == "117559403_A_G" and a["PGT"] == "1|0"
    assert "PID" not in b and "PGT" not in b
    # the phasing group is realistic: three records within a read pair, one PID, phased PGTs
    group = [k for k, f in fields.items() if f.get("PID") == "117559403_A_G"]
    assert group == ["7:117559403:A:G", "7:117559479:G:A", F508DEL]
    assert [fields[k]["PGT"] for k in group] == ["0|1", "0|1", "1|0"]


def test_bgzipped_copy_matches_the_source_and_is_indexed():
    with gzip.open(FIX / "public.vcf.gz", "rt") as f:
        assert f.read() == (FIX / "public.vcf").read_text()
    raw = (FIX / "public.vcf.gz").read_bytes()
    assert raw[:2] == b"\x1f\x8b" and raw[3] & 0x04 and b"BC" in raw[:20]  # gzip with the BGZF extra field
    assert (FIX / "public.vcf.gz.tbi").stat().st_size > 0


def test_case_file_names_the_public_proband_and_verified_hpo_terms():
    case = CaseConfig.load(FIX / "case.yaml")
    assert case.proband_id == "PUBLIC01"
    assert case.vcf == (FIX / "public.vcf.gz").resolve()
    assert case.hpo == HPO
    assert case.reference_fasta is None and case.regions is None and case.sample is None


def test_script_is_valid_bash_and_points_at_the_fixture():
    text = SCRIPT.read_text()
    assert text.startswith("#!/usr/bin/env bash")
    assert os.access(SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assert 'CASE="$REPO/tests/fixtures/public_case/case.yaml"' in text
    assert 'CHAIN_FIXTURE="$REPO/tests/fixtures/public_case/05_reason/chain_CFTR_comphet.json"' in text
    for stage in ("engine ingest", "engine retrieve", "engine filter", "engine rank", "engine reason", "engine medicine"):
        assert stage in text, stage
    assert "--dry-run" in text and "--funnel" not in text.split("engine retrieve", 1)[1].split("\n", 1)[0]
    # the run and cache default to a plain temp directory, never a machine- or session-specific path
    assert 'PUBLIC_TMP="${TMPDIR:-/tmp}/engine-public-case"' in text
    for leak in ("scratchpad", "/private/tmp", "claude-", "/Users/"):
        assert leak not in text, leak
    assert not re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", text)  # no session id
    # <M37>/work and <M37>/cache are refused, not merely warned about
    assert 'for name in ("work", "cache"):' in text and "holds the real case" in text
    # stage 5 runs dry by default and live only on request; stage 6 always gets a chain
    assert "PUBLIC_LIVE_MODEL" in text and "ANTHROPIC_API_KEY" in text
    assert 'cp "$CHAIN_FIXTURE" "$RUN_DIR/05_reason/chains/CFTR:comphet.json"' in text
    assert 'rmdir "$RUN_DIR/06_medicine"' in text


# ------------------------------------------------------------------ stage 1 on the fixture

@needs_htslib
def test_ingest_row_counts(tmp_path: Path):
    case = CaseConfig.load(FIX / "case.yaml")
    run = tmp_path / "run"
    manifest_path = run_ingest(case, run, threads=1)

    rows = list(read_variants(run))
    assert list(rows[0].keys()) == list(COLUMNS)
    assert [(f"{r['chrom']}:{r['pos']}:{r['ref']}:{r['alt']}", r["id"], r["gt"]) for r in rows] == \
        [(k, rs, gt) for k, rs, _, gt in ALLELES]
    assert all(r["filter"] == "PASS" and r["quality_flag"] == "" and r["split_from"] == "" for r in rows)
    assert all(r["chrom"] == r["chrom_source"] for r in rows)  # bare names in, canonical names out

    by_key = {f"{r['chrom']}:{r['pos']}:{r['ref']}:{r['alt']}": r for r in rows}
    assert by_key[F508DEL]["vtype"] == "indel" and by_key[G542X]["vtype"] == "snv"
    assert by_key[F508DEL]["pid"] == "117559403_A_G" and by_key[F508DEL]["pgt"] == "1|0"
    assert by_key[G542X]["pid"] == "." and by_key[G542X]["pgt"] == "."
    assert by_key[F508DEL]["pid"] != by_key[G542X]["pid"]
    assert by_key[R175H]["ad"] == "26,24" and by_key[R175H]["dp"] == "50" and by_key[R175H]["gq"] == "99"
    assert by_key[F508DEL]["info_dp"] == "42" and by_key[F508DEL]["qd"] == "19.35"

    m = json.loads(manifest_path.read_text())
    c = m["counts"]
    assert c["rows_out"] == 12 and c["sites_in"] == 12 and c["pass_rows"] == 12 and c["flagged_rows"] == 0
    assert c["multiallelic_sites_split"] == 0 and c["rows_from_splits"] == 0 and c["rows_realigned"] == 0
    assert c["dropped_non_primary"] == 0 and c["dropped_star_allele"] == 0 and c["dropped_star_sites"] == 0
    assert c["per_type"] == {"snv": 11, "indel": 1}
    assert c["per_gt"] == {"0/1": 10, "1/1": 2}
    assert c["rows_per_contig"] == {"1": 2, "7": 7, "15": 1, "17": 2}
    assert c["records_in_file"] == 12 == c["records_on_primary_contigs"] and c["records_excluded_non_primary"] == 0
    assert any(n.startswith("Consistency check passed") for n in m["notes"])
    assert m["params"]["sample"] == "PUBLIC01" and m["params"]["naming_style"] == "ensembl"
    assert m["params"]["format_tags_present"] == ["GT", "AD", "DP", "GQ", "PGT", "PID"]
    assert m["params"]["info_tags_present"] == ["DP", "QD", "MQ", "FS"]
    assert m["inputs"]["vcf"]["sha256"] and m["inputs"]["vcf_index"]["path"].endswith("public.vcf.gz.tbi")


# ------------------------------------------------------------------ stages 3 and 5 on the recorded stage-2 output

def make_run_from_recorded_stage2(tmp_path: Path) -> Path:
    """A run directory whose ``02_retrieve/`` is the recorded live output: the evidence
    store copied as is, the annotated table rebuilt from ``annotated_rows.json`` with
    the writer's own header (checked against the retriever classes)."""
    from engine.filter.rules import ANNOTATED_COLUMNS

    rows = json.loads((FIX / "02_retrieve" / "annotated_rows.json").read_text())
    assert [list(r) for r in rows] == [list(ANNOTATED_COLUMNS)] * len(rows)
    run = tmp_path / "run"
    stage = run / "02_retrieve"
    shutil.copytree(FIX / "02_retrieve" / "evidence", stage / "evidence")
    with gzip.GzipFile(stage / "variants.annotated.tsv.gz", "wb", mtime=0) as raw, io.TextIOWrapper(raw, encoding="utf-8") as out:
        out.write("\t".join(ANNOTATED_COLUMNS) + "\n")
        for r in rows:
            out.write("\t".join(r[c] for c in ANNOTATED_COLUMNS) + "\n")
    return run


def test_recorded_stage2_output_is_consistent_with_the_vcf():
    rows = json.loads((FIX / "02_retrieve" / "annotated_rows.json").read_text())
    assert [(f"{r['chrom']}:{r['pos']}:{r['ref']}:{r['alt']}", r["id"], r["gene_symbol"], r["gt"]) for r in rows] == ALLELES
    index = json.loads((FIX / "02_retrieve" / "evidence" / "index.json").read_text())
    cited = {rid for r in rows for rid in r["evidence_ids"].split(";") if rid}
    assert cited == set(index) and len(index) == 27
    assert {rid.split(":", 1)[0] for rid in index} == {"vep", "clinvar", "gnomad"}
    assert sum(1 for rid in index if rid.startswith("gnomad:")) == 3  # the three consequential rare alleles only
    for rid, meta in index.items():
        path = FIX / "02_retrieve" / "evidence" / meta["path"]
        rec = json.loads(path.read_text())
        assert rec["record_id"] == rid and rec["url"].startswith("https://") and rec["source_version"]
    assert index["clinvar:VCV000007105"]["source_version"] == "ClinVar 2026-09-05 GRCh38 VCF"
    assert index["vep:" + G542X]["source_version"].startswith("Ensembl VEP 116")
    assert index["gnomad:7-117587778-G-T"]["source_version"].startswith("gnomad_r4")


def test_filter_puts_the_cftr_compound_het_first(tmp_path: Path):
    run = make_run_from_recorded_stage2(tmp_path)
    manifest_path = run_filter(run)
    doc = read_candidates(run)
    ids = [c["candidate_id"] for c in doc["candidates"]]
    assert ids == ["CFTR:comphet", "TP53:het_single"]

    cftr = doc["candidates"][0]
    assert cftr["priority"] == 1 and cftr["gene_symbol"] == "CFTR" and cftr["gene_id"] == "ENSG00000001626"
    assert [v["key"] for v in cftr["variants"]] == [F508DEL, G542X]
    assert [v["clinvar_pathogenicity"] for v in cftr["variants"]] == ["Pathogenic", "Pathogenic"]
    assert [v["clinvar_stars"] for v in cftr["variants"]] == ["4", "4"]
    assert [v["impact"] for v in cftr["variants"]] == ["MODERATE", "HIGH"]
    assert [v["gt"] for v in cftr["variants"]] == ["0/1", "0/1"]
    assert cftr["phase"]["status"] == "unknown" and "no shared PID" in cftr["phase"]["evidence"]
    assert cftr["caveats"] == [] and all(v["caveats"] == [] for v in cftr["variants"])
    assert "clinvar:always_keep=Pathogenic" in cftr["rule_hits"]  # F508del (AF 0.0119) rescued by ClinVar P at 4 stars

    tp53 = doc["candidates"][1]
    assert tp53["priority"] == 2 and [v["key"] for v in tp53["variants"]] == [R175H]
    assert tp53["variants"][0]["clinvar_pathogenicity"] == "Pathogenic"

    c = json.loads(manifest_path.read_text())["counts"]
    assert c["rows_in"] == 12 and c["kept"] == 3 and c["dropped"] == 9
    assert c["dropped_by_rule"] == {"clinvar_benign": 4, "consequence:LOW": 1, "consequence:MODIFIER": 3, "rarity": 1}
    assert c["candidates_by_model"] == {"comphet": 1, "het_single": 1}


def test_reason_dry_run_writes_the_prompt_for_the_cftr_pair(tmp_path: Path):
    from engine.agents.client import FINAL_INSTRUCTION
    from engine.reason.prompts import SYSTEM_PROMPT
    from engine.reason.run import run_reason

    run = make_run_from_recorded_stage2(tmp_path)
    run_filter(run)
    manifest_path = run_reason(run, top_n=3, dry_run=True, case_path=FIX / "case.yaml", cache_root=tmp_path / "cache")
    out = run / "05_reason"
    assert sorted(p.name for p in (out / "prompts").iterdir()) == \
        ["CFTR:comphet.json", "CFTR:comphet.md", "TP53:het_single.json", "TP53:het_single.md"]
    assert not (out / "chains").exists()  # a dry run writes no chain; the script copies the recorded one in before stage 6

    prompt = json.loads((out / "prompts" / "CFTR:comphet.json").read_text())
    assert prompt["system"] == SYSTEM_PROMPT and prompt["final_instruction"] == FINAL_INSTRUCTION
    assert [t["name"] for t in prompt["tool_definitions"]] == ["get_record", "search_literature", "get_paper"]
    assert prompt["output_schema"]["$defs"]["VariantChain"]["properties"]["key"]["enum"] == [F508DEL, G542X]
    user = prompt["user"]
    assert user.startswith("Write the ACMG/AMP evidence chain for the candidate below.")
    assert "# Candidate CFTR:comphet" in user
    assert [line for line in user.splitlines() if line.startswith("## Variant ")] == [f"## Variant {F508DEL}", f"## Variant {G542X}"]
    assert "phase: unknown — no shared PID; 28.2 kb apart" in user
    assert "exomiser: no stage-4 output" in user  # stage 4 was not run here
    assert "case HPO: " + ", ".join(sorted(HPO)) in user
    assert "p.Phe508del" in user and "p.Gly542Ter" in user and "4 stars (practice_guideline)" in user
    for rid in ("clinvar:VCV000007105", "clinvar:VCV000007115", "gnomad:7-117559590-ATCT-A", "gnomad:7-117587778-G-T",
                f"vep:{F508DEL}", f"vep:{G542X}"):
        assert f"[{rid}]" in user
    assert "17:7675088" not in user  # the other candidate's variant is not in this prompt

    m = json.loads(manifest_path.read_text())
    assert m["counts"]["candidates_selected"] == 2 and m["counts"]["candidates_total"] == 2
    assert m["params"]["hpo"] == HPO and m["params"]["hpo_source"].startswith("case file")


# ------------------------------------------------------------------ stage 6 on the recorded stage-5 chain

def cited_ids(text: str) -> list[str]:
    return re.findall(r"\[([a-z]+:[^\]\s]+)\]", text)


def test_recorded_chain_is_stage5_output_over_the_recorded_evidence():
    """``05_reason/chain_CFTR_comphet.json`` is what the real stage 5 wrote for
    ``CFTR:comphet`` over the recorded stage-2 store with a scripted client (no
    literature turn): every id it cites is one of the pair's six records, the
    classification is the engine's own combination of the criteria, and the file is
    the stage's deterministic rendering (sorted keys, indent 1)."""
    from engine.agents.schema import EvidenceChain, combine_acmg

    raw = CHAIN.read_text()
    doc = json.loads(raw)
    assert raw == json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    chain = EvidenceChain.model_validate(doc)
    assert chain.candidate_id == "CFTR:comphet" and [v.key for v in chain.variants] == [F508DEL, G542X]
    assert chain.literature == []  # no paper was fetched, so no pmid: id anywhere and no PS3
    assert all(c.code != "PS3" for v in chain.variants for c in v.criteria)

    index = json.loads((FIX / "02_retrieve" / "evidence" / "index.json").read_text())
    cited = {rid for v in chain.variants for c in v.criteria for rid in c.evidence_ids}
    prose = " ".join([chain.phase_statement, chain.mechanism_hypothesis, *chain.limits, *chain.what_would_change_the_call]
                     + [v.summary for v in chain.variants] + [c.justification for v in chain.variants for c in v.criteria])
    cited |= set(cited_ids(prose))
    assert cited == set(CFTR_PAIR_IDS) and cited <= set(index)
    assert "[citation removed" not in prose  # stage 5 redacted nothing
    for v in chain.variants:
        assert v.classification == combine_acmg(v.criteria), v.key
        codes = [c.code for c in v.criteria]
        assert len(codes) == len(set(codes))
        pm2 = next(c for c in v.criteria if c.code == "PM2")
        assert pm2.met is False and pm2.evidence_ids == [f"gnomad:{v.key.replace(':', '-')}"]  # AF 0.0119 / 0.00036 > 0.0001
        assert next(c for c in v.criteria if c.code == "PP4").evidence_ids == []  # cites the case, as the schema allows
        assert next(c for c in v.criteria if c.code == "PM3").met  # the partner allele, cited by its own records
    by_key = {v.key: v for v in chain.variants}
    assert by_key[F508DEL].classification == "likely_pathogenic" and by_key[G542X].classification == "pathogenic"
    assert [c.code for c in by_key[G542X].criteria if c.code == "PVS1" and c.strength == "very_strong" and c.met] == ["PVS1"]
    assert "cannot show" in chain.phase_statement and "no PID" in chain.phase_statement  # phase unknown, said as such
    assert "no literature search" in " ".join(chain.limits)
    assert "17:7675088" not in raw  # the other candidate's variant is not in this chain


def test_medicine_dry_run_builds_on_the_recorded_chain(tmp_path: Path):
    """What the script does after the stage-5 dry run: copy the recorded chain into
    ``05_reason/chains/`` and run stage 6 dry. Stage 6 reads the chain, finds every
    id it cites in the run's stage-2 store, and writes its bundle, prompt and manifest."""
    from engine.medicine.run import run_medicine
    from engine.reason.run import run_reason

    run = make_run_from_recorded_stage2(tmp_path)
    run_filter(run)
    run_reason(run, top_n=3, dry_run=True, case_path=FIX / "case.yaml", cache_root=tmp_path / "cache")
    chains = run / "05_reason" / "chains"
    chains.mkdir()
    shutil.copy(CHAIN, chains / "CFTR:comphet.json")

    manifest_path = run_medicine(run, dry_run=True, case_path=FIX / "case.yaml", cache_root=tmp_path / "cache")
    out = run / "06_medicine"
    assert manifest_path == out / "manifest.json"
    assert [p.name for p in (out / "bundles").iterdir()] == ["CFTR:comphet.json"]
    assert sorted(p.name for p in (out / "prompts").iterdir()) == ["CFTR:comphet.json", "CFTR:comphet.md"]
    assert not (out / "report.json").exists() and not (out / "report.md").exists()

    bundle = json.loads((out / "bundles" / "CFTR:comphet.json").read_text())
    assert bundle["candidate_id"] == "CFTR:comphet" and bundle["gene_symbol"] == "CFTR" and bundle["gene_id"] == "ENSG00000001626"
    assert bundle["missing_ids"] == [] and set(bundle["record_ids"]) == set(CFTR_PAIR_IDS)
    assert [(v["key"], v["classification"]) for v in bundle["chain"]["variants"]] == \
        [(F508DEL, "likely_pathogenic"), (G542X, "pathogenic")]
    text = bundle["text"]
    assert text.startswith("# Candidate CFTR:comphet\n") and f"## Variant {F508DEL}" in text and f"## Variant {G542X}" in text
    assert "case HPO: " + ", ".join(sorted(HPO)) in text and "17:7675088" not in text

    prompt = json.loads((out / "prompts" / "CFTR:comphet.json").read_text())
    assert [t["name"] for t in prompt["tool_definitions"]] == ["get_record", "search_literature", "get_paper", "drugs_for_gene", "search_trials"]
    assert prompt["output_schema"]["properties"]["candidate_id"]["enum"] == ["CFTR:comphet"]
    assert prompt["output_schema"]["properties"]["gene_symbol"]["enum"] == ["CFTR"]
    assert prompt["user"].endswith(text)

    m = json.loads(manifest_path.read_text())
    assert m["stage"] == "medicine" and m["params"]["dry_run"] is True and m["params"]["candidate"] == "CFTR:comphet"
    assert m["counts"]["chains_available"] == 1 and m["counts"]["reports_written"] == 0
    assert m["params"]["hpo"] == HPO and m["params"]["hpo_source"].startswith("case file")
    assert m["inputs"]["chain"]["path"].endswith("05_reason/chains/CFTR:comphet.json")
    assert m["inputs"]["chain"]["sha256"] == hashlib.sha256(CHAIN.read_bytes()).hexdigest()
    assert not any("not citable" in n for n in m["notes"])


# ------------------------------------------------------------------ stages 5 and 6 for real, with a scripted client

def test_stages_5_and_6_run_on_the_recorded_evidence_with_a_scripted_client(tmp_path: Path):
    """The non-dry stage-5 and stage-6 code paths over the demo, network-free: the
    scripted model reads the pair's records and answers with the recorded chain's
    content (classification left to the engine), and stage 5 reproduces
    ``05_reason/chain_CFTR_comphet.json`` byte for byte — which is how that fixture was
    made. Then stage 6 with the real drug, trial and paper retrievers over the recorded
    Open Targets / DGIdb / ChEMBL / ClinicalTrials.gov fixtures: the model calls
    ``drugs_for_gene`` and ``search_trials``, cites only what came back, and the report
    is validated and rendered with every candidate intact."""
    from engine.agents import client as ac
    from engine.agents.schema import EvidenceChain, MedicineReport
    from engine.medicine.run import run_medicine
    from engine.reason.run import run_reason
    from engine.retrieve.store import EvidenceStore
    from tests.test_medicine import (CH_IVACAFTOR, CH_IVACAFTOR_MEC, N_DRUG_RECORDS, OT_CF, OT_IVACAFTOR, OT_TARGET, TRIAL, TRIALS,
                                     retrievers, stub_http)

    run = make_run_from_recorded_stage2(tmp_path)
    run_filter(run)

    # -- stage 5: top 1 (CFTR:comphet), the chain's content as the scripted answer, no literature turn
    answer = json.loads(CHAIN.read_text())
    for v in answer["variants"]:
        v["classification"] = None
    reader = ac.FakeClient(answer, turns=[ac.FakeTurn([("get_record", {"record_id": rid}) for rid in CFTR_PAIR_IDS], text="reading")])
    manifest_path = run_reason(run, 1, reader, "fake-model", "low", False, http=stub_http(), case_path=FIX / "case.yaml")
    out5 = run / "05_reason"
    assert [c.name for c in reader.calls] == ["get_record"] * 6 and not any(c.is_error for c in reader.calls)
    assert (out5 / "chains" / "CFTR:comphet.json").read_bytes() == CHAIN.read_bytes()
    chain = EvidenceChain.model_validate(json.loads((out5 / "chains" / "CFTR:comphet.json").read_text()))
    assert [(v.key, v.classification) for v in chain.variants] == [(F508DEL, "likely_pathogenic"), (G542X, "pathogenic")]
    validation = json.loads((out5 / "validation" / "CFTR:comphet.json").read_text())
    assert validation["rejections"] == [] and validation.get("redactions", []) == []
    m5 = json.loads(manifest_path.read_text())
    assert m5["counts"]["chains_written"] == 1 and m5["counts"]["candidates_selected"] == 1 and m5["counts"]["rejections"] == 0
    assert m5["counts"]["evidence_records_added"] == 0  # no paper fetched: the stage-5 store stays empty
    assert sorted(p.name for p in (out5 / "transcripts").iterdir()) == ["CFTR:comphet.json"]
    md = (out5 / "evidence_chain.md").read_text()
    assert "CFTR:comphet" in md and "likely pathogenic" in md and f"## Variant {G542X}" in md and "17:7675088" not in md

    # -- stage 6: the real retrievers over the recorded fixtures; the model cites only what the tools returned
    report = {
        "candidate_id": "CFTR:comphet", "gene_symbol": "CFTR",
        "mechanism": [
            {"statement": f"p.Phe508del misfolds CFTR and p.Gly542Ter truncates it: two loss-of-function alleles [vep:{F508DEL}] [vep:{G542X}] [clinvar:VCV000007105]",
             "evidence_ids": [f"vep:{F508DEL}", f"vep:{G542X}", "clinvar:VCV000007105"]},
        ],
        "pathway_targets": [{"statement": f"CFTR's strongest Open Targets association is cystic fibrosis [{OT_CF}]; the target is tractable by small molecules [{OT_TARGET}]",
                             "evidence_ids": [OT_CF, OT_TARGET]}],
        "candidates": [{
            "name": "Ivacaftor", "chembl_id": "CHEMBL2010601", "mechanism_of_action": "CFTR potentiator",
            "approval_status": "approved 2012, cystic fibrosis (ChEMBL max_phase 4; Open Targets APPROVAL)",
            "rationale": f"Potentiates the residual p.Phe508del channel at the membrane [{CH_IVACAFTOR}] [{CH_IVACAFTOR_MEC}] [{OT_IVACAFTOR}]",
            "counter_arguments": ["p.Gly542Ter produces no protein to potentiate; benefit rests on the p.Phe508del allele alone",
                                  "systemic exposure of a child; the trials cited were in other genotype combinations"],
            "evidence_ids": [CH_IVACAFTOR, CH_IVACAFTOR_MEC, OT_IVACAFTOR], "trial_ids": [TRIAL],
        }],
        "follow_up_experiments": [f"Sweat chloride and nasal potential difference after ivacaftor exposure in vitro [{CH_IVACAFTOR}]"],
        "limits": ["No paper was retrieved in this run; the mechanism rests on the variant and drug records alone"],
        "literature": [],
    }
    turns = [ac.FakeTurn([("get_record", {"record_id": f"vep:{F508DEL}"}), ("drugs_for_gene", {"gene": "CFTR"})], text="mechanism first"),
             ac.FakeTurn([("search_trials", {"condition": "cystic fibrosis", "intervention": "ivacaftor", "term": None, "max_results": 3}),
                          ("get_record", {"record_id": OT_CF})], text="drugs and trials")]
    writer = ac.FakeClient(report, turns=turns)
    http = stub_http()
    manifest_path = run_medicine(run, None, writer, "fake-model", "low", False, retrievers=retrievers(http), case_path=FIX / "case.yaml")
    out6 = run / "06_medicine"
    assert [(c.name, c.is_error) for c in writer.calls] == [("get_record", False), ("drugs_for_gene", False), ("search_trials", False), ("get_record", False)]
    drugs = json.loads(writer.calls[1].result)
    assert drugs["ensembl_id"] == "ENSG00000001626" and drugs["opentargets"]["known"] and drugs["dgidb"]["known"] and drugs["chembl"]["known"]
    assert drugs["opentargets"]["diseases"][0]["record_id"] == OT_CF and drugs["opentargets"]["diseases"][0]["disease_name"] == "cystic fibrosis"
    assert json.loads(writer.calls[3].result)["record_id"] == OT_CF
    assert not any("117559590" in json.dumps([u, p, b]) for _, u, p, b in http.calls)  # nothing genomic left the process

    final = MedicineReport.model_validate(json.loads((out6 / "report.json").read_text()))
    assert final.candidate_id == "CFTR:comphet" and final.gene_symbol == "CFTR"
    assert [c.name for c in final.candidates] == ["Ivacaftor"] and final.candidates[0].chembl_id == "CHEMBL2010601"
    assert final.candidates[0].trial_ids == [TRIAL] and len(final.candidates[0].counter_arguments) == 2
    assert [c.evidence_ids for c in final.pathway_targets] == [[OT_CF, OT_TARGET]]
    validation = json.loads((out6 / "validation" / "CFTR:comphet.json").read_text())
    assert validation["rejections"] == []
    md = (out6 / "report.md").read_text()
    assert "Ivacaftor" in md and "cystic fibrosis" in md and "[citation removed" not in md and "17:7675088" not in md

    store = EvidenceStore(out6 / "evidence")
    assert store.count() == N_DRUG_RECORDS + 4 and store.count("nct") == 3 and store.count("pmid") == 0
    assert sorted(r.record_id for r in store.iter() if r.record_id.startswith("nct:")) == sorted(TRIALS)  # store order is by id
    m6 = json.loads(manifest_path.read_text())
    assert m6["counts"]["reports_written"] == 1 and m6["counts"]["evidence_records"] == N_DRUG_RECORDS + 4
    assert m6["inputs"]["chain"]["sha256"] == hashlib.sha256(CHAIN.read_bytes()).hexdigest()  # the chain stage 5 just wrote
    assert m6["params"]["candidate"] == "CFTR:comphet" and m6["params"]["hpo"] == HPO and m6["params"]["dry_run"] is False


# ------------------------------------------------------------------ live: the whole script

@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ENGINE_LIVE_TESTS"), reason="set ENGINE_LIVE_TESTS=1 to hit Ensembl VEP, gnomAD (and Docker)")
def test_script_runs_the_public_case_end_to_end(tmp_path: Path):
    """Runs ``scripts/run_public_case.sh`` with a cold cache into a temporary run
    directory, on the default path: stage 5 dry, the recorded chain copied in, stage 6
    dry (no model, no key). Stage 4 runs only when Docker answers; otherwise the
    script skips it and says so (PUBLIC_SKIP_RANK)."""
    docker = subprocess.run(["docker", "info"], capture_output=True, text=True, check=False).returncode == 0
    env = {**os.environ, "PUBLIC_RUN_DIR": str(tmp_path / "run"), "PUBLIC_CACHE": str(tmp_path / "cache")}
    env.pop("PUBLIC_LIVE_MODEL", None)
    if not docker:
        env["PUBLIC_SKIP_RANK"] = "1"
    result = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=env, cwd=REPO, check=False, timeout=900)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    out = result.stdout
    run = tmp_path / "run"

    assert "candidate #1 is CFTR:comphet: YES" in out
    assert "copied the recorded stage-5 chain into" in out and "no model was called" in out
    assert "medicine  ok (dry run: bundle and prompt on the recorded chain, no model)" in out
    assert "06_medicine candidate CFTR:comphet (CFTR)" in out and "0 cited id(s) missing from the run" in out
    ids = [c["candidate_id"] for c in read_candidates(run)["candidates"]]
    assert ids[:2] == ["CFTR:comphet", "TP53:het_single"]
    assert [v["key"] for v in read_candidates(run)["candidates"][0]["variants"]] == [F508DEL, G542X]

    retrieve = json.loads((run / "02_retrieve" / "manifest.json").read_text())["counts"]
    assert retrieve["unique_variants"] == 12 and retrieve["vep_records"] == 12 and retrieve["clinvar_records"] == 12
    assert retrieve["gnomad_records"] >= 2 and retrieve["http"]["live_requests"] > 0
    assert (run / "05_reason" / "prompts" / "CFTR:comphet.json").exists()
    reason = json.loads((run / "05_reason" / "manifest.json").read_text())
    assert reason["params"]["dry_run"] is True and reason["counts"]["chains_written"] == 0
    assert (run / "05_reason" / "chains" / "CFTR:comphet.json").read_bytes() == CHAIN.read_bytes()
    medicine = json.loads((run / "06_medicine" / "manifest.json").read_text())
    assert medicine["params"]["dry_run"] is True and medicine["params"]["candidate"] == "CFTR:comphet"
    assert medicine["inputs"]["chain"]["sha256"] == hashlib.sha256(CHAIN.read_bytes()).hexdigest()
    assert (run / "06_medicine" / "bundles" / "CFTR:comphet.json").exists()
    assert json.loads((run / "06_medicine" / "bundles" / "CFTR:comphet.json").read_text())["missing_ids"] == []
    if docker:
        joined = json.loads((run / "04_rank" / "joined.json").read_text())
        ranks = {c["candidate_id"]: c["exomiser_rank"] for c in joined["candidates"]}
        assert ranks["CFTR:comphet"] == 1 and joined["hpo"] == HPO
    else:
        assert "skipped: PUBLIC_SKIP_RANK=1" in out and not (run / "04_rank" / "manifest.json").exists()


def test_public_fixture_files_are_committable_but_nothing_else_vcf_shaped_is():
    """The repository ignores every VCF and case file by design; the nested .gitignore
    re-includes only the four public fixture files."""
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout")

    def ignored(path: str) -> bool:
        return subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", path], check=False).returncode == 0

    for name in ("public.vcf", "public.vcf.gz", "public.vcf.gz.tbi", "case.yaml", "README.md", "02_retrieve/annotated_rows.json",
                 "05_reason/chain_CFTR_comphet.json"):
        assert not ignored(f"tests/fixtures/public_case/{name}"), name
    for path in ("tests/fixtures/public_case/other.vcf", "tests/fixtures/public_case/case.proband.yaml",
                 # the same four names below the fixture directory: the patterns are anchored, so still ignored
                 "tests/fixtures/public_case/02_retrieve/case.yaml", "tests/fixtures/public_case/02_retrieve/public.vcf",
                 "tests/fixtures/public_case/x/public.vcf.gz", "tests/fixtures/public_case/sub/dir/public.vcf.gz.tbi",
                 "tests/fixtures/public_case/05_reason/case.yaml",
                 "tests/fixtures/other.vcf.gz", "tests/case.yaml", "case.yaml", "data/proband.vcf.gz",
                 "work/r/01_ingest/variants.tsv.gz"):
        assert ignored(path), path
