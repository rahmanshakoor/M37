"""Agent core: the scripted client, the validator, the bundle and the renderers.

Network-free. The evidence store under tests/fixtures/agents/evidence/ holds public
records only (MTHFR rs1801133, CFTR p.Phe508del, TP53 p.Arg175His, two papers, one
trial; one ChEMBL molecule record sits beside it). The Anthropic SDK is exercised through a fake ``client``
object that records every call the loop makes and answers with real SDK message
types, plus one real SDK client pointed at a closed local port; one live test talks
to the API when ``ENGINE_LIVE_TESTS`` is set.

Variant keys: only the three public variants above appear as real coordinates; the
keys ``MT:1:A:G`` and ``1:5:N:A`` are deliberately impossible spellings, not variants.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
from pydantic import BaseModel

from engine.agents import client as ac
from engine.agents.bundle import build_bundle, load_rank, project_columns, write_bundle
from engine.agents.render import cited_ids, render_evidence_chain, render_medicine_report
from engine.agents.schema import ALL_CODES, Criterion, EvidenceChain, MechanismClaim, MedicineReport, VariantChain, combine_acmg
from engine.agents.validator import (
    EvidenceIndex, canonical_key, citation_tokens, frequency_criterion_met, gnomad_record_id, observed_frequency,
    validate,
)
from engine.retrieve.gnomad import GnomadError
from engine.retrieve.http import HttpError
from engine.retrieve.store import EvidenceRecord, EvidenceStore

FIXTURES = Path(__file__).parent / "fixtures" / "agents" / "evidence"
LUMACAFTOR_RECORD = Path(__file__).parent / "fixtures" / "agents" / "chembl_CHEMBL2103870.json"
"""A live-fetched ChEMBL molecule record kept *beside* the store: stages 5 and 6 copy
the store as their stage-2 evidence, and stage 6 retrieves this molecule itself."""

CFTR = "7:117559590:ATCT:A"
CFTR_GNOMAD = "gnomad:7-117559590-ATCT-A"
CFTR_VEP = f"vep:{CFTR}"
CFTR_CLINVAR = "clinvar:VCV000007105"
MTHFR = "1:11796321:G:A"
MTHFR_GNOMAD = "gnomad:1-11796321-G-A"
MTHFR_VEP = f"vep:{MTHFR}"
TP53 = "17:7675088:C:T"
TP53_GNOMAD = "gnomad:17-7675088-C-T"
TP53_VEP = f"vep:{TP53}"
TP53_CLINVAR = "clinvar:VCV000012374"
PAPER = "pmid:7647779"
VX809_PAPER = "pmid:21976485"     # Van Goor 2011, PMC3219147, doi 10.1073/pnas.1105787108
LUMACAFTOR = "chembl:CHEMBL2103870"
TRIAL = "nct:NCT01807923"
FAKE_PMID = "pmid:99999999"       # Europe PMC answers hitCount 0 for it (tests/fixtures/literature)
FAKE_CLINVAR = "clinvar:VCV999999999"
FAKE_TRIAL = "nct:NCT99999999"
MT_KEY = "MT:1:A:G"               # an impossible spelling: gnomAD's variant() cannot be asked about MT


# ------------------------------------------------------------------------ fixtures

@pytest.fixture
def store() -> EvidenceStore:
    return EvidenceStore(FIXTURES)


@pytest.fixture
def index(store: EvidenceStore) -> EvidenceIndex:
    return EvidenceIndex([store])


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    shutil.copytree(FIXTURES, run / "02_retrieve" / "evidence")
    return run


def index_without(store: EvidenceStore, *ids: str) -> EvidenceIndex:
    """The fixture store minus the named records — the store as stage 2 leaves it
    when gnomAD was not asked (prefiltered) or answered "not found"."""
    return EvidenceIndex.from_records(r for r in store.iter() if r.record_id not in ids)


def variant(key: str, ids: list[str], gt: str = "0/1", **extra: Any) -> dict[str, Any]:
    v = {"key": key, "gt": gt, "ad": "21,25", "dp": "46", "gq": "99", "quality_flag": "",
         "af_used": "", "af_source": "", "caveats": [], "evidence_ids": ids}
    v.update(extra)
    return v


def cftr_candidate() -> dict[str, Any]:
    return {"candidate_id": "CFTR:hom", "gene_symbol": "CFTR", "gene_id": "ENSG00000001626", "model": "hom",
            "priority": 1, "variants": [variant(CFTR, [CFTR_VEP, CFTR_CLINVAR, CFTR_GNOMAD], gt="1/1",
                                                af_used="0.0119", af_source="gnomad_af")],
            "phase": {"status": "n/a", "evidence": "homozygous"}, "rule_hits": ["consequence:MODERATE"], "caveats": []}


def criterion(code: str, strength: str, ids: list[str], met: bool = True, just: str = "because") -> dict[str, Any]:
    return {"code": code, "strength": strength, "met": met, "justification": just, "evidence_ids": ids}


def chain(criteria: list[dict[str, Any]], key: str = CFTR, **extra: Any) -> dict[str, Any]:
    d = {"candidate_id": "CFTR:hom", "variants": [{"key": key, "criteria": criteria, "summary": "s"}],
         "phase_statement": "homozygous; phase not in question", "mechanism_hypothesis": f"LoF [{CFTR_VEP}]",
         "limits": [], "what_would_change_the_call": [], "literature": []}
    d.update(extra)
    return d


def get_record_tool(index: EvidenceIndex) -> ac.ToolSpec:
    def handler(inputs: dict[str, Any]) -> dict[str, Any]:
        rec = index.get(inputs["record_id"])
        if rec is None:
            raise KeyError(f"no record {inputs['record_id']}")
        return asdict(rec)
    return ac.ToolSpec("get_record", "Fetch one evidence record by id.",
                       {"type": "object", "properties": {"record_id": {"type": "string"}}}, handler)


def failing_tool(name: str, exc: Exception) -> ac.ToolSpec:
    def handler(inputs: dict[str, Any]) -> Any:
        raise exc
    return ac.ToolSpec(name, "Always fails.", {"type": "object", "properties": {"q": {"type": "string"}}}, handler)


def request(index: EvidenceIndex, model: type[BaseModel] = EvidenceChain, **kw: Any) -> ac.AgentRequest:
    kw.setdefault("tools", [get_record_tool(index)])
    return ac.AgentRequest(system="You reason over records.", user="bundle text", output_model=model, max_turns=3, **kw)


# ------------------------------------------------------------------- happy path

def test_happy_path_chain_validates_cleanly_and_renders(index: EvidenceIndex):
    out = chain([
        criterion("PS3", "strong", [PAPER], just="functional study [pmid:7647779]"),
        criterion("PM2", "moderate", [CFTR_GNOMAD], met=False, just="af 0.0119 is not rare"),
        criterion("PP3", "supporting", [CFTR_VEP], met=False, just="CADD 17.6, below the 25.3 calibration"),
        criterion("PM4", "moderate", [CFTR_VEP], just=f"in-frame deletion of one residue; ClinVar concordant [{CFTR_CLINVAR}]"),
        criterion("PP4", "supporting", [], just="sweat chloride phenotype"),
    ], literature=[PAPER])
    fake = ac.FakeClient(out, turns=[ac.FakeTurn([("get_record", {"record_id": CFTR_GNOMAD}),
                                                   ("get_record", {"record_id": "nope:1"})], text="looking")])
    result = fake.run(request(index))

    assert isinstance(result.output, EvidenceChain)
    assert [c.name for c in fake.calls] == ["get_record", "get_record"]
    assert not fake.calls[0].is_error and '"variant_id": "7-117559590-ATCT-A"' in fake.calls[0].result
    assert fake.calls[1].is_error and fake.calls[1].result.startswith("Error: KeyError")
    assert result.transcript[0].tool_calls[0].id == "toolu_fake_1_1"
    assert result.transcript[-1].stop_reason == "end_turn" and result.final_text == result.output.model_dump_json()
    assert "no API call" in result.disclosure
    assert (result.usage.tool_calls, result.usage.tool_errors, result.usage.api_calls) == (2, 1, 0)
    assert result.request["tools"] == ["get_record"] and result.request["max_turns"] == 3
    assert result.as_dict()["request"]["thinking"] == {"type": "adaptive"}

    cleaned, report = validate(result.output, index)
    assert report.clean and report.rejections == [] and report.disputes == [] and report.notes == []
    assert report.counts["frequency_recomputed"] == 1 and report.counts["frequency_disputed"] == 0
    assert report.counts["computational_recomputed"] == 1 and report.counts["computational_disputed"] == 0
    crit = cleaned.variants[0].criteria
    assert [c.code for c in crit] == ["PS3", "PM2", "PP3", "PM4", "PP4"]
    assert cleaned.variants[0].classification == combine_acmg(crit) == "likely_pathogenic"  # the engine's, not the model's
    assert cleaned.variants[0].points == 7 and cleaned.variants[0].classification_richards_2015 == "likely_pathogenic"
    assert report.rules["af_field"] == "gnomad_af" and report.rules["PM2"] == "met iff af < 0.0001 or absent"

    md = render_evidence_chain(cleaned, index, disclosure=ac.disclosure("claude-opus-5", "high"))
    assert "## Variant 7:117559590:ATCT:A — likely pathogenic" in md
    assert f"[{CFTR_CLINVAR}]" in md and f"[{PAPER}] Frosst P et al. (1995) Nature genetics" in md
    assert "(case-level criterion; cites no record)" in md
    refs = md.split("## References")[1]
    assert f"- [{CFTR_CLINVAR}] — https://www.ncbi.nlm.nih.gov/clinvar/variation/7105/" in refs
    assert f"- [{CFTR_GNOMAD}] — https://gnomad.broadinstitute.org/variant/7-117559590-ATCT-A?dataset=gnomad_r4" in refs
    assert f"- [{PAPER}] — https://europepmc.org/article/MED/7647779" in refs
    assert cited_ids(cleaned, index) == sorted({PAPER, CFTR_GNOMAD, CFTR_VEP, CFTR_CLINVAR})
    assert md.rstrip().endswith("_Anthropic API, model claude-opus-5, effort high; API inputs are not used for "
                                "training under Anthropic's commercial terms_")
    assert render_evidence_chain(cleaned, index) == render_evidence_chain(cleaned, index)


# ------------------------------------------------------------------- fabrication

def test_fabricated_evidence_id_drops_the_criterion(index: EvidenceIndex):
    out = chain([
        criterion("PP3", "supporting", [CFTR_VEP]),
        criterion("PS1", "strong", [FAKE_CLINVAR, CFTR_CLINVAR], just="same amino acid change"),
        criterion("PM1", "moderate", [], just="hot spot, no record"),
    ])
    result = ac.FakeClient(out).run(request(index))
    assert FAKE_CLINVAR in result.output.model_dump_json()  # the client does not check; the validator does

    cleaned, report = validate(result.output, index)
    assert [c.code for c in cleaned.variants[0].criteria] == ["PP3"]
    assert [(r.path, r.reason) for r in report.rejections] == [
        ("variants[0].criteria[1]", f"unknown evidence id(s): {FAKE_CLINVAR}"),
        ("variants[0].criteria[2]", "PM1 cites no evidence record and is not a case-level criterion"),
    ]
    assert report.counts["items_dropped"] == 2 and report.counts["ids_unknown"] == 1
    assert FAKE_CLINVAR not in render_evidence_chain(cleaned, index)
    assert cleaned.variants[0].classification == "vus"


def test_fabricated_pmid_is_dropped_everywhere(index: EvidenceIndex):
    out = chain(
        [criterion("PS3", "strong", [FAKE_PMID], just="see PMID 99999999"),
         criterion("PP3", "supporting", [CFTR_VEP], just="in silico; also PMID 7647779 and PMID: 99999999")],
        literature=[PAPER, FAKE_PMID, "7647779", "not-a-pmid"],
        mechanism_hypothesis=f"LoF [{CFTR_VEP}] [{FAKE_PMID}]",
    )
    cleaned, report = validate(out, index)
    assert cleaned.literature == [PAPER]
    assert [c.code for c in cleaned.variants[0].criteria] == ["PP3"]
    pp3 = cleaned.variants[0].criteria[0]
    assert pp3.met is False and pp3.justification.startswith("[DISPUTED — PP3 recomputed from vep:7:117559590:ATCT:A: CADD 17.55")
    assert pp3.justification.endswith("in silico; also PMID 7647779 and PMID [citation removed: no such record]")
    assert cleaned.mechanism_hypothesis == f"LoF [{CFTR_VEP}] [citation removed: no such record]"
    reasons = {r.path: r.reason for r in report.rejections}
    assert reasons["literature[1]"] == f"no such literature record in the store: {FAKE_PMID}"
    assert reasons["literature[3]"] == "not a pmid:<n> id: 'not-a-pmid'"
    assert reasons["variants[0].criteria[0]"] == f"unknown evidence id(s): {FAKE_PMID}"
    assert report.counts["literature_removed"] == 2 and report.counts["redactions"] == 2
    assert "99999999" not in render_evidence_chain(cleaned, index)


def test_inline_citations_are_found_in_every_spelling_and_redacted_per_id(index: EvidenceIndex):
    prose = (f"pmid:99999999 shows X; (pmid:99999999); pmid 99999999; [ {FAKE_CLINVAR} ] with spaces; "
             f"see gnomad:9-9-A-T. Also [PMID:7647779], [pmid:7647779, pmid:99999999] and [{CFTR_VEP}; {FAKE_PMID}]. "
             f"Bare {CFTR_GNOMAD}: fine. HP:0002205 and chr7:117559590 and NM_000492.4:c.1521_1523del stay.")
    out = chain([criterion("PP3", "supporting", [CFTR_VEP])], phase_statement=prose)
    cleaned, report = validate(out, index)
    r = "[citation removed: no such record]"
    assert cleaned.phase_statement == (
        f"{r} shows X; ({r}); PMID {r}; {r} with spaces; see {r}. Also [pmid:7647779], [pmid:7647779] {r} and "
        f"[{CFTR_VEP}] {r}. Bare {CFTR_GNOMAD}: fine. HP:0002205 and chr7:117559590 and NM_000492.4:c.1521_1523del stay.")
    assert report.counts["redactions"] == 7 and "99999999" not in render_evidence_chain(cleaned, index)
    assert all(rj.path == "phase_statement" for rj in report.rejections)
    # the renderer's inventory uses the same tokenizer: References lists exactly what survived
    assert cited_ids(cleaned, index) == sorted({PAPER, CFTR_VEP, CFTR_GNOMAD})
    assert citation_tokens(f"x [PMID:7647779] y pmid:1 z [hp:1, HP:2] HP:3 [{CFTR_VEP}] VCV000007105 NCT01807923 PMC1 rs1") == \
        ["pmid:7647779", "pmid:1", CFTR_VEP, CFTR_CLINVAR, TRIAL]  # HP: is a term, not a source; bare accessions map to records


def test_bracketed_ontology_terms_are_not_citations(index: EvidenceIndex):
    prose = f"the proband shows [HP:0002205] and [HP:0006528, HP:0002205]; also [OMIM:219700, {CFTR_CLINVAR}] and [sic]"
    out = chain([criterion("PP4", "supporting", [], just="phenotype [HP:0002205] fits")], phase_statement=prose)
    cleaned, report = validate(out, index)
    assert cleaned.phase_statement == f"the proband shows [HP:0002205] and [HP:0006528, HP:0002205]; also [OMIM:219700] [{CFTR_CLINVAR}] and [sic]"
    assert cleaned.variants[0].criteria[0].justification == "phenotype [HP:0002205] fits"
    assert report.rejections == [] and report.counts["redactions"] == 0
    assert report.counts["identifiers_unverified"] == 5
    assert report.notes[0] == "variants[0].criteria[0].justification: bracketed identifier of no evidence source, left in place (not a citation): HP:0002205"
    assert "OMIM:219700" in report.notes[-1]
    md = render_evidence_chain(cleaned, index)
    assert "[HP:0006528, HP:0002205]" in md and "HP:0002205" not in md.split("## References")[1]


def test_bare_accessions_in_prose_resolve_through_the_store_or_are_redacted(index: EvidenceIndex):
    """``VCV…``/``NCT…`` name records directly; ``PMC…``/``doi:`` resolve through the
    pmid: records' own ids; ``rs`` ids are identifiers, checked against the cited
    records and only noted; a numbered ``[1]`` names nothing."""
    r = "[citation removed: no such record]"
    prose = ("ClinVar VCV999999999 and VCV000007105 (vcv000007105); trials NCT99999999 and NCT01807923; "
             "PubMed 99999999 and PubMed ID 7647779 and PMC9999999 and PMC3219147 (see doi:10.1073/pnas.1105787108, "
             "doi:10.1000/fake.123); dbSNP rs999999999 and rs1801133; refs [1] and [2, 3] and [4-6]; "
             f"ChEMBL CHEMBL999999 and CHEMBL2103870 (a record) and CHEMBL2010601 (carried); DOI 10.1038/ng0595-111 stays [{MTHFR_VEP}].")
    index.add(EvidenceRecord.from_json(LUMACAFTOR_RECORD.read_text()))
    index.add(EvidenceRecord(record_id="opentargets:knownDrug:stub", source="opentargets", source_version="test stub", query={},
                             url="https://platform.opentargets.org/drug/CHEMBL2010601", retrieved_at="1970-01-01T00:00:00+00:00",
                             payload={"drug": {"id": "CHEMBL2010601", "name": "IVACAFTOR"}}))
    out = chain([criterion("PP3", "supporting", [CFTR_VEP, "opentargets:knownDrug:stub"])], mechanism_hypothesis=prose)
    cleaned, report = validate(out, index)
    assert cleaned.mechanism_hypothesis == (
        f"ClinVar {r} and VCV000007105 (vcv000007105); trials {r} and NCT01807923; "
        f"PMID {r} and PubMed ID 7647779 and {r} and PMC3219147 [{VX809_PAPER}] (see doi:10.1073/pnas.1105787108, "
        f"{r}); dbSNP rs999999999 and rs1801133; refs {r} and {r} and {r}; "
        f"ChEMBL {r} and CHEMBL2103870 (a record) and CHEMBL2010601 (carried); DOI 10.1038/ng0595-111 stays [{MTHFR_VEP}].")
    reasons = [rj.reason for rj in report.rejections]
    assert reasons == [
        "inline accession not in the store: clinvar:VCV999999999",
        "inline accession not in the store: nct:NCT99999999",
        "inline PMID not in the store: pmid:99999999",
        "inline PMC id not carried by any pmid: record in the store: PMC9999999",
        "inline DOI not carried by any pmid: record in the store: 10.1000/fake.123",
        "numbered reference [1] names no record",
        "numbered reference [2, 3] names no record",
        "numbered reference [4-6] names no record",
        "inline ChEMBL id neither a chembl: record in the store nor carried by a cited record: CHEMBL999999",
    ]
    assert all(rj.path == "mechanism_hypothesis" for rj in report.rejections) and report.counts["redactions"] == 9
    assert report.notes == ["mechanism_hypothesis: dbSNP id not carried by any record the answer cites, left in place "
                            "(not a citation): rs999999999"]  # rs1801133 is in the cited MTHFR VEP record
    assert report.counts["identifiers_unverified"] == 1
    refs = render_evidence_chain(cleaned, index).split("## References")[1]
    assert f"[{CFTR_CLINVAR}]" in refs and f"[{TRIAL}]" in refs and f"[{VX809_PAPER}]" in refs and f"[{PAPER}]" in refs
    assert "99999999" not in refs and "fake" not in refs
    assert cited_ids(cleaned, index) == sorted({CFTR_VEP, MTHFR_VEP, CFTR_CLINVAR, TRIAL, PAPER, VX809_PAPER, "opentargets:knownDrug:stub"})


# ------------------------------------------------------------------ frequencies

def test_pm2_disputed_and_recomputed_from_gnomad(index: EvidenceIndex):
    out = {
        "candidate_id": "MIX", "phase_statement": "p", "mechanism_hypothesis": "m",
        "variants": [
            {"key": MTHFR, "summary": "", "criteria": [
                criterion("PM2", "moderate", [MTHFR_GNOMAD], met=True, just="rare"),   # af 0.318: wrong
                criterion("BA1", "stand_alone", [MTHFR_GNOMAD], met=False, just="not common"),  # wrong
                criterion("BS2", "strong", [], met=False, just="cites nothing"),         # dropped before any recompute
                criterion("BS1", "strong", [MTHFR_GNOMAD], met=True, just="common"),   # right
            ]},
            {"key": TP53, "summary": "", "criteria": [
                criterion("PM2", "moderate", [TP53_GNOMAD], met=True, just="7 alleles in 1.6M"),  # right
            ]},
            {"key": CFTR, "summary": "", "criteria": [
                criterion("BS1", "strong", [CFTR_GNOMAD], met=True, just="af 0.0119 > 0.01"),   # right
                criterion("BA1", "stand_alone", [CFTR_GNOMAD], met=True, just="common"),        # wrong
            ]},
        ],
    }
    cleaned, report = validate(out, index)
    mthfr, tp53, cftr = cleaned.variants

    pm2, ba1, bs1 = mthfr.criteria
    assert pm2.met is False and pm2.justification.startswith("[DISPUTED — PM2 recomputed from gnomad:1-11796321-G-A: af=0.318")
    assert pm2.justification.endswith("the model said met] rare")
    assert ba1.met is True and ba1.justification.startswith("[DISPUTED — BA1 recomputed from gnomad:1-11796321-G-A")
    assert bs1.met is True and not bs1.justification.startswith("[DISPUTED")
    assert tp53.criteria[0].met is True and not tp53.criteria[0].justification.startswith("[DISPUTED")
    assert cftr.criteria[0].met is True and cftr.criteria[1].met is False

    assert [(d.path, d.code, d.claimed_met, d.recomputed_met) for d in report.disputes] == [
        ("variants[0].criteria[0]", "PM2", True, False),
        ("variants[0].criteria[1]", "BA1", False, True),
        ("variants[2].criteria[1]", "BA1", True, False),
    ]
    assert report.disputes[0].af == pytest.approx(513548 / 1613846)
    assert report.counts["frequency_recomputed"] == 6 and report.counts["frequency_disputed"] == 3
    assert [(r.path, r.reason) for r in report.rejections] == [
        ("variants[0].criteria[2]", "BS2 cites no evidence record and is not a case-level criterion")]
    assert report.thresholds == {"ba1_min_af": 0.05, "bs1_min_af": 0.01, "pm2_max_af": 0.0001}
    assert (mthfr.classification, tp53.classification, cftr.classification) == ("benign", "vus", "likely_benign")
    assert (mthfr.points, tp53.points, cftr.points) == (-12, 2, -4)  # BS1 alone is likely benign in the point system
    assert cftr.classification_richards_2015 == "vus"
    md = render_evidence_chain(cleaned, index)
    assert "- **PM2** · moderate · not met — [DISPUTED — PM2 recomputed" in md


def test_frequency_criterion_citing_another_variants_gnomad_record_is_rejected(index: EvidenceIndex):
    """The engine recomputes from the record for the variant's *key*; a PM2 on the common
    MTHFR allele that points at TP53's gnomAD record cannot borrow its rarity."""
    out = chain([
        criterion("PM2", "moderate", [TP53_GNOMAD], met=True, just="absent"),      # another variant's record
        criterion("PS3", "strong", [PAPER]),
        criterion("PP3", "supporting", [MTHFR_VEP]),
        criterion("BA1", "stand_alone", [MTHFR_VEP], met=False, just="not common"),  # cites no gnomad: at all
    ], key=MTHFR)
    cleaned, report = validate(out, index)
    assert [c.code for c in cleaned.variants[0].criteria] == ["PS3", "PP3", "BA1"]
    assert [(r.path, r.reason) for r in report.rejections] == [
        ("variants[0].criteria[0]", f"PM2 cites the gnomAD record of a different variant: {TP53_GNOMAD}")]
    ba1 = cleaned.variants[0].criteria[2]
    assert ba1.met is True and ba1.evidence_ids == [MTHFR_VEP, MTHFR_GNOMAD]  # recomputed from the key's own record
    assert report.disputes[0].record_id == MTHFR_GNOMAD and report.counts["frequency_disputed"] == 1
    assert cleaned.variants[0].classification == "benign"  # BA1 stands alone whatever PS3 + PP3 add up to
    assert cleaned.variants[0].points == -8 + 4 + 1 and cleaned.variants[0].criteria[1].strength == "supporting"  # stated below REVEL 0.842's moderate: honoured

    swapped = chain([criterion("BA1", "stand_alone", [MTHFR_GNOMAD], met=True, just="common")], key=CFTR)
    cleaned, report = validate(swapped, index)
    assert cleaned.variants[0].criteria == [] and cleaned.variants[0].classification == "vus"
    assert report.rejections[0].reason == f"BA1 cites the gnomAD record of a different variant: {MTHFR_GNOMAD}"


def test_frequency_falls_back_to_veps_copy_of_gnomad(store: EvidenceStore):
    """Stage 2 fetches no gnomAD record for a variant VEP already shows as common; the
    VEP record still carries the frequency, so BA1 is still checked."""
    index = index_without(store, MTHFR_GNOMAD)
    out = chain([criterion("PM2", "moderate", [MTHFR_VEP], met=True, just="absent from gnomAD"),
                 criterion("BA1", "stand_alone", [MTHFR_VEP], met=False, just="not common")], key=MTHFR)
    cleaned, report = validate(out, index)
    pm2, ba1 = cleaned.variants[0].criteria
    assert pm2.met is False and ba1.met is True
    assert pm2.justification.startswith(f"[DISPUTED — PM2 recomputed from {MTHFR_VEP}: VEP's copy of gnomAD: "
                                        "exomes 0.3227, genomes 0.2752; max 0.323 used; met iff af < 0.0001 or absent → not met")
    assert [d.record_id for d in report.disputes] == [MTHFR_VEP, MTHFR_VEP]
    assert report.counts["frequency_from_vep"] == 2 and report.counts["frequency_recomputed"] == 2
    assert report.disputes[0].af == pytest.approx(0.3227)
    assert cleaned.variants[0].classification == "benign"
    obs = observed_frequency(MTHFR, index)
    assert obs is not None and obs.record.record_id == MTHFR_VEP and obs.af == pytest.approx(0.3227)


def test_frequency_criterion_without_any_record_is_unverified_and_not_met(store: EvidenceStore):
    """What the store cannot check does not count: a PM2 the model says is met, with
    no gnomAD or VEP record for the key, is set to not met — it never reaches the
    combining rules as pathogenic evidence."""
    index = index_without(store, TP53_GNOMAD, TP53_VEP)
    out = chain([criterion("PM2", "moderate", [TP53_CLINVAR], met=True, just="not seen in gnomAD"),
                 criterion("PS3", "strong", [PAPER]),
                 criterion("PP5", "supporting", [TP53_CLINVAR]),
                 criterion("BS1", "strong", [TP53_CLINVAR], met=False, just="?")], key=TP53)
    cleaned, report = validate(out, index)
    pm2, ps3, pp5, bs1 = cleaned.variants[0].criteria
    reason = f"no gnomAD or VEP record for {TP53} in the store; PM2 cannot be checked → not met; the model said met"
    assert pm2.met is False and pm2.justification == f"[UNVERIFIED — {reason}] not seen in gnomAD"
    assert bs1.met is False and bs1.justification.startswith("[UNVERIFIED — ") and "the model said not met" in bs1.justification
    assert [(d.path, d.code, d.record_id, d.claimed_met, d.recomputed_met, d.af) for d in report.disputes] == \
        [("variants[0].criteria[0]", "PM2", None, True, False, None)]  # BS1 was already not met: no dispute
    assert report.counts["frequency_unverified"] == 2 and report.counts["frequency_recomputed"] == 0
    assert report.counts["frequency_disputed"] == 1
    assert report.notes[0] == f"variants[0].criteria[0]: PM2 not recomputed — {reason}"
    assert cleaned.variants[0].classification == "vus"  # PS3 + PP5 alone; with PM2 it would have been likely pathogenic
    assert report.rules["unverified"].endswith("→ not met, marked [UNVERIFIED]")
    assert f"- **PM2** · moderate · not met — [UNVERIFIED — {reason}] not seen in gnomAD" in render_evidence_chain(cleaned, index)
    assert observed_frequency(TP53, index) is None and observed_frequency(MT_KEY, index) is None

    mito = chain([criterion("PM2", "moderate", [TP53_CLINVAR], met=True)], key=MT_KEY)  # gnomAD cannot be asked
    cleaned, report = validate(mito, index)
    assert cleaned.variants[0].criteria[0].met is False and cleaned.variants[0].criteria[0].justification.startswith("[UNVERIFIED")


def test_variant_keys_are_pinned_to_the_candidates(index: EvidenceIndex):
    """A respelled key is matched back to the engine's spelling so the frequency
    recompute finds the record; a key that is not the candidate's is dropped with
    its criteria — it can neither borrow the candidate's verdict nor escape the
    recompute under a spelling the store does not know."""
    out = {
        "candidate_id": "CFTR:hom", "phase_statement": "p", "mechanism_hypothesis": "m",
        "variants": [
            {"key": "chr7:117559590:ATCT:A", "summary": "", "criteria": [   # the model's spelling
                criterion("PM2", "moderate", [CFTR_CLINVAR], met=True, just="rare"),   # af 0.0119: wrong
                criterion("PS3", "strong", [PAPER])]},
            {"key": "7-117559590-atct-a", "summary": "", "criteria": [criterion("PP3", "supporting", [CFTR_VEP])]},  # same variant again
            {"key": TP53, "summary": "", "criteria": [criterion("PM2", "moderate", [TP53_GNOMAD], met=True)]},  # not this candidate's
            {"key": "not a key", "summary": "", "criteria": []},
        ],
    }
    cleaned, report = validate(out, index, keys=[CFTR])
    assert [v.key for v in cleaned.variants] == [CFTR]
    pm2 = cleaned.variants[0].criteria[0]
    assert pm2.met is False and pm2.justification.startswith(f"[DISPUTED — PM2 recomputed from {CFTR_GNOMAD}: af=0.0119")
    assert pm2.evidence_ids == [CFTR_CLINVAR, CFTR_GNOMAD] and cleaned.variants[0].classification == "vus"
    assert report.counts["keys_respelled"] == 1 and report.counts["items_dropped"] == 3
    assert [(r.path, r.reason) for r in report.rejections] == [
        ("variants[1]", f"a second entry for variant {CFTR!r}; the first is kept"),
        ("variants[2]", f"key {TP53!r} is not a variant of this candidate ({CFTR}); the entry and its 1 criteria were dropped"),
        ("variants[3]", f"key 'not a key' is not a variant of this candidate ({CFTR}); the entry and its 0 criteria were dropped"),
    ]
    assert report.notes[0] == f"variants[0].key: the model wrote 'chr7:117559590:ATCT:A'; matched to {CFTR!r}"

    # without keys: any real key is accepted, canonically spelled; the recompute still finds the record
    cleaned, report = validate(out, index)
    assert [v.key for v in cleaned.variants] == [CFTR, TP53, "not a key"]
    assert cleaned.variants[0].criteria[0].met is False and cleaned.variants[1].criteria[0].met is True
    assert [r.path for r in report.rejections] == ["variants[1]"]
    assert canonical_key("chrM:1:A:G") == MT_KEY and canonical_key("1:11796321:g:a") == MTHFR
    assert canonical_key("7:117559590") is None and canonical_key("7:x:A:T") is None and canonical_key("Q:1:A:T") is None


def test_strength_is_capped_at_what_the_code_may_carry(index: EvidenceIndex):
    """The engine combines at the stated strength, so an inflated one would set the
    verdict; a code counts at most at its ACMG default (PP3/BP4: strong), a lower
    stated strength is honoured."""
    out = chain([criterion("PP3", "very_strong", [CFTR_VEP], just="REVEL 0.99"),
                 criterion("PM1", "moderate", [CFTR_CLINVAR]),
                 criterion("PM2", "supporting", [CFTR_GNOMAD], met=False, just="PM2_Supporting"),
                 criterion("BP4", "strong", [CFTR_VEP], met=False),
                 criterion("BS1", "stand_alone", [CFTR_GNOMAD], met=True, just="af 0.0119 > 0.01")])
    cleaned, report = validate(out, index)
    pp3, pm1, pm2, bp4, bs1 = cleaned.variants[0].criteria
    assert (pp3.strength, pm1.strength, pm2.strength, bp4.strength, bs1.strength) == \
        ("strong", "moderate", "supporting", "strong", "strong")
    # capped to strong first, then recomputed: the record holds no REVEL and CADD 17.55, so the claimed PP3 is disputed
    assert pp3.met is False and pp3.justification.startswith("[DISPUTED — PP3 recomputed from vep:7:117559590:ATCT:A: CADD 17.55")
    assert pp3.justification.endswith("[STRENGTH CAPPED — PP3 at most strong (a code counts at most at its ACMG/AMP 2015 default "
                                      "level (PP3/BP4 at most strong)); the model said very_strong] REVEL 0.99")
    assert bp4.met is False and not bp4.justification.startswith("[DISPUTED")  # not met and the scores support nothing: agreed
    assert bs1.justification.startswith("[STRENGTH CAPPED — BS1 at most strong")
    assert pm2.justification == "PM2_Supporting"  # a downgrade is the model's call
    assert report.counts["strength_capped"] == 2 and report.rejections == []
    assert [n for n in report.notes if ".strength:" in n] == [
        "variants[0].criteria[0].strength: PP3 at most strong (a code counts at most at its ACMG/AMP 2015 default level "
        "(PP3/BP4 at most strong)); the model said very_strong; lowered to strong",
        "variants[0].criteria[4].strength: BS1 at most strong (a code counts at most at its ACMG/AMP 2015 default level "
        "(PP3/BP4 at most strong)); the model said stand_alone; lowered to strong",
    ]
    # PP3 was disputed away (no calibrated score), so what counts is PM1 (+2) against BS1 (−4): −2 → likely benign
    assert cleaned.variants[0].classification == "likely_benign" and cleaned.variants[0].points == -2
    assert "strength_cap" in report.rules and "- **PP3** · strong · not met — [DISPUTED" in render_evidence_chain(cleaned, index)

    inflated = chain([criterion("BP4", "stand_alone", [CFTR_VEP])])
    cleaned, _ = validate(inflated, index)
    assert cleaned.variants[0].criteria[0].strength == "strong" and cleaned.variants[0].classification == "vus"


def test_af_field_is_a_recorded_parameter(index: EvidenceIndex):
    out = chain([criterion("BS1", "strong", [CFTR_GNOMAD], met=False, just="?"),
                 criterion("BA1", "stand_alone", [CFTR_GNOMAD], met=False, just="?")])
    cleaned, report = validate(out, index, af_field="gnomad_faf95_popmax")
    bs1, ba1 = cleaned.variants[0].criteria
    assert bs1.met is True and ba1.met is False  # faf95 0.01476: > 0.01, not > 0.05
    assert bs1.justification.startswith("[DISPUTED — BS1 recomputed from gnomad:7-117559590-ATCT-A: gnomad_faf95_popmax=0.0148")
    assert report.rules["af_field"] == "gnomad_faf95_popmax" and report.disputes[0].af == pytest.approx(0.01475724)
    assert report.as_dict()["rules"]["BS1"] == "met iff af > 0.01"
    with pytest.raises(ValueError, match="af_field"):
        validate(out, index, af_field="gnomad_nhom")


def test_frequency_helpers():
    th = {"ba1_min_af": 0.05, "bs1_min_af": 0.01, "pm2_max_af": 0.0001}
    assert frequency_criterion_met("PM2", None, th) and frequency_criterion_met("PM2", 5e-5, th)
    assert not frequency_criterion_met("PM2", 1e-4, th)
    assert frequency_criterion_met("BS1", 0.011, th) and not frequency_criterion_met("BS1", 0.01, th)
    assert frequency_criterion_met("BA1", 0.051, th) and not frequency_criterion_met("BA1", None, th)
    with pytest.raises(ValueError):
        frequency_criterion_met("PP3", 0.1, th)
    assert gnomad_record_id(MTHFR) == MTHFR_GNOMAD
    assert gnomad_record_id(MT_KEY) is None and gnomad_record_id("1:5:N:A") is None


# ----------------------------------------------------------- criteria arithmetic

def test_classification_is_the_engines_never_the_models(index: EvidenceIndex):
    out = chain([criterion("PS3", "strong", [PAPER]), criterion("PM1", "moderate", [CFTR_CLINVAR])])
    out["variants"][0]["classification"] = "benign"
    cleaned, report = validate(out, index)
    assert cleaned.variants[0].classification == "likely_pathogenic"
    assert report.counts["classification_replaced"] == 1
    assert report.notes == [("variants[0].classification: the model said 'benign'; replaced by the engine's "
                             "'likely_pathogenic' (+6 SVI points over the met criteria)")]
    assert "— likely pathogenic" in render_evidence_chain(cleaned, index)

    empty = chain([criterion("PS1", "strong", [FAKE_CLINVAR])])
    empty["variants"][0]["classification"] = "pathogenic"
    cleaned, _ = validate(empty, index)
    assert cleaned.variants[0].criteria == [] and cleaned.variants[0].classification == "vus"

    same = validate(EvidenceChain.model_validate(dict(out, variants=[dict(out["variants"][0], classification="likely_pathogenic")])), index)
    assert same[0].variants[0].classification == "likely_pathogenic" and "same verdict" in same[1].notes[0]


def test_duplicate_criterion_codes_count_once(index: EvidenceIndex):
    out = chain([criterion("PS3", "strong", [FAKE_PMID]),      # dropped: fabricated → the next PS3 is "first"
                 criterion("PS3", "strong", [PAPER]),
                 criterion("ps3", "strong", [CFTR_CLINVAR]),   # same code, spelled differently
                 criterion("PS3", "strong", [CFTR_VEP])])
    cleaned, report = validate(out, index)
    assert [(c.code, c.evidence_ids) for c in cleaned.variants[0].criteria] == [("PS3", [PAPER])]
    assert [r.reason for r in report.rejections] == [
        f"unknown evidence id(s): {FAKE_PMID}",
        "duplicate criterion PS3 (the first occurrence is kept)",
        "duplicate criterion PS3 (the first occurrence is kept)",
    ]
    assert report.counts["duplicates_dropped"] == 2
    assert cleaned.variants[0].classification == "vus"  # one PS3, not "pathogenic" from ps >= 2


def test_missing_citation_keys_and_unknown_codes_are_judged_not_exempt(index: EvidenceIndex):
    out = chain([{"code": "PS1", "strength": "strong", "met": True, "justification": "no key at all"},
                 {"code": "PVS1", "strength": "very_strong", "met": True, "justification": "none"},
                 {"code": "PP4", "strength": "supporting", "met": True, "justification": "case-level, no key"},
                 {"code": "PM2_SUPPORTING", "strength": "supporting", "met": True, "justification": "modified code",
                  "evidence_ids": [CFTR_GNOMAD]}])
    cleaned, report = validate(out, index)
    assert [c.code for c in cleaned.variants[0].criteria] == ["PP4"] and cleaned.variants[0].criteria[0].evidence_ids == []
    assert [r.reason for r in report.rejections] == [
        "PS1 cites no evidence record and is not a case-level criterion",
        "PVS1 cites no evidence record and is not a case-level criterion",
        "unknown ACMG code 'PM2_SUPPORTING'",
    ]
    assert cleaned.variants[0].classification == "vus"

    # the same hole through the scripted client, which may hand back an unvalidated dict
    result = ac.FakeClient(out, validate_output=False).run(request(index))
    cleaned, report = validate(result.output, index)
    assert [c.code for c in cleaned.variants[0].criteria] == ["PP4"] and report.counts["items_dropped"] == 3


# --------------------------------------------------------------- medicine report

def test_medicine_report_drops_one_sided_and_unverified_candidates(index: EvidenceIndex):
    good = {"name": "Lumacaftor", "chembl_id": "chembl2103870", "mechanism_of_action": "CFTR corrector",
            "approval_status": "approved (F508del homozygous, with ivacaftor)", "rationale": f"targets the folding defect [{CFTR_CLINVAR}]",
            "counter_arguments": ["modest FEV1 gain", " "], "evidence_ids": [CFTR_CLINVAR, CFTR_VEP], "trial_ids": [TRIAL]}
    one_sided = dict(good, name="Ivacaftor alone", counter_arguments=[])
    fake_trial = dict(good, name="Elexacaftor", trial_ids=[TRIAL, FAKE_TRIAL])
    bare_trial = dict(good, name="Bare NCT", chembl_id="CHEMBL9999999", trial_ids=["NCT01807923", "nct:nct01807923"])
    # a drug-source record (stub of an Open Targets known-drug row; ivacaftor is CHEMBL2010601) that names the id
    index.add(EvidenceRecord(record_id="opentargets:knownDrug:stub", source="opentargets", source_version="test stub",
                             query={}, url="https://platform.opentargets.org/drug/CHEMBL2010601", retrieved_at="1970-01-01T00:00:00+00:00",
                             payload={"drug": {"id": "CHEMBL2010601", "name": "IVACAFTOR"}}))
    carried = dict(good, name="Ivacaftor (carried by a cited record)", chembl_id="CHEMBL2010601",
                   evidence_ids=[CFTR_VEP, "opentargets:knownDrug:stub"])
    uncited = {k: v for k, v in good.items() if k not in ("evidence_ids", "trial_ids")}
    uncited["name"] = "No citations at all"
    out = {
        "candidate_id": "CFTR:hom", "gene_symbol": "CFTR",
        "mechanism": [{"statement": f"misfolded CFTR is degraded [{CFTR_VEP}]", "evidence_ids": [CFTR_VEP]},
                      {"statement": "made up", "evidence_ids": [FAKE_CLINVAR]},
                      {"statement": "uncited", "evidence_ids": []},
                      {"statement": "totally uncited mechanism"}],
        "pathway_targets": [],
        "candidates": [one_sided, good, fake_trial, bare_trial, uncited, carried],
        "follow_up_experiments": ["sweat chloride after 4 weeks"], "limits": ["no PK data in the store"],
        "literature": [PAPER, FAKE_PMID],
    }
    fake = ac.FakeClient(out, validate_output=False)  # a pydantic MedicineReport could not even hold []
    result = fake.run(request(index, MedicineReport))
    assert isinstance(result.output, dict) and result.output["candidates"][0]["counter_arguments"] == []

    index.add(EvidenceRecord.from_json(LUMACAFTOR_RECORD.read_text()))  # chembl:CHEMBL2103870, a record in the store
    cleaned, report = validate(result.output, index)
    assert isinstance(cleaned, MedicineReport)
    assert [d.name for d in cleaned.candidates] == ["Lumacaftor", "Bare NCT", "Ivacaftor (carried by a cited record)"]
    assert cleaned.candidates[0].counter_arguments == ["modest FEV1 gain"]
    assert cleaned.candidates[0].chembl_id == "CHEMBL2103870"  # a chembl: record in the store, spelled as the record
    assert cleaned.candidates[1].trial_ids == [TRIAL]  # a bare NCT id is the same claim, spelled as the record
    assert cleaned.candidates[1].chembl_id is None  # no such record, no cited record carries it
    assert cleaned.candidates[2].chembl_id == "CHEMBL2010601"  # no chembl: record, but a cited record's payload names it
    assert [m.statement for m in cleaned.mechanism] == [f"misfolded CFTR is degraded [{CFTR_VEP}]"]
    assert cleaned.literature == [PAPER]
    reasons = {r.path: r.reason for r in report.rejections}
    assert reasons["candidates[0]"] == "drug candidate has no counter-arguments"
    assert reasons["candidates[2]"] == f"trial id(s) not nct: records in the store: {FAKE_TRIAL}"
    assert reasons["candidates[3].chembl_id"] == ("chembl_id 'CHEMBL9999999' is neither a chembl: record in the store nor "
                                                  "carried by a record the candidate cites; cleared")
    assert reasons["candidates[4]"] == "cites no evidence record"
    assert reasons["mechanism[1]"] == f"unknown evidence id(s): {FAKE_CLINVAR}"
    assert reasons["mechanism[2]"] == reasons["mechanism[3]"] == "cites no evidence record"
    assert "candidates[3]" not in reasons and "candidates[5]" not in reasons and report.counts["chembl_cleared"] == 1

    md = render_medicine_report(cleaned, index, disclosure="FakeClient")
    assert md.index("## Mechanism") < md.index("## Drug candidates") < md.index("## Follow-up experiments") < md.index("## Limits")
    assert "### 1. Lumacaftor (CHEMBL2103870)" in md and "### 2. Bare NCT\n" in md and "Ivacaftor alone" not in md
    assert "CHEMBL9999999" not in md
    assert f"- misfolded CFTR is degraded [{CFTR_VEP}]\n" in md  # cited inline once, not repeated after the statement
    assert f"- trials: [{TRIAL}]" in md and "  - modest FEV1 gain" in md
    assert f"- [{TRIAL}] — https://clinicaltrials.gov/study/NCT01807923" in md
    assert f"- [{LUMACAFTOR}] — https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL2103870" in md
    assert "99999999" not in md and "totally uncited" not in md


def test_validate_rejects_malformed_object(index: EvidenceIndex):
    with pytest.raises(TypeError):
        validate({"nothing": 1}, index)
    with pytest.raises(TypeError):
        validate("text", index)  # type: ignore[arg-type]


def test_index_add_and_ids_cover_memory_and_stores(store: EvidenceStore):
    """A record added in memory is a member and is listed once with the stores' ids."""
    index = EvidenceIndex([store])
    on_disk = index.ids()
    assert on_disk == sorted(r.record_id for r in store.iter()) and CFTR_VEP in on_disk
    rec = store.get(CFTR_VEP)
    new = EvidenceRecord(record_id="pmid:1", source="europepmc", source_version="v", query={}, url="https://x",
                         retrieved_at="2026-01-01T00:00:00+00:00", payload={"title": "t"})
    index.add(new)
    index.add(rec)  # already on disk: not listed twice
    assert "pmid:1" in index and index.get("pmid:1") is new and index.url("pmid:1") == "https://x"
    assert index.ids() == sorted(on_disk + ["pmid:1"]) and 12345 not in index and index.get("nope:0") is None


def test_render_medicine_report_with_no_surviving_candidate_says_so(index: EvidenceIndex):
    report = MedicineReport(candidate_id="CFTR:hom", gene_symbol="CFTR",
                            mechanism=[MechanismClaim(statement=f"LoF [{CFTR_VEP}]", evidence_ids=[CFTR_VEP])],
                            candidates=[], limits=["no drug record cited survived"])
    md = render_medicine_report(report, index, disclosure="FakeClient")
    assert "## Drug candidates\n\n- none survived validation\n" in md
    assert md.index("## Mechanism") < md.index("## Drug candidates") < md.index("## Limits")


# -------------------------------------------------------------------------- bundle

def test_bundle_says_when_the_store_holds_no_gnomad_record(run_dir: Path):
    """A variant stage 2 never sent to gnomAD (prefiltered as common on VEP's copy):
    the prompt says so and names what stage 3 used instead of a frequency line."""
    shutil.rmtree(run_dir / "02_retrieve" / "evidence" / "gnomad")
    idx = json.loads((run_dir / "02_retrieve" / "evidence" / "index.json").read_text())
    (run_dir / "02_retrieve" / "evidence" / "index.json").write_text(json.dumps({k: v for k, v in idx.items() if not k.startswith("gnomad:")}))
    cand = cftr_candidate()
    cand["variants"][0].update(evidence_ids=[CFTR_VEP, CFTR_CLINVAR], af_used="0.0119", af_source="vep_gnomade_af")
    b = build_bundle(cand, run_dir, [])
    assert "gnomAD: no record in the store (af_used 0.0119 from vep_gnomade_af)" in b.text
    assert CFTR_GNOMAD not in b.record_ids and b.variants[0].missing_ids == []


def test_bundle_is_deterministic_and_complete(run_dir: Path):
    cand = cftr_candidate()
    cand["variants"][0]["evidence_ids"].append("gnomad:9-9-A-T")  # listed by stage 3, not in the store
    a = build_bundle(cand, run_dir, ["HP:0006528", "HP:0002205"])
    b = build_bundle(cand, run_dir, ["HP:0002205", "HP:0006528"])
    assert a.to_json() == b.to_json() and a.text == b.text
    p1 = write_bundle(a, run_dir / "05_reason")
    p2 = write_bundle(b, run_dir / "05_reason")
    assert p1 == p2 == run_dir / "05_reason" / "bundles" / "CFTR:hom.json"
    assert p1.read_bytes() == a.to_json().encode()

    doc = json.loads(p1.read_text())
    assert doc["candidate_id"] == "CFTR:hom" and doc["case_hpo"] == ["HP:0002205", "HP:0006528"]
    assert doc["record_ids"] == [CFTR_CLINVAR, CFTR_GNOMAD, CFTR_VEP]
    assert doc["rank"] is None and doc["gene_records"] == []
    v = doc["variants"][0]
    assert [r["record_id"] for r in v["records"]] == [CFTR_CLINVAR, CFTR_GNOMAD, CFTR_VEP]
    assert v["records"][1]["payload"]["joint"]["ac"] == 19237  # the full payload, not a summary
    assert v["missing_ids"] == ["gnomad:9-9-A-T"]
    assert v["columns"]["hgvsp"] == "ENSP00000003084.6:p.Phe508del" and v["columns"]["clinvar_stars"] == "4"
    assert v["columns"]["gnomad_nhom"] == "58" and v["columns"]["clinvar_revstat"] == "practice_guideline"

    text = a.text
    assert text.startswith("# Candidate CFTR:hom\ngene: CFTR (ENSG00000001626) · model: hom · priority: 1\n")
    assert "exomiser: no stage-4 output" in text and "case HPO: HP:0002205, HP:0006528" in text
    assert "## Variant 7:117559590:ATCT:A\ngenotype: 1/1 · AD 21,25 · DP 46 · GQ 99 · filter: PASS" in text
    assert f"c.1521_1523del · ENSP00000003084.6:p.Phe508del [{CFTR_VEP}]" in text
    assert "gnomAD: af 0.01193 (ac 19237 / an 1612320) · hom 58 · " in text and f"[{CFTR_GNOMAD}]" in text
    assert "stage-3 AF: 0.0119 from gnomad_af" in text
    assert "ClinVar: Pathogenic · 4 stars (practice_guideline) · " in text and f"[{CFTR_CLINVAR}]" in text
    assert "listed but not in the store: gnomad:9-9-A-T" in text
    assert text.rstrip().endswith(f"citable record ids: [{CFTR_CLINVAR}] [{CFTR_GNOMAD}] [{CFTR_VEP}]")
    assert "retrieved_at" not in text and "2026-" not in text  # no clock in the prompt


def test_bundle_reads_rank_and_shortlist_and_candidates_json(run_dir: Path):
    cand = cftr_candidate()
    cand["variants"][0]["evidence_ids"] = []  # canonical ids are found even when stage 3 listed none
    (run_dir / "03_filter").mkdir()
    (run_dir / "03_filter" / "candidates.json").write_text(json.dumps({"candidates": [cand], "config": {}, "counts": {}}))
    with gzip.open(run_dir / "03_filter" / "shortlist.tsv.gz", "wt") as f:
        f.write("chrom\tpos\tref\talt\tgt\tpid\tgnomad_nhom\tmodel\n7\t117559590\tATCT\tA\t1/1\t\t58\thom\n")
    (run_dir / "04_rank").mkdir()
    (run_dir / "04_rank" / "joined.json").write_text(json.dumps({"candidates": [
        {"candidate_id": "CFTR:hom", "gene_symbol": "CFTR", "exomiser_rank": 1, "exomiser_score": 0.98,
         "phenotype_score": 0.9, "moi": "AR", "extra": "ignored"},
        {"candidate_id": "OTHER:hom", "gene_symbol": "OTHER", "exomiser_rank": None}]}))

    b = build_bundle("CFTR:hom", run_dir, [])
    assert b.rank == {"exomiser_rank": 1, "exomiser_score": 0.98, "phenotype_score": 0.9, "moi": "AR"}
    assert "exomiser: rank 1 · combined 0.98 · phenotype 0.9 · moi AR\n" in b.text
    assert "case HPO: none given" in b.text
    assert b.record_ids == [CFTR_GNOMAD, CFTR_VEP]  # clinvar is not a canonical id and was not listed
    cols = b.variants[0].columns
    assert "pid" not in cols  # empty row cells do not overwrite projections
    assert cols["model"] == "hom" and cols["gnomad_nhom"] == "58" and cols["gene_symbol"] == "CFTR"
    assert load_rank(run_dir, "OTHER:hom", "OTHER") == {"exomiser_rank": None}  # stage 4 says: not ranked
    with pytest.raises(ValueError, match="no entry for candidate 'NOPE:hom'"):  # stage 4 says nothing: a mismatch, not "unranked"
        load_rank(run_dir, "NOPE:hom", "NOPE")
    with pytest.raises(ValueError, match="no entry for candidate"):
        build_bundle(dict(cand, candidate_id="NOPE:hom", gene_symbol="NOPE"), run_dir)
    with pytest.raises(KeyError):
        build_bundle("NOPE:hom", run_dir)
    with pytest.raises(FileNotFoundError):
        build_bundle(cand, run_dir / "nowhere")

    (run_dir / "04_rank" / "joined.json").write_text(json.dumps({"ranking": []}))  # a shape this reader does not know
    with pytest.raises(ValueError, match="candidates"):
        load_rank(run_dir, "CFTR:hom", "CFTR")


def test_bundle_carries_the_stage4_exomiser_record_in_the_writers_shape(run_dir: Path):
    """``engine.rank.run`` spells the MOI ``exomiser_moi``, names the rank's record in
    ``exomiser_evidence_id`` and writes it to ``04_rank/evidence``: the bundle tells
    the model it may cite it, and the validator accepts the citation."""
    ranking = {"rank": 1, "gene_symbol": "CFTR", "entrez_id": "1080", "moi": "AR", "exomiser_score": 0.98,
               "phenotype_score": 0.9, "variant_score": 0.8, "n_variants": 1, "variants": [CFTR], "by_moi": {}}
    rank_store = EvidenceStore(run_dir / "04_rank" / "evidence")
    rank_store.put(EvidenceRecord(record_id="exomiser:CFTR", source="exomiser", source_version="exomiser-cli 14.0.0 · data 2406",
                                  query={"hpoIds": ["HP:0006528"], "genomeAssembly": "hg38", "analysis_yaml_sha256": "0"},
                                  url="https://www.ncbi.nlm.nih.gov/gene/1080", retrieved_at="2026-09-12T00:00:00+00:00",
                                  payload={"ranking": ranking, "citation": "Exomiser"}))
    rank_store.write_index()
    entry = dict(cftr_candidate(), exomiser_rank=1, exomiser_score=0.98, phenotype_score=0.9, variant_score=0.8,
                 exomiser_moi="AR", exomiser_gene_symbol="CFTR", exomiser_variants=[CFTR], exomiser_variants_matched=[CFTR],
                 exomiser_by_moi={}, exomiser_match="symbol", exomiser_evidence_id="exomiser:CFTR")
    (run_dir / "04_rank" / "joined.json").write_text(json.dumps({"candidates": [entry], "exomiser_only": [], "hpo": ["HP:0006528"]}))

    b = build_bundle(cftr_candidate(), run_dir, ["HP:0006528"])
    assert b.rank == {"exomiser_rank": 1, "exomiser_score": 0.98, "phenotype_score": 0.9, "variant_score": 0.8,
                      "moi": "AR", "evidence_id": "exomiser:CFTR"}
    assert [r["record_id"] for r in b.gene_records] == ["exomiser:CFTR"] and b.gene_records[0]["payload"]["ranking"]["rank"] == 1
    assert b.record_ids == [CFTR_CLINVAR, "exomiser:CFTR", CFTR_GNOMAD, CFTR_VEP]
    assert "exomiser: rank 1 · combined 0.98 · phenotype 0.9 · variant 0.8 · moi AR [exomiser:CFTR]\n" in b.text
    assert b.text.rstrip().endswith(f"citable record ids: [{CFTR_CLINVAR}] [exomiser:CFTR] [{CFTR_GNOMAD}] [{CFTR_VEP}]")

    index = EvidenceIndex.from_run(run_dir)
    assert [s.root for s in index.stores] == [run_dir / "02_retrieve" / "evidence", run_dir / "04_rank" / "evidence"]
    assert "exomiser:CFTR" in index and "exomiser" in index.sources()
    out = chain([criterion("PP4", "supporting", ["exomiser:CFTR"], just="phenotype match, exomiser:CFTR rank 1")])
    cleaned, report = validate(out, index)
    assert report.clean and cleaned.variants[0].criteria[0].evidence_ids == ["exomiser:CFTR"]
    assert "- [exomiser:CFTR] — https://www.ncbi.nlm.nih.gov/gene/1080" in render_evidence_chain(cleaned, index)


def test_project_columns_pins_the_clinvar_projection_stand_in(store: EvidenceStore):
    """``ClinvarRetriever.extract`` is called with a stand-in ``self`` that has only
    ``columns``; this test fails the day the projection needs anything else."""
    rec = store.get(CFTR_CLINVAR)
    cols = project_columns([rec])
    assert cols["clinvar_vcv"] == "VCV000007105" and cols["clinvar_pathogenicity"] == "Pathogenic"
    assert cols["clinvar_stars"] == "4" and cols["clinvar_revstat"] == "practice_guideline"
    assert set(cols) <= {"clinvar_vcv", "clinvar_clnsig", "clinvar_revstat", "clinvar_stars", "clinvar_conditions",
                         "clinvar_alleleid", "clinvar_hgvs", "clinvar_pathogenicity", "clinvar_release"}


# ------------------------------------------------------------ Anthropic client

class FakeSdk:
    """Stands in for ``anthropic.Anthropic()`` with the SDK's streaming shape: every
    call is ``messages.stream(...)``, the stream carries ``request_id`` and the
    message ``get_final_message()`` returns carries none. Loop turns are answered
    from a script of real SDK ``Message`` objects; the final structured call is
    recognised by ``output_config.format``. ``messages.create`` refuses, so the
    tests prove nothing goes unstreamed."""

    def __init__(self, turns: list[Message], final: Any = None, raise_on_stream: Exception | None = None):
        self.turns = list(turns)
        self.final = final
        self.raise_on_stream = raise_on_stream
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create, stream=self._stream)

    @property
    def loop_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if "format" not in c["output_config"]]

    @property
    def final_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if "format" in c["output_config"]]

    def _create(self, **kw: Any) -> Any:
        raise AssertionError("messages.create must not be used: every call is streamed")

    @contextmanager
    def _stream(self, **kw: Any):
        self.calls.append(dict(kw, messages=list(kw["messages"])))  # the loop appends to its list
        if self.raise_on_stream is not None:
            raise self.raise_on_stream
        if "format" in kw["output_config"]:
            message, rid = self.final, "req_final"
        else:
            scripted = self.turns.pop(0) if len(self.turns) > 1 else self.turns[0]
            message, rid = scripted.model_copy(), scripted._request_id
            message._request_id = None  # as the SDK: a streamed message carries no request id
        yield SimpleNamespace(get_final_message=lambda: message, request_id=rid)


def msg(content: list[Any], stop: str, rid: str = "req_1", **extra: Any) -> Message:
    m = Message(id="msg_1", type="message", role="assistant", model="claude-opus-5", content=content,
                stop_reason=stop, stop_sequence=None, usage=Usage(input_tokens=100, output_tokens=10), **extra)
    m._request_id = rid  # what FakeSdk serves on the stream for this turn
    return m


def final_message(answer: Any, stop: str = "end_turn", category: str | None = None) -> SimpleNamespace:
    """The final call's message: ``answer`` is a pydantic object or the raw text. No
    ``_request_id`` — the SDK's streamed message has none; the stream has it."""
    text = answer.model_dump_json() if isinstance(answer, BaseModel) else str(answer)
    details = SimpleNamespace(type="refusal", category=category) if category else None
    return SimpleNamespace(content=[TextBlock(type="text", text=text)], stop_reason=stop, stop_details=details,
                           usage=Usage(input_tokens=500, output_tokens=50, cache_read_input_tokens=400))


def test_anthropic_client_tool_loop_and_final_parse(index: EvidenceIndex):
    answer = EvidenceChain.model_validate(chain([criterion("PP3", "supporting", [CFTR_VEP])]))
    sdk = FakeSdk([
        msg([TextBlock(type="text", text="I will look."),
             ToolUseBlock(type="tool_use", id="toolu_1", name="get_record", input={"record_id": CFTR_GNOMAD}),
             ToolUseBlock(type="tool_use", id="toolu_2", name="get_record", input={"record_id": "nope:0"}),
             ToolUseBlock(type="tool_use", id="toolu_3", name="not_a_tool", input={})], "tool_use", rid="req_a"),
        msg([TextBlock(type="text", text="Done.")], "end_turn", rid="req_b"),
    ], final=final_message(answer))
    seen: list[str] = []
    req = request(index)
    result = ac.AnthropicClient(sdk, log=seen.append).run(req)

    assert result.output == answer and result.stop_reason == "end_turn"
    assert result.model == "claude-opus-5" and result.effort == "high"
    assert result.disclosure == ("Anthropic API, model claude-opus-5, effort high; API inputs are not used for "
                                 "training under Anthropic's commercial terms")
    assert asdict(result.usage) == {"input_tokens": 700, "output_tokens": 70, "cache_read_input_tokens": 400,
                                    "cache_creation_input_tokens": 0, "api_calls": 3, "tool_calls": 3, "tool_errors": 2}
    assert seen == ["turn 1: stop_reason=tool_use tool_calls=3", "turn 2: stop_reason=end_turn tool_calls=0"]
    assert result.request == {"model": "claude-opus-5", "effort": "high", "thinking": {"type": "adaptive"}, "max_turns": 3,
                              "max_tokens": 32000, "final_max_tokens": 16000, "tools": ["get_record"],
                              "tool_definitions_sha256": ac.sha256_json([get_record_tool(index).param()]),
                              "output_model": "EvidenceChain", "output_schema_sha256": ac.sha256_json(req.output_schema),
                              "api": "messages.stream"}
    assert result.as_dict()["request"] == result.request
    assert req.params()["output_schema_sha256"] != ac.AgentRequest(system="s", user="u", output_model=EvidenceChain,
                                                                  output_schema=ac.answer_schema(EvidenceChain)).params()["output_schema_sha256"]

    # request shape, as the SDK skill documents it; every call streamed
    assert len(sdk.calls) == 3 and len(sdk.loop_calls) == 2 and len(sdk.final_calls) == 1
    first = sdk.loop_calls[0]
    assert first["model"] == "claude-opus-5" and first["max_tokens"] == 32000
    assert first["thinking"] == {"type": "adaptive"} and first["output_config"] == {"effort": "high"}
    assert first["cache_control"] == {"type": "ephemeral"} and first["system"] == "You reason over records."
    assert first["tools"] == [{"name": "get_record", "description": "Fetch one evidence record by id.", "strict": True,
                               "input_schema": {"type": "object", "properties": {"record_id": {"type": "string"}},
                                                "additionalProperties": False, "required": ["record_id"]}}]
    assert first["messages"] == [{"role": "user", "content": "bundle text"}]

    # all three tool results went back in ONE user message, errors flagged
    second = sdk.loop_calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert [b.type for b in second[1]["content"]] == ["text", "tool_use", "tool_use", "tool_use"]
    results = second[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["toolu_1", "toolu_2", "toolu_3"]
    assert "is_error" not in results[0] and '"variant_id": "7-117559590-ATCT-A"' in results[0]["content"]
    assert results[1] == {"type": "tool_result", "tool_use_id": "toolu_2", "content": "Error: KeyError: 'no record nope:0'", "is_error": True}
    assert results[2] == {"type": "tool_result", "tool_use_id": "toolu_3", "content": "Error: unknown tool 'not_a_tool'", "is_error": True}

    # final call: structured by output_config.format, its own max_tokens, tools declared but not callable
    fin = sdk.final_calls[0]
    assert fin["output_config"] == {"effort": "high", "format": {"type": "json_schema", "schema": req.output_schema}}
    assert "output_format" not in fin and fin["max_tokens"] == 16000 and fin["tool_choice"] == {"type": "none"}
    assert fin["messages"][-1] == {"role": "user", "content": ac.FINAL_INSTRUCTION}
    assert fin["messages"][-2]["role"] == "assistant" and fin["cache_control"] == {"type": "ephemeral"}

    # transcript: every tool call, its result, the request ids — read from the streams, never from the messages
    t1, t2, t3 = result.transcript
    assert (t1.n, t1.stop_reason, t1.text, t1.request_id) == (1, "tool_use", "I will look.", "req_a")
    assert [(c.name, c.is_error) for c in t1.tool_calls] == [("get_record", False), ("get_record", True), ("not_a_tool", True)]
    assert (t2.stop_reason, t2.text, t2.request_id) == ("end_turn", "Done.", "req_b")
    assert t3.request_id == "req_final" and t3.text == answer.model_dump_json() == result.final_text
    assert result.as_dict()["transcript"][0]["tool_calls"][0]["input"] == {"record_id": CFTR_GNOMAD}


def test_anthropic_client_without_tools_or_system_still_streams_both_calls(index: EvidenceIndex):
    answer = EvidenceChain.model_validate(chain([]))
    sdk = FakeSdk([msg([TextBlock(type="text", text="ok")], "end_turn")], final=final_message(answer))
    req = ac.AgentRequest(system="", user="u", output_model=EvidenceChain)  # no tools, no system
    result = ac.AnthropicClient(sdk).run(req)
    assert result.output == answer and result.request["api"] == "messages.stream"
    assert "system" not in sdk.loop_calls[0] and "tools" not in sdk.loop_calls[0]
    fin = sdk.final_calls[0]
    assert "tool_choice" not in fin and fin["output_config"]["format"]["schema"] == req.output_schema
    assert fin["cache_control"] == {"type": "ephemeral"} and fin["max_tokens"] == 16000


def test_real_sdk_client_against_a_closed_port_maps_to_agent_error(index: EvidenceIndex, monkeypatch: pytest.MonkeyPatch):
    """The real ``anthropic.Anthropic`` client, pointed at a port nothing listens on:
    the request is built and sent by the SDK itself (a streamed call at the final
    call's 64000 max_tokens, which the SDK would refuse unstreamed), the connection
    is refused, and the loop reports one :class:`AgentError` — with the SDK's
    non-streaming guard exercised alongside for the record."""
    sdk = anthropic.Anthropic(api_key="sk-ant-dummy", base_url="http://127.0.0.1:9", max_retries=0, timeout=2.0)
    req = ac.AgentRequest(system="s", user="u", output_model=EvidenceChain, tools=[get_record_tool(index)], max_tokens=64000)
    with pytest.raises(ac.AgentError, match="^could not reach the Anthropic API"):
        ac.AnthropicClient(sdk).run(req)
    guarded = anthropic.Anthropic(api_key="sk-ant-dummy", base_url="http://127.0.0.1:9", max_retries=0)  # default timeout
    with pytest.raises(ValueError, match="Streaming is required"):
        guarded.messages.create(model="claude-opus-5", max_tokens=64000, messages=[{"role": "user", "content": "u"}])
    with pytest.raises(ac.AgentError, match="^could not reach the Anthropic API"):
        ac.AnthropicClient(guarded).run(req)  # the same client streams here, so the guard never fires

    # the SDK's own refusal to send (no credentials resolved) is an AgentError too, not a bare TypeError
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    bare = anthropic.Anthropic(base_url="http://127.0.0.1:9", max_retries=0, timeout=2.0)
    try:
        ac.AnthropicClient(bare).run(req)
    except ac.AgentError as e:
        assert str(e).startswith(("the Anthropic SDK refused the request: TypeError: ", "could not reach the Anthropic API"))
    else:  # an `ant auth login` profile on this machine resolved credentials; the connection still fails
        pytest.fail("expected an AgentError")


def test_anthropic_client_resends_after_a_server_pause_turn(index: EvidenceIndex):
    """``pause_turn``: the server stopped mid-turn; the loop appends the partial
    assistant content and re-sends without adding a user message, then carries on."""
    answer = EvidenceChain.model_validate(chain([criterion("PP3", "supporting", [CFTR_VEP])]))
    sdk = FakeSdk([
        msg([TextBlock(type="text", text="thinking…")], "pause_turn", rid="req_p"),
        msg([ToolUseBlock(type="tool_use", id="toolu_1", name="get_record", input={"record_id": CFTR_VEP})], "tool_use", rid="req_t"),
        msg([TextBlock(type="text", text="Done.")], "end_turn", rid="req_e"),
    ], final=final_message(answer))
    result = ac.AnthropicClient(sdk).run(ac.AgentRequest(system="s", user="u", output_model=EvidenceChain,
                                                         tools=[get_record_tool(index)], max_turns=3))
    assert result.output == answer and result.stop_reason == "end_turn"
    assert [(t.stop_reason, t.request_id) for t in result.transcript[:3]] == [("pause_turn", "req_p"), ("tool_use", "req_t"), ("end_turn", "req_e")]
    assert result.transcript[0].tool_calls == [] and result.transcript[1].tool_calls[0].name == "get_record"
    second = sdk.loop_calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant"]  # the paused turn's content re-sent as is, no new user message
    assert second[-1]["content"][0].text == "thinking…"
    assert result.usage.api_calls == 4 and result.usage.tool_calls == 1


def test_anthropic_client_stops_after_max_turns_tool_calling_turns(index: EvidenceIndex):
    always = msg([ToolUseBlock(type="tool_use", id="toolu_x", name="get_record", input={"record_id": CFTR_VEP})], "tool_use")
    sdk = FakeSdk([always], final=final_message(EvidenceChain.model_validate(chain([]))))
    req = ac.AgentRequest(system="s", user="u", output_model=EvidenceChain, tools=[get_record_tool(index)], max_turns=2)
    result = ac.AnthropicClient(sdk).run(req)
    assert result.stop_reason == "max_turns" and len(sdk.loop_calls) == 3  # two turns honoured, the third refused
    assert not result.transcript[0].tool_calls[0].is_error and not result.transcript[1].tool_calls[0].is_error
    last = result.transcript[2].tool_calls[0]
    assert last.is_error and last.result == ac.TOOL_BUDGET_ERROR
    assert sdk.final_calls[0]["messages"][-2]["content"][0]["is_error"] is True
    assert (result.usage.tool_calls, result.usage.tool_errors) == (3, 1)


def test_anthropic_client_error_mapping(index: EvidenceIndex):
    req = ac.AgentRequest(system="s", user="u", output_model=EvidenceChain)
    http_req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")

    limited = anthropic.RateLimitError("slow down", response=httpx2.Response(429, headers={"retry-after": "7"}, request=http_req), body=None)
    with pytest.raises(ac.AgentError, match=r"rate limit \(retry-after 7s\): slow down"):
        ac.AnthropicClient(FakeSdk([], raise_on_stream=limited)).run(req)

    status = anthropic.APIStatusError("overloaded", response=httpx2.Response(529, request=http_req), body=None)
    with pytest.raises(ac.AgentError, match="Anthropic API error 529: overloaded"):
        ac.AnthropicClient(FakeSdk([], raise_on_stream=status)).run(req)

    with pytest.raises(ac.AgentError, match="could not reach the Anthropic API"):
        ac.AnthropicClient(FakeSdk([], raise_on_stream=anthropic.APIConnectionError(request=http_req))).run(req)

    # what the SDK raises before sending (no credentials, an argument it rejects) is an AgentError too
    for exc in (TypeError("Could not resolve authentication method."), ValueError("Streaming is required …")):
        with pytest.raises(ac.AgentError, match=rf"^the Anthropic SDK refused the request: {type(exc).__name__}: ") as e:
            ac.AnthropicClient(FakeSdk([], raise_on_stream=exc)).run(req)
        assert e.value.__cause__ is exc


def test_final_answer_failures_raise_agent_error_without_quoting_the_text(index: EvidenceIndex):
    ok = msg([TextBlock(type="text", text="ok")], "end_turn")
    req = ac.AgentRequest(system="s", user="u", output_model=EvidenceChain)
    secret = chain([criterion("PP3", "supporting", [CFTR_VEP])], candidate_id="GENE:comphet")

    # refused outright
    with pytest.raises(ac.AgentError, match=r"^final answer not usable: the model refused \(category cyber\); 0 chars") as e:
        ac.AnthropicClient(FakeSdk([ok], final=final_message("", stop="refusal", category="cyber"))).run(req)
    assert e.value.request_id == "req_final" and e.value.final_text == "" and [t.n for t in e.value.transcript] == [1, 2]

    # cut off: the truncated JSON is on the exception, never in its message
    truncated = json.dumps(secret)[:60]
    with pytest.raises(ac.AgentError, match=r"cut off at max_tokens; 60 chars, sha256 [0-9a-f]{16} \(request id req_final\)$") as e:
        ac.AnthropicClient(FakeSdk([ok], final=final_message(truncated, stop="max_tokens"))).run(req)
    assert e.value.final_text == truncated and CFTR not in str(e.value) and "GENE:comphet" not in str(e.value)

    # complete JSON the schema rejects (a code the enum would have stopped)
    bad = json.dumps(dict(secret, variants=[dict(secret["variants"][0], criteria=[
        criterion("PM2_SUPPORTING", "supporting", [CFTR_GNOMAD], just="")])]))
    with pytest.raises(ac.AgentError, match=r"does not fit EvidenceChain: 2 error\(s\) — variants\.0\.criteria\.0\.code: value_error; "
                                              r"variants\.0\.criteria\.0\.justification: string_too_short; \d+ chars") as e:
        ac.AnthropicClient(FakeSdk([ok], final=final_message(bad))).run(req)
    assert e.value.final_text == bad and CFTR not in str(e.value) and "PM2_SUPPORTING" not in str(e.value)
    assert isinstance(e.value.__cause__, Exception) and type(e.value.__cause__).__name__ == "ValidationError"

    # not JSON at all
    with pytest.raises(ac.AgentError, match=r"does not fit EvidenceChain: 1 error\(s\) — <root>: json_invalid"):
        ac.AnthropicClient(FakeSdk([ok], final=final_message("Sorry, I cannot."))).run(req)


def test_loop_turn_refusal_or_truncation_raises(index: EvidenceIndex):
    req = ac.AgentRequest(system="s", user="u", output_model=EvidenceChain, tools=[get_record_tool(index)])
    refused = msg([TextBlock(type="text", text="")], "refusal", rid="req_r",
                  stop_details={"type": "refusal", "category": "bio", "explanation": "no"})
    sdk = FakeSdk([refused], final=final_message(EvidenceChain.model_validate(chain([]))))
    with pytest.raises(ac.AgentError, match=r"^turn 1: the model refused \(category bio\) \(request id req_r\)$") as e:
        ac.AnthropicClient(sdk).run(req)
    assert sdk.final_calls == [] and e.value.transcript[0].stop_reason == "refusal"

    cut = msg([TextBlock(type="text", text="I will now fetch the rec")], "max_tokens", rid="req_m")
    with pytest.raises(ac.AgentError, match=r"^turn 1: the turn was cut off at max_tokens \(request id req_m\)$"):
        ac.AnthropicClient(FakeSdk([cut], final=final_message(EvidenceChain.model_validate(chain([]))))).run(req)


def test_tool_service_failures_raise_but_bad_arguments_go_back_to_the_model(index: EvidenceIndex):
    answer = EvidenceChain.model_validate(chain([]))
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    cases = [
        (HttpError(503, url + "?query=CFTR", "<html>down</html>"), "HTTP 503 from www.ebi.ac.uk"),
        (RuntimeError(f"offline: no cached response for GET {url}"), "offline: no cached response for the request"),
        (GnomadError("Something went wrong"), "GnomadError"),
        (ConnectionResetError(104, "reset"), "ConnectionResetError"),
        (TimeoutError("timed out"), "TimeoutError"),
    ]
    for exc, detail in cases:
        tools = [get_record_tool(index), failing_tool("search_literature", exc)]
        turn = msg([ToolUseBlock(type="tool_use", id="toolu_1", name="get_record", input={"record_id": CFTR_VEP}),
                    ToolUseBlock(type="tool_use", id="toolu_2", name="search_literature", input={"q": "CFTR"})], "tool_use", rid="req_t")
        sdk = FakeSdk([turn], final=final_message(answer))
        req = ac.AgentRequest(system="s", user="u", output_model=EvidenceChain, tools=tools)
        with pytest.raises(ac.ToolFailure, match=rf"^tool 'search_literature' failed: {detail} \(request id req_t\)$") as e:
            ac.AnthropicClient(sdk).run(req)
        assert e.value.tool == "search_literature" and e.value.__cause__ is exc and sdk.final_calls == []
        assert [c.name for c in e.value.transcript[0].tool_calls] == ["get_record"]  # what ran before the failure
        assert str(exc) not in str(e.value)  # the URL and body stay on the chained exception

        fake = ac.FakeClient(answer, turns=[ac.FakeTurn([("search_literature", {"q": "CFTR"})])])
        with pytest.raises(ac.ToolFailure, match=rf"^tool 'search_literature' failed: {detail} \(request id fake_1\)$"):
            fake.run(ac.AgentRequest(system="s", user="u", output_model=EvidenceChain, tools=tools))

    # a bad argument or a handler's own complaint is the model's to route around
    for exc in (KeyError("no record x"), ValueError("not a PMID"), RuntimeError("stage: no such candidate"), LookupError("?")):
        text, is_error = ac.call_tool({"t": failing_tool("t", exc).handler}, "t", {})
        assert is_error and text.startswith(f"Error: {type(exc).__name__}")
    assert not ac.is_infrastructure_failure(KeyError("x")) and ac.is_infrastructure_failure(HttpError(500, url, ""))


def test_request_and_schema_guards(index: EvidenceIndex):
    with pytest.raises(ValueError, match="effort"):
        ac.AgentRequest(system="s", user="u", output_model=EvidenceChain, effort="extreme")
    with pytest.raises(ValueError, match="max_turns"):
        ac.AgentRequest(system="s", user="u", output_model=EvidenceChain, max_turns=0)
    t = get_record_tool(index)
    with pytest.raises(ValueError, match="duplicate tool names"):
        ac.AgentRequest(system="s", user="u", output_model=EvidenceChain, tools=[t, t])
    closed = ac.strict_schema({"type": "object", "properties": {
        "q": {"type": "string"}, "n": {"type": "object", "properties": {"a": {"type": "integer"}}},
        "xs": {"type": "array", "items": {"type": "object", "properties": {"b": {"type": "boolean"}}}}}, "required": ["q"]})
    assert closed["additionalProperties"] is False and closed["required"] == ["n", "q", "xs"]
    assert closed["properties"]["n"] == {"type": "object", "properties": {"a": {"type": "integer"}}, "additionalProperties": False, "required": ["a"]}
    assert closed["properties"]["xs"]["items"]["required"] == ["b"]
    assert isinstance(ac.FakeClient({}), ac.ModelClient) and isinstance(ac.AnthropicClient(FakeSdk([])), ac.ModelClient)


def test_answer_schema_asks_for_every_field_but_never_the_engines_and_pins_the_codes():
    """The default schema drops ``classification`` (the engine's) and pins ``code``
    to the ACMG codes — the API refuses an invented code before parse_answer sees it."""
    default = ac.AgentRequest(system="s", user="u", output_model=EvidenceChain).output_schema
    assert default == ac.default_answer_schema(EvidenceChain) == \
        ac.answer_schema(EvidenceChain, drop=("classification", "points", "classification_richards_2015"), enum={"code": sorted(ALL_CODES)})
    crit = default["$defs"]["Criterion"]
    assert crit["required"] == ["code", "evidence_ids", "justification", "met", "strength"]  # nothing may be omitted
    assert crit["additionalProperties"] is False and crit["properties"]["code"]["enum"] == sorted(ALL_CODES)
    assert "minLength" not in crit["properties"]["justification"]  # transform_schema moved it to the description
    vc = default["$defs"]["VariantChain"]
    assert "classification" not in vc["properties"] and vc["required"] == ["criteria", "key", "summary"]
    assert default["required"] == ["candidate_id", "limits", "literature", "mechanism_hypothesis", "phase_statement",
                                   "variants", "what_would_change_the_call"]
    assert "classification" not in json.dumps(default)

    plain = ac.answer_schema(EvidenceChain)  # the unpinned schema is still there for a caller that wants it
    assert "classification" in plain["$defs"]["VariantChain"]["properties"] and "enum" not in plain["$defs"]["Criterion"]["properties"]["code"]
    narrowed = ac.answer_schema(EvidenceChain, drop=("classification",), enum={"code": ["PS3"], "key": [CFTR]})
    assert narrowed["$defs"]["Criterion"]["properties"]["code"]["enum"] == ["PS3"]
    assert narrowed["$defs"]["VariantChain"]["properties"]["key"]["enum"] == [CFTR]

    med = ac.AgentRequest(system="s", user="u", output_model=MedicineReport).output_schema
    assert med["$defs"]["DrugCandidate"]["properties"]["counter_arguments"]["minItems"] == 1
    assert "trial_ids" in med["$defs"]["DrugCandidate"]["required"] and "enum" not in json.dumps(med)

    class Other(BaseModel):  # a "code" that is not an ACMG code, and a "classification" the engine fills
        code: str
        classification: str | None = None
        inner: list[LiveAnswer] = []
    other = ac.AgentRequest(system="s", user="u", output_model=Other).output_schema
    assert "enum" not in other["properties"]["code"] and "classification" not in other["properties"]
    assert ac.models_used(EvidenceChain) == {EvidenceChain, VariantChain, Criterion} and ac.models_used(Other) == {Other, LiveAnswer}


def test_fake_client_consumes_scripted_outputs_in_order(index: EvidenceIndex):
    first, second = chain([], candidate_id="A"), chain([], candidate_id="B")
    fake = ac.FakeClient([first, lambda req: dict(second, phase_statement=req.user)])
    assert fake.run(request(index)).output.candidate_id == "A"
    assert fake.run(request(index)).output.phase_statement == "bundle text"
    assert fake.run(request(index)).output.candidate_id == "B"  # the last output repeats
    assert fake.requests[0].effort == "high" and fake.run(request(index)).usage.api_calls == 0


# ------------------------------------------------------------------------- live

class LiveAnswer(BaseModel):
    variant_id: str
    allele_count: int
    allele_number: int
    record_ids: list[str]


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ENGINE_LIVE_TESTS"), reason="set ENGINE_LIVE_TESTS=1 to call the Anthropic API")
def test_live_anthropic_tool_loop(index: EvidenceIndex):
    """Public data only: the model fetches the gnomAD record of CFTR p.Phe508del through
    the tool and reports its joint allele count — a number the fixture holds."""
    req = ac.AgentRequest(
        system="You answer only from records returned by the get_record tool. Cite record ids exactly.",
        user=f"Fetch the record {CFTR_GNOMAD} with get_record and report the joint allele count and allele number "
             f"in the JSON answer, listing every record id you used.",
        output_model=LiveAnswer, tools=[get_record_tool(index)], max_turns=3, effort="low")
    result = ac.AnthropicClient().run(req)
    assert result.output.variant_id == "7-117559590-ATCT-A"
    assert (result.output.allele_count, result.output.allele_number) == (19237, 1612320)
    assert CFTR_GNOMAD in result.output.record_ids and all(r in index for r in result.output.record_ids)
    assert result.usage.api_calls >= 2 and result.usage.input_tokens > 0 and result.usage.tool_calls >= 1
    assert any(c.name == "get_record" and not c.is_error for t in result.transcript for c in t.tool_calls)
    assert result.disclosure.startswith("Anthropic API, model claude-opus-5, effort low;")
