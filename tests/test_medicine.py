"""Stage 6 over a synthetic run directory built from public data only.

The run directory is stage 5's from ``tests/test_reason.py`` — ``tests/fixtures/agents/
evidence`` (MTHFR rs1801133, CFTR p.Phe508del, TP53 p.Arg175His), ``tests/fixtures/
reason/`` (stage-3 ``candidates.json``, stage-4 ``joined.json``) — plus the validated
chains in ``tests/fixtures/medicine/`` and the stage-5 evidence store rebuilt from the
recorded Europe PMC search, plus three ``hpo:`` term records written by hand (the
public CF terms stage 5 fetches) and a ``case.yaml`` naming the public disease id
MONDO_0009061. The drug sources are the real retrievers over a stub ``Http`` that
serves the recorded Open Targets (target, drugs, associations and the cystic-fibrosis
disease block), DGIdb, ChEMBL (the gene route and the two recorded text searches) and
ClinicalTrials.gov fixtures for the exact requests they make, or hand-written fakes
with the retrievers' methods where a shape (absence, a service failure) has no
recording. The model is ``FakeClient``, scripted to call every tool — including with
bad arguments — and to answer with a report that breaks every gate of the ladder: a
class backed by no search, a candidate with one counter-argument, one without a
paediatric-safety argument, one without an approved indication, one in a class that
was never searched, a drug it remembers but no source returned, a fabricated ChEMBL
id, a fabricated trial, a rejected entry citing nothing, a follow-up item citing
nothing, a bare NCT id in prose and a paper it never fetched — so what the tests check
is the stage's plumbing and its gates, never a model's judgement. One live test talks
to the four public APIs when ``ENGINE_LIVE_TESTS`` is set (CFTR only).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from engine.agents import client as ac
from engine.agents.schema import MedicineReport
from engine.agents.validator import EvidenceIndex
from engine.cli import main
from engine.medicine import run as mr
from engine.medicine.chembl import ChemblError, ChemblRetriever
from engine.medicine.ctgov import TrialsRetriever
from engine.medicine.dgidb import DgidbRetriever, GeneResult
from engine.medicine.opentargets import OpenTargetsRetriever
from engine.medicine.prompts import PROMPT_VERSION, SYSTEM_PROMPT, instructions_sha256, prompt_sha256
from engine.medicine.render import NO_CANDIDATE, NO_CANDIDATE_NO_CLASS
from engine.medicine.tools import DISEASES_PER_GENE, FIELD_CHARS, MAX_GENES, MedicineRetrievers, MedicineTools
from engine.retrieve.http import Http, HttpCache, HttpError, RateLimiter, Response
from engine.retrieve.literature import LiteratureRetriever
from engine.retrieve.store import EvidenceRecord, EvidenceStore

FIXTURES = Path(__file__).parent / "fixtures"
EVIDENCE = FIXTURES / "agents" / "evidence"
REASON = FIXTURES / "reason"
MEDICINE = FIXTURES / "medicine"

CFTR = "7:117559590:ATCT:A"
CFTR_VEP = f"vep:{CFTR}"
CFTR_GNOMAD = "gnomad:7-117559590-ATCT-A"
CFTR_CLINVAR = "clinvar:VCV000007105"
ENSG = "ENSG00000001626"
TP53_VEP = "vep:17:7675088:C:T"
PAPER = "pmid:42616613"                 # cited by the CFTR chain; in 05_reason/evidence
FROSST = "pmid:7647779"                 # the MTHFR paper in the stage-2 store; not the candidate's
FAKE_PMID = "pmid:99999999"
SEARCH_QUERY = 'TITLE_ABS:"CFTR" AND TITLE_ABS:"F508del"'
OT_TARGET = f"opentargets:{ENSG}"
OT_IVACAFTOR = f"opentargets:drug:{ENSG}:CHEMBL2010601"
OT_CF = f"opentargets:association:{ENSG}:MONDO_0009061"   # the top-scored association: cystic fibrosis
CH_IVACAFTOR = "chembl:CHEMBL2010601"
CH_IVACAFTOR_MEC = "chembl:mechanism:965"
CH_LUMACAFTOR = "chembl:CHEMBL2103870"
CH_LUMACAFTOR_MEC = "chembl:mechanism:2346"
CH_ELEXACAFTOR = "chembl:CHEMBL4298128"
DG_IVACAFTOR = "dgidb:CFTR:rxcui:1243041"
DG_TEZACAFTOR = "dgidb:CFTR:rxcui:1999382"
DG_GENE = "dgidb-gene:CFTR"
TRIAL = "nct:NCT06191640"               # first result of the recorded cystic fibrosis / ivacaftor search
TRIALS = ["nct:NCT06191640", "nct:NCT05331183", "nct:NCT07809867"]
FAKE_TRIAL = "nct:NCT99999999"
HPO = ["HP:0002205", "HP:0006528", "HP:0012236"]
HPO_LABELS = {"HP:0002205": "Recurrent respiratory infections", "HP:0006528": "Chronic lung disease",
              "HP:0012236": "Elevated sweat chloride"}
"""The public CF terms and the labels their JAX records carry (see ``tests/fixtures/hpo``)."""
DISEASE = "MONDO_0009061"
OT_DISEASE = f"opentargets:disease:{DISEASE}"
PMID_SEARCH = "pmid-search:757345ee6bd5216b0430087292fea44082c169691b9ee6bef71d2a826a682243"
NCT_SEARCH = "nct-search:94b4ef2f2811d172330be30b3013ad17d679f7e76eb9de2561a641b62afaa5ae"
CHEMBL_SEARCH_MEC = "chembl-search:bf297f7769358dacc9efd790446fb7df825563f274edd3006b94c823951f740c"
CHEMBL_SEARCH_IND = "chembl-search:fcd5947138c0aeaf5729f98e179da8ba942ccb58d51de5a39aebf2ffd69b537e"
"""The search records the scripted conversation makes — the ids an intervention class
may name in ``searched`` (the recorded Europe PMC, ClinicalTrials.gov and ChEMBL
searches, keyed by their own request)."""
CH_ATALUREN = "chembl:CHEMBL256997"
N_DRUG_RECORDS = 123                    # 1 + 5 + 13 Open Targets, 1 + 49 DGIdb, 54 ChEMBL for CFTR
# Open Targets: the target, its top DISEASES_PER_GENE (5) target-disease associations by score (each an
# opentargets:association: record) and 13 known drugs.
# ChEMBL: 1 target + 13 mechanisms + 13 molecules + 27 indication rows. The indication (and warning)
# rows are fetched per *family* (parent_molecule_chembl_id), so the two rows ChEMBL files on another
# form of a family — drugind 147820 on CHEMBL4298159 (bamocaftor) and 156570 on CHEMBL6068396
# (vanzacaftor) — are distinct citable drug_indication records with their own detail URL; a per-molecule
# filter used to miss them (52). tests/test_chembl.py::test_drugs_for_target_symbol_cftr_end_to_end pins 54.


# ------------------------------------------------------------------------ fixtures

class StubHttp:
    """Serves the recorded fixtures of every source keyed by the exact request; refuses
    anything else. ``cache_ok=False`` (a live version check) is served from the same
    recordings; ``cache_404`` returns the recorded 404 the way ``Http`` does."""

    def __init__(self, *dirs: str, offline: bool = False):
        self.limiter = RateLimiter(default_per_second=0)
        self.offline = offline
        self.calls: list[tuple[str, str, Any, Any]] = []
        self._by_key: dict[str, dict] = {}
        for d in dirs:
            for p in sorted((FIXTURES / d).glob("*.json")):
                self.add(json.loads(p.read_text()))

    def add(self, fx: dict) -> None:
        r = fx["request"]
        self._by_key[HttpCache.key(r["method"], r["url"], r.get("params"), r.get("body"))] = fx

    def request(self, method: str, url: str, *, params=None, json_body=None, cache_404: bool = False, **kw) -> Response:
        self.calls.append((method, url, params, json_body))
        fx = self._by_key.get(HttpCache.key(method, url, params, json_body))
        if fx is None:
            raise AssertionError(f"no fixture for {method} {url} {params} {json.dumps(json_body)[:100] if json_body else ''}")
        text = fx["text"] if fx.get("body") is None else json.dumps(fx["body"], ensure_ascii=False)
        if fx["status"] == 404 and cache_404:
            return Response(404, text, fx["retrieved_at"], False, "stub", fx.get("headers", {}))
        if fx["status"] >= 400:
            raise HttpError(fx["status"], url, text)
        return Response(fx["status"], text, fx["retrieved_at"], False, "stub", fx.get("headers", {}))

    def get(self, url: str, **kw) -> Response:
        return self.request("GET", url, **kw)

    def post(self, url: str, json_body, **kw) -> Response:
        return self.request("POST", url, json_body=json_body, **kw)


def stub_http() -> StubHttp:
    return StubHttp("opentargets", "dgidb", "chembl", "ctgov", "literature")


def retrievers(http: StubHttp | None = None) -> MedicineRetrievers:
    """The real retrievers over the stub: page size 3 for trials, as recorded."""
    http = http or stub_http()
    return MedicineRetrievers(OpenTargetsRetriever(http), DgidbRetriever(http), ChemblRetriever(http),
                              TrialsRetriever(http, page_size=3), LiteratureRetriever(http))


@pytest.fixture
def case_file(tmp_path: Path) -> Path:
    """A case file of this test's own: the public CF terms and the public disease id.
    ``tests/fixtures/public_case/case.yaml`` is not touched (stage 5's tests own it)."""
    path = tmp_path / "case.yaml"
    vcf = tmp_path / "public.vcf.gz"
    vcf.write_bytes(b"")
    (tmp_path / "public.vcf.gz.tbi").write_bytes(b"")
    path.write_text("proband_id: PUBLIC01\n"
                    f"vcf: {vcf.name}\n"
                    "hpo: [" + ", ".join(HPO) + "]\n"
                    f"disease: [{DISEASE}]\n")
    return path


def hpo_record(term: str) -> EvidenceRecord:
    """The ``hpo:`` record stage 5 leaves in ``05_reason/evidence`` for a case term —
    the JAX term object's citable fields, written by hand from the public fixture."""
    definitions = {"HP:0012236": "An increased concentration of chloride in the sweat.",
                   "HP:0002205": "Recurrent bacterial or viral infections of the respiratory tract.",
                   "HP:0006528": "A chronic disorder of the lungs."}
    return EvidenceRecord(
        record_id=f"hpo:{term}", source="hpo", source_version="JAX ontology API hp (the service exposes no release identifier)",
        query={"id": term, "api": f"https://ontology.jax.org/api/hp/terms/{term}"},
        url=f"https://hpo.jax.org/browse/term/{term}", retrieved_at="2026-09-17T00:00:00+00:00",
        payload={"id": term, "name": HPO_LABELS[term], "definition": definitions[term], "synonyms": []})


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    shutil.copytree(EVIDENCE, run / "02_retrieve" / "evidence")
    (run / "03_filter").mkdir()
    shutil.copy(REASON / "candidates.json", run / "03_filter" / "candidates.json")
    (run / "04_rank").mkdir()
    shutil.copy(REASON / "joined.json", run / "04_rank" / "joined.json")
    rank_store = EvidenceStore(run / "04_rank" / "evidence")
    rank_store.put(EvidenceRecord(
        record_id="exomiser:CFTR", source="exomiser", source_version="exomiser-cli 14.0.0 · data 2406",
        query={"hpoIds": HPO, "genomeAssembly": "hg38", "analysis_yaml_sha256": "0" * 64},
        url="https://www.ncbi.nlm.nih.gov/gene/1080", retrieved_at="2026-09-12T00:00:00+00:00",
        payload={"ranking": {"rank": 1, "gene_symbol": "CFTR", "moi": "AR", "exomiser_score": 0.97}, "citation": "Exomiser"}))
    rank_store.write_index()
    chains = run / "05_reason" / "chains"
    chains.mkdir(parents=True)
    shutil.copy(MEDICINE / "chain_CFTR_hom.json", chains / "CFTR:hom.json")
    shutil.copy(MEDICINE / "chain_TP53_het_single.json", chains / "TP53:het_single.json")
    # the stage-5 store as stage 5 left it: the recorded search and its five papers
    reason_store = EvidenceStore(run / "05_reason" / "evidence")
    found = LiteratureRetriever(StubHttp("literature")).search(SEARCH_QUERY, max_results=5)
    reason_store.put(found.record)
    for paper in found.papers:
        reason_store.put(paper)
    for term in HPO:  # the hpo: records stage 5 fetched for the case terms
        reason_store.put(hpo_record(term))
    reason_store.write_index()
    return run


def read(path: Path) -> Any:
    return json.loads(path.read_text())


# ------------------------------------------------------------------------- answers

def candidate(name: str, **over: Any) -> dict[str, Any]:
    base = {
        "name": name, "chembl_id": None, "intervention_class": POTENTIATION,
        "mechanism_of_action": "CFTR potentiator", "approval_status": "approved (CF)",
        "approved_indication": "cystic fibrosis in patients with a gating mutation (ChEMBL first_approval 2012)",
        "rationale": "acts on the residual protein",
        "counter_arguments": ["systemic exposure of a child whose every cell carries the defect",
                              "shown on other genotypes"],
        "paediatric_safety": f"approved from 4 months of age in the records the search returned [{CH_IVACAFTOR}]",
        "evidence_ids": [CH_IVACAFTOR], "trial_ids": [],
    }
    return {**base, **over}


POTENTIATION = "potentiation of the residual channel at the membrane"
READ_THROUGH = "read-through of a premature stop"
UNSEARCHED = "gene replacement by an inhaled vector"
"""The three class names the scripted answer uses: one with candidates, one rejected
on a record, and one the model never searched (dropped, and its candidate with it)."""


def intervention_class(name: str, **over: Any) -> dict[str, Any]:
    base = {
        "name": name, "acts_on": f"the residual channel the chain describes [{CFTR_VEP}]", "targets": ["CFTR"],
        "searched": [PMID_SEARCH, NCT_SEARCH, CHEMBL_SEARCH_MEC], "verdict": "candidates_proposed",
        "rejection_reason": "", "evidence_ids": [CH_IVACAFTOR],
    }
    return {**base, **over}


def cftr_answer() -> dict[str, Any]:
    """What a model that mostly walks the ladder — and breaks every gate once — writes."""
    return {
        "candidate_id": "CFTR", "gene_symbol": "cftr",  # respelled identity: pinned back by the stage
        "mechanism": [
            {"statement": f"F508del-CFTR misfolds, is retained in the ER and degraded; residual channel has low open probability [{CFTR_VEP}] [{PAPER}]",
             "evidence_ids": [CFTR_VEP, PAPER]},
            {"statement": "a mechanism the model made up", "evidence_ids": ["chembl:CHEMBL999999999"]},
        ],
        "consequence": [
            {"statement": f"Apical chloride secretion fails, the airway surface dehydrates and infection recurs — the disease "
                          f"definition names recurrent bronchopulmonary infections [{OT_DISEASE}], which the case lists as "
                          f"HP:0002205 Recurrent respiratory infections [hpo:HP:0002205]",
             "evidence_ids": [OT_DISEASE, "hpo:HP:0002205"]},
        ],
        "pathway_targets": [{"statement": f"Apical chloride secretion; CFTR sits in the 'Defective CFTR causes cystic fibrosis' pathway [{OT_TARGET}]",
                             "evidence_ids": [OT_TARGET]}],
        "intervention_classes": [
            intervention_class(POTENTIATION),
            intervention_class(READ_THROUGH, acts_on=f"a premature stop, which this genotype does not carry [{CFTR_VEP}]",
                               verdict="considered_and_rejected", searched=[CHEMBL_SEARCH_IND, OT_TARGET],
                               rejection_reason="the candidate's variant is an in-frame deletion, not a premature stop; "
                                                "read-through has nothing to act on here",
                               evidence_ids=[OT_TARGET]),
            intervention_class(UNSEARCHED, searched=[], evidence_ids=[]),                # dropped: no search backs it
            intervention_class("modulation of a pathway the model invented",
                               searched=["pmid-search:deadbeef"]),                        # dropped: not this conversation's
            intervention_class("clearance of the stressed cell state", verdict="no_evidence_found",
                               rejection_reason=""),                                      # dropped: no reason for the verdict
        ],
        "candidates": [
            candidate("Ivacaftor", chembl_id="CHEMBL2010601",
                      approval_status="approved 2012, cystic fibrosis (ChEMBL max_phase 4; Open Targets APPROVAL)",
                      rationale=f"Potentiator of CFTR at the membrane [{CH_IVACAFTOR}] [{CH_IVACAFTOR_MEC}] [{OT_IVACAFTOR}]; "
                                f"in trial NCT06191640 with F508del combinations; ChEMBL CHEMBL2010601.",
                      evidence_ids=[CH_IVACAFTOR, CH_IVACAFTOR_MEC, OT_IVACAFTOR, DG_IVACAFTOR], trial_ids=[TRIAL]),
            candidate("Lumacaftor", chembl_id="CHEMBL2103870", mechanism_of_action="CFTR corrector (stabiliser, F508del)",
                      evidence_ids=[CH_LUMACAFTOR_MEC, CH_LUMACAFTOR],
                      counter_arguments=["corrects folding, not gating"]),                # one counter-argument → stage check
            candidate("Elexacaftor", chembl_id="CHEMBL4298128", evidence_ids=[CH_ELEXACAFTOR],
                      paediatric_safety="   "),                                           # no paediatric safety → stage check
            candidate("Ivacaftor", evidence_ids=[CH_IVACAFTOR], approved_indication=""),  # no approved indication → stage check
            candidate("Ataluren", intervention_class=READ_THROUGH, evidence_ids=[CH_ATALUREN],
                      mechanism_of_action="premature-stop read-through"),                  # a rejected class is still a class
            candidate("Ivacaftor", intervention_class=UNSEARCHED, evidence_ids=[CH_IVACAFTOR]),  # its class was dropped
            candidate("Tezacaftor", chembl_id="CHEMBL25", evidence_ids=[CH_ELEXACAFTOR],
                      trial_ids=[TRIAL, FAKE_TRIAL]),                                      # fabricated trial → validator
            candidate("A drug the model remembers", evidence_ids=[CFTR_VEP, CFTR_CLINVAR]),  # no drug record → stage check
            candidate("Tezacaftor", chembl_id="CHEMBL25",                                   # aspirin's id: cleared
                      rationale=f"Corrector [{DG_TEZACAFTOR}]; cf. NCT99999999 and doi:10.1000/made-up.",
                      evidence_ids=[DG_TEZACAFTOR]),
        ],
        "considered_and_rejected": [
            {"name": "Crofelemer", "intervention_class": POTENTIATION,
             "reason": f"a CFTR inhibitor: the action type is the opposite of what this mechanism needs [{OT_TARGET}]",
             "evidence_ids": [OT_TARGET]},
            {"name": "a compound the model remembers", "intervention_class": "", "reason": "no record rejects it either",
             "evidence_ids": []},                                                          # cites nothing → validator
            {"name": "gene replacement", "intervention_class": UNSEARCHED, "reason": "not searched",
             "evidence_ids": [OT_TARGET]},                                                 # class dropped → so is it
        ],
        "surveillance": [
            {"statement": f"the disease record names hepatobiliary complications and pancreatic insufficiency as "
                          f"organ findings to watch [{OT_DISEASE}]", "evidence_ids": [OT_DISEASE]},
        ],
        "follow_up_experiments": [f"Ussing-chamber assay in the proband's nasal epithelial cells with the potentiator [{CH_IVACAFTOR}]",
                                  f"Sweat chloride before and after exposure, against NCT99999999 [{TRIAL}]",
                                  "Compare with PMID 99999999 before any exposure"],       # cites nothing → stage check
        "limits": ["No paediatric pharmacokinetic data in any record", f"The Frosst paper is about MTHFR, not CFTR [{FROSST}]"],
        "literature": [PAPER, FAKE_PMID],
    }


SCRIPT = [
    ac.FakeTurn([("get_record", {"record_id": CFTR_VEP}),
                 ("get_record", {"record_id": PAPER}),
                 ("get_record", {"record_id": CH_IVACAFTOR}),                                   # not returned yet
                 ("drugs_for_gene", {"gene": "CFTR"})], text="mechanism first"),
    ac.FakeTurn([("get_record", {"record_id": CH_IVACAFTOR}),
                 ("search_trials", {"condition": "cystic fibrosis", "intervention": "ivacaftor", "term": None, "max_results": 3}),
                 ("search_trials", {"condition": f"cystic fibrosis {CFTR}", "intervention": None, "term": None, "max_results": None}),
                 ("drugs_for_gene", {"gene": "7:117559590"}),
                 ("search_chembl", {"mechanism_text": "conductance regulator", "indication_text": "cystic fibrosis",
                                    "max_results": 3}),
                 ("search_chembl", {"mechanism_text": None, "indication_text": None, "max_results": None}),
                 ("get_paper", {"pmid": "7647779"}),
                 ("search_literature", {"query": SEARCH_QUERY, "max_results": 5})], text="classes and searches"),
]


def fake_client(outputs: Any = None, turns: list[ac.FakeTurn] | None = None) -> ac.FakeClient:
    return ac.FakeClient(outputs if outputs is not None else cftr_answer(), turns=SCRIPT if turns is None else turns,
                         validate_output=False)  # a pydantic MedicineReport could not even hold counter_arguments=[]


# ------------------------------------------------------------------------ dry run

def test_dry_run_writes_bundle_and_prompt_without_a_model_or_a_source(run_dir: Path, case_file: Path):
    """The dry run makes one kind of network request — the public disease record, which
    is engine-driven and citable — and calls no model and no drug source."""
    manifest_path = mr.run_medicine(run_dir, dry_run=True, case_path=case_file, http=StubHttp("opentargets"))
    out = run_dir / "06_medicine"
    assert manifest_path == out / "manifest.json"
    assert [p.name for p in (out / "bundles").iterdir()] == ["CFTR:hom.json"]
    assert sorted(p.name for p in (out / "prompts").iterdir()) == ["CFTR:hom.json", "CFTR:hom.md"]
    assert not (out / "report.json").exists() and not (out / "report.md").exists() and not (out / "transcripts").exists()
    assert list(read(out / "evidence" / "index.json")) == [OT_DISEASE]  # the disease record, fetched once

    bundle = read(out / "bundles" / "CFTR:hom.json")
    assert bundle["candidate_id"] == "CFTR:hom" and bundle["gene_symbol"] == "CFTR" and bundle["gene_id"] == ENSG
    assert bundle["chain"]["variants"][0]["classification"] == "likely_pathogenic"
    assert bundle["chain_record_ids"] == [CFTR_CLINVAR, "exomiser:CFTR", CFTR_GNOMAD, PAPER, CFTR_VEP] and bundle["missing_ids"] == []
    assert bundle["record_ids"] == [CFTR_CLINVAR, "exomiser:CFTR", CFTR_GNOMAD, *(f"hpo:{t}" for t in HPO),
                                    OT_DISEASE, PAPER, CFTR_VEP]
    assert [r["record_id"] for r in bundle["hpo_terms"]] == [f"hpo:{t}" for t in HPO]
    assert [r["record_id"] for r in bundle["disease_records"]] == [OT_DISEASE]
    assert bundle["dossier"] is None and bundle["secondary_findings"] == []
    text = bundle["text"]
    assert text.startswith("# Candidate CFTR:hom\ngene: CFTR (ENSG00000001626) · model: hom · priority: 1\n")
    assert "## Patient context (records only — nothing here comes from a person's notes)" in text
    assert f"disease: cystic fibrosis [{OT_DISEASE}] — Autosomal recessive disorder caused by pathogenic variants in the CFTR gene" in text
    assert "OMIM 219700; Orphanet 586; HPO annotations of the disease (34): Nasal polyposis" in text
    assert "case HPO: HP:0002205 Recurrent respiratory infections [hpo:HP:0002205]; HP:0006528 Chronic lung disease " \
           "[hpo:HP:0006528]; HP:0012236 Elevated sweat chloride [hpo:HP:0012236]" in text
    assert "HPO definitions: HP:0002205 — Recurrent bacterial or viral infections" in text
    assert f"## Variant {CFTR} — likely pathogenic (+8 SVI points)" in text
    assert f"genotype 1/1 · inframe_deletion (MODERATE) · ENST00000003084 (MANE NM_000492.4) · ENST00000003084.11:c.1521_1523del · ENSP00000003084.6:p.Phe508del [{CFTR_VEP}]" in text
    assert f"gnomAD af 0.01193 · hom 58 [{CFTR_GNOMAD}] · ClinVar Pathogenic (4 stars) VCV000007105 [{CFTR_CLINVAR}]" in text
    assert f"criteria met: PS3 (strong) [{PAPER}]; PM4 (moderate) [{CFTR_VEP}]; PP5 (supporting) [{CFTR_CLINVAR}]; PP4 (supporting) [exomiser:CFTR]; PP3 (supporting) [{CFTR_VEP}] · not met: PM2" in text
    assert "mechanism hypothesis: Loss of function by misfolding" in text and "exomiser: rank 1 · combined 0.97 · phenotype 0.91 · moi AR [exomiser:CFTR]" in text
    assert f"literature: [{PAPER}]" in text
    assert "## Secondary findings in this run (not targets of this report)\nnone" in text
    assert "## Gene dossier" not in text  # stage 5 wrote none for this candidate
    assert text.rstrip("\n").endswith(f"citable record ids: [{CFTR_CLINVAR}] [exomiser:CFTR] [{CFTR_GNOMAD}] "
                                      f"[hpo:HP:0002205] [hpo:HP:0006528] [hpo:HP:0012236] [{OT_DISEASE}] [{PAPER}] [{CFTR_VEP}]")

    prompt = read(out / "prompts" / "CFTR:hom.json")
    assert prompt["system"] == SYSTEM_PROMPT and prompt["system_prompt_sha256"] == prompt_sha256()
    assert prompt["instructions_sha256"] == instructions_sha256() != prompt_sha256()
    assert prompt["user"].endswith(text) and "up to 16 tool-calling turns" in prompt["user"]
    assert prompt["final_instruction"] == ac.FINAL_INSTRUCTION and prompt["tool_budget_error"] == ac.TOOL_BUDGET_ERROR
    assert [t["name"] for t in prompt["tool_definitions"]] == list(mr.TOOLS)
    assert all(t["strict"] is True and t["input_schema"]["additionalProperties"] is False for t in prompt["tool_definitions"])
    assert prompt["tool_definitions"][4]["input_schema"]["required"] == ["condition", "intervention", "max_results", "term"]
    assert prompt["tool_definitions"][5]["input_schema"]["required"] == ["indication_text", "max_results", "mechanism_text"]
    assert "drugs_for_gene" in prompt["tool_definitions"][0]["description"]
    schema = prompt["output_schema"]
    assert schema["properties"]["candidate_id"]["enum"] == ["CFTR:hom"] and schema["properties"]["gene_symbol"]["enum"] == ["CFTR"]
    # the engine fills these; the model is never asked for them
    assert "patient_context" not in schema["properties"] and "secondary_findings" not in schema["properties"]
    assert "patient_context" not in schema["required"] and "secondary_findings" not in schema["required"]
    assert schema["$defs"]["DrugCandidate"]["properties"]["counter_arguments"]["minItems"] == 1  # two is the stage's rule
    assert schema["$defs"]["InterventionClass"]["properties"]["verdict"]["enum"] == \
        ["candidates_proposed", "considered_and_rejected", "no_evidence_found"]
    for name in ("trial_ids", "approved_indication", "paediatric_safety", "intervention_class"):
        assert name in schema["$defs"]["DrugCandidate"]["required"], name
    for name in ("consequence", "intervention_classes", "considered_and_rejected", "surveillance"):
        assert name in schema["required"], name
    md = (out / "prompts" / "CFTR:hom.md").read_text()
    assert md.startswith("# Prompt — CFTR:hom\n") and "## System\n" in md and text in md and ac.FINAL_INSTRUCTION in md

    m = read(manifest_path)
    assert m["stage"] == "medicine" and m["params"]["dry_run"] is True and m["params"]["candidate"] == "CFTR:hom"
    assert m["params"]["candidate_requested"] is None and m["params"]["retrievers"] is None and m["params"]["cache_root"] is None
    assert m["params"]["prompt_version"] == PROMPT_VERSION == "2026-09-17.1"
    assert m["params"]["instructions_sha256"] == instructions_sha256()
    assert m["params"]["tools"] == list(mr.TOOLS) == ["get_record", "search_literature", "get_paper", "drugs_for_gene",
                                                      "search_trials", "search_chembl"]
    assert m["params"]["evidence_stages"] == ["02_retrieve", "04_rank", "05_reason", "06_medicine"]
    assert m["params"]["drugs"] == {"max_genes": MAX_GENES, "sources": ["opentargets", "dgidb", "chembl"], "field_chars": FIELD_CHARS,
                                    "opentargets_diseases_per_gene": DISEASES_PER_GENE}
    assert MAX_GENES == 8 and mr.DEFAULT_MAX_TURNS == 16
    assert m["params"]["trials"]["max_results_cap"] == 25 and "5+ digits" in m["params"]["trials"]["coordinates_in_queries"]
    assert m["params"]["trials"]["field_chars"] == FIELD_CHARS == 240
    assert m["params"]["chembl_search"] == {"default_max_results": 10, "max_results_cap": 25,
                                            "fields": ["mechanism_of_action__icontains", "mesh_heading__icontains"],
                                            "coordinates_in_queries": m["params"]["trials"]["coordinates_in_queries"]}
    assert m["params"]["stage_checks"]["drug_candidate"] == f"dropped unless {mr.NAME_RULE}"
    assert m["params"]["stage_checks"]["chembl_id"].startswith("cleared unless a record the candidate itself cites")
    assert set(mr.LADDER_RULES) <= set(m["params"]["stage_checks"])
    assert "fewer than two non-empty counter-arguments is dropped" in m["params"]["stage_checks"]["counter_arguments"]
    assert m["params"]["hpo"] == HPO and m["params"]["hpo_source"] == f"case file {case_file}"
    assert m["params"]["hpo_terms"]["served"] == [f"hpo:{t}" for t in HPO] and m["params"]["hpo_terms"]["missing"] == []
    assert m["params"]["hpo_terms"]["fetched"] == [] and m["params"]["hpo_terms"]["store"] == "06_medicine/evidence/hpo"
    assert "no HPO release" in m["params"]["hpo_terms"]["version_note"]  # stage 5 fetched them; nothing to fetch here
    assert m["params"]["disease"] == [DISEASE] and m["params"]["disease_source"] == f"case file {case_file}"
    assert m["params"]["disease_records"] == {"served": [], "fetched": [OT_DISEASE], "missing": []}
    assert m["params"]["patient_context"]["disease"][0] == {
        "id": DISEASE, "name": "cystic fibrosis", "record_id": OT_DISEASE,
        "description": m["params"]["patient_context"]["disease"][0]["description"]}
    assert m["params"]["patient_context"]["hpo"] == [{"id": t, "label": HPO_LABELS[t], "record_id": f"hpo:{t}"} for t in HPO]
    assert m["params"]["dossier"] is None and m["params"]["ladder_walked"] is None
    assert m["params"]["disclosure"].startswith("Anthropic API, model claude-opus-5, effort high;")
    assert m["counts"] == {"chains_available": 2, "candidates_selected": 1, "reports_written": 0,
                           "secondary_findings": 0, "evidence_records": 1, "evidence_records_added": 1,
                           "hpo_terms": 3, "hpo_records": 3}
    assert set(m["inputs"]) == {"chain", "candidates", "case", "retrieve_evidence_index", "rank_joined",
                                "rank_evidence_index", "reason_evidence_index"}
    assert m["inputs"]["chain"]["path"].endswith("05_reason/chains/CFTR:hom.json")
    assert m["tools"]["anthropic-sdk"] and any(n.startswith("dry run:") for n in m["notes"])

    before = {p.name: p.read_bytes() for d in ("bundles", "prompts") for p in (out / d).iterdir()}
    mr.run_medicine(run_dir, dry_run=True, case_path=case_file, http=StubHttp("opentargets"))
    assert {p.name: p.read_bytes() for d in ("bundles", "prompts") for p in (out / d).iterdir()} == before
    m2 = read(manifest_path)
    assert m2["params"]["disease_records"] == {"served": [OT_DISEASE], "fetched": [], "missing": []}  # the store holds it now


def test_a_disease_the_platform_does_not_know_is_a_note_not_an_error(run_dir: Path):
    m = read(mr.run_medicine(run_dir, dry_run=True, case_disease=["MONDO_9999999"], http=StubHttp("opentargets")))
    assert m["params"]["disease_records"] == {"served": [], "fetched": [], "missing": ["MONDO_9999999"]}
    assert any("Open Targets has no disease MONDO_9999999" in n for n in m["notes"])
    bundle = read(run_dir / "06_medicine" / "bundles" / "CFTR:hom.json")
    assert bundle["disease_records"] == [] and "disease: none given" in bundle["text"]
    for bad in (["MONDO:0009061"], ["mondo_0009061"], ["CFTR"]):
        with pytest.raises(ValueError, match="not Open Targets disease ids"):
            mr.run_medicine(run_dir, dry_run=True, case_disease=bad)


def test_a_case_term_with_no_hpo_record_is_visible_and_not_citable(run_dir: Path):
    """A term no store of the run holds is fetched here (stage 5's own store is the
    first place it is looked for); one the JAX API has no term for is shown bare and
    is not citable — never silently dropped."""
    store = EvidenceStore(run_dir / "05_reason" / "evidence")
    store.path_for("hpo:HP:0012236").unlink()
    store.write_index()
    http = StubHttp("hpo/requests")
    m = read(mr.run_medicine(run_dir, dry_run=True, case_hpo=[*HPO, "HP:9999999"], http=http))
    assert m["params"]["hpo_terms"]["served"] == ["hpo:HP:0002205", "hpo:HP:0006528"]
    assert m["params"]["hpo_terms"]["fetched"] == ["hpo:HP:0012236"]  # re-fetched into this stage's store
    assert m["params"]["hpo_terms"]["missing"] == ["HP:9999999"]
    assert [url for _, url, _, _ in http.calls] == ["https://ontology.jax.org/api/hp/terms/HP:0012236",
                                                    "https://ontology.jax.org/api/hp/terms/HP:9999999"]
    assert any("HPO term HP:9999999 has no record at the JAX API (404)" in n for n in m["notes"])
    bundle = read(run_dir / "06_medicine" / "bundles" / "CFTR:hom.json")
    assert "HP:9999999 (no hpo: record in the store; label unknown)" in bundle["text"]
    assert "HP:0012236 Elevated sweat chloride [hpo:HP:0012236]" in bundle["text"]
    assert "hpo:HP:9999999" not in bundle["record_ids"] and "hpo:HP:0012236" in bundle["record_ids"]
    assert EvidenceStore(run_dir / "06_medicine" / "evidence").get("hpo:HP:0012236") is not None


def test_the_gene_dossier_stage_5_wrote_is_in_the_bundle_and_citable(run_dir: Path):
    """Stage 5's optional gene dossier: its validated claims reach the prompt and the
    ids it cites that the run holds join the citable list. Its absence is no error."""
    dossier = {
        "candidate_id": "CFTR:hom", "gene_symbol": "CFTR", "uniprot_accession": "P13569",
        "protein": [{"statement": f"CFTR is an ABC transporter of 1480 residues [uniprot:P13569]", "evidence_ids": ["uniprot:P13569"]}],
        "mechanism_of_disease": [{"statement": f"Loss of the channel causes cystic fibrosis [{PAPER}]", "evidence_ids": [PAPER]}],
        "variant_positions": [{"key": CFTR, "protein_position": 508, "region": ["Domain: ABC transporter 1 (423–646)"],
                               "consequence": f"deletion inside NBD1 [{CFTR_VEP}]", "evidence_ids": [CFTR_VEP]}],
        "region_knowledge": [], "genotype_patterns": [], "functional_test": [],
        "limits": ["no assay on this proband's cells"], "literature": [PAPER],
    }
    path = run_dir / "05_reason" / "dossier" / "CFTR:hom.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dossier, sort_keys=True, indent=1) + "\n")
    m = read(mr.run_medicine(run_dir, dry_run=True, case_hpo=HPO))
    bundle = read(run_dir / "06_medicine" / "bundles" / "CFTR:hom.json")
    text = bundle["text"]
    assert "## Gene dossier (validated in stage 5; cite its ids)" in text
    assert "protein: P13569 CFTR is an ABC transporter of 1480 residues [uniprot:P13569]" in text
    assert f"variant positions: {CFTR} → residue 508, Domain: ABC transporter 1 (423–646) — deletion inside NBD1" in text
    assert PAPER in bundle["dossier_record_ids"] and "uniprot:P13569" not in bundle["dossier_record_ids"]  # no such record here
    assert PAPER in bundle["record_ids"]
    assert m["params"]["dossier"]["path"].endswith("05_reason/dossier/CFTR:hom.json") and m["params"]["dossier"]["sha256"]
    assert "dossier" in m["inputs"]


def test_prompt_states_the_rules_the_stage_enforces():
    for phrase in ("walk every rung", "act on the consequence, not on the gene symbol", "NMD caveat", "tumour compartment",
                   "treatment-tolerance", "considered_and_rejected", "at least two counter-arguments", "paediatric_safety",
                   "Absence rule", "never attach a label from memory",
                   "Prefer, in this order: an approved drug", "every cell", "still developing", "systemic and may be lifelong",
                   "Hypotheses for follow-up, not treatment", "never as a plan to treat", "never a treatment recommendation",
                   "Never invent a record id, a ChEMBL id, an NCT id, a PMID or a DOI", "`trial_ids` may contain only `nct:<id>` records",
                   "`chembl_id` must be the ChEMBL id of a record that same candidate cites", "cites at least one record that names the drug",
                   "records about another drug is deleted", "A ChEMBL `mechanism` row alone does not name the drug",
                   "A bare accession in prose", "Never put a genomic coordinate in a query", "Absence is a result",
                   "not targets of this report"):
        assert phrase in SYSTEM_PROMPT, phrase
    assert "2026-" not in SYSTEM_PROMPT  # no clock, so the cached prefix is stable across runs
    # disease-agnostic and drug-free: the disease enters through the patient-context block, a drug through a record
    for word in ("cystic fibrosis", "CFTR", "ivacaftor", "ataluren", "MONDO", "aneuploidy", "BUB1B", "F508"):
        assert word.lower() not in SYSTEM_PROMPT.lower(), word


# ---------------------------------------------------------------------- end to end

def test_end_to_end_report_validated_and_rendered(run_dir: Path, case_file: Path):
    http = stub_http()
    fake = fake_client()
    manifest_path = mr.run_medicine(run_dir, None, fake, "fake-model", "low", False, retrievers=retrievers(http),
                                    case_path=case_file, chembl_max_results=3)
    out = run_dir / "06_medicine"

    # -- tools: every scripted call ran against the real handlers and the real retrievers
    calls = [(c.name, c.is_error) for c in fake.calls]
    assert calls == [("get_record", False), ("get_record", False), ("get_record", True), ("drugs_for_gene", False),
                     ("get_record", False), ("search_trials", False), ("search_trials", True), ("drugs_for_gene", True),
                     ("search_chembl", False), ("search_chembl", True), ("get_paper", False), ("search_literature", False)]
    assert fake.calls[2].result.startswith("Error: KeyError") and "bundle or among this conversation" in fake.calls[2].result
    drugs = json.loads(fake.calls[3].result)
    assert drugs["gene"] == "CFTR" and drugs["ensembl_id"] == ENSG
    assert drugs["opentargets"]["known"] and drugs["opentargets"]["target"]["record_id"] == OT_TARGET
    assert drugs["opentargets"]["target"]["n_drug_rows"] == "13" and drugs["opentargets"]["target"]["tractable_modalities"] == "AB;PR;SM"
    assert len(drugs["opentargets"]["drugs"]) == 13 and drugs["opentargets"]["version"] == "Open Targets Platform 26.06 (API 26.6.3, platform2606)"
    diseases = drugs["opentargets"]["diseases"]
    assert [d["record_id"] for d in diseases][:1] == [OT_CF] and len(diseases) == 5 == DISEASES_PER_GENE
    assert drugs["dgidb"]["known"] and drugs["dgidb"]["gene_record"] == DG_GENE and drugs["dgidb"]["n_interactions"] == 49
    assert drugs["chembl"]["known"] and drugs["chembl"]["target_records"] == ["chembl:CHEMBL4051"] and drugs["chembl"]["n_records"] == 54
    trials = json.loads(fake.calls[5].result)
    assert trials["total_count"] == 187 and trials["returned"] == 3 and [t["record_id"] for t in trials["trials"]] == TRIALS
    assert trials["search_record"] == NCT_SEARCH
    assert "must not contain a genomic coordinate" in fake.calls[6].result
    assert fake.calls[7].result.startswith("Error: ValueError: not a gene symbol")

    # -- the ChEMBL text searches: rows, molecules and one search record per table
    chembl = json.loads(fake.calls[8].result)
    assert chembl["search_records"] == [CHEMBL_SEARCH_MEC, CHEMBL_SEARCH_IND]
    assert (chembl["total_mechanisms"], chembl["total_indications"]) == (13, 135)
    assert [m["record_id"] for m in chembl["mechanisms"]] == ["chembl:mechanism:964", "chembl:mechanism:965", "chembl:mechanism:2346"]
    iva = next(m for m in chembl["mechanisms"] if m["chembl_id"] == "CHEMBL2010601")
    assert iva["name"] == "IVACAFTOR" and iva["max_phase"] == "4" and iva["first_approval"] == "2012"
    assert "conductance regulator" in iva["mechanism_of_action"].lower() and iva["target_chembl_id"] == "CHEMBL4051"
    ataluren = next(i for i in chembl["indications"] if i["chembl_id"] == "CHEMBL256997")
    assert ataluren["name"] == "ATALUREN" and ataluren["mesh_heading"] == "Cystic Fibrosis" and ataluren["first_approval"] == "2014"
    assert "Cite the molecule record" in chembl["note"]
    assert fake.calls[9].result.startswith("Error: ValueError: a ChEMBL search needs a mechanism_text")
    assert json.loads(fake.calls[10].result)["record_id"] == FROSST
    assert json.loads(fake.calls[11].result)["papers"][0]["record_id"] == PAPER
    # nothing genomic left the process: every request the stub saw carried a symbol, an id or a query text
    assert not any("117559590" in json.dumps([u, p, b]) for _, u, p, b in http.calls)

    # -- the stage store holds what the tools fetched, the disease record included
    store = EvidenceStore(out / "evidence")
    assert store.count("opentargets") == 20 and store.exists(OT_DISEASE)  # 19 gene records + the disease
    assert store.count("dgidb") == 49 and store.count("dgidb-gene") == 1 and store.count("nct") == 3
    assert store.count("nct-search") == 1 and store.count("chembl-search") == 2 and store.count("pmid") == 0
    # the gene route's 54 plus what the two text searches added: the three indication rows and ataluren and liprotamase
    # (of which ataluren and liprotamase are new; the mechanism search adds none - its rows are the gene route's own)
    assert store.count("chembl") == 54 + 4
    index = read(out / "evidence" / "index.json")
    assert index[OT_DISEASE]["url"] == "https://platform.opentargets.org/disease/MONDO_0009061"
    assert index[CHEMBL_SEARCH_MEC]["source_version"] == "ChEMBL_37 (2026-05-01)"

    # -- the report: the ladder's rungs, the gates applied, the engine's fields filled
    report = MedicineReport.model_validate(read(out / "report.json"))
    assert report.candidate_id == "CFTR:hom" and report.gene_symbol == "CFTR"
    assert [m.statement[:12] for m in report.mechanism] == ["F508del-CFTR"]
    assert len(report.consequence) == 1 and "HP:0002205 Recurrent respiratory infections" in report.consequence[0].statement
    assert [(c.name, c.verdict) for c in report.intervention_classes] == [(POTENTIATION, "candidates_proposed"),
                                                                          (READ_THROUGH, "considered_and_rejected")]
    assert report.intervention_classes[0].searched == [PMID_SEARCH, NCT_SEARCH, CHEMBL_SEARCH_MEC]
    assert report.intervention_classes[1].rejection_reason.startswith("the candidate's variant is an in-frame deletion")
    assert [d.name for d in report.candidates] == ["Ivacaftor", "Ataluren", "Tezacaftor"]
    ivacaftor = report.candidates[0]
    assert ivacaftor.chembl_id == "CHEMBL2010601" and ivacaftor.trial_ids == [TRIAL]
    assert ivacaftor.intervention_class == POTENTIATION and len(ivacaftor.counter_arguments) == 2
    assert ivacaftor.approved_indication.startswith("cystic fibrosis in patients with a gating mutation")
    assert "approved from 4 months of age" in ivacaftor.paediatric_safety
    assert "in trial NCT06191640 with F508del combinations; ChEMBL CHEMBL2010601." in ivacaftor.rationale  # carried by records
    tezacaftor = report.candidates[2]
    assert tezacaftor.chembl_id is None  # aspirin's id, carried by no record
    assert tezacaftor.rationale == f"Corrector [{DG_TEZACAFTOR}]; cf. [^1] and [^2]."  # the stage check's, numbered first
    assert [r.name for r in report.considered_and_rejected] == ["Crofelemer"]
    assert len(report.surveillance) == 1 and report.surveillance[0].evidence_ids == [OT_DISEASE]
    assert len(report.follow_up_experiments) == 2 and "Compare with PMID" not in json.dumps(report.follow_up_experiments)
    assert report.literature == [PAPER]
    # the engine's own fields: filled from the records, never asked of the model
    assert report.patient_context is not None
    assert [(t.id, t.label, t.record_id) for t in report.patient_context.hpo] == [(t, HPO_LABELS[t], f"hpo:{t}") for t in HPO]
    assert [(d.id, d.name, d.record_id) for d in report.patient_context.disease] == [(DISEASE, "cystic fibrosis", OT_DISEASE)]
    assert report.patient_context.disease[0].description.startswith("Autosomal recessive disorder caused by")
    assert report.secondary_findings == []
    dumped = report.model_dump_json()
    assert "99999999" not in dumped and "CHEMBL25\"" not in dumped and "made up" not in dumped and "remembers" not in dumped
    assert UNSEARCHED not in dumped and "pathway the model invented" not in dumped

    validation = read(out / "validation" / "CFTR:hom.json")
    reasons = {(r["path"], r["reason"]) for r in validation["rejections"]}
    assert reasons == {
        ("intervention_classes[2]", f"intervention class {UNSEARCHED!r} is not backed by a search made in this conversation"),
        ("intervention_classes[3]", "intervention class 'modulation of a pathway the model invented' is not backed by a "
                                    "search made in this conversation"),
        ("intervention_classes[4]", "intervention class 'clearance of the stressed cell state' has verdict "
                                    "'no_evidence_found' but no rejection_reason"),
        ("candidates[1]", "drug candidate 'lumacaftor' carries 1 counter-argument(s); at least two are required"),
        ("candidates[2]", "drug candidate 'elexacaftor' gives no paediatric-safety reasoning"),
        ("candidates[3]", "drug candidate 'ivacaftor' states no approved indication as recorded"),
        ("candidates[5]", "drug candidate 'ivacaftor' belongs to no searched intervention class"),
        ("candidates[6]", f"trial id(s) not nct: records in the store: {FAKE_TRIAL}"),
        ("candidates[6].chembl_id", "chembl_id not carried by any record the candidate cites: CHEMBL25; cleared"),
        ("candidates[7]", ("drug candidate 'a drug the model remembers' cites no record from a drug source that names it "
                           "(a drug row, an interaction, a ChEMBL molecule, a trial or a paper carrying the name)")),
        ("candidates[8].chembl_id", "chembl_id not carried by any record the candidate cites: CHEMBL25; cleared"),
        ("candidates[8].rationale", "bare accession not carried by any citable record: NCT99999999"),
        ("candidates[8].rationale", "bare accession not carried by any citable record: doi:10.1000/made-up"),
        ("considered_and_rejected[1]", "cites no evidence record"),
        ("considered_and_rejected[2]", "rejected entry 'gene replacement' names an intervention class that was not searched"),
        ("follow_up_experiments[1]", "bare accession not carried by any citable record: NCT99999999"),
        ("follow_up_experiments[2]", "follow-up experiment cites no record"),
        ("mechanism[1]", "unknown evidence id(s): chembl:CHEMBL999999999"),
        ("literature[1]", f"no such literature record in the store: {FAKE_PMID}"),
    }
    counts = validation["counts"]
    assert (counts["counter_arguments_short"], counts["paediatric_safety_missing"], counts["approved_indication_missing"]) == (1, 1, 1)
    assert (counts["classes_unbacked"], counts["classes_unreasoned"], counts["candidates_unclassed"], counts["follow_up_uncited"]) == (2, 1, 2, 1)
    assert counts["items_dropped"] == 13 and counts["literature_removed"] == 1 and counts["citations_out_of_scope"] == 0
    assert validation["notes"][:2] == ["candidate_id: the model wrote 'CFTR'; replaced by the candidate's 'CFTR:hom'",
                                       "gene_symbol: the model wrote 'cftr'; replaced by the candidate's 'CFTR'"]

    # -- the manifest: the ladder's counts, every rejection as a note, versions, disclosure
    m = read(manifest_path)
    c = m["counts"]
    assert c["reports_written"] == 1 and c["drug_candidates"] == 3 and c["drug_candidates_with_trials"] == 1
    assert c["mechanism_claims"] == 1 and c["consequence_claims"] == 1 and c["pathway_targets"] == 1
    assert c["intervention_classes"] == 2 and c["classes_by_verdict"] == {"candidates_proposed": 1, "considered_and_rejected": 1}
    assert c["considered_and_rejected"] == 1 and c["surveillance"] == 1 and c["follow_up_experiments"] == 2
    assert c["secondary_findings"] == 0
    assert c["searches"] == {"literature": 1, "trials": 1, "chembl": 2, "genes": 1}
    assert c["rejections"] == 19 and c["validation"]["items_dropped"] == 13
    assert c["evidence_records"] == store.count() and c["evidence_records_added"] == store.count()
    assert c["tool_calls"] == {"drugs_for_gene": 2, "get_paper": 1, "get_record": 4, "search_chembl": 2,
                               "search_literature": 1, "search_trials": 2}
    assert m["params"]["ladder_walked"] is True
    assert "CFTR:hom: rejected candidates[1]: drug candidate 'lumacaftor' carries 1 counter-argument(s); at least two are required" in m["notes"]
    assert "CFTR:hom: candidate_id: the model wrote 'CFTR'; replaced by the candidate's 'CFTR:hom'" in m["notes"]
    used = m["params"]["tools_used"]
    assert used["genes"] == ["CFTR"] and used["trials"] == TRIALS and used["chembl_searches"] == [CHEMBL_SEARCH_MEC, CHEMBL_SEARCH_IND]
    assert used["records_read"] == [CFTR_VEP, PAPER, CH_IVACAFTOR] and used["papers"][0] == FROSST
    assert m["params"]["source_versions"]["chembl-search"] == ["ChEMBL_37 (2026-05-01)"]
    assert m["params"]["model"] == "fake-model" and m["params"]["disclosure"].startswith("FakeClient")
    assert m["params"]["agent"]["request"]["tools"] == list(mr.TOOLS)
    assert m["outputs"]["report_md"]["sha256"] and m["outputs"]["report_json"]["sha256"]

    transcript = read(out / "transcripts" / "CFTR:hom.json")
    assert [t["stop_reason"] for t in transcript["transcript"]] == ["tool_use", "tool_use", "end_turn"]
    assert json.loads(transcript["final_text"])["gene_symbol"] == "cftr"  # as written, before the gate
    assert transcript["tools"]["chembl_searches"] == [CHEMBL_SEARCH_MEC, CHEMBL_SEARCH_IND]

    # -- the markdown: the rubric's order, the ladder visible, nothing fabricated, no "survived"
    md = (out / "report.md").read_text()
    assert md.startswith("# Medicine — CFTR:hom\n\ncandidate CFTR:hom · gene CFTR · model hom · stage-5 classification: "
                         f"{CFTR} likely pathogenic · model fake-model · effort low\n")
    assert "not a treatment recommendation" in md and "6 proposed candidate(s) were removed by the validator" in md
    order = ["# Medicine report — CFTR (CFTR:hom)", "## Patient context", "## Variant mechanism",
             "## Cellular and disease consequence", "## Intervention classes searched", "## Drug candidates",
             "### 1. Ivacaftor (CHEMBL2010601)", "- counter-arguments:", "- paediatric safety:", "### 3. Tezacaftor",
             "## Considered and rejected", "## Surveillance", "## Follow-up experiments", "## Limits",
             "## Secondary findings", "## Literature", "## References"]
    positions = [md.index(x) for x in order]
    assert positions == sorted(positions)
    assert f"- disease: cystic fibrosis [{OT_DISEASE}] — Autosomal recessive disorder" in md
    assert "- case HPO: HP:0012236 Elevated sweat chloride [hpo:HP:0012236]" in md
    assert "| class | acts on | targets | verdict | searches | records |" in md
    assert f"| {POTENTIATION} | the residual channel the chain describes [{CFTR_VEP}] | CFTR | candidates proposed |" in md
    assert f"[{PMID_SEARCH}] [{NCT_SEARCH}] [{CHEMBL_SEARCH_MEC}]" in md
    assert "| read-through of a premature stop |" in md and "considered and rejected — the candidate's variant" in md
    assert "- class: potentiation of the residual channel at the membrane" in md
    assert "- approved indication: cystic fibrosis in patients with a gating mutation" in md
    assert "  - systemic exposure of a child whose every cell carries the defect" in md
    assert f"- trials: [{TRIAL}]" in md
    assert "- **Crofelemer** (class: potentiation of the residual channel at the membrane) — a CFTR inhibitor" in md
    assert "## Secondary findings\n\n- none recorded" in md
    refs = md.split("## References")[1]
    assert f"- [{CH_IVACAFTOR}] — https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL2010601" in refs
    assert f"- [{OT_DISEASE}] — https://platform.opentargets.org/disease/MONDO_0009061" in refs
    assert f"- [{TRIAL}] — https://clinicaltrials.gov/study/NCT06191640" in refs
    assert f"- [{PMID_SEARCH}] — " in refs and f"- [{CHEMBL_SEARCH_MEC}] — " in refs
    assert f"[{PAPER}] " in md.split("## Literature")[1] and "Alyftrek" in md.split("## Literature")[1]
    prose = "\n".join(line for line in md.splitlines() if not line.startswith("[^"))
    for token in ("99999999", "(CHEMBL25)", "made-up", "Lumacaftor", "Elexacaftor", "remembers", "survived"):
        assert token not in prose, token  # the prose carries markers; each footnote names the hole it left
    assert "[^1]: citation removed by the validator: bare accession not carried by any citable record: NCT99999999" in md
    assert md.count("_FakeClient (scripted, no API call), model fake-model, effort low_") == 1
    assert "2026-" not in md

    # -- deterministic: a rerun in place and a rerun in a fresh copy of the run directory give the same bytes
    def snapshot(stage: Path) -> dict[str, bytes]:
        return {str(p.relative_to(stage)): p.read_bytes() for p in stage.rglob("*") if p.is_file() and p.name != "manifest.json"}
    before = snapshot(out)
    mr.run_medicine(run_dir, None, fake_client(), "fake-model", "low", False, retrievers=retrievers(),
                    case_path=case_file, chembl_max_results=3)
    assert snapshot(out) == before
    fresh = run_dir.parent / "fresh"
    shutil.copytree(run_dir, fresh, ignore=shutil.ignore_patterns("06_medicine"))
    mr.run_medicine(fresh, None, fake_client(), "fake-model", "low", False, retrievers=retrievers(),
                    case_path=case_file, chembl_max_results=3)
    assert snapshot(fresh / "06_medicine") == before
    m2 = read(fresh / "06_medicine" / "manifest.json")
    assert m2["counts"] == m["counts"] and m2["params"]["tools_used"] == m["params"]["tools_used"]


def test_an_empty_candidate_list_names_the_classes_searched_and_never_says_survived(run_dir: Path, case_file: Path):
    """The failure of the first live run, in one test: nothing was proposed, so the
    report says so and lists what was searched — the word "survived" never appears."""
    answer = dict(cftr_answer(), candidates=[], considered_and_rejected=[], follow_up_experiments=[])
    mr.run_medicine(run_dir, None, fake_client(answer), "fake-model", "low", False, retrievers=retrievers(),
                    case_path=case_file, chembl_max_results=3)
    md = (run_dir / "06_medicine" / "report.md").read_text()
    assert f"{NO_CANDIDATE} Classes searched: {POTENTIATION} (candidates proposed), {READ_THROUGH} (considered and rejected)" in md
    assert "survived" not in md and "[citation removed" not in md.split("## Drug candidates")[1].split("##")[0]
    assert "proposed candidate(s) were removed by the validator" not in md  # nothing was proposed to remove
    m = read(run_dir / "06_medicine" / "manifest.json")
    assert m["counts"]["drug_candidates"] == 0 and m["params"]["ladder_walked"] is True


def test_no_class_at_all_says_the_ladder_was_not_walked(run_dir: Path, case_file: Path):
    answer = dict(cftr_answer(), candidates=[], intervention_classes=[], considered_and_rejected=[],
                  follow_up_experiments=[])
    m = read(mr.run_medicine(run_dir, None, fake_client(answer), "fake-model", "low", False, retrievers=retrievers(),
                             case_path=case_file, chembl_max_results=3))
    md = (run_dir / "06_medicine" / "report.md").read_text()
    assert NO_CANDIDATE_NO_CLASS in md and "survived" not in md
    assert m["params"]["ladder_walked"] is False
    assert any("the ladder was not walked: no intervention class survived the checks" in n for n in m["notes"])
    # a class backed only by a literature search is not a walked ladder either
    one = dict(cftr_answer(), candidates=[], considered_and_rejected=[], follow_up_experiments=[],
               intervention_classes=[intervention_class(POTENTIATION, searched=[PMID_SEARCH])])
    m = read(mr.run_medicine(run_dir, None, fake_client(one), "fake-model", "low", False, retrievers=retrievers(),
                             case_path=case_file, chembl_max_results=3))
    assert m["params"]["ladder_walked"] is False
    assert any("no class is backed by a trial or ChEMBL search" in n for n in m["notes"])


def test_a_secondary_finding_is_recorded_and_is_not_a_target(run_dir: Path, case_file: Path):
    """Another chain of the run — a lone heterozygote whose criteria the engine
    combines to P/LP — is a secondary finding: on the bundle, in the report and in the
    rendered document, and named to the model as a non-target, never as a candidate."""
    chain_path = run_dir / "05_reason" / "chains" / "TP53:het_single.json"
    chain = read(chain_path)
    chain["variants"][0]["criteria"] = [
        {"code": "PVS1", "strength": "very_strong", "met": True, "justification": "null variant", "evidence_ids": [TP53_VEP]},
        {"code": "PM2", "strength": "supporting", "met": True, "justification": "absent from gnomAD",
         "evidence_ids": ["gnomad:17-7675088-C-T"]},
    ]
    chain["variants"][0]["classification"] = "vus"  # the file's own value; the stage recomputes it
    chain_path.write_text(json.dumps(chain, sort_keys=True, indent=1) + "\n")
    mr.run_medicine(run_dir, None, fake_client(), "fake-model", "low", False, retrievers=retrievers(),
                    case_path=case_file, chembl_max_results=3)
    out = run_dir / "06_medicine"
    bundle = read(out / "bundles" / "CFTR:hom.json")
    assert bundle["secondary_findings"] == [{
        "candidate_id": "TP53:het_single", "gene_symbol": "TP53", "model": "het_single",
        "classifications": {"17:7675088:C:T": "likely_pathogenic"},
        "note": bundle["secondary_findings"][0]["note"]}]
    assert "a secondary finding, not a repurposing target" in bundle["secondary_findings"][0]["note"]
    assert "TP53:het_single TP53 het_single: 17:7675088:C:T likely pathogenic" in bundle["text"]
    report = MedicineReport.model_validate(read(out / "report.json"))
    assert [f.candidate_id for f in report.secondary_findings] == ["TP53:het_single"]
    md = (out / "report.md").read_text()
    assert "| TP53:het_single | TP53 | het_single | 17:7675088:C:T likely pathogenic |" in md
    m = read(out / "manifest.json")
    assert m["counts"]["secondary_findings"] == 1 and m["params"]["secondary_findings"] == ["TP53:het_single"]
    # as shipped the TP53 chain stays a VUS: no secondary finding
    shutil.copy(MEDICINE / "chain_TP53_het_single.json", chain_path)
    mr.run_medicine(run_dir, None, fake_client(), "fake-model", "low", False, retrievers=retrievers(),
                    case_path=case_file, chembl_max_results=3)
    assert read(out / "bundles" / "CFTR:hom.json")["secondary_findings"] == []
    assert read(out / "report.json")["secondary_findings"] == []


def test_the_candidate_is_the_best_priority_chain_unless_asked_otherwise(run_dir: Path):
    assert list(mr.list_chains(run_dir)) == ["CFTR:hom", "TP53:het_single"]
    assert mr.select_candidate(run_dir, None)[0] == "CFTR:hom"
    m = read(mr.run_medicine(run_dir, "TP53:het_single", dry_run=True))
    assert m["params"]["candidate"] == "TP53:het_single" and m["params"]["candidate_requested"] == "TP53:het_single"
    assert [p.name for p in (run_dir / "06_medicine" / "bundles").iterdir()] == ["TP53:het_single.json"]
    bundle = read(run_dir / "06_medicine" / "bundles" / "TP53:het_single.json")
    assert bundle["gene_symbol"] == "TP53" and "## Variant 17:7675088:C:T — vus" in bundle["text"]
    assert bundle["record_ids"] == ["clinvar:VCV000012374", "gnomad:17-7675088-C-T", *(f"hpo:{t}" for t in HPO), TP53_VEP]
    with pytest.raises(KeyError, match="no stage-5 chain for candidate"):
        mr.run_medicine(run_dir, "MTHFR:hom", dry_run=True)
    (run_dir / "05_reason" / "chains" / "TP53:het_single.json").unlink()
    (run_dir / "05_reason" / "chains" / "CFTR:hom.json").unlink()
    with pytest.raises(FileNotFoundError, match="wrote no chain"):
        mr.run_medicine(run_dir, dry_run=True)
    shutil.rmtree(run_dir / "05_reason")
    with pytest.raises(FileNotFoundError, match="stage 5"):
        mr.run_medicine(run_dir, dry_run=True)


def test_the_classification_shown_is_recomputed_from_the_chains_criteria(run_dir: Path):
    """Stage 5 wrote the engine's verdict; a chain file edited to disagree with its own
    criteria is shown at the recomputed value, and the disagreement is noted."""
    chain_path = run_dir / "05_reason" / "chains" / "CFTR:hom.json"
    chain = read(chain_path)
    chain["variants"][0]["classification"] = "pathogenic"  # PS3 + PM4 + PP3/PP4/PP5 combine to likely_pathogenic
    chain_path.write_text(json.dumps(chain, sort_keys=True, indent=1) + "\n")
    m = read(mr.run_medicine(run_dir, dry_run=True))
    bundle = read(run_dir / "06_medicine" / "bundles" / "CFTR:hom.json")
    assert bundle["chain"]["variants"][0]["classification"] == "likely_pathogenic"
    assert f"## Variant {CFTR} — likely pathogenic" in bundle["text"] and "— pathogenic" not in bundle["text"]
    note = ("chain variants[0].classification: the file says 'pathogenic', its criteria combine to 'likely_pathogenic' "
            "(ACMG/AMP 2015 rules); the engine's value is used")
    assert bundle["notes"] == [note] and f"CFTR:hom: {note}" in m["notes"]
    assert "117559590" not in note
    # as shipped, the chain agrees with its criteria: no note
    shutil.copy(MEDICINE / "chain_CFTR_hom.json", chain_path)
    m = read(mr.run_medicine(run_dir, dry_run=True))
    assert read(run_dir / "06_medicine" / "bundles" / "CFTR:hom.json")["notes"] == [] and not any("combine to" in n for n in m["notes"])


def test_the_manifest_describes_the_http_the_sources_ran_on(run_dir: Path, tmp_path: Path):
    """``offline`` and ``cache_root`` come from the client actually used, not from the
    arguments alone; with injected retrievers there is no client to describe."""
    answer = dict(cftr_answer(), candidates=[candidate("Ivacaftor", evidence_ids=[PAPER])], mechanism=[cftr_answer()["mechanism"][0]],
                  pathway_targets=[], literature=[PAPER], follow_up_experiments=[], limits=[])
    http = Http(HttpCache(tmp_path / "cache-a"), offline=True)
    m = read(mr.run_medicine(run_dir, None, fake_client(answer, turns=[]), "fake-model", "low", False, http=http))
    assert m["params"]["offline"] is True and m["params"]["cache_root"] == str(tmp_path / "cache-a")
    m = read(mr.run_medicine(run_dir, None, fake_client(answer, turns=[]), "fake-model", "low", False, retrievers=retrievers(), offline=True))
    assert m["params"]["offline"] is True and m["params"]["cache_root"] is None
    m = read(mr.run_medicine(run_dir, None, fake_client(answer, turns=[]), "fake-model", "low", False, cache_root=tmp_path / "cache-b"))
    assert m["params"]["offline"] is False and m["params"]["cache_root"] == str(tmp_path / "cache-b")


def test_a_chain_citation_no_store_holds_is_visible_and_not_citable(run_dir: Path):
    chain_path = run_dir / "05_reason" / "chains" / "CFTR:hom.json"
    chain = read(chain_path)
    chain["limits"].append("See [pmid:11111111].")
    chain_path.write_text(json.dumps(chain, sort_keys=True, indent=1) + "\n")
    m = read(mr.run_medicine(run_dir, dry_run=True))
    bundle = read(run_dir / "06_medicine" / "bundles" / "CFTR:hom.json")
    assert bundle["missing_ids"] == ["pmid:11111111"] and "pmid:11111111" not in bundle["record_ids"]
    assert "cited by the chain but not in any store (not citable): pmid:11111111" in bundle["text"]
    assert any("no store of the run holds" in n for n in m["notes"])


def test_a_run_replaces_earlier_outputs_but_keeps_the_evidence_store(run_dir: Path):
    mr.run_medicine(run_dir, None, fake_client(), "fake-model", "low", False, retrievers=retrievers())
    out = run_dir / "06_medicine"
    assert (out / "report.json").exists() and read(out / "evidence" / "index.json")

    class Refusing:
        def run(self, request: ac.AgentRequest) -> ac.AgentResult:
            raise ac.AgentError("final answer not usable: the model refused", request_id="req_x")
    with pytest.raises(ac.AgentError, match="refused"):
        mr.run_medicine(run_dir, None, Refusing(), "fake-model", "low", False, retrievers=retrievers())
    assert sorted(p.name for p in (out / "transcripts").iterdir()) == ["CFTR:hom.failed.json"]
    assert read(out / "transcripts" / "CFTR:hom.failed.json")["request_id"] == "req_x"
    assert not (out / "report.json").exists() and not (out / "report.md").exists() and not (out / "manifest.json").exists()
    assert not (out / "validation").exists() and (out / "bundles" / "CFTR:hom.json").exists()

    held = EvidenceStore(out / "evidence").count()
    answer = dict(cftr_answer(), candidates=[candidate("Ivacaftor", chembl_id="CHEMBL2010601", evidence_ids=[CH_IVACAFTOR, OT_IVACAFTOR])],
                  mechanism=[cftr_answer()["mechanism"][0]], consequence=[], intervention_classes=[
                      intervention_class(POTENTIATION, searched=[DG_GENE], evidence_ids=[DG_GENE])],
                  considered_and_rejected=[], surveillance=[], literature=[PAPER],
                  follow_up_experiments=[f"organoid swelling assay [{CH_IVACAFTOR}]"],
                  limits=["No paediatric pharmacokinetic data in any record"])
    m = read(mr.run_medicine(run_dir, None, fake_client(answer, turns=SCRIPT[:1]), "fake-model", "low", False, retrievers=retrievers()))
    for d in ("bundles", "transcripts", "validation"):
        assert sorted(p.name for p in (out / d).iterdir()) == ["CFTR:hom.json"], d
    # the store is a store: the first run's trials and searches are still there; this run added nothing new
    assert EvidenceStore(out / "evidence").count("nct") == 3
    assert m["counts"]["evidence_records"] == held and m["counts"]["evidence_records_added"] == 0
    assert m["counts"]["rejections"] == 0 and m["counts"]["drug_candidates"] == 1
    assert m["params"]["ladder_walked"] is False  # one class, backed by a gene lookup alone
    # a dry run is a run too
    mr.run_medicine(run_dir, dry_run=True)
    assert not (out / "report.md").exists() and (out / "evidence" / "index.json").exists()


# ------------------------------------------------------------------------------ tools

def test_tools_refuse_bad_arguments_before_any_request(run_dir: Path):
    http = stub_http()
    store = EvidenceStore(run_dir / "06_medicine" / "evidence")
    tools = MedicineTools(EvidenceIndex.from_run(run_dir, stages=mr.EVIDENCE_STAGES), store, retrievers(http),
                          gene_symbol="CFTR", gene_id=ENSG, citable=[CFTR_VEP])
    for bad in ("", "  ", "7:117559590", "CFTR F508del", "a" * 41, None,
                "117559590", "7-117559590", "chr7-117559590", "g.117559590", "X-153000000", "MT:8993", "NM_000492.4"):
        with pytest.raises(ValueError, match="not a gene symbol"):
            tools.drugs_for_gene({"gene": bad})
    assert http.calls == []  # a coordinate in any spelling never reaches a source
    with pytest.raises(ValueError, match="needs a condition, an intervention or a free-text term"):
        tools.search_trials({"condition": " ", "intervention": None, "term": None, "max_results": None})
    with pytest.raises(ValueError, match="genomic coordinate|five or more digits"):
        tools.search_trials({"condition": "cystic fibrosis", "intervention": None, "term": "CFTR 117559590", "max_results": None})
    with pytest.raises(ValueError, match="between 1 and 25"):
        tools.search_trials({"condition": "cystic fibrosis", "intervention": None, "term": None, "max_results": 26})
    with pytest.raises(KeyError):
        tools.get_record({"record_id": CH_IVACAFTOR})  # real elsewhere? no — not returned by a tool yet
    assert http.calls == [] and store.count() == 0
    with pytest.raises(ValueError, match="a ChEMBL search needs a mechanism_text"):
        tools.search_chembl({"mechanism_text": " ", "indication_text": None, "max_results": None})
    with pytest.raises(ValueError, match="genomic coordinate|five or more digits"):
        tools.search_chembl({"mechanism_text": None, "indication_text": "cystic fibrosis 117559590", "max_results": None})
    with pytest.raises(ValueError, match="at least 3 characters"):
        tools.search_chembl({"mechanism_text": "cf", "indication_text": None, "max_results": None})
    with pytest.raises(ValueError, match="between 1 and 25"):
        tools.search_chembl({"mechanism_text": "conductance regulator", "indication_text": None, "max_results": 26})
    assert [t.param()["name"] for t in tools.specs()] == list(mr.TOOLS)
    assert tools.specs()[3].param()["input_schema"]["required"] == ["gene"]
    assert "8 distinct genes" in tools.specs()[3].description


def test_the_gene_cap_and_absence_per_source_with_fake_retrievers(run_dir: Path):
    """A symbol no source knows: every part says so, nothing is written; and the cap on
    distinct genes is a refusal the model can read, not a request."""
    empty = StubHttp()

    class NoOpenTargets(OpenTargetsRetriever):
        def resolve_symbol(self, symbol, *, allow_synonym=False):
            return None

    class NoDgidb(DgidbRetriever):
        def fetch(self, gene_symbol, *, resolve_aliases=True):
            return GeneResult(gene_symbol.upper(), None, [])

    class NoChembl(ChemblRetriever):
        def drugs_for_target_symbol(self, symbol):
            return []
    fakes = MedicineRetrievers(NoOpenTargets(empty), NoDgidb(empty), NoChembl(empty), TrialsRetriever(empty), LiteratureRetriever(empty))
    store = EvidenceStore(run_dir / "06_medicine" / "evidence")
    tools = MedicineTools(EvidenceIndex.from_run(run_dir, stages=mr.EVIDENCE_STAGES), store, fakes,
                          gene_symbol="CFTR", gene_id=ENSG, max_genes=2)
    out = tools.drugs_for_gene({"gene": "notagene1"})
    assert out["gene"] == "NOTAGENE1" and out["ensembl_id"] is None
    assert out["opentargets"] == {"known": False, "note": "Open Targets maps no target to this symbol", "target": None, "diseases": [], "drugs": []}
    assert out["dgidb"]["known"] is False and out["dgidb"]["interactions"] == [] and out["dgidb"]["gene_record"] is None
    assert out["chembl"]["known"] is False and out["chembl"]["drugs"] == [] and out["chembl"]["n_records"] == 0
    assert store.count() == 0 and empty.calls == [] and tools.log.genes == ["NOTAGENE1"] and tools.log.drug_records == []
    tools.drugs_for_gene({"gene": "NOTAGENE2"})
    tools.drugs_for_gene({"gene": "notagene1"})  # already asked: not a new gene
    with pytest.raises(ValueError, match="at most 2 distinct genes"):
        tools.drugs_for_gene({"gene": "NOTAGENE3"})
    assert tools.log.genes == ["NOTAGENE1", "NOTAGENE2"]
    # the candidate's own gene uses stage 3's Ensembl id and never asks Open Targets to map it
    assert tools._ensembl_id("CFTR", fakes) == ENSG and tools._ensembl_id("MTHFR", fakes) is None


def test_a_tool_service_failure_aborts_the_run_and_leaves_the_transcript(run_dir: Path):
    class BrokenChembl(ChemblRetriever):
        def drugs_for_target_symbol(self, symbol):
            raise ChemblError("status.json reports ChEMBL_37 but status 'DOWN', not 'UP': nothing served now can be cited")
    r = retrievers()
    r.chembl = BrokenChembl(StubHttp())
    fake = fake_client(turns=[ac.FakeTurn([("get_record", {"record_id": CFTR_VEP}), ("drugs_for_gene", {"gene": "CFTR"})])])
    with pytest.raises(ac.ToolFailure, match="tool 'drugs_for_gene' failed: ChemblError"):
        mr.run_medicine(run_dir, None, fake, "fake-model", "low", False, retrievers=r)
    out = run_dir / "06_medicine"
    failed = read(out / "transcripts" / "CFTR:hom.failed.json")
    assert failed["candidate_id"] == "CFTR:hom" and failed["transcript"][0]["tool_calls"][0]["name"] == "get_record"
    assert failed["tools"]["records_read"] == [CFTR_VEP] and failed["tools"]["genes"] == []
    assert not (out / "report.json").exists() and not (out / "manifest.json").exists()
    # the Open Targets and DGIdb records fetched before ChEMBL failed are in the store — a store, not a claim
    assert EvidenceStore(out / "evidence").count("opentargets") == 19
    # an offline cache miss is a service failure too
    fake = fake_client(turns=[ac.FakeTurn([("drugs_for_gene", {"gene": "CFTR"})])])
    with pytest.raises(ac.ToolFailure, match="offline"):
        mr.run_medicine(run_dir, None, fake, "fake-model", "low", False, http=Http(HttpCache(run_dir.parent / "cache"), offline=True))


# ------------------------------------------------------------------ after the model

def drug_record(record_id: str, payload: Any, url: str = "https://example.org/x") -> EvidenceRecord:
    """A hand-written record of a drug source (public names only)."""
    return EvidenceRecord(record_id=record_id, source=record_id.split(":", 1)[0], source_version="test", query={},
                          url=url, retrieved_at="2026-09-12T00:00:00+00:00", payload=payload)


IVACAFTOR_MOLECULE = drug_record(CH_IVACAFTOR, {"molecule_chembl_id": "CHEMBL2010601", "pref_name": "IVACAFTOR",
                                                "synonyms": ["KALYDECO", "VX-770"]},
                                 "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL2010601")
TEZACAFTOR_INTERACTION = drug_record(DG_TEZACAFTOR, {"drug": {"name": "TEZACAFTOR", "conceptId": "rxcui:1999382"}},
                                     "https://dgidb.org/genes/CFTR")
IVACAFTOR_MECHANISM = drug_record(CH_IVACAFTOR_MEC, {"molecule_chembl_id": "CHEMBL2010601", "action_type": "POSITIVE MODULATOR"})


BACKING = [PMID_SEARCH, NCT_SEARCH, CHEMBL_SEARCH_MEC, CHEMBL_SEARCH_IND, OT_TARGET]
"""The search records the scripted conversation made — what :func:`stage_checks` is
told to accept in a class's ``searched``."""


def test_stage_checks_drop_clear_and_redact_against_the_candidates_records(run_dir: Path):
    index = EvidenceIndex.from_run(run_dir, stages=mr.EVIDENCE_STAGES)
    trial = EvidenceStore(EVIDENCE).get("nct:NCT01807923")  # ivacaftor + lumacaftor, per its interventions
    records = [index.get(CFTR_VEP), index.get(PAPER), trial]
    resolver = mr.AccessionResolver(records)
    assert resolver.carries_chembl("CHEMBL2010601") is False
    text = ("NCT01807923 https://clinicaltrials.gov/study/NCT01807923 CHEMBL2010601 chembl:CHEMBL2010601 "
            "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL25 PMC1234567 rs113993960 VCV000007105 doi:10.1000/x "
            "PubMed 42616613 https://platform.opentargets.org/drug/CHEMBL2010601 HP:0002205 MONDO:0009061 lumacaftor")
    # ``chembl:CHEMBL2010601`` is a record-id citation, the validator's business, not a bare accession
    assert mr.accession_tokens(text) == ["NCT01807923", "https://clinicaltrials.gov/study/NCT01807923", "CHEMBL2010601",
                                         "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL25", "PMC1234567",
                                         "rs113993960", "VCV000007105", "doi:10.1000/x", "PubMed 42616613",
                                         "https://platform.opentargets.org/drug/CHEMBL2010601"]
    # the trial names ivacaftor, so the candidate stands on it; its ChEMBL id is carried by no record it cites
    data = dict(cftr_answer(), candidates=[candidate("Ivacaftor", rationale=text, chembl_id="chembl2010601", evidence_ids=[CH_IVACAFTOR],
                                                     trial_ids=["NCT01807923"])],
                mechanism=[{"statement": text, "evidence_ids": [CFTR_VEP]}], limits=[text])
    out = mr.stage_checks(data, records, searched=BACKING)
    def expected(first: int) -> str:
        """Five holes per field, numbered in the walk's order: mechanism, candidates, limits."""
        h = [f"[^{k}]" for k in range(first, first + 5)]
        return (f"NCT01807923 https://clinicaltrials.gov/study/NCT01807923 {h[0]} chembl:CHEMBL2010601 {h[1]} {h[2]} rs113993960 "
                f"VCV000007105 {h[3]} PubMed 42616613 {h[4]} HP:0002205 MONDO:0009061 lumacaftor")
    assert out.report["mechanism"][0]["statement"] == expected(1) and out.report["limits"] == [expected(11)]
    assert out.report["candidates"][0]["rationale"] == expected(6) and out.report["candidates"][0]["chembl_id"] is None
    assert out.positions["candidates"] == [0]
    assert [r.path for r in out.dropped if r.path.startswith("candidates")] == []  # the ladder's own drops are tested below
    assert [r.path for r in out.redacted][:2] == ["mechanism[0].statement", "mechanism[0].statement"]
    assert ("candidates[0].chembl_id", "chembl_id not carried by any record the candidate cites: chembl2010601; cleared") in [(r.path, r.reason) for r in out.redacted]
    # with the ChEMBL molecule among the records the id stands, upper-cased, and the bare mentions resolve
    with_drug = records + [IVACAFTOR_MOLECULE]
    out = mr.stage_checks(data, with_drug, searched=BACKING)
    assert out.report["candidates"][0]["chembl_id"] == "CHEMBL2010601"
    assert out.report["candidates"][0]["rationale"].startswith("NCT01807923 https://clinicaltrials.gov/study/NCT01807923 CHEMBL2010601 chembl:CHEMBL2010601 ")
    assert "platform.opentargets.org/drug/CHEMBL2010601" not in out.report["candidates"][0]["rationale"]  # a URL no record carries

    # -- the name rule: a candidate stands only on a cited drug-source record that names it
    with_drug = with_drug + [TEZACAFTOR_INTERACTION, IVACAFTOR_MECHANISM]
    data["candidates"] = [
        candidate("only variants", evidence_ids=[CFTR_VEP]),                                   # 0: no drug source at all
        candidate("Ivacaftor", evidence_ids=[" chembl:CHEMBL2010601"]),                        # 1: kept — id spelled with a space
        candidate("Lumacaftor", evidence_ids=[], trial_ids=["NCT01807923"]),                   # 2: kept — bare trial id, trial names it
        candidate("Aspirin", evidence_ids=[DG_TEZACAFTOR]),                                    # 3: a drug from memory on another drug's record
        candidate("Ivacaftor", evidence_ids=[CH_IVACAFTOR_MEC]),                               # 4: a mechanism row does not name the drug
        candidate("Ivacaftor", evidence_ids=[PAPER]),                                          # 5: kept — the paper mentions it
        candidate("Aspirin", evidence_ids=[PAPER]),                                            # 6: a paper about the gene, not the drug
        candidate("Tezacaftor", chembl_id="CHEMBL2010601", evidence_ids=[DG_TEZACAFTOR]),      # 7: kept; ivacaftor's id cleared
        candidate("Elexacaftor/Tezacaftor/Ivacaftor", evidence_ids=[DG_TEZACAFTOR, CH_IVACAFTOR]),  # 8: one component uncited
        candidate("Tezacaftor/ivacaftor (Symdeko)", evidence_ids=[DG_TEZACAFTOR, CH_IVACAFTOR]),  # 9: kept — a record per component
        candidate("Kalydeco (ivacaftor)", evidence_ids=[CH_IVACAFTOR]),                        # 10: kept — synonym in the molecule record
        candidate("x", evidence_ids=[CH_IVACAFTOR]),                                           # 11: too short to mean anything
        candidate("Ivacaftor", evidence_ids=["PMID:42616613"]),                                # 12: kept — prefix case-folded
    ]
    out = mr.stage_checks(data, with_drug, searched=BACKING)
    assert [c["name"] for c in out.report["candidates"]] == ["Ivacaftor", "Lumacaftor", "Ivacaftor", "Tezacaftor",
                                                             "Tezacaftor/ivacaftor (Symdeko)", "Kalydeco (ivacaftor)", "Ivacaftor"]
    assert out.positions["candidates"] == [1, 2, 5, 7, 9, 10, 12]
    assert [r.path for r in out.dropped if r.path.startswith("candidates")] == \
        ["candidates[0]", "candidates[3]", "candidates[4]", "candidates[6]", "candidates[8]", "candidates[11]"]
    assert [r for r in out.dropped if r.path == "candidates[3]"][0].reason == \
        ("drug candidate 'aspirin' cites no record from a drug source that names it "
         "(a drug row, an interaction, a ChEMBL molecule, a trial or a paper carrying the name)")
    assert out.report["candidates"][3]["chembl_id"] is None
    assert ("candidates[7].chembl_id", "chembl_id not carried by any record the candidate cites: CHEMBL2010601; cleared") in [(r.path, r.reason) for r in out.redacted]
    assert out.report["candidates"][1]["trial_ids"] == ["NCT01807923"]  # spelled as the model wrote it; the validator rewrites it
    assert mr._original_path("candidates[1].rationale", out.positions) == "candidates[2].rationale"
    assert mr._original_path("mechanism[1]", out.positions) == "mechanism[1]"  # a list the stage does not prune
    assert mr._original_path("intervention_classes[1]", out.positions) == "intervention_classes[1]"
    assert mr.cited_record_ids({"evidence_ids": [" PMID:1 ", "pmid:1", None], "trial_ids": ["nct01807923", "NCT01807923", "nct:NCT01807923"]}) == ["pmid:1", "nct:NCT01807923"]

    # -- the engine's own auxiliary ids are accessions too: they stand only when the record is citable
    gene_record = drug_record(DG_GENE, {"gene": {"name": "CFTR"}}, "https://dgidb.org/genes/CFTR")
    data["candidates"], data["limits"] = [], []
    data["mechanism"] = [{"statement": f"see {DG_GENE}, dgidb-gene:BRCA1, nct-search:deadbeef, chembl-search:f00d "
                                       "and pmid-search:cafe.", "evidence_ids": [CFTR_VEP]}]
    out = mr.stage_checks(data, with_drug + [gene_record], searched=BACKING)
    assert out.report["mechanism"][0]["statement"] == f"see {DG_GENE}, [^1], [^2], [^3] and [^4]."
    assert [r.reason for r in out.redacted] == [f"bare accession not carried by any citable record: {x}"
                                                for x in ("dgidb-gene:BRCA1", "nct-search:deadbeef", "chembl-search:f00d",
                                                          "pmid-search:cafe")]
    # the search records of this conversation are citable like any other record
    search_record = drug_record(CHEMBL_SEARCH_MEC, {"sent": {"needle": "conductance regulator"}, "total_count": 13, "ids": []},
                                "https://www.ebi.ac.uk/chembl/api/data/mechanism.json?mechanism_of_action__icontains=conductance+regulator")
    data["mechanism"] = [{"statement": f"ChEMBL curates 13 mechanism rows for the phrase [{CHEMBL_SEARCH_MEC}].",
                          "evidence_ids": [CHEMBL_SEARCH_MEC]}]
    out = mr.stage_checks(data, with_drug + [gene_record, search_record], searched=BACKING)
    assert "[^" not in out.report["mechanism"][0]["statement"]


def test_check_report_paths_name_the_models_positions_and_out_of_scope_ids(run_dir: Path):
    """The validator sees the list after the stage's drops; its paths are mapped back.
    A real record outside the candidate's scope (TP53's VEP record) is rejected and noted."""
    index = EvidenceIndex.from_run(run_dir, stages=mr.EVIDENCE_STAGES)
    chain = mr.EvidenceChain.model_validate(read(run_dir / "05_reason" / "chains" / "CFTR:hom.json"))
    bundle = mr.build_medicine_bundle(mr.find_candidate(run_dir, "CFTR:hom"), chain, run_dir, HPO, index)
    records = [index.get(rid) for rid in bundle.record_ids]
    scoped = EvidenceIndex.from_records(records)
    answer = dict(cftr_answer(), candidates=[
        candidate("Ivacaftor", evidence_ids=[CFTR_VEP]),                          # stage check: candidates[0] — no drug record
        candidate("Tezacaftor", evidence_ids=[TP53_VEP, PAPER]),                  # validator: candidates[1] — out of scope
        candidate("Ivacaftor", evidence_ids=[PAPER]),                             # kept: the paper names it
    ], mechanism=[cftr_answer()["mechanism"][0]], consequence=[], pathway_targets=[],
        intervention_classes=[intervention_class(POTENTIATION, searched=[PMID_SEARCH], evidence_ids=[PMID_SEARCH])],
        considered_and_rejected=[], surveillance=[], literature=[PAPER], follow_up_experiments=[], limits=[])
    cleaned, report = mr.check_report(answer, bundle, records, scoped, index, searched=[PMID_SEARCH])
    assert [d.name for d in cleaned.candidates] == ["Ivacaftor"]
    assert [(r.path, r.reason) for r in report.rejections] == [
        ("candidates[0]", ("drug candidate 'ivacaftor' cites no record from a drug source that names it "
                           "(a drug row, an interaction, a ChEMBL molecule, a trial or a paper carrying the name)")),
        ("intervention_classes[0]", f"unknown evidence id(s): {PMID_SEARCH}"),  # no search record is in this scope
        ("candidates[1]", f"unknown evidence id(s): {TP53_VEP}"),
        ("candidates[2].paediatric_safety", f"inline citation to a record not in the store: {CH_IVACAFTOR}"),
    ]
    assert report.counts["items_dropped"] == 3
    assert any(n == (f"out of scope: {TP53_VEP} exists in the run's evidence but was neither in the candidate's list "
                     "nor returned by a tool in this conversation; rejected") for n in report.notes)


# ------------------------------------------------------------------------------ cli

def test_cli_dry_run_prints_counts_and_no_gene(run_dir: Path):
    result = CliRunner().invoke(main, ["medicine", "--run", str(run_dir), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "medicine · " in result.output and "DRY RUN" in result.output and "best-priority candidate" in result.output
    assert "candidate: bundle and prompt written · chains available: 2 · no model called" in result.output
    assert "CFTR" not in result.output and "117559590" not in result.output
    assert result.output.rstrip().endswith(f"manifest: {run_dir / '06_medicine' / 'manifest.json'}")
    result = CliRunner().invoke(main, ["medicine", "--run", str(run_dir), "--dry-run", "--candidate", "MTHFR:hom"])
    assert result.exit_code == 1 and "FAILED: " in result.output and "no stage-5 chain" in result.output


def test_cli_runs_the_stage_and_names_the_candidate_only_when_asked(run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """With a scripted model that calls no tool, the real (offline, empty-cache) services are built and never asked."""
    answer = dict(cftr_answer(), candidates=[candidate("Ivacaftor", evidence_ids=[PAPER])], mechanism=[cftr_answer()["mechanism"][0]],
                  consequence=[], pathway_targets=[], intervention_classes=[], considered_and_rejected=[], surveillance=[],
                  literature=[PAPER], follow_up_experiments=[f"organoid assay [{PAPER}]"], limits=["no drug source was asked"])
    monkeypatch.setattr(ac, "AnthropicClient", lambda **kw: fake_client(answer, turns=[]))
    args = ["medicine", "--run", str(run_dir), "--model", "fake-model", "--effort", "low", "--cache", str(tmp_path / "cache"), "--offline"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "candidate: agent (fake-model, effort low)" in result.output
    assert ("candidate: candidates proposed: 0 (with trials: 0) · classes: 0 · rejected: 0 · follow-up: 1 · "
            "validator rejections: 1 · records added: 0") in result.output
    assert "the ladder was not walked: see the manifest's notes" in result.output
    assert "tokens in/out: 0/0 · api calls: 0 · tool calls: 0" in result.output
    assert "CFTR" not in result.output and "Ivacaftor" not in result.output and "117559590" not in result.output
    m = read(run_dir / "06_medicine" / "manifest.json")
    assert m["params"]["offline"] is True and m["params"]["cache_root"] == str(tmp_path / "cache")
    assert m["params"]["retrievers"]["opentargets"]["per_second"] == 3.0 and m["params"]["retrievers"]["trials"]["page_size"] == 50
    assert not (tmp_path / "cache").exists() or not any((tmp_path / "cache").iterdir())  # nothing was fetched
    md = (run_dir / "06_medicine" / "report.md").read_text()
    # the model named no class, so the candidate it proposed belongs to none and is dropped: the report says so
    assert NO_CANDIDATE_NO_CLASS in md and "survived" not in md
    order = [md.index(x) for x in ("## Variant mechanism", "## Intervention classes searched", "## Drug candidates",
                                   "## Follow-up experiments", "## Limits")]
    assert order == sorted(order)
    result = CliRunner().invoke(main, args + ["-v"])
    assert result.exit_code == 0, result.output
    assert "CFTR:hom: candidates proposed: 0" in result.output


def test_cli_reports_an_agent_failure_and_exits_1(run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class Failing:
        def run(self, request: ac.AgentRequest) -> ac.AgentResult:
            raise ac.AgentError("Anthropic API rate limit (retry-after 30s): slow down")
    monkeypatch.setattr(ac, "AnthropicClient", lambda **kw: Failing())
    result = CliRunner().invoke(main, ["medicine", "--run", str(run_dir), "--offline", "--cache", str(tmp_path / "cache")])
    assert result.exit_code == 1 and "FAILED: Anthropic API rate limit" in result.output
    assert not (run_dir / "06_medicine" / "manifest.json").exists()


def test_a_client_that_cannot_be_built_is_one_agent_error_and_a_real_bug_still_surfaces(run_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Stage 6's twin of stage 5's check: the SDK's "Could not resolve authentication"
    is a TypeError raised inside the constructor, converted to one AgentError before any
    candidate is touched — while a TypeError from a programming error in a client's
    ``run`` is not swallowed."""
    def no_client(**kw: Any) -> None:
        raise TypeError("Could not resolve authentication method.")
    monkeypatch.setattr(ac, "AnthropicClient", no_client)
    with pytest.raises(ac.AgentError, match="could not build the Anthropic client: Could not resolve authentication"):
        mr.run_medicine(run_dir, None, None, "claude-opus-5", "high", False, retrievers=retrievers())
    out = run_dir / "06_medicine"
    assert not (out / "report.json").exists() and not (out / "manifest.json").exists()

    class Broken:
        def run(self, request: ac.AgentRequest) -> ac.AgentResult:
            raise TypeError("unrelated programming error")
    with pytest.raises(TypeError, match="unrelated"):
        mr.run_medicine(run_dir, None, Broken(), "fake-model", "low", False, retrievers=retrievers())


def test_cli_reports_a_malformed_chain_by_field_path_never_by_value(run_dir: Path, tmp_path: Path):
    chain_path = run_dir / "05_reason" / "chains" / "CFTR:hom.json"
    chain = read(chain_path)
    chain["variants"][0]["criteria"][0]["strength"] = f"very strong, see {CFTR}"   # a value the message must not echo
    chain["phase_statement"] = None
    chain_path.write_text(json.dumps(chain, sort_keys=True, indent=1) + "\n")
    with pytest.raises(ValueError, match="does not fit EvidenceChain: 2 error") as info:
        mr.run_medicine(run_dir, dry_run=True)
    assert "variants.0.criteria.0.strength: literal_error" in str(info.value) and "phase_statement: string_type" in str(info.value)
    assert "117559590" not in str(info.value) and "very strong" not in str(info.value)
    result = CliRunner().invoke(main, ["medicine", "--run", str(run_dir), "--dry-run", "--cache", str(tmp_path / "cache")])
    assert result.exit_code == 1 and "FAILED: the stage-5 chain for the selected candidate does not fit" in result.output
    assert "117559590" not in result.output and "CFTR" not in result.output


# ------------------------------------------------------------------------------ live

@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ENGINE_LIVE_TESTS"), reason="set ENGINE_LIVE_TESTS=1 to hit the public drug and trial APIs")
def test_live_drug_and_trial_tools_write_citable_records(tmp_path: Path):
    """Public gene only (CFTR): Open Targets, DGIdb, ChEMBL and ClinicalTrials.gov, one
    request each way, through the real Http and a throw-away cache."""
    http = Http(HttpCache(tmp_path / "cache"), limiter=RateLimiter({"www.ebi.ac.uk": 3.0}))
    store = EvidenceStore(tmp_path / "evidence")
    tools = MedicineTools(EvidenceIndex([store]), store, mr.default_retrievers(http), gene_symbol="CFTR", gene_id=ENSG)
    out = tools.drugs_for_gene({"gene": "CFTR"})
    assert out["opentargets"]["known"] and out["dgidb"]["known"] and out["chembl"]["known"]
    assert any(d["name"] == "IVACAFTOR" for d in out["opentargets"]["drugs"])
    assert any(d["chembl_id"] == "CHEMBL2010601" and d["max_phase"] == "4" for d in out["chembl"]["drugs"])
    assert store.exists(OT_TARGET) and store.exists(CH_IVACAFTOR) and store.exists(DG_GENE)
    trials = tools.search_trials({"condition": "cystic fibrosis", "intervention": "ivacaftor", "term": None, "max_results": 3})
    assert trials["returned"] == 3 and trials["total_count"] >= 100 and all(store.exists(t["record_id"]) for t in trials["trials"])
    assert store.get(trials["trials"][0]["record_id"]).source_version.startswith("ClinicalTrials.gov API")
    assert http.live_requests > 0 and len(tools.log.drug_records) >= 50 and len(tools.log.trials) == 3
