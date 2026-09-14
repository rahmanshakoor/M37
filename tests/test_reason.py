"""Stage 5 over a synthetic run directory built from public data only.

The run directory is assembled per test from ``tests/fixtures/agents/evidence`` (MTHFR
rs1801133, CFTR p.Phe508del, TP53 p.Arg175His and the Frosst 1995 paper) plus
``tests/fixtures/reason/`` (a stage-3 ``candidates.json`` and a stage-4 ``joined.json``
in the writers' shapes). The model is ``FakeClient`` — scripted to call every tool,
including with bad arguments, and to answer with a chain that cites a paper it
fetched, a paper it never fetched, a fabricated ClinVar accession and a frequency
criterion the data contradict, and in other tests a paper it remembers but never
retrieved, another candidate's record, a bare accession and a foreign variant — so
what the tests check is the stage's plumbing and the gates, never a model's
judgement. Europe PMC is a stub ``Http`` serving
the recorded fixtures in ``tests/fixtures/literature`` for the exact requests the
tools make; one live test talks to Europe PMC when ``ENGINE_LIVE_TESTS`` is set.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import anthropic
import pytest
from click.testing import CliRunner

from engine.agents import client as ac
from engine.agents.bundle import build_bundle
from engine.agents.schema import EvidenceChain
from engine.agents.validator import EvidenceIndex
from engine.cli import main
from engine.reason import run as rr
from engine.reason.checks import AccessionResolver, accession_tokens, stage_checks
from engine.reason.prompts import PROMPT_VERSION, SYSTEM_PROMPT, instructions_sha256, prompt_sha256
from engine.reason.tools import ReasonTools, check_query, parse_pmid
from engine.retrieve.http import Http, HttpCache, HttpError, RateLimiter, Response
from engine.retrieve.store import EvidenceRecord, EvidenceStore

FIXTURES = Path(__file__).parent / "fixtures"
EVIDENCE = FIXTURES / "agents" / "evidence"
REASON = FIXTURES / "reason"
LITERATURE = FIXTURES / "literature"

CFTR = "7:117559590:ATCT:A"
CFTR_GNOMAD = "gnomad:7-117559590-ATCT-A"
CFTR_VEP = f"vep:{CFTR}"
CFTR_CLINVAR = "clinvar:VCV000007105"
TP53 = "17:7675088:C:T"
TP53_GNOMAD = "gnomad:17-7675088-C-T"
TP53_VEP = f"vep:{TP53}"
TP53_CLINVAR = "clinvar:VCV000012374"
FROSST = "pmid:7647779"            # already in the stage-2 store
ZELICHA = "pmid:42269400"          # fetched by get_paper in the tests (fixture fetch_42269400)
SEARCH_QUERY = 'TITLE_ABS:"CFTR" AND TITLE_ABS:"F508del"'
SEARCH_TOP = "pmid:42616613"       # first result of the recorded search
FAKE_PMID = "pmid:99999999"        # Europe PMC answers hitCount 0 (fixture fetch_99999999)
FAKE_CLINVAR = "clinvar:VCV999999999"
HPO = ["HP:0002205", "HP:0006528", "HP:0012236"]


# ------------------------------------------------------------------------ fixtures

def fixture(name: str) -> dict:
    return json.loads((LITERATURE / f"{name}.json").read_text())


class StubHttp:
    """Serves recorded Europe PMC responses keyed by the exact request; refuses anything else."""

    def __init__(self, *fixtures: dict):
        self.limiter = RateLimiter(default_per_second=0)
        self.calls: list[tuple[str, str, dict | None]] = []
        self._by_key: dict[str, dict] = {}
        for fx in fixtures:
            r = fx["request"]
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


def stub_http() -> StubHttp:
    return StubHttp(fixture("search_page1"), fixture("fetch_7647779"), fixture("fetch_42269400"), fixture("fetch_99999999"))


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
    return run


def criterion(code: str, strength: str, ids: list[str], met: bool = True, just: str = "because") -> dict[str, Any]:
    return {"code": code, "strength": strength, "met": met, "justification": just, "evidence_ids": ids}


def cftr_answer() -> dict[str, Any]:
    """What a model that mostly follows the rules — and breaks three of them — writes."""
    return {
        "candidate_id": "CFTR:hom",
        "variants": [{"key": CFTR, "classification": "pathogenic", "summary": f"Homozygous p.Phe508del [{CFTR_VEP}] [{CFTR_CLINVAR}].",
                      "criteria": [
            criterion("PS3", "strong", [SEARCH_TOP], just=f"functional rescue assays characterise F508del-CFTR [{SEARCH_TOP}]"),
            criterion("PM2", "moderate", [CFTR_GNOMAD], met=True, just="claimed rare"),            # af 0.0119: the data disagree
            criterion("PM4", "moderate", [CFTR_VEP], just="in-frame deletion of one residue in a non-repeat region"),
            criterion("PP5", "supporting", [CFTR_CLINVAR], just="ClinVar Pathogenic, practice guideline (4 stars)"),
            criterion("PP4", "supporting", ["exomiser:CFTR"], just="CF phenotype terms; Exomiser rank 1"),
            criterion("PS1", "strong", [FAKE_CLINVAR], just="fabricated accession"),                 # not in any store
            criterion("PP3", "supporting", [CFTR_VEP], just="CADD 17.55; see also PMID 99999999"),   # inline fabricated PMID
        ]}],
        "phase_statement": f"Homozygous genotype: phase is not in question [{CFTR_GNOMAD}]; a deletion of the other allele cannot be excluded.",
        "mechanism_hypothesis": f"Loss of function by misfolding and ER retention [{CFTR_VEP}] [{ZELICHA}] [{FROSST}].",
        "limits": ["No parental samples", "No functional assay on this proband's cells"],
        "what_would_change_the_call": ["Sweat chloride measurement", "Parental genotypes"],
        "literature": [SEARCH_TOP, ZELICHA, FROSST, FAKE_PMID],
    }


def cftr_bundle_only_answer() -> dict[str, Any]:
    """An answer that cites nothing but the bundle's own records — valid without any tool call."""
    a = cftr_answer()
    a["variants"][0]["criteria"] = [
        criterion("PM2", "moderate", [CFTR_GNOMAD], met=False, just="af 0.0119 with 58 homozygotes"),
        criterion("PM4", "moderate", [CFTR_VEP], just="in-frame deletion"),
        criterion("PP5", "supporting", [CFTR_CLINVAR], just="ClinVar Pathogenic, practice guideline"),
        criterion("PP4", "supporting", [], just="CF phenotype terms"),
    ]
    a["variants"][0].pop("classification")
    a["mechanism_hypothesis"] = f"Loss of function by misfolding [{CFTR_VEP}]"
    a["literature"] = []
    return a


def tp53_answer() -> dict[str, Any]:
    return {
        "candidate_id": "TP53:het_single",
        "variants": [{"key": TP53, "summary": "Heterozygous p.Arg175His at low depth.", "criteria": [
            criterion("PM2", "moderate", [TP53_GNOMAD], just="7 alleles in 1.6 M, no homozygotes"),
            criterion("PP3", "supporting", [TP53_VEP], just="CADD 25.9"),
            criterion("PP5", "supporting", [TP53_CLINVAR], just="ClinVar Pathogenic, expert panel"),
        ]}],
        "phase_statement": "Single heterozygous variant; phase does not arise.",
        "mechanism_hypothesis": f"Dominant negative [{TP53_VEP}].",
        "limits": ["Depth 16"], "what_would_change_the_call": ["Orthogonal confirmation of the genotype"], "literature": [],
    }


SCRIPT = [ac.FakeTurn([("get_record", {"record_id": CFTR_GNOMAD}),
                       ("get_record", {"record_id": "nope:1"}),
                       ("search_literature", {"query": SEARCH_QUERY, "max_results": 5})], text="reading"),
          ac.FakeTurn([("get_paper", {"pmid": "42269400"}),
                       ("get_paper", {"pmid": "pmid:7647779"}),
                       ("get_paper", {"pmid": "99999999"}),
                       ("search_literature", {"query": f"CFTR {CFTR}", "max_results": None})], text="checking")]


def fake_client(outputs: Any = None, turns: list[ac.FakeTurn] | None = None) -> ac.FakeClient:
    return ac.FakeClient(outputs if outputs is not None else [cftr_answer(), tp53_answer()], turns=SCRIPT if turns is None else turns)


def read(path: Path) -> Any:
    return json.loads(path.read_text())


# ------------------------------------------------------------------------ dry run

def test_dry_run_writes_bundles_and_the_exact_prompt_without_a_model(run_dir: Path):
    manifest_path = rr.run_reason(run_dir, top_n=2, dry_run=True)
    out = run_dir / "05_reason"
    assert manifest_path == out / "manifest.json"
    assert sorted(p.name for p in (out / "bundles").iterdir()) == ["CFTR:hom.json", "TP53:het_single.json"]
    assert sorted(p.name for p in (out / "prompts").iterdir()) == ["CFTR:hom.json", "CFTR:hom.md", "TP53:het_single.json", "TP53:het_single.md"]
    assert not (out / "chains").exists() and not (out / "transcripts").exists() and not (out / "evidence_chain.md").exists()
    assert read(out / "evidence" / "index.json") == {}

    prompt = read(out / "prompts" / "CFTR:hom.json")
    bundle = read(out / "bundles" / "CFTR:hom.json")
    assert prompt["system"] == SYSTEM_PROMPT and prompt["system_prompt_sha256"] == prompt_sha256()
    assert prompt["user"].endswith(bundle["text"]) and "up to 10 tool-calling turns" in prompt["user"]
    # every fixed text the model receives is in the file, and one hash covers them all
    assert prompt["final_instruction"] == ac.FINAL_INSTRUCTION and prompt["tool_budget_error"] == ac.TOOL_BUDGET_ERROR
    assert prompt["instructions_sha256"] == instructions_sha256() != prompt_sha256()
    assert prompt["model"] == "claude-opus-5" and prompt["effort"] == "high" and prompt["thinking"] == {"type": "adaptive"}
    assert [t["name"] for t in prompt["tool_definitions"]] == ["get_record", "search_literature", "get_paper"]
    assert all(t["strict"] is True and t["input_schema"]["additionalProperties"] is False for t in prompt["tool_definitions"])
    assert prompt["tool_definitions"][1]["input_schema"]["required"] == ["max_results", "query"]
    variant_schema = prompt["output_schema"]["$defs"]["VariantChain"]["properties"]
    assert "classification" not in variant_schema  # the engine's, never asked of the model
    assert variant_schema["key"]["enum"] == [CFTR]  # the API itself refuses a respelled or foreign variant
    assert "PVS1" in prompt["output_schema"]["$defs"]["Criterion"]["properties"]["code"]["enum"]
    # the model sees the case HPO terms, the stage-4 record and every citable id
    assert "case HPO: HP:0002205, HP:0006528, HP:0012236" in bundle["text"]
    assert "exomiser: rank 1 · combined 0.97 · phenotype 0.91 · variant 0.95 · moi AR [exomiser:CFTR]" in bundle["text"]
    assert bundle["record_ids"] == [CFTR_CLINVAR, "exomiser:CFTR", CFTR_GNOMAD, CFTR_VEP]
    md = (out / "prompts" / "CFTR:hom.md").read_text()
    assert md.startswith("# Prompt — CFTR:hom\n") and "## System\n" in md and "## User\n" in md and bundle["text"] in md
    assert ac.FINAL_INSTRUCTION in md and ac.TOOL_BUDGET_ERROR in md

    m = read(manifest_path)
    assert m["stage"] == "reason" and m["params"]["dry_run"] is True
    assert m["params"]["candidates"] == ["CFTR:hom", "TP53:het_single"] and m["params"]["top_n"] == 2
    assert m["params"]["prompt_version"] == PROMPT_VERSION and m["params"]["system_prompt_sha256"] == prompt_sha256()
    assert m["params"]["instructions_sha256"] == instructions_sha256()
    assert m["params"]["final_instruction_sha256"] == prompt_sha256(ac.FINAL_INSTRUCTION)
    assert m["params"]["evidence_stages"] == ["02_retrieve", "04_rank", "05_reason"]
    assert m["params"]["literature"]["abstract_chars"] == 6000 and "5+ digits" in m["params"]["literature"]["coordinates_in_queries"]
    assert m["params"]["hpo"] == HPO and m["params"]["hpo_source"] == "04_rank/joined.json"
    validator = m["params"]["validator"]
    assert validator["thresholds"] == {"ba1_min_af": 0.05, "bs1_min_af": 0.01, "pm2_max_af": 0.0001} and validator["af_field"] == "gnomad_af"
    assert validator["rules"]["PM2"].startswith("met iff af < 0.0001, or the gnomAD record says") and validator["rules"]["BA1"] == "met iff af > 0.05"
    assert set(validator["rules"]) >= {"af_field", "fallback", "PM2", "BS1", "BA1", "unverified", "strength_cap"}
    assert m["params"]["disclosure"].startswith("Anthropic API, model claude-opus-5, effort high;")
    assert m["counts"] == {"candidates_total": 3, "candidates_selected": 2, "candidates_reasoned": 0, "chains_written": 0,
                           "evidence_records": 0, "evidence_records_added": 0}
    assert set(m["inputs"]) == {"candidates", "retrieve_evidence_index", "rank_joined", "rank_evidence_index"}
    assert all(v["sha256"] for v in m["inputs"].values())
    assert m["tools"]["anthropic-sdk"]
    assert any(n.startswith("dry run:") for n in m["notes"])

    # deterministic: the same run directory gives the same bytes
    before = {p.name: p.read_bytes() for d in ("bundles", "prompts") for p in (out / d).iterdir()}
    rr.run_reason(run_dir, top_n=2, dry_run=True)
    assert {p.name: p.read_bytes() for d in ("bundles", "prompts") for p in (out / d).iterdir()} == before


def test_prompt_states_the_rules_the_validator_enforces():
    for phrase in ("Never invent a record id", "Applied at <strength> strength", "phase_statement", "cannot establish",
                   "Do not state a classification", "PP4, PM3, PS2, PM6, PS4, PP1, BS4, BP2, BP5",
                   "Never put a genomic coordinate in a search query", "PM2 at supporting",
                   "A bare accession in prose", "cites no `pmid:` record is deleted", "`get_record` serves only",
                   "Never use numbered references", "lowered, never raised", "PP3 and BP4 at most strong"):
        assert phrase in SYSTEM_PROMPT, phrase
    assert "2026-" not in SYSTEM_PROMPT  # no clock, so the cached prefix is stable across runs


# ---------------------------------------------------------------------- end to end

def test_end_to_end_chain_validated_classified_and_rendered(run_dir: Path):
    http = stub_http()
    fake = fake_client()
    manifest_path = rr.run_reason(run_dir, 2, fake, "fake-model", "low", False, http=http, progress=lambda s: None)
    out = run_dir / "05_reason"

    # -- tools: every scripted call ran against the real handlers
    calls = [(c.name, c.is_error) for c in fake.calls[:7]]
    assert calls == [("get_record", False), ("get_record", True), ("search_literature", False),
                     ("get_paper", False), ("get_paper", False), ("get_paper", True), ("search_literature", True)]
    assert '"variant_id": "7-117559590-ATCT-A"' in fake.calls[0].result
    assert fake.calls[1].result.startswith("Error: KeyError") and "nope:1" in fake.calls[1].result
    search = json.loads(fake.calls[2].result)
    assert search["hit_count"] == 1310 and search["returned"] == 5 and search["papers"][0]["record_id"] == SEARCH_TOP
    assert search["sent"] == f"({SEARCH_QUERY}) AND SRC:MED" and search["search_record"].startswith("pmid-search:")
    paper = json.loads(fake.calls[3].result)
    assert paper["record_id"] == ZELICHA and paper["year"] == "2026" and paper["abstract"] and paper["url"].endswith("/MED/42269400")
    served = json.loads(fake.calls[4].result)
    assert served["record_id"] == FROSST and served["title"].startswith("A candidate genetic risk factor")
    assert "no PubMed record for PMID 99999999" in fake.calls[5].result
    assert "must not contain a genomic coordinate" in fake.calls[6].result
    # the second candidate's identical script asks for the first candidate's gnomAD record: not its to read
    assert fake.calls[7].name == "get_record" and fake.calls[7].is_error
    assert fake.calls[7].result.startswith("Error: KeyError") and "bundle or among this conversation" in fake.calls[7].result
    # the coordinate never left the process; the second candidate's identical calls hit the stub again except
    # for the paper now held in the stage store
    queries = [p.get("query") for _, _, p in http.calls]
    assert not any("117559590" in (q or "") for q in queries)
    assert queries.count(f"({SEARCH_QUERY}) AND SRC:MED") == 2 and queries.count("EXT_ID:42269400 AND SRC:MED") == 1
    assert queries.count("EXT_ID:99999999 AND SRC:MED") == 2 and "EXT_ID:7647779 AND SRC:MED" not in queries

    # -- the stage store holds what the tools fetched, in the stage-2 format
    stage_store = EvidenceStore(out / "evidence")
    ids = [r.record_id for r in stage_store.iter()]
    assert ZELICHA in ids and SEARCH_TOP in ids and FROSST not in ids and FAKE_PMID not in ids
    assert sum(1 for i in ids if i.startswith("pmid:")) == 6 and sum(1 for i in ids if i.startswith("pmid-search:")) == 1
    index = read(out / "evidence" / "index.json")
    assert index[ZELICHA]["url"] == "https://europepmc.org/article/MED/42269400" and index[ZELICHA]["source_version"] == "Europe PMC REST 6.9"
    rec = stage_store.get(ZELICHA)
    assert rec.query == {"pmid": "42269400"} and rec.payload["pmid"] == "42269400"

    # -- the chain: fabricated ids gone, the frequency claim overruled, the verdict the engine's
    chain = EvidenceChain.model_validate(read(out / "chains" / "CFTR:hom.json"))
    v = chain.variants[0]
    assert v.key == CFTR and v.classification == "likely_pathogenic"  # PS3 4 + PM4 2 + PP4 1 = 7; the model had said "pathogenic"
    assert v.points == 7 and v.classification_richards_2015 == "likely_pathogenic"
    assert [c.code for c in v.criteria] == ["PS3", "PM2", "PM4", "PP5", "PP4", "PP3"]
    assert v.criteria[1].met is False and v.criteria[1].justification.startswith("[DISPUTED — PM2 recomputed from gnomad:7-117559590-ATCT-A: af=0.0119")
    assert v.criteria[3].met is False and v.criteria[3].justification.startswith("[RETIRED — PP5 is retired")  # ClinVar is concordance, not a criterion
    assert v.criteria[5].met is False and v.criteria[5].justification.startswith("[DISPUTED — PP3 recomputed from vep:7:117559590:ATCT:A: CADD 17.55")
    assert v.criteria[5].justification.endswith("CADD 17.55; see also PMID [citation removed: no such record]")
    assert chain.literature == [SEARCH_TOP, ZELICHA, FROSST]
    assert FAKE_CLINVAR not in chain.model_dump_json() and "99999999" not in chain.model_dump_json()
    tp53 = EvidenceChain.model_validate(read(out / "chains" / "TP53:het_single.json"))
    assert tp53.variants[0].classification == "vus" and [c.code for c in tp53.variants[0].criteria] == ["PM2", "PP3", "PP5"]
    assert tp53.variants[0].points == 3 and tp53.variants[0].criteria[1].strength == "supporting"  # REVEL 0.922 allows moderate; the model's lower call stands

    validation = read(out / "validation" / "CFTR:hom.json")
    assert {(r["path"], r["reason"]) for r in validation["rejections"]} == {
        ("variants[0].criteria[5]", f"unknown evidence id(s): {FAKE_CLINVAR}"),
        ("variants[0].criteria[6].justification", f"inline PMID not in the store: {FAKE_PMID}"),
        ("literature[3]", f"no such literature record in the store: {FAKE_PMID}"),
    }
    assert validation["disputes"][0]["code"] == "PM2" and validation["disputes"][0]["recomputed_met"] is False
    assert validation["counts"]["classification_replaced"] == 1 and validation["rules"]["af_field"] == "gnomad_af"

    # -- the manifest logs every rejection, the usage, the disclosure and every threshold
    m = read(manifest_path)
    assert m["counts"]["candidates_reasoned"] == 2 and m["counts"]["chains_written"] == 2 and m["counts"]["rejections"] == 3
    assert m["counts"]["classifications"] == {"CFTR:hom": {CFTR: "likely_pathogenic"}, "TP53:het_single": {TP53: "vus"}}
    assert m["counts"]["validation"]["CFTR:hom"]["items_dropped"] == 1 and m["counts"]["validation"]["CFTR:hom"]["criteria_kept"] == 6
    assert m["counts"]["usage"]["tool_calls"] == 14 and m["counts"]["usage"]["tool_errors"] == 7 and m["counts"]["usage"]["api_calls"] == 0
    assert m["counts"]["evidence_records_added"] == 7 and m["counts"]["evidence_records"] == 7
    assert f"CFTR:hom: rejected variants[0].criteria[5]: unknown evidence id(s): {FAKE_CLINVAR}" in m["notes"]
    assert any(n.startswith("CFTR:hom: disputed variants[0].criteria[1]: PM2 recomputed") for n in m["notes"])
    assert any("the model said 'pathogenic'; replaced by the engine's 'likely_pathogenic'" in n for n in m["notes"])
    assert m["params"]["validation"]["CFTR:hom"]["rejections"][0]["reason"] == f"unknown evidence id(s): {FAKE_CLINVAR}"
    assert m["params"]["model"] == "fake-model" and m["params"]["effort"] == "low" and m["params"]["disclosure"].startswith("FakeClient")
    assert m["params"]["agent"]["CFTR:hom"]["request"]["tools"] == ["get_record", "search_literature", "get_paper"]
    # one fact per paper — a tool returned it — whichever store served it; the same for both candidates
    papers = m["params"]["tools_used"]["CFTR:hom"]["papers"]
    assert papers[0] == SEARCH_TOP and papers[-2:] == [ZELICHA, FROSST] and len(papers) == 7
    assert m["params"]["tools_used"]["TP53:het_single"]["papers"] == papers
    assert set(m["params"]["tools_used"]["CFTR:hom"]) == {"searches", "papers", "records_read"}
    assert m["params"]["tools_used"]["CFTR:hom"]["records_read"] == [CFTR_GNOMAD]
    assert m["params"]["tools_used"]["TP53:het_single"]["records_read"] == []
    assert m["params"]["validator"]["thresholds"]["pm2_max_af"] == 0.0001 and m["params"]["max_turns"] == 10
    assert m["outputs"]["evidence_chain"]["sha256"] and m["outputs"]["evidence_index"]["sha256"]

    # -- the transcript: what the model asked, what it was told, what it wrote
    transcript = read(out / "transcripts" / "CFTR:hom.json")
    assert [t["stop_reason"] for t in transcript["transcript"]] == ["tool_use", "tool_use", "end_turn"]
    assert transcript["transcript"][0]["tool_calls"][1]["is_error"] is True
    assert json.loads(transcript["final_text"])["variants"][0]["classification"] == "pathogenic"  # as written, before the gate
    assert transcript["tools"]["searches"][0].startswith("pmid-search:")

    # -- the markdown: every chain, references with URLs, nothing fabricated
    md = (out / "evidence_chain.md").read_text()
    assert md.startswith("# Evidence chains\n\n2 candidate(s) · model fake-model · effort low")
    assert f"| 1 | CFTR:hom | CFTR | hom | {CFTR} | likely pathogenic |" in md
    assert f"| 2 | TP53:het_single | TP53 | het_single | {TP53} | vus |" in md
    assert "# Evidence chain — CFTR:hom" in md and f"## Variant {CFTR} — likely pathogenic" in md
    assert "- **PM2** · moderate · not met — [DISPUTED — PM2 recomputed from" in md
    assert "- **PP4** · supporting · met — CF phenotype terms; Exomiser rank 1 [exomiser:CFTR]" in md
    refs = md.split("# Evidence chain — CFTR:hom")[1].split("## References")[1].split("---")[0]
    assert f"- [{ZELICHA}] — https://europepmc.org/article/MED/42269400" in refs
    assert f"- [{SEARCH_TOP}] — https://europepmc.org/article/MED/42616613" in refs
    assert "- [exomiser:CFTR] — https://www.ncbi.nlm.nih.gov/gene/1080" in refs
    assert f"- [{CFTR_GNOMAD}] — https://gnomad.broadinstitute.org/variant/7-117559590-ATCT-A?dataset=gnomad_r4" in refs
    assert f"[{ZELICHA}] Zelicha" in md.split("## Literature")[1]
    assert "VCV999999999" not in md and "99999999" not in md
    assert md.count("_FakeClient (scripted, no API call), model fake-model, effort low_") == 2
    assert "2026-" not in md  # no clock in the report

    # -- deterministic, transcripts included: a rerun in place (papers now served from the stage store) and a
    # rerun in a fresh copy of the run directory both give the same bytes; only the manifest carries a clock
    def snapshot(stage: Path) -> dict[str, bytes]:
        return {str(p.relative_to(stage)): p.read_bytes() for p in stage.rglob("*") if p.is_file() and p.name != "manifest.json"}
    before = snapshot(out)
    rr.run_reason(run_dir, 2, fake_client(), "fake-model", "low", False, http=stub_http())
    assert snapshot(out) == before
    fresh = run_dir.parent / "fresh"
    shutil.copytree(run_dir, fresh, ignore=shutil.ignore_patterns("05_reason"))
    rr.run_reason(fresh, 2, fake_client(), "fake-model", "low", False, http=stub_http())
    assert snapshot(fresh / "05_reason") == before
    m2 = read(fresh / "05_reason" / "manifest.json")
    assert {k: v for k, v in m2["counts"].items()} == m["counts"] and m2["params"]["tools_used"] == m["params"]["tools_used"]


def test_top_n_selects_candidates_in_priority_order(run_dir: Path):
    fake = fake_client([cftr_bundle_only_answer()], turns=[])
    manifest = read(rr.run_reason(run_dir, 1, fake, "fake-model", "low", False, http=stub_http()))
    assert manifest["params"]["candidates"] == ["CFTR:hom"] and manifest["counts"]["candidates_total"] == 3
    assert [p.name for p in (run_dir / "05_reason" / "chains").iterdir()] == ["CFTR:hom.json"]
    assert len(fake.requests) == 1 and fake.requests[0].max_turns == 10
    # an answer citing only records the bundle already held needs nothing from the tools, and says so
    assert manifest["params"]["tools_used"]["CFTR:hom"] == {"searches": [], "papers": [], "records_read": []}
    assert manifest["counts"]["rejections"] == 0 and manifest["counts"]["classifications"] == {"CFTR:hom": {CFTR: "vus"}}
    with pytest.raises(ValueError):
        rr.run_reason(run_dir, 0, fake, dry_run=True)


def test_a_citation_outside_the_candidates_scope_is_rejected_even_when_the_record_exists(run_dir: Path):
    """The model remembers papers and records it was never shown here: the Frosst paper
    (in the stage-2 store), a paper an earlier run left in the stage store, TP53's VEP
    and ClinVar records, and a record planted by a later stage. Every one of them
    resolves in the run directory; none is in this candidate's bundle or tool results."""
    stage_store = EvidenceStore(run_dir / "05_reason" / "evidence")
    stage_store.put(EvidenceRecord(record_id="pmid:11111111", source="pmid", source_version="Europe PMC REST 6.9", query={"pmid": "11111111"},
                                   url="https://europepmc.org/article/MED/11111111", retrieved_at="2026-09-01T00:00:00+00:00",
                                   payload={"pmid": "11111111", "title": "left by an earlier run", "pubYear": "2020"}))
    later = EvidenceStore(run_dir / "06_medicine" / "evidence")
    later.put(EvidenceRecord(record_id="pmid:12345678", source="pmid", source_version="Europe PMC REST 6.9", query={"pmid": "12345678"},
                             url="https://europepmc.org/article/MED/12345678", retrieved_at="2026-09-01T00:00:00+00:00",
                             payload={"pmid": "12345678", "title": "planted by stage 6", "pubYear": "2021"}))
    answer = cftr_bundle_only_answer()
    answer["variants"][0]["summary"] += " See also [pmid:11111111]."
    answer["variants"][0]["criteria"] += [
        criterion("PS3", "strong", [FROSST], just="a functional study the model remembers"),
        criterion("PM1", "moderate", [TP53_VEP], just="hot spot (another variant's VEP record)"),
        criterion("PM5", "moderate", [TP53_CLINVAR], just="another variant's ClinVar record"),
        criterion("PP2", "supporting", ["pmid:12345678"], just="from a later stage's store"),
    ]
    answer["literature"] = [FROSST, "pmid:11111111", "pmid:12345678"]
    fake = fake_client([answer], turns=[])
    m = read(rr.run_reason(run_dir, 1, fake, "fake-model", "low", False, http=stub_http()))
    chain = EvidenceChain.model_validate(read(run_dir / "05_reason" / "chains" / "CFTR:hom.json"))
    assert [c.code for c in chain.variants[0].criteria] == ["PM2", "PM4", "PP5", "PP4"] and chain.literature == []
    assert m["counts"]["classifications"] == {"CFTR:hom": {CFTR: "vus"}}
    v = read(run_dir / "05_reason" / "validation" / "CFTR:hom.json")
    assert {r["reason"] for r in v["rejections"]} >= {
        f"unknown evidence id(s): {FROSST}", f"unknown evidence id(s): {TP53_VEP}", f"unknown evidence id(s): {TP53_CLINVAR}",
        "unknown evidence id(s): pmid:12345678", "inline citation to a record not in the store: pmid:11111111",
    }
    # four of them exist in the stages this one may read and are noted as out of scope; the stage-6 plant is
    # simply unknown here — a later stage's evidence is not an input
    assert v["counts"]["citations_out_of_scope"] == 4
    for rid in (FROSST, "pmid:11111111", TP53_VEP, TP53_CLINVAR):
        assert f"CFTR:hom: out of scope: {rid} exists in the run's evidence but was neither in the bundle nor returned by a tool " \
               "in this conversation; rejected" in m["notes"]
    assert not any("12345678" in n and "out of scope" in n for n in m["notes"])
    assert "06_medicine" not in json.dumps(m["inputs"]) and "06_medicine" not in m["params"]["evidence_stages"]
    md = (run_dir / "05_reason" / "evidence_chain.md").read_text()
    assert "planted by stage 6" not in md and "left by an earlier run" not in md and "12345678" not in md and "11111111" not in md
    # the same paper, retrieved here, is citable — the scope is what this conversation returned, not the run
    fake = fake_client([answer], turns=[ac.FakeTurn([("get_paper", {"pmid": "7647779"})])])
    m = read(rr.run_reason(run_dir, 1, fake, "fake-model", "low", False, http=stub_http()))
    chain = EvidenceChain.model_validate(read(run_dir / "05_reason" / "chains" / "CFTR:hom.json"))
    assert [c.code for c in chain.variants[0].criteria] == ["PM2", "PM4", "PP5", "PP4", "PS3"] and chain.literature == [FROSST]
    assert m["params"]["tools_used"]["CFTR:hom"]["papers"] == [FROSST] and m["counts"]["evidence_records_added"] == 0


def test_missing_stage3_output_is_an_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="stage 3"):
        rr.run_reason(tmp_path / "nowhere", dry_run=True)


def test_every_file_the_bundle_reads_is_a_checksummed_input(run_dir: Path):
    """03_filter/shortlist.tsv.gz and 04_rank/evidence both change what the model sees,
    so they are manifest inputs whenever they exist."""
    import gzip
    m = read(rr.run_reason(run_dir, 1, dry_run=True))
    assert set(m["inputs"]) == {"candidates", "retrieve_evidence_index", "rank_joined", "rank_evidence_index"}
    bundle_before = (run_dir / "05_reason" / "bundles" / "CFTR:hom.json").read_bytes()
    shortlist = run_dir / "03_filter" / "shortlist.tsv.gz"
    with gzip.open(shortlist, "wt") as f:
        f.write("chrom\tpos\tref\talt\tcadd_phred\n7\t117559590\tATCT\tA\t99.9\n")
    m = read(rr.run_reason(run_dir, 1, dry_run=True))
    assert m["inputs"]["filter_shortlist"]["sha256"] and m["inputs"]["filter_shortlist"]["path"] == str(shortlist)
    assert (run_dir / "05_reason" / "bundles" / "CFTR:hom.json").read_bytes() != bundle_before
    assert "CADD 99.9" in read(run_dir / "05_reason" / "bundles" / "CFTR:hom.json")["text"]


def test_a_run_replaces_earlier_outputs_but_keeps_the_evidence_store(run_dir: Path):
    """--top 2, then a refusing model at --top 1, then a good --top 1: the directory
    holds exactly what the last manifest claims, plus the papers fetched by any run."""
    rr.run_reason(run_dir, 2, fake_client(), "fake-model", "low", False, http=stub_http())
    out = run_dir / "05_reason"
    assert (out / "chains" / "TP53:het_single.json").exists() and read(out / "evidence" / "index.json")

    class Refusing:
        def run(self, request: ac.AgentRequest) -> ac.AgentResult:
            raise ac.AgentError("final answer not usable: the model refused")
    with pytest.raises(ac.AgentError):
        rr.run_reason(run_dir, 1, Refusing(), "fake-model", "low", False, http=stub_http())
    assert sorted(p.name for p in (out / "transcripts").iterdir()) == ["CFTR:hom.failed.json"]
    assert not (out / "chains").exists() and not (out / "manifest.json").exists() and not (out / "evidence_chain.md").exists()
    assert sorted(p.name for p in (out / "bundles").iterdir()) == ["CFTR:hom.json"]

    m = read(rr.run_reason(run_dir, 1, fake_client([cftr_bundle_only_answer()], turns=[]), "fake-model", "low", False, http=stub_http()))
    assert m["params"]["candidates"] == ["CFTR:hom"]
    for d in ("bundles", "chains", "validation", "transcripts"):
        assert sorted(p.name for p in (out / d).iterdir()) == ["CFTR:hom.json"], d
    assert sorted(p.name for p in (out / "prompts").iterdir()) == ["CFTR:hom.json", "CFTR:hom.md"]
    assert (out / "evidence_chain.md").read_text().count("# Evidence chain — ") == 1
    # the store is a store: the first run's papers are still there, and this run added none
    assert sum(1 for r in EvidenceStore(out / "evidence").iter("pmid")) == 6
    assert m["counts"]["evidence_records"] == 7 and m["counts"]["evidence_records_added"] == 0
    # a dry run is a run too
    rr.run_reason(run_dir, 1, dry_run=True)
    assert not (out / "chains").exists() and not (out / "evidence_chain.md").exists() and (out / "evidence" / "index.json").exists()


def test_no_candidate_means_no_client_and_says_so(run_dir: Path, monkeypatch: pytest.MonkeyPatch):
    (run_dir / "03_filter" / "candidates.json").write_text('{"candidates": []}\n')

    def never(**kw: Any) -> None:
        raise AssertionError("a client was built with nothing to reason about")
    monkeypatch.setattr(ac, "AnthropicClient", never)
    m = read(rr.run_reason(run_dir, 3, None, "claude-opus-5", "high", False, http=stub_http()))
    assert m["counts"]["candidates_selected"] == 0 and m["counts"]["chains_written"] == 0
    assert "no candidate selected: stage 3 listed none; no model was called." in m["notes"]
    assert (run_dir / "05_reason" / "evidence_chain.md").read_text().startswith("# Evidence chains\n\n0 candidate(s)")


def test_hpo_comes_from_the_case_file_or_an_explicit_list_before_stage_4(run_dir: Path, tmp_path: Path):
    case = tmp_path / "case.yaml"
    case.write_text("proband_id: DEMO\nvcf: demo.vcf.gz\nhpo: [HP:0012236]\n")
    m = read(rr.run_reason(run_dir, 1, dry_run=True, case_path=case))
    assert m["params"]["hpo"] == ["HP:0012236"] and m["params"]["hpo_source"].startswith("case file")
    assert "case HPO: HP:0012236\n" in read(run_dir / "05_reason" / "bundles" / "CFTR:hom.json")["text"]
    m = read(rr.run_reason(run_dir, 1, dry_run=True, case_hpo=["HP:0006528"]))
    assert m["params"]["hpo"] == ["HP:0006528"] and m["params"]["hpo_source"] == "argument"
    (run_dir / "04_rank" / "joined.json").unlink()
    m = read(rr.run_reason(run_dir, 1, dry_run=True))
    assert m["params"]["hpo"] == [] and m["params"]["hpo_source"] == "none" and "rank_joined" not in m["inputs"]
    with pytest.raises(ValueError, match="HPO"):
        rr.run_reason(run_dir, 1, dry_run=True, case_hpo=["HP:12"])


# ------------------------------------------------------------------ after the model

def test_align_chain_pins_identity_and_covers_every_variant(run_dir: Path):
    bundle = build_bundle("CFTR:hom", run_dir, HPO)
    answer = cftr_answer()
    answer["candidate_id"] = "CFTR"
    answer["variants"][0]["key"] = "chr7-117559590-ATCT-A"                       # respelled, still this variant
    answer["variants"].append({"key": TP53, "summary": "not a variant of this candidate",       # foreign, with criteria
                               "criteria": [criterion("PS3", "strong", [TP53_VEP]), criterion("PM1", "moderate", [TP53_VEP])]})
    answer["variants"].append({"key": "7:117559592:CTT:-", "criteria": [criterion("PM4", "moderate", [CFTR_VEP])],
                               "summary": "a spelling the bundle does not use"})
    answer["variants"].append({"key": CFTR, "criteria": [], "summary": "a second entry for the real key"})
    chain, notes, rejections = rr.align_chain(EvidenceChain.model_validate(answer), bundle)
    assert chain.candidate_id == "CFTR:hom" and [v.key for v in chain.variants] == [CFTR]
    assert notes == [
        "candidate_id: the model wrote 'CFTR'; replaced by the bundle's 'CFTR:hom'",
        f"variants: key 'chr7-117559590-ATCT-A' respelled by the model; matched to the bundle's '{CFTR}'",
    ]
    assert [(r.path, r.reason) for r in rejections] == [
        ("variants[1]", f"key '{TP53}' is not a variant of this candidate (CFTR:hom: {CFTR}); the entry and its 2 criteria were dropped"),
        ("variants[2]", f"key '7:117559592:CTT:-' is not a variant of this candidate (CFTR:hom: {CFTR}); the entry and its 1 criteria were dropped"),
        ("variants[3]", f"a second entry for '{CFTR}'; the first is kept"),
    ]
    empty = EvidenceChain.model_validate(dict(cftr_answer(), variants=[]))
    chain, notes, rejections = rr.align_chain(empty, bundle)
    assert [v.key for v in chain.variants] == [CFTR] and chain.variants[0].criteria == [] and rejections == []
    assert notes == [f"variants: the model returned no entry for '{CFTR}'; an empty entry was added (no criteria → vus)"]
    assert rr.classify(chain).variants[0].classification == "vus"


def test_a_foreign_variant_entry_never_reaches_the_chain_or_the_report(run_dir: Path):
    answer = cftr_bundle_only_answer()
    answer["variants"].append({"key": TP53, "summary": "TP53 smuggled into the CFTR chain",
                               "criteria": [criterion("PS3", "strong", [TP53_VEP]), criterion("PM1", "moderate", [TP53_VEP]),
                                            criterion("PM2", "moderate", [TP53_GNOMAD]), criterion("PP3", "supporting", [TP53_VEP])]})
    m = read(rr.run_reason(run_dir, 1, fake_client([answer], turns=[]), "fake-model", "low", False, http=stub_http()))
    chain = read(run_dir / "05_reason" / "chains" / "CFTR:hom.json")
    assert [v["key"] for v in chain["variants"]] == [CFTR] and m["counts"]["classifications"] == {"CFTR:hom": {CFTR: "vus"}}
    assert m["counts"]["validation"]["CFTR:hom"]["items_dropped"] == 1 and m["counts"]["rejections"] == 1
    assert f"CFTR:hom: rejected variants[1]: key '{TP53}' is not a variant of this candidate (CFTR:hom: {CFTR}); " \
           "the entry and its 4 criteria were dropped" in m["notes"]
    md = (run_dir / "05_reason" / "evidence_chain.md").read_text()
    assert md.count("\n| 1 | CFTR:hom |") == 1 and TP53 not in md and "smuggled" not in md


def test_bare_accessions_in_prose_must_be_carried_by_a_citable_record(run_dir: Path):
    """Every accession the reviewer's probe smuggled past the validator, plus the same
    shapes where a citable record carries them."""
    answer = cftr_bundle_only_answer()
    answer["variants"][0]["summary"] = ("Reported as VCV999999999 and SCV000012345 (see doi:10.1038/ng9999-999, PMC9999999, "
                                        "Smith et al. 2019, rs199826652, NCT99999999, pubmed 87654321, "
                                        "https://europepmc.org/article/MED/11111111). Listed as VCV000007105 — "
                                        f"rs113993960 [{CFTR_VEP}] — at https://www.ncbi.nlm.nih.gov/clinvar/variation/7105/.")
    answer["variants"][0]["criteria"][1]["justification"] = "in-frame deletion [VCV7105]; cf. [VCV999999999] and RCV000000001."
    answer["phase_statement"] = "Phase per gnomAD rs113993960; a PubMed ID 12345 claim and https://pubmed.ncbi.nlm.nih.gov/7647779/."
    answer["limits"] = ["No assay; DOI 10.1038/ng0595-111 covers MTHFR, not CFTR."]
    fake = fake_client([answer], turns=[ac.FakeTurn([("get_paper", {"pmid": "7647779"})])])
    m = read(rr.run_reason(run_dir, 1, fake, "fake-model", "low", False, http=stub_http()))
    chain = EvidenceChain.model_validate(read(run_dir / "05_reason" / "chains" / "CFTR:hom.json"))
    gone = "[citation removed: no such record]"
    assert chain.variants[0].summary == (f"Reported as {gone} and {gone} (see {gone}, {gone}, Smith et al. 2019, {gone}, {gone}, {gone}, "
                                         f"{gone}). Listed as VCV000007105 — rs113993960 [{CFTR_VEP}] — at "
                                         "https://www.ncbi.nlm.nih.gov/clinvar/variation/7105/.")
    assert chain.variants[0].criteria[1].justification == f"in-frame deletion [VCV7105]; cf. {gone} and {gone}."
    assert chain.phase_statement == f"Phase per gnomAD rs113993960; a {gone} claim and https://pubmed.ncbi.nlm.nih.gov/7647779/."
    assert chain.limits == ["No assay; DOI 10.1038/ng0595-111 covers MTHFR, not CFTR."]  # the Frosst paper was retrieved here
    assert [c.code for c in chain.variants[0].criteria] == ["PM2", "PM4", "PP5", "PP4"]  # redaction never drops a criterion
    v = read(run_dir / "05_reason" / "validation" / "CFTR:hom.json")
    reasons = [(r["path"], r["reason"]) for r in v["rejections"]]
    assert reasons[:2] == [("variants[0].summary", "bare accession not carried by any citable record: VCV999999999"),
                           ("variants[0].summary", "bare accession not carried by any citable record: SCV000012345")]
    assert ("variants[0].summary", "bare accession not carried by any citable record: https://europepmc.org/article/MED/11111111") in reasons
    assert ("variants[0].criteria[1].justification", "bare accession not carried by any citable record: RCV000000001") in reasons
    assert ("phase_statement", "bare accession not carried by any citable record: PubMed ID 12345") in reasons
    assert v["counts"]["redactions"] == 11 and m["counts"]["rejections"] == 11  # 8 in the summary, 2 in a justification, 1 in the phase
    md = (run_dir / "05_reason" / "evidence_chain.md").read_text()
    for token in ("VCV999999999", "SCV000012345", "ng9999-999", "PMC9999999", "rs199826652", "NCT99999999", "87654321", "11111111", "12345 "):
        assert token not in md, token
    assert "rs113993960" in md and "ng0595-111" in md and "VCV000007105" in md and "Smith et al. 2019" in md


def test_stage_checks_resolve_accessions_against_the_candidates_records_and_need_a_paper_for_ps3(run_dir: Path):
    store = EvidenceStore(run_dir / "02_retrieve" / "evidence")
    resolver = AccessionResolver([store.get(CFTR_VEP), store.get(CFTR_CLINVAR), store.get(FROSST), store.get("nct:NCT01807923")])
    text = ("VCV000007105 VCV7105 vcv000007105 rs113993960 RS113993960 NCT01807923 https://clinicaltrials.gov/study/NCT01807923 "
            "doi:10.1038/ng0595-111 10.1038/ng0595-111. PubMed 7647779 europepmc.org/article/MED/7647779 PMC0000000 "
            "clinvar:VCV000007105 HP:0002205 OMIM:219700 years2019 1000 Genomes")
    assert accession_tokens(text) == ["VCV000007105", "VCV7105", "vcv000007105", "rs113993960", "RS113993960", "NCT01807923",
                                      "https://clinicaltrials.gov/study/NCT01807923", "doi:10.1038/ng0595-111", "10.1038/ng0595-111.",
                                      "PubMed 7647779", "europepmc.org/article/MED/7647779", "PMC0000000"]
    chain = EvidenceChain.model_validate(dict(cftr_bundle_only_answer(), phase_statement=text))
    chain.variants[0].criteria.append(EvidenceChain.model_validate(cftr_answer()).variants[0].criteria[0].model_copy(update={"evidence_ids": [CFTR_VEP]}))
    chain.variants[0].criteria.append(EvidenceChain.model_validate(cftr_answer()).variants[0].criteria[0].model_copy(update={"code": "BS3", "evidence_ids": []}))
    out = stage_checks(chain, resolver)
    assert out.chain.phase_statement == text.replace("PMC0000000", "[citation removed: no such record]")
    assert [(r.path, r.reason) for r in out.redacted] == [("phase_statement", "bare accession not carried by any citable record: PMC0000000")]
    assert [(r.path, r.reason) for r in out.dropped] == [
        ("variants[0].criteria[4]", "PS3 asserts a functional study but cites no paper record (pmid:)"),
        ("variants[0].criteria[5]", "BS3 asserts a functional study but cites no paper record (pmid:)"),
    ]
    assert [c.code for c in out.chain.variants[0].criteria] == ["PM2", "PM4", "PP5", "PP4"]


def test_an_unverifiable_frequency_criterion_is_not_met_and_a_raised_strength_is_capped(run_dir: Path):
    """TP53 with its gnomAD and VEP records removed from the run: nothing in the store
    can check PM2, so the model's "met" does not count (a dispute with no record); a
    PP3 the model wrote at very_strong is lowered to the code's cap. The verdict is
    computed from what survives — with PM2 believed, PM2 + PP3(strong) + PP5 would
    have been likely pathogenic."""
    store = EvidenceStore(run_dir / "02_retrieve" / "evidence")
    for rid in (TP53_GNOMAD, TP53_VEP):
        store.path_for(rid).unlink()
    store.write_index()
    answer = tp53_answer()
    answer["variants"][0]["criteria"] = [
        criterion("PM2", "moderate", [TP53_CLINVAR], just="absent from gnomAD"),
        criterion("PP3", "very_strong", [TP53_CLINVAR], just="all predictors agree"),
        criterion("PP5", "supporting", [TP53_CLINVAR], just="ClinVar Pathogenic, expert panel"),
    ]
    fake = fake_client([answer], turns=[])
    m = read(rr.run_reason(run_dir, 2, fake, "fake-model", "low", False, http=stub_http()))
    bundle = read(run_dir / "05_reason" / "bundles" / "TP53:het_single.json")
    assert bundle["variants"][0]["missing_ids"] == [TP53_GNOMAD, TP53_VEP] and "listed but not in the store" in bundle["text"]
    chain = EvidenceChain.model_validate(read(run_dir / "05_reason" / "chains" / "TP53:het_single.json"))
    pm2, pp3, pp5 = chain.variants[0].criteria
    assert pm2.met is False and pm2.justification.startswith(
        f"[UNVERIFIED — no gnomAD or VEP record for {TP53} in the store; PM2 cannot be checked → not met; the model said met]")
    assert pp3.strength == "strong" and pp3.met is False
    assert pp3.justification.startswith(f"[UNVERIFIED — no VEP record for {TP53} in the store; PP3 cannot be checked → not met; the model said met] [STRENGTH CAPPED — PP3 at most strong")
    assert pp5.met is False and pp5.justification.startswith("[RETIRED — PP5") and chain.variants[0].classification == "vus"
    v = read(run_dir / "05_reason" / "validation" / "TP53:het_single.json")
    assert v["counts"]["frequency_unverified"] == 1 and v["counts"]["frequency_disputed"] == 1 and v["counts"]["strength_capped"] == 1
    assert v["counts"]["computational_unverified"] == 1 and v["counts"]["computational_disputed"] == 1 and v["counts"]["retired_not_counted"] == 1
    assert v["disputes"][0] == {"path": "variants[0].criteria[0]", "code": "PM2", "record_id": None, "claimed_met": True,
                                "recomputed_met": False, "af": None,
                                "reason": f"no gnomAD or VEP record for {TP53} in the store; PM2 cannot be checked → not met; the model said met"}
    assert v["disputes"][1]["code"] == "PP3" and v["disputes"][1]["record_id"] is None
    assert m["counts"]["classifications"]["TP53:het_single"] == {TP53: "vus"}
    assert any(n.startswith(f"TP53:het_single: disputed variants[0].criteria[0]: no gnomAD or VEP record for {TP53}") for n in m["notes"])
    assert any(n.startswith("TP53:het_single: variants[0].criteria[1].strength: PP3 at most strong") for n in m["notes"])
    md = (run_dir / "05_reason" / "evidence_chain.md").read_text()
    assert "- **PM2** · moderate · not met — [UNVERIFIED — no gnomAD or VEP record" in md
    assert "- **PP3** · strong · not met — [UNVERIFIED — no VEP record" in md


def test_the_validator_is_told_the_candidates_keys(run_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Alignment pins the keys first; the validator is still given them, so its
    frequency lookups and its own key check rest on the candidate's variants."""
    seen: list[list[str] | None] = []
    real = rr.validate

    def spy(obj: Any, index: EvidenceIndex, **kw: Any) -> Any:
        seen.append(kw.get("keys"))
        return real(obj, index, **kw)
    monkeypatch.setattr(rr, "validate", spy)
    rr.run_reason(run_dir, 2, fake_client([cftr_bundle_only_answer(), tp53_answer()], turns=[]), "fake-model", "low", False, http=stub_http())
    assert seen == [[CFTR], [TP53]]


def test_an_omitted_variant_is_classified_vus_and_noted_in_the_manifest(run_dir: Path):
    fake = fake_client([dict(cftr_answer(), variants=[])], turns=[])
    m = read(rr.run_reason(run_dir, 1, fake, "fake-model", "low", False, http=stub_http()))
    assert m["counts"]["classifications"] == {"CFTR:hom": {CFTR: "vus"}}
    assert f"CFTR:hom: variants: the model returned no entry for '{CFTR}'; an empty entry was added (no criteria → vus)" in m["notes"]
    md = (run_dir / "05_reason" / "evidence_chain.md").read_text()
    assert f"## Variant {CFTR} — vus" in md and "- none survived validation" in md


# ------------------------------------------------------------------------ failures

def test_a_tool_service_failure_aborts_the_run_and_leaves_the_transcript(run_dir: Path):
    broken = dict(fixture("fetch_42269400"), status=502, body=None, text="bad gateway")
    fake = fake_client(turns=[ac.FakeTurn([("get_record", {"record_id": CFTR_VEP}), ("get_paper", {"pmid": "42269400"})])])
    with pytest.raises(ac.ToolFailure, match="tool 'get_paper' failed: HTTP 502 from www.ebi.ac.uk"):
        rr.run_reason(run_dir, 1, fake, "fake-model", "low", False, http=StubHttp(broken))
    out = run_dir / "05_reason"
    failed = read(out / "transcripts" / "CFTR:hom.failed.json")
    assert failed["candidate_id"] == "CFTR:hom" and failed["transcript"][0]["tool_calls"][0]["name"] == "get_record"
    assert failed["tools"]["records_read"] == [CFTR_VEP]
    assert not (out / "chains").exists() and not (out / "manifest.json").exists() and not (out / "evidence_chain.md").exists()
    assert (out / "bundles" / "CFTR:hom.json").exists() and (out / "prompts" / "CFTR:hom.json").exists()


def test_a_model_failure_aborts_the_run(run_dir: Path):
    class Refusing:
        def run(self, request: ac.AgentRequest) -> ac.AgentResult:
            raise ac.AgentError("final answer not usable: the model refused", request_id="req_x")
    with pytest.raises(ac.AgentError, match="refused"):
        rr.run_reason(run_dir, 1, Refusing(), "fake-model", "low", False, http=stub_http())
    assert read(run_dir / "05_reason" / "transcripts" / "CFTR:hom.failed.json")["request_id"] == "req_x"


# --------------------------------------------------------------------------- tools

def test_tools_refuse_bad_arguments_before_any_request(run_dir: Path):
    http = StubHttp()
    tools = ReasonTools(EvidenceIndex.from_run(run_dir), EvidenceStore(run_dir / "05_reason" / "evidence"),
                        rr._literature(http, None, False))
    with pytest.raises(ValueError, match="genomic coordinate"):
        tools.search_literature({"query": f"CFTR {CFTR}", "max_results": None})
    with pytest.raises(ValueError, match="genomic coordinate"):
        tools.search_literature({"query": "NC_000007.14:g.117559592_117559594del", "max_results": 3})
    with pytest.raises(ValueError, match="between 1 and 25"):
        tools.search_literature({"query": SEARCH_QUERY, "max_results": 26})
    with pytest.raises(ValueError, match="empty"):
        tools.search_literature({"query": "  ", "max_results": None})
    with pytest.raises(ValueError, match="not a PMID"):
        tools.get_paper({"pmid": "10.1038/ng0595-111"})
    with pytest.raises(KeyError):
        tools.get_record({"record_id": FAKE_CLINVAR})
    # a bare position — no chromosome, no g. — is a coordinate too, in any of the spellings a model might try
    for bad in ("CFTR 117559590 deletion", "7:117,559,590", "chr7 117559590", "CFTR position 117559590-117559593",
                "NC_000007.14 117559590", "MTHFR 11796321", "TITLE_ABS:\"100,000 Genomes Project\" AND CFTR"):
        with pytest.raises(ValueError, match="genomic coordinate|five or more digits"):
            check_query(bad)
    assert http.calls == []
    for ok in ('TITLE_ABS:"CFTR" AND TITLE_ABS:"p.Phe508del"', "rs1801133 MTHFR", "1-100 patients with c.1521_1523del", "HP:0006528",
               "DMD c.10086C>T", "NM_000492.4 CFTR", "ENSG00000001626", "OMIM 219700", "PMID: 7647779", "EXT_ID:7647779",
               "NCT01807923 ivacaftor", "VCV000007105", "1,000 Genomes", "m.8993T>G NARP", "PMC1234567", "MONDO:0009061"):
        check_query(ok)
    assert parse_pmid("pmid:7647779") == parse_pmid("PMID 7647779") == parse_pmid(" 07647779 ") == "7647779"
    # a paper already in an earlier stage's store is served from there — no request, no duplicate record
    served = tools.get_paper({"pmid": "7647779"})
    assert served["record_id"] == FROSST and served["abstract"].startswith("Hyperhomocysteinaemia") and http.calls == []
    assert tools.log.papers == [FROSST] and tools.log.written == [] and not (run_dir / "05_reason" / "evidence" / "pmid").exists()
    assert tools.log.as_dict() == {"searches": [], "papers": [FROSST], "records_read": []}
    assert [t.param()["name"] for t in tools.specs()] == ["get_record", "search_literature", "get_paper"]
    assert tools.specs()[1].param()["input_schema"]["properties"]["max_results"]["anyOf"] == [{"type": "integer"}, {"type": "null"}]


def test_get_record_serves_only_the_candidates_records_and_the_papers_the_tools_returned(run_dir: Path):
    bundle = build_bundle("CFTR:hom", run_dir, HPO)
    index = EvidenceIndex.from_run(run_dir)
    tools = ReasonTools(index, EvidenceStore(run_dir / "05_reason" / "evidence"), rr._literature(stub_http(), None, False),
                        citable=bundle.record_ids)
    assert tools.get_record({"record_id": CFTR_GNOMAD})["record_id"] == CFTR_GNOMAD
    assert tools.get_record({"record_id": "exomiser:CFTR"})["source"] == "exomiser"
    for foreign in (TP53_VEP, TP53_CLINVAR, "nct:NCT01807923", FROSST):  # all real, none this candidate's
        with pytest.raises(KeyError, match="bundle or among this conversation"):
            tools.get_record({"record_id": foreign})
    assert index.get(TP53_VEP) is not None
    tools.get_paper({"pmid": "7647779"})  # retrieved here → readable and citable from now on
    assert tools.get_record({"record_id": FROSST})["record_id"] == FROSST
    assert tools.citable_ids() == sorted(bundle.record_ids + [FROSST])
    assert tools.log.records_read == [CFTR_GNOMAD, "exomiser:CFTR", FROSST] and tools.log.ids() == [FROSST, CFTR_GNOMAD, "exomiser:CFTR"]


# ------------------------------------------------------------------------------ cli

def test_cli_dry_run_prints_counts_and_no_variant(run_dir: Path):
    result = CliRunner().invoke(main, ["reason", "--run", str(run_dir), "--dry-run", "--top", "1"])
    assert result.exit_code == 0, result.output
    assert "reason · " in result.output and "DRY RUN" in result.output
    assert "candidates: 1 of 3 · bundles and prompts written · no model called" in result.output
    assert "117559590" not in result.output and "CFTR" not in result.output
    assert result.output.rstrip().endswith(f"manifest: {run_dir / '05_reason' / 'manifest.json'}")


def test_cli_runs_the_stage_and_names_candidates_only_when_asked(run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ac, "AnthropicClient", lambda **kw: fake_client([cftr_bundle_only_answer(), tp53_answer()], turns=[]))
    args = ["reason", "--run", str(run_dir), "--top", "2", "--model", "fake-model", "--effort", "low",
            "--cache", str(tmp_path / "cache"), "--offline"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "candidate 1/2: vus · 0 rejected · 0 disputed" in result.output
    assert "candidates: 2 of 3 · chains: 2 · rejections: 0 · records added: 0 · tokens in/out: 0/0 · api calls: 0" in result.output
    assert "candidate 1: vus" in result.output and "candidate 2: vus" in result.output
    assert "CFTR" not in result.output and "117559590" not in result.output
    m = read(run_dir / "05_reason" / "manifest.json")
    assert m["params"]["offline"] is True and m["params"]["cache_root"] == str(tmp_path / "cache")
    assert not (tmp_path / "cache").exists() or not any((tmp_path / "cache").iterdir())  # nothing was fetched

    monkeypatch.setattr(ac, "AnthropicClient", lambda **kw: fake_client([cftr_bundle_only_answer()], turns=[]))
    result = CliRunner().invoke(main, args[:5] + ["--top", "1", "-v"])
    assert result.exit_code == 0, result.output
    assert "CFTR:hom: vus" in result.output


def test_the_real_sdk_without_credentials_or_a_server_is_one_agent_error(run_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """The stage builds ``anthropic.Anthropic()`` zero-arg. Here that constructor is
    pointed at a port nothing listens on and stripped of credentials, so the SDK either
    refuses to send (no authentication resolved — a TypeError inside the SDK) or is
    refused by the socket; the client maps both to one AgentError, the stage leaves
    the failed transcript and no chain, and the CLI reports it and exits 1. Nothing is
    sent anywhere: the only address is 127.0.0.1:9."""
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    real = anthropic.Anthropic
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: real(base_url="http://127.0.0.1:9", max_retries=0, timeout=2.0, **kw))
    with pytest.raises(ac.AgentError) as e:
        rr.run_reason(run_dir, 1, None, "claude-opus-5", "high", False, http=stub_http())
    assert str(e.value).startswith(("the Anthropic SDK refused the request: TypeError: ", "could not reach the Anthropic API"))
    out = run_dir / "05_reason"
    assert not (out / "chains").exists() and not (out / "manifest.json").exists()
    assert read(out / "transcripts" / "CFTR:hom.failed.json")["error"] == str(e.value)
    result = CliRunner().invoke(main, ["reason", "--run", str(run_dir), "--top", "1", "--offline"])
    assert result.exit_code == 1 and result.output.count("FAILED: ") == 1
    assert "117559590" not in result.output and "CFTR" not in result.output

    def no_client(**kw: Any) -> None:
        raise TypeError("Could not resolve authentication method.")
    monkeypatch.setattr(ac, "AnthropicClient", no_client)
    with pytest.raises(ac.AgentError, match="could not build the Anthropic client"):
        rr.run_reason(run_dir, 1, None, "claude-opus-5", "high", False, http=stub_http())

    class Broken:
        def run(self, request: ac.AgentRequest) -> ac.AgentResult:
            raise TypeError("unrelated programming error")
    with pytest.raises(TypeError, match="unrelated"):  # a real bug still surfaces as itself
        rr.run_reason(run_dir, 1, Broken(), "fake-model", "low", False, http=stub_http())


def test_cli_reports_an_agent_failure_and_exits_1(run_dir: Path, monkeypatch: pytest.MonkeyPatch):
    class Failing:
        def run(self, request: ac.AgentRequest) -> ac.AgentResult:
            raise ac.AgentError("Anthropic API rate limit (retry-after 30s): slow down")
    monkeypatch.setattr(ac, "AnthropicClient", lambda **kw: Failing())
    result = CliRunner().invoke(main, ["reason", "--run", str(run_dir), "--top", "1", "--offline"])
    assert result.exit_code == 1 and "FAILED: Anthropic API rate limit" in result.output


# ------------------------------------------------------------------------------ live

@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ENGINE_LIVE_TESTS"), reason="set ENGINE_LIVE_TESTS=1 to hit Europe PMC")
def test_live_literature_tools_write_citable_records(tmp_path: Path):
    """Public queries only: a CFTR/F508del search and the Frosst 1995 MTHFR paper."""
    http = Http(HttpCache(tmp_path / "cache"), limiter=RateLimiter({"www.ebi.ac.uk": 3.0}))
    store = EvidenceStore(tmp_path / "evidence")
    tools = ReasonTools(EvidenceIndex([store]), store, rr._literature(http, None, False))
    found = tools.search_literature({"query": SEARCH_QUERY, "max_results": 3})
    assert found["returned"] == 3 and found["hit_count"] >= 1000
    assert all(p["record_id"].startswith("pmid:") and store.exists(p["record_id"]) for p in found["papers"])
    paper = tools.get_paper({"pmid": "7647779"})
    assert "methylenetetrahydrofolate reductase" in paper["title"].lower() and paper["year"] == "1995"
    rec = store.get(FROSST)
    assert rec is not None and rec.source_version.startswith("Europe PMC REST") and rec.url == "https://europepmc.org/article/MED/7647779"
    assert [r.record_id for r in store.iter("pmid-search")] == found["search_record"].split() and http.live_requests == 2
