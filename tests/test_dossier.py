"""The dossier step over the public CFTR case — ``tests/fixtures/public_case/``.

The run directory is the recorded stage-2 output (the twelve public alleles), stage 3
run over it for real, and the recorded stage-5 chain for ``CFTR:comphet`` copied into
``05_reason/chains/`` — the same files ``scripts/run_public_case.sh`` gives stage 6.
UniProt and Europe PMC are the real retrievers over a ``StubHttp`` that serves the
recorded fixtures (``tests/fixtures/uniprot``, ``tests/fixtures/dossier``) for the
exact requests they make. The model is ``FakeClient``, scripted to answer with a
dossier that writes a wrong residue, mentions an accession no record carries, cites a
paper it never fetched and names a variant of another candidate — so what the tests
check is the step's plumbing and its gates, never a model's judgement.

Nothing here touches the network; one live test runs under ``ENGINE_LIVE_TESTS``.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from engine.agents import client as ac
from engine.cli import main
from engine.dossier import run as dr
from engine.dossier.checks import DossierResolver, accession_tokens
from engine.dossier.prompts import PROMPT_VERSION, SYSTEM_PROMPT, instructions_sha256, prompt_sha256
from engine.dossier.render import render_dossier_html
from engine.dossier.schema import GeneDossier
from engine.dossier.tools import DISEASES_SEARCHED, QUERY_TEMPLATES, DossierRetrievers, fixed_queries
from engine.dossier.views import dossier_view
from engine.retrieve.http import HttpCache, HttpError, RateLimiter, Response
from engine.retrieve.literature import LiteratureRetriever
from engine.retrieve.store import EvidenceStore
from engine.retrieve.uniprot import UniprotRetriever

FIXTURES = Path(__file__).parent / "fixtures"
PUBLIC = FIXTURES / "public_case"

CID = "CFTR:comphet"
F508DEL = "7:117559590:ATCT:A"
G542X = "7:117587778:G:T"
R175H = "17:7675088:C:T"
ACC = "P13569"
UNIPROT = f"uniprot:{ACC}"
VEP_F508 = f"vep:{F508DEL}"
VEP_G542 = f"vep:{G542X}"
CLINVAR_F508 = "clinvar:VCV000007105"
PAPER = "pmid:42687801"      # first result of the recorded CFTR / cystic fibrosis search
OTHER_PAPER = "pmid:42589573"  # first result of the recorded mechanism search
FAKE_PMID = "pmid:99999999"
DOSSIER_DIR = "05_reason/dossier"


# ------------------------------------------------------------------------ fixtures

class StubHttp:
    """Serves the recorded fixtures of every source keyed by the exact request; refuses
    anything else — so a request the step invents (a coordinate in a query, an
    unrecorded accession) fails the test rather than reaching a service."""

    def __init__(self, *dirs: str, offline: bool = False):
        self.limiter = RateLimiter(default_per_second=0)
        self.offline = offline
        self.calls: list[tuple[str, str, Any, Any]] = []
        self._by_key: dict[str, dict] = {}
        for d in dirs:
            for p in sorted((FIXTURES / d).glob("*.json")):
                fx = json.loads(p.read_text())
                if isinstance(fx, dict) and "request" in fx:  # the directory also holds the scripted answer
                    self.add(fx)

    def add(self, fx: dict) -> None:
        r = fx["request"]
        self._by_key[HttpCache.key(r["method"], r["url"], r.get("params"), r.get("body"))] = fx

    def request(self, method: str, url: str, *, params=None, json_body=None, cache_404: bool = False, **kw) -> Response:
        self.calls.append((method, url, params, json_body))
        fx = self._by_key.get(HttpCache.key(method, url, params, json_body))
        if fx is None:
            raise AssertionError(f"no fixture for {method} {url} {params}")
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

    def sent_text(self) -> str:
        return json.dumps(self.calls, default=str)


def stub_http(**kw: Any) -> StubHttp:
    return StubHttp("uniprot", "dossier", **kw)


def retrievers(http: StubHttp | None = None) -> DossierRetrievers:
    """The real retrievers over the stub: page size 8, as the searches were recorded."""
    http = http or stub_http()
    return DossierRetrievers(uniprot=UniprotRetriever(http), literature=LiteratureRetriever(http, page_size=8))


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """The public case as stage 3 and stage 5 leave it: the recorded stage-2 store and
    annotated table, stage 3 run for real, the recorded chain in ``05_reason/chains``."""
    from engine.filter.rules import ANNOTATED_COLUMNS
    from engine.filter.run import run_filter

    run = tmp_path / "run"
    stage = run / "02_retrieve"
    shutil.copytree(PUBLIC / "02_retrieve" / "evidence", stage / "evidence")
    rows = json.loads((PUBLIC / "02_retrieve" / "annotated_rows.json").read_text())
    with gzip.GzipFile(stage / "variants.annotated.tsv.gz", "wb", mtime=0) as raw, \
            io.TextIOWrapper(raw, encoding="utf-8") as out:
        out.write("\t".join(ANNOTATED_COLUMNS) + "\n")
        for r in rows:
            out.write("\t".join(r[c] for c in ANNOTATED_COLUMNS) + "\n")
    run_filter(run)
    chains = run / "05_reason" / "chains"
    chains.mkdir(parents=True)
    shutil.copy(PUBLIC / "05_reason" / "chain_CFTR_comphet.json", chains / f"{CID}.json")
    return run


def with_evidence_chain(run: Path, text: str = "# Evidence chains\n\none candidate.\n") -> Path:
    path = run / "05_reason" / "evidence_chain.md"
    path.write_text(text)
    return path


def read(path: Path) -> Any:
    return json.loads(path.read_text())


# ------------------------------------------------------------------------- answers

def claim(statement: str, ids: list[str]) -> dict[str, Any]:
    return {"statement": statement, "evidence_ids": ids}


def cftr_answer() -> dict[str, Any]:
    """The scripted answer in ``tests/fixtures/dossier/answer_CFTR_comphet.json``: what
    a model that mostly follows the rules — and breaks several — writes. A wrong
    residue (999) for F508del and a remembered region; a bare ``P13569`` (the record
    carries it) beside a bare ``O60566`` (no record does) and a ``PF99999`` the entry
    does not cross-reference; a paper it never fetched; a variant of another
    candidate; and a protein claim resting on nothing from UniProt."""
    return json.loads((FIXTURES / "dossier" / "answer_CFTR_comphet.json").read_text())


SCRIPT = [ac.FakeTurn([("get_record", {"record_id": UNIPROT}), ("get_paper", {"pmid": "42687801"})])]


def fake_client(output: Any = None, turns: list[ac.FakeTurn] | None = None) -> ac.FakeClient:
    return ac.FakeClient(output if output is not None else cftr_answer(), turns=SCRIPT if turns is None else turns,
                         validate_output=False)


def run_step(run: Path, *, client: Any = None, http: StubHttp | None = None, dry_run: bool = False, **kw: Any) -> Path:
    http = http or stub_http()
    return dr.run_dossier(run, kw.pop("candidate_id", None), client if client is not None else fake_client(),
                          "fake-model", "low", dry_run, retrievers=retrievers(http), **kw)


# ------------------------------------------------------------------ fixed searches

def test_the_fixed_queries_name_no_coordinate_and_follow_the_templates():
    queries = fixed_queries("CFTR", ["Cystic fibrosis", "Congenital bilateral absence of the vas deferens", "A", "D4"])
    assert len(queries) == DISEASES_SEARCHED + 2
    assert queries[0] == 'TITLE_ABS:"CFTR" AND TITLE_ABS:"Cystic fibrosis"'
    assert queries[3] == QUERY_TEMPLATES[1].format(gene="CFTR") and queries[4] == QUERY_TEMPLATES[2].format(gene="CFTR")
    assert fixed_queries("CFTR", []) == [QUERY_TEMPLATES[1].format(gene="CFTR"), QUERY_TEMPLATES[2].format(gene="CFTR")]
    for q in queries:
        assert "117559590" not in q and "p.Phe508del" not in q and "7:" not in q
    with pytest.raises(ValueError, match="genomic coordinate"):
        fixed_queries("CFTR", ["chr7:117559590 disease"])


# ------------------------------------------------------------------------ dry run

def test_dry_run_retrieves_writes_the_bundle_and_the_prompt_and_no_dossier(run_dir: Path):
    http = stub_http()
    manifest_path = run_step(run_dir, http=http, dry_run=True)
    out = run_dir / DOSSIER_DIR
    assert sorted(p.name for p in out.iterdir()) == ["bundles", "manifest.json", "prompts"]
    assert sorted(p.name for p in (out / "prompts").iterdir()) == [f"{CID}.json", f"{CID}.md"]
    assert not (out / f"{CID}.json").exists() and not (out / "transcripts").exists()

    # the retrieval is the engine's and happens in a dry run too
    store = EvidenceStore(run_dir / "05_reason" / "evidence")
    assert store.exists(UNIPROT) and store.exists(PAPER)
    assert sorted(d.name for d in (run_dir / "05_reason" / "evidence").iterdir() if d.is_dir()) == ["pmid", "pmid-search", "uniprot"]
    assert store.get(UNIPROT).payload["primaryAccession"] == ACC

    m = read(manifest_path)
    assert m["stage"] == "dossier"
    p, c = m["params"], m["counts"]
    assert p["candidate"] == CID and p["accession"] == ACC and p["dry_run"] is True
    assert p["uniprot_release"] == "UniProt release 2026_03 (02-September-2026)"
    assert p["prompt_version"] == PROMPT_VERSION and p["system_prompt_sha256"] == prompt_sha256()
    assert p["instructions_sha256"] == instructions_sha256() and p["tools"] == ["get_record", "search_literature", "get_paper"]
    assert [q["query"] for q in p["queries"]] == fixed_queries("CFTR", ["Cystic fibrosis", "Congenital bilateral absence of the vas deferens"])
    assert all(q["sent"].endswith("AND SRC:MED") and q["hit_count"] > 0 and q["returned"] == 8 for q in p["queries"])
    assert all(q["search_record"].startswith("pmid-search:") for q in p["queries"])
    assert p["evidence_chain_md"] is None  # stage 5 wrote none in this run
    assert c["features_in_map"] == 39 and c["papers_retrieved"] >= 8 and c["dossiers_written"] == 0
    assert c["natural_variants_at_positions"] == 3  # two at residue 508, one at 542
    assert any("dry run" in n for n in m["notes"])
    assert set(m["inputs"]) >= {"chain", "candidates", "retrieve_evidence_index"}

    # no request carried a coordinate or an HGVS string
    sent = http.sent_text()
    for leak in ("117559590", "117587778", "Phe508del", "1521_1523del", "PUBLIC01"):
        assert leak not in sent, leak


def test_the_bundle_and_the_prompt_carry_the_map_and_the_engine_residues(run_dir: Path):
    run_step(run_dir, dry_run=True)
    out = run_dir / DOSSIER_DIR
    bundle = read(out / "bundles" / f"{CID}.json")
    assert bundle["candidate_id"] == CID and bundle["gene_symbol"] == "CFTR" and bundle["accession"] == ACC
    assert [v["key"] for v in bundle["variants"]] == [F508DEL, G542X]
    f508, g542 = bundle["variants"]
    assert f508["residue"] == 508 and f508["reference_residue"] == "F" and f508["hgvsp"].endswith("p.Phe508del")
    assert f508["region"][:2] == ["Topological domain: Cytoplasmic (359–858)", "Domain: ABC transporter 1 (423–646)"]
    assert any(r.startswith("VAR_000171 F→del") for r in f508["region"])
    assert g542["residue"] == 542 and g542["classification"] == "pathogenic" and g542["points"] == 10  # PP4 unverified: the recorded chain was made without stage 4
    assert len(bundle["feature_map"]) == 39
    assert bundle["uniprot"]["record_id"] == UNIPROT
    assert UNIPROT in bundle["record_ids"] and VEP_F508 in bundle["record_ids"] and PAPER in bundle["record_ids"]
    assert any(r.startswith("pmid-search:") for r in bundle["record_ids"])

    prompt = read(out / "prompts" / f"{CID}.json")
    assert prompt["system"] == SYSTEM_PROMPT and prompt["output_model"] == "GeneDossier"
    assert [t["name"] for t in prompt["tool_definitions"]] == ["get_record", "search_literature", "get_paper"]
    assert "the bundle already carries its domain map" in prompt["tool_definitions"][0]["description"]
    schema = prompt["output_schema"]
    assert schema["$defs"]["VariantPosition"]["properties"]["key"]["enum"] == [F508DEL, G542X]
    assert "protein_position" not in schema["$defs"]["VariantPosition"]["properties"]      # the engine's
    assert "region" not in schema["$defs"]["VariantPosition"]["properties"]
    assert schema["properties"]["uniprot_accession"]["enum"] == [ACC]
    assert schema["properties"]["candidate_id"]["enum"] == [CID]

    user = prompt["user"]
    assert user.startswith("Write the gene dossier for the candidate below.")
    assert f"# Candidate {CID}" in user
    assert f"{ACC} Cystic fibrosis transmembrane conductance regulator · 1480 aa" in user
    assert f"Domain 423–646 ABC transporter 1 [{UNIPROT}]" in user
    assert "residue (engine, from hgvsp): 508" in user and "residue (engine, from hgvsp): 542" in user
    assert "- Domain: ABC transporter 1 (423–646) [uniprot:P13569]" in user
    assert "VAR_000171 F→del" in user and "(20 UniProt evidence references)" in user
    assert "Cystic fibrosis: A common generalized disorder of the exocrine glands" in user
    assert f"- [{PAPER}]" in user and "## Papers retrieved" in user
    assert "17:7675088" not in user                       # the other candidate's variant is not in this prompt
    assert "PubMed:" not in user                          # UniProt's own references are not citable ids
    md = (out / "prompts" / f"{CID}.md").read_text()
    assert md.startswith(f"# Prompt — {CID}") and "## System" in md and "## User" in md


def test_a_gene_uniprot_does_not_know_stops_with_a_manifest_and_no_dossier(run_dir: Path, tmp_path: Path,
                                                                                monkeypatch: pytest.MonkeyPatch):
    """An absence, recorded: the chain's gene is renamed to a symbol UniProt has no
    reviewed human entry for, and the step ends at exit 0 with a note."""
    candidates = run_dir / "03_filter" / "candidates.json"
    doc = json.loads(candidates.read_text())
    doc["candidates"][0]["gene_symbol"] = "NOTAGENE1"
    candidates.write_text(json.dumps(doc))
    manifest_path = run_step(run_dir, dry_run=True)
    m = read(manifest_path)
    assert m["params"]["accession"] is None
    assert any("no reviewed human entry" in n for n in m["notes"]) and any("no dossier" in n for n in m["notes"])
    assert not (run_dir / DOSSIER_DIR / f"{CID}.json").exists()
    assert not (run_dir / DOSSIER_DIR / "bundles").exists()
    monkeypatch.setattr(dr, "default_retrievers", lambda http: retrievers())
    result = CliRunner().invoke(main, ["dossier", "--run", str(run_dir), "--dry-run", "--cache", str(tmp_path / "c")])
    assert result.exit_code == 0, result.output
    assert "UniProt has no reviewed human entry for the gene · no dossier" in result.output


# --------------------------------------------------------------- the scripted run

def test_a_scripted_run_writes_a_validated_dossier(run_dir: Path):
    http = stub_http()
    manifest_path = run_step(run_dir, http=http)
    out = run_dir / DOSSIER_DIR
    doc = read(out / f"{CID}.json")
    GeneDossier.model_validate(doc)
    assert doc["candidate_id"] == CID and doc["gene_symbol"] == "CFTR" and doc["uniprot_accession"] == ACC

    # 1. the engine's residue replaces the model's, and its region replaces the model's
    by_key = {p["key"]: p for p in doc["variant_positions"]}
    assert list(by_key) == [F508DEL, G542X]                       # the foreign key is gone, the chain's order kept
    assert by_key[F508DEL]["protein_position"] == 508             # the model wrote 999
    assert by_key[F508DEL]["region"][:2] == ["Topological domain: Cytoplasmic (359–858)", "Domain: ABC transporter 1 (423–646)"]
    assert any(r.startswith("VAR_000171 F→del") for r in by_key[F508DEL]["region"])
    assert "a domain I remember" not in by_key[F508DEL]["region"]
    assert by_key[G542X]["protein_position"] == 542
    assert any("VAR_080305" in r for r in by_key[G542X]["region"])

    # 2. a bare accession a record carries stands; one no record carries is redacted
    protein = [c["statement"] for c in doc["protein"]]
    assert any("P13569 is the reviewed human entry" in s for s in protein)
    assert any("the paralogue [^1] is not" in s for s in protein)  # a footnote marker, never a sentence in the prose
    assert not any("as I recall it" in s for s in protein)        # the claim citing no uniprot: record
    assert "[^2]" in doc["limits"][1]                             # the bare Pfam id the entry does not carry
    assert "PF99999" not in json.dumps(doc)

    # 3. unknown ids drop the claim and are pruned from the literature
    assert [c["statement"] for c in doc["mechanism_of_disease"]] == \
        [f"Loss of apical chloride conductance causes cystic fibrosis [{UNIPROT}] [{PAPER}]"]
    assert doc["literature"] == [PAPER, OTHER_PAPER]
    assert FAKE_PMID not in json.dumps(doc)
    assert len(doc["region_knowledge"]) == len(doc["genotype_patterns"]) == len(doc["functional_test"]) == 1

    validation = read(out / "validation" / f"{CID}.json")
    assert validation["candidate_id"] == CID and validation["model"] == "GeneDossier"
    reasons = {r["path"]: r["reason"] for r in validation["rejections"]}
    assert reasons["protein[1]"] == "protein claim cites no UniProt record"
    assert "is not a variant of this candidate" in reasons["variant_positions[2]"]
    assert f"unknown evidence id(s): {FAKE_PMID}" in reasons["mechanism_of_disease[1]"]
    assert any("bare accession not carried by any citable record: O60566" in r for r in reasons.values())
    assert any("PF99999" in r for r in reasons.values())
    notes = " | ".join(validation["notes"])
    assert "the model wrote 999; the vep: record's hgvsp gives 508" in notes
    assert "candidate_id: the model wrote 'CFTR'" in notes and "gene_symbol: the model wrote 'cftr'" in notes
    assert "replaced by the features the uniprot: record carries" in notes

    m = read(manifest_path)
    c = m["counts"]
    assert c["dossiers_written"] == 1 and c["variant_positions"] == 2
    assert c["claims_kept"] == 6 and c["items_dropped"] == 3 and c["positions_replaced"] == 2
    assert c["redactions"] == 2 and c["validation"]["literature_removed"] == 1
    assert c["tool_calls"] == {"get_paper": 1, "get_record": 1}
    assert m["params"]["disclosure"].startswith("FakeClient")
    assert m["params"]["source_versions"]["uniprot"] == ["UniProt release 2026_03 (02-September-2026)"]
    assert set(m["outputs"]) == {"dossier_json", "dossier_md", "evidence_index"}
    for leak in ("117559590", "Phe508del", "PUBLIC01"):
        assert leak not in http.sent_text(), leak


def test_the_markdown_shows_the_engine_columns_and_resolves_the_references(run_dir: Path):
    run_step(run_dir)
    md = (run_dir / DOSSIER_DIR / f"{CID}.md").read_text()
    assert md.startswith(f"# Gene dossier — CFTR ({CID})")
    for title in ("## Protein", "## Mechanism of disease", "## Where the variants fall",
                  "## What is known about those regions", "## Published genotype patterns",
                  "## What a functional test would show", "## Limits", "## Literature", "## References"):
        assert title in md, title
    assert md.index("## Protein") < md.index("## Where the variants fall") < md.index("## Limits") < md.index("## References")
    assert "| variant | residue | region | consequence (cited) |" in md
    assert f"| {F508DEL} | 508 | Topological domain: Cytoplasmic (359–858); Domain: ABC transporter 1 (423–646);" in md
    assert f"| {G542X} | 542 |" in md
    assert f"- [{UNIPROT}] — https://www.uniprot.org/uniprotkb/{ACC}/entry" in md
    assert f"- [{PAPER}]" in md and "(1995)" not in md
    assert R175H not in md and FAKE_PMID not in md
    assert "_FakeClient (scripted, no API call), model fake-model, effort low_" in md


def test_the_evidence_chain_section_is_appended_once_and_replaced_on_a_rerun(run_dir: Path):
    chain_md = with_evidence_chain(run_dir)
    manifest_path = run_step(run_dir)
    m = read(manifest_path)
    text = chain_md.read_text()
    assert text.startswith("# Evidence chains\n")
    assert text.count(f"<!-- dossier:{CID} -->") == 1 and text.count(f"<!-- /dossier:{CID} -->") == 1
    assert text.count(f"## Gene dossier — {CID}") == 1
    assert "### Protein" in text and "#### " not in text          # the dossier's headings one level down
    assert f"# Gene dossier — CFTR ({CID})" not in text            # the dossier's own title line is dropped
    hashes = m["params"]["evidence_chain_md"]
    assert set(hashes) == {"sha256_before", "sha256_after"} and hashes["sha256_before"] != hashes["sha256_after"]

    again = run_step(run_dir)
    text2 = chain_md.read_text()
    assert text2 == text                                            # idempotent: one section, same bytes
    assert text2.count(f"## Gene dossier — {CID}") == 1
    h2 = read(again)["params"]["evidence_chain_md"]
    assert h2["sha256_before"] == h2["sha256_after"] == hashes["sha256_after"]

    # a rerun of stage 5 rewrites the file without the section; the step appends it again
    chain_md.write_text("# Evidence chains\n\nrewritten by stage 5.\n")
    run_step(run_dir)
    assert chain_md.read_text().count(f"## Gene dossier — {CID}") == 1


def test_a_rerun_replaces_the_step_outputs_and_keeps_the_store(run_dir: Path):
    run_step(run_dir, dry_run=True)
    out = run_dir / DOSSIER_DIR
    (out / "validation").mkdir(exist_ok=True)
    (out / "validation" / f"{CID}.json").write_text("{}")
    (out / "OTHER:cand.json").write_text("{}")                      # another candidate's dossier stays
    records_before = EvidenceStore(run_dir / "05_reason" / "evidence").count()
    run_step(run_dir)
    assert read(out / "validation" / f"{CID}.json")["candidate_id"] == CID
    assert (out / "OTHER:cand.json").exists()
    assert EvidenceStore(run_dir / "05_reason" / "evidence").count() == records_before  # nothing refetched, nothing lost
    assert read(out / "manifest.json")["counts"]["evidence_records_added"] == 0


def test_the_agent_failure_leaves_a_failed_transcript_and_no_dossier(run_dir: Path):
    class Failing:
        def run(self, request: ac.AgentRequest) -> ac.AgentResult:
            raise ac.AgentError("Anthropic API rate limit (retry-after 30s): slow down", request_id="req_1")

    with pytest.raises(ac.AgentError, match="rate limit"):
        run_step(run_dir, client=Failing())
    out = run_dir / DOSSIER_DIR
    failed = read(out / "transcripts" / f"{CID}.failed.json")
    assert failed["candidate_id"] == CID and failed["request_id"] == "req_1"
    assert not (out / f"{CID}.json").exists() and not (out / "manifest.json").exists()
    assert (run_dir / "05_reason" / "evidence" / "index.json").exists()  # the store's index is still written


def test_an_unknown_candidate_and_a_missing_stage_5_are_errors(run_dir: Path, tmp_path: Path):
    with pytest.raises(KeyError, match="no stage-5 chain for candidate"):
        run_step(run_dir, candidate_id="MTHFR:hom", dry_run=True)
    with pytest.raises(FileNotFoundError, match="stage 5 has not run"):
        run_step(tmp_path / "empty", dry_run=True)
    with pytest.raises(ValueError, match="max_turns must be at least 1"):
        run_step(run_dir, dry_run=True, max_turns=0)
    with pytest.raises(ValueError, match="literature_max_results must be between 1 and 25"):
        run_step(run_dir, dry_run=True, literature_max_results=99)


# ------------------------------------------------------------------- checks alone

def test_the_resolver_answers_for_uniprot_pfam_and_interpro(run_dir: Path):
    entry = UniprotRetriever(stub_http()).entry(ACC)
    resolver = DossierResolver([entry])
    assert accession_tokens(f"{ACC} and O60566, PF00005 with IPR003439, PF99999 and rs113993960") == \
        ["P13569", "O60566", "PF00005", "IPR003439", "PF99999", "rs113993960"]
    import re
    from engine.dossier.checks import _ACCESSION
    verdicts = {m.group(0): resolver.resolves(m) for m in _ACCESSION.finditer(
        f"{ACC} O60566 PF00005 PF99999 IPR003439 IPR999999 VCV000007105")}
    assert verdicts == {ACC: True, "O60566": False, "PF00005": True, "PF99999": False, "IPR003439": True,
                        "IPR999999": False, "VCV000007105": False}
    assert re.search(r"IPR003439", json.dumps(entry.payload))  # the entry's own cross-reference


def test_an_isoform_mismatch_refuses_to_apply_the_map(run_dir: Path):
    """When the transcript's reference amino acid is not what the canonical sequence
    carries at that residue, the features are numbered on another isoform: the region
    says so and the map is not applied."""
    rows = read(run_dir / "03_filter" / "candidates.json")
    entry = UniprotRetriever(stub_http()).entry(ACC)
    from engine.agents.bundle import build_bundle
    from engine.dossier.run import variant_entry
    bundle = build_bundle(rows["candidates"][0], run_dir)
    v = bundle.variants[0]
    v.columns["hgvsp"] = "ENSP00000003084.6:p.Trp508del"           # W, where the canonical sequence has F
    out = variant_entry(v, None, entry)
    assert out["residue"] == 508 and out["reference_residue"] == "W"
    assert out["region"] == ["residue 508: the transcript's reference amino acid (W) is not the UniProt canonical "
                             "sequence's (F); the feature map is numbered on another isoform and is not applied"]
    assert "region not applied" in out["note"]


def test_an_omitted_chain_variant_is_filled_by_the_engine(run_dir: Path):
    answer = dict(cftr_answer(), variant_positions=[p for p in cftr_answer()["variant_positions"] if p["key"] == F508DEL])
    run_step(run_dir, client=fake_client(answer))
    doc = read(run_dir / DOSSIER_DIR / f"{CID}.json")
    filled = {p["key"]: p for p in doc["variant_positions"]}
    assert list(filled) == [F508DEL, G542X]
    assert filled[G542X]["protein_position"] == 542 and filled[G542X]["consequence"] == ""
    assert any("VAR_080305" in r for r in filled[G542X]["region"])
    notes = " | ".join(read(run_dir / DOSSIER_DIR / "validation" / f"{CID}.json")["notes"])
    assert f"the model returned no entry for '{G542X}'" in notes


# ------------------------------------------------------------------- footnotes

def test_the_markdown_carries_footnotes_for_redaction_markers_and_tolerates_their_absence():
    """Task D replaces the redaction sentence with a ``[^k]`` marker and puts ``marker``
    on the rejection. The renderer already prints the footnote for a marker it finds,
    with the reason when a rejection carries it and the fixed sentence when none does;
    a dossier without markers gets no footnote line."""
    from dataclasses import dataclass

    from engine.dossier.render import FOOTNOTE_UNKNOWN, footnotes, render_dossier

    @dataclass
    class Marked:
        path: str
        reason: str
        marker: int | None = None

    assert footnotes(["a [^2] b [^1]", "c [^2]"], [Marked("limits[0]", "no such record: pmid:9", 1)]) == [
        "[^2]: " + FOOTNOTE_UNKNOWN, "[^1]: citation removed by the validator: no such record: pmid:9"]
    assert footnotes(["nothing here"], []) == []
    assert footnotes(["x [^3]"], [{"path": "limits[0]", "reason": "gone", "marker": 3}]) == [
        "[^3]: citation removed by the validator: gone"]

    dossier = GeneDossier(candidate_id=CID, gene_symbol="CFTR", uniprot_accession=ACC,
                          protein=[{"statement": f"A domain map claim [{UNIPROT}] and a lost one [^1]", "evidence_ids": [UNIPROT]}],
                          mechanism_of_disease=[], variant_positions=[], region_knowledge=[], genotype_patterns=[],
                          functional_test=[], limits=["a limit"], literature=[])
    md = render_dossier(dossier, None, rejections=[Marked("protein[0].statement", "no such record: pmid:99999999", 1)])
    assert "and a lost one [^1]" in md
    assert "[^1]: citation removed by the validator: no such record: pmid:99999999" in md
    assert md.index("[^1]: citation removed") < md.index("## Mechanism of disease")
    assert "[^" not in render_dossier(dossier.model_copy(update={"protein": []}), None)


def test_the_manifest_records_the_four_stage_checks_and_the_citation_scope(run_dir: Path):
    run_step(run_dir, dry_run=True)
    p = read(run_dir / DOSSIER_DIR / "manifest.json")["params"]
    assert set(p["stage_checks"]) == {"keys", "engine_fields", "bare_accessions_in_prose", "protein_claims"}
    assert "protein_position and region are set by the engine" in p["stage_checks"]["engine_fields"]
    assert p["citation_scope"] == dr.CITATION_SCOPE and "uniprot: entry" in p["citation_scope"]
    assert p["output_schema"].startswith("GeneDossier without protein_position/region")
    assert p["uniprot"]["fields"] == p["uniprot_fields"] and p["uniprot"]["per_second"] == 3.0
    assert p["literature"]["max_results_per_query"] == 8 and p["literature"]["peer_reviewed_only"] is True
    assert p["evidence_stages"] == ["02_retrieve", "04_rank", "05_reason"]


def test_the_recorded_fixtures_are_public_and_documented():
    readme = (FIXTURES / "dossier" / "README.md").read_text()
    assert "2026-09-17" in readme and "pageSize" in readme
    for name in ("search_CFTR_cystic_fibrosis", "search_CFTR_cbavd", "search_CFTR_mechanism", "search_CFTR_genotype"):
        fx = json.loads((FIXTURES / "dossier" / f"{name}.json").read_text())
        assert fx["status"] == 200 and fx["request"]["params"]["pageSize"] == 8
        assert fx["request"]["params"]["query"].endswith("AND SRC:MED") and fx["body"]["hitCount"] > 0
        assert len(fx["body"]["resultList"]["result"]) == 8
        for leak in ("117559590", "117587778", "Phe508del", "PUBLIC01"):
            assert leak not in fx["request"]["params"]["query"], leak


# -------------------------------------------------------------------------- view

def test_the_view_resolves_every_id_and_says_when_nothing_ran(run_dir: Path, tmp_path: Path):
    assert dossier_view(tmp_path / "nothing") == {"present": False, "dossier": None}
    run_step(run_dir, dry_run=True)
    dry = dossier_view(run_dir)
    assert dry["present"] and dry["dry_run"] is True and dry["dossier"] is None
    assert dry["uniprot"]["url"] == f"https://www.uniprot.org/uniprotkb/{ACC}/entry"
    assert dry["gene_symbol"] == "CFTR"

    run_step(run_dir)
    view = dossier_view(run_dir)
    assert view["candidate_id"] == CID and view["model"] == "fake-model" and view["dry_run"] is False
    assert view["file"].endswith(f"{CID}.json") and view["validation"]["counts"]["items_dropped"] == 3
    d = view["dossier"]
    assert d["uniprot_accession"] == ACC and len(d["protein"]) == 2
    assert d["protein"][0]["evidence"][0] == {"id": UNIPROT, "url": f"https://www.uniprot.org/uniprotkb/{ACC}/entry",
                                              "source": "uniprot", "stage": "05_reason"}
    positions = {p["key"]: p for p in d["variant_positions"]}
    assert positions[F508DEL]["protein_position"] == 508
    assert positions[F508DEL]["evidence"][0]["url"].startswith("https://")
    assert [e["id"] for e in d["literature"]] == [PAPER, OTHER_PAPER]
    refs = [e["id"] for e in d["references"]]
    assert UNIPROT in refs and PAPER in refs and CLINVAR_F508 not in refs
    assert all(e["url"] for e in d["references"])
    assert dossier_view(run_dir, candidate_id=CID)["dossier"] == d

    html = render_dossier_html(view, prose=lambda t: str(t or ""), link=lambda e: f"<a>{e['id']}</a>")
    assert html.startswith("<article class=\"dossier\">") and html.endswith("</article>")
    for title in ("Protein", "Mechanism of disease", "Where the variants fall", "Limits", "References"):
        assert f"<h4>{title}</h4>" in html or f"<h3>{title}</h3>" in html, title
    assert f"<a>{UNIPROT}</a>" in html and "<td class=\"num\">508</td>" in html
    assert "Domain: ABC transporter 1 (423–646)" in html
    assert render_dossier_html({"present": False}, prose=str, link=lambda e: "")\
        .count("No dossier step in this run") == 1


# --------------------------------------------------------------------------- cli

def test_cli_dry_run_prints_counts_and_no_gene(run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dr, "default_retrievers", lambda http: retrievers())
    result = CliRunner().invoke(main, ["dossier", "--run", str(run_dir), "--dry-run", "--cache", str(tmp_path / "cache")])
    assert result.exit_code == 0, result.output
    assert "dossier · " in result.output and "DRY RUN" in result.output and "best-priority candidate" in result.output
    assert "bundle and prompt written · papers retrieved: " in result.output and "no model called" in result.output
    assert "CFTR" not in result.output and "P13569" not in result.output and "117559590" not in result.output
    assert result.output.rstrip().endswith(f"manifest: {run_dir / DOSSIER_DIR / 'manifest.json'}")
    result = CliRunner().invoke(main, ["dossier", "--run", str(run_dir), "--dry-run", "--candidate", "MTHFR:hom"])
    assert result.exit_code == 1 and "FAILED: " in result.output and "no stage-5 chain" in result.output


def test_cli_runs_the_step_and_names_the_candidate_only_when_asked(run_dir: Path, tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch):
    with_evidence_chain(run_dir)
    monkeypatch.setattr(dr, "default_retrievers", lambda http: retrievers())
    monkeypatch.setattr(ac, "AnthropicClient", lambda **kw: fake_client())
    args = ["dossier", "--run", str(run_dir), "--model", "fake-model", "--effort", "low", "--cache", str(tmp_path / "cache")]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "candidate: agent (fake-model, effort low)" in result.output
    assert "candidate: claims kept: 6 · variant positions: 2 (replaced: 2) · rejections: 6 · records added: 35" in result.output
    assert "tokens in/out: 0/0 · api calls: 0 · tool calls: 2" in result.output
    assert "evidence_chain.md: dossier section written" in result.output
    assert "CFTR" not in result.output and "P13569" not in result.output and "117559590" not in result.output
    result = CliRunner().invoke(main, args + ["-v"])
    assert result.exit_code == 0, result.output
    assert f"{CID}: claims kept: 6" in result.output


def test_cli_reports_an_agent_failure_and_exits_1(run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class Failing:
        def run(self, request: ac.AgentRequest) -> ac.AgentResult:
            raise ac.AgentError("Anthropic API rate limit (retry-after 30s): slow down")

    monkeypatch.setattr(dr, "default_retrievers", lambda http: retrievers())
    monkeypatch.setattr(ac, "AnthropicClient", lambda **kw: Failing())
    result = CliRunner().invoke(main, ["dossier", "--run", str(run_dir), "--cache", str(tmp_path / "cache")])
    assert result.exit_code == 1 and "FAILED: Anthropic API rate limit" in result.output
    assert not (run_dir / DOSSIER_DIR / "manifest.json").exists()


# -------------------------------------------------------------------- determinism

def test_two_runs_write_the_same_bytes(run_dir: Path, tmp_path: Path):
    run_step(run_dir)
    out = run_dir / DOSSIER_DIR
    first = {p.name: p.read_text() for p in (out.glob(f"{CID}.*"))}
    first["bundle"] = (out / "bundles" / f"{CID}.json").read_text()
    first["prompt"] = (out / "prompts" / f"{CID}.json").read_text()
    run_step(run_dir)
    second = {p.name: p.read_text() for p in (out.glob(f"{CID}.*"))}
    second["bundle"] = (out / "bundles" / f"{CID}.json").read_text()
    second["prompt"] = (out / "prompts" / f"{CID}.json").read_text()
    assert first == second
    assert "2026-09-17" not in first[f"{CID}.md"]  # no clock anywhere but the manifest


# ------------------------------------------------------------------------- live

@pytest.mark.skipif(not os.environ.get("ENGINE_LIVE_TESTS"), reason="set ENGINE_LIVE_TESTS=1 to talk to UniProt and Europe PMC")
def test_live_dry_run_over_the_public_case(run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A dry run against the live services with the public CFTR case: the release
    headers must survive ``Http``'s header filter (the integrator's request 1)."""
    from engine.retrieve import http as http_mod

    monkeypatch.setattr(http_mod, "_KEEP", tuple(http_mod._KEEP) + ("x-uniprot-release", "x-uniprot-release-date", "x-total-results"))
    manifest_path = dr.run_dossier(run_dir, None, None, "claude-opus-5", "low", True, cache_root=tmp_path / "cache")
    m = read(manifest_path)
    assert m["params"]["accession"] == ACC and m["params"]["uniprot_release"].startswith("UniProt release 20")
    assert m["counts"]["features_in_map"] >= 39 and m["counts"]["papers_retrieved"] > 0
    user = read(run_dir / DOSSIER_DIR / "prompts" / f"{CID}.json")["user"]
    assert "residue (engine, from hgvsp): 508" in user and "Domain: ABC transporter 1 (423–646)" in user
