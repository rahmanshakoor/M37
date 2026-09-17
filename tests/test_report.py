"""The report — ``engine.report`` (CONTRACTS.md, "Report").

Network-free: the run directory is assembled from the public demo fixtures the way
``tests/test_public_case.py`` does — the recorded stage-2 store, the ``hpo:`` term
records of the public case (stage 5 fetches the case terms, so its store is seeded
from ``tests/fixtures/hpo/records``), the real stage 3, stage 4 over the recorded
Exomiser outputs with the fake Docker runner, stage 5 with a scripted client answering
the recorded chain (plus one fabricated citation, so the validator has something to
reject), stage 6 with the real drug, trial and disease retrievers over the recorded
fixtures — so every view and the rendered document are checked against files real
stages wrote. The public run ``scripts/run_public_case.sh`` leaves
in ``${TMPDIR:-/tmp}/engine-public-case/run`` is checked too when it is present.
Nothing here derives from a person.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from html import escape
from html.parser import HTMLParser
from pathlib import Path

import pytest
from click.testing import CliRunner

import engine.cli
from engine.agents import client as ac
from engine.filter.run import run_filter
from engine.medicine.run import run_medicine
from engine.rank.exomiser import DEFAULT_CONFIG, ExomiserConfig
from engine.rank.run import run_rank
from engine.reason.run import run_reason
from engine.report import views
from engine.report.cli import commands, report, report_command
from engine.report.render import CSS, FONTS_URL, _Linker, _claims, render_run, write_report
from tests.test_medicine import (CH_IVACAFTOR, CH_IVACAFTOR_MEC, DISEASE, NCT_SEARCH, OT_CF, OT_DISEASE, OT_IVACAFTOR,
                                 OT_TARGET, TRIAL, retrievers, stub_http)
from tests.test_public_case import CFTR_PAIR_IDS, CHAIN, F508DEL, FIX, G542X, HPO, R175H, make_run_from_recorded_stage2
from tests.test_rank import FakeDocker, make_data_dir

HPO_RECORDS = Path(__file__).parent / "fixtures" / "hpo" / "records"
"""The five public ``hpo:`` term records; stage 5 serves the case terms from a store
that holds them instead of asking the JAX API."""
CLASS = "potentiation of the residual F508del channel"
FAKE_PMID = "pmid:999999"
"""A paper no store holds: the scripted chain cites it under PS3, which the stage-5
checks drop — the rejection must be on the page."""

PUBLIC_RUN = Path(os.environ.get("PUBLIC_RUN_DIR") or Path(os.environ.get("TMPDIR", "/tmp")) / "engine-public-case" / "run")


# ------------------------------------------------------------------------ fixtures

def scripted_chain() -> dict:
    answer = json.loads(CHAIN.read_text())
    for v in answer["variants"]:
        v["classification"] = None
        # the recorded fixture was made without stage 4, so its PP4 is unverified; this
        # run has the ranker, and the model claims the criterion for it to check
        for c in v["criteria"]:
            if c["code"] == "PP4":
                c["met"] = True
    answer["variants"][1]["criteria"].append({
        "code": "PS3", "strength": "strong", "met": True,
        "justification": "A functional study the model made up", "evidence_ids": [FAKE_PMID],
    })
    return answer


def scripted_report() -> dict:
    """The medicine report the scripted model writes: one class backed by the trial
    search it made, one candidate inside it, one rejected on a record, and a
    consequence claim citing the disease record — the ladder, in miniature."""
    return {
        "candidate_id": "CFTR:comphet", "gene_symbol": "CFTR",
        "mechanism": [{"statement": f"Two loss-of-function alleles [vep:{F508DEL}] [vep:{G542X}]",
                       "evidence_ids": [f"vep:{F508DEL}", f"vep:{G542X}"]},
                      {"statement": f"p.Phe508del is cited bare here, vep:{F508DEL}, as the validator leaves a resolving id",
                       "evidence_ids": [f"vep:{F508DEL}"]}],
        "consequence": [{"statement": f"Chloride secretion fails and the airway surface dehydrates; the disease record "
                                      f"names recurrent bronchopulmonary infections [{OT_DISEASE}]",
                         "evidence_ids": [OT_DISEASE]}],
        "pathway_targets": [{"statement": f"Cystic fibrosis is the top association [{OT_CF}]; tractable [{OT_TARGET}]",
                             "evidence_ids": [OT_CF, OT_TARGET]}],
        "intervention_classes": [{
            "name": CLASS, "acts_on": f"the residual channel at the apical membrane [vep:{F508DEL}]",
            "targets": ["CFTR"], "searched": [NCT_SEARCH, OT_TARGET], "verdict": "candidates_proposed",
            "rejection_reason": "", "evidence_ids": [OT_TARGET],
        }],
        "candidates": [{
            "name": "Ivacaftor", "chembl_id": "CHEMBL2010601", "intervention_class": CLASS,
            "mechanism_of_action": "CFTR potentiator", "approval_status": "approved (ChEMBL max_phase 4)",
            "approved_indication": "cystic fibrosis with a gating mutation (ChEMBL first_approval 2012)",
            "rationale": f"Potentiates the residual p.Phe508del channel [{CH_IVACAFTOR}] [{CH_IVACAFTOR_MEC}] [{OT_IVACAFTOR}]",
            "counter_arguments": ["p.Gly542Ter produces no protein to potentiate",
                                  "the trials cited were in other genotype combinations"],
            "paediatric_safety": f"the record carries no exposure data below the approved age [{CH_IVACAFTOR}]",
            "evidence_ids": [CH_IVACAFTOR, CH_IVACAFTOR_MEC, OT_IVACAFTOR], "trial_ids": [TRIAL],
        }],
        "considered_and_rejected": [{"name": "Crofelemer", "intervention_class": CLASS,
                                     "reason": f"an inhibitor, the opposite action type [{OT_TARGET}]",
                                     "evidence_ids": [OT_TARGET]}],
        "surveillance": [{"statement": f"the disease record names hepatobiliary complications [{OT_DISEASE}]",
                          "evidence_ids": [OT_DISEASE]}],
        "follow_up_experiments": [f"Sweat chloride after ivacaftor exposure in vitro [{CH_IVACAFTOR}]"],
        "limits": ["No paper was retrieved in this run"],
        "literature": [],
    }


@pytest.fixture(scope="module")
def full_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Stages 2–6 of the public demo, every one written by the real stage code."""
    tmp = tmp_path_factory.mktemp("report")
    run = make_run_from_recorded_stage2(tmp)
    shutil.copytree(HPO_RECORDS, run / "05_reason" / "evidence", dirs_exist_ok=True)  # stage 5 serves the case terms
    run_filter(run)
    cfg = ExomiserConfig.load(DEFAULT_CONFIG)
    run_rank(run, FIX / "case.yaml", make_data_dir(tmp, cfg), run=FakeDocker(cfg.image_digest))
    reader = ac.FakeClient(scripted_chain(), turns=[ac.FakeTurn([("get_record", {"record_id": rid}) for rid in CFTR_PAIR_IDS], text="reading")])
    run_reason(run, 1, reader, "fake-model", "low", False, http=stub_http(), case_path=FIX / "case.yaml")
    writer = ac.FakeClient(scripted_report(), turns=[ac.FakeTurn([("drugs_for_gene", {"gene": "CFTR"})], text="drugs"),
                                                     ac.FakeTurn([("search_trials", {"condition": "cystic fibrosis", "intervention": "ivacaftor",
                                                                                     "term": None, "max_results": 3})], text="trials")])
    run_medicine(run, None, writer, "fake-model", "low", False, retrievers=retrievers(stub_http()),
                 case_path=FIX / "case.yaml", case_disease=[DISEASE])
    return run


@pytest.fixture
def run_copy(full_run: Path, tmp_path: Path) -> Path:
    """A private copy of the full run, for tests that damage it."""
    run = tmp_path / "run"
    shutil.copytree(full_run, run)
    return run


@pytest.fixture
def early_run(tmp_path: Path) -> Path:
    """A run that stopped after stage 3 (stage 1 is not even there)."""
    run = make_run_from_recorded_stage2(tmp_path)
    run_filter(run)
    return run


def validation_with_a_marker(run: Path, marker: int = 1) -> None:
    """Give the stage-6 validation record one redaction that carries a marker, as the
    validator will once a citation is replaced by ``[^k]`` — the footnote's reason."""
    path = run / "06_medicine" / "validation" / "CFTR:comphet.json"
    doc = json.loads(path.read_text())
    doc["rejections"].append({"path": "limits[0]", "marker": marker,
                              "reason": "inline citation to a record not in the store: pmid:99999999"})
    path.write_text(json.dumps(doc, sort_keys=True, indent=1) + "\n")


class TagBalance(HTMLParser):
    VOID = {"meta", "link", "br", "hr", "img", "input", "col", "wbr"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in self.VOID:
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.errors.append(f"</{tag}> at {self.getpos()} closes <{self.stack[-1] if self.stack else '?'}>")


def assert_well_formed(html: str) -> None:
    p = TagBalance()
    p.feed(html)
    assert not p.errors and not p.stack, (p.errors[:5], p.stack[:5])


def link(rid: str, url: str) -> str:
    return f'<a class="id" href="{url}" rel="noopener">{rid}</a>'


# --------------------------------------------------------------------------- views

def test_run_summary_reads_the_manifests(full_run: Path):
    s = views.run_summary(full_run)
    assert s["sample"] is None  # stage 1 did not run here; the sample comes from its manifest only
    assert s["hpo"] == HPO and s["hpo_source"] == "04_rank/joined.json"
    # every case term carries the label its own hpo: record spells, never one from elsewhere
    assert [(t["id"], t["label"], t["record_id"]) for t in s["hpo_terms"]][:1] == \
        [("HP:0012236", "Elevated sweat chloride", "hpo:HP:0012236")]
    assert all(t["url"].startswith("https://hpo.jax.org/browse/term/") for t in s["hpo_terms"])
    assert views.hpo_terms(full_run, ["HP:9999999"]) == [{"id": "HP:9999999", "label": "", "record_id": None, "url": None}]
    assert s["stages_present"] == ["02_retrieve", "03_filter", "04_rank", "05_reason", "06_medicine"]
    by_dir = {st["dir"]: st for st in s["stages"]}
    assert by_dir["01_ingest"] == {"dir": "01_ingest", "stage": "ingest", "present": False, "manifest": False}
    assert by_dir["02_retrieve"]["manifest"] is False  # the fixture store has no stage-2 manifest
    assert by_dir["03_filter"]["headline"] == [("rows in", 12), ("kept", 3), ("dropped", 9), ("candidates", 2)]
    assert by_dir["05_reason"]["dry_run"] is False and by_dir["05_reason"]["engine_version"] == "0.1.0"
    assert by_dir["05_reason"]["duration_s"] is not None and by_dir["05_reason"]["started_at"]
    assert by_dir["06_medicine"]["headline"][0] == ("candidate", "CFTR:comphet")


def test_candidates_view_joins_stage_4_and_marks_agreement(full_run: Path):
    c = views.candidates_view(full_run)
    assert c["present"] and c["rank_present"]
    assert [(x["candidate_id"], x["priority"]) for x in c["candidates"]] == [("CFTR:comphet", 1), ("TP53:het_single", 2)]
    cftr, tp53 = c["candidates"]
    assert cftr["exomiser"]["rank"] == 1 and cftr["exomiser"]["moi"] == "AR" and cftr["exomiser"]["match"] == "symbol"
    assert cftr["exomiser"]["score"] == 0.9894 and cftr["exomiser"]["evidence"]["id"] == "exomiser:CFTR"
    assert cftr["exomiser"]["evidence"]["url"] == "https://www.ncbi.nlm.nih.gov/gene/1080"
    assert tp53["exomiser"]["rank"] == 2
    assert [x["agreement"] for x in c["candidates"]] == ["agrees", "agrees"] and [x["join"] for x in c["candidates"]] == ["ranked", "ranked"]
    assert c["top_agreement"] is True and c["order_agreement"] is True
    assert c["join_check"]["stale"] is False and c["join_check"]["missing"] == [] and c["join_check"]["dropped"] == []
    assert c["join_check"]["candidates_sha256_recorded"] == c["join_check"]["candidates_sha256_current"] == \
        json.loads((full_run / "04_rank" / "manifest.json").read_text())["inputs"]["candidates"]["sha256"]
    # every stage-3 field passes through as written, with the evidence ids resolved
    assert cftr["rule_hits"] == json.loads((full_run / "03_filter" / "candidates.json").read_text())["candidates"][0]["rule_hits"]
    assert [v["key"] for v in cftr["variants"]] == [F508DEL, G542X]
    assert cftr["variants"][0]["af_used"] == "0.011931254341569912"  # the file's text, not a recomputed float
    assert [e["id"] for e in cftr["variants"][0]["evidence"]] == [f"vep:{F508DEL}", "clinvar:VCV000007105", "gnomad:7-117559590-ATCT-A"]
    assert all(e["url"].startswith("https://") and e["stage"] == "02_retrieve" for e in cftr["variants"][0]["evidence"])
    assert cftr["clinvar_plp"] is True and cftr["phase"]["status"] == "unknown"
    assert c["counts"]["candidates_by_model"] == {"comphet": 1, "het_single": 1}


def test_candidates_view_without_stage_4(early_run: Path):
    c = views.candidates_view(early_run)
    assert c["present"] and not c["rank_present"]
    assert [x["candidate_id"] for x in c["candidates"]] == ["CFTR:comphet", "TP53:het_single"]
    assert all(x["exomiser"] is None and x["agreement"] == "unranked" and x["join"] is None for x in c["candidates"])
    assert c["top_agreement"] is None and c["order_agreement"] is None and c["join_check"] is None


def test_agreement_marks_a_reordered_and_an_unranked_candidate():
    cands = [{"candidate_id": "A", "exomiser": {"rank": 5}}, {"candidate_id": "B", "exomiser": {"rank": 1}}, {"candidate_id": "C", "exomiser": None},
             {"candidate_id": "D", "exomiser": None, "join": "missing"}]
    views._mark_agreement(cands)
    assert [c["agreement"] for c in cands] == ["disagrees", "disagrees", "unranked", "not_joined"]


def test_ranking_view_keeps_the_file_text_and_marks_the_shortlist(full_run: Path):
    r = views.ranking_view(full_run, top_n=1)
    assert r["present"] and r["exomiser_version"] == "14.0.0" and r["data_version"] == "2406"
    rows = r["rows"]
    assert [(x["rank"], x["gene_symbol"], x["exomiser_score"], x["in_shortlist"], x["candidate_id"], x["joined"]) for x in rows] == \
        [("1", "CFTR", "0.9894", True, "CFTR:comphet", True), ("2", "TP53", "0.9828", True, "TP53:het_single", True)]  # TP53 is beyond top_n but on the shortlist
    first = (full_run / "04_rank" / "ranking.tsv").read_text().splitlines()[1].split("\t")
    assert rows[0]["variants"] == first[-1].split(";") == [F508DEL]  # the contributing alleles as the file lists them
    assert rows[0]["evidence"]["url"] == "https://www.ncbi.nlm.nih.gov/gene/1080"
    assert r["rows_total"] == 2 and r["exomiser_only"] == [] and r["hpo"] == HPO
    assert r["join_check"]["stale"] is False
    assert views.ranking_view(full_run.parent / "nowhere") == {"present": False, "rows": [], "exomiser_only": [], "counts": {}, "join_check": None}


def test_chain_view_resolves_every_evidence_id_and_keeps_the_rejection(full_run: Path):
    ch = views.chain_view(full_run)
    assert ch["present"] and ch["dry_run"] is False and ch["model"] == "fake-model" and ch["chains_written"] == 1 and ch["failures"] == []
    assert [c["candidate_id"] for c in ch["chains"]] == ["CFTR:comphet"]
    chain = ch["chains"][0]
    assert chain["claimed_by_manifest"] is True and chain["manifest_note"] is None
    assert chain["gene_symbol"] == "CFTR" and chain["model"] == "comphet" and chain["priority"] == 1
    assert [(v["key"], v["classification"]) for v in chain["variants"]] == [(F508DEL, "vus"), (G542X, "pathogenic")]
    index = json.loads((full_run / "02_retrieve" / "evidence" / "index.json").read_text())
    index.update(json.loads((full_run / "04_rank" / "evidence" / "index.json").read_text()))  # PP4 cites the ranker's record
    for v in chain["variants"]:
        for c in v["criteria"]:
            assert c["code"] != "PS3"  # the fabricated citation was rejected, so the criterion is gone
            for e in c["evidence"]:
                assert e["url"] == index[e["id"]]["url"]
        pp4 = next(c for c in v["criteria"] if c["code"] == "PP4")
        assert [e["id"] for e in pp4["evidence"]] == ["exomiser:CFTR"] and pp4["met"] is True  # CFTR is the ranker's gene 1
    assert [e["id"] for e in chain["references"]] == sorted(CFTR_PAIR_IDS + ["exomiser:CFTR"])
    assert chain["validation"] is not None
    assert any(FAKE_PMID in r["reason"] or "PS3" in r["path"] + r["reason"] for r in chain["validation"]["rejections"])
    assert chain["validation"]["counts"]["items_dropped"] >= 1
    assert "PM2" in chain["validation"]["rules"]
    assert views.chain_view(full_run, "CFTR:comphet")["chains"][0]["candidate_id"] == "CFTR:comphet"
    assert views.chain_view(full_run, "TP53:het_single")["chains"] == []


def test_medicine_view_in_rubric_order(full_run: Path):
    m = views.medicine_view(full_run)
    assert m["present"] and m["dry_run"] is False and m["candidate_id"] == "CFTR:comphet" and m["gene_symbol"] == "CFTR" and m["failures"] == []
    assert m["stage5_verdicts"] == [{"key": F508DEL, "classification": "vus"}, {"key": G542X, "classification": "pathogenic"}]
    r = m["report"]
    assert list(r) == ["candidate_id", "gene_symbol", "patient_context", "mechanism", "consequence",
                       "intervention_classes", "pathway_targets", "candidates", "considered_and_rejected",
                       "surveillance", "follow_up_experiments", "limits", "secondary_findings", "literature", "references"]
    assert [e["id"] for e in r["mechanism"][0]["evidence"]] == [f"vep:{F508DEL}", f"vep:{G542X}"]
    assert [e["id"] for e in r["consequence"][0]["evidence"]] == [OT_DISEASE]
    assert [e["id"] for e in r["surveillance"][0]["evidence"]] == [OT_DISEASE]
    # the patient context resolves to the records a judge opens
    assert [(d["id"], d["name"], d["record"]["id"]) for d in r["patient_context"]["disease"]] == \
        [(DISEASE, "cystic fibrosis", OT_DISEASE)]
    assert r["patient_context"]["disease"][0]["record"]["url"].endswith(DISEASE)
    assert ("HP:0012236", "Elevated sweat chloride") in [(t["id"], t["label"]) for t in r["patient_context"]["hpo"]]
    assert [t["id"] for t in r["patient_context"]["hpo"]] == sorted(HPO)  # the case's terms, in the bundle's order
    assert all(t["record"]["url"].startswith("https://hpo.jax.org/") for t in r["patient_context"]["hpo"])
    assert "engine-filled" in r["patient_context"]["source"]
    # the class and its searches
    cls = r["intervention_classes"][0]
    assert cls["name"] == CLASS and cls["verdict"] == "candidates_proposed" and cls["targets"] == ["CFTR"]
    assert [e["id"] for e in cls["searched"]] == [NCT_SEARCH, OT_TARGET] and all(e["url"] for e in cls["searched"])
    drug = r["candidates"][0]
    assert drug["n"] == 1 and drug["name"] == "Ivacaftor" and drug["chembl"]["id"] == CH_IVACAFTOR
    assert drug["intervention_class"] == CLASS and drug["approved_indication"].startswith("cystic fibrosis with a gating")
    assert "no exposure data below the approved age" in drug["paediatric_safety"]
    assert [e["id"] for e in drug["trials"]] == [TRIAL] and drug["trials"][0]["url"].startswith("https://clinicaltrials.gov/")
    assert all(e["url"] for e in drug["evidence"]) and len(drug["counter_arguments"]) == 2
    assert [(x["name"], [e["id"] for e in x["evidence"]]) for x in r["considered_and_rejected"]] == [("Crofelemer", [OT_TARGET])]
    assert r["secondary_findings"] == []  # the run's other candidate is a VUS
    assert {e["id"] for e in r["references"]} >= {CH_IVACAFTOR, CH_IVACAFTOR_MEC, OT_IVACAFTOR, OT_CF, OT_TARGET, TRIAL,
                                                  OT_DISEASE, NCT_SEARCH, f"vep:{F508DEL}"}
    assert all(e["url"] for e in r["references"])
    assert m["validation"] is not None and m["validation"]["rejections"] == []


def test_provenance_view_lists_every_manifest_with_checksums(full_run: Path):
    p = views.provenance_view(full_run)
    assert [st["dir"] for st in p["stages"]] == [d for d, _ in views.STAGES]
    by_dir = {st["dir"]: st for st in p["stages"]}
    assert by_dir["01_ingest"]["manifest"] is None and by_dir["01_ingest"]["present"] is False
    m5 = by_dir["05_reason"]["manifest"]
    raw = json.loads((full_run / "05_reason" / "manifest.json").read_text())
    assert {i["name"]: i["sha256"] for i in m5["inputs"] if "sha256" in i} == {k: v["sha256"] for k, v in raw["inputs"].items() if "sha256" in v}
    assert m5["params"] == raw["params"] and m5["counts"] == raw["counts"] and m5["notes"] == raw["notes"]
    assert m5["tools"]["anthropic-sdk"]


def test_views_over_a_missing_or_empty_run(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert views.run_summary(empty)["stages_present"] == [] and views.run_summary(empty)["hpo"] == []
    assert views.candidates_view(empty)["present"] is False
    assert views.chain_view(empty)["present"] is False and views.chain_view(empty)["chains"] == [] and views.chain_view(empty)["failures"] == []
    assert views.medicine_view(empty) == {"present": False, "report": None}
    assert views.join_status(empty) is None and views.failures(empty / "05_reason") == []
    assert all(st["manifest"] is None for st in views.provenance_view(empty)["stages"])
    assert views.evidence_index(empty) == {}
    # a half-written file is reported as absent, never raised on
    (empty / "03_filter").mkdir()
    (empty / "03_filter" / "candidates.json").write_text("{not json")
    assert views.candidates_view(empty)["present"] is False


def test_a_candidate_missing_from_the_join_is_not_called_unranked(run_copy: Path):
    """Stage 3 rerun after stage 4: ``joined.json`` no longer holds TP53 while
    ``ranking.tsv`` still ranks it 2. The page must say the join is stale — never that
    Exomiser did not rank the gene, nor that it is off the shortlist."""
    joined_path = run_copy / "04_rank" / "joined.json"
    joined = json.loads(joined_path.read_text())
    joined["candidates"] = [c for c in joined["candidates"] if c["candidate_id"] != "TP53:het_single"]
    joined_path.write_text(json.dumps(joined))
    c = views.candidates_view(run_copy)
    cftr, tp53 = c["candidates"]
    assert cftr["join"] == "ranked" and cftr["agreement"] == "agrees"
    assert tp53["join"] == "missing" and tp53["exomiser"] is None and tp53["agreement"] == "not_joined"
    assert c["order_agreement"] is True  # only the joined candidates are compared; a missing one is not a disagreement
    assert c["join_check"] == {**c["join_check"], "missing": ["TP53:het_single"], "dropped": [], "stale": True}
    r = views.ranking_view(run_copy)
    assert [(x["gene_symbol"], x["in_shortlist"], x["candidate_id"], x["joined"]) for x in r["rows"]] == \
        [("CFTR", True, "CFTR:comphet", True), ("TP53", True, "TP53:het_single", False)]  # the shortlist as stage 3 holds it now
    html = render_run(run_copy)
    assert_well_formed(html)
    assert "Exomiser did not rank this gene" not in html and "not on shortlist" not in html and ">unranked<" not in html
    assert html.count("not in the stage-4 join") >= 3  # the candidates table, the candidate block, the ranking row
    assert "The stage-4 join is stale" in html and "engine rank --join-only" in html and "TP53:het_single</span> are not in" in html
    assert '<span class="mark warn">not in the stage-4 join</span>' in html.split('<section id="ranking">')[1]


def test_a_shortlist_rewritten_since_stage_4_read_it_is_a_stale_join(run_copy: Path):
    """The other shape of the same problem: every id still joins, but the file stage 4
    read is not the file on disk (a caveat added; a candidate dropped)."""
    path = run_copy / "03_filter" / "candidates.json"
    doc = json.loads(path.read_text())
    doc["candidates"] = [c for c in doc["candidates"] if c["candidate_id"] != "TP53:het_single"]
    doc["candidates"][0]["caveats"].append("added after stage 4 ran")
    path.write_text(json.dumps(doc))
    check = views.join_status(run_copy)
    assert check["stale"] is True and check["missing"] == [] and check["dropped"] == ["TP53:het_single"]
    assert check["candidates_sha256_recorded"] != check["candidates_sha256_current"]
    r = views.ranking_view(run_copy)
    assert [(x["gene_symbol"], x["in_shortlist"], x["joined"]) for x in r["rows"]] == [("CFTR", True, True), ("TP53", False, None)]
    html = render_run(run_copy)
    assert "The stage-4 join is stale" in html and "has changed since stage 4 read it" in html
    assert "are no longer on the shortlist" in html and "not on shortlist" in html.split('<section id="ranking">')[1]  # true now


def test_a_failed_live_stage_is_stated_not_hidden(run_copy: Path):
    """A live stage 5 or 6 that failed on the model writes ``transcripts/<id>.failed.json``
    and stops before its manifest. The page must say so — error and request id — not
    'no manifest' as if nothing had been launched; the model's partial text stays out."""
    for stage, product in (("05_reason", "chains"), ("06_medicine", "report.json")):
        stage_dir = run_copy / stage
        (stage_dir / "manifest.json").unlink()
        target = stage_dir / product
        shutil.rmtree(target) if target.is_dir() else target.unlink()
        (stage_dir / "transcripts" / "CFTR:comphet.json").unlink()
        (stage_dir / "transcripts" / "CFTR:comphet.failed.json").write_text(json.dumps({
            "candidate_id": "CFTR:comphet", "error": "model call failed: 529 overloaded (request id req_abc)", "request_id": "req_abc",
            "transcript": [{"role": "assistant", "text": "half a turn"}], "final_text": "half an answer the model gave", "tools": {}}))
    ch = views.chain_view(run_copy)
    assert ch["manifest"] is False and ch["chains"] == []
    assert ch["failures"] == [{"candidate_id": "CFTR:comphet", "error": "model call failed: 529 overloaded (request id req_abc)",
                               "request_id": "req_abc", "turns": 1, "file": "05_reason/transcripts/CFTR:comphet.failed.json"}]
    m = views.medicine_view(run_copy)
    assert m["report"] is None and m["manifest"] is False and m["candidate_id"] == "CFTR:comphet"
    assert m["failures"][0]["file"] == "06_medicine/transcripts/CFTR:comphet.failed.json"
    by_dir = {st["dir"]: st for st in views.run_summary(run_copy)["stages"]}
    assert by_dir["05_reason"]["failed"] == ["CFTR:comphet"] and by_dir["06_medicine"]["failed"] == ["CFTR:comphet"]
    html = render_run(run_copy)
    assert_well_formed(html)
    chains = html.split('<section id="chains">')[1].split("</section>")[0]
    med = html.split('<section id="medicine">')[1].split("</section>")[0]
    assert 'Stage 5 failed on <span class="id">CFTR:comphet</span>: model call failed: 529 overloaded' in chains and "req_abc" in chains
    assert "no chain was written for it" in chains and "1 turn(s) completed before the failure" in chains
    assert 'Stage 6 failed on <span class="id">CFTR:comphet</span>' in med and "no report was written for it" in med
    assert "06_medicine/report.json</span> in this run" not in med  # the failure is the reason; it is not restated as an absence
    assert html.count('<span class="mark crit">failed</span>') == 2  # the header's stage table
    assert "half an answer" not in html and "half a turn" not in html


# -------------------------------------------------------------------------- render

def test_html_is_self_contained_and_well_formed(full_run: Path):
    html = render_run(full_run)
    assert_well_formed(html)
    assert html.startswith("<!doctype html>") and html.count("<html") == 1 and "<title>Run report · run</title>" in html
    assert FONTS_URL in html and "Spectral" in CSS and "IBM Plex Sans" in CSS and "IBM Plex Mono" in CSS
    assert "Georgia" in CSS and "monospace" in CSS  # real fallbacks
    assert "tabular-nums" in CSS and "--accent: #2A5D7C" in CSS
    assert "@media (prefers-color-scheme: dark)" in CSS and ':root:not([data-theme="light"])' in CSS and ':root[data-theme="dark"]' in CSS
    assert "body { margin: 0; background: var(--bg)" in CSS and "overflow-x: auto" in CSS
    assert html.count("<script") == 1 and html.count("<link") == 3  # the fonts and their preconnects; nothing else external
    assert not re.search(r"<(script|img|iframe)[^>]*src=", html)
    assert not re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", html)  # no emoji, no dingbats
    assert "gradient" not in CSS and "border-radius" not in CSS
    # every HP: id in the document is followed by the label its own hpo: record carries
    head = html.split("Case HPO terms")[1].split("</table>")[0]
    assert ('<a class="id" href="https://hpo.jax.org/browse/term/HP:0012236" rel="noopener">HP:0012236</a> '
            "Elevated sweat chloride") in head
    for term, label in (("HP:0002205", "Recurrent respiratory infections"), ("HP:0002110", "Bronchiectasis")):
        assert f">{term}</a> {label}" in head
    assert "Gene dossier" in html.split("<nav")[1].split("</nav>")[0]  # the dossier step has its own section


def test_html_shows_candidates_ranks_and_agreement(full_run: Path):
    html = render_run(full_run)
    assert "CFTR:comphet" in html and "TP53:het_single" in html
    assert 'id="cand-CFTR:comphet"' in html and 'href="#cand-CFTR:comphet"' in html
    assert '<span class="mark good">agrees</span>' in html
    assert "0.9894" in html and "0.9828" in html  # Exomiser scores as ranking.tsv wrote them
    assert "is also the blind ranker&#x27;s rank 1" in html or "is also the blind ranker's rank 1" in html
    for hit in ("consequence:HIGH", "clinvar:always_keep=Pathogenic", "model:comphet", "model:het_single"):
        assert hit in html
    assert 'title="0.011931254341569912">0.0119</span>' in html  # the AF shortened for the eye, the file's text kept
    for key in (F508DEL, G542X, R175H):
        assert key in html


def test_html_links_every_evidence_id_to_its_record(full_run: Path):
    html = render_run(full_run)
    index = views.evidence_index(full_run)
    chain = json.loads((full_run / "05_reason" / "chains" / "CFTR:comphet.json").read_text())
    cited = {rid for v in chain["variants"] for c in v["criteria"] for rid in c["evidence_ids"]}
    cands = json.loads((full_run / "03_filter" / "candidates.json").read_text())["candidates"]
    cited |= {rid for c in cands for v in c["variants"] for rid in v["evidence_ids"]}
    report = json.loads((full_run / "06_medicine" / "report.json").read_text())
    cited |= {rid for d in report["candidates"] for rid in d["evidence_ids"] + d["trial_ids"]}
    cited |= {"exomiser:CFTR", "exomiser:TP53"}
    assert cited and cited <= set(index)
    for rid in cited:
        assert link(rid, index[rid]["url"].replace("&", "&amp;")) in html, rid
    # inline citations in prose become links too; an ontology term in brackets does not
    assert f"[{link(f'vep:{F508DEL}', index[f'vep:{F508DEL}']['url'].replace('&', '&amp;'))}" in html
    assert "HP:0012236" in html and 'href="HP:' not in html
    # a record cited bare (the spelling the validator leaves) is linked where it stands, and not repeated after the sentence
    report = json.loads((full_run / "06_medicine" / "report.json").read_text())
    assert report["mechanism"][1]["statement"] == f"p.Phe508del is cited bare here, vep:{F508DEL}, as the validator leaves a resolving id"
    bullet = html.split("is cited bare here, ")[1].split("</li>")[0]
    assert bullet.startswith(link(f"vep:{F508DEL}", index[f"vep:{F508DEL}"]["url"].replace("&", "&amp;")) + ", as the validator")
    assert bullet.count("<a ") == 1


def test_html_shows_classification_phase_limits_and_the_validator(full_run: Path):
    html = render_run(full_run)
    assert "Classification (engine-computed)" in html
    assert '<span class="mark crit">pathogenic</span>' in html and ">vus<" in html.replace('class="mark crit">', '>')
    chain = json.loads((full_run / "05_reason" / "chains" / "CFTR:comphet.json").read_text())
    assert "this data cannot show whether they lie on different chromosomes" in html  # the phase statement
    for text in chain["limits"] + chain["what_would_change_the_call"]:
        assert escape(text.split("[")[0].strip()[:60]) in html
    for v in chain["variants"]:
        for c in v["criteria"]:
            assert f'<td class="id">{c["code"]}</td>' in html
            assert escape(c["justification"].split("[")[0].strip()[:50]) in html
    assert "<h4>Validator</h4>" in html and "Rejections (" in html and "Rejections (0)" not in html.split("<h2>Medicine")[0]
    assert "PS3" in html and (FAKE_PMID in html or "not carried by a citable record" in html or "paper" in html)
    assert "case-level; cites no record" in html or "exomiser:CFTR" in html  # PP4: case-level, or cited to the ranker's record


def test_html_medicine_in_rubric_order_and_provenance_sha256s(full_run: Path):
    html = render_run(full_run)
    assert_well_formed(html)
    med = html.split('<section id="medicine">')[1].split("</section>")[0]
    order = [med.index(h) for h in ("<h3>Patient context</h3>", "<h3>Variant mechanism</h3>",
                                     "<h3>Cellular and disease consequence</h3>", "<h3>Intervention classes searched</h3>",
                                     "<h3>Drug candidates</h3>", "<h3>Considered and rejected</h3>", "<h3>Surveillance</h3>",
                                     "<h3>Follow-up experiments</h3>", "<h3>Limits</h3>", "<h3>Secondary findings</h3>",
                                     "<h3>Literature</h3>", "<h3>References</h3>", "<h4>Validator</h4>")]
    assert order == sorted(order)
    assert "Ivacaftor" in med and "p.Gly542Ter produces no protein to potentiate" in med and "never a treatment recommendation" in med
    assert "Elevated sweat chloride" in med and "cystic fibrosis" in med  # the patient context, from records
    assert "<th>class</th><th>acts on</th><th>targets</th><th>verdict</th><th>searches</th><th>records</th>" in med
    assert CLASS in med and '<span class="mark good">candidates proposed</span>' in med
    assert "Approved indication" in med and "Paediatric safety" in med and "Intervention class" in med
    assert "Crofelemer" in med and "none recorded" in med.split("<h3>Secondary findings</h3>")[1]
    assert "survived" not in med  # never said of a list nothing was proposed into
    prov = html.split('<section id="provenance">')[1]
    for stage_dir in ("03_filter", "04_rank", "05_reason", "06_medicine"):
        m = json.loads((full_run / stage_dir / "manifest.json").read_text())
        for entry in list(m["inputs"].values()) + list(m["outputs"].values()):
            if "sha256" in entry:
                assert entry["sha256"] in prov, (stage_dir, entry["path"])
        for note in m["notes"]:
            assert escape(note[:40]) in prov
    assert "<summary>Parameters</summary>" in prov and "system_prompt_sha256" in prov


def test_render_with_only_stages_2_and_3(early_run: Path):
    html = render_run(early_run)
    assert_well_formed(html)
    assert "CFTR:comphet" in html and "TP53:het_single" in html
    assert "stage 4 has not run" in html and "Stage 4 has not run" in html
    assert "Stage 5 has not run" in html and "Stage 6 has not run" in html
    assert '<span class="mark">not run</span>' in html
    assert "blind ranker" not in html.split('<section id="ranking">')[0].split("<table")[1]  # no Exomiser columns without stage 4
    index = views.evidence_index(early_run)
    for rid in CFTR_PAIR_IDS:
        assert link(rid, index[rid]["url"].replace("&", "&amp;")) in html


def test_render_an_empty_run_directory(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    html = render_run(empty)
    assert_well_formed(html)
    assert "Stage 3 has not run" in html and "not run" in html and "Provenance" in html


def test_render_is_deterministic(full_run: Path, tmp_path: Path):
    a = render_run(full_run)
    b = render_run(full_run)
    assert a == b
    out1 = write_report(full_run, tmp_path / "r1.html")
    out2 = write_report(full_run, tmp_path / "r2.html")
    assert out1.read_bytes() == out2.read_bytes()
    assert not re.search(r"20\d\d-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+", a)  # no render-time timestamp anywhere


def test_linker_links_records_and_leaves_terms_alone():
    index = {"vep:1:2:A:G": {"url": "https://x/vep", "source": "vep"}}
    linker = _Linker(index, {"vep", "pmid"})
    assert linker.prose("Seen [vep:1:2:A:G] and [HP:0000001] and [pmid:12] & <b>") == \
        'Seen [<a class="id" href="https://x/vep" rel="noopener">vep:1:2:A:G</a>] and [HP:0000001] and ' \
        '[<span class="id unresolved" title="not in the evidence store of this run">pmid:12</span>] &amp; &lt;b&gt;'
    assert linker.prose("[vep:1:2:A:G, pmid:12]").count("<a ") == 1 and "unresolved" in linker.prose("[vep:1:2:A:G, pmid:12]")
    assert linker.prose(None) == ""


def test_linker_links_bare_citations_the_validator_counts_and_no_others():
    """Whatever ``citation_tokens`` counts as a citation is linked — bare ``source:id``,
    ``PMID n``, a bare ClinVar or trial accession, the known tokens of a mixed bracket
    — and a coordinate, a transcript or an ontology term is not."""
    index = {"pmid:12345": {"url": "https://x/p", "source": "pmid"}, "clinvar:VCV000007105": {"url": "https://x/c", "source": "clinvar"},
             "opentargets:association:ENSG1:MONDO_1": {"url": "https://x/o", "source": "opentargets"}}
    linker = _Linker(index, {"pmid", "clinvar", "vep", "opentargets"})
    pm, cv = link("pmid:12345", "https://x/p"), link("clinvar:VCV000007105", "https://x/c")
    assert linker.linked("Shown functionally, pmid:12345.") == (f"Shown functionally, {pm}.", ["pmid:12345"])
    assert linker.linked("ClinVar lists it as VCV000007105.") == (f"ClinVar lists it as {cv}.", ["clinvar:VCV000007105"])
    assert linker.linked("PMID 12345, PMID:12345 and [pmid:12345]")[0] == f"{pm}, {pm} and [{pm}]"
    assert linker.linked("[HP:0002205, pmid:12345]") == (f"[HP:0002205, {pm}]", ["pmid:12345"])
    assert linker.linked("[see VCV000007105; pmid:12345]") == (f"[see {cv}; {pm}]", ["clinvar:VCV000007105", "pmid:12345"])
    assert linker.linked("[opentargets:association:ENSG1:MONDO_1]")[1] == ["opentargets:association:ENSG1:MONDO_1"]  # an id with an underscore, whole
    plain = "chr7:117559590, NM_000492.4:c.1521_1523del, HP:0002205, rs113993960, [1] and doi:10.1000/x stay text"
    assert linker.linked(plain) == (plain, [])
    # a claim's trailing list leaves out exactly what the sentence links, whichever spelling it used
    evidence = views.resolve(["pmid:12345", "clinvar:VCV000007105"], index)
    for statement in ("Shown functionally, pmid:12345.", "ClinVar lists it as VCV000007105.", "Shown [pmid:12345].", "PMID 12345 shows it."):
        bullet = _claims([{"statement": statement, "evidence": evidence}], linker)
        assert bullet.count(pm) == 1 and bullet.count(cv) == 1 and bullet.count("<a ") == 2, (statement, bullet)
    assert _claims([{"statement": "Says nothing citable.", "evidence": evidence}], linker).count("<a ") == 2


def test_a_redaction_is_a_footnote_in_the_chain_and_in_the_medicine_section(run_copy: Path):
    """The validator leaves ``[^k]`` in the prose and the reason in its record; the page
    prints the marker as a reference and the reason as a footnote item — never the
    sentence ``[citation removed: no such record]`` in the middle of a claim."""
    report_path = run_copy / "06_medicine" / "report.json"
    report = json.loads(report_path.read_text())
    report["limits"] = ["No paper was retrieved in this run[^1]"]
    report_path.write_text(json.dumps(report, sort_keys=True, indent=1) + "\n")
    validation_with_a_marker(run_copy)
    chain_path = run_copy / "05_reason" / "chains" / "CFTR:comphet.json"
    chain = json.loads(chain_path.read_text())
    chain["mechanism_hypothesis"] += " A paper the validator could not resolve[^1]."
    chain_path.write_text(json.dumps(chain, sort_keys=True, indent=1) + "\n")
    chain_validation = run_copy / "05_reason" / "validation" / "CFTR:comphet.json"
    doc = json.loads(chain_validation.read_text())
    doc["rejections"].append({"path": "mechanism_hypothesis", "marker": 1,
                              "reason": "inline PMID not in the store: pmid:99999999"})
    chain_validation.write_text(json.dumps(doc, sort_keys=True, indent=1) + "\n")

    html = render_run(run_copy)
    assert_well_formed(html)
    assert "[^1]" not in html and "citation removed: no such record" not in html
    med = html.split('<section id="medicine">')[1].split("</section>")[0]
    assert '<sup class="fn"><a href="#fn-medicine-1">1</a></sup>' in med
    assert ('<ol class="footnotes"><li id="fn-medicine-1">citation removed by the validator: inline citation to a '
            'record not in the store: pmid:99999999</li></ol>') in med
    chains = html.split('<section id="chains">')[1].split('<section id="medicine">')[0]
    assert '<sup class="fn"><a href="#fn-CFTR:comphet-1">1</a></sup>' in chains
    assert ('<li id="fn-CFTR:comphet-1">citation removed by the validator: inline PMID not in the store: '
            'pmid:99999999</li>') in chains
    # a marker no rejection carries (one the model wrote itself) still gets a footnote
    doc["rejections"] = [r for r in doc["rejections"] if r.get("marker") != 1]
    chain_validation.write_text(json.dumps(doc, sort_keys=True, indent=1) + "\n")
    html = render_run(run_copy)
    assert '<li id="fn-CFTR:comphet-1">citation removed by the validator (reason in the validation record)</li>' in html


def test_the_html_says_no_candidate_was_proposed_and_names_the_classes(run_copy: Path):
    report_path = run_copy / "06_medicine" / "report.json"
    report = json.loads(report_path.read_text())
    report["candidates"] = []
    report_path.write_text(json.dumps(report, sort_keys=True, indent=1) + "\n")
    med = render_run(run_copy).split('<section id="medicine">')[1].split("</section>")[0]
    assert f"No candidate proposed. Classes searched: {escape(CLASS)} (candidates proposed)" in med
    assert "survived" not in med
    report["intervention_classes"] = []
    report_path.write_text(json.dumps(report, sort_keys=True, indent=1) + "\n")
    med = render_run(run_copy).split('<section id="medicine">')[1].split("</section>")[0]
    assert "No candidate proposed and no intervention class was searched: the ladder was not walked (see the manifest)." in med
    assert "No intervention class was searched." in med


def test_the_html_shows_a_secondary_finding_as_a_finding_not_a_target(run_copy: Path):
    report_path = run_copy / "06_medicine" / "report.json"
    report = json.loads(report_path.read_text())
    report["secondary_findings"] = [{"candidate_id": "TP53:het_single", "gene_symbol": "TP53", "model": "het_single",
                                     "classifications": {R175H: "likely_pathogenic"},
                                     "note": "a secondary finding, not a repurposing target; disclosure is the clinical "
                                             "team's decision under the study's recontact rules"}]
    report_path.write_text(json.dumps(report, sort_keys=True, indent=1) + "\n")
    view = views.medicine_view(run_copy)["report"]["secondary_findings"]
    assert view[0]["candidate_id"] == "TP53:het_single" and view[0]["classifications"] == {R175H: "likely_pathogenic"}
    med = render_run(run_copy).split('<section id="medicine">')[1].split("</section>")[0]
    findings = med.split("<h3>Secondary findings</h3>")[1]
    assert "TP53:het_single" in findings and "likely pathogenic" in findings
    assert "not a repurposing target" in findings


# ----------------------------------------------------------------------------- CLI

def test_cli_writes_the_default_and_a_chosen_path(full_run: Path, tmp_path: Path):
    assert commands == [report] and report_command is report and report.name == "report"
    runner = CliRunner()
    result = runner.invoke(report, ["--run", str(full_run)])
    assert result.exit_code == 0, result.output
    default = full_run / "report.html"
    assert default.exists() and default.read_text(encoding="utf-8") == render_run(full_run)
    assert "02_retrieve, 03_filter, 04_rank, 05_reason, 06_medicine" in result.output and str(default) in result.output
    for leak in (F508DEL, G542X, R175H, "CFTR", "TP53"):
        assert leak not in result.output  # counts and paths only on the terminal
    out = tmp_path / "elsewhere" / "r.html"
    result = runner.invoke(report, ["--run", str(full_run), "--out", str(out)])
    assert result.exit_code == 0 and out.read_bytes() == default.read_bytes()
    result = runner.invoke(report, ["--run", str(full_run), "--out", str(out), "--top", "5"])
    assert result.exit_code == 0 and "Showing the top 5 of" in out.read_text(encoding="utf-8")
    default.unlink()


def test_cli_refuses_a_missing_run(tmp_path: Path):
    result = CliRunner().invoke(report, ["--run", str(tmp_path / "missing")])
    assert result.exit_code != 0 and "does not exist" in result.output


@pytest.mark.xfail("report" not in engine.cli.main.commands, strict=True,
                   reason="engine.cli.STAGE_PACKAGES has no 'report' entry, so `engine report` is not registered: "
                          "add `\"report\": (\"report\", \"report\", \"P7\")` to STAGE_PACKAGES in src/engine/cli.py "
                          "(a file the report package does not own); this test then runs for real")
def test_engine_cli_registers_report(full_run: Path, tmp_path: Path):
    """The contract's entry point is ``engine report --run <dir> [--out <file>]`` — a
    sub-command of ``engine.cli.main``, not only a click object in this package."""
    main = engine.cli.main
    assert "report" in main.commands and main.commands["report"] is report
    out = tmp_path / "r.html"
    result = CliRunner().invoke(main, ["report", "--run", str(full_run), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == render_run(full_run)
    assert "report" in CliRunner().invoke(main, ["--help"]).output


# ------------------------------------------------------------ the public run, if present

@pytest.mark.skipif(not (PUBLIC_RUN / "03_filter" / "candidates.json").exists(),
                    reason="no public demo run; run scripts/run_public_case.sh (PUBLIC_RUN_DIR to point elsewhere)")
def test_public_run_renders_as_recorded():
    """The directory ``scripts/run_public_case.sh`` writes: stages 1–4 for real (4 only
    with Docker), stage 5 dry with the recorded chain copied in, stage 6 dry. Public
    data only. The directory is shared, so a run still being written is skipped."""
    if not (PUBLIC_RUN / "06_medicine" / "manifest.json").exists():
        pytest.skip("public demo run is incomplete or being rewritten")
    s = views.run_summary(PUBLIC_RUN)
    assert s["sample"] == "PUBLIC01" and s["hpo"] == HPO
    c = views.candidates_view(PUBLIC_RUN)
    assert [x["candidate_id"] for x in c["candidates"]] == ["CFTR:comphet", "TP53:het_single"]
    if c["rank_present"]:
        ranking = {r["gene_symbol"]: r["rank"] for r in views.ranking_view(PUBLIC_RUN)["rows"]}
        assert c["candidates"][0]["exomiser"]["rank"] == int(ranking["CFTR"]) == 1
        assert c["candidates"][0]["agreement"] == "agrees"
    ch = views.chain_view(PUBLIC_RUN)
    if ch["chains"]:
        chain = ch["chains"][0]
        assert [(v["key"], v["classification"]) for v in chain["variants"]] == [(F508DEL, "vus"), (G542X, "pathogenic")]
        if ch["dry_run"]:
            assert chain["claimed_by_manifest"] is False and "dry run" in chain["manifest_note"]
    html = render_run(PUBLIC_RUN)
    assert_well_formed(html)
    assert html == render_run(PUBLIC_RUN)
    index = views.evidence_index(PUBLIC_RUN)
    for rid in CFTR_PAIR_IDS:
        assert link(rid, index[rid]["url"].replace("&", "&amp;")) in html
    for stage_dir in s["stages_present"]:
        m = json.loads((PUBLIC_RUN / stage_dir / "manifest.json").read_text())
        for entry in m["inputs"].values():
            if "sha256" in entry:
                assert entry["sha256"] in html
    if ch["chains"] and ch["dry_run"]:
        assert "Not claimed by the stage-5 manifest" in html
    if views.medicine_view(PUBLIC_RUN).get("dry_run"):
        assert "Stage 6 ran dry" in html
