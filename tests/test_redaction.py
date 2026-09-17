"""Redaction markers, footnotes and near-miss respelling (engine.agents.redaction),
and the stage-5 checks that number their redactions through the same counter.

Network-free. The evidence store under tests/fixtures/agents/evidence/ (public
records only) and in-memory indexes built from it; the ids below are the public
CFTR, MTHFR and TP53 records and two papers.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from engine.agents.redaction import (
    FOOTNOTE_UNKNOWN, Redactions, footnotes, markers, near_misses, prefixed, respell, respellable, scan, unwrap,
)
from engine.agents.schema import EvidenceChain
from engine.agents.validator import EvidenceIndex, Rejection, citation_tokens, validate
from engine.reason.checks import AccessionResolver, accession_tokens, stage_checks
from engine.retrieve.store import EvidenceRecord, EvidenceStore

FIXTURES = Path(__file__).parent / "fixtures" / "agents" / "evidence"
CFTR_VEP = "vep:7:117559590:ATCT:A"
CFTR_GNOMAD = "gnomad:7-117559590-ATCT-A"
CFTR_CLINVAR = "clinvar:VCV000007105"
MTHFR_GNOMAD = "gnomad:1-11796321-G-A"
PAPER = "pmid:7647779"


def record(rid: str) -> EvidenceRecord:
    return EvidenceRecord(record_id=rid, source=rid.split(":", 1)[0], source_version="test", query={}, url="https://x",
                          retrieved_at="2026-01-01T00:00:00+00:00", payload={})


@pytest.fixture
def store() -> EvidenceStore:
    return EvidenceStore(FIXTURES)


# ---------------------------------------------------------------------- respelling

@pytest.mark.parametrize("written, stored, expected", [
    ("gnomad:15-1-T:G", "gnomad:15-1-T-G", True),                       # one separator substituted
    ("vep:7:117559590:ATCT:A", "vep:7-117559590-ATCT-A", False),          # three separator edits
    ("pmid:7647779", "pmid:7647778", False),                              # a digit: another paper
    ("clinvar:vcv000007105", "clinvar:VCV000007105", True),               # case only
    ("GNOMAD:7-117559590-ATCT-A", "gnomad:7-117559590-ATCT-A", True),
    ("clinvar:VCV7105", "clinvar:VCV000007105", False),                   # digits missing, not a separator
    ("gnomad:7-117559590-ATCT-A", "gnomad:7-117559590-ATCT-A", True),     # equal
    ("gnomad:7_117559590-ATCT-A", "gnomad:7-117559590-ATCT-A", True),
    ("gnomad:7-117559590-ATCT-A-", "gnomad:7-117559590-ATCT-A", True),    # one separator appended
    ("gnomad:7-117559590ATCT-A", "gnomad:7-117559590-ATCT-A", True),      # one separator deleted
    ("gnomad:7--117559590-ATCT-A", "gnomad:7-117559590-ATCT-A", True),    # one separator inserted
    ("gnomad:7-117559590-ATCT-AT", "gnomad:7-117559590-ATCT-A", False),   # a letter appended
    ("gnomad:7-117559590-ATCT-", "gnomad:7-117559590-ATCT-A", False),     # a letter missing
    ("gnomad:7-117559590-ATCT:A", "gnomad:7-117559590-ACTT-A", False),    # a separator and a letter
    ("gnomad:7-117559590-ATCT-A", "gnomad:7-11755959-ATCT-A", False),     # a digit missing
    ("pmid:7647779", "pmid:76477790", False),
    ("", "", True),
    ("a", "", False),
])
def test_respellable_truth_table(written: str, stored: str, expected: bool):
    assert respellable(written, stored) is expected
    assert respellable(stored, written) is expected  # symmetric


def test_respell_needs_exactly_one_near_miss_of_the_same_source(store: EvidenceStore):
    index = EvidenceIndex([store])
    assert respell("gnomad:7-117559590-ATCT:A", index) == CFTR_GNOMAD
    assert respell("GNOMAD:7:117559590-ATCT-A", index) == CFTR_GNOMAD
    assert respell("gnomad:7-117559590-ATCT-T", index) is None                       # a letter: another variant
    assert respell("pmid:7647778", index) is None and respell("pmid:7647779", index) == PAPER
    assert respell("vep:7-117559590-ATCT-A", index) is None                          # three edits from the vep: id
    assert respell("clinvar:7-117559590-ATCT:A", index) is None                      # right shape, other source
    assert respell("nothing:7-117559590-ATCT:A", index) is None
    # two candidates one edit away each: no guess
    two = EvidenceIndex.from_records([record("gnomad:1-5-A-T"), record("gnomad:1-5-A.T"), record("gnomad:1-5-A-TT")])
    assert near_misses("gnomad:1-5-A:T", two) == ["gnomad:1-5-A-T", "gnomad:1-5-A.T"]
    assert respell("gnomad:1-5-A:T", two) is None
    assert respell("gnomad:1-5-A-TT-", two) == "gnomad:1-5-A-TT"
    # in-memory records first, then the store — each id once
    both = EvidenceIndex([store], records=[record("gnomad:1-5-A-T")])
    assert both.ids_of("gnomad") == ["gnomad:1-5-A-T", MTHFR_GNOMAD, "gnomad:17-7675088-C-T", CFTR_GNOMAD]
    assert respell("gnomad:1-5-A:T", both) == "gnomad:1-5-A-T"


# ------------------------------------------------------------------------- markers

def test_redactions_number_in_order_and_continue_after_what_the_object_carries():
    r = Redactions()
    assert r.next == 1 and r.rejections == []
    assert r.redact("phase_statement", "why") == "[^1]" and r.redact("limits[0]", "again") == "[^2]"
    assert r.rejections == [Rejection("phase_statement", "why", marker=1), Rejection("limits[0]", "again", marker=2)]
    assert r.next == 3
    assert asdict(r.rejections[0]) == {"path": "phase_statement", "reason": "why", "marker": 1}
    assert Rejection("p", "r").marker is None  # a dropped item carries no marker

    carrying = {"variants": [{"summary": "see [^2] and [^1]", "criteria": [{"justification": "x"}]}], "limits": ["[^1] again"]}
    assert scan(carrying) == [("variants[0].summary", 2), ("variants[0].summary", 1), ("limits[0]", 1)]
    assert Redactions.continuing(carrying).next == 3
    assert Redactions.continuing({"a": ["[^7]"]}).next == 8 and Redactions.continuing({"a": "none"}).next == 1
    assert Redactions.continuing([]).next == 1 and Redactions.continuing("[^3] text [^12]").next == 13
    chain = EvidenceChain(candidate_id="c", variants=[], phase_statement="p [^4]", mechanism_hypothesis="m",
                          limits=[], what_would_change_the_call=[], literature=[])
    assert Redactions.continuing(chain).next == 5 and scan(chain) == [("phase_statement", 4)]
    assert Redactions.continuing("[^0]").next == 1  # a model's [^0] is left alone; numbering starts at 1


def test_markers_prefixed_and_unwrap():
    text = "a [^3] b [^1] c [^3] d [^12] [^x] [1] [^ 2]"
    assert markers(text) == [3, 1, 12] and markers("nothing") == []
    assert prefixed(text, "2-") == "a [^2-3] b [^2-1] c [^2-3] d [^2-12] [^x] [1] [^ 2]"
    assert prefixed(text, "") == text
    assert unwrap("cf. [[^1]] and [[^2]]; [^3] [[x]]") == "cf. [^1] and [^2]; [^3] [[x]]"
    # the validator's tokenizer does not read a marker as a citation
    assert citation_tokens("see [^1] and [^12] and [pmid:7647779]") == [PAPER]


def test_footnotes_from_objects_dicts_and_missing_rejections():
    rejections = [Rejection("p", "dropped item"), Rejection("q", "no such paper", marker=2),
                  {"path": "r", "reason": "no such trial", "marker": 3}, {"path": "s", "reason": "dup", "marker": 3}]
    texts = ["see [^3] and [^2]", "and [^1] again [^3]", "nothing"]
    assert footnotes(texts, rejections) == [
        "[^3]: citation removed by the validator: no such trial",
        "[^2]: citation removed by the validator: no such paper",
        f"[^1]: {FOOTNOTE_UNKNOWN}",
    ]
    assert footnotes(texts, rejections, prefix="1-") == [
        "[^1-3]: citation removed by the validator: no such trial",
        "[^1-2]: citation removed by the validator: no such paper",
        f"[^1-1]: {FOOTNOTE_UNKNOWN}",
    ]
    assert footnotes(texts, []) == [f"[^3]: {FOOTNOTE_UNKNOWN}", f"[^2]: {FOOTNOTE_UNKNOWN}", f"[^1]: {FOOTNOTE_UNKNOWN}"]
    assert footnotes(["plain"], rejections) == [] and footnotes([], rejections) == []
    assert footnotes(["[^4]"], [{"marker": 4, "reason": ""}]) == [f"[^4]: {FOOTNOTE_UNKNOWN}"]  # an empty reason is no reason


# ---------------------------------------------------------------- stage-5 checks

def test_stage_checks_number_their_redactions_and_the_validator_continues(store: EvidenceStore):
    """The stage's accession check writes [^1]…; the validator, told nothing, scans
    the chain and continues at the next number; one report carries both."""
    resolver = AccessionResolver([store.get(CFTR_VEP), store.get(CFTR_CLINVAR)])
    chain = EvidenceChain.model_validate({
        "candidate_id": "CFTR:hom",
        "variants": [{"key": "7:117559590:ATCT:A", "summary": "As VCV000007105 and [VCV999999999], PMC9999999.",
                      "criteria": [{"code": "PM4", "strength": "moderate", "met": True, "justification": "in-frame [SCV000000001]",
                                    "evidence_ids": [CFTR_VEP]}]}],
        "phase_statement": "n/a", "mechanism_hypothesis": f"LoF [{CFTR_VEP}] [pmid:99999999]",
        "limits": ["pubmed 87654321."], "what_would_change_the_call": [], "literature": [],
    })
    assert accession_tokens(chain.variants[0].summary) == ["VCV000007105", "VCV999999999", "PMC9999999"]
    out = stage_checks(chain, resolver)
    assert out.chain.variants[0].summary == "As VCV000007105 and [^1], [^2]."   # brackets the model wrote are not doubled
    assert out.chain.variants[0].criteria[0].justification == "in-frame [^3]"
    assert out.chain.limits == ["[^4]."]  # the sentence keeps its full stop
    assert [(r.path, r.marker) for r in out.redacted] == [("variants[0].summary", 1), ("variants[0].summary", 2),
                                                          ("variants[0].criteria[0].justification", 3), ("limits[0]", 4)]
    assert out.redacted[0].reason == "bare accession not carried by any citable record: VCV999999999"
    assert out.dropped == []

    counter = Redactions.continuing(chain)
    assert counter.next == 1
    again = stage_checks(chain, resolver, counter)
    assert again.redacted is counter.rejections and counter.next == 5

    cleaned, report = validate(out.chain, EvidenceIndex([store]))
    assert cleaned.mechanism_hypothesis == f"LoF [{CFTR_VEP}] [^5]"
    assert [r.marker for r in report.rejections] == [5] and report.counts["redactions"] == 1
    assert cleaned.variants[0].summary == "As VCV000007105 and [^1], [^2]."  # untouched by the validator
    assert Redactions.continuing(cleaned).next == 6
