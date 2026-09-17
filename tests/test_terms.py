"""The HPO term check (``engine.agents.terms``) over an in-memory index of the recorded
public term records (``tests/fixtures/hpo/records``): the three attachment forms, the
stop-word rule, redaction of an id no record carries, the fields it skips, the paths it
spells and the counts it keeps. No network, no run directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from engine.agents.terms import DEFAULT_SKIP, DISPUTED_LABEL, HPO_TOKEN, STOP_WORDS, TermCheck, attached_phrase, check_terms
from engine.agents.validator import Dispute, EvidenceIndex, Rejection
from engine.retrieve.store import EvidenceStore

RECORDS = Path(__file__).parent / "fixtures" / "hpo" / "records"
SWEAT = "HP:0012236"       # Elevated sweat chloride; synonyms Elevated sweat Cl, Elevated sweat Cl-, Elevated sweat chloride
LUNG = "HP:0006528"        # Chronic lung disease
RESP = "HP:0002205"        # Recurrent respiratory infections; synonyms include "Frequent respiratory infections"
UNKNOWN = "HP:0000252"     # microcephaly: a real term, no record in scope
GONE = "[citation removed: no such record]"


def index() -> EvidenceIndex:
    return EvidenceIndex.from_records(EvidenceStore(RECORDS).iter())


def redact(path: str, reason: str) -> str:
    return GONE


def run(data: dict[str, Any], **kw: Any) -> TermCheck:
    return check_terms(data, index(), redact=redact, **kw)


def mark(hpo_id: str, label: str, written: str) -> str:
    return DISPUTED_LABEL.format(record_id=f"hpo:{hpo_id}", label=label, written=written)


# ------------------------------------------------------------------------ the forms

def test_form_a_parenthesised_label_matches_or_is_disputed():
    data = {"summary": f"{SWEAT} (Elevated sweat chloride) and {LUNG} (chronic lung disease) and {RESP} (frequent respiratory infections)"}
    out = run(data)
    assert data["summary"].count("[DISPUTED") == 0 and out.disputes == [] and out.rejections == []
    assert out.counts == {"hp_ids_checked": 3, "hp_ids_redacted": 0, "hp_labels_checked": 3, "hp_labels_disputed": 0}

    data = {"summary": f"the case lists {SWEAT} (hearing loss) too."}
    out = run(data)
    assert data["summary"] == f"the case lists {SWEAT}{mark(SWEAT, 'Elevated sweat chloride', 'hearing loss')} (hearing loss) too."
    assert out.disputes == [Dispute("summary", "HPO", f"hpo:{SWEAT}", True, False, None,
                                    f'the text "hearing loss" attached to {SWEAT} is not the label of hpo:{SWEAT} '
                                    '("Elevated sweat chloride") or one of its synonyms')]
    assert out.notes == [f'summary: {SWEAT} label disputed; the record spells it "Elevated sweat chloride" '
                         "(synonyms: Elevated sweat Cl; Elevated sweat Cl-; Elevated sweat chloride)"]
    assert out.counts["hp_labels_checked"] == 1 and out.counts["hp_labels_disputed"] == 1


def test_form_b_label_after_the_id_up_to_the_next_delimiter():
    good = (f"{SWEAT} Elevated sweat chloride [hpo:{SWEAT}]; {LUNG}: Chronic lung disease, {RESP} — Frequent respiratory infections "
            f"and {SWEAT} elevated sweat Cl- (a synonym) or {LUNG} - Chronic lung disease of the child.\n{RESP} Recurrent respiratory infections")
    data = {"summary": good}
    out = run(data)
    assert data["summary"] == good and out.disputes == []
    assert out.counts == {"hp_ids_checked": 6, "hp_ids_redacted": 0, "hp_labels_checked": 6, "hp_labels_disputed": 0}
    # a longer phrase that starts with the label still matches; one that does not is disputed in place, nothing else moves
    data = {"summary": f"{LUNG} Chronic lung disease with bronchiectasis; {SWEAT} Sweat chloride elevation [hpo:{SWEAT}]."}
    out = run(data)
    assert data["summary"] == (f"{LUNG} Chronic lung disease with bronchiectasis; "
                               f"{SWEAT}{mark(SWEAT, 'Elevated sweat chloride', 'Sweat chloride elevation')} Sweat chloride elevation [hpo:{SWEAT}].")
    assert [d.path for d in out.disputes] == ["summary"] and out.counts["hp_labels_disputed"] == 1
    # a bracketed id is separated from what follows by its bracket: no form applies
    data = {"s": f"[{LUNG}] Chronic lung disease; [{SWEAT}] hearing loss"}
    assert run(data).counts["hp_labels_checked"] == 0 and "[DISPUTED" not in data["s"]


def test_form_c_label_before_the_parenthesised_id():
    data = {"limits": ["Elevated sweat chloride (HP:0012236) is present.", "The case shows chronic lung disease (HP:0006528), "
                       "recurrent respiratory infections (HP:0002205) and frequent respiratory infections (HP:0002205)."]}
    out = run(data)
    assert "[DISPUTED" not in "".join(data["limits"]) and out.disputes == []
    assert out.counts == {"hp_ids_checked": 4, "hp_ids_redacted": 0, "hp_labels_checked": 4, "hp_labels_disputed": 0}
    # a sentence ending in the label matches (the label is the tail); one ending in something else is disputed
    data = {"limits": ["The proband has documented elevated sweat chloride (HP:0012236).", "The proband has hearing loss (HP:0012236)."]}
    out = run(data)
    assert data["limits"][0] == "The proband has documented elevated sweat chloride (HP:0012236)."
    assert data["limits"][1] == f"The proband has hearing loss (HP:0012236{mark(SWEAT, 'Elevated sweat chloride', 'proband has hearing loss')})."
    assert [d.path for d in out.disputes] == ["limits[1]"]
    # more than eight words before the parenthesis is prose, not a label; so is a phrase cut by a delimiter
    data = {"s": "Sweat chloride was 98 mmol/L on two occasions in this child (HP:0012236); high, (HP:0012236)"}
    out = run(data)
    assert "[DISPUTED" not in data["s"] and out.counts["hp_labels_checked"] == 0 and out.counts["hp_ids_checked"] == 2


def test_a_phrase_opening_with_a_stop_word_is_prose_not_a_label():
    for text in (f"{SWEAT} was reported in the case", f"{SWEAT} is not a hearing-loss term", f"{LUNG}, which the case lists",
                 f"{RESP} and {LUNG} are both listed", f"{SWEAT} — the sweat test — is high", f"{SWEAT} [hpo:{SWEAT}] is present"):
        data = {"s": text}
        out = run(data)
        assert data["s"] == text, text
        assert out.counts["hp_labels_checked"] == 0 and out.disputes == [], text
    # any other word after the id is read as a label — the prompt asks for `HP:nnnnnnn <label>` — and held to the record
    data = {"s": f"{RESP} and {LUNG} both fit"}
    out = run(data)
    assert data["s"] == f"{RESP} and {LUNG}{mark(LUNG, 'Chronic lung disease', 'both fit')} both fit" and out.counts["hp_labels_disputed"] == 1
    assert {"is", "was", "which", "and", "the", "a", "not", "than"} <= STOP_WORDS
    assert attached_phrase(f"{SWEAT} Elevated sweat chloride", next(HPO_TOKEN.finditer(f"{SWEAT} Elevated sweat chloride"))) == "Elevated sweat chloride"


# ---------------------------------------------------------------------- redaction

def test_an_id_no_record_carries_is_redacted_with_its_brackets():
    data = {"variants": [{"summary": f"See [{UNKNOWN}] and {UNKNOWN} (microcephaly); also {UNKNOWN}: microcephaly.",
                          "criteria": [{"code": "PP4", "justification": f"the case lists no {UNKNOWN} term", "evidence_ids": [UNKNOWN]}]}],
            "limits": [f"[{SWEAT}] stays"]}
    out = run(data)
    assert data["variants"][0]["summary"] == f"See {GONE} and {GONE} (microcephaly); also {GONE}: microcephaly."
    assert data["variants"][0]["criteria"][0]["justification"] == f"the case lists no {GONE} term"
    assert data["variants"][0]["criteria"][0]["evidence_ids"] == [UNKNOWN]  # skipped: the validator's field
    assert data["limits"] == [f"[{SWEAT}] stays"]
    reason = f"HP id not carried by any hpo: record in scope: {UNKNOWN}"
    assert out.rejections == [Rejection("variants[0].summary", reason)] * 3 + [Rejection("variants[0].criteria[0].justification", reason)]
    assert out.counts == {"hp_ids_checked": 5, "hp_ids_redacted": 4, "hp_labels_checked": 0, "hp_labels_disputed": 0}
    assert out.disputes == [] and out.notes == []
    # the redaction text is the caller's, with the path and the reason it was asked for
    asked: list[tuple[str, str]] = []
    data = {"s": f"{UNKNOWN} here"}
    check_terms(data, index(), redact=lambda p, r: (asked.append((p, r)), f"[^{len(asked)}]")[1])
    assert data["s"] == "[^1] here" and asked == [("s", reason)]


def test_a_known_id_without_a_phrase_is_left_alone_and_hpo_citations_are_not_tokens():
    text = f"{SWEAT}, {LUNG}; [{RESP}] [hpo:{SWEAT}] hpo:{LUNG} HP:12 HP:00122360 xHP:0012236 (HP:0006528)"
    data = {"s": text}
    out = run(data)
    assert data["s"] == text
    assert out.counts == {"hp_ids_checked": 4, "hp_ids_redacted": 0, "hp_labels_checked": 0, "hp_labels_disputed": 0}
    assert [m.group(1) or m.group(2) for m in HPO_TOKEN.finditer(text)] == [SWEAT, LUNG, RESP, LUNG]


# ------------------------------------------------------------------- walk and paths

def test_skip_fields_are_untouched_and_paths_are_spelled_as_the_validator_spells_them():
    data = {
        "candidate_id": f"{UNKNOWN}:x", "key": UNKNOWN, "code": UNKNOWN, "literature": [UNKNOWN], "trial_ids": [UNKNOWN],
        "patient_context": {"hpo": [{"id": UNKNOWN, "label": "microcephaly"}]},
        "variants": [{"key": UNKNOWN, "summary": f"{UNKNOWN} one", "criteria": [{"justification": f"{UNKNOWN} two", "evidence_ids": [UNKNOWN]}]}],
        "limits": ["fine", f"{UNKNOWN} three"],
        "candidates": [{"rationale": "ok"}, {"rationale": f"{UNKNOWN} four", "counter_arguments": [f"{UNKNOWN} five"]}],
        "nested": {"deep": {"text": f"{SWEAT} hearing loss"}},
        "number": 3, "flag": None,
    }
    out = run(data)
    assert [r.path for r in out.rejections] == ["variants[0].summary", "variants[0].criteria[0].justification", "limits[1]",
                                                "candidates[1].rationale", "candidates[1].counter_arguments[0]"]
    assert [d.path for d in out.disputes] == ["nested.deep.text"]
    assert data["candidate_id"] == f"{UNKNOWN}:x" and data["key"] == UNKNOWN and data["code"] == UNKNOWN
    assert data["literature"] == [UNKNOWN] and data["trial_ids"] == [UNKNOWN] and data["patient_context"] == {"hpo": [{"id": UNKNOWN, "label": "microcephaly"}]}
    assert data["variants"][0]["key"] == UNKNOWN and data["variants"][0]["criteria"][0]["evidence_ids"] == [UNKNOWN]
    assert data["limits"] == ["fine", f"{GONE} three"] and data["candidates"][0] == {"rationale": "ok"}
    assert data["nested"]["deep"]["text"] == f"{SWEAT}{mark(SWEAT, 'Elevated sweat chloride', 'hearing loss')} hearing loss"
    assert out.counts == {"hp_ids_checked": 6, "hp_ids_redacted": 5, "hp_labels_checked": 1, "hp_labels_disputed": 1}
    assert DEFAULT_SKIP == ("evidence_ids", "trial_ids", "literature", "candidate_id", "key", "code", "patient_context")
    # a narrower skip walks the rest
    data = {"key": f"{UNKNOWN} x", "literature": [f"{UNKNOWN} y"]}
    out = run(data, skip=("literature",))
    assert data == {"key": f"{GONE} x", "literature": [f"{UNKNOWN} y"]} and [r.path for r in out.rejections] == ["key"]


def test_an_empty_or_clean_object_counts_nothing():
    data: dict[str, Any] = {"summary": "no terms here", "list": [], "obj": {}}
    out = run(data)
    assert data == {"summary": "no terms here", "list": [], "obj": {}}
    assert out == TermCheck()
