"""The HPO term check — every ``HP:`` id the model writes must be a record, and every
label it attaches must be the record's.

The validator (:mod:`engine.agents.validator`) exempts ontology terms on purpose: a
bracketed ``[HP:0002205]`` is not a citation of a record, so it is neither resolved
nor redacted. That left one hole on the first live run: the model was handed the
case terms as bare ids, attached a label from memory to one of them, misremembered,
and the wrong label reached the report unflagged. Now the case terms are records
(``hpo:HP:nnnnnnn``, :mod:`engine.retrieve.hpo`) whose payload spells the label and
its synonyms, and this module — a *stage check*, run by stages 5 and 6 (and the
dossier) after their own checks and before the validator — walks every string of the
answer and holds each ``HP:`` token to two rules:

1. **Carried.** The id must be one an ``hpo:`` record in the candidate's citation
   scope carries. One that is not is replaced — with its brackets, when the model
   bracketed it — by whatever the caller's ``redact`` returns, and rejected on the
   record. The prompt tells the model to state a phenotype the case does not list in
   words, never by an id it was not shown.
2. **Labelled as the record says.** A phrase the model attached to the id — in one of
   three shapes only: ``HP:nnnnnnn (phrase)``, ``HP:nnnnnnn phrase`` up to the next
   delimiter, or ``phrase (HP:nnnnnnn)`` — must equal, extend or end with the
   record's label or one of its synonyms, compared in
   :func:`~engine.retrieve.hpo.normalise_label` form. A phrase that matches nothing
   gets a ``[DISPUTED — …]`` mark right after the id (no other surgery on the text)
   and a :class:`~engine.agents.validator.Dispute`, so the reader sees the record's
   label beside the model's. A phrase after the id that starts with a function word
   (``HP:0012236 was reported …``) is prose, not a label, and is left alone; before
   the id (``the case shows hearing loss (HP:0012236)``) the function words are the
   sentence leading up to the label and are dropped before the comparison.

Paths are spelled as the validator spells them (``variants[0].summary``,
``limits[2]``, ``candidates[1].rationale``), so the rejections and disputes fold into
the same report. Fields named in ``skip`` — citation lists, ids, keys, the
engine-filled patient context — are never walked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from engine.agents.validator import Dispute, Rejection
from engine.retrieve.hpo import label_of, normalise_label, synonyms_of

HPO_TOKEN = re.compile(r"\[(HP:\d{7})\]|(?<![\w:])(HP:\d{7})\b")
"""An ``HP:`` id in prose: bracketed (consumed with its brackets) or bare. ``hpo:HP:…``
is a record citation and the validator's business, so a colon before ``HP`` excludes it."""
STOP_WORDS = frozenset("is was were are be been has have had which that and or but in of for with as at on to from by "
                       "not no does do did can may might would could should will an a the this these those it its than "
                       "then".split())
"""A phrase whose first word is one of these is a sentence continuing after the id,
not a label attached to it."""
DISPUTED_LABEL = '[DISPUTED — {record_id} names this term "{label}"; the adjacent text "{written}" is not its label or a synonym]'
MAX_PHRASE_WORDS = 8
DEFAULT_SKIP = ("evidence_ids", "trial_ids", "literature", "candidate_id", "key", "code", "patient_context")

_FORM_A = re.compile(r"\s*\(([^()\n]*)\)")
"""``HP:nnnnnnn (phrase)`` — the text after the token."""
_FORM_B_SEP = re.compile(r"[ :—–-]+")
"""``HP:nnnnnnn phrase``: the separator run between the id and the phrase."""
_FORM_B_END = re.compile(r"[,;.)\]\[\n]| and | or |HP:")
"""Where a form-(b) phrase ends: the first delimiter, conjunction or next id."""
_FORM_C_START = re.compile(r"[,;.:()\[\n]")
"""Where a form-(c) phrase (``phrase (HP:nnnnnnn)``) begins, scanning back: a
delimiter, an earlier parenthesis (open or closed — a phrase never spans one) or
the line start."""


@dataclass
class TermCheck:
    """What the check did: one :class:`Rejection` per redacted id, one :class:`Dispute`
    per disputed label, a note per dispute, and the counts."""
    rejections: list[Rejection] = field(default_factory=list)
    disputes: list[Dispute] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=lambda: {
        "hp_ids_checked": 0, "hp_ids_redacted": 0, "hp_labels_checked": 0, "hp_labels_disputed": 0,
    })


def check_terms(data: dict[str, Any], index: Any, *, redact: Callable[[str, str], str],
                skip: Iterable[str] = DEFAULT_SKIP) -> TermCheck:
    """Walk every string of ``data`` (in place) and apply the two rules above against
    ``index`` — the candidate's scoped index: ``hpo:<id>`` must be in it. ``redact(path,
    reason)`` returns the text that replaces an id no record carries."""
    out = TermCheck()
    skipped = frozenset(skip)
    _walk(data, "", index, redact, skipped, out)
    return out


def _walk(node: Any, path: str, index: Any, redact: Callable[[str, str], str], skip: frozenset[str], out: TermCheck) -> None:
    if isinstance(node, dict):
        for name, value in node.items():
            if name in skip:
                continue
            child = f"{path}.{name}" if path else str(name)
            if isinstance(value, str):
                node[name] = check_text(value, child, index, redact, out)
            else:
                _walk(value, child, index, redact, skip, out)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            child = f"{path}[{i}]"
            if isinstance(value, str):
                node[i] = check_text(value, child, index, redact, out)
            else:
                _walk(value, child, index, redact, skip, out)


def check_text(text: str, path: str, index: Any, redact: Callable[[str, str], str], out: TermCheck) -> str:
    """One string: every ``HP:`` token redacted or label-checked; the text as it goes on."""
    pieces: list[str] = []
    pos = 0
    for m in HPO_TOKEN.finditer(text):
        hpo_id = m.group(1) or m.group(2)
        rid = f"hpo:{hpo_id}"
        out.counts["hp_ids_checked"] += 1
        pieces.append(text[pos:m.start()])
        pos = m.end()
        rec = index.get(rid)
        if rec is None:
            reason = f"HP id not carried by any hpo: record in scope: {hpo_id}"
            out.rejections.append(Rejection(path, reason))
            out.counts["hp_ids_redacted"] += 1
            pieces.append(redact(path, reason))
            continue
        pieces.append(m.group(0))
        written = attached_phrase(text, m)
        if written is None:
            continue
        out.counts["hp_labels_checked"] += 1
        label = label_of(rec)
        if _matches(written, label, synonyms_of(rec), suffix=written_before(text, m)):
            continue
        out.counts["hp_labels_disputed"] += 1
        reason = (f'the text "{written}" attached to {hpo_id} is not the label of {rid} ("{label}") or one of its synonyms')
        out.disputes.append(Dispute(path, "HPO", rid, True, False, None, reason))
        synonyms = synonyms_of(rec)
        out.notes.append(f'{path}: {hpo_id} label disputed; the record spells it "{label}"'
                         + (f" (synonyms: {'; '.join(synonyms)})" if synonyms else ""))
        pieces.append(DISPUTED_LABEL.format(record_id=rid, label=label, written=written))
    pieces.append(text[pos:])
    return "".join(pieces)


def written_before(text: str, m: re.Match[str]) -> bool:
    """True when the phrase stands before the id — form (c), ``phrase (HP:nnnnnnn)``."""
    return m.group(2) is not None and m.start() > 0 and text[m.start() - 1] == "(" and text[m.end():m.end() + 1] == ")"


def attached_phrase(text: str, m: re.Match[str]) -> str | None:
    """The label the model attached to the bare token at ``m``, in one of the three
    forms, or ``None`` when there is none worth comparing (a bracketed id, an empty
    phrase, one longer than :data:`MAX_PHRASE_WORDS`, or one opening with a stop word)."""
    if m.group(2) is None:
        return None  # a bracketed id: the closing bracket separates it from whatever follows
    after = text[m.end():]
    if written_before(text, m):
        head = text[:m.start() - 1].rstrip()
        last = max((x.end() for x in _FORM_C_START.finditer(head)), default=0)
        return _phrase(head[last:], leading=True)
    a = _FORM_A.match(after)
    if a:
        return _phrase(a.group(1))
    sep = _FORM_B_SEP.match(after)
    if not sep:
        return None
    rest = after[sep.end():]
    end = _FORM_B_END.search(rest)
    return _phrase(rest[: end.start()] if end else rest)


def _phrase(raw: str, *, leading: bool = False) -> str | None:
    """The phrase to compare, or ``None``. After the id (forms a/b) a phrase opening
    with a stop word is the sentence going on, not a label. Before the id (form c,
    ``leading``) the sentence comes first — ``the case shows hearing loss (HP:…)`` —
    so leading stop words are dropped and what remains is compared."""
    words = raw.strip().split()
    if leading:
        while words and words[0].lower() in STOP_WORDS:
            words.pop(0)
    elif words and words[0].lower() in STOP_WORDS:
        return None
    if not words or len(words) > MAX_PHRASE_WORDS or not any(ch.isalpha() for ch in raw):
        return None  # a label has letters; digits alone are a neighbouring id's tail
    return " ".join(words)


def _matches(written: str, label: str, synonyms: list[str], *, suffix: bool) -> bool:
    """Form (c) matches when the normalised phrase ends with the label or a synonym
    (the words before it are the sentence); forms (a)/(b) when it starts with one."""
    n = normalise_label(written)
    if not n:
        return True  # nothing to compare
    for candidate in (label, *synonyms):
        c = normalise_label(candidate)
        if not c:
            continue
        if n == c or (n.endswith(c) if suffix else n.startswith(c)):
            return True
    return False
