"""Redactions as footnotes, and near-miss record ids respelled instead of removed.

The validator's rule is that a citation which names no record in the store is removed
from the prose. On the first live run that left the sentence
``[citation removed: no such record]`` twice in a document whose thesis is citation
discipline — once for an id the model had spelled with a colon where the store has a
dash. Two things follow.

A redaction is a numbered marker, ``[^k]``, not a sentence. The marker is the Markdown
footnote reference; the validation record (a :class:`~engine.agents.validator.Rejection`
with ``marker=k``) carries the reason; a renderer prints, at the end of the section the
marker appears in, ``[^k]: citation removed by the validator: <reason>``. Numbering is
per validated object and continues across the stage checks and the validator: each
starts from ``max(marker in the text) + 1`` (:meth:`Redactions.continuing`), so the
validator needs no argument to pick up where a stage check left off. A marker the
model itself wrote is left alone (the validator's tokenizer does not read ``[^k]`` as a
citation), rendered with :data:`FOOTNOTE_UNKNOWN`, and noted by ``validate()``.

An id one *separator* edit away from exactly one store id of the same source is the
same claim misspelled, not a fabrication: it is respelled to the store's id and counted
(``ids_respelled``). A digit or letter difference is never respelled — that is a
different variant or a different paper, and the original rule applies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

from pydantic import BaseModel

if TYPE_CHECKING:
    from engine.agents.validator import EvidenceIndex, Rejection

MARK = "[^{k}]"
"""The footnote reference a redaction leaves in the prose."""
_MARK = re.compile(r"\[\^(\d+)\]")
_WRAPPED = re.compile(r"\[(\[\^\d+\])\]")
"""``[[^k]]`` — an accession the model had bracketed on its own, redacted inside its brackets."""
FOOTNOTE = "citation removed by the validator: {reason}"
FOOTNOTE_UNKNOWN = "citation removed by the validator (reason in the validation record)"
"""The footnote for a marker no rejection carries — one the model wrote, or one whose
validation record the renderer was not given."""
SEPARATORS = frozenset(":-_/.")
"""The characters a near-miss may differ in (substituted, inserted or deleted), besides case."""


@dataclass
class Redactions:
    """Numbers the redactions of one validated object, in order of discovery; every
    redaction is a Rejection that carries its marker."""
    next: int = 1
    rejections: list[Rejection] = field(default_factory=list)

    @classmethod
    def continuing(cls, obj: Any) -> "Redactions":
        """A counter that starts after every marker ``obj`` already carries (a dict,
        list, pydantic object or string) — so a second pass never reuses a number."""
        found = [k for _, k in scan(obj)]
        return cls(next=max(found, default=0) + 1)

    def redact(self, path: str, reason: str) -> str:
        """Record one redaction at ``path`` and return the marker that replaces it."""
        from engine.agents.validator import Rejection  # the validator imports this module; the cycle ends here
        k = self.next
        self.next += 1
        self.rejections.append(Rejection(path, reason, marker=k))
        return MARK.format(k=k)


def scan(obj: Any, path: str = "") -> list[tuple[str, int]]:
    """Every marker in every string of ``obj``, as ``(path, k)`` in order of
    appearance, paths spelled as the validator spells them (``variants[0].summary``)."""
    if isinstance(obj, BaseModel):
        obj = obj.model_dump()
    found: list[tuple[str, int]] = []
    if isinstance(obj, dict):
        for name, value in obj.items():
            found.extend(scan(value, f"{path}.{name}" if path else str(name)))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            found.extend(scan(item, f"{path}[{i}]"))
    elif isinstance(obj, str):
        found.extend((path, int(m.group(1))) for m in _MARK.finditer(obj))
    return found


def markers(text: str) -> list[int]:
    """The markers in ``text``, in order of first appearance."""
    out: list[int] = []
    for m in _MARK.finditer(text):
        k = int(m.group(1))
        if k not in out:
            out.append(k)
    return out


def prefixed(text: str, prefix: str) -> str:
    """``[^3]`` → ``[^<prefix>3]``, so a file holding several documents (one chain
    per candidate) keeps its footnote references unique."""
    if not prefix:
        return text
    return _MARK.sub(lambda m: MARK.format(k=f"{prefix}{m.group(1)}"), text)


def unwrap(text: str) -> str:
    """``[[^k]]`` → ``[^k]``: a marker that replaced an accession the model had
    bracketed on its own keeps one pair of brackets, its own."""
    return _WRAPPED.sub(r"\1", text)


def footnotes(texts: Iterable[str], rejections: Iterable[Any], *, prefix: str = "") -> list[str]:
    """One ``[^<prefix>k]: citation removed by the validator: <reason>`` line per marker
    found in ``texts`` (order of appearance), the reason from the rejection whose
    ``marker == k`` — a ``Rejection`` or its dict — else :data:`FOOTNOTE_UNKNOWN`."""
    reasons: dict[int, str] = {}
    for r in rejections:
        k = r.get("marker") if isinstance(r, dict) else getattr(r, "marker", None)
        reason = r.get("reason") if isinstance(r, dict) else getattr(r, "reason", None)
        if isinstance(k, int) and k not in reasons:
            reasons[k] = FOOTNOTE.format(reason=reason) if reason else FOOTNOTE_UNKNOWN
    seen: list[int] = []
    for text in texts:
        seen.extend(k for k in markers(str(text)) if k not in seen)
    return [f"{MARK.format(k=f'{prefix}{k}')}: {reasons.get(k, FOOTNOTE_UNKNOWN)}" for k in seen]


def respellable(written: str, stored: str) -> bool:
    """Whether ``written`` is ``stored`` up to case and one separator edit: equal
    ignoring case; or the same length with exactly one differing position where both
    characters are separators; or one character longer or shorter where the extra
    character is a separator and the rest is equal. A digit or letter difference is
    never a respelling."""
    a, b = written.lower(), stored.lower()
    if a == b:
        return True
    if len(a) == len(b):
        diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        return len(diffs) == 1 and a[diffs[0]] in SEPARATORS and b[diffs[0]] in SEPARATORS
    if abs(len(a) - len(b)) != 1:
        return False
    longer, shorter = (a, b) if len(a) > len(b) else (b, a)
    i = next((i for i in range(len(shorter)) if longer[i] != shorter[i]), len(shorter))
    return longer[i] in SEPARATORS and longer[i + 1:] == shorter[i:]


def near_misses(rid: str, index: EvidenceIndex) -> list[str]:
    """Every id of ``rid``'s source (the prefix before the first ``:``) in ``index``
    that ``rid`` is a respelling of. Stages 5 and 6 validate against the in-memory
    scoped index of one candidate, so the search is small; on a file store it reads
    one source directory."""
    source = rid.split(":", 1)[0]
    return [sid for sid in index.ids_of(source) if respellable(rid, sid)]


def respell(rid: str, index: EvidenceIndex) -> str | None:
    """The one index id ``rid`` is a respelling of; ``None`` when there is none or
    more than one (an ambiguous near-miss is left to the unknown-id rule)."""
    found = near_misses(rid, index)
    return found[0] if len(found) == 1 else None
