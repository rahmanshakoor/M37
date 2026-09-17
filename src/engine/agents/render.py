"""Markdown for the two agent outputs — what a reader checks, line by line.

The rendering rule mirrors the validation rule: every claim shows the record it rests
on as ``[record_id]`` right where the claim is made (once — an id the prose already
cites inline is not repeated after it), and the document ends with a References
section that maps every cited id to the URL a judge opens. Only ids the
object actually cites are listed, so a reference list is also an inventory of what
the argument used. Nothing is looked up that the validator did not already accept —
the index is consulted for URLs and paper titles, never for new facts.

The evidence chain prints the classification the validator computed from the
surviving criteria (never one the model asserted), and a disputed or unverified
criterion prints with the validator's ``[DISPUTED …]`` / ``[UNVERIFIED …]`` mark
intact. A redaction is a footnote: the ``[^k]`` marker the validator left in the prose
stays where it is and every ``##`` section ends with the ``[^k]: citation removed by
the validator: <reason>`` lines for the markers it printed, reasons from the
``rejections`` the caller hands in (a marker no rejection carries prints the reason
as unknown); ``footnote_prefix`` keeps the references unique when several documents
share one file. The medicine report follows the rubric's order: mechanism →
candidates with counter-arguments → follow-up → limits (the legacy renderer;
``engine.medicine.render`` writes stage 6's ``report.md``).
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from pydantic import BaseModel

from engine.agents.redaction import footnotes, prefixed
from engine.agents.schema import Criterion, DrugCandidate, EvidenceChain, MedicineReport, VariantChain
from engine.agents.validator import KNOWN_SOURCES, EvidenceIndex, citation_tokens

_WS = re.compile(r"\s+")


# ------------------------------------------------------------------- evidence chain

def render_evidence_chain(chain: EvidenceChain, index: EvidenceIndex | None = None, *,
                          disclosure: str | None = None, rejections: Iterable[Any] = (),
                          footnote_prefix: str = "") -> str:
    notes = _Footnotes(rejections, footnote_prefix)
    out = [f"# Evidence chain — {chain.candidate_id}", ""]
    for v in chain.variants:
        out.extend(notes.section(_variant_section(v)))
    out.extend(notes.section(["## Phase", "", _para(chain.phase_statement), ""]))
    out.extend(notes.section(["## Mechanism hypothesis", "", _para(chain.mechanism_hypothesis), ""]))
    out.extend(notes.section(_bullets("Limits", chain.limits)))
    out.extend(notes.section(_bullets("What would change the call", chain.what_would_change_the_call)))
    out.extend(_literature(chain.literature, index))
    out.extend(_references(cited_ids(chain, index), index))
    out.extend(_footer(disclosure))
    return "\n".join(out).rstrip("\n") + "\n"


def _variant_section(v: VariantChain) -> list[str]:
    label = v.classification.replace("_", " ") if v.classification else "not computed"
    if v.points is not None:
        table = f"; 2015 Table 5: {v.classification_richards_2015.replace('_', ' ')}" if v.classification_richards_2015 else ""
        label += f" ({v.points:+d} SVI points{table})"
    out = [f"## Variant {v.key} — {label}", ""]
    if v.summary.strip():
        out.extend([_para(v.summary), ""])
    out.append("Criteria (classification computed by the engine from the met criteria: ClinGen SVI points, Tavtigian 2020; "
               "PP5/BP6 retired and never counted):")
    out.append("")
    if not v.criteria:
        out.append("- none survived validation")
    for c in v.criteria:
        out.append(_criterion_line(c))
    out.append("")
    return out


def _criterion_line(c: Criterion) -> str:
    met = "met" if c.met else "not met"
    cites = _cite(c.evidence_ids, after=c.justification) or ("" if c.evidence_ids else "(case-level criterion; cites no record)")
    return f"- **{c.code}** · {c.strength} · {met} — {_para(c.justification)} {cites}".rstrip()


# ------------------------------------------------------------------ medicine report

def render_medicine_report(report: MedicineReport, index: EvidenceIndex | None = None, *,
                           disclosure: str | None = None, rejections: Iterable[Any] = (),
                           footnote_prefix: str = "") -> str:
    notes = _Footnotes(rejections, footnote_prefix)
    out = [f"# Medicine report — {report.gene_symbol} ({report.candidate_id})", ""]
    out.extend(notes.section(_claims("Mechanism", report.mechanism)))
    out.extend(notes.section(_claims("Pathway targets", report.pathway_targets)))
    drugs = ["## Drug candidates", ""]
    if not report.candidates:
        drugs.extend(["- No candidate proposed (the legacy renderer; see 06_medicine/report.md for the classes searched)", ""])
    for i, d in enumerate(report.candidates, 1):
        drugs.extend(_drug_section(i, d))
    out.extend(notes.section(drugs))
    out.extend(notes.section(_bullets("Follow-up experiments", report.follow_up_experiments)))
    out.extend(notes.section(_bullets("Limits", report.limits)))
    out.extend(_literature(report.literature, index))
    out.extend(_references(cited_ids(report, index), index))
    out.extend(_footer(disclosure))
    return "\n".join(out).rstrip("\n") + "\n"


def _claims(title: str, claims: list[Any]) -> list[str]:
    out = [f"## {title}", ""]
    if not claims:
        out.append("- none")
    for m in claims:
        out.append(f"- {_para(m.statement)} {_cite(m.evidence_ids, after=m.statement)}".rstrip())
    out.append("")
    return out


def _drug_section(i: int, d: DrugCandidate) -> list[str]:
    head = f"### {i}. {d.name}" + (f" ({d.chembl_id})" if d.chembl_id else "")
    out = [head, ""]
    out.append(f"- mechanism of action: {_para(d.mechanism_of_action)}")
    out.append(f"- approval status: {_para(d.approval_status)}")
    out.append(f"- rationale: {_para(d.rationale)}")
    out.append(f"- evidence: {_cite(d.evidence_ids) or 'none'}")
    out.append(f"- trials: {_cite(d.trial_ids) or 'none'}")
    out.append("- counter-arguments:")
    for a in d.counter_arguments:
        out.append(f"  - {_para(a)}")
    out.append("")
    return out


# ------------------------------------------------------------------------- shared

class _Footnotes:
    """The footnotes of one document: the reasons behind every ``[^k]`` marker, and
    the prefix that keeps the references unique when several documents share a file."""

    def __init__(self, rejections: Iterable[Any], prefix: str):
        self.rejections = list(rejections)
        self.prefix = prefix

    def section(self, lines: list[str]) -> list[str]:
        """One ``##`` section's lines (ending with its blank line) with every marker
        prefixed and the section's footnotes appended before that blank line."""
        notes = footnotes(lines, self.rejections, prefix=self.prefix)
        body = [prefixed(line, self.prefix) for line in lines]
        if not notes:
            return body
        end = len(body) - 1 if body and body[-1] == "" else len(body)
        return body[:end] + [""] + notes + [""]


def cited_ids(obj: BaseModel | dict[str, Any] | list[Any] | str, index: EvidenceIndex | None = None) -> list[str]:
    """Every record id the object cites, sorted: ``evidence_ids``, ``trial_ids``,
    ``literature``, inline mentions in any string — found by the validator's own
    tokenizer (:func:`~engine.agents.validator.citation_tokens`) so the two agree on
    what a citation is — and a ``chembl_id`` the index holds as a ``chembl:`` record
    (it is printed in the candidate's heading, so it is openable from References).
    ``index`` widens the bare-token sources to those it holds."""
    sources = set(KNOWN_SOURCES) | (index.sources() if index is not None else set())
    found: set[str] = set()
    _collect(obj.model_dump() if isinstance(obj, BaseModel) else obj, found, sources, index)
    return sorted(found)


def _collect(node: Any, found: set[str], sources: set[str], index: EvidenceIndex | None) -> None:
    if isinstance(node, dict):
        for name, value in node.items():
            if name in ("evidence_ids", "trial_ids", "literature") and isinstance(value, list):
                found.update(str(x) for x in value if isinstance(x, str) and x)
            elif name == "chembl_id":
                if isinstance(value, str) and value and index is not None and f"chembl:{value}" in index:
                    found.add(f"chembl:{value}")
            else:
                _collect(value, found, sources, index)
    elif isinstance(node, list):
        for item in node:
            _collect(item, found, sources, index)
    elif isinstance(node, str):
        found.update(citation_tokens(node, sources))


def _references(ids: Iterable[str], index: EvidenceIndex | None) -> list[str]:
    out = ["## References", ""]
    ids = list(ids)
    if not ids:
        out.append("- none")
    for rid in ids:
        url = index.url(rid) if index is not None else None
        out.append(f"- [{rid}] — {url}" if url else f"- [{rid}] — (no URL in the evidence store)")
    out.append("")
    return out


def _literature(ids: list[str], index: EvidenceIndex | None) -> list[str]:
    out = ["## Literature", ""]
    if not ids:
        out.append("- none")
    for rid in ids:
        label = paper_label(index.get(rid)) if index is not None else ""
        out.append(f"- [{rid}]" + (f" {label}" if label else ""))
    out.append("")
    return out


def paper_label(rec: Any) -> str:
    """``Frosst P et al. (1995) Nature genetics — title`` from a Europe PMC payload;
    '' for anything else."""
    p = getattr(rec, "payload", None)
    if not isinstance(p, dict) or not p.get("title"):
        return ""
    authors = str(p.get("authorString") or "").strip().rstrip(".")
    first = authors.split(",")[0].strip() if authors else ""
    if first and "," in authors:
        first += " et al."
    year = p.get("pubYear") or (p.get("firstPublicationDate") or "")[:4]
    journal = ((p.get("journalInfo") or {}).get("journal") or {}).get("title") or ""
    head = " ".join(x for x in (first, f"({year})" if year else "", journal) if x)
    title = _WS.sub(" ", re.sub(r"<[^>]+>", "", str(p["title"]))).strip().rstrip(".")
    return f"{head} — {title}" if head else title


def _bullets(title: str, items: list[str]) -> list[str]:
    out = [f"## {title}", ""]
    if not items:
        out.append("- none")
    out.extend(f"- {_para(x)}" for x in items)
    out.append("")
    return out


def _cite(ids: list[str], after: str = "") -> str:
    """``[id] [id]`` for the ids not already cited inline in ``after`` — a statement
    that ends in ``[pmid:1]`` is not followed by a second ``[pmid:1]``."""
    inline = set(citation_tokens(after)) if after else set()
    return " ".join(f"[{i}]" for i in ids if i not in inline)


def _para(text: str) -> str:
    """One line: whitespace collapsed so a justification never breaks a list item; a
    ``[^k]`` marker is text like any other and passes through."""
    return _WS.sub(" ", str(text)).strip()


def _footer(disclosure: str | None) -> list[str]:
    return ["---", "", f"_{disclosure}_", ""] if disclosure else []
