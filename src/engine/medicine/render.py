"""``06_medicine/report.md`` — the medicine report in the rubric's order.

Why stage 6 renders its own Markdown rather than reusing
:func:`engine.agents.render.render_medicine_report` (which stays for the legacy
shape): the report is read against the challenge rubric, whose 35 % criterion asks
whether the variant mechanism is sound *and* whether the repurposing candidate is
well supported by that mechanism. So the document is the ladder, in order — who the
patient is as records (the disease definition, the case terms with their labels), what
the variant does, what that does to a cell and to the person, which classes of
intervention were searched and with which searches, which candidates were proposed
with their counter-arguments and paediatric-safety argument, which were considered and
rejected, what to watch, what to do next, what the data cannot show, and the secondary
findings the run turned up. A reader can check every rung against the References.

Two wordings matter and are fixed here. An empty candidate list says **No candidate
proposed** and names the classes that were searched with their verdicts — the first
live run rendered "none survived validation" for a list nothing had been proposed
into, and the word *survived* never appears in this document. A redaction is a
footnote: the ``[^k]`` marker the validator left in the prose stays where it is, and
every ``##`` section ends with ``[^k]: citation removed by the validator: <reason>``
for the markers it printed (:mod:`engine.agents.redaction`), so prose reads as prose
and the reason is one line below.

Nothing is looked up that the validator did not accept: the index supplies URLs and
paper labels, never new facts, and the bundle (optional) supplies nothing the report
does not already carry — it is there for the patient-context block of a report whose
``patient_context`` the caller did not fill.
"""

from __future__ import annotations

from typing import Any, Iterable

from engine.agents.redaction import footnotes
from engine.agents.render import cited_ids, paper_label
from engine.agents.schema import (ConsideredAndRejected, DrugCandidate, InterventionClass, MechanismClaim, MedicineReport,
                                  PatientContext, SecondaryFinding)
from engine.agents.validator import EvidenceIndex, citation_tokens

NO_CANDIDATE = "No candidate proposed."
"""The first half of the empty-list sentence; the classes searched follow it."""
NO_CANDIDATE_NO_CLASS = ("No candidate proposed and no intervention class was searched: the ladder was not walked "
                         "(see the manifest).")
DESCRIPTION_CHARS = 600
"""How much of a disease description the report prints; the record holds the rest."""


def render_medicine_report(report: MedicineReport, index: EvidenceIndex | None = None, *,
                           disclosure: str | None = None, rejections: Iterable[Any] = (),
                           bundle: Any = None) -> str:
    """The report as Markdown, in the order above. ``rejections`` are the validation
    record's (:class:`~engine.agents.validator.Rejection` objects or their dicts): they
    supply the reason of every ``[^k]`` footnote. ``bundle`` is the stage-6 bundle,
    used only for the patient context of a report that carries none."""
    notes = _Footnotes(rejections)
    context = report.patient_context or _context_of(bundle)
    out = [f"# Medicine report — {report.gene_symbol} ({report.candidate_id})", ""]
    out.extend(notes.section(_patient_context(context)))
    out.extend(notes.section(_claims("Variant mechanism", report.mechanism)))
    out.extend(notes.section(_claims("Cellular and disease consequence", report.consequence)))
    out.extend(notes.section(_classes(report.intervention_classes, report.pathway_targets)))
    out.extend(notes.section(_candidates(report)))
    out.extend(notes.section(_rejected(report.considered_and_rejected)))
    out.extend(notes.section(_claims("Surveillance", report.surveillance)))
    out.extend(notes.section(_bullets("Follow-up experiments", report.follow_up_experiments)))
    out.extend(notes.section(_bullets("Limits", report.limits)))
    out.extend(notes.section(_secondary(report.secondary_findings)))
    out.extend(_literature(report.literature, index))
    out.extend(_references(cited_ids(report, index), index))
    out.extend(_footer(disclosure))
    return "\n".join(out).rstrip("\n") + "\n"


# ------------------------------------------------------------------------ sections

def _patient_context(context: PatientContext | None) -> list[str]:
    """The disease line(s) and the case terms — records only, never free text."""
    out = ["## Patient context", ""]
    if context is None or not (context.disease or context.hpo):
        out.extend(["- No disease record and no HPO term record in this run.", ""])
        return out
    for d in context.disease:
        head = f"- disease: {d.name or d.id} [{d.record_id}]"
        out.append(f"{head} — {_short(_para(d.description), DESCRIPTION_CHARS)}" if d.description else head)
    for t in context.hpo:
        if t.record_id:
            out.append(f"- case HPO: {t.id} {t.label or '(no label in the record)'} [{t.record_id}]")
        else:
            out.append(f"- case HPO: {t.id} (no hpo: record in the store; label unknown)")
    out.append("")
    return out


def _claims(title: str, claims: list[MechanismClaim]) -> list[str]:
    out = [f"## {title}", ""]
    if not claims:
        out.append("- none stated")
    for m in claims:
        out.append(f"- {_para(m.statement)} {_cite(m.evidence_ids, after=m.statement)}".rstrip())
    out.append("")
    return out


def _classes(classes: list[InterventionClass], pathway_targets: list[MechanismClaim]) -> list[str]:
    """The ladder's third rung as a table — one row per class, with the searches behind
    it — then the gene products those classes act on."""
    out = ["## Intervention classes searched", ""]
    if not classes:
        out.extend(["- No intervention class was searched.", ""])
    else:
        out.append("| class | acts on | targets | verdict | searches | records |")
        out.append("| --- | --- | --- | --- | --- | --- |")
        for c in classes:
            verdict = c.verdict.replace("_", " ")
            if c.rejection_reason.strip():
                verdict += f" — {_para(c.rejection_reason)}"
            out.append(f"| {_para(c.name)} | {_para(c.acts_on)} | {', '.join(c.targets) or 'none'} | {verdict} "
                       f"| {_cite(c.searched) or 'none'} | {_cite(c.evidence_ids) or 'none'} |")
        out.append("")
    out.append("Gene products the classes act on:")
    out.append("")
    if not pathway_targets:
        out.append("- none stated")
    for m in pathway_targets:
        out.append(f"- {_para(m.statement)} {_cite(m.evidence_ids, after=m.statement)}".rstrip())
    out.append("")
    return out


def _candidates(report: MedicineReport) -> list[str]:
    out = ["## Drug candidates", ""]
    if not report.candidates:
        out.extend([_no_candidate(report.intervention_classes), ""])
        return out
    for i, d in enumerate(report.candidates, 1):
        out.extend(_drug_section(i, d))
    return out


def _no_candidate(classes: list[InterventionClass]) -> str:
    """The empty-list sentence: what was searched, never a claim about survival."""
    if not classes:
        return NO_CANDIDATE_NO_CLASS
    return f"{NO_CANDIDATE} Classes searched: " + ", ".join(f"{_para(c.name)} ({c.verdict.replace('_', ' ')})" for c in classes)


def _drug_section(i: int, d: DrugCandidate) -> list[str]:
    head = f"### {i}. {d.name}" + (f" ({d.chembl_id})" if d.chembl_id else "")
    out = [head, ""]
    out.append(f"- class: {_para(d.intervention_class) or 'none stated'}")
    out.append(f"- mechanism of action: {_para(d.mechanism_of_action)}")
    out.append(f"- approval status: {_para(d.approval_status)}")
    out.append(f"- approved indication: {_para(d.approved_indication) or 'none stated'}")
    out.append(f"- rationale: {_para(d.rationale)}")
    out.append(f"- evidence: {_cite(d.evidence_ids) or 'none'}")
    out.append(f"- trials: {_cite(d.trial_ids) or 'none'}")
    out.append("- counter-arguments:")
    for a in d.counter_arguments:
        out.append(f"  - {_para(a)}")
    out.append(f"- paediatric safety: {_para(d.paediatric_safety) or 'none stated'}")
    out.append("")
    return out


def _rejected(items: list[ConsideredAndRejected]) -> list[str]:
    out = ["## Considered and rejected", ""]
    if not items:
        out.append("- nothing was raised and rejected on a record")
    for r in items:
        cls = f" (class: {_para(r.intervention_class)})" if r.intervention_class.strip() else ""
        out.append(f"- **{_para(r.name)}**{cls} — {_para(r.reason)} {_cite(r.evidence_ids, after=r.reason)}".rstrip())
    out.append("")
    return out


def _secondary(findings: list[SecondaryFinding]) -> list[str]:
    """The run's other P/LP lone heterozygotes — recorded, never targets."""
    out = ["## Secondary findings", ""]
    if not findings:
        out.extend(["- none recorded", ""])
        return out
    out.append("| candidate | gene | model | engine classification | note |")
    out.append("| --- | --- | --- | --- | --- |")
    for f in findings:
        calls = "; ".join(f"{k} {v.replace('_', ' ')}" for k, v in sorted(f.classifications.items())) or "none computed"
        out.append(f"| {f.candidate_id} | {f.gene_symbol or '-'} | {f.model or '-'} | {calls} | {_para(f.note)} |")
    out.append("")
    return out


# -------------------------------------------------------------------------- shared

class _Footnotes:
    """The reasons behind every ``[^k]`` marker, appended to the section that printed it."""

    def __init__(self, rejections: Iterable[Any]):
        self.rejections = list(rejections)

    def section(self, lines: list[str]) -> list[str]:
        notes = footnotes(lines, self.rejections)
        if not notes:
            return lines
        end = len(lines) - 1 if lines and lines[-1] == "" else len(lines)
        return lines[:end] + [""] + notes + [""]


def _context_of(bundle: Any) -> PatientContext | None:
    """The patient context of a report that carries none, from the stage-6 bundle."""
    if bundle is None:
        return None
    from engine.medicine.run import patient_context  # the stage imports this module; the cycle ends here
    return patient_context(bundle)


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


def _bullets(title: str, items: list[str]) -> list[str]:
    out = [f"## {title}", ""]
    if not items:
        out.append("- none")
    out.extend(f"- {_para(x)}" for x in items)
    out.append("")
    return out


def _cite(ids: list[str], after: str = "") -> str:
    """``[id] [id]`` for the ids not already cited inline in ``after``."""
    inline = set(citation_tokens(after)) if after else set()
    return " ".join(f"[{i}]" for i in ids if i not in inline)


def _para(text: Any) -> str:
    """One line: whitespace collapsed, so a statement never breaks a list item or a
    table row, and a pipe escaped, so it never ends a cell."""
    return " ".join(str(text if text is not None else "").split()).replace("|", "\\|")


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _footer(disclosure: str | None) -> list[str]:
    return ["---", "", f"_{disclosure}_", ""] if disclosure else []
