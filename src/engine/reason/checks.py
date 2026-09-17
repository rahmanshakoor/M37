"""Stage-5 checks the shared validator does not make, or makes more leniently.

The validator (:mod:`engine.agents.validator`) redacts the citations it can name a
record for: ``[id]``, ``source:id`` of a known source, ``PMID n``, a bare nine-digit
``VCV…``, ``NCT…``, ``PMC…`` or ``doi:`` accession, a numbered ``[1]``; a dbSNP
``rs`` id it only notes. A model can still make an unsupported claim look supported
with a spelling that names no record — a Europe PMC / PubMed / ClinVar /
ClinicalTrials.gov URL, an ``SCV``/``RCV`` accession, a short ``VCV7105``, a DOI
without its ``doi:`` prefix, ``pubmed id 87654321`` — and an ``rs`` id that no citable
record carries is, for an evidence chain, a claim about the variant's identity that
nothing supports. This module holds every such form to one rule, ahead of the
validator: an accession may stand in prose only if a citable record carries it — as
the record's id (``clinvar:VCV000007105``, ``nct:NCT01807923``, ``pmid:7647779``), its
URL, or in its payload (rsIDs in VEP and gnomAD payloads; DOIs and PMC ids in Europe
PMC payloads) — otherwise it is redacted exactly as an unresolvable ``[id]`` is — a
footnote marker ``[^k]`` in the prose (:mod:`engine.agents.redaction`), numbered from 1
per chain, the validator continuing the count — and the redaction is a rejection in
the validation report carrying its marker. The forms the validator also
knows are handled here first for the same reason they are handled at all: the
stage's rule is stricter and one mark should mean one thing.

Redaction rather than deletion of the criterion, on purpose: the validator's policy is
that ``evidence_ids`` decide whether a criterion stands and that prose citations which
do not resolve are removed from the text. A justification that leans on a fabricated
accession loses the accession; the criterion still has to stand on its ``evidence_ids``.

The second check is about what kind of record a criterion may rest on. PS3 and BS3
assert a well-established functional study; only a paper can show one, so a PS3/BS3
that cites no ``pmid:`` record is dropped — a VEP or ClinVar record cannot stand in
for an assay, however real the record is.

Both checks run on the aligned chain *before* the validator, so every path in the
report (theirs and the validator's) names the model's own positions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from engine.agents.redaction import Redactions, unwrap
from engine.agents.schema import EvidenceChain
from engine.agents.validator import Rejection
from engine.retrieve.store import EvidenceRecord

PAPER_BACKED_CODES = ("PS3", "BS3")
"""Criteria that assert a functional study — a claim only a paper record can carry."""

_ACCESSION = re.compile(
    r"(?P<url>(?:https?://)?(?:www\.)?(?:"
    r"europepmc\.org/(?:article|abstract)/MED/(?P<url_pmid>\d+)"
    r"|pubmed\.ncbi\.nlm\.nih\.gov/(?P<url_pmid2>\d+)"
    r"|(?:www\.)?ncbi\.nlm\.nih\.gov/pubmed/(?P<url_pmid3>\d+)"
    r"|(?:www\.)?ncbi\.nlm\.nih\.gov/clinvar/variation/(?P<url_vcv>\d+)"
    r"|clinicaltrials\.gov/(?:study|ct2/show)/(?P<url_nct>NCT\d{8})"
    r")[^\s\]\)>\"']*)"
    r"|(?<![\w:/.])(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?(?P<doi>10\.\d{4,9}/[^\s\[\]()<>\"']+)"
    r"|(?<![\w:/.])pubmed\s*(?:id)?\s*:?\s*(?P<pubmed>\d{4,9})\b"
    r"|(?<![\w:/.])(?P<cv_prefix>VCV|SCV|RCV)0*(?P<cv_number>\d{1,9})\b"
    r"|(?<![\w:/.])rs(?P<rs>\d{3,})\b"
    r"|(?<![\w:/.])NCT(?P<nct>\d{8})\b"
    r"|(?<![\w:/.])PMC(?P<pmc>\d{4,})\b",
    re.IGNORECASE,
)
_TRAIL = ".,;:"


class AccessionResolver:
    """Whether a bare accession is carried by one of the candidate's citable records."""

    def __init__(self, records: Iterable[EvidenceRecord]):
        records = list(records)
        self.ids = {r.record_id for r in records}
        self.corpus = "\n".join(f"{r.record_id}\n{r.url}\n{json.dumps(r.payload, ensure_ascii=False)}" for r in records).lower()

    def resolves(self, m: re.Match[str]) -> bool:
        pmid = m.group("url_pmid") or m.group("url_pmid2") or m.group("url_pmid3") or m.group("pubmed")
        if pmid:
            return f"pmid:{int(pmid)}" in self.ids
        if m.group("url_vcv"):
            return f"clinvar:VCV{int(m.group('url_vcv')):09d}" in self.ids
        if m.group("url_nct"):
            return f"nct:{m.group('url_nct').upper()}" in self.ids
        if m.group("cv_prefix"):
            acc = f"{m.group('cv_prefix').upper()}{int(m.group('cv_number')):09d}"
            return f"clinvar:{acc}" in self.ids or acc.lower() in self.corpus
        if m.group("nct"):
            return f"nct:NCT{m.group('nct')}" in self.ids or f"nct{m.group('nct')}" in self.corpus
        token = m.group("doi") or (f"rs{m.group('rs')}" if m.group("rs") else f"pmc{m.group('pmc')}")
        return token.rstrip(_TRAIL).lower() in self.corpus


@dataclass
class StageChecks:
    """What the checks did: the chain as it goes on to the validator, the criteria
    dropped and the accessions redacted — one :class:`~engine.agents.validator.Rejection`
    each, path as the validator spells it."""
    chain: EvidenceChain
    dropped: list[Rejection] = field(default_factory=list)
    redacted: list[Rejection] = field(default_factory=list)
    """The redactions, with their markers — the counter's own list."""


def stage_checks(chain: EvidenceChain, resolver: AccessionResolver, redactions: Redactions | None = None) -> StageChecks:
    """Redact every unresolvable accession and drop every paper-less PS3/BS3. The
    classification is left for the validator to compute. ``redactions`` is the
    chain's marker counter; without one the count continues from the markers the
    chain already carries (the validator does the same, so it needs no argument)."""
    data = chain.model_dump()
    redactions = redactions or Redactions.continuing(data)
    out = StageChecks(chain, redacted=redactions.rejections)
    for i, v in enumerate(data["variants"]):
        vpath = f"variants[{i}]"
        v["summary"] = _redact(v["summary"], f"{vpath}.summary", resolver, redactions)
        kept = []
        for j, c in enumerate(v["criteria"]):
            cpath = f"{vpath}.criteria[{j}]"
            if c["code"] in PAPER_BACKED_CODES and not any(str(x).lower().startswith("pmid:") for x in c["evidence_ids"]):
                out.dropped.append(Rejection(cpath, f"{c['code']} asserts a functional study but cites no paper record (pmid:)"))
                continue
            c["justification"] = _redact(c["justification"], f"{cpath}.justification", resolver, redactions)
            kept.append(c)
        v["criteria"] = kept
    for name in ("phase_statement", "mechanism_hypothesis"):
        data[name] = _redact(data[name], name, resolver, redactions)
    for name in ("limits", "what_would_change_the_call"):
        data[name] = [_redact(x, f"{name}[{k}]", resolver, redactions) for k, x in enumerate(data[name])]
    out.chain = EvidenceChain.model_validate(data)
    return out


def accession_tokens(text: str) -> list[str]:
    """Every bare accession in ``text`` as written, in order — for tests and audits."""
    return [m.group(0) for m in _ACCESSION.finditer(text)]


def _redact(text: Any, path: str, resolver: AccessionResolver, redactions: Redactions) -> str:
    def repl(m: re.Match[str]) -> str:
        if resolver.resolves(m):
            return m.group(0)
        marker = redactions.redact(path, f"bare accession not carried by any citable record: {m.group(0).rstrip(_TRAIL)}")
        return marker + (m.group(0)[len(m.group(0).rstrip(_TRAIL)):])  # keep the sentence's punctuation
    return unwrap(_ACCESSION.sub(repl, str(text)))  # an accession the model bracketed on its own keeps one pair
