"""The gate between a model's answer and anything a reader sees.

The one rule downstream of stage 2 is that a claim may cite only a record that exists
in the evidence store. A language model will, sooner or later, produce a record id
that looks right and does not exist, a PMID it never fetched, or a frequency criterion
that contradicts the gnomAD record for the variant. None of that is allowed to reach a
report, so this module walks the model's output *by its schema* — every list of
``Criterion``, ``MechanismClaim`` or ``DrugCandidate`` is known from the pydantic model,
so an item that simply omits ``evidence_ids`` is judged like one that wrote ``[]`` — and:

- drops any criterion, claim or drug candidate whose ``evidence_ids`` (or
  ``trial_ids``) name a record the store does not hold — the whole item, because an
  item that leans on a fabricated citation cannot be trusted on its remaining ones;
- drops a criterion that cites nothing unless its code may legitimately rest on the
  case itself (:data:`~engine.agents.schema.CASE_LEVEL_CODES`), one whose code is not
  an ACMG code, and a mechanism claim or drug candidate that cites nothing at all;
- drops a frequency criterion (PM2, BS1, BA1) that cites the gnomAD record of a
  *different* variant — the one way to make a common variant look rare while every
  cited id resolves;
- pins every variant entry to a real key: a respelled key (``chr7:…``, ``7-…-…``) is
  matched back to the engine's spelling, and — when the caller passes the candidate's
  ``keys`` — an entry for a variant that is not the candidate's is dropped with its
  criteria, because the frequency checks below are only as good as the key they look up;
- keeps the first surviving criterion of each code per variant and drops the rest, so
  a repeated PS3 cannot count twice in the combining rules;
- lowers a criterion's strength to the most its code may carry (its ACMG/AMP 2015
  default; PP3/BP4 up to strong per the ClinGen SVI computational recommendation) — a
  ``PP3`` at ``very_strong`` would otherwise drive the engine's own verdict;
- drops a drug candidate with no counter-arguments — a one-sided proposal is not a
  proposal — and clears a ``chembl_id`` that is neither a ``chembl:`` record in the
  store nor carried by a record the candidate cites;
- removes ``literature`` entries that are not ``pmid:<n>`` records in the store, and
  redacts inline citations in prose that do not resolve — ``[id]``, ``[id, id]``, a bare
  ``pmid:<n>`` / ``gnomad:<id>`` token of a known source, ``PMID <n>`` / ``PubMed <n>``,
  a bare ClinVar ``VCV…``, trial ``NCT…``, ``PMC…``, ``doi:`` or ``CHEMBL…`` accession
  (``PMC``/``doi:`` resolve through the store's ``pmid:`` records; ``CHEMBL`` through a
  ``chembl:`` record or a cited record that carries it), and a numbered ``[1]``
  reference that points at nothing — each replaced by a numbered footnote marker
  ``[^k]`` whose :class:`Rejection` carries ``marker=k`` (:mod:`engine.agents.redaction`),
  never by a sentence; a bracketed ontology term (``[HP:0002205]``) is
  not a citation and is left alone; a dbSNP ``rs`` id no cited record carries is left
  in place but noted, since it is an identifier rather than a citation;
- respells, before calling it unknown, an id one separator edit (``:`` ``-`` ``_``
  ``/`` ``.`` or case) away from exactly one store id of the same source — in
  ``evidence_ids``/``trial_ids`` and in prose — and counts it ``ids_respelled``; a digit
  or letter difference is a different record and is never respelled;
- recomputes PM2, BS1 and BA1 from the store's gnomAD record *for the variant's key*
  (never from whichever record the model chose), falling back to VEP's copy of the
  gnomAD frequency when stage 2 fetched no gnomAD record, and, where the model
  disagrees with the data, keeps the criterion with the *recomputed* value and a
  ``[DISPUTED …]`` mark on its justification; a frequency criterion the store cannot
  check at all is marked ``[UNVERIFIED …]`` and **not met** — what cannot be checked
  does not count, in either direction;
- fills every ``classification`` from the surviving criteria with the ACMG combining
  rules — the model is never believed on the verdict, only on the criteria.

Every removal is a :class:`Rejection` with a path into the object; every override is a
:class:`Dispute` or a note. The report — with the thresholds, the frequency field and
the rules applied — goes into the stage manifest. Nothing here raises for a bad
citation; only a malformed object (one that does not fit its own schema after
cleaning) raises, because that is a programming error, not a model error.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, get_args, get_origin

from pydantic import BaseModel

from engine.agents.redaction import Redactions, respellable, scan, unwrap
from engine.agents.schema import (
    ALL_CODES, BENIGN_CODES, CASE_LEVEL_CODES, PATHOGENIC_CODES, RETIRED_CODES, EvidenceChain, MedicineReport,
    acmg_points, combine_acmg, combine_richards_2015,
)
from engine.contigs import canonical
from engine.retrieve.gnomad import GnomadRetriever, unaskable, variant_id
from engine.retrieve.store import EvidenceRecord, EvidenceStore, key_str, parse_key
from engine.retrieve.vep import VepRetriever

THRESHOLDS: dict[str, float] = {"ba1_min_af": 0.05, "bs1_min_af": 0.01, "pm2_max_af": 0.0001}
"""BA1: af > 0.05. BS1: af > 0.01 (a ceiling for a recessive disease; a dominant
disease would use a lower one). PM2: af < 0.0001, or absent from gnomAD."""
AF_FIELD = "gnomad_af"
"""Which projected gnomAD column the thresholds are applied to. ``gnomad_af`` is the
joint (exome+genome) ``ac/an`` — the headline number in the bundle and stage 3's
``af_used``. ClinGen applies BA1/BS1 to the filtering allele frequency; pass
``af_field="gnomad_faf95_popmax"`` to :func:`validate` for that (it falls back to
``gnomad_af`` where the record carries none). Recorded in the report either way."""
AF_FIELDS = ("gnomad_af", "gnomad_faf95_popmax", "gnomad_grpmax_af")
VEP_AF_FIELDS = ("vep_gnomade_af", "vep_gnomadg_af")
VEP_AF_LABELS = {"vep_gnomade_af": "exomes", "vep_gnomadg_af": "genomes"}
FREQUENCY_CODES = ("PM2", "BS1", "BA1")
COMPUTATIONAL_CODES = ("PP3", "BP4")
GENE_CONSTRAINT_CODES = ("PP2", "BP1")
PHENOTYPE_CODES = ("PP4",)
"""PP4 (phenotype highly specific for a disease with a single genetic aetiology) is a
case-level claim the engine can partly check: it is counted only when the gene-blind
phenotype ranker (stage 4) put the candidate's gene first for the case's HPO terms.
The single-aetiology half of the definition is not checked and the report says so."""
"""PP2 (missense in a gene with a low rate of benign missense variation) and BP1
(missense in a gene where truncations are the mechanism) are gene-level facts about
constraint that only a ``constraint:`` record (gnomAD missense o/e) could establish;
the store holds none today, so a model claiming either is marked unverified and it
is not counted."""
REVEL_PP3 = ((0.932, "strong"), (0.773, "moderate"), (0.644, "supporting"))
REVEL_BP4 = ((0.016, "strong"), (0.183, "moderate"), (0.290, "supporting"))
CADD_PP3_MIN = 25.3
CADD_BP4_MAX = 22.7
SPLICEAI_PP3_MIN = 0.2
SPLICEAI_BP4_MAX = 0.1
"""The ClinGen SVI calibration of computational evidence (Pejaver et al. 2022, Am J
Hum Genet 109:2163): REVEL ≥ 0.644 / 0.773 / 0.932 → PP3 supporting / moderate /
strong, REVEL ≤ 0.290 / 0.183 / 0.016 → BP4 supporting / moderate / strong; CADD
PHRED ≥ 25.3 → PP3 supporting, ≤ 22.7 → BP4 supporting, used only when REVEL is
absent (a non-missense variant). SpliceAI ≥ 0.2 → PP3 supporting for a splicing
effect (ClinGen SVI splicing recommendations, Walker et al. 2023), ≤ 0.1 → BP4
supporting, applied to non-coding and synonymous changes. AlphaMissense is carried
in the bundle for the reader but is not calibrated by the SVI and never counts."""
EVIDENCE_STAGES = ("02_retrieve", "04_rank", "05_reason", "06_medicine")
"""Run-directory stages that hold an ``evidence/`` directory in the stage-2 format."""
KNOWN_SOURCES = frozenset({"vep", "clinvar", "gnomad", "exomiser", "pmid", "nct", "opentargets", "dgidb", "chembl",
                           "hpo", "uniprot"})
"""Record-id prefixes the engine's retrievers write. A bare ``<source>:<id>`` token in
prose counts as a citation only for these and for the sources present in the index —
so ``HP:0002205`` or ``chr7:117559590`` in a sentence is left alone (``hpo:HP:0002205``
is a citation, even in a scoped index that happens to hold no ``hpo:`` record)."""

STRENGTH_RANK = {"supporting": 1, "moderate": 2, "strong": 3, "very_strong": 4, "stand_alone": 5}
STRENGTH_CAP: dict[str, str] = {**PATHOGENIC_CODES, **BENIGN_CODES, "PP3": "strong", "BP4": "strong"}
"""The strongest level each code may be applied at: its ACMG/AMP 2015 default
(Richards et al., Tables 3–4), except PP3/BP4, which the ClinGen SVI computational
recommendation (Pejaver et al. 2022) allows up to strong. A lower level is always
allowed (``PM2_Supporting``, ``PVS1_Moderate`` — the model states it, the engine
honours it); a higher one is lowered to the cap and marked."""

DISPUTED_MARK = "[DISPUTED — {reason}]"
UNVERIFIED_MARK = "[UNVERIFIED — {reason}]"
CAPPED_MARK = "[STRENGTH CAPPED — {reason}]"
_TRAIL = ".,;:"

_PMID_ID = re.compile(r"^pmid:\d+$")
_NCT_ID = re.compile(r"^nct:NCT\d{8}$")
_BARE_NCT = re.compile(r"^(?:nct:)?(NCT\d{8})$", re.IGNORECASE)
_ID_LIKE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*:\S+$")
_NUMBERED = re.compile(r"^\d{1,3}(?:\s*[-–]\s*\d{1,3})?$")
_SPLIT = re.compile(r"[\s,;]+")
_KEY_SEP = re.compile(r"[:\-_/]")
_ALLELE = re.compile(r"^[A-Z]+$")
_CITATION = re.compile(
    r"\[(?P<group>[^\[\]]+)\]"                                                                   # [id] / [id, id] / [1]
    r"|(?<![\w:/.])(?P<doi>doi:\s*10\.\d{4,9}/[^\s\[\]()]+)"                                       # doi:10.xxxx/… (before source:id)
    r"|(?<![\w:])(?P<src>[A-Za-z][A-Za-z0-9_-]*):(?P<id>[A-Za-z0-9](?:[A-Za-z0-9._:-]*[A-Za-z0-9])?)"  # bare source:id
    r"|\b(?:PMID|PubMed(?:\s+ID)?)\s*:?\s*(?P<pmid>\d{4,9})\b"                                    # PMID 123 / PubMed 123
    r"|(?<![\w:])(?P<vcv>VCV\d{9})\b"                                                            # ClinVar accession
    r"|(?<![\w:])(?P<nct>NCT\d{8})\b"                                                            # trial id
    r"|(?<![\w:])(?P<pmc>PMC\d{5,9})\b"                                                          # PubMed Central id
    r"|(?<![\w:])(?P<chembl>CHEMBL\d+)\b"                                                         # ChEMBL molecule/target id
    r"|(?<![\w:])(?P<rs>rs\d{3,})\b",                                                            # dbSNP id (identifier, not citation)
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------- index

class EvidenceIndex:
    """Membership and payload access over one or more evidence directories, plus
    records held in memory (a stage may add ``pmid:`` records as it runs). Membership
    is checked against the files at call time, so a record written mid-run counts."""

    def __init__(self, stores: Iterable[EvidenceStore] = (), records: Iterable[EvidenceRecord] = ()):
        self.stores = list(stores)
        self._memory: dict[str, EvidenceRecord] = {r.record_id: r for r in records}

    @classmethod
    def from_run(cls, run_dir: str | Path, stages: Iterable[str] = EVIDENCE_STAGES) -> "EvidenceIndex":
        run_dir = Path(run_dir)
        dirs = [run_dir / s / "evidence" for s in stages if (run_dir / s / "evidence").is_dir()]
        return cls(EvidenceStore(d) for d in dirs)

    @classmethod
    def from_records(cls, records: Iterable[EvidenceRecord]) -> "EvidenceIndex":
        return cls(records=records)

    def add(self, record: EvidenceRecord) -> None:
        self._memory[record.record_id] = record

    def __contains__(self, record_id: object) -> bool:
        if not isinstance(record_id, str):
            return False
        return record_id in self._memory or any(s.exists(record_id) for s in self.stores)

    def get(self, record_id: str) -> EvidenceRecord | None:
        rec = self._memory.get(record_id)
        if rec is not None:
            return rec
        for s in self.stores:
            rec = s.get(record_id)
            if rec is not None:
                return rec
        return None

    def url(self, record_id: str) -> str | None:
        rec = self.get(record_id)
        return rec.url if rec is not None else None

    def ids(self) -> list[str]:
        out = set(self._memory)
        for s in self.stores:
            out.update(r.record_id for r in s.iter())
        return sorted(out)

    def iter_source(self, source: str) -> Iterator[EvidenceRecord]:
        """Every record of one source, memory first, each id once — reads only that
        source's directory of each store."""
        seen: set[str] = set()
        for rec in self._memory.values():
            if rec.source == source:
                seen.add(rec.record_id)
                yield rec
        for s in self.stores:
            for rec in s.iter(source):
                if rec.record_id not in seen:
                    seen.add(rec.record_id)
                    yield rec

    def sources(self) -> set[str]:
        """Every record-id prefix held: the store directories plus in-memory records."""
        out = {rid.split(":", 1)[0] for rid in self._memory}
        for s in self.stores:
            out.update(d.name for d in s.root.iterdir() if d.is_dir())
        return out

    def ids_of(self, source: str) -> list[str]:
        """Every record id of one source, memory records first, then each store's
        ``<source>/`` directory (read through :meth:`iter_source`) — what a near-miss
        respelling searches. Stages 5 and 6 validate against the in-memory scoped index
        of one candidate, so the list is short; on a file store this reads one
        source directory."""
        return [rec.record_id for rec in self.iter_source(source)]


# --------------------------------------------------------------------------- report

@dataclass
class Rejection:
    path: str
    reason: str
    marker: int | None = None
    """The footnote marker ``[^k]`` left in the prose when the rejection is a
    redaction; ``None`` for a dropped item or a cleared field."""


@dataclass
class Dispute:
    path: str
    code: str
    record_id: str | None
    """The record the recomputation rests on; ``None`` when nothing in the store
    could check the criterion and it was set to not met for that reason."""
    claimed_met: bool
    recomputed_met: bool
    af: float | None
    reason: str


@dataclass
class ValidationReport:
    model: str
    thresholds: dict[str, float]
    rules: dict[str, str] = field(default_factory=dict)
    """The frequency field and the comparison applied per code — what the manifest
    needs to say how every dispute was computed."""
    rejections: list[Rejection] = field(default_factory=list)
    disputes: list[Dispute] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=lambda: {
        "ids_checked": 0, "ids_unknown": 0, "items_dropped": 0, "duplicates_dropped": 0, "literature_removed": 0,
        "redactions": 0, "identifiers_unverified": 0, "keys_respelled": 0, "ids_respelled": 0, "strength_capped": 0,
        "chembl_cleared": 0,
        "frequency_recomputed": 0, "frequency_from_vep": 0, "frequency_disputed": 0, "frequency_unverified": 0,
        "computational_recomputed": 0, "computational_disputed": 0, "computational_unverified": 0,
        "constraint_unverified": 0, "phenotype_disputed": 0, "phenotype_unverified": 0, "retired_not_counted": 0, "classification_replaced": 0,
    })

    @property
    def clean(self) -> bool:
        return not self.rejections and not self.disputes

    def reject(self, path: str, reason: str) -> None:
        self.rejections.append(Rejection(path, reason))

    def redact(self, path: str, reason: str, redactions: Redactions) -> str:
        """Record one redaction — the rejection with its marker, the counts an unknown
        id adds — and return the marker that replaces the citation in the prose."""
        marker = redactions.redact(path, reason)
        self.rejections.append(redactions.rejections[-1])
        self.counts["ids_checked"] += 1
        self.counts["ids_unknown"] += 1
        self.counts["redactions"] += 1
        return marker

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def frequency_rules(th: dict[str, float], af_field: str) -> dict[str, str]:
    """The rules the report states it applied — every comparison a judge would redo."""
    return {
        "af_field": af_field,
        "fallback": "max(vep_gnomade_af, vep_gnomadg_af) from the vep: record when the store holds no gnomAD record",
        "PM2": (f"met iff af < {th['pm2_max_af']:g}, or the gnomAD record says the variant was queried and not found; "
                "a variant with no gnomAD record whose VEP copy lists no frequency is left as the model called it (a gap is not an observation)"),
        "BS1": f"met iff af > {th['bs1_min_af']:g}",
        "BA1": f"met iff af > {th['ba1_min_af']:g}",
        "PP3": ("met at the strength the vep: record's scores support: missense by REVEL "
                f"(≥{REVEL_PP3[2][0]} supporting, ≥{REVEL_PP3[1][0]} moderate, ≥{REVEL_PP3[0][0]} strong); "
                f"otherwise CADD PHRED ≥{CADD_PP3_MIN} supporting; a splice effect by SpliceAI ≥{SPLICEAI_PP3_MIN} supporting "
                "(Pejaver 2022; Walker 2023). A stated strength above what the scores support is lowered; "
                "no qualifying score → not met; no vep: record → unverified, not met"),
        "BP4": ("met at the strength the vep: record's scores support: missense by REVEL "
                f"(≤{REVEL_BP4[2][0]} supporting, ≤{REVEL_BP4[1][0]} moderate, ≤{REVEL_BP4[0][0]} strong); "
                f"otherwise CADD PHRED ≤{CADD_BP4_MAX} supporting; non-coding/synonymous by SpliceAI ≤{SPLICEAI_BP4_MAX} supporting"),
        "PP3+PVS1": "PP3 is not counted beside a met PVS1 on the same variant (ClinGen SVI)",
        "PP4": "counted only when the gene-blind phenotype ranker (stage 4) ranks the candidate's gene first for the case's "
               "HPO terms; unverified (not met) when stage 4 did not run; the single-aetiology condition is not checked",
        "PP2/BP1": "counted only with a constraint: record (gnomAD missense o/e) in evidence_ids; otherwise unverified, not met",
        "PP5/BP6": "retired by the ClinGen SVI (Biesecker & Harrison 2018): accepted for the record, marked, never counted",
        "classification": "ClinGen SVI points (Tavtigian 2020) over the met, non-retired criteria: supporting 1, moderate 2, "
                          "strong 4, very strong 8, benign negative; P ≥ 10, LP 6–9, VUS 0–5, LB −1 to −6, B ≤ −7. "
                          "The 2015 Table 5 verdict is recorded beside it",
        "unverified": "PM2/BS1/BA1 with no gnomAD or VEP record for the variant's key in the store → not met, marked [UNVERIFIED]",
        "strength_cap": ("a code counts at most at its ACMG/AMP 2015 default level (PP3/BP4 at most strong); "
                         "a higher stated strength is lowered to the cap and marked [STRENGTH CAPPED]"),
        "respelling": ("an id one separator edit (: - _ / . or case) away from exactly one store id of the same source "
                       "is respelled to the store's id and counted ids_respelled; a digit or letter difference is never respelled"),
        "redaction": "an inline citation that names no record is replaced by a footnote marker [^k]; the rejection carries marker k",
    }


# ------------------------------------------------------------------------- validate

@dataclass
class _Walk:
    """What every step of the walk needs."""
    index: EvidenceIndex
    report: ValidationReport
    th: dict[str, float]
    af_field: str
    sources: frozenset[str]
    keys: dict[str, str] | None
    """Canonical key → the candidate's own spelling; ``None`` = any real key is allowed."""
    redactions: Redactions = field(default_factory=Redactions)
    """The marker counter — continued from the markers the object already carried."""
    cited: list[str] = field(default_factory=list)
    """Every record id the answer cites anywhere (as first read) — the records an
    identifier in prose may be carried by."""
    pvs1_met: dict[str, bool] = field(default_factory=dict)
    phenotype_rank: dict[str, Any] | None = None
    """Variant key → whether the model wrote a met PVS1 for it (read before its criteria
    are cleaned, so PP3 on the same variant can be refused)."""
    _corpus: dict[str, str] = field(default_factory=dict)
    _papers: dict[str, str] | None = None
    _ids_of: dict[str, list[str]] = field(default_factory=dict)

    def near_misses(self, rid: str) -> list[str]:
        """The index ids of ``rid``'s source that ``rid`` is a respelling of
        (:func:`~engine.agents.redaction.respellable`); each source's ids are read once
        per walk, so a file-backed index reads a source directory once, not per id."""
        source = rid.split(":", 1)[0]
        if source not in self._ids_of:
            self._ids_of[source] = self.index.ids_of(source)
        return [sid for sid in self._ids_of[source] if respellable(rid, sid)]

    def corpus(self, record_id: str) -> str:
        """A record's id, URL and payload as one lower-cased string, cached."""
        if record_id not in self._corpus:
            rec = self.index.get(record_id)
            self._corpus[record_id] = "" if rec is None else \
                f"{rec.record_id}\n{rec.url}\n{json.dumps(rec.payload, ensure_ascii=False)}".lower()
        return self._corpus[record_id]

    def carried(self, token: str, record_ids: Iterable[str]) -> bool:
        needle = token.lower()
        return any(needle in self.corpus(rid) for rid in record_ids)

    def paper(self, accession: str) -> str | None:
        """The ``pmid:`` record that carries a ``PMC…`` id or a DOI, built once from
        the store's ``pmid:`` records (their Europe PMC ``pmcid``/``doi`` fields)."""
        if self._papers is None:
            self._papers = {}
            for rec in self.index.iter_source("pmid"):
                p = rec.payload if isinstance(rec.payload, dict) else {}
                for name in ("pmcid", "doi"):
                    if p.get(name):
                        self._papers.setdefault(str(p[name]).strip().lower(), rec.record_id)
        return self._papers.get(accession.strip().lower())


def validate(obj: BaseModel | dict[str, Any], index: EvidenceIndex, *,
             model: type[BaseModel] | None = None,
             thresholds: dict[str, float] | None = None,
             af_field: str = AF_FIELD,
             keys: Iterable[str] | None = None,
             phenotype_rank: dict[str, Any] | None = None,
             redactions: Redactions | None = None) -> tuple[BaseModel, ValidationReport]:
    """Return ``(cleaned, report)``. ``obj`` is the model's answer as a pydantic object
    or a raw dict (then ``model`` says which schema, or it is inferred for the two
    known ones). ``keys`` are the candidate's variant keys: given, a variant entry
    whose key is none of them is dropped; absent, any parseable key is accepted (and
    respelled canonically). The returned object is a fresh instance; ``obj`` is not
    modified. ``phenotype_rank`` (``{"rank": int | None, "record_id": str | None}``) is
    the gene-blind phenotype ranker's verdict on the candidate's gene, when stage 4
    ran: PP4 is counted only for the ranker's top gene. ``redactions`` is the marker
    counter a stage check already used on this object; without it the counter
    continues from the markers found in the text, and each of those — the model's
    own, as far as this call can tell — is noted. Raises only if the cleaned object
    does not fit its schema."""
    if isinstance(obj, BaseModel):
        model = model or type(obj)
        data = obj.model_dump()
    elif isinstance(obj, dict):
        model = model or _infer_model(obj)
        data = copy.deepcopy(obj)
    else:
        raise TypeError(f"expected a pydantic object or a dict, got {type(obj).__name__}")
    if af_field not in AF_FIELDS:
        raise ValueError(f"af_field must be one of {AF_FIELDS}, got {af_field!r}")
    th = {**THRESHOLDS, **(thresholds or {})}
    report = ValidationReport(model=model.__name__, thresholds=th, rules=frequency_rules(th, af_field))
    sources = frozenset(KNOWN_SOURCES | index.sources())
    wanted = None if keys is None else {canonical_key(str(k)) or str(k): str(k) for k in keys}
    found = scan(data)
    if redactions is None:
        redactions = Redactions.continuing(data)
        report.notes.extend(f"{path}: model-written footnote marker [^{k}] left in place (no rejection of this "
                            "validation carries it; a renderer prints its reason as unknown)" for path, k in found)
    else:
        redactions.next = max(redactions.next, max((k for _, k in found), default=0) + 1)
    walk = _Walk(index, report, th, af_field, sources, wanted, redactions=redactions,
                 cited=_collect_cited(data, sources), phenotype_rank=phenotype_rank)
    _clean_object(data, model, "", walk, key=None)
    return model.model_validate(data), report


def canonical_key(s: str) -> str | None:
    """The engine's spelling of a variant key from any common one — ``chr7:…``,
    ``7-117559590-ATCT-A``, lower-case alleles — or ``None`` for a string that is not
    a ``chrom pos ref alt`` key at all."""
    parts = _KEY_SEP.split(s.strip())
    if len(parts) != 4:
        return None
    chrom = canonical(parts[0])
    ref, alt = parts[2].strip().upper(), parts[3].strip().upper()
    if chrom is None or not parts[1].strip().isdigit() or not (_ALLELE.match(ref) and _ALLELE.match(alt)):
        return None
    return key_str((chrom, int(parts[1]), ref, alt))


def _collect_cited(node: Any, sources: frozenset[str], found: list[str] | None = None) -> list[str]:
    """Every record id the answer cites, in order of first appearance: the citation
    lists and the inline citations of every string."""
    found = [] if found is None else found
    if isinstance(node, dict):
        for name, value in node.items():
            if name in ("evidence_ids", "trial_ids", "literature") and isinstance(value, list):
                ids = _norm_trial_ids(value) if name == "trial_ids" else _norm_ids(value)
                found.extend(i for i in ids if i not in found)
            else:
                _collect_cited(value, sources, found)
    elif isinstance(node, list):
        for item in node:
            _collect_cited(item, sources, found)
    elif isinstance(node, str):
        found.extend(i for i in citation_tokens(node, sources) if i not in found)
    return found


def _infer_model(d: dict[str, Any]) -> type[BaseModel]:
    if "variants" in d and "phase_statement" in d:
        return EvidenceChain
    if "candidates" in d and "mechanism" in d:
        return MedicineReport
    raise TypeError("cannot infer the output model from a dict; pass model=")


# --------------------------------------------------------------------------- walking

def _item_model(annotation: Any) -> type[BaseModel] | None:
    """The pydantic model of a ``list[Model]`` annotation, else ``None``."""
    if get_origin(annotation) is list:
        args = get_args(annotation)
        if args and _is_model(args[0]):
            return args[0]
    return None


def _is_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _clean_object(data: dict[str, Any], model: type[BaseModel], path: str, w: _Walk, key: str | None) -> None:
    """Walk one object by its model's fields, in place. ``key`` is the enclosing
    variant's key, for the frequency checks."""
    fields = model.model_fields
    if "key" in fields and "criteria" in fields and isinstance(data.get("key"), str):
        key = data["key"]
        # PVS1 is the model's claim; whether it survives the citation rules is decided
        # below, so a PVS1 on an unknown id would still refuse PP3 — an acceptable
        # asymmetry: the model asserted a null mechanism either way.
        w.pvs1_met[key] = any(isinstance(c, dict) and _code_of(c) == "PVS1" and bool(c.get("met"))
                              for c in data.get("criteria") or [])
    for name, info in fields.items():
        if name in ("evidence_ids", "trial_ids"):
            continue  # judged once, by _drop_reason
        value = data.get(name)
        child = _join(path, name)
        item_model = _item_model(info.annotation)
        if name == "literature" and isinstance(value, list):
            data[name] = _clean_literature(value, child, w)
        elif isinstance(value, str):
            data[name] = _redact(value, child, w)
        elif isinstance(value, list) and item_model is not None:
            data[name] = _clean_items(value, item_model, child, w, key)
        elif isinstance(value, list):
            data[name] = [_redact(x, f"{child}[{i}]", w) if isinstance(x, str) else x for i, x in enumerate(value)]
        elif isinstance(value, dict) and _is_model(info.annotation):
            _clean_object(value, info.annotation, child, w, key)
    if "criteria" in fields and "classification" in fields:
        _set_classification(data, _item_model(fields["criteria"].annotation), path, w)


def _clean_items(items: list[Any], model: type[BaseModel], path: str, w: _Walk, key: str | None) -> list[Any]:
    """The items that survive the citation rules, cleaned; the rest are rejections."""
    fields = model.model_fields
    cites = "evidence_ids" in fields or "trial_ids" in fields
    is_variant = "key" in fields and "criteria" in fields
    kept: list[Any] = []
    seen_codes: set[str] = set()
    seen_keys: set[str] = set()
    for i, item in enumerate(items):
        child = f"{path}[{i}]"
        if not isinstance(item, dict):
            kept.append(item)  # not this module's call; the schema check at the end raises
            continue
        reason = _align_key(item, child, w, seen_keys) if is_variant else None
        if reason is None and cites:
            reason = _drop_reason(item, model, child, w, key)
        code = _code_of(item) if "code" in fields else None
        if reason is None and code is not None and code in seen_codes:
            reason = f"duplicate criterion {code} (the first occurrence is kept)"
            w.report.counts["duplicates_dropped"] += 1
        if reason:
            w.report.reject(child, reason)
            w.report.counts["items_dropped"] += 1
            continue
        _clean_object(item, model, child, w, key)
        if code is not None:
            seen_codes.add(code)
            # after the prose walk: the marks these add are not citations
            _cap_strength(item, child, w)
            _recompute_frequency(item, child, w, key)
            _recompute_computational(item, child, w, key)
            _unverified_gene_constraint(item, child, w)
            _check_phenotype(item, child, w)
            _mark_retired(item, child, w)
        kept.append(item)
    return kept


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def _code_of(item: dict[str, Any]) -> str:
    return str(item.get("code", "")).strip().upper()


def _align_key(item: dict[str, Any], path: str, w: _Walk, seen: set[str]) -> str | None:
    """Pin a variant entry's ``key`` to a real one: canonical spelling, and one of the
    candidate's when the caller named them. Returns the reason to drop the entry, or
    ``None`` with ``item['key']`` rewritten."""
    raw = str(item.get("key", ""))
    norm = canonical_key(raw)
    if w.keys is None:
        match = norm or raw
    else:
        match = w.keys.get(norm) if norm else None
        if match is None:
            n = len(item.get("criteria") or [])
            return (f"key {raw!r} is not a variant of this candidate ({', '.join(w.keys.values())}); "
                    f"the entry and its {n} criteria were dropped")
    if match in seen:
        return f"a second entry for variant {match!r}; the first is kept"
    seen.add(match)
    if match != raw:
        w.report.counts["keys_respelled"] += 1
        w.report.notes.append(f"{path}.key: the model wrote {raw!r}; matched to {match!r}")
        item["key"] = match
    return None


def _drop_reason(item: dict[str, Any], model: type[BaseModel], path: str, w: _Walk, key: str | None) -> str | None:
    """Why this item must go, or ``None``. A missing citation list is an empty one."""
    fields = model.model_fields
    ids = _norm_ids(item.get("evidence_ids")) if "evidence_ids" in fields else []
    trials = _norm_trial_ids(item.get("trial_ids")) if "trial_ids" in fields else []
    ids = [i if i in w.index else _respelled(i, f"{path}.evidence_ids", w) or i for i in ids]
    trials = [t if t in w.index else _respelled(t, f"{path}.trial_ids", w) or t for t in trials]
    if "evidence_ids" in fields:
        item["evidence_ids"] = ids
    if "trial_ids" in fields:
        item["trial_ids"] = trials
    w.report.counts["ids_checked"] += len(ids) + len(trials)
    unknown = [i for i in ids if i not in w.index]
    if unknown:
        w.report.counts["ids_unknown"] += len(unknown)
        return f"unknown evidence id(s): {', '.join(unknown)}"
    bad_trials = [t for t in trials if not _NCT_ID.match(t) or t not in w.index]
    if bad_trials:
        w.report.counts["ids_unknown"] += len(bad_trials)
        return f"trial id(s) not nct: records in the store: {', '.join(bad_trials)}"
    if "code" in fields:
        code = _code_of(item)
        if code not in ALL_CODES:
            return f"unknown ACMG code {str(item.get('code', '')).strip()!r}"
        if not ids and code not in CASE_LEVEL_CODES:
            return f"{code} cites no evidence record and is not a case-level criterion"
        if code in FREQUENCY_CODES:
            own = gnomad_record_id(key) if key else None
            foreign = [i for i in ids if i.startswith("gnomad:") and i != own]
            if foreign:
                return f"{code} cites the gnomAD record of a different variant: {', '.join(foreign)}"
        return None
    if "counter_arguments" in fields:
        args = [a.strip() for a in item.get("counter_arguments") or [] if isinstance(a, str) and a.strip()]
        item["counter_arguments"] = args
        if not args:
            return "drug candidate has no counter-arguments"
    if not ids and not trials:
        return "cites no evidence record"
    if "chembl_id" in fields:
        _check_chembl(item, ids + trials, path, w)
    return None


def _check_chembl(item: dict[str, Any], cited: list[str], path: str, w: _Walk) -> None:
    """A ``chembl_id`` stands only as a ``chembl:<id>`` record in the store or as an
    id a record the candidate cites carries (a drug row naming the molecule);
    otherwise it is cleared and the clearing logged — a made-up ChEMBL id in a
    heading reads as verified."""
    raw = item.get("chembl_id")
    cid = str(raw or "").strip().upper()
    if not cid:
        item["chembl_id"] = None
        return
    w.report.counts["ids_checked"] += 1
    if f"chembl:{cid}" in w.index or w.carried(cid, cited):
        item["chembl_id"] = cid
        return
    w.report.counts["ids_unknown"] += 1
    w.report.counts["chembl_cleared"] += 1
    w.report.reject(f"{path}.chembl_id", f"chembl_id {raw!r} is neither a chembl: record in the store nor carried "
                                         "by a record the candidate cites; cleared")
    item["chembl_id"] = None


def _cap_strength(item: dict[str, Any], path: str, w: _Walk) -> None:
    """Lower a stated strength to :data:`STRENGTH_CAP` for the code, marking the
    justification; the verdict is combined at the lowered level."""
    code, stated = _code_of(item), item.get("strength")
    cap = STRENGTH_CAP.get(code)
    if cap is None or stated not in STRENGTH_RANK or STRENGTH_RANK[stated] <= STRENGTH_RANK[cap]:
        return
    w.report.counts["strength_capped"] += 1
    reason = f"{code} at most {cap} ({w.report.rules['strength_cap'].split(';')[0]}); the model said {stated}"
    w.report.notes.append(f"{path}.strength: {reason}; lowered to {cap}")
    item["strength"] = cap
    item["justification"] = f"{CAPPED_MARK.format(reason=reason)} {item.get('justification', '')}".rstrip()


def canonical_id(s: str) -> str:
    """``PMID:7647779`` → ``pmid:7647779``: the source prefix is case-folded, the id
    is left as written (``VCV…``, ``NCT…`` and gnomAD ids are upper-case by nature)."""
    src, sep, rest = s.strip().partition(":")
    return f"{src.lower()}{sep}{rest}" if sep else s.strip()


def _norm_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for v in value:
        s = canonical_id(str(v))
        if s and s not in out:
            out.append(s)
    return out


def _norm_trial_ids(value: Any) -> list[str]:
    """Trial ids as ``nct:NCT<8 digits>``; a bare ``NCT01807923`` is the same claim."""
    out: list[str] = []
    for s in _norm_ids(value):
        m = _BARE_NCT.match(s)
        s = f"nct:{m.group(1).upper()}" if m else s
        if s not in out:
            out.append(s)
    return out


def _clean_literature(items: list[Any], path: str, w: _Walk) -> list[str]:
    kept: list[str] = []
    for i, v in enumerate(items):
        s = canonical_id(str(v))
        if s.isdigit():
            s = f"pmid:{s}"  # a bare number is the same claim; spell it as the record id
        w.report.counts["ids_checked"] += 1
        if not _PMID_ID.match(s):
            w.report.reject(f"{path}[{i}]", f"not a pmid:<n> id: {v!r}")
        elif s not in w.index:
            w.report.counts["ids_unknown"] += 1
            w.report.reject(f"{path}[{i}]", f"no such literature record in the store: {s}")
        elif s in kept:
            continue
        else:
            kept.append(s)
            continue
        w.report.counts["literature_removed"] += 1
    return kept


# ------------------------------------------------------------------- citations in prose

def _group_tokens(inner: str) -> list[str] | None:
    """The tokens of a bracket group ``id, id; id`` — or ``None`` when the bracket
    holds anything that is not spelled ``source:id`` (``[sic]``, ``[see below]``)."""
    tokens = [t.strip(_TRAIL) for t in _SPLIT.split(inner.strip())]
    tokens = [t for t in tokens if t]
    if tokens and all(_ID_LIKE.match(t) for t in tokens):
        return tokens
    return None


def _numbered_group(inner: str) -> bool:
    """``[1]``, ``[2, 3]``, ``[4-6]`` — a numbered reference, which names no record."""
    tokens = [t for t in _SPLIT.split(inner.strip()) if t]
    return bool(tokens) and all(_NUMBERED.match(t) for t in tokens)


def _source_of(token: str) -> str:
    return token.split(":", 1)[0].lower()


def _accession_id(m: re.Match[str]) -> str | None:
    """The record id a bare accession names, when the mapping needs no lookup:
    ``VCV…`` → ``clinvar:``, ``NCT…`` → ``nct:``, ``PMID n``/``PubMed n`` → ``pmid:``."""
    if m.group("pmid") is not None:
        return f"pmid:{m.group('pmid')}"
    if m.group("vcv") is not None:
        return f"clinvar:{m.group('vcv').upper()}"
    if m.group("nct") is not None:
        return f"nct:{m.group('nct').upper()}"
    return None


def citation_tokens(text: str, sources: Iterable[str] = KNOWN_SOURCES) -> list[str]:
    """Every record id cited in ``text``, canonical, in order of appearance: bracket
    groups (the tokens of a known source — ``[HP:0002205]`` is a term, not a
    citation), bare ``source:id`` tokens of a known source, ``PMID <n>`` /
    ``PubMed <n>``, and bare ``VCV…`` / ``NCT…`` accessions as their record ids."""
    known = {s.lower() for s in sources}
    found: list[str] = []

    def take(rid: str) -> None:
        if rid not in found:
            found.append(rid)

    for m in _CITATION.finditer(text):
        if m.group("group") is not None:
            tokens = _group_tokens(m.group("group"))
            if tokens is None:
                found.extend(x for x in citation_tokens(m.group("group"), known) if x not in found)
            else:
                for tok in tokens:
                    if _source_of(tok) in known:
                        take(canonical_id(tok))
        elif m.group("src") is not None:
            if m.group("src").lower() in known:
                take(canonical_id(m.group(0)))
        else:
            rid = _accession_id(m)
            if rid is not None:
                take(rid)
    return found


def _respelled(rid: str, path: str, w: _Walk) -> str | None:
    """The store's spelling of an id that is not in the store but is one separator
    edit away from exactly one id of its source — noted and counted; ``None`` when
    there is none (the unknown-id rule applies) or several (noted, no guess)."""
    found = w.near_misses(rid)
    if len(found) == 1:
        w.report.counts["ids_respelled"] += 1
        w.report.notes.append(f"{path}: {rid!r} respelled to {found[0]!r} (one separator edit; the store's spelling)")
        return found[0]
    if found:
        w.report.notes.append(f"{path}: {rid!r} ambiguous near-miss: {', '.join(found)}")
    return None


def _redact(text: str, path: str, w: _Walk) -> str:
    """Replace inline citations that do not resolve by a footnote marker. The text
    stays; the id goes. A resolving id is rewritten in its canonical spelling (a
    near-miss in the store's) so the References list and the prose agree; a resolving
    ``PMC…``/``doi:`` gets its ``pmid:`` record appended once, for the same reason. A
    bracketed token of no known source is left alone and noted; a dbSNP id no cited
    record carries is left alone and noted."""
    added: set[str] = set()

    def hole(reason: str) -> str:
        return w.report.redact(path, reason, w.redactions)

    def check(rid: str, what: str) -> tuple[str, bool]:
        """``(the id as the store spells it, True)`` or ``(the marker, False)``."""
        if rid in w.index:
            w.report.counts["ids_checked"] += 1
            return rid, True
        alt = _respelled(rid, path, w)
        if alt is not None:
            w.report.counts["ids_checked"] += 1
            return alt, True
        return hole(f"{what}: {rid}"), False

    def note(what: str, token: str) -> None:
        w.report.counts["identifiers_unverified"] += 1
        w.report.notes.append(f"{path}: {what}: {token}")

    def group(m: re.Match[str]) -> str:
        inner = m.group("group")
        tokens = _group_tokens(inner)
        if tokens is None:
            if _numbered_group(inner):
                return hole(f"numbered reference {m.group(0)} names no record")
            return "[" + _CITATION.sub(repl, inner) + "]"
        cited = [tok for tok in tokens if _source_of(tok) in w.sources]
        if not cited:
            for tok in tokens:
                note("bracketed identifier of no evidence source, left in place (not a citation)", tok)
            return m.group(0)
        out = []
        for tok in tokens:
            if tok not in cited:
                note("bracketed identifier of no evidence source, left in place (not a citation)", tok)
                out.append(f"[{tok}]")
                continue
            shown, ok = check(canonical_id(tok), "inline citation to a record not in the store")
            out.append(f"[{shown}]" if ok else shown)
        return " ".join(out)

    def paper(shown: str, what: str, accession: str) -> str:
        rid = w.paper(accession)
        if rid is None:
            return hole(f"{what}: {accession}")
        w.report.counts["ids_checked"] += 1
        if rid in text or rid in added:
            return shown
        added.add(rid)
        return f"{shown} [{rid}]"

    def repl(m: re.Match[str]) -> str:
        if m.group("group") is not None:
            return group(m)
        if m.group("doi") is not None:
            shown = m.group(0).rstrip(_TRAIL)  # the regex cannot tell a DOI's end from the sentence's
            tail = m.group(0)[len(shown):]
            return paper(shown, "inline DOI not carried by any pmid: record in the store", shown.split(":", 1)[1].strip()) + tail
        if m.group("src") is not None:
            if m.group("src").lower() not in w.sources:
                return m.group(0)
            return check(canonical_id(m.group(0)), "inline citation to a record not in the store")[0]
        if m.group("pmc") is not None:
            return paper(m.group(0), "inline PMC id not carried by any pmid: record in the store", m.group("pmc").upper())
        if m.group("chembl") is not None:
            cid = m.group("chembl").upper()
            if f"chembl:{cid}" in w.index or w.carried(cid, w.cited):
                w.report.counts["ids_checked"] += 1
                return m.group(0)
            return hole(f"inline ChEMBL id neither a chembl: record in the store nor carried by a cited record: {cid}")
        if m.group("rs") is not None:
            if not w.carried(m.group("rs"), w.cited):
                note("dbSNP id not carried by any record the answer cites, left in place (not a citation)", m.group("rs"))
            return m.group(0)
        rid = _accession_id(m)
        assert rid is not None  # every remaining alternative maps to a record id
        what = "inline PMID not in the store" if m.group("pmid") is not None else "inline accession not in the store"
        shown, ok = check(rid, what)
        return m.group(0) if ok else shown  # "PMID 99999999" becomes the marker alone, not "PMID [^k]"

    return unwrap(_CITATION.sub(repl, text))


# ------------------------------------------------------------------- frequencies

@dataclass(frozen=True)
class ObservedFrequency:
    """What the store says about a variant's population frequency, and from where."""
    record: EvidenceRecord
    af: float | None
    """``None`` = absent from gnomAD (by the record consulted)."""
    detail: str
    """``af=0.318 (ac=…, an=…, hom=…)`` or ``VEP's copy of gnomAD: exomes …, genomes …``."""


def gnomad_frequency(rec: EvidenceRecord, af_field: str = AF_FIELD) -> tuple[float | None, int | None, int | None, int | None] | None:
    """``(af, ac, an, nhom)`` from a gnomAD record by the retriever's own projection.
    ``af`` is the ``af_field`` column, falling back to ``gnomad_af`` (joint ``ac/an``)
    when that column is empty; ``None`` when the record has no allele number at all.
    The whole result is ``None`` when the payload is not a gnomAD variant object."""
    p = rec.payload
    if isinstance(p, dict) and p.get("absent"):
        return None, 0, 0, 0  # asked for and not found: an observed absence
    if not isinstance(p, dict) or not any(isinstance(p.get(b), dict) for b in ("joint", "exome", "genome")):
        return None
    cols = GnomadRetriever(None).extract([rec])  # projection only; no request is made
    raw = cols.get(af_field) or cols["gnomad_af"]
    af = float(raw) if raw else None
    ac = int(cols["gnomad_ac"]) if cols["gnomad_ac"] else None
    an = int(cols["gnomad_an"]) if cols["gnomad_an"] else None
    nhom = int(cols["gnomad_nhom"]) if cols["gnomad_nhom"] else None
    return af, ac, an, nhom


def vep_gnomad_frequency(rec: EvidenceRecord) -> tuple[float | None, dict[str, str]] | None:
    """The gnomAD frequencies VEP carries as a courtesy: ``(max of exomes and genomes,
    {column: value})``; ``(None, {})`` when VEP lists none; ``None`` when the payload
    is not a VEP result."""
    if not isinstance(rec.payload, (dict, list)):
        return None
    cols = VepRetriever(None).extract([rec])  # projection only
    afs = {c: cols[c] for c in VEP_AF_FIELDS if cols.get(c)}
    values = []
    for v in afs.values():
        try:
            values.append(float(v))
        except ValueError:
            pass
    return (max(values) if values else None), afs


def observed_frequency(key: str | None, index: EvidenceIndex, af_field: str = AF_FIELD) -> ObservedFrequency | None:
    """The store's answer for a variant key: its gnomAD record, else its VEP record's
    copy of the gnomAD frequency, else ``None`` (nothing in the store can say)."""
    if not key:
        return None
    rid = gnomad_record_id(key)
    rec = index.get(rid) if rid else None
    if rec is not None:
        freq = gnomad_frequency(rec, af_field)
        if freq is not None:
            af, ac, an, nhom = freq
            label = "af" if af_field == "gnomad_af" else af_field
            detail = ("absent from gnomAD (queried, not found)" if af is None and an == 0 else
                      "absent from gnomAD (no allele number)" if af is None else
                      f"{label}={af:.3g} (ac={ac}, an={an}, hom={nhom})")
            return ObservedFrequency(rec, af, detail)
    vrec = index.get(f"vep:{key}")
    if vrec is not None:
        freq = vep_gnomad_frequency(vrec)
        if freq is not None:
            af, cols = freq
            shown = ", ".join(f"{VEP_AF_LABELS[c]} {v}" for c, v in cols.items())
            detail = ("VEP's copy of gnomAD lists no frequency (absent)" if af is None
                      else f"VEP's copy of gnomAD: {shown}; max {af:.3g} used")
            return ObservedFrequency(vrec, af, detail)
    return None


def frequency_criterion_met(code: str, af: float | None, th: dict[str, float]) -> bool:
    """The data's answer for PM2/BS1/BA1 given the gnomAD allele frequency
    (``None`` = absent from gnomAD)."""
    if code == "PM2":
        return af is None or af < th["pm2_max_af"]
    if code == "BS1":
        return af is not None and af > th["bs1_min_af"]
    if code == "BA1":
        return af is not None and af > th["ba1_min_af"]
    raise ValueError(f"not a frequency criterion: {code}")


def gnomad_record_id(key: str) -> str | None:
    """``gnomad:1-11796321-G-A`` for a variant key, or ``None`` for a key gnomAD
    cannot be asked about (mitochondrial, non-ACGT) — the retriever's own rule."""
    try:
        k = parse_key(key)
    except ValueError:
        return None
    return None if unaskable(k) else f"gnomad:{variant_id(k)}"


def _recompute_frequency(item: dict[str, Any], path: str, w: _Walk, key: str | None) -> None:
    code = _code_of(item)
    if code not in FREQUENCY_CODES:
        return
    obs = observed_frequency(key, w.index, w.af_field)
    if obs is None:
        _unverified(item, code, path, w, key)
        return
    w.report.counts["frequency_recomputed"] += 1
    if obs.record.source == "vep":
        w.report.counts["frequency_from_vep"] += 1
    if obs.record.record_id not in item["evidence_ids"]:
        item["evidence_ids"].append(obs.record.record_id)  # the number rests on this record
    recomputed = frequency_criterion_met(code, obs.af, w.th)
    claimed = bool(item.get("met"))
    if recomputed == claimed:
        return
    if obs.record.source == "vep" and obs.af is None:
        # VEP's courtesy copy lists nothing: the variant may be absent from gnomAD, or
        # gnomAD may simply never have been asked (the stage-2 prefilter skips what the
        # copy already shows common, and the copy is keyed on dbSNP). A gap is not an
        # observation, so the model's call stands either way and the gap is noted.
        w.report.counts["frequency_unverified"] += 1
        w.report.notes.append(f"{path}: {code} not recomputed — no gnomAD record and VEP's copy lists no frequency for "
                              f"{key}; the model's '{'met' if claimed else 'not met'}' stands, resting on absence only")
        return
    w.report.counts["frequency_disputed"] += 1
    reason = (f"{code} recomputed from {obs.record.record_id}: {obs.detail}; {w.report.rules[code]} → "
              f"{'met' if recomputed else 'not met'}; the model said {'met' if claimed else 'not met'}")
    w.report.disputes.append(Dispute(path, code, obs.record.record_id, claimed, recomputed, obs.af, reason))
    item["met"] = recomputed
    item["justification"] = f"{DISPUTED_MARK.format(reason=reason)} {item.get('justification', '')}".rstrip()


def _unverified(item: dict[str, Any], code: str, path: str, w: _Walk, key: str | None) -> None:
    """Nothing in the store can check the criterion: it does not count. The model's
    ``met`` is overridden to ``False`` (a dispute, when it said met) and the
    justification says why."""
    claimed = bool(item.get("met"))
    w.report.counts["frequency_unverified"] += 1
    reason = (f"no gnomAD or VEP record for {key or 'this variant'} in the store; {code} cannot be checked → not met"
              f"; the model said {'met' if claimed else 'not met'}")
    w.report.notes.append(f"{path}: {code} not recomputed — {reason}")
    if claimed:
        w.report.counts["frequency_disputed"] += 1
        w.report.disputes.append(Dispute(path, code, None, True, False, None, reason))
    item["met"] = False
    item["justification"] = f"{UNVERIFIED_MARK.format(reason=reason)} {item.get('justification', '')}".rstrip()


# ---------------------------------------------------------------- computational

RETIRED_MARK = "[RETIRED — {reason}]"


def _check_phenotype(item: dict[str, Any], path: str, w: _Walk) -> None:
    """PP4 stands only on the phenotype ranker's top gene; elsewhere it is disputed
    to not met, and without a stage-4 verdict it is left as the model called it."""
    code = _code_of(item)
    if code not in PHENOTYPE_CODES or not item.get("met"):
        return
    if w.phenotype_rank is None:
        return  # the caller does not run a phenotype ranker at all: the claim stands as written
    if not w.phenotype_rank.get("ran", True):
        # The ranker exists but did not run for this candidate: nothing can check the claim.
        w.report.counts["phenotype_unverified"] += 1
        reason = (f"{code} cannot be checked: the gene-blind phenotype ranker did not run for this candidate, so the "
                  "phenotype's specificity to this gene rests on the model's word alone → not met; the model said met")
        w.report.notes.append(f"{path}: {reason}")
        w.report.disputes.append(Dispute(path, code, None, True, False, None, reason))
        item["met"] = False
        item["justification"] = f"{UNVERIFIED_MARK.format(reason=reason)} {item.get('justification', '')}".rstrip()
        return
    rank, rid = w.phenotype_rank.get("rank"), w.phenotype_rank.get("record_id")
    if rank == 1:
        if rid and rid in w.index and rid not in item["evidence_ids"]:
            item["evidence_ids"].append(rid)
        return
    w.report.counts["phenotype_disputed"] += 1
    reason = (f"{code} recomputed from the gene-blind phenotype ranker: the gene is ranked "
              f"{rank if rank is not None else 'unranked'} for the case's HPO terms, not first, so the phenotype is not "
              "specific to it → not met; the model said met")
    w.report.disputes.append(Dispute(path, code, rid, True, False, None, reason))
    item["met"] = False
    item["justification"] = f"{DISPUTED_MARK.format(reason=reason)} {item.get('justification', '')}".rstrip()


def _unverified_gene_constraint(item: dict[str, Any], path: str, w: _Walk) -> None:
    """PP2/BP1 rest on gene-level constraint the store cannot show: not counted."""
    code = _code_of(item)
    if code not in GENE_CONSTRAINT_CODES or not item.get("met"):
        return
    if any(rid.startswith("constraint:") for rid in item.get("evidence_ids") or []):
        return
    w.report.counts["constraint_unverified"] += 1
    reason = (f"{code} needs gene-level constraint evidence (gnomAD missense o/e) as a constraint: record; none is in the "
              "store, so the claim cannot be checked → not met; the model said met")
    w.report.notes.append(f"{path}: {reason}")
    w.report.disputes.append(Dispute(path, code, None, True, False, None, reason))
    item["met"] = False
    item["justification"] = f"{UNVERIFIED_MARK.format(reason=reason)} {item.get('justification', '')}".rstrip()


def _mark_retired(item: dict[str, Any], path: str, w: _Walk) -> None:
    """PP5/BP6 stay in the chain as the model's note of another laboratory's view,
    marked and set to not met so nothing downstream counts them."""
    code = _code_of(item)
    if code not in RETIRED_CODES:
        return
    w.report.counts["retired_not_counted"] += 1
    reason = f"{code} is retired ({w.report.rules['PP5/BP6']}); the ClinVar classification is concordance, not a criterion"
    w.report.notes.append(f"{path}: {reason}")
    item["met"] = False
    if not str(item.get("justification", "")).startswith("[RETIRED — "):  # a re-validated chain is not marked twice
        item["justification"] = f"{RETIRED_MARK.format(reason=reason)} {item.get('justification', '')}".rstrip()


def computational_support(cols: dict[str, str]) -> dict[str, tuple[str | None, str]]:
    """What the VEP columns of one variant support for PP3 and BP4: ``code → (strength
    or None, the score line it rests on)``. A missense is judged by REVEL, else by
    CADD; a change with no predicted protein effect (intronic, synonymous, UTR,
    splice-region) by SpliceAI; any variant earns a PP3 at supporting for a SpliceAI
    splice effect. The calibrations are for missense and splicing — an in-frame or
    truncating change has no calibrated predictor, so PP3/BP4 are not supported for
    it (PVS1/PM4 are its criteria)."""
    revel = _float(cols.get("revel"))
    cadd = _float(cols.get("cadd_phred"))
    splice = _float(cols.get("spliceai_ds_max"))
    consequence = cols.get("consequence") or ""
    missense = "missense_variant" in consequence
    protein_changing = missense or any(t in consequence for t in ("inframe", "stop_", "frameshift", "start_lost",
                                                                  "protein_altering", "splice_acceptor", "splice_donor"))
    pp3: str | None = None
    bp4: str | None = None
    lines: list[str] = []
    if missense and revel is not None:
        lines.append(f"REVEL {revel:g}")
        pp3 = next((lvl for cut, lvl in REVEL_PP3 if revel >= cut), None)
        bp4 = next((lvl for cut, lvl in REVEL_BP4 if revel <= cut), None)
    elif missense and cadd is not None:
        lines.append(f"CADD {cadd:g}")
        pp3 = "supporting" if cadd >= CADD_PP3_MIN else None
        bp4 = "supporting" if cadd <= CADD_BP4_MAX else None
    elif cadd is not None:
        lines.append(f"CADD {cadd:g} (not calibrated for this consequence)")
    if splice is not None:
        lines.append(f"SpliceAI max {splice:g}")
        if splice >= SPLICEAI_PP3_MIN and pp3 is None:
            pp3 = "supporting"
        if not protein_changing:
            bp4 = "supporting" if splice <= SPLICEAI_BP4_MAX else None
    am = cols.get("alphamissense_score")
    if am:
        lines.append(f"AlphaMissense {am} ({cols.get('alphamissense_class') or '?'}; not calibrated, not counted)")
    detail = "; ".join(lines) if lines else "no REVEL, CADD or SpliceAI score in the record"
    return {"PP3": (pp3, detail), "BP4": (bp4, detail)}


def _float(text: str | None) -> float | None:
    try:
        return float(text) if text not in (None, "") else None
    except ValueError:
        return None


def _recompute_computational(item: dict[str, Any], path: str, w: _Walk, key: str | None) -> None:
    code = _code_of(item)
    if code not in COMPUTATIONAL_CODES:
        return
    if code == "PP3" and item.get("met") and w.pvs1_met.get(key or ""):
        # ClinGen SVI: PP3 is not applied alongside PVS1 for the same variant — the
        # predicted impact is already the very-strong criterion's premise.
        w.report.counts["computational_disputed"] += 1
        reason = "PP3 is not counted beside a met PVS1 on the same variant (ClinGen SVI PVS1/splicing guidance) → not met; the model said met"
        w.report.disputes.append(Dispute(path, code, None, True, False, None, reason))
        item["met"] = False
        item["justification"] = f"{DISPUTED_MARK.format(reason=reason)} {item.get('justification', '')}".rstrip()
        return
    rec = w.index.get(f"vep:{key}") if key else None
    claimed = bool(item.get("met"))
    stated = item.get("strength")
    if rec is None:
        w.report.counts["computational_unverified"] += 1
        reason = (f"no VEP record for {key or 'this variant'} in the store; {code} cannot be checked → not met"
                  f"; the model said {'met' if claimed else 'not met'}")
        w.report.notes.append(f"{path}: {code} not recomputed — {reason}")
        if claimed:
            w.report.counts["computational_disputed"] += 1
            w.report.disputes.append(Dispute(path, code, None, True, False, None, reason))
        item["met"] = False
        item["justification"] = f"{UNVERIFIED_MARK.format(reason=reason)} {item.get('justification', '')}".rstrip()
        return
    w.report.counts["computational_recomputed"] += 1
    cols = VepRetriever(None).extract([rec])  # type: ignore[arg-type]  # extract reads the payload only
    supported, detail = computational_support(cols)[code]
    if rec.record_id not in item["evidence_ids"]:
        item["evidence_ids"].append(rec.record_id)
    if supported is None:
        if claimed:
            w.report.counts["computational_disputed"] += 1
            reason = f"{code} recomputed from {rec.record_id}: {detail}; below every calibrated threshold → not met; the model said met"
            w.report.disputes.append(Dispute(path, code, rec.record_id, True, False, None, reason))
            item["met"] = False
            item["justification"] = f"{DISPUTED_MARK.format(reason=reason)} {item.get('justification', '')}".rstrip()
        return
    if not claimed:
        w.report.counts["computational_disputed"] += 1
        reason = f"{code} recomputed from {rec.record_id}: {detail}; supports {supported} → met; the model said not met"
        w.report.disputes.append(Dispute(path, code, rec.record_id, False, True, None, reason))
        item["met"] = True
        item["strength"] = supported
        item["justification"] = f"{DISPUTED_MARK.format(reason=reason)} {item.get('justification', '')}".rstrip()
        return
    if stated in STRENGTH_RANK and STRENGTH_RANK[stated] > STRENGTH_RANK[supported]:
        w.report.counts["strength_capped"] += 1
        reason = f"{code} from {rec.record_id}: {detail}; the scores support {supported}, the model said {stated}"
        w.report.notes.append(f"{path}.strength: {reason}; lowered to {supported}")
        item["strength"] = supported
        item["justification"] = f"{CAPPED_MARK.format(reason=reason)} {item.get('justification', '')}".rstrip()


# ---------------------------------------------------------------- classification

def _set_classification(data: dict[str, Any], criterion_model: type[BaseModel] | None, path: str, w: _Walk) -> None:
    """``classification`` from the surviving criteria by the ACMG combining rules. A
    value the model wrote is replaced and noted — the verdict is never the model's."""
    items = [c for c in data.get("criteria") or [] if isinstance(c, dict)]
    if criterion_model is None:
        return
    criteria = [criterion_model.model_validate(c) for c in items]  # a malformed criterion raises, as documented
    computed = combine_acmg(criteria)  # type: ignore[arg-type]
    points = acmg_points(criteria)  # type: ignore[arg-type]
    claimed = data.get("classification")
    if claimed is not None and claimed != computed:
        w.report.counts["classification_replaced"] += 1
        w.report.notes.append(f"{_join(path, 'classification')}: the model said {claimed!r}; "
                              f"replaced by the engine's {computed!r} ({points:+d} SVI points over the met criteria)")
    elif claimed is not None:
        w.report.counts["classification_replaced"] += 1
        w.report.notes.append(f"{_join(path, 'classification')}: the model wrote {claimed!r}; recomputed by the engine (same verdict)")
    data["classification"] = computed
    data["points"] = points
    data["classification_richards_2015"] = combine_richards_2015(criteria)  # type: ignore[arg-type]
