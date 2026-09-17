"""Output schemas for the two agents, and the ACMG combining rules.

The model produces criteria with justifications and evidence ids; the *engine* combines
them into a classification. Keeping the arithmetic out of the model means the verdict
is reproducible from the criteria, and a judge can recompute it by hand.

The combining rule is the ClinGen SVI Bayesian point system (Tavtigian et al. 2020,
Genet Med 22:1001: supporting 1, moderate 2, strong 4, very strong 8; benign evidence
negative; pathogenic ≥ 10, likely pathogenic 6–9, uncertain 0–5, likely benign −1 to
−6, benign ≤ −7), which is the 2015 Table 5 made additive and closes its gaps — Table 5
has no rule for PVS1 + PM2_Supporting, the commonest null-variant chain, and points
give it likely pathogenic. The 2015 Table 5 verdict is computed beside it for the
record. PP5 and BP6 (another laboratory's classification as evidence) were retired
by the SVI (Biesecker & Harrison 2018, Genet Med 20:1687) and never count: a ClinVar
classification is reported as concordance, not as a criterion.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Strength = Literal["very_strong", "strong", "moderate", "supporting", "stand_alone"]
Classification = Literal["pathogenic", "likely_pathogenic", "vus", "likely_benign", "benign"]

PATHOGENIC_CODES = {
    "PVS1": "very_strong",
    "PS1": "strong", "PS2": "strong", "PS3": "strong", "PS4": "strong",
    "PM1": "moderate", "PM2": "moderate", "PM3": "moderate", "PM4": "moderate", "PM5": "moderate", "PM6": "moderate",
    "PP1": "supporting", "PP2": "supporting", "PP3": "supporting", "PP4": "supporting", "PP5": "supporting",
}
BENIGN_CODES = {
    "BA1": "stand_alone",
    "BS1": "strong", "BS2": "strong", "BS3": "strong", "BS4": "strong",
    "BP1": "supporting", "BP2": "supporting", "BP3": "supporting", "BP4": "supporting",
    "BP5": "supporting", "BP6": "supporting", "BP7": "supporting",
}
ALL_CODES = set(PATHOGENIC_CODES) | set(BENIGN_CODES)

RETIRED_CODES = frozenset({"PP5", "BP6"})
"""Retired by the ClinGen SVI (Biesecker & Harrison 2018): a reputable source's
classification is not evidence. Accepted from the model for the record, marked, and
never counted."""

POINTS = {"supporting": 1, "moderate": 2, "strong": 4, "very_strong": 8, "stand_alone": 8}
"""Tavtigian et al. 2020: the exponent ladder that makes ACMG/AMP 2015 additive."""

CASE_LEVEL_CODES = {"PP4", "PM3", "PS2", "PS4", "PP1", "PM6", "BS4", "BP2", "BP5"}
"""Criteria that may legitimately cite the case itself (phenotype, segregation, phase)
rather than a database record. A criterion outside this set with no evidence id is rejected."""


class Criterion(BaseModel):
    code: str
    strength: Strength
    met: bool
    justification: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("code")
    @classmethod
    def _known_code(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in ALL_CODES:
            raise ValueError(f"unknown ACMG code {v!r}")
        return v


class VariantChain(BaseModel):
    key: str
    criteria: list[Criterion]
    classification: Classification | None = None
    points: int | None = None
    """The SVI point total the classification is read from; the engine's, never the model's."""
    classification_richards_2015: Classification | None = None
    """The verdict the 2015 Table 5 gives the same criteria, for comparison."""
    summary: str = ""


class EvidenceChain(BaseModel):
    candidate_id: str
    variants: list[VariantChain]
    phase_statement: str
    mechanism_hypothesis: str
    limits: list[str] = Field(default_factory=list)
    what_would_change_the_call: list[str] = Field(default_factory=list)
    literature: list[str] = Field(default_factory=list)


class MechanismClaim(BaseModel):
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)


ClassVerdict = Literal["candidates_proposed", "considered_and_rejected", "no_evidence_found"]
"""What the search for one intervention class came to. A class is reported whatever
the outcome: the search records behind it say where the engine looked."""


class InterventionClass(BaseModel):
    """One rung-3 entry of the medicine ladder: a kind of intervention that could act
    on a node of the disease consequence, and the searches made for it. Stage 6 drops
    a class whose ``searched`` names no search record of its own conversation."""

    name: str
    acts_on: str
    """The consequence node the class addresses, with inline ``[ids]``."""
    targets: list[str] = Field(default_factory=list)
    """Gene symbols looked up with ``drugs_for_gene`` for this class."""
    searched: list[str] = Field(default_factory=list)
    """The search records behind the class: ``pmid-search:``, ``nct-search:``,
    ``chembl-search:``, ``dgidb-gene:`` and ``opentargets:<ENSG>`` ids."""
    verdict: ClassVerdict
    rejection_reason: str = ""
    """Required text when ``verdict`` is not ``candidates_proposed``."""
    evidence_ids: list[str] = Field(default_factory=list)


class DrugCandidate(BaseModel):
    name: str
    chembl_id: str | None = None
    intervention_class: str = ""
    """Must equal the ``name`` of a surviving :class:`InterventionClass`."""
    mechanism_of_action: str
    approval_status: str
    approved_indication: str = ""
    """The indication the record says the drug was approved for — never this disease
    unless a record says so. A research compound still states what the record says."""
    rationale: str
    counter_arguments: list[str] = Field(min_length=1)
    """The stage requires at least two non-empty entries."""
    paediatric_safety: str = ""
    """The systemic-exposure argument for a child whose every cell carries the defect, cited."""
    evidence_ids: list[str] = Field(default_factory=list)
    trial_ids: list[str] = Field(default_factory=list)


class ConsideredAndRejected(BaseModel):
    """A drug or a class the evidence argues against — a result, not padding."""

    name: str
    intervention_class: str = ""
    reason: str
    evidence_ids: list[str] = Field(default_factory=list)
    """The record(s) that reject it; the validator drops an entry citing nothing."""


class HpoTermRef(BaseModel):
    id: str
    label: str
    record_id: str | None = None


class DiseaseRef(BaseModel):
    id: str
    name: str
    description: str = ""
    record_id: str


class PatientContext(BaseModel):
    """Engine-filled from the case terms' ``hpo:`` records and the public disease
    records; dropped from the answer schema so the model never writes it."""

    hpo: list[HpoTermRef] = Field(default_factory=list)
    disease: list[DiseaseRef] = Field(default_factory=list)
    source: str = "engine-filled from the case terms' hpo: records and the public disease records; no free text"


class SecondaryFinding(BaseModel):
    """Another chain of the run whose lone heterozygous variant the engine classifies
    P/LP — recorded, never a repurposing target."""

    candidate_id: str
    gene_symbol: str
    model: str
    classifications: dict[str, str] = Field(default_factory=dict)
    """Variant key → engine classification."""
    note: str = ("a secondary finding, not a repurposing target; disclosure is the clinical team's decision "
                 "under the study's recontact rules")


class MedicineReport(BaseModel):
    candidate_id: str
    gene_symbol: str
    mechanism: list[MechanismClaim]
    """Rung 1: the variant-level mechanism, from the chain."""
    consequence: list[MechanismClaim] = Field(default_factory=list)
    """Rung 2: the cellular and disease-level consequence chain."""
    pathway_targets: list[MechanismClaim] = Field(default_factory=list)
    """Gene products the classes act on (rendered inside the classes section)."""
    intervention_classes: list[InterventionClass] = Field(default_factory=list)
    """Rung 3."""
    candidates: list[DrugCandidate]
    """Rung 4, proposed."""
    considered_and_rejected: list[ConsideredAndRejected] = Field(default_factory=list)
    """Rung 4, rejected."""
    surveillance: list[MechanismClaim] = Field(default_factory=list)
    """Rung 5, cited."""
    follow_up_experiments: list[str] = Field(default_factory=list)
    """Rung 5; each must carry at least one resolving inline citation."""
    limits: list[str] = Field(default_factory=list)
    literature: list[str] = Field(default_factory=list)
    patient_context: PatientContext | None = None
    """Engine-filled; dropped from the answer schema."""
    secondary_findings: list[SecondaryFinding] = Field(default_factory=list)
    """Engine-filled; dropped from the answer schema."""


def _count(criteria: list[Criterion], codes: dict[str, str], strength: str) -> int:
    return sum(1 for c in criteria if c.met and c.code in codes and c.code not in RETIRED_CODES and c.strength == strength)


def acmg_points(criteria: list[Criterion]) -> int:
    """Tavtigian et al. 2020 point total over the *met*, non-retired criteria at their
    stated strength (a modified strength, e.g. PM2_Supporting, is honoured because
    the model states it). Pathogenic evidence adds, benign evidence subtracts."""
    total = 0
    for c in criteria:
        if not c.met or c.code in RETIRED_CODES:
            continue
        if c.code in PATHOGENIC_CODES:
            total += POINTS[c.strength]
        elif c.code in BENIGN_CODES:
            total -= POINTS[c.strength]
    return total


def classify_points(points: int) -> Classification:
    """Tavtigian et al. 2020 categories."""
    if points >= 10:
        return "pathogenic"
    if points >= 6:
        return "likely_pathogenic"
    if points >= 0:
        return "vus"
    if points >= -6:
        return "likely_benign"
    return "benign"


def combine_acmg(criteria: list[Criterion]) -> Classification:
    """The engine's classification: the SVI point system over the met criteria, with
    BA1 kept stand-alone as Tavtigian et al. recommend (a variant above the BA1
    frequency is benign whatever else is claimed for it)."""
    if any(c.met and c.code == "BA1" for c in criteria):
        return "benign"
    return classify_points(acmg_points(criteria))


def combine_richards_2015(criteria: list[Criterion]) -> Classification:
    """Richards et al. 2015 Table 5 over the same criteria, for the record."""
    pvs = _count(criteria, PATHOGENIC_CODES, "very_strong")
    ps = _count(criteria, PATHOGENIC_CODES, "strong")
    pm = _count(criteria, PATHOGENIC_CODES, "moderate")
    pp = _count(criteria, PATHOGENIC_CODES, "supporting")
    ba = _count(criteria, BENIGN_CODES, "stand_alone")
    bs = _count(criteria, BENIGN_CODES, "strong")
    bp = _count(criteria, BENIGN_CODES, "supporting")

    pathogenic = (
        (pvs >= 1 and (ps >= 1 or pm >= 2 or (pm == 1 and pp == 1) or pp >= 2))
        or ps >= 2
        or (ps == 1 and (pm >= 3 or (pm == 2 and pp >= 2) or (pm == 1 and pp >= 4)))
    )
    likely_pathogenic = (
        (pvs == 1 and pm == 1)
        or (ps == 1 and pm in (1, 2))
        or (ps == 1 and pp >= 2)
        or pm >= 3
        or (pm == 2 and pp >= 2)
        or (pm == 1 and pp >= 4)
    )
    benign = ba >= 1 or bs >= 2
    likely_benign = (bs == 1 and bp == 1) or bp >= 2

    if (pathogenic or likely_pathogenic) and (benign or likely_benign):
        return "vus"  # contradictory evidence → uncertain, per the guideline
    if pathogenic:
        return "pathogenic"
    if likely_pathogenic:
        return "likely_pathogenic"
    if benign:
        return "benign"
    if likely_benign:
        return "likely_benign"
    return "vus"
