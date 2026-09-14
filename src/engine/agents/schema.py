"""Output schemas for the two agents, and the ACMG combining rules.

The model produces criteria with justifications and evidence ids; the *engine* combines
them into a classification (Richards et al. 2015, Table 5). Keeping the arithmetic out of
the model means the verdict is reproducible from the criteria, and a judge can recompute
it by hand.
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


class DrugCandidate(BaseModel):
    name: str
    chembl_id: str | None = None
    mechanism_of_action: str
    approval_status: str
    rationale: str
    counter_arguments: list[str] = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    trial_ids: list[str] = Field(default_factory=list)


class MedicineReport(BaseModel):
    candidate_id: str
    gene_symbol: str
    mechanism: list[MechanismClaim]
    pathway_targets: list[MechanismClaim] = Field(default_factory=list)
    candidates: list[DrugCandidate]
    follow_up_experiments: list[str] = Field(default_factory=list)
    limits: list[str] = Field(default_factory=list)
    literature: list[str] = Field(default_factory=list)


def _count(criteria: list[Criterion], codes: dict[str, str], strength: str) -> int:
    return sum(1 for c in criteria if c.met and c.code in codes and c.strength == strength)


def combine_acmg(criteria: list[Criterion]) -> Classification:
    """Richards et al. 2015 Table 5. Applied to *met* criteria at their stated strength
    (a modified strength, e.g. PM2_Supporting, is honoured because the model states it)."""
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
