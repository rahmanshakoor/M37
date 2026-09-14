"""Stage 3 thresholds, loaded from ``configs/filter.yaml``.

Every number a rule compares against lives in the config file and nowhere else, so
that the manifest can echo the exact thresholds a run applied and a re-run with the
same file is the same computation. The model is strict: an unknown key is a typo,
not an extension, and fails loudly rather than silently doing nothing.

Vocabularies are checked too, against the code that produces the values: an AF
column must be one a retriever writes, an impact must be one VEP's ranking knows,
a ClinVar class must be one the ClinVar retriever emits. A misspelt ``gnomad_AF``
or ``High`` would otherwise match no row and turn a filter into a pass-through
with only a manifest note to say so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from engine.retrieve.clinvar import PATHOGENICITY_CLASSES, ClinvarRetriever
from engine.retrieve.gnomad import GnomadRetriever
from engine.retrieve.vep import IMPACT_ORDER, VepRetriever

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "filter.yaml"

IMPACT_RANK: dict[str, int] = {imp: i for i, imp in enumerate(reversed(IMPACT_ORDER))}  # HIGH=3 … MODIFIER=0
IMPACT_RANK["SPLICE"] = IMPACT_RANK["MODERATE"]  # stage 2's marker for a splice term below MODERATE
IMPACT_RANK[""] = -1
"""Rank of every ``impact_any_coding`` value, from the order stage 2 applies (``vep.IMPACT_ORDER``)."""

RETRIEVER_COLUMNS: frozenset[str] = (
    frozenset(VepRetriever.columns) | frozenset(ClinvarRetriever.columns) | frozenset(GnomadRetriever.columns)
)
"""Every column a stage-2 retriever writes — the only names an AF source may use."""

IMPACTS: frozenset[str] = frozenset(k for k in IMPACT_RANK if k)
"""The ``impact_any_coding`` vocabulary, from the ranking stage 2 applies."""

CLINVAR_CLASSES: frozenset[str] = frozenset(PATHOGENICITY_CLASSES)
"""The ClinVar germline classes ``clinvar_pathogenicity`` is built from."""


def _known(values: list[str], allowed: frozenset[str], what: str) -> list[str]:
    unknown = [v for v in values if v not in allowed]
    if unknown:
        raise ValueError(f"{what}: unknown value(s) {unknown}; allowed: {sorted(allowed)}")
    return values


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RarityConfig(_Strict):
    recessive_max_af: float = Field(ge=0.0, le=1.0)
    dominant_max_af: float = Field(ge=0.0, le=1.0)
    homozygote_max_nhom: int = Field(ge=0)
    rescue_max_af: float = Field(ge=0.0, le=1.0)
    af_source_order: list[str] = Field(min_length=1)
    af_fallback_pick: Literal["first", "max"]

    @field_validator("af_source_order")
    @classmethod
    def _columns_exist(cls, v: list[str]) -> list[str]:
        return _known(v, RETRIEVER_COLUMNS, "rarity.af_source_order")


class ConsequenceConfig(_Strict):
    keep_impacts: list[str]
    keep_splice_min_ds: float = Field(ge=0.0, le=1.0)
    drop_terms: list[str] = Field(default_factory=list)

    @field_validator("keep_impacts")
    @classmethod
    def _impacts_exist(cls, v: list[str]) -> list[str]:
        return _known(v, IMPACTS, "consequence.keep_impacts")


class ClinvarConfig(_Strict):
    always_keep: list[str]
    drop_if_benign_min_stars: int = Field(ge=0, le=4)
    rescue_min_stars: int = Field(ge=0, le=4)

    @field_validator("always_keep")
    @classmethod
    def _classes_exist(cls, v: list[str]) -> list[str]:
        return _known(v, CLINVAR_CLASSES, "clinvar.always_keep")


class QualityConfig(_Strict):
    cluster_min_rows: int = Field(default=5, ge=2)
    cluster_window_bp: int = Field(default=200, ge=1)
    caveat_min_dp: int = Field(ge=0)
    caveat_min_gq: int = Field(ge=0)
    caveat_flagged_filters: bool = True


class PhaseConfig(_Strict):
    trust_pgt: bool


class FilterConfig(_Strict):
    rarity: RarityConfig
    consequence: ConsequenceConfig
    clinvar: ClinvarConfig
    quality: QualityConfig
    phase: PhaseConfig

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> "FilterConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"filter config not found: {path} (configs/filter.yaml in the source tree)")
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        return cls(**raw)

    def as_params(self) -> dict[str, Any]:
        """Plain dict for the manifest and candidates.json — every threshold, verbatim."""
        return self.model_dump(mode="json")
