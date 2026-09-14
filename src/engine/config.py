"""Case configuration — the only place the engine learns anything about the proband.

Loaded from a ``case.yaml`` that is git-ignored. Relative paths resolve against the
directory containing the case file, so the file can sit next to the data it points at.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

HPO_ID = re.compile(r"^HP:\d{7}$")


class CaseConfig(BaseModel):
    proband_id: str = Field(min_length=1)
    vcf: Path
    reference_fasta: Path | None = None
    hpo: list[str] = Field(default_factory=list)
    regions: Path | None = None
    sample: str | None = None
    """Sample name inside the VCF. Optional for a single-sample file."""
    sex: str = "unknown"
    """``male``, ``female`` or ``unknown``. Stated sex wins over the sex inferred from
    the calls; the filter stage uses it to refuse genotypes a karyotype cannot carry
    (see :mod:`engine.sex`)."""

    @field_validator("sex")
    @classmethod
    def _sex(cls, v: str) -> str:
        v = (v or "unknown").strip().lower()
        if v not in ("male", "female", "unknown"):
            raise ValueError(f"sex must be male, female or unknown (got {v!r})")
        return v

    @field_validator("hpo")
    @classmethod
    def _hpo_ids(cls, terms: list[str]) -> list[str]:
        bad = [t for t in terms if not HPO_ID.match(t)]
        if bad:
            raise ValueError(f"not HPO ids (expected HP:0000000): {bad}")
        return terms

    @classmethod
    def load(cls, path: str | Path) -> "CaseConfig":
        path = Path(path).resolve()
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        base = path.parent
        for key in ("vcf", "reference_fasta", "regions"):
            if raw.get(key):
                raw[key] = (base / raw[key]).resolve()
        cfg = cls(**raw)
        cfg.check_files()
        return cfg

    def check_files(self) -> None:
        if not self.vcf.exists():
            raise FileNotFoundError(f"vcf not found: {self.vcf}")
        if not any(self.vcf.with_name(self.vcf.name + ext).exists() for ext in (".tbi", ".csi")):
            raise FileNotFoundError(
                f"vcf is not indexed (no .tbi/.csi beside it): {self.vcf}\n"
                f"  fix: bcftools index -t {self.vcf}"
            )
        if self.reference_fasta and not self.reference_fasta.exists():
            raise FileNotFoundError(f"reference_fasta not found: {self.reference_fasta}")
        if self.regions and not self.regions.exists():
            raise FileNotFoundError(f"regions file not found: {self.regions}")
