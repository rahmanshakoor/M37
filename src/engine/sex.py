"""Sample sex: stated in the case file, inferred from the calls, and applied where a
genotype can contradict a karyotype.

A male has one X and one Y outside the pseudoautosomal regions, so a heterozygous
call there is not a genotype a haploid chromosome can carry — it is a mapping or
calling artefact (or mosaicism), and two of them in one gene are never a compound
heterozygote. A female has no Y. The filter stage applies both facts as recorded
rules; this module holds the coordinates and the inference so ingest, filter and
submit agree.

Inference uses the fraction of X non-PAR carrier calls that are heterozygous (a male
sits near 0 — a few percent of artefacts — and a female near one half) and the
number of Y non-PAR carrier calls. It is recorded beside the stated sex and never
overrides it; a disagreement is a manifest note for a person to read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# GRCh38 pseudoautosomal regions (1-based, inclusive), per the assembly's own definition.
PAR_X: tuple[tuple[int, int], ...] = ((10_001, 2_781_479), (155_701_383, 156_030_895))
PAR_Y: tuple[tuple[int, int], ...] = ((10_001, 2_781_479), (56_887_903, 57_217_415))

SEXES = ("male", "female", "unknown")

MIN_X_CALLS = 50
"""Fewer X non-PAR carrier calls than this and no inference is made (a panel run)."""
MALE_MAX_X_HET_FRACTION = 0.20
FEMALE_MIN_X_HET_FRACTION = 0.35


def in_par(chrom: str, pos: int) -> bool:
    """True when ``pos`` on canonical ``X``/``Y`` lies in a pseudoautosomal region."""
    pars = PAR_X if chrom == "X" else PAR_Y if chrom == "Y" else ()
    return any(a <= pos <= b for a, b in pars)


def _alleles(gt: str) -> list[str]:
    return [a for a in gt.replace("|", "/").split("/") if a != ""]


@dataclass
class SexTally:
    """Counts fed one call at a time (canonical chromosome, position, GT string)."""

    x_nonpar_carrier: int = 0
    x_nonpar_het: int = 0
    y_nonpar_carrier: int = 0
    extra: dict[str, int] = field(default_factory=dict)

    def add(self, chrom: str, pos: int, gt: str) -> None:
        if chrom not in ("X", "Y") or in_par(chrom, pos):
            return
        alleles = _alleles(gt)
        if "1" not in alleles:
            return
        if chrom == "Y":
            self.y_nonpar_carrier += 1
            return
        self.x_nonpar_carrier += 1
        if len(alleles) > 1 and any(a != "1" for a in alleles):
            self.x_nonpar_het += 1

    @property
    def x_het_fraction(self) -> float | None:
        return self.x_nonpar_het / self.x_nonpar_carrier if self.x_nonpar_carrier else None

    def inferred(self) -> str:
        if self.x_nonpar_carrier < MIN_X_CALLS or self.x_het_fraction is None:
            return "unknown"
        if self.x_het_fraction <= MALE_MAX_X_HET_FRACTION:
            return "male"
        if self.x_het_fraction >= FEMALE_MIN_X_HET_FRACTION:
            return "female"
        return "unknown"

    def as_dict(self) -> dict[str, object]:
        f = self.x_het_fraction
        return {
            "x_nonpar_carrier_calls": self.x_nonpar_carrier,
            "x_nonpar_het_calls": self.x_nonpar_het,
            "x_nonpar_het_fraction": None if f is None else round(f, 4),
            "y_nonpar_carrier_calls": self.y_nonpar_carrier,
            "thresholds": {"min_x_calls": MIN_X_CALLS, "male_max_x_het_fraction": MALE_MAX_X_HET_FRACTION,
                           "female_min_x_het_fraction": FEMALE_MIN_X_HET_FRACTION},
            "inferred": self.inferred(),
        }


def effective_sex(stated: str, inferred: str) -> str:
    """The sex later stages act on: what the case file says, else what the calls say."""
    return stated if stated in ("male", "female") else inferred if inferred in ("male", "female") else "unknown"
