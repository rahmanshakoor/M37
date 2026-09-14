"""Chromosome naming, reconciled once.

Three conventions meet in this project and disagree with each other:

* the source VCF and Ensembl use ``1``…``22``, ``X``, ``Y`` — with the mitochondrion
  spelled ``M`` in the VCF and ``MT`` by Ensembl;
* the submission template and UCSC use ``chr1``…``chr22``, ``chrX``, ``chrY``, ``chrM``;
* the challenge scorer compares chromosome strings exactly, so a mismatch scores zero
  without an error.

Internally the engine stores every chromosome in one canonical form (Ensembl style,
mitochondrion ``MT``) and converts at the edges with :func:`to_submission` and
:func:`to_annotation`. Anything that is not a primary chromosome — unplaced, random,
alt and decoy contigs — has no canonical name and is dropped at ingest with a count.
"""

from __future__ import annotations

from typing import Iterable, Literal

AUTOSOMES: tuple[str, ...] = tuple(str(i) for i in range(1, 23))
PRIMARY: tuple[str, ...] = AUTOSOMES + ("X", "Y", "MT")

NamingStyle = Literal["ensembl", "ucsc"]


def canonical(chrom: str) -> str | None:
    """Canonical name for a primary chromosome, or ``None`` for any other contig.

    >>> canonical("chr15"), canonical("15"), canonical("M"), canonical("chrMT")
    ('15', '15', 'MT', 'MT')
    >>> canonical("1_KI270706v1_random") is None
    True
    """
    c = chrom.strip()
    if c[:3].lower() == "chr":
        c = c[3:]
    u = c.upper()
    if u in ("M", "MT"):
        return "MT"
    if u in ("X", "Y"):
        return u
    if c in AUTOSOMES:
        return c
    return None


def is_primary(chrom: str) -> bool:
    return canonical(chrom) is not None


def to_submission(chrom: str) -> str:
    """Submission/UCSC form: ``chr15``, ``chrX``, ``chrM``.

    Raises ``ValueError`` for a non-primary contig — nothing outside the primary
    assembly can appear in a submission row.
    """
    c = canonical(chrom)
    if c is None:
        raise ValueError(f"not a primary chromosome: {chrom!r}")
    return "chrM" if c == "MT" else f"chr{c}"


def to_annotation(chrom: str) -> str:
    """Ensembl form: ``15``, ``X``, ``MT`` — what the REST endpoints expect."""
    c = canonical(chrom)
    if c is None:
        raise ValueError(f"not a primary chromosome: {chrom!r}")
    return c


def naming_style(contig_ids: Iterable[str]) -> NamingStyle:
    """Detect whether a VCF header names chromosomes ``chr1`` or ``1``.

    Decided by the primary autosomes only, so odd extra contigs cannot tip it.
    """
    ids = set(contig_ids)
    ucsc = sum(1 for a in AUTOSOMES if f"chr{a}" in ids)
    ensembl = sum(1 for a in AUTOSOMES if a in ids)
    if ucsc == 0 and ensembl == 0:
        raise ValueError("no primary autosomes found among contig ids")
    return "ucsc" if ucsc >= ensembl else "ensembl"


def primary_names(contig_ids: Iterable[str]) -> list[str]:
    """The primary chromosomes as spelled in *this* VCF, in header order.

    Used to build the region list handed to bcftools, which must use the file's own
    spelling. Only contigs actually declared in the header are returned.
    """
    return [cid for cid in contig_ids if canonical(cid) is not None]
