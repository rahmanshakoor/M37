"""The dossier agent's output model.

Two fields are the engine's, not the model's: ``VariantPosition.protein_position``
(the residue the ``vep:`` record's HGVS protein change names) and ``region`` (the
UniProt features covering it, and the natural variants at it). They are dropped from
the answer schema the API enforces and filled by :mod:`engine.dossier.checks`, so a
residue number in the dossier is always one the records carry. ``key`` and
``uniprot_accession`` are pinned by enum to the chain's keys and the fetched entry.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

ENGINE_FIELDS = ("protein_position", "region")
"""Dropped from the answer schema; set by the engine after the model answers."""


class DossierClaim(BaseModel):
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)


class VariantPosition(BaseModel):
    key: str
    """Pinned by enum to the chain's variant keys."""
    protein_position: int | None = None
    """ENGINE-FILLED from the ``vep:`` record's hgvsp (:func:`engine.retrieve.uniprot.residue_of`)."""
    region: list[str] = Field(default_factory=list)
    """ENGINE-FILLED: ``regions_at`` + ``natural_variants_at`` over the ``uniprot:`` record."""
    consequence: str = ""
    """The model's cited statement of what the variant does at that position."""
    evidence_ids: list[str] = Field(default_factory=list)


class GeneDossier(BaseModel):
    candidate_id: str
    gene_symbol: str
    uniprot_accession: str
    """Pinned by enum to the entry the engine fetched."""
    protein: list[DossierClaim]
    """Name, length, function, domain organisation — cites ``uniprot:``."""
    mechanism_of_disease: list[DossierClaim]
    """How loss (or gain) of this product causes the disease — cites ``uniprot:`` and ``pmid:``."""
    variant_positions: list[VariantPosition]
    """One per chain variant."""
    region_knowledge: list[DossierClaim]
    """What is known about the regions the variants fall in."""
    genotype_patterns: list[DossierClaim]
    """Published genotype–phenotype patterns for the gene and disease."""
    functional_test: list[DossierClaim]
    """What a functional test in the patient's cells would show, and which."""
    limits: list[str] = Field(default_factory=list)
    literature: list[str] = Field(default_factory=list)
