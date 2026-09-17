"""The dossier agent's tools and the engine's fixed literature searches.

The tools are stage 5's three (:class:`engine.reason.tools.ReasonTools`: ``get_record``,
``search_literature``, ``get_paper``) bound to the same store, with one difference in
what ``get_record`` says about itself: the UniProt payload is large and the bundle
already carries the domain map, so the description tells the model when reading it is
worth a turn.

The searches the engine runs *before* the model are fixed templates over the gene
symbol and the UniProt DISEASE names — a coordinate or an HGVS string never enters a
query (:func:`engine.reason.tools.check_query` guards every one, and the templates
carry none) — so every dossier starts from the same kind of evidence, recorded in the
manifest as sent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.agents.client import ToolSpec
from engine.reason.tools import ReasonTools, check_query

DISEASES_SEARCHED = 3
"""How many of the entry's DISEASE names get a search of their own."""
QUERY_TEMPLATES = (
    'TITLE_ABS:"{gene}" AND TITLE_ABS:"{disease}"',
    'TITLE_ABS:"{gene}" AND (TITLE_ABS:"loss of function" OR TITLE_ABS:"mechanism" OR TITLE_ABS:"pathogenic variants")',
    'TITLE_ABS:"{gene}" AND (TITLE_ABS:"genotype" OR TITLE_ABS:"functional assay" OR TITLE_ABS:"functional characterization")',
)
"""The first is run once per disease name (up to :data:`DISEASES_SEARCHED`); the
other two once. Recorded in the manifest as ``params.query_templates``."""

GET_RECORD_NOTE = (
    " The uniprot: record is large (every feature, comment and cross-reference of the entry); the bundle already "
    "carries its domain map, FUNCTION and DISEASE texts, so read it only for what the bundle does not show "
    "(SUBUNIT, SUBCELLULAR LOCATION, PTM, the Pfam/InterPro cross-references, the full natural-variant list)."
)


@dataclass
class DossierRetrievers:
    """The services behind the step. Duck-typed: the stage passes the real retrievers,
    tests pass the same classes over a stub ``Http``."""

    uniprot: Any
    """``accession_for_symbol(symbol) -> str | None``, ``entry(accession) -> EvidenceRecord``, ``version()``, ``params``."""
    literature: Any
    """``search(query, max_results=) -> SearchResult``, ``fetch(pmid)``, ``extract(record)``."""


class DossierTools(ReasonTools):
    """Stage 5's tools with the dossier's ``get_record`` description."""

    def specs(self) -> list[ToolSpec]:
        specs = super().specs()
        get_record = specs[0]
        specs[0] = ToolSpec(get_record.name, get_record.description + GET_RECORD_NOTE, get_record.input_schema, get_record.handler)
        return specs


def fixed_queries(gene: str, diseases: list[str]) -> list[str]:
    """The engine's searches for one gene, in order: one per disease name (the first
    :data:`DISEASES_SEARCHED`), then the mechanism and the genotype templates. Every
    query passes :func:`~engine.reason.tools.check_query`."""
    out = [QUERY_TEMPLATES[0].format(gene=_term(gene), disease=_term(d)) for d in diseases[:DISEASES_SEARCHED] if d.strip()]
    out.extend(t.format(gene=_term(gene)) for t in QUERY_TEMPLATES[1:])
    for q in out:
        check_query(q)
    return out


def _term(text: str) -> str:
    """A phrase inside a Europe PMC quoted term: quotes cannot nest."""
    return " ".join(str(text).replace('"', " ").split())
