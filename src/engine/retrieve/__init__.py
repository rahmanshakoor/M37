"""Stage 2 — retrieve, don't recall.

Every fact a later stage may cite is fetched from a public source and stored as an
:class:`~engine.retrieve.store.EvidenceRecord` with the source, its version, the exact
query, a human-resolvable URL, the retrieval time and the raw payload. The variant
table this stage writes is a *view* over those records — regenerable from the store,
never the other way round.

Retrievers (one module each) implement :class:`Retriever`; the orchestrator in
:mod:`engine.retrieve.run` wires them together.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from engine.retrieve.store import EvidenceRecord, VariantKey


@runtime_checkable
class Retriever(Protocol):
    """A source of evidence records keyed by variant.

    Implementations must be deterministic given a warm cache: same keys in, same
    record bytes out. They must never raise on a variant that is simply absent from
    the source — absence is a result (an empty list), not an error.
    """

    source: str
    columns: tuple[str, ...]
    """Column names this retriever contributes to the variant table."""

    def version(self) -> str:
        """Citable version string of the source as observed at retrieval time."""
        ...

    def retrieve(self, keys: list[VariantKey]) -> dict[VariantKey, list[EvidenceRecord]]:
        """Fetch records for ``keys``. Keys absent from the source map to ``[]``."""
        ...

    def extract(self, records: list[EvidenceRecord]) -> dict[str, str]:
        """Project one variant's records onto ``columns`` (strings; '' when unknown)."""
        ...
