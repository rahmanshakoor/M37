"""Evidence records and the store that holds them.

A record is the unit of citation. Its ``record_id`` is what a downstream claim carries;
its ``url`` is what a judge opens; its ``payload`` is the raw source response (or the
relevant slice of it) so that any number quoted later can be checked against it.

Records are written as JSON with sorted keys and a fixed indent, so a rerun that
produces the same record produces the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

VariantKey = tuple[str, int, str, str]
"""(canonical chrom, pos, ref, alt) — chrom as ``1``…``22``, ``X``, ``Y``, ``MT``."""


def key_str(k: VariantKey) -> str:
    """``7:117559590:ATCT:A`` — the engine's spelling of a variant, used in ids and tables."""
    return f"{k[0]}:{k[1]}:{k[2]}:{k[3]}"


def parse_key(s: str) -> VariantKey:
    c, p, r, a = s.split(":")
    return (c, int(p), r, a)


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(record_id: str) -> str:
    """Filesystem-safe, still readable, collision-resistant file stem for a record id."""
    stem = _SAFE.sub("_", record_id).strip("_")[:120]
    digest = hashlib.sha256(record_id.encode()).hexdigest()[:10]
    return f"{stem}.{digest}"


@dataclass(frozen=True)
class EvidenceRecord:
    record_id: str
    """``<source>:<stable id>`` — e.g. ``gnomad:1-11796321-G-A``, ``clinvar:VCV000003520``,
    ``vep:1:11796321:G:A``, ``pmid:21083385``."""
    source: str
    source_version: str
    query: dict[str, Any]
    url: str
    retrieved_at: str
    payload: Any

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=1, ensure_ascii=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "EvidenceRecord":
        d = json.loads(text)
        return cls(**d)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()


class EvidenceStore:
    """``<root>/<source>/<safe_name>.json`` per record, plus ``index.json`` on demand."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, record_id: str) -> Path:
        source = record_id.split(":", 1)[0]
        return self.root / source / f"{safe_name(record_id)}.json"

    def put(self, rec: EvidenceRecord) -> Path:
        p = self.path_for(rec.record_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        text = rec.to_json()
        if p.exists() and p.read_text() == text:
            return p  # identical bytes already there — leave mtime alone
        p.write_text(text)
        return p

    def get(self, record_id: str) -> EvidenceRecord | None:
        p = self.path_for(record_id)
        return EvidenceRecord.from_json(p.read_text()) if p.exists() else None

    def exists(self, record_id: str) -> bool:
        return self.path_for(record_id).exists()

    def iter(self, source: str | None = None) -> Iterator[EvidenceRecord]:
        dirs = [self.root / source] if source else sorted(d for d in self.root.iterdir() if d.is_dir())
        for d in dirs:
            if not d.exists():
                continue
            for p in sorted(d.glob("*.json")):
                yield EvidenceRecord.from_json(p.read_text())

    def write_index(self) -> Path:
        """``index.json``: record id → source, url, retrieved_at, path, sha256. The audit trail."""
        index: dict[str, dict[str, str]] = {}
        for rec in self.iter():
            index[rec.record_id] = {
                "source": rec.source,
                "source_version": rec.source_version,
                "url": rec.url,
                "retrieved_at": rec.retrieved_at,
                "path": str(self.path_for(rec.record_id).relative_to(self.root)),
                "sha256": rec.sha256,
            }
        p = self.root / "index.json"
        p.write_text(json.dumps(index, sort_keys=True, indent=1) + "\n")
        return p

    def count(self, source: str | None = None) -> int:
        return sum(1 for _ in self.iter(source))
