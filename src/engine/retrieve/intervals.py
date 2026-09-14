"""Interval membership for BED-defined regions, for ~5M lookups a run.

Intervals are stored per canonical chromosome as sorted, merged, half-open
``[start, end)`` arrays; membership is a bisect. Merged means non-overlapping, so a
single binary search answers the query.
"""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

from engine.contigs import canonical


class IntervalIndex:
    def __init__(self, intervals: dict[str, list[tuple[int, int]]]):
        self.starts: dict[str, list[int]] = {}
        self.ends: dict[str, list[int]] = {}
        self.names: dict[str, list[str]] = {}
        for chrom, ivs in intervals.items():
            merged = _merge(sorted(ivs))
            self.starts[chrom] = [s for s, _e, _n in merged]
            self.ends[chrom] = [e for _s, e, _n in merged]
            self.names[chrom] = [n for _s, _e, n in merged]

    @classmethod
    def from_bed(cls, path: Path) -> "IntervalIndex":
        raw: dict[str, list[tuple[int, int, str]]] = {}
        with open(path) as f:
            for line in f:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                parts = line.rstrip("\n").split("\t")
                chrom = canonical(parts[0])
                if chrom is None:
                    continue
                name = parts[3] if len(parts) > 3 else ""
                raw.setdefault(chrom, []).append((int(parts[1]), int(parts[2]), name))
        return cls(raw)  # type: ignore[arg-type]

    def contains(self, chrom: str, pos: int) -> bool:
        """``pos`` is 1-based (VCF); BED is 0-based half-open, so test ``pos-1``."""
        return self.lookup(chrom, pos) is not None

    def lookup(self, chrom: str, pos: int) -> str | None:
        """Name of the interval containing 1-based ``pos``, or None."""
        starts = self.starts.get(chrom)
        if not starts:
            return None
        p = pos - 1
        i = bisect_right(starts, p) - 1
        if i >= 0 and p < self.ends[chrom][i]:
            return self.names[chrom][i]
        return None

    def overlaps(self, chrom: str, pos: int, ref: str) -> bool:
        """True if the reference span of a variant at 1-based ``pos`` touches any interval."""
        starts = self.starts.get(chrom)
        if not starts:
            return False
        lo, hi = pos - 1, pos - 1 + max(1, len(ref))
        i = bisect_right(starts, hi - 1) - 1
        return i >= 0 and self.ends[chrom][i] > lo

    def total_bp(self) -> int:
        return sum(e - s for c in self.starts for s, e in zip(self.starts[c], self.ends[c]))

    def n_intervals(self) -> int:
        return sum(len(v) for v in self.starts.values())


def _merge(ivs: list[tuple]) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for iv in ivs:
        s, e = iv[0], iv[1]
        n = iv[2] if len(iv) > 2 else ""
        if out and s <= out[-1][1]:
            ps, pe, pn = out[-1]
            out[-1] = (ps, max(pe, e), pn if pn == n or not n else (pn + "," + n if pn else n))
        else:
            out.append((s, e, n))
    return out
